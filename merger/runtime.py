from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .data import CorruptionConfig, IAMMergerDataset
from .metrics import compute_metrics, edge_copy_baseline
from .model import MergerNet


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corruption_config(config: dict[str, Any]) -> CorruptionConfig:
    return CorruptionConfig(**config.get("corruption", {}))


def build_model(config: dict[str, Any]) -> MergerNet:
    return MergerNet(**config.get("model", {}))


def build_dataset(
    config: dict[str, Any],
    stage: str,
    split: str,
    deterministic: bool,
    force_corruption_type: str | None = None,
) -> IAMMergerDataset:
    data = config["data"]
    return IAMMergerDataset(
        data_root=data["root"],
        gt_file=data[f"{split}_gt"],
        stage=stage,
        corruption=corruption_config(config),
        deterministic=deterministic,
        seed=int(config.get("seed", 123456)),
        max_samples=data.get(f"max_{split}_samples"),
        force_corruption_type=force_corruption_type,
        min_text_length=int(data.get("min_text_length", 2)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: sum(item[key] for item in items) / len(items) for key in items[0]}


def save_labeled_preview(
    rows: list[tuple[str, torch.Tensor]],
    column_labels: list[str],
    path: Path,
) -> None:
    """Save a comparison table whose rows and sample columns are labeled."""
    if not rows or not column_labels:
        return
    columns = min(len(column_labels), rows[0][1].size(0))
    figure, axes = plt.subplots(
        len(rows),
        columns,
        figsize=(max(6.0, 1.55 * columns), max(8.0, 1.35 * len(rows))),
        squeeze=False,
    )
    for row_index, (row_name, images) in enumerate(rows):
        images = images[:columns].detach().float().cpu()
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            axis.imshow(images[column_index, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(column_labels[column_index], fontsize=9)
            if column_index == 0:
                axis.set_ylabel(
                    row_name,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=9,
                )
    figure.suptitle("Merger preview: rows = outputs, columns = validation samples", fontsize=12)
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.98), pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def relative_improvement(model: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    higher_is_better = {"psnr", "ssim", "ink_precision", "ink_recall", "ink_f1"}
    result = {}
    for name, model_value in model.items():
        baseline_value = baseline[name]
        denominator = max(abs(baseline_value), 1e-8)
        delta = model_value - baseline_value if name in higher_is_better else baseline_value - model_value
        result[name] = 100.0 * delta / denominator
    return result


@torch.no_grad()
def evaluate_loader(
    model: MergerNet,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    preview_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    model.eval()
    model_metrics: list[dict[str, float]] = []
    noop_metrics: list[dict[str, float]] = []
    edge_metrics: list[dict[str, float]] = []
    preview_rows: list[tuple[str, torch.Tensor]] | None = None
    preview_columns: list[str] = []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        prediction = model(
            batch["left_image"],
            batch["right_image"],
            batch["left_corruption_mask"],
            batch["right_corruption_mask"],
            batch["prefix_context"],
            batch["prefix_mask"],
        )
        noop = torch.cat((batch["left_image"], batch["right_image"]), dim=-1)
        edge_baseline = edge_copy_baseline(noop, batch["seam_mask"])
        model_metrics.append(
            compute_metrics(
                prediction,
                batch["target"],
                batch["edit_mask"],
                batch["valid_mask"],
                noop,
                batch["left_preservation_mask"],
                batch["seam_mask"],
            )
        )
        noop_metrics.append(
            compute_metrics(
                noop, batch["target"], batch["edit_mask"], batch["valid_mask"],
                noop, batch["left_preservation_mask"], batch["seam_mask"]
            )
        )
        edge_metrics.append(
            compute_metrics(
                edge_baseline,
                batch["target"],
                batch["edit_mask"],
                batch["valid_mask"],
                noop,
                batch["left_preservation_mask"],
                batch["seam_mask"],
            )
        )
        if preview_rows is None:
            count = min(8, batch["target"].size(0))
            to_image = lambda value: value[:count].add(1).div(2).clamp(0, 1)
            prefix = torch.nn.functional.interpolate(
                batch["prefix_context"][:count],
                size=batch["target"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preview_rows = [
                ("Target", to_image(batch["target"])),
                ("Full prefix (resized)", to_image(prefix)),
                ("Left input", to_image(batch["left_image"])),
                ("Right input", to_image(batch["right_image"])),
                ("Naive concatenation", to_image(noop)),
                ("Corruption mask", batch["corruption_mask"][:count].clamp(0, 1)),
                ("Edit mask", batch["edit_mask"][:count].clamp(0, 1)),
                ("Left preserve mask", batch["left_preservation_mask"][:count].clamp(0, 1)),
                ("No-op baseline", to_image(noop)),
                ("Edge-copy baseline", to_image(edge_baseline)),
                ("Model prediction", to_image(prediction)),
                ("Absolute error", (prediction[:count] - batch["target"][:count]).abs().div(2).clamp(0, 1)),
            ]
            texts = batch.get("text", [])
            preview_columns = [
                f"#{index + 1}: {str(texts[index])[:16]}" if index < len(texts) else f"Sample {index + 1}"
                for index in range(count)
            ]
    if preview_path is not None and preview_rows is not None:
        save_labeled_preview(preview_rows, preview_columns, preview_path)
    model_result = aggregate_metrics(model_metrics)
    noop_result = aggregate_metrics(noop_metrics)
    edge_result = aggregate_metrics(edge_metrics)
    return {
        "model": model_result,
        "noop": noop_result,
        "edge_copy": edge_result,
        "improvement_vs_noop_pct": relative_improvement(model_result, noop_result),
        "improvement_vs_edge_copy_pct": relative_improvement(model_result, edge_result),
    }


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def cosine_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1e-8, step / max(1, warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
