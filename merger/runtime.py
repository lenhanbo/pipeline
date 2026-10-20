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
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

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
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: sum(item[key] for item in items) / len(items) for key in items[0]}


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
    preview = None
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        prediction = model(batch["corrupted"], batch["corruption_mask"], batch["edit_mask"])
        noop = batch["corrupted"]
        edge_baseline = edge_copy_baseline(batch["corrupted"], batch["edit_mask"])
        model_metrics.append(
            compute_metrics(
                prediction, batch["target"], batch["edit_mask"], batch["valid_mask"], batch["corrupted"]
            )
        )
        noop_metrics.append(
            compute_metrics(noop, batch["target"], batch["edit_mask"], batch["valid_mask"], batch["corrupted"])
        )
        edge_metrics.append(
            compute_metrics(
                edge_baseline, batch["target"], batch["edit_mask"], batch["valid_mask"], batch["corrupted"]
            )
        )
        if preview is None:
            error = (prediction - batch["target"]).abs().clamp(0, 2).sub(1)
            images = torch.cat(
                (
                    batch["target"][:8],
                    batch["corrupted"][:8],
                    batch["corruption_mask"][:8].mul(2).sub(1),
                    batch["edit_mask"][:8].mul(2).sub(1),
                    noop[:8],
                    edge_baseline[:8],
                    prediction[:8],
                    error[:8],
                )
            )
            preview = make_grid(images.add(1).div(2), nrow=min(8, batch["target"].size(0)), padding=2, pad_value=0.5)
    if preview_path is not None and preview is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(preview, preview_path)
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
