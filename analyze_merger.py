from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from merger.data import merger_collate
from merger.metrics import compute_metrics, edge_copy_baseline
from merger.runtime import (
    build_dataset,
    build_model,
    load_config,
    move_batch,
    save_json,
    save_labeled_preview,
    seed_everything,
)


SEAM_SUITES = ("clean", "mask", "translation", "rotation", "shear_scale", "photometric", "combined")
SELECTIONS = ("random", "worst_mae", "best_mae", "worst_seam", "worst_noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and rank merger checkpoint examples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/merger_iam.yml")
    parser.add_argument("--stage", choices=("continuation", "seam"), default=None)
    parser.add_argument("--suite", choices=SEAM_SUITES, default="combined")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--selection", choices=SELECTIONS, default="worst_noise")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--candidate-samples", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _image01(value: torch.Tensor) -> torch.Tensor:
    return value.add(1).div(2).clamp(0, 1)


def _noise_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # IAM has dark ink on a light background. Measure unwanted dark texture only
    # where the clean target is confidently background.
    background = target > 0.8
    count = background.sum().clamp_min(1)
    error = (prediction - target).abs()
    background_noise_mae = (error * background).sum() / count
    false_ink_rate = ((prediction < 0.8) & background).sum() / count
    return {
        "background_noise_mae": float(background_noise_mae),
        "background_false_ink_rate": float(false_ink_rate),
    }


def _record_batch(
    prediction: torch.Tensor,
    batch: dict[str, Any],
    dataset_indices: list[int],
) -> list[dict[str, Any]]:
    edge = edge_copy_baseline(batch["corrupted"], batch["seam_mask"])
    records: list[dict[str, Any]] = []
    for index in range(prediction.size(0)):
        item = {key: value[index : index + 1] for key, value in batch.items() if isinstance(value, torch.Tensor)}
        metrics = compute_metrics(
            prediction[index : index + 1],
            item["target"],
            item["edit_mask"],
            item["valid_mask"],
            item["corrupted"],
            item["left_preservation_mask"],
            item["seam_mask"],
        )
        metrics.update(_noise_metrics(prediction[index : index + 1], item["target"]))
        records.append(
            {
                "dataset_index": dataset_indices[index],
                "path": batch["path"][index],
                "text": batch["text"][index],
                "global_cut": int(batch["global_cut"][index]),
                "metrics": metrics,
                "target": item["target"].cpu(),
                "corrupted": item["corrupted"].cpu(),
                "corruption_mask": item["corruption_mask"].cpu(),
                "left_preservation_mask": item["left_preservation_mask"].cpu(),
                "seam_mask": item["seam_mask"].cpu(),
                "edge": edge[index : index + 1].cpu(),
                "prediction": prediction[index : index + 1].cpu(),
            }
        )
    return records


def _select(records: list[dict[str, Any]], selection: str, count: int) -> list[dict[str, Any]]:
    if selection == "random":
        return records[:count]
    key_and_reverse = {
        "worst_mae": ("mae", True),
        "best_mae": ("mae", False),
        "worst_seam": ("seam_discontinuity", True),
        "worst_noise": ("background_noise_mae", True),
    }
    metric, reverse = key_and_reverse[selection]
    return sorted(records, key=lambda item: item["metrics"][metric], reverse=reverse)[:count]


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    metric_names = list(records[0]["metrics"]) if records else []
    fields = ["rank", "dataset_index", "text", "path", "global_cut", *metric_names]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "dataset_index": record["dataset_index"],
                    "text": record["text"],
                    "path": record["path"],
                    "global_cut": record["global_cut"],
                    **record["metrics"],
                }
            )


def _save_pages(records: list[dict[str, Any]], output_dir: Path, page_size: int) -> None:
    for page_start in range(0, len(records), page_size):
        page = records[page_start : page_start + page_size]
        stack = lambda name: torch.cat([item[name] for item in page], dim=0)
        target = stack("target")
        corrupted = stack("corrupted")
        prediction = stack("prediction")
        rows = [
            ("Target", _image01(target)),
            ("Corrupted input", _image01(corrupted)),
            ("Corruption mask", stack("corruption_mask").clamp(0, 1)),
            ("Left preserve mask", stack("left_preservation_mask").clamp(0, 1)),
            ("Seam mask", stack("seam_mask").clamp(0, 1)),
            ("No-op baseline", _image01(corrupted)),
            ("Edge-copy baseline", _image01(stack("edge"))),
            ("Model prediction", _image01(prediction)),
            ("Absolute error", (prediction - target).abs().div(2).clamp(0, 1)),
            ("Model change", (prediction - corrupted).abs().div(2).clamp(0, 1)),
        ]
        labels = [
            f"#{item['dataset_index']} {item['text'][:12]}\n"
            f"MAE {item['metrics']['mae']:.3f} noise {item['metrics']['background_noise_mae']:.3f}"
            for item in page
        ]
        page_number = page_start // page_size + 1
        save_labeled_preview(rows, labels, output_dir / f"page_{page_number:03d}.png")


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.candidate_samples < 1 or args.page_size < 1:
        raise ValueError("sample counts and page size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    config["data"]["min_text_length"] = max(2, int(config["data"].get("min_text_length", 2)))
    seed_everything(args.seed)

    stage = args.stage or checkpoint.get("stage", "seam")
    suite = None if stage == "continuation" else args.suite
    requested = args.device or config.get("device", "cuda")
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(
        config,
        stage,
        args.split,
        deterministic=True,
        force_corruption_type=suite,
    )
    if not dataset:
        raise ValueError("no samples remain after filtering transcripts shorter than two characters")
    candidate_count = min(args.candidate_samples, len(dataset))
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:candidate_count]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=min(args.batch_size, candidate_count),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=merger_collate,
        pin_memory=device.type == "cuda",
    )

    records: list[dict[str, Any]] = []
    offset = 0
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            current_indices = indices[offset : offset + len(raw_batch["text"])]
            offset += len(current_indices)
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(
                    batch["corrupted"],
                    batch["corruption_mask"],
                    batch["edit_mask"],
                    batch["prefix_context"],
                    batch["prefix_mask"],
                )
            records.extend(_record_batch(prediction.float(), batch, current_indices))

    selected = _select(records, args.selection, min(args.num_samples, len(records)))
    checkpoint_name = Path(args.checkpoint).stem
    analysis_name = f"analysis_{checkpoint_name}_{stage}_{suite or 'default'}_{args.selection}_seed{args.seed}"
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(selected, output_dir / "samples.csv")
    _save_pages(selected, output_dir, args.page_size)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "stage": stage,
        "suite": suite,
        "split": args.split,
        "selection": args.selection,
        "seed": args.seed,
        "dataset_samples_after_filter": len(dataset),
        "candidate_samples": len(records),
        "selected_samples": len(selected),
        "mean_selected_metrics": {
            name: sum(item["metrics"][name] for item in selected) / len(selected)
            for name in selected[0]["metrics"]
        } if selected else {},
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Analyzed {len(records)} candidates; wrote {len(selected)} examples to {output_dir}")


if __name__ == "__main__":
    main()
