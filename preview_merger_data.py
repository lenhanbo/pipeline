from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import torch

from merger.runtime import build_dataset, load_config, save_labeled_preview, seed_everything


MERGE_SUITES = (
    "clean",
    "mask",
    "translation",
    "rotation",
    "shear_scale",
    "photometric",
    "combined",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the 32+32 merger inputs without loading a checkpoint"
    )
    parser.add_argument("--config", default="configs/merger_iam.yml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--suite", choices=MERGE_SUITES, default="combined")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output-dir", default="runs/merger_data_preview")
    return parser.parse_args()


def _image01(value: torch.Tensor) -> torch.Tensor:
    return value.add(1).div(2).clamp(0, 1)


def _stack(samples: list[dict[str, Any]], name: str) -> torch.Tensor:
    return torch.stack([sample[name] for sample in samples])


def _background_mean(image: torch.Tensor) -> float:
    # The normalized representation uses exactly +1 for paper. Restrict this
    # diagnostic to confidently light pixels so ink does not bias the value.
    pixels = image[image > 0.8]
    return float(pixels.mean()) if pixels.numel() else float("nan")


def save_pages(samples: list[dict[str, Any]], output_dir: Path, page_size: int) -> None:
    for start in range(0, len(samples), page_size):
        page = samples[start : start + page_size]
        target = _stack(page, "target")
        cut = target.size(-1) // 2
        left_target = target[..., :cut]
        right_target = target[..., cut:]
        left_input = _stack(page, "left_image")
        right_input = _stack(page, "right_image")
        naive = _stack(page, "naive_concat")
        rows = [
            ("Clean left target", _image01(left_target)),
            ("Clean right target", _image01(right_target)),
            ("Left input", _image01(left_input)),
            ("Right input (augmented)", _image01(right_input)),
            ("Boundary erase mask", _stack(page, "boundary_erase_mask").clamp(0, 1)),
            ("Right corruption mask", _stack(page, "right_corruption_mask").clamp(0, 1)),
            ("Input pair (diagnostic)", _image01(naive)),
            ("Clean 64px target", _image01(target)),
            ("Input vs target error", (naive - target).abs().div(2).clamp(0, 1)),
        ]
        labels = []
        for sample in page:
            labels.append(
                f"#{int(sample['record_index'])} {sample['text'][:12]}\n"
                f"x={int(sample['global_cut']) - 32}/{int(sample['source_width']) - 64} "
                f"a={float(sample['angle']):.1f} "
                f"t=({int(sample['translate_x'])},{int(sample['translate_y'])})"
            )
        number = start // page_size + 1
        save_labeled_preview(rows, labels, output_dir / f"page_{number:03d}.png")


def write_csv(samples: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset_index",
        "record_index",
        "text",
        "path",
        "global_cut",
        "window_start",
        "source_width",
        "angle",
        "translate_x",
        "translate_y",
        "scale",
        "shear",
        "boundary_erased_pixels",
        "left_ink_pixels",
        "right_ink_pixels",
        "left_background_mean",
        "right_background_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            target = sample["target"]
            cut = target.size(-1) // 2
            writer.writerow(
                {
                    "dataset_index": sample["dataset_index"],
                    "record_index": int(sample["record_index"]),
                    "text": sample["text"],
                    "path": sample["path"],
                    "global_cut": int(sample["global_cut"]),
                    "window_start": int(sample["global_cut"]) - 32,
                    "source_width": int(sample["source_width"]),
                    "angle": float(sample["angle"]),
                    "translate_x": int(sample["translate_x"]),
                    "translate_y": int(sample["translate_y"]),
                    "scale": float(sample["scale"]),
                    "shear": float(sample["shear"]),
                    "boundary_erased_pixels": int(sample["boundary_erase_mask"].sum()),
                    "left_ink_pixels": int((target[..., :cut] < 0.9).sum()),
                    "right_ink_pixels": int((target[..., cut:] < 0.9).sum()),
                    "left_background_mean": _background_mean(sample["left_image"]),
                    "right_background_mean": _background_mean(sample["right_image"]),
                }
            )


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.page_size < 1:
        raise ValueError("num-samples and page-size must be positive")
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    config["seed"] = args.seed
    config["data"]["min_text_length"] = max(
        2, int(config["data"].get("min_text_length", 2))
    )
    seed_everything(args.seed)
    dataset = build_dataset(
        config,
        stage="merge",
        split=args.split,
        deterministic=True,
        force_corruption_type=args.suite,
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: min(args.num_samples, len(indices))]
    samples: list[dict[str, Any]] = []
    for index in indices:
        sample = dataset[index]
        sample["dataset_index"] = index
        samples.append(sample)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pages(samples, output_dir, args.page_size)
    write_csv(samples, output_dir / "samples.csv")
    print(f"Wrote {len(samples)} samples to {output_dir.resolve()}")
    print("Rows show the two independent model inputs before any model inference.")


if __name__ == "__main__":
    main()
