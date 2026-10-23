from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch

from merger.runtime import build_dataset, load_config, seed_everything


SUITES = ("clean", "mask", "translation", "rotation", "shear_scale", "photometric", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize every merger data-preparation step")
    parser.add_argument("--config", default="configs/merger_iam.yml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--suite", choices=SUITES, default="combined")
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output-dir", default="runs/merger_step_visualization")
    return parser.parse_args()


def _image01(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().add(1).div(2).clamp(0, 1)


def _show(axis: Any, image: torch.Tensor, title: str, mask: bool = False) -> None:
    array = image.detach().float().cpu()[0].numpy() if mask else _image01(image)[0].numpy()
    axis.imshow(array, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _source_with_padding(sample: dict[str, Any]) -> torch.Tensor:
    source = sample["source_image"]
    width = int(sample["source_width"])
    if source.size(-1) >= width:
        return source
    padded = source.new_ones((1, source.size(-2), width))
    padded[..., : source.size(-1)] = source
    return padded


def save_sample(sample: dict[str, Any], requested_index: int, path: Path) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(13, 13), squeeze=False)
    axes = axes.ravel()
    source = _source_with_padding(sample)
    _show(axes[0], source, "1. Normalized complete word")
    window_start = int(sample["global_cut"]) - int(sample["cut"])
    axes[0].add_patch(
        Rectangle(
            (window_start - 0.5, -0.5),
            int(sample["valid_width"]),
            source.size(-2),
            fill=False,
            edgecolor="red",
            linewidth=1.8,
        )
    )
    axes[0].axvline(float(sample["global_cut"]) - 0.5, color="gold", linewidth=1.5)

    target = sample["target"]
    cut = int(sample["cut"])
    _show(axes[1], target, "2. Selected clean 64px window")
    axes[1].axvline(cut - 0.5, color="gold", linewidth=1.5)
    _show(axes[2], sample["left_image"], "3. Clean left input (unchanged)")
    _show(axes[3], sample["trace_clean_right"], "4. Clean right crop")
    _show(axes[4], sample["boundary_erase_mask"], "5. Boundary erase mask", mask=True)
    _show(axes[5], sample["trace_after_boundary_erase"], "6. Right after boundary erase")
    _show(axes[6], sample["trace_after_photometric"], "7. After random ink noise")
    _show(axes[7], sample["trace_centered_right"], "8. Standalone right centered")
    _show(axes[8], sample["trace_rotation_only"], "9. Rotation only (diagnostic)")
    _show(
        axes[9],
        sample["trace_before_translation"],
        "10. Rotation + scale/shear, no shift",
    )
    _show(axes[10], sample["right_image"], "11. Final affine + translation")
    _show(axes[11], sample["naive_concat"], "12. Final input pair (view only)")
    _show(axes[12], target, "13. Clean 64px training target")
    error = (sample["naive_concat"] - target).abs().div(2).clamp(0, 1)
    _show(axes[13], error, "14. Input-to-target difference", mask=True)
    _show(axes[14], sample["right_corruption_mask"], "15. Right corruption mask", mask=True)
    axes[15].axis("off")
    axes[15].text(
        0.02,
        0.95,
        "Geometry used by training\n\n"
        f"angle: {float(sample['angle']):.2f}°\n"
        f"translate x/y: {int(sample['translate_x'])}, {int(sample['translate_y'])}\n"
        f"scale: {float(sample['scale']):.3f}\n"
        f"shear: {float(sample['shear']):.2f}°\n\n"
        "Panels 9–10 are diagnostics.\n"
        "Training applies panel 11 in\n"
        "one interpolation operation.",
        va="top",
        fontsize=11,
    )

    title = (
        f"slot #{requested_index} → record #{int(sample['record_index'])}: {sample['text']} | "
        f"x={window_start}, angle={float(sample['angle']):.2f}°, "
        f"translate=({int(sample['translate_x'])},{int(sample['translate_y'])}), "
        f"scale={float(sample['scale']):.3f}, shear={float(sample['shear']):.2f}°"
    )
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    config["seed"] = args.seed
    seed_everything(args.seed)
    dataset = build_dataset(
        config,
        stage="merge",
        split=args.split,
        deterministic=True,
        force_corruption_type=args.suite,
        include_trace=True,
    )
    if args.indices:
        indices = args.indices
    else:
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        indices = indices[: min(args.num_samples, len(indices))]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError(f"indices must be in [0, {len(dataset) - 1}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for order, index in enumerate(indices, 1):
        sample = dataset[index]
        filename = f"sample_{order:03d}_record_{int(sample['record_index'])}.png"
        save_sample(sample, index, output_dir / filename)
        records.append(
            {
                "file": filename,
                "requested_index": index,
                "record_index": int(sample["record_index"]),
                "text": sample["text"],
                "path": sample["path"],
                "window_start": int(sample["global_cut"]) - int(sample["cut"]),
                "angle": float(sample["angle"]),
                "translate_x": int(sample["translate_x"]),
                "translate_y": int(sample["translate_y"]),
                "scale": float(sample["scale"]),
                "shear": float(sample["shear"]),
                "boundary_erased_pixels": int(sample["boundary_erase_mask"].sum()),
            }
        )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} step-by-step visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
