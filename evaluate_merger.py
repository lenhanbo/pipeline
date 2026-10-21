from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from merger.data import merger_collate
from merger.runtime import build_dataset, build_model, evaluate_loader, load_config, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate merger corruption suites")
    parser.add_argument("--config", default="configs/merger_iam.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=("continuation", "seam"), default="seam")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    seed_everything(int(config.get("seed", 123456)))
    requested = args.device or config.get("device", "cuda")
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / "evaluation")
    suites = ["continuation"] if args.stage == "continuation" else [
        "clean", "mask", "translation", "rotation", "shear_scale", "photometric", "combined"
    ]
    results = {}
    for suite in suites:
        stage = "continuation" if suite == "continuation" else "seam"
        dataset = build_dataset(
            config,
            stage,
            "val",
            deterministic=True,
            force_corruption_type=None if stage == "continuation" else suite,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            collate_fn=merger_collate,
        )
        results[suite] = evaluate_loader(
            model,
            loader,
            device,
            max_batches=int(config["training"].get("validation_batches", 50)),
            preview_path=output_dir / f"{suite}.png",
        )
        print(suite, results[suite])
    save_json(output_dir / "metrics.json", results)


if __name__ == "__main__":
    main()
