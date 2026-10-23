from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from merger.runtime import build_model


class AnalyzeMergerTests(unittest.TestCase):
    def test_cli_writes_ranked_pages_and_filters_single_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "dataset"
            words = data_root / "words"
            words.mkdir(parents=True)
            rows = []
            for index, text in enumerate(("alpha", "beta", "gamma", "x")):
                image = Image.new("L", (96, 32), 255)
                draw = ImageDraw.Draw(image)
                draw.line((4, 20, 90, 8 + index), fill=0, width=2)
                relative = f"words/sample_{index}.png"
                image.save(data_root / relative)
                rows.append(f"{relative}\t{text}\n")
            (data_root / "train_gt.txt").write_text("".join(rows), encoding="utf-8")
            (data_root / "val_gt.txt").write_text("".join(rows), encoding="utf-8")

            config = {
                "seed": 7,
                "device": "cpu",
                "data": {
                    "root": str(data_root),
                    "train_gt": "train_gt.txt",
                    "val_gt": "val_gt.txt",
                    "min_text_length": 2,
                    "max_train_samples": None,
                    "max_val_samples": None,
                },
                "corruption": {
                    "patch_width": 32,
                    "window_width": 64,
                    "image_height": 32,
                    "center_gap_width": 2,
                    "window_center_std_fraction": 0.18,
                    "window_sample_attempts": 16,
                    "record_sample_attempts": 8,
                    "min_half_ink_pixels": 8,
                    "boundary_erase_probability": 0.6,
                },
                "model": {
                    "image_height": 32,
                    "patch_width": 32,
                    "window_width": 64,
                    "vit_patch_size": 4,
                    "embed_dim": 32,
                    "transformer_layers": 1,
                    "transformer_heads": 4,
                    "transformer_ffn_dim": 64,
                    "prefix_transformer_layers": 1,
                    "dropout": 0.0,
                },
                "training": {"batch_size": 2, "num_workers": 0},
            }
            checkpoint = root / "checkpoint.pth"
            torch.save(
                {"model": build_model(config).state_dict(), "config": config, "stage": "merge", "step": 1},
                checkpoint,
            )
            output = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    "analyze_merger.py",
                    "--checkpoint", str(checkpoint),
                    "--stage", "merge",
                    "--suite", "clean",
                    "--selection", "worst_noise",
                    "--candidate-samples", "3",
                    "--num-samples", "2",
                    "--page-size", "2",
                    "--batch-size", "2",
                    "--device", "cpu",
                    "--output-dir", str(output),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertTrue((output / "page_001.png").exists())
            self.assertTrue((output / "samples.csv").exists())
            self.assertTrue((output / "summary.json").exists())
            summary = (output / "summary.json").read_text(encoding="utf-8")
            self.assertIn('"dataset_samples_after_filter": 3', summary)


if __name__ == "__main__":
    unittest.main()
