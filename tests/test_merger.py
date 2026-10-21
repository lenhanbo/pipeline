from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from merger.data import (
    CorruptionConfig,
    IAMMergerDataset,
    make_continuation_sample,
    make_seam_sample,
    merger_collate,
)
from merger.losses import MergerLoss
from merger.metrics import compute_metrics
from merger.model import MergerNet


class MergerTests(unittest.TestCase):
    def _dataset_root(self) -> tempfile.TemporaryDirectory:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        words = root / "words" / "a01" / "a01-000u"
        words.mkdir(parents=True)
        rows = []
        for index in range(4):
            image = Image.new("L", (96 + index * 8, 32), 255)
            draw = ImageDraw.Draw(image)
            draw.line((4, 20, image.width - 4, 10 + index), fill=0, width=2)
            draw.arc((20, 8, 60, 28), 0, 300, fill=20, width=2)
            relative = f"words/a01/a01-000u/a01-000u-00-0{index}.png"
            image.save(root / relative)
            rows.append(f"{relative}\ttest\n")
        (root / "train_gt.txt").write_text("".join(rows), encoding="utf-8")
        (root / "val_gt.txt").write_text("".join(rows), encoding="utf-8")
        temporary.root = root  # type: ignore[attr-defined]
        return temporary

    def test_datasets_create_binary_masks(self) -> None:
        temporary = self._dataset_root()
        try:
            for stage in ("continuation", "seam"):
                dataset = IAMMergerDataset(
                    temporary.root, "train_gt.txt", stage, deterministic=True  # type: ignore[attr-defined]
                )
                sample = dataset[0]
                self.assertEqual(sample["target"].shape, (1, 32, 32))
                self.assertEqual(sample["prefix_context"].shape[0:2], (1, 32))
                self.assertEqual(sample["prefix_context"].shape[-1], sample["global_cut"])
                for key in ("corruption_mask", "edit_mask", "valid_mask"):
                    self.assertTrue(set(torch.unique(sample[key]).tolist()).issubset({0.0, 1.0}))
                if stage == "seam":
                    coverage = float(sample["edit_mask"].sum()) / 32
                    self.assertEqual(coverage, 32)
                    corruption_coverage = float(sample["corruption_mask"].sum())
                    self.assertLessEqual(corruption_coverage, float(sample["edit_mask"].sum()))
                    self.assertTrue(torch.all(sample["corrupted"][sample["seam_mask"].bool()] == 1.0))
                    self.assertEqual(
                        float((sample["left_preservation_mask"] * sample["seam_mask"]).sum()), 0.0
                    )
        finally:
            temporary.cleanup()

    def test_forced_rotation_and_mask_keep_shapes(self) -> None:
        target = torch.ones(1, 32, 32)
        target[:, 12:20, 4:28] = -1
        cfg = CorruptionConfig()
        for kind in ("rotation", "translation", "shear_scale", "mask", "combined", "clean"):
            sample = make_seam_sample(
                target,
                32,
                cfg,
                __import__("random").Random(7),
                torch.Generator().manual_seed(7),
                force_type=kind,
            )
            self.assertEqual(sample["corrupted"].shape, target.shape)
            self.assertTrue(set(torch.unique(sample["corruption_mask"]).tolist()).issubset({0.0, 1.0}))
            cut = int(sample["cut"])
            self.assertGreater(float(sample["corruption_mask"][..., cut - 1 : cut + 1].sum()), 0)

    def test_continuation_keeps_content_visible_and_uses_mild_geometry(self) -> None:
        target = torch.ones(1, 32, 32)
        target[:, 12:20, 4:28] = -1
        cfg = CorruptionConfig(
            continuation_geometry_probability=1.0,
            continuation_noise_probability=1.0,
        )
        sample = make_continuation_sample(
            target,
            32,
            cfg,
            __import__("random").Random(7),
            torch.Generator().manual_seed(7),
        )
        corrupted = sample["corrupted"]
        self.assertEqual(corrupted.shape, target.shape)
        self.assertGreater(float((corrupted < 0.9).sum()), 0)
        self.assertFalse(bool(torch.all(corrupted[sample["edit_mask"].bool()] == 1.0)))
        self.assertEqual(sample["corruption_type"], "alignment_noise")

    def test_model_hard_blend_and_backward(self) -> None:
        model = MergerNet(
            window_width=32,
            vit_patch_size=4,
            embed_dim=32,
            transformer_layers=1,
            prefix_transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=64,
            dropout=0.0,
        )
        image = torch.rand(2, 1, 32, 32) * 2 - 1
        prefix = torch.rand(2, 1, 32, 32) * 2 - 1
        prefix_mask = torch.ones_like(prefix)
        corruption = torch.zeros_like(image)
        edit = torch.zeros_like(image)
        edit[..., 8:24] = 1
        target = image.clone()
        target[..., 8:24] = 0
        prediction = model(image, corruption, edit, prefix, prefix_mask)
        self.assertTrue(torch.equal(prediction * (1 - edit), image * (1 - edit)))
        loss, parts = MergerLoss()(
            prediction,
            target,
            edit,
            torch.ones_like(edit),
            input_image=image,
            left_preservation_mask=(1 - edit),
        )
        self.assertEqual(set(parts), {"loss", "reconstruction_l1", "left_identity_l1"})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        metrics = compute_metrics(prediction.detach(), target, edit, torch.ones_like(edit))
        self.assertEqual(metrics["outside_mask_mae"], 0.0)

    def test_collate_pads_prefix_after_encoding_input(self) -> None:
        temporary = self._dataset_root()
        try:
            dataset = IAMMergerDataset(
                temporary.root, "train_gt.txt", "seam", deterministic=True  # type: ignore[attr-defined]
            )
            batch = merger_collate([dataset[0], dataset[1]])
            self.assertEqual(batch["prefix_context"].shape, batch["prefix_mask"].shape)
            for index, width in enumerate(batch["global_cut"].tolist()):
                self.assertTrue(torch.all(batch["prefix_mask"][index, ..., :width] == 1))
                self.assertTrue(torch.all(batch["prefix_mask"][index, ..., width:] == 0))
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
