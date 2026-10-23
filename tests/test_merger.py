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
    _content_preserving_affine,
    _merge_window,
    make_merge_sample,
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
        rows.append(f"{relative}\tx\n")
        (root / "train_gt.txt").write_text("".join(rows), encoding="utf-8")
        (root / "val_gt.txt").write_text("".join(rows), encoding="utf-8")
        temporary.root = root  # type: ignore[attr-defined]
        return temporary

    def test_datasets_create_binary_masks(self) -> None:
        temporary = self._dataset_root()
        try:
            for stage in ("merge",):
                dataset = IAMMergerDataset(
                    temporary.root, "train_gt.txt", stage, deterministic=True  # type: ignore[attr-defined]
                )
                sample = dataset[0]
                self.assertEqual(len(dataset), 4)
                self.assertEqual(sample["target"].shape, (1, 32, 64))
                self.assertEqual(sample["left_image"].shape, (1, 32, 32))
                self.assertEqual(sample["right_image"].shape, (1, 32, 32))
                self.assertTrue(
                    torch.equal(
                        torch.cat((sample["left_image"], sample["right_image"]), dim=-1),
                        sample["naive_concat"],
                    )
                )
                self.assertEqual(sample["prefix_context"].shape[0:2], (1, 32))
                self.assertEqual(sample["prefix_valid_width"], sample["global_cut"])
                self.assertGreaterEqual(sample["prefix_context"].shape[-1], sample["global_cut"])
                self.assertEqual(sample["prefix_context"].shape[-1] % 32, 0)
                for key in ("corruption_mask", "edit_mask", "valid_mask"):
                    self.assertTrue(set(torch.unique(sample[key]).tolist()).issubset({0.0, 1.0}))
                coverage = float(sample["edit_mask"].sum()) / 32
                self.assertEqual(coverage, 64)
                corruption_coverage = float(sample["corruption_mask"].sum())
                self.assertLessEqual(corruption_coverage, float(sample["edit_mask"].sum()))
                self.assertTrue(torch.equal(sample["left_image"], sample["target"][..., :32]))
                self.assertGreater(float((sample["right_image"] == 1.0).sum()), 0)
                self.assertGreaterEqual(float((sample["target"][..., :32] < 0.9).sum()), 8)
                self.assertGreaterEqual(float((sample["target"][..., 32:] < 0.9).sum()), 8)
        finally:
            temporary.cleanup()

    def test_forced_rotation_and_mask_keep_shapes(self) -> None:
        target = torch.ones(1, 32, 64)
        target[:, 12:20, 4:60] = -1
        cfg = CorruptionConfig()
        for kind in ("rotation", "translation", "shear_scale", "mask", "combined", "clean"):
            sample = make_merge_sample(
                target,
                64,
                cfg,
                __import__("random").Random(7),
                torch.Generator().manual_seed(7),
                force_type=kind,
            )
            self.assertEqual(sample["naive_concat"].shape, target.shape)
            self.assertTrue(set(torch.unique(sample["corruption_mask"]).tolist()).issubset({0.0, 1.0}))
            self.assertGreater(float((sample["right_image"] < 0.9).sum()), 0)
            if kind == "mask":
                self.assertEqual(float(sample["boundary_erase_mask"].sum()), 32 * 2)
                self.assertTrue(
                    torch.all(sample["right_image"][sample["boundary_erase_mask"].bool()] == 1.0)
                )

    def test_augmentation_does_not_tint_empty_background(self) -> None:
        target = torch.ones(1, 32, 64)
        cfg = CorruptionConfig(boundary_erase_probability=0.0)
        for kind in ("rotation", "translation", "shear_scale", "photometric", "combined"):
            sample = make_merge_sample(
                target,
                64,
                cfg,
                __import__("random").Random(11),
                torch.Generator().manual_seed(11),
                force_type=kind,
            )
            self.assertTrue(torch.equal(sample["right_image"], torch.ones(1, 32, 32)))

    def test_affine_keeps_all_compact_ink_inside_right_patch(self) -> None:
        target = torch.ones(1, 32, 64)
        target[:, 10:22, 38:58] = -1
        cfg = CorruptionConfig(boundary_erase_probability=0.0)
        original_ink = float(target[..., 32:].lt(0.9).sum())
        for kind in ("rotation", "translation", "shear_scale"):
            sample = make_merge_sample(
                target,
                64,
                cfg,
                __import__("random").Random(7),
                torch.Generator().manual_seed(7),
                force_type=kind,
            )
            transformed_ink = float(sample["right_image"].lt(0.9).sum())
            # Bilinear interpolation changes edge pixels, but the padded affine
            # and fit-back step must retain the complete stroke body.
            self.assertGreater(transformed_ink, 0.70 * original_ink)

    def test_window_sampling_is_centered_and_keeps_ink_in_both_halves(self) -> None:
        image = torch.ones(1, 32, 256)
        image[:, 12:20, 32:224] = -1
        cfg = CorruptionConfig()
        starts = []
        for seed in range(200):
            target, _, global_cut, _ = _merge_window(
                image, cfg, __import__("random").Random(seed)
            )
            starts.append(global_cut - cfg.patch_width)
            self.assertGreaterEqual(float((target[..., :32] < 0.9).sum()), 8)
            self.assertGreaterEqual(float((target[..., 32:] < 0.9).sum()), 8)
        expected_center = (image.size(-1) - cfg.window_width) / 2
        self.assertLess(abs(sum(starts) / len(starts) - expected_center), 5.0)
        self.assertGreater(sum(48 <= start <= 144 for start in starts), 140)

    def test_right_patch_translation_is_not_canceled_by_fit_back(self) -> None:
        right = torch.ones(1, 32, 32)
        right[:, 8:24, 10:18] = -1
        centered = _content_preserving_affine(
            right, angle=0, tx=0, ty=0, scale=1, shear=0, margin_x=6, margin_y=4
        )
        shifted = _content_preserving_affine(
            right, angle=0, tx=5, ty=0, scale=1, shear=0, margin_x=6, margin_y=4
        )

        def ink_center_x(image: torch.Tensor) -> float:
            points = torch.nonzero(image[0] < 0.9, as_tuple=False)
            return float(points[:, 1].float().mean())

        self.assertGreater(ink_center_x(shifted) - ink_center_x(centered), 3.5)

    def test_model_generates_full_merge_and_backward(self) -> None:
        model = MergerNet(
            patch_width=32,
            window_width=64,
            vit_patch_size=4,
            embed_dim=32,
            transformer_layers=1,
            prefix_transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=64,
            dropout=0.0,
        )
        left_image = torch.rand(2, 1, 32, 32) * 2 - 1
        right_image = torch.rand(2, 1, 32, 32) * 2 - 1
        prefix = torch.rand(2, 1, 32, 64) * 2 - 1
        prefix_mask = torch.ones_like(prefix)
        left_corruption = torch.zeros_like(left_image)
        right_corruption = torch.zeros_like(right_image)
        target = torch.cat((left_image, right_image), dim=-1)
        edit = torch.ones_like(target)
        prediction = model(
            left_image,
            right_image,
            left_corruption,
            right_corruption,
            prefix,
            prefix_mask,
        )
        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(torch.all(prediction >= -1.0))
        self.assertTrue(torch.all(prediction <= 1.0))
        loss, parts = MergerLoss()(
            prediction,
            target,
            edit,
            torch.ones_like(edit),
            input_image=target,
            left_preservation_mask=torch.cat(
                (torch.ones_like(left_image), torch.zeros_like(right_image)), dim=-1
            ),
        )
        self.assertEqual(set(parts), {"loss", "reconstruction_mse", "left_identity_mse"})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        metrics = compute_metrics(prediction.detach(), target, edit, torch.ones_like(edit))
        self.assertEqual(metrics["outside_mask_mae"], 0.0)

    def test_collate_pads_prefix_after_encoding_input(self) -> None:
        temporary = self._dataset_root()
        try:
            dataset = IAMMergerDataset(
                temporary.root, "train_gt.txt", "merge", deterministic=True  # type: ignore[attr-defined]
            )
            batch = merger_collate([dataset[0], dataset[1]])
            self.assertEqual(batch["prefix_context"].shape, batch["prefix_mask"].shape)
            for index, width in enumerate(batch["prefix_valid_width"].tolist()):
                self.assertTrue(torch.all(batch["prefix_mask"][index, ..., :width] == 1))
                self.assertTrue(torch.all(batch["prefix_mask"][index, ..., width:] == 0))
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
