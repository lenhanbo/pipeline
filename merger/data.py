from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class CorruptionConfig:
    patch_width: int = 32
    window_width: int = 64
    image_height: int = 32
    edit_width: int = 32
    center_gap_width: int = 2
    min_side_width: int = 32
    continuation_rotation_degrees: float = 3.0
    continuation_translate_x: int = 3
    continuation_translate_y: int = 2
    continuation_shear_degrees: float = 2.0
    continuation_scale_min: float = 0.97
    continuation_scale_max: float = 1.03
    continuation_geometry_probability: float = 0.9
    continuation_noise_probability: float = 0.8
    continuation_noise_std_min: float = 0.005
    continuation_noise_std_max: float = 0.04
    continuation_blur_probability: float = 0.25
    mask_probability: float = 0.8
    rotation_degrees: float = 8.0
    translate_x: int = 6
    translate_y: int = 4
    shear_degrees: float = 8.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    geometry_identity_probability: float = 0.2


def _read_records(data_root: Path, gt_file: str) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    gt_path = data_root / gt_file
    with gt_path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            raw = raw.rstrip("\n\r")
            if not raw:
                continue
            try:
                relative, text = raw.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"invalid ground-truth line {line_number}: {raw!r}") from exc
            path = data_root.joinpath(*relative.split("/"))
            records.append((path, text))
    if not records:
        raise ValueError(f"no samples found in {gt_path}")
    return records


def _to_normalized_tensor(image: Image.Image, height: int) -> torch.Tensor:
    image = image.convert("L")
    width = max(1, round(image.width * height / image.height))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).mul(2.0).sub(1.0)


def _merge_window(
    image: torch.Tensor, cfg: CorruptionConfig, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return [last prefix patch | next patch] and the complete raw prefix."""
    expected_width = 2 * cfg.patch_width
    if cfg.window_width != expected_width:
        raise ValueError(
            f"window_width must equal 2 * patch_width ({expected_width}), got {cfg.window_width}"
        )
    padded_width = max(
        expected_width,
        math.ceil(image.size(-1) / cfg.patch_width) * cfg.patch_width,
    )
    if image.size(-1) != padded_width:
        padded = torch.ones((1, image.size(-2), padded_width), dtype=image.dtype)
        padded[..., : image.size(-1)] = image
        image = padded

    # Simulate the boundary between generated characters: one 32 px character patch
    # becomes one prefix token, so every cut is patch-aligned.
    cut_patch = rng.randint(1, image.size(-1) // cfg.patch_width - 1)
    global_cut = cut_patch * cfg.patch_width
    start = global_cut - cfg.patch_width
    target = image[..., start : start + cfg.window_width]

    # Do not pool pixels here. The model must encode the complete prefix first,
    # then pool its feature sequence (the same ordering used by an LM encoder).
    full_prefix = image[..., :global_cut]
    return target, full_prefix, global_cut


def merger_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-width prefixes per batch and return a valid-pixel mask."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    contexts = [sample["prefix_context"] for sample in samples]
    max_width = max(context.size(-1) for context in contexts)
    batch_context = contexts[0].new_ones(
        (len(contexts), 1, contexts[0].size(-2), max_width)
    )
    prefix_mask = contexts[0].new_zeros(batch_context.shape)
    for index, context in enumerate(contexts):
        width = context.size(-1)
        batch_context[index, :, :, :width] = context
        prefix_mask[index, :, :, :width] = 1.0

    fixed = [{key: value for key, value in sample.items() if key != "prefix_context"} for sample in samples]
    batch = default_collate(fixed)
    batch["prefix_context"] = batch_context
    batch["prefix_mask"] = prefix_mask
    return batch


def _column_mask(height: int, width: int, start: int, end: int) -> torch.Tensor:
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[..., max(0, start) : min(width, end)] = 1.0
    return mask


def _draw_occlusions(
    mask: torch.Tensor,
    left: int,
    right: int,
    rng: random.Random,
) -> None:
    height = mask.size(-2)
    region_width = max(1, right - left)
    target = rng.uniform(0.10, 0.35) * height * region_width
    covered_before = float(mask[..., left:right].sum())

    if rng.random() < 0.7:
        band_width = rng.randint(2, min(8, region_width))
        center = rng.randint(left, max(left, right - 1))
        mask[..., max(left, center - band_width // 2) : min(right, center + math.ceil(band_width / 2))] = 1

    for _ in range(rng.randint(1, 3)):
        rect_width = rng.randint(2, min(12, region_width))
        rect_height = rng.randint(4, min(16, height))
        x0 = rng.randint(left, max(left, right - rect_width))
        y0 = rng.randint(0, height - rect_height)
        mask[..., y0 : y0 + rect_height, x0 : x0 + rect_width] = 1
        if float(mask[..., left:right].sum()) - covered_before >= target:
            break

    if rng.random() < 0.35:
        thickness = rng.randint(1, 3)
        x0 = rng.randint(left, max(left, right - 1))
        slope = rng.uniform(-0.4, 0.4)
        for y in range(height):
            x = round(x0 + slope * (y - height / 2))
            mask[..., y, max(left, x - thickness) : min(right, x + thickness + 1)] = 1

    region = mask[..., left:right]
    while float(region.sum()) - covered_before < target:
        rect_width = rng.randint(2, min(6, region_width))
        rect_height = rng.randint(2, min(8, height))
        x0 = rng.randint(0, max(0, region_width - rect_width))
        y0 = rng.randint(0, height - rect_height)
        region[..., y0 : y0 + rect_height, x0 : x0 + rect_width] = 1
    max_pixels = int(0.5 * height * region_width)
    if int(region.sum()) > max_pixels:
        indices = torch.nonzero(region[0], as_tuple=False).tolist()
        rng.shuffle(indices)
        for y, x in indices[max_pixels:]:
            region[0, y, x] = 0


def _photometric(right: torch.Tensor, rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    image01 = right.add(1).div(2)
    contrast = rng.uniform(0.8, 1.2)
    brightness = rng.uniform(-0.08, 0.08)
    image01 = ((image01 - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
    if rng.random() < 0.35:
        sigma = rng.uniform(0.2, 1.0)
        image01 = TF.gaussian_blur(image01, kernel_size=[3, 3], sigma=[sigma, sigma])
    if rng.random() < 0.35:
        noise = torch.randn(image01.shape, generator=generator) * rng.uniform(0.005, 0.035)
        image01 = (image01 + noise).clamp(0, 1)
    if rng.random() < 0.25:
        ink = 1.0 - image01
        if rng.random() < 0.5:
            ink = torch.nn.functional.max_pool2d(ink.unsqueeze(0), 3, 1, 1).squeeze(0)
        else:
            ink = -torch.nn.functional.max_pool2d(-ink.unsqueeze(0), 3, 1, 1).squeeze(0)
        image01 = 1.0 - ink
    return image01.mul(2).sub(1)


def make_continuation_sample(
    target: torch.Tensor,
    valid_width: int,
    cfg: CorruptionConfig,
    rng: random.Random,
    generator: torch.Generator,
) -> dict[str, torch.Tensor | int | str]:
    """Create a mild registration-and-denoising task without hiding content."""
    valid_width = max(1, min(valid_width, cfg.window_width))
    use_geometry = rng.random() < cfg.continuation_geometry_probability
    angle = rng.uniform(-cfg.continuation_rotation_degrees, cfg.continuation_rotation_degrees) if use_geometry else 0.0
    tx = rng.randint(-cfg.continuation_translate_x, cfg.continuation_translate_x) if use_geometry else 0
    ty = rng.randint(-cfg.continuation_translate_y, cfg.continuation_translate_y) if use_geometry else 0
    shear = rng.uniform(-cfg.continuation_shear_degrees, cfg.continuation_shear_degrees) if use_geometry else 0.0
    scale = rng.uniform(cfg.continuation_scale_min, cfg.continuation_scale_max) if use_geometry else 1.0

    cut = cfg.patch_width
    left_mask = _column_mask(cfg.image_height, cfg.window_width, 0, cut)
    right_mask = _column_mask(cfg.image_height, cfg.window_width, cut, valid_width)
    right = target * right_mask + (1.0 - right_mask)
    image01 = right.add(1).div(2)
    if rng.random() < cfg.continuation_blur_probability:
        sigma = rng.uniform(0.2, 0.8)
        image01 = TF.gaussian_blur(image01, kernel_size=[3, 3], sigma=[sigma, sigma])
    if rng.random() < cfg.continuation_noise_probability:
        std = rng.uniform(cfg.continuation_noise_std_min, cfg.continuation_noise_std_max)
        noise = torch.randn(image01.shape, generator=generator) * std
        image01 = image01 + noise * right_mask
    right = image01.clamp(0, 1).mul(2).sub(1)
    right = right * right_mask + (1.0 - right_mask)

    right = TF.affine(
        right,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[shear, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=1.0,
        center=[0.5 * (cut + valid_width), cfg.image_height / 2],
    )
    warped_right_mask = TF.affine(
        right_mask,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[shear, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
        center=[0.5 * (cut + valid_width), cfg.image_height / 2],
    )
    corrupted = target.clone()
    corrupted[..., cut:valid_width] = 1.0
    corrupted = torch.minimum(corrupted, right)

    valid_mask = _column_mask(cfg.image_height, cfg.window_width, 0, valid_width)
    edit_mask = valid_mask.clone()  # predict both adjacent patches
    corruption_mask = torch.maximum(right_mask, warped_right_mask)
    seam_mask = _column_mask(cfg.image_height, cfg.window_width, cut - 1, cut + 1)
    left_preservation_mask = left_mask * (1.0 - corruption_mask)
    return {
        "target": target,
        "corrupted": corrupted,
        "corruption_mask": corruption_mask,
        "edit_mask": edit_mask,
        "valid_mask": valid_mask,
        "left_preservation_mask": left_preservation_mask,
        "seam_mask": seam_mask,
        "cut": cut,
        "corruption_type": "alignment_noise",
    }


def make_seam_sample(
    target: torch.Tensor,
    valid_width: int,
    cfg: CorruptionConfig,
    rng: random.Random,
    generator: torch.Generator,
    force_type: str | None = None,
    cut: int | None = None,
) -> dict[str, torch.Tensor | int | str]:
    min_side = min(cfg.min_side_width, max(1, valid_width // 3))
    cut = cut if cut is not None else rng.randint(min_side, max(min_side, valid_width - min_side))
    valid_mask = _column_mask(cfg.image_height, cfg.window_width, 0, valid_width)
    edit_mask = valid_mask.clone()  # both patches are predicted and may be adjusted
    half_edit = cfg.edit_width // 2
    edit_left, edit_right = max(0, cut - half_edit), min(valid_width, cut + half_edit)
    left_mask = _column_mask(cfg.image_height, cfg.window_width, 0, cut)

    allowed = {"clean", "rotation", "translation", "shear_scale", "photometric", "mask", "combined"}
    corruption_type = force_type or "combined"
    if corruption_type not in allowed:
        raise ValueError(f"unknown corruption type: {corruption_type}")
    right = torch.ones_like(target)
    right[..., cut:valid_width] = target[..., cut:valid_width]
    if corruption_type in {"photometric", "combined"}:
        right = _photometric(right, rng, generator)
        # Photometric changes belong to the candidate/right component only;
        # otherwise a brightness offset would tint the preserved prefix too.
        component_mask = _column_mask(cfg.image_height, cfg.window_width, cut, valid_width)
        right = right * component_mask + (1.0 - component_mask)
    use_geometry = corruption_type not in {"clean", "photometric", "mask"}
    identity = rng.random() < cfg.geometry_identity_probability
    angle = 0.0 if identity or not use_geometry else rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
    tx = 0 if identity or not use_geometry else rng.randint(-cfg.translate_x, cfg.translate_x)
    ty = 0 if identity or not use_geometry else rng.randint(-cfg.translate_y, cfg.translate_y)
    shear = 0.0 if identity or not use_geometry else rng.uniform(-cfg.shear_degrees, cfg.shear_degrees)
    scale = 1.0 if identity or not use_geometry else rng.uniform(cfg.scale_min, cfg.scale_max)
    if corruption_type == "rotation":
        tx = ty = 0
        shear, scale = 0.0, 1.0
    elif corruption_type == "translation":
        angle, shear, scale = 0.0, 0.0, 1.0
    elif corruption_type == "shear_scale":
        angle, tx, ty = 0.0, 0, 0

    right_valid = _column_mask(cfg.image_height, cfg.window_width, cut, valid_width)
    if use_geometry:
        center = [0.5 * (cut + valid_width), cfg.image_height / 2]
        right = TF.affine(
            right,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[shear, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
            center=center,
        )
        warped_valid = TF.affine(
            right_valid,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[shear, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
            center=center,
        )
    else:
        warped_valid = right_valid

    corrupted = target.clone()
    corrupted[..., cut:valid_width] = 1.0
    corrupted = torch.minimum(corrupted, right)
    corruption_mask = torch.maximum(right_valid, warped_valid)

    # Always remove a minimal vertical seam after splitting. This prevents the
    # network from solving seam restoration by simply copying two already
    # connected halves. With width=2, one column is removed on each side.
    gap_width = max(1, cfg.center_gap_width)
    gap_left = max(edit_left, cut - gap_width // 2)
    gap_right = min(edit_right, gap_left + gap_width)
    center_gap = _column_mask(cfg.image_height, cfg.window_width, gap_left, gap_right) * valid_mask
    overlap = warped_valid * left_mask
    seam_removal = torch.maximum(center_gap, overlap)
    corrupted[seam_removal.bool()] = 1.0
    corruption_mask = torch.maximum(corruption_mask, seam_removal)

    add_mask = corruption_type == "mask" or (
        corruption_type == "combined" and rng.random() < cfg.mask_probability
    )
    if add_mask:
        occlusion = torch.zeros_like(target)
        _draw_occlusions(occlusion, edit_left, edit_right, rng)
        occlusion *= valid_mask
        corrupted[occlusion.bool()] = 1.0
        corruption_mask = torch.maximum(corruption_mask, occlusion)

    if corruption_type == "clean":
        # Clean-seam keeps the mandatory center gap but disables all other
        # perturbations, making it a focused boundary-repair evaluation.
        corrupted = target.clone()
        corrupted[center_gap.bool()] = 1.0
        corruption_mask = center_gap
        seam_removal = center_gap
    left_preservation_mask = left_mask * (1.0 - corruption_mask)
    return {
        "target": target,
        "corrupted": corrupted,
        "corruption_mask": corruption_mask,
        "edit_mask": edit_mask,
        "valid_mask": valid_mask,
        "left_preservation_mask": left_preservation_mask,
        "seam_mask": seam_removal,
        "cut": cut,
        "corruption_type": corruption_type,
    }


class IAMMergerDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        gt_file: str,
        stage: str,
        corruption: CorruptionConfig | None = None,
        deterministic: bool = False,
        seed: int = 123456,
        max_samples: int | None = None,
        force_corruption_type: str | None = None,
    ) -> None:
        if stage not in {"continuation", "seam"}:
            raise ValueError("stage must be 'continuation' or 'seam'")
        self.data_root = Path(data_root)
        self.records = _read_records(self.data_root, gt_file)
        if max_samples:
            self.records = self.records[:max_samples]
        self.stage = stage
        self.cfg = corruption or CorruptionConfig()
        self.deterministic = deterministic
        self.seed = seed
        self.force_corruption_type = force_corruption_type

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int) -> tuple[random.Random, torch.Generator]:
        seed = self.seed + index if self.deterministic else random.randrange(2**31)
        return random.Random(seed), torch.Generator().manual_seed(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng, generator = self._rng(index)
        path, text = self.records[index]
        with Image.open(path) as image:
            tensor = _to_normalized_tensor(image, self.cfg.image_height)
        target, prefix_context, global_cut = _merge_window(tensor, self.cfg, rng)
        valid_width = self.cfg.window_width

        if self.stage == "continuation":
            sample = make_continuation_sample(target, valid_width, self.cfg, rng, generator)
        else:
            sample = make_seam_sample(
                target,
                valid_width,
                self.cfg,
                rng,
                generator,
                force_type=self.force_corruption_type,
                cut=self.cfg.patch_width,
            )
        sample.update(
            {
                "path": str(path),
                "text": text,
                "valid_width": valid_width,
                "global_cut": global_cut,
                "prefix_context": prefix_context,
            }
        )
        return sample
