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
    center_gap_width: int = 2
    window_center_std_fraction: float = 0.18
    window_sample_attempts: int = 16
    record_sample_attempts: int = 8
    min_half_ink_pixels: int = 8
    boundary_erase_probability: float = 0.6
    boundary_erase_width: int = 2
    rotation_degrees: float = 8.0
    translate_x: int = 6
    translate_y: int = 4
    shear_degrees: float = 8.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    geometry_identity_probability: float = 0.2
    noise_probability: float = 0.30
    noise_std_min: float = 0.005
    noise_std_max: float = 0.025
    standalone_margin_x: int = 6
    standalone_margin_y: int = 4


def _read_records(
    data_root: Path,
    gt_file: str,
    min_text_length: int = 2,
) -> list[tuple[Path, str]]:
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
            if sum(not character.isspace() for character in text) < min_text_length:
                continue
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
    # Estimate paper and foreground once on the complete word, before choosing
    # the 64px window. Otsu separates dark handwriting from scanner/paper tone;
    # the small margin retains antialiased stroke edges. Values above the cutoff
    # become exact white in both halves, so geometry cannot rotate a gray paper
    # rectangle into only the right input.
    paper = max(float(np.percentile(array, 95)), 0.5)
    array = np.clip(array / paper, 0.0, 1.0)
    histogram, _ = np.histogram(array, bins=256, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64) / max(int(histogram.sum()), 1)
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256, dtype=np.float64))
    total_mean = cumulative_mean[-1]
    variance = (total_mean * cumulative_probability - cumulative_mean) ** 2
    variance /= cumulative_probability * (1.0 - cumulative_probability) + 1e-12
    otsu = float(np.argmax(variance)) / 255.0
    paper_cutoff = min(0.95, max(0.35, otsu + 0.12))
    array = np.clip(array / paper_cutoff, 0.0, 1.0)
    return torch.from_numpy(array).unsqueeze(0).mul(2.0).sub(1.0)


def _merge_window(
    image: torch.Tensor, cfg: CorruptionConfig, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Sample a horizontal 64-pixel interval and retain the complete prefix."""
    expected_width = 2 * cfg.patch_width
    if cfg.window_width != expected_width:
        raise ValueError(
            f"window_width must equal 2 * patch_width ({expected_width}), got {cfg.window_width}"
        )
    padded_width = max(expected_width, image.size(-1))
    if image.size(-1) != padded_width:
        padded = torch.ones((1, image.size(-2), padded_width), dtype=image.dtype)
        padded[..., : image.size(-1)] = image
        image = padded

    max_start = image.size(-1) - cfg.window_width
    center_start = max_start / 2.0
    start_std = max(1.0, max_start * cfg.window_center_std_fraction)
    start = int(round(center_start))
    best_score = -1
    # Draw from a truncated normal distribution around the word center. A draw
    # is accepted only when both 32px halves contain ink; otherwise retain the
    # best candidate seen so words with an unavoidable blank half still work.
    for _ in range(max(1, cfg.window_sample_attempts)):
        candidate = round(rng.gauss(center_start, start_std)) if max_start else 0
        candidate = min(max(int(candidate), 0), max_start)
        window = image[..., candidate : candidate + cfg.window_width]
        left_ink = int((window[..., : cfg.patch_width] < 0.9).sum())
        right_ink = int((window[..., cfg.patch_width :] < 0.9).sum())
        score = min(left_ink, right_ink)
        if score > best_score:
            start, best_score = candidate, score
        if score >= cfg.min_half_ink_pixels:
            start = candidate
            break
    global_cut = start + cfg.patch_width
    target = image[..., start : start + cfg.window_width]

    # Do not pool pixels here. The model must encode the complete prefix first,
    # then pool its feature sequence (the same ordering used by an LM encoder).
    raw_prefix = image[..., :global_cut]
    prefix_valid_width = raw_prefix.size(-1)
    prefix_width = math.ceil(prefix_valid_width / cfg.patch_width) * cfg.patch_width
    full_prefix = image.new_ones((1, cfg.image_height, prefix_width))
    full_prefix[..., :prefix_valid_width] = raw_prefix
    return target, full_prefix, global_cut, prefix_valid_width


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
        valid_width = int(samples[index]["prefix_valid_width"])
        batch_context[index, :, :, :width] = context
        prefix_mask[index, :, :, :valid_width] = 1.0

    fixed = [{key: value for key, value in sample.items() if key != "prefix_context"} for sample in samples]
    batch = default_collate(fixed)
    batch["prefix_context"] = batch_context
    batch["prefix_mask"] = prefix_mask
    return batch


def _column_mask(height: int, width: int, start: int, end: int) -> torch.Tensor:
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[..., max(0, start) : min(width, end)] = 1.0
    return mask


def _photometric(
    right: torch.Tensor,
    cfg: CorruptionConfig,
    rng: random.Random,
    generator: torch.Generator,
) -> torch.Tensor:
    # Work on ink opacity rather than image brightness. Zero ink stays exactly
    # zero, so augmentation can never introduce a mismatched gray background.
    ink = right.neg().add(1).div(2).clamp(0, 1)
    if rng.random() < cfg.noise_probability:
        support = (ink > 0.01).float()
        noise = torch.randn(ink.shape, generator=generator) * rng.uniform(
            cfg.noise_std_min, cfg.noise_std_max
        )
        ink = (ink + noise * support).clamp(0, 1)
    return 1.0 - 2.0 * ink.clamp(0, 1)


def _ink_bbox(ink: torch.Tensor, threshold: float = 1e-4) -> tuple[int, int, int, int] | None:
    points = torch.nonzero(ink[0] > threshold, as_tuple=False)
    if points.numel() == 0:
        return None
    y0, x0 = points.amin(dim=0).tolist()
    y1, x1 = (points.amax(dim=0) + 1).tolist()
    return int(y0), int(y1), int(x0), int(x1)


def _erase_boundary_columns(
    right: torch.Tensor,
    cfg: CorruptionConfig,
    rng: random.Random,
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    erase_mask = torch.zeros_like(right)
    if not force and rng.random() >= cfg.boundary_erase_probability:
        return right, erase_mask
    width = min(max(0, cfg.boundary_erase_width), right.size(-1))
    erase_mask[..., :width] = 1.0
    erased = right.clone()
    erased[erase_mask.bool()] = 1.0
    return erased, erase_mask


def _center_standalone_patch(
    right: torch.Tensor,
    margin_x: int,
    margin_y: int,
) -> torch.Tensor:
    """Fit the right-side ink into a centered standalone 32px patch."""
    height, width = right.shape[-2:]
    ink = right.neg().add(1).div(2).clamp(0, 1)
    source_bbox = _ink_bbox(ink, threshold=0.01)
    if source_bbox is None:
        return torch.ones_like(right)
    source_y0, source_y1, source_x0, source_x1 = source_bbox
    content = ink[..., source_y0:source_y1, source_x0:source_x1]

    # A generated character patch is normally centered rather than already in
    # its final word position. Reserve explicit margins before affine so x/y
    # translation remains visible instead of being canceled by fit-back.
    available_height = max(1, height - 2 * max(0, margin_y))
    available_width = max(1, width - 2 * max(0, margin_x))
    content_height, content_width = content.shape[-2:]
    if content_height > available_height or content_width > available_width:
        factor = min(available_height / content_height, available_width / content_width)
        content_height = max(1, round(content_height * factor))
        content_width = max(1, round(content_width * factor))
        content = torch.nn.functional.interpolate(
            content.unsqueeze(0),
            size=(content_height, content_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    output_ink = torch.zeros_like(ink)
    paste_y = (height - content_height) // 2
    paste_x = (width - content_width) // 2
    output_ink[..., paste_y : paste_y + content_height, paste_x : paste_x + content_width] = content
    return 1.0 - 2.0 * output_ink.clamp(0, 1)


def _affine_centered_patch(
    centered_right: torch.Tensor,
    angle: float,
    tx: int,
    ty: int,
    scale: float,
    shear: float,
) -> torch.Tensor:
    """Affine an already centered patch on a padded canvas without clipping."""
    height, width = centered_right.shape[-2:]
    padding = max(height, width)
    canvas = centered_right.new_zeros((1, height + 2 * padding, width + 2 * padding))
    ink = centered_right.neg().add(1).div(2).clamp(0, 1)
    canvas[..., padding : padding + height, padding : padding + width] = ink
    warped = TF.affine(
        canvas,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[shear, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
        center=[canvas.size(-1) / 2, canvas.size(-2) / 2],
    )
    bbox = _ink_bbox(warped, threshold=0.01)
    if bbox is None:
        return torch.ones_like(centered_right)
    y0, y1, x0, x1 = bbox
    content = warped[..., y0:y1, x0:x1]
    content_height, content_width = content.shape[-2:]
    if content_height > height or content_width > width:
        factor = min(height / content_height, width / content_width)
        resized_height = max(1, min(height, round(content_height * factor)))
        resized_width = max(1, min(width, round(content_width * factor)))
        content = torch.nn.functional.interpolate(
            content.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        content_height, content_width = content.shape[-2:]

    center_y = 0.5 * (y0 + y1) - padding
    center_x = 0.5 * (x0 + x1) - padding
    paste_y = min(max(round(center_y - content_height / 2), 0), height - content_height)
    paste_x = min(max(round(center_x - content_width / 2), 0), width - content_width)
    output_ink = torch.zeros_like(ink)
    output_ink[..., paste_y : paste_y + content_height, paste_x : paste_x + content_width] = content
    return 1.0 - 2.0 * output_ink.clamp(0, 1)


def _content_preserving_affine(
    right: torch.Tensor,
    angle: float,
    tx: int,
    ty: int,
    scale: float,
    shear: float,
    margin_x: int,
    margin_y: int,
) -> torch.Tensor:
    """Center a standalone patch, transform it, and retain every ink pixel."""
    centered = _center_standalone_patch(right, margin_x=margin_x, margin_y=margin_y)
    return _affine_centered_patch(
        centered,
        angle=angle,
        tx=tx,
        ty=ty,
        scale=scale,
        shear=shear,
    )


def make_merge_sample(
    target: torch.Tensor,
    valid_width: int,
    cfg: CorruptionConfig,
    rng: random.Random,
    generator: torch.Generator,
    force_type: str | None = None,
    cut: int | None = None,
    include_trace: bool = False,
) -> dict[str, torch.Tensor | int | str]:
    cut = cut if cut is not None else cfg.patch_width
    if cut != cfg.patch_width or valid_width != cfg.window_width:
        raise ValueError("merge samples must be two equal patches split at the center")
    valid_mask = _column_mask(cfg.image_height, cfg.window_width, 0, valid_width)
    edit_mask = valid_mask.clone()
    left_mask = _column_mask(cfg.image_height, cfg.window_width, 0, cut)

    allowed = {"clean", "rotation", "translation", "shear_scale", "photometric", "mask", "combined"}
    corruption_type = force_type or "combined"
    if corruption_type not in allowed:
        raise ValueError(f"unknown corruption type: {corruption_type}")
    left_image = target[..., :cut].clone()
    clean_right = target[..., cut:valid_width].clone()
    right_image = clean_right.clone()

    erase_boundary = corruption_type == "mask" or corruption_type == "combined"
    if erase_boundary:
        right_image, boundary_erase_mask = _erase_boundary_columns(
            right_image,
            cfg,
            rng,
            force=corruption_type == "mask",
        )
    else:
        boundary_erase_mask = torch.zeros_like(right_image)
    right_after_boundary_erase = right_image.clone()

    if corruption_type in {"photometric", "combined"}:
        right_image = _photometric(right_image, cfg, rng, generator)
    right_after_photometric = right_image.clone()
    use_geometry = corruption_type not in {"clean", "photometric", "mask"}
    identity = corruption_type == "combined" and rng.random() < cfg.geometry_identity_probability
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

    if use_geometry:
        right_centered = _center_standalone_patch(
            right_image,
            margin_x=cfg.standalone_margin_x,
            margin_y=cfg.standalone_margin_y,
        )
        right_image = _affine_centered_patch(
            right_centered,
            angle=angle,
            tx=tx,
            ty=ty,
            scale=scale,
            shear=shear,
        )
        if include_trace:
            right_rotation_only = _affine_centered_patch(
                right_centered,
                angle=angle,
                tx=0,
                ty=0,
                scale=1.0,
                shear=0.0,
            )
            right_before_translation = _affine_centered_patch(
                right_centered,
                angle=angle,
                tx=0,
                ty=0,
                scale=scale,
                shear=shear,
            )
    else:
        right_centered = right_image.clone()
        right_rotation_only = right_image.clone()
        right_before_translation = right_image.clone()

    left_corruption_mask = torch.zeros_like(left_image)
    right_corruption_mask = ((right_image - clean_right).abs() > 1e-4).float()
    naive_concat = torch.cat((left_image, right_image), dim=-1)
    corruption_mask = torch.cat((left_corruption_mask, right_corruption_mask), dim=-1)
    join_width = max(1, cfg.center_gap_width)
    join_left = cut - join_width // 2
    join_mask = _column_mask(
        cfg.image_height,
        cfg.window_width,
        join_left,
        join_left + join_width,
    )
    left_preservation_mask = left_mask
    sample: dict[str, torch.Tensor | int | str] = {
        "target": target,
        "left_image": left_image,
        "right_image": right_image,
        "left_corruption_mask": left_corruption_mask,
        "right_corruption_mask": right_corruption_mask,
        "boundary_erase_mask": boundary_erase_mask,
        "naive_concat": naive_concat,
        "corruption_mask": corruption_mask,
        "edit_mask": edit_mask,
        "valid_mask": valid_mask,
        "left_preservation_mask": left_preservation_mask,
        "seam_mask": join_mask,
        "cut": cut,
        "angle": angle,
        "translate_x": tx,
        "translate_y": ty,
        "scale": scale,
        "shear": shear,
        "corruption_type": corruption_type,
    }
    if include_trace:
        sample.update(
            {
                "trace_clean_right": clean_right,
                "trace_after_boundary_erase": right_after_boundary_erase,
                "trace_after_photometric": right_after_photometric,
                "trace_centered_right": right_centered,
                "trace_rotation_only": right_rotation_only,
                "trace_before_translation": right_before_translation,
            }
        )
    return sample


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
        min_text_length: int = 2,
        include_trace: bool = False,
    ) -> None:
        if stage != "merge":
            raise ValueError("stage must be 'merge'")
        self.data_root = Path(data_root)
        self.records = _read_records(self.data_root, gt_file, min_text_length=min_text_length)
        if max_samples:
            self.records = self.records[:max_samples]
        self.stage = stage
        self.cfg = corruption or CorruptionConfig()
        self.deterministic = deterministic
        self.seed = seed
        self.force_corruption_type = force_corruption_type
        self.include_trace = include_trace

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int) -> tuple[random.Random, torch.Generator]:
        seed = self.seed + index if self.deterministic else random.randrange(2**31)
        return random.Random(seed), torch.Generator().manual_seed(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng, generator = self._rng(index)
        best: tuple[
            int, Path, str, torch.Tensor, torch.Tensor, torch.Tensor, int, int
        ] | None = None
        best_score = -1
        for attempt in range(max(1, self.cfg.record_sample_attempts)):
            record_index = index if attempt == 0 else rng.randrange(len(self.records))
            path, text = self.records[record_index]
            with Image.open(path) as image:
                tensor = _to_normalized_tensor(image, self.cfg.image_height)
            target, prefix_context, global_cut, prefix_valid_width = _merge_window(
                tensor, self.cfg, rng
            )
            left_ink = int((target[..., : self.cfg.patch_width] < 0.9).sum())
            right_ink = int((target[..., self.cfg.patch_width :] < 0.9).sum())
            score = min(left_ink, right_ink)
            if score > best_score:
                best_score = score
                best = (
                    record_index,
                    path,
                    text,
                    tensor,
                    target,
                    prefix_context,
                    global_cut,
                    prefix_valid_width,
                )
            if score >= self.cfg.min_half_ink_pixels:
                break
        if best is None:
            raise RuntimeError("failed to sample a merger window")
        (
            record_index,
            path,
            text,
            tensor,
            target,
            prefix_context,
            global_cut,
            prefix_valid_width,
        ) = best
        valid_width = self.cfg.window_width

        sample = make_merge_sample(
            target,
            valid_width,
            self.cfg,
            rng,
            generator,
            force_type=self.force_corruption_type,
            cut=self.cfg.patch_width,
            include_trace=self.include_trace,
        )
        sample.update(
            {
                "path": str(path),
                "text": text,
                "record_index": record_index,
                "valid_width": valid_width,
                "global_cut": global_cut,
                "source_width": max(self.cfg.window_width, tensor.size(-1)),
                "prefix_valid_width": prefix_valid_width,
                "prefix_context": prefix_context,
            }
        )
        if self.include_trace:
            sample["source_image"] = tensor
        return sample
