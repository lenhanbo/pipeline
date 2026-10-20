from __future__ import annotations

import math

import torch

from .losses import _sobel, _ssim_map, boundary_mask, masked_mean


@torch.no_grad()
def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    input_image: torch.Tensor | None = None,
) -> dict[str, float]:
    mask = edit_mask * valid_mask
    error = prediction - target
    mae = masked_mean(error.abs(), mask)
    mse = masked_mean(error.square(), mask)
    psnr = 10.0 * math.log10(4.0 / max(float(mse), 1e-12))
    ssim = masked_mean(_ssim_map(prediction, target), mask)
    pred_x, pred_y = _sobel(prediction)
    target_x, target_y = _sobel(target)
    gradient_mae = masked_mean((pred_x - target_x).abs() + (pred_y - target_y).abs(), mask)
    edges = boundary_mask(edit_mask) * valid_mask
    seam = masked_mean((pred_x - target_x).abs(), edges)

    pred_ink = (prediction < 0).float() * mask
    target_ink = (target < 0).float() * mask
    true_positive = (pred_ink * target_ink).sum()
    precision = true_positive / pred_ink.sum().clamp_min(1)
    recall = true_positive / target_ink.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    outside = valid_mask * (1.0 - edit_mask)
    preservation_error = prediction - (input_image if input_image is not None else target)
    outside_mae = masked_mean(preservation_error.abs(), outside)
    return {
        "mae": float(mae),
        "psnr": psnr,
        "ssim": float(ssim),
        "gradient_mae": float(gradient_mae),
        "seam_discontinuity": float(seam),
        "ink_precision": float(precision),
        "ink_recall": float(recall),
        "ink_f1": float(f1),
        "outside_mask_mae": float(outside_mae),
    }


def edge_copy_baseline(corrupted: torch.Tensor, edit_mask: torch.Tensor) -> torch.Tensor:
    result = corrupted.clone()
    for batch_index in range(result.size(0)):
        columns = torch.nonzero(edit_mask[batch_index, 0, 0] > 0.5, as_tuple=False).flatten()
        if columns.numel() == 0:
            continue
        start, end = int(columns.min()), int(columns.max()) + 1
        source = result[batch_index : batch_index + 1, :, :, max(0, start - 1) : start]
        if source.numel():
            result[batch_index : batch_index + 1, :, :, start:end] = source
    return result
