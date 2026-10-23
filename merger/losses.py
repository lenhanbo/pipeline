from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def _sobel(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    kernel_y = kernel_x.transpose(-1, -2)
    return F.conv2d(image, kernel_x, padding=1), F.conv2d(image, kernel_y, padding=1)


def _ssim_map(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.add(1).div(2)
    target = target.add(1).div(2)
    mu_x = F.avg_pool2d(prediction, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    sigma_x = F.avg_pool2d(prediction.square(), 7, 1, 3) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), 7, 1, 3) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 7, 1, 3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )


def boundary_mask(edit_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.zeros_like(edit_mask)
    edge = (edit_mask[..., 1:] - edit_mask[..., :-1]).abs()
    diff[..., 1:] = torch.maximum(diff[..., 1:], edge)
    diff[..., :-1] = torch.maximum(diff[..., :-1], edge)
    return F.max_pool2d(diff, kernel_size=(3, 3), stride=1, padding=1)


class MergerLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        left_identity_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.left_identity_weight = left_identity_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        edit_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        input_image: torch.Tensor | None = None,
        left_preservation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = edit_mask * valid_mask
        reconstruction_mse = masked_mean((prediction - target).square(), mask)
        if input_image is not None and left_preservation_mask is not None:
            left_identity_mse = masked_mean(
                (prediction - input_image).square(),
                left_preservation_mask * valid_mask,
            )
        else:
            left_identity_mse = prediction.new_zeros(())
        total = (
            self.mse_weight * reconstruction_mse
            + self.left_identity_weight * left_identity_mse
        )
        values = {
            "loss": total,
            "reconstruction_mse": reconstruction_mse,
            "left_identity_mse": left_identity_mse,
        }
        return total, values
