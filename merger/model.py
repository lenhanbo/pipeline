from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class MergerNet(nn.Module):
    """Residual U-Net that may alter pixels only inside ``edit_mask``.

    The three input channels are the corrupted image, a missing-pixel mask, and
    the editable-region mask. Images are expected in [-1, 1], masks in [0, 1].
    """

    def __init__(
        self,
        image_height: int = 32,
        window_width: int = 256,
        base_channels: int = 64,
        bottleneck_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 1024,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if image_height % 8 or window_width % 8:
            raise ValueError("image_height and window_width must be divisible by 8")
        if bottleneck_dim % transformer_heads:
            raise ValueError("bottleneck_dim must be divisible by transformer_heads")

        c = base_channels
        self.image_height = image_height
        self.window_width = window_width
        self.residual_scale = residual_scale
        self.stem = ConvBlock(3, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, bottleneck_dim)

        token_count = (image_height // 8) * (window_width // 8)
        self.position = nn.Parameter(torch.zeros(1, token_count, bottleneck_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, norm=nn.LayerNorm(bottleneck_dim)
        )

        self.up2 = UpBlock(bottleneck_dim, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(
        self,
        image: torch.Tensor,
        corruption_mask: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if image.shape != corruption_mask.shape or image.shape != edit_mask.shape:
            raise ValueError("image and masks must have identical [B,1,H,W] shapes")
        if image.shape[-2:] != (self.image_height, self.window_width):
            raise ValueError(
                f"expected spatial shape {(self.image_height, self.window_width)}, "
                f"got {tuple(image.shape[-2:])}"
            )

        e0 = self.stem(torch.cat((image, corruption_mask, edit_mask), dim=1))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        bottleneck = self.down3(e2)

        batch, channels, height, width = bottleneck.shape
        tokens = bottleneck.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens + self.position[:, : tokens.size(1)])
        bottleneck = tokens.transpose(1, 2).reshape(batch, channels, height, width)

        decoded = self.up2(bottleneck, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        delta = torch.tanh(self.head(decoded)) * self.residual_scale
        return (image + edit_mask * delta).clamp(-1.0, 1.0)
