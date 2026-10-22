from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_position(length: int, dimension: int, reference: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(length, device=reference.device, dtype=reference.dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=reference.device, dtype=reference.dtype)
        * (-math.log(10000.0) / dimension)
    )
    encoding = reference.new_zeros((1, length, dimension))
    encoding[..., 0::2] = torch.sin(positions * frequencies)
    encoding[..., 1::2] = torch.cos(positions * frequencies[: encoding[..., 1::2].size(-1)])
    return encoding


class MergerNet(nn.Module):
    """Residual ViT merger for two adjacent handwriting patches."""

    def __init__(
        self,
        image_height: int = 32,
        patch_width: int = 32,
        window_width: int = 64,
        vit_patch_size: int = 4,
        embed_dim: int = 256,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 1024,
        prefix_transformer_layers: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if window_width != 2 * patch_width:
            raise ValueError("window_width must equal 2 * patch_width")
        if image_height % vit_patch_size or window_width % vit_patch_size:
            raise ValueError("image dimensions must be divisible by vit_patch_size")
        if embed_dim % transformer_heads:
            raise ValueError("embed_dim must be divisible by transformer_heads")

        self.image_height = image_height
        self.patch_width = patch_width
        self.window_width = window_width
        self.vit_patch_size = vit_patch_size
        self.embed_dim = embed_dim
        self.residual_scale = residual_scale

        # One complete generated character patch becomes one prefix token.
        prefix_pixels = image_height * patch_width
        self.prefix_patch_encoder = nn.Sequential(
            nn.LayerNorm(prefix_pixels),
            nn.Linear(prefix_pixels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        prefix_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prefix_transformer = nn.TransformerEncoder(
            prefix_layer,
            num_layers=prefix_transformer_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Local ViT input: grayscale image + corruption mask + edit mask.
        self.local_patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=vit_patch_size,
            stride=vit_patch_size,
        )
        local_token_count = (image_height // vit_patch_size) * (window_width // vit_patch_size)
        self.local_position = nn.Parameter(torch.zeros(1, local_token_count, embed_dim))
        self.prefix_role = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.local_position, std=0.02)
        nn.init.trunc_normal_(self.prefix_role, std=0.02)

        merger_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.merger_transformer = nn.TransformerEncoder(
            merger_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.patch_decoder = nn.Linear(embed_dim, vit_patch_size * vit_patch_size)
        # Identity initialization stabilizes the first updates of the merger.
        nn.init.zeros_(self.patch_decoder.weight)
        nn.init.zeros_(self.patch_decoder.bias)

    def _encode_prefix(
        self, prefix_context: torch.Tensor, prefix_mask: torch.Tensor
    ) -> torch.Tensor:
        batch = prefix_context.size(0)
        prefix_patches = prefix_context.unfold(-1, self.patch_width, self.patch_width)
        prefix_patches = prefix_patches.permute(0, 3, 1, 2, 4).contiguous()
        prefix_patches = prefix_patches.view(batch, -1, self.image_height * self.patch_width)
        prefix_tokens = self.prefix_patch_encoder(prefix_patches)

        patch_mask = prefix_mask.unfold(-1, self.patch_width, self.patch_width)
        token_valid = patch_mask.amax(dim=(1, 2, 4)).bool()
        causal_mask = torch.triu(
            torch.ones(
                prefix_tokens.size(1),
                prefix_tokens.size(1),
                dtype=torch.bool,
                device=prefix_tokens.device,
            ),
            diagonal=1,
        )
        prefix_tokens = self.prefix_transformer(
            prefix_tokens
            + _sinusoidal_position(prefix_tokens.size(1), self.embed_dim, prefix_tokens),
            mask=causal_mask,
            src_key_padding_mask=~token_valid,
        )
        last_indices = token_valid.sum(dim=1).clamp_min(1) - 1
        return prefix_tokens[
            torch.arange(batch, device=prefix_tokens.device), last_indices
        ].unsqueeze(1)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.size(0)
        grid_height = self.image_height // self.vit_patch_size
        grid_width = self.window_width // self.vit_patch_size
        patches = patches.view(
            batch,
            grid_height,
            grid_width,
            1,
            self.vit_patch_size,
            self.vit_patch_size,
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(batch, 1, self.image_height, self.window_width)

    def forward(
        self,
        image: torch.Tensor,
        corruption_mask: torch.Tensor,
        edit_mask: torch.Tensor,
        prefix_context: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        if image.shape != corruption_mask.shape or image.shape != edit_mask.shape:
            raise ValueError("image and masks must have identical [B,1,H,W] shapes")
        if image.shape[-2:] != (self.image_height, self.window_width):
            raise ValueError(
                f"expected spatial shape {(self.image_height, self.window_width)}, "
                f"got {tuple(image.shape[-2:])}"
            )
        if prefix_context.shape != prefix_mask.shape or prefix_context.size(0) != image.size(0):
            raise ValueError("prefix_context and prefix_mask must have identical [B,1,H,W] shapes")
        if prefix_context.size(-2) != self.image_height:
            raise ValueError(f"prefix height must be {self.image_height}")
        if prefix_context.size(-1) % self.patch_width:
            raise ValueError("padded prefix width must be divisible by patch_width")

        prefix_state = self._encode_prefix(prefix_context, prefix_mask)
        local = torch.cat((image, corruption_mask, edit_mask), dim=1)
        local_tokens = self.local_patch_embed(local).flatten(2).transpose(1, 2)
        fused = torch.cat(
            (prefix_state + self.prefix_role, local_tokens + self.local_position), dim=1
        )
        local_output = self.merger_transformer(fused)[:, 1:]
        delta = torch.tanh(self._unpatchify(self.patch_decoder(local_output)))
        return (image + edit_mask * delta * self.residual_scale).clamp(-1.0, 1.0)
