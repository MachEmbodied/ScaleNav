"""
Transformer blocks and output heads for Wan Transformer.
"""

import math

import torch
import torch.nn as nn

from .attention import (
    WanLayerNorm,
)


class Head(nn.Module):
    """Output head with modulation."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_total)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [ei.squeeze(2) for ei in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)

        original_type = x.dtype
        x = x.to(self.head.weight.dtype)
        e = (e[0].to(x.dtype), e[1].to(x.dtype))
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0])).to(original_type)
        return x


class DeepActionHead(nn.Module):
    """AdaLN-conditioned action head with two hidden GELU layers."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple,
        eps: float = 1e-6,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        # Deeper trunk replacing Head's single Linear.
        # The final Linear is the one zero-initialized externally to
        # match the original `Head.head` zero-init contract.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim_total)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [ei.squeeze(2) for ei in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)

        original_type = x.dtype
        x = x.to(self.head.weight.dtype)
        e = (e[0].to(x.dtype), e[1].to(x.dtype))
        x = self.norm(x) * (1 + e[1]) + e[0]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.head(x)
        return x.to(original_type)


class MLPProj(nn.Module):
    """MLP projection for CLIP features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)
