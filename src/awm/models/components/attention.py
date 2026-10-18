"""
Attention modules for Wan Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import rope_apply_qk


class WanRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)


class WanLayerNorm(nn.LayerNorm):
    """Layer normalization with float32 computation."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class WanSelfAttention(nn.Module):
    """Self-attention module with optional QK normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0,
        temporal_stride: int = 4
    ) -> torch.Tensor:
        from .attention_utils import attention

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs, temporal_stride=temporal_stride)

        x = attention(
            q.to(dtype), k.to(dtype), v=v.to(dtype),
            k_lens=seq_lens, window_size=self.window_size)
        x = x.to(dtype).flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross attention."""

    def precompute_kv(
        self,
        context: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project immutable text context once for repeated inference steps."""
        b, n, d = context.size(0), self.num_heads, self.head_dim
        context = context.to(dtype)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return k, v

    def forward_with_kv(
        self,
        x: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0,
    ) -> torch.Tensor:
        """Cross-attend using K/V projected by :meth:`precompute_kv`."""
        from .attention_utils import attention

        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k, v = kv
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.to(dtype).flatten(2)
        return self.o(x)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0
    ) -> torch.Tensor:
        return self.forward_with_kv(
            x,
            self.precompute_kv(context, dtype),
            context_lens,
            dtype,
            t=t,
        )


class WanI2VCrossAttention(WanSelfAttention):
    """Image-to-video cross attention with separate image keys/values."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def precompute_kv(
        self,
        context: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_img = context[:, :257].to(dtype)
        context_text = context[:, 257:].to(dtype)
        b, n, d = context.size(0), self.num_heads, self.head_dim
        k = self.norm_k(self.k(context_text)).view(b, -1, n, d)
        v = self.v(context_text).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        return k, v, k_img, v_img

    def forward_with_kv(
        self,
        x: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0,
    ) -> torch.Tensor:
        from .attention_utils import attention

        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k, v, k_img, v_img = kv
        img_x = attention(
            q.to(dtype), k_img.to(dtype), v_img.to(dtype), k_lens=None
        ).to(dtype)
        text_x = attention(
            q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens
        ).to(dtype)
        return self.o(text_x.flatten(2) + img_x.flatten(2))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0
    ) -> torch.Tensor:
        return self.forward_with_kv(
            x,
            self.precompute_kv(context, dtype),
            context_lens,
            dtype,
            t=t,
        )


class WanCrossAttention(WanSelfAttention):
    """Generic cross attention."""

    def precompute_kv(
        self,
        context: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = context.size(0), self.num_heads, self.head_dim
        context = context.to(dtype)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return k, v

    def forward_with_kv(
        self,
        x: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0,
    ) -> torch.Tensor:
        from .attention_utils import attention

        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k, v = kv
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.flatten(2)
        return self.o(x.to(dtype))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0
    ) -> torch.Tensor:
        return self.forward_with_kv(
            x,
            self.precompute_kv(context, dtype),
            context_lens,
            dtype,
            t=t,
        )


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}
