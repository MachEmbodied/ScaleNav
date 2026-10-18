"""Causal video/action diffusion with cached continuous history."""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version

from .components.attention import WanLayerNorm, WanRMSNorm, WAN_CROSSATTENTION_CLASSES
from .components.attention_utils import attention
from .components.embeddings import (
    build_rope_freqs,
    rope_apply_qk,
    rope_apply_multimodal_qk,
    rope_params,
    sinusoidal_embedding_1d,
    zero_initialize_layer,
)
from .components.blocks import DeepActionHead, Head, MLPProj


class CausalChunkSelfAttention(nn.Module):
    """
    Self-attention with block-wise causal masking across chunks.

    Same Q/K/V/O linears + RMSNorm as WanSelfAttention, but supports:
      - Teacher forcing: clean chunks (block-wise causal) + noisy chunk (attends to all clean + self)
      - KV caching for autoregressive inference
      - Sliding window via max_attn_chunks
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        max_attn_chunks: int = -1,
        action_temporal_scale: float = 1.0,
        use_special_memory: bool = False,
        use_continuous_history: bool = False,
        special_memory_recent_chunks: int = 2,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attn_chunks = max_attn_chunks
        self.action_temporal_scale = action_temporal_scale
        if use_special_memory:
            raise ValueError("ScaleNav uses continuous history, without special memory tokens")
        self.use_special_memory = False
        self.use_continuous_history = use_continuous_history
        self.special_memory_recent_chunks = special_memory_recent_chunks

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @property
    def use_anchor_history(self) -> bool:
        """Whether old history is reduced to a permanent first-frame anchor."""
        return self.use_continuous_history

    def _recent_history_chunks(self, reference_position: int) -> int:
        """Number of full recent chunks retained alongside the anchor."""
        if self.max_attn_chunks == -1:
            return reference_position
        # The finite window contains anchor + recent history + current query.
        return max(0, self.max_attn_chunks - 2)

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        temporal_stride: int = 4,
        chunk_boundaries: Optional[List[Tuple[int, int]]] = None,
        is_tf: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t: int = 0,
        num_clean_chunks: int = 0,
        past_chunk_valid: Optional[torch.Tensor] = None,
        collect_kv: bool = False,
        static_cache: bool = False,
        chunk_layouts: Optional[List[Tuple[int, int, int, int]]] = None,
        chunk_time_offsets: Optional[List[float]] = None,
        chunk_anchor_mask: Optional[torch.Tensor] = None,
        decouple_noisy_chunks: bool = False,
        joint_rope_freqs: Optional[torch.Tensor] = None,
        kv_workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [B, S, dim] input hidden states
            seq_lens: per-sample sequence lengths
            grid_sizes: [B, 3] per-sample (F, H, W) for RoPE
            freqs: RoPE frequency tensor
            chunk_boundaries: list of (start, end) for each chunk in the sequence
            is_tf: teacher forcing mode (training)
            kv_cache: (cached_k, cached_v) from previous chunks (inference)
            collect_kv: prefill mode — also return this call's post-RoPE (K, V)
                so the caller can build a static cache of the clean past.
            static_cache: read-only cache mode — attend over [cached past +
                current] WITHOUT appending the current (noisy) K/V to the cache.
        Returns:
            (output, updated_kv_cache_or_None)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)

        if self.use_continuous_history:
            if chunk_boundaries is None or chunk_layouts is None or chunk_time_offsets is None:
                raise ValueError(
                    "Global-time RoPE requires explicit chunk position metadata"
                )
            q, k = rope_apply_multimodal_qk(
                q, k, grid_sizes, freqs,
                chunk_boundaries=chunk_boundaries,
                chunk_layouts=chunk_layouts,
                chunk_time_offsets=chunk_time_offsets,
                temporal_stride=temporal_stride,
            )
        elif chunk_boundaries is not None and len(chunk_boundaries) > 1:
            q, k = self._apply_rope_per_chunk(
                q, k, grid_sizes, freqs, chunk_boundaries, temporal_stride
            )
        else:
            q, k = rope_apply_qk(
                q, k, grid_sizes, freqs,
                temporal_stride=temporal_stride,
                action_temporal_scale=self.action_temporal_scale,
                precomputed_freqs=joint_rope_freqs,
            )

        if chunk_boundaries is not None and len(chunk_boundaries) > 1 and kv_cache is None:
            # Block-causal attention: each chunk ci attends only to chunks 0..ci.
            # When num_clean_chunks > 0 we apply the DreamZero-style routing:
            # the first num_clean_chunks chunks form a clean-context "memory
            # bank" (block-causal among themselves); subsequent chunks are
            # noisy queries that attend to the clean past + their own tokens
            # only (never other noisy chunks, never their own clean copy).
            out = self._teacher_forcing_attention(
                q, k, v, seq_lens, chunk_boundaries, dtype,
                num_clean_chunks=num_clean_chunks,
                past_chunk_valid=past_chunk_valid,
                chunk_layouts=chunk_layouts,
                chunk_anchor_mask=chunk_anchor_mask,
            )
            if collect_kv:
                return out, self._collect_memory_kv(
                    k, v, chunk_boundaries, chunk_layouts, chunk_anchor_mask
                )
            return out, None

        elif kv_cache is not None:
            # Inference with KV cache
            cached_k, cached_v = kv_cache
            # Keep only the trained attention window; retain K/V computed with full past context.
            if static_cache and not self.use_anchor_history and self.max_attn_chunks > 0:
                max_past_tokens = max(0, self.max_attn_chunks - 1) * k.shape[1]
                if cached_k.shape[1] > max_past_tokens:
                    if max_past_tokens == 0:
                        cached_k = cached_k[:, :0]
                        cached_v = cached_v[:, :0]
                    else:
                        cached_k = cached_k[:, -max_past_tokens:]
                        cached_v = cached_v[:, -max_past_tokens:]
            # Reuse the clean prefix; overwrite only the current video/action tail each step.
            if kv_workspace is not None:
                if not static_cache or torch.is_grad_enabled():
                    raise ValueError(
                        "KV workspaces are only valid for no-grad static inference"
                    )
                full_k, full_v = kv_workspace
                expected_shape = (
                    cached_k.shape[0], cached_k.shape[1] + k.shape[1],
                    cached_k.shape[2], cached_k.shape[3],
                )
                if tuple(full_k.shape) != expected_shape or tuple(full_v.shape) != expected_shape:
                    raise ValueError(
                        f"KV workspace shape mismatch: expected {expected_shape}, "
                        f"got K={tuple(full_k.shape)}, V={tuple(full_v.shape)}"
                    )
                current_start = cached_k.shape[1]
                full_k[:, current_start:].copy_(k)
                full_v[:, current_start:].copy_(v)
            else:
                full_k = torch.cat([cached_k, k], dim=1)
                full_v = torch.cat([cached_v, v], dim=1)

            if static_cache:
                # The noisy chunk attends bidirectionally to itself and clean cached history.
                # Never write noisy K/V back into the clean cache.
                if decouple_noisy_chunks:
                    if not chunk_layouts or len(chunk_layouts) != 1:
                        raise ValueError(
                            "Decoupled static inference requires one current layout"
                        )
                    out = self._decoupled_chunk_attention(
                        q,
                        full_k,
                        full_v,
                        current_start=cached_k.shape[1],
                        layout=chunk_layouts[0],
                        dtype=dtype,
                        history_ranges=(
                            [(0, cached_k.shape[1])]
                            if cached_k.shape[1] else []
                        ),
                        allow_contiguous_view=kv_workspace is not None,
                    )
                else:
                    # Dense fixed-size workspaces need no length mask; omitting it avoids
                    # CUDA scalar reads during graph capture.
                    k_lens = None if kv_workspace is not None else torch.full(
                        (b,), full_k.shape[1], dtype=torch.long, device=x.device
                    )
                    out = attention(
                        q.to(dtype), full_k.to(dtype), v=full_v.to(dtype),
                        k_lens=k_lens, causal=False, window_size=self.window_size,
                    )
                out = out.to(dtype).flatten(2)
                out = self.o(out)
                # Incremental prefill retains the appended clean chunk.
                # Denoising uses collect_kv=False and leaves history unchanged.
                appended_cache = (
                    (full_k.detach(), full_v.detach()) if collect_kv else None
                )
                return out, appended_cache

            if self.max_attn_chunks > 0 and chunk_boundaries is not None:
                max_ctx_len = sum(
                    end - start
                    for start, end in chunk_boundaries[-self.max_attn_chunks:]
                )
                if full_k.shape[1] > max_ctx_len + k.shape[1]:
                    trim = full_k.shape[1] - max_ctx_len - k.shape[1]
                    full_k = full_k[:, trim:]
                    full_v = full_v[:, trim:]

            # Attention: current Q against full KV (past cached + current)
            # Use causal=True to match training behavior: current chunk tokens
            # attend to all past context + causal within current chunk.
            # Flash Attention bottom-right alignment handles Lq < Lk correctly.
            k_lens = torch.tensor([full_k.shape[1]] * b, dtype=torch.long, device=x.device)
            out = attention(
                q.to(dtype), full_k.to(dtype), v=full_v.to(dtype),
                k_lens=k_lens, causal=True, window_size=self.window_size,
            )

            out = out.to(dtype).flatten(2)
            out = self.o(out)

            new_cache = (full_k.detach(), full_v.detach())
            return out, new_cache

        else:
            # Standard bidirectional attention (single chunk or no chunk boundaries)
            if chunk_boundaries is not None and len(chunk_boundaries) > 1:
                warnings.warn(
                    "CausalChunkSelfAttention: multiple chunks reaching standard "
                    "bidirectional attention path. This should not happen — check "
                    "that block-causal masking is applied for multi-chunk inputs.",
                    stacklevel=2,
                )
            out = attention(
                q.to(dtype), k.to(dtype), v=v.to(dtype),
                k_lens=seq_lens, window_size=self.window_size,
            )
            out = out.to(dtype).flatten(2)
            out = self.o(out)
            if collect_kv:
                if chunk_boundaries is None:
                    return out, (k.detach(), v.detach())
                return out, self._collect_memory_kv(
                    k, v, chunk_boundaries, chunk_layouts, chunk_anchor_mask
                )
            return out, None

    def _apply_rope_per_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        chunk_boundaries: List[Tuple[int, int]],
        temporal_stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE independently to each chunk, resetting positions per chunk.

        Each chunk uses the same grid_sizes (same F, H, W) but gets local positions
        starting from 0, as if it were an independent video.
        """
        q_parts = []
        k_parts = []
        for c_start, c_end in chunk_boundaries:
            q_chunk = q[:, c_start:c_end]
            k_chunk = k[:, c_start:c_end]
            # Apply RoPE with local positions (same grid_sizes for each chunk)
            # action_temporal_scale gives action tokens an extended temporal range
            q_chunk, k_chunk = rope_apply_qk(
                q_chunk, k_chunk, grid_sizes, freqs,
                temporal_stride=temporal_stride,
                action_temporal_scale=self.action_temporal_scale,
            )
            q_parts.append(q_chunk)
            k_parts.append(k_chunk)
        q = torch.cat(q_parts, dim=1)
        k = torch.cat(k_parts, dim=1)
        return q, k

    def _per_sample_attn(self, q_chunk, k, v, per_sample_key_ranges, dtype):
        """Run attention per batch-sample with sample-specific valid KEY ranges.

        Used by the padded-past masking path: each sample's zero-padded past
        chunks are excluded from its key set. flash-attn varlen cannot express an
        arbitrary per-sample key mask (it only masks a contiguous suffix), and the
        padding sits at the PREFIX, so we slice per sample and call attention B
        times. FLOPs are ~unchanged (attention is already per-sample over the batch
        dim) and padded keys are dropped, so it is if anything cheaper; the only
        overhead is B kernel launches per masked chunk (B is the micro-batch, ~2).

        Args:
            q_chunk: [B, Lq, n, d] queries for this chunk (all samples).
            k, v: full [B, S, n, d] key/value tensors (already RoPE'd).
            per_sample_key_ranges: list (len B); entry bi is a list of
                (tok_start, tok_end) slices into k/v giving sample bi's valid keys.
            dtype: attention dtype.
        Returns: [B, Lq, n, d]
        """
        outs = []
        for bi in range(len(per_sample_key_ranges)):
            ranges = per_sample_key_ranges[bi]
            k_ctx = torch.cat([k[bi:bi + 1, s:e] for s, e in ranges], dim=1)
            v_ctx = torch.cat([v[bi:bi + 1, s:e] for s, e in ranges], dim=1)
            klens = torch.tensor([k_ctx.shape[1]], dtype=torch.long, device=q_chunk.device)
            outs.append(attention(
                q_chunk[bi:bi + 1].to(dtype), k_ctx.to(dtype), v=v_ctx.to(dtype),
                k_lens=klens, window_size=self.window_size,
            ))
        return torch.cat(outs, dim=0)

    def _memory_key_ranges(
        self,
        history_indices: List[int],
        reference_position: int,
        chunk_boundaries: List[Tuple[int, int]],
        chunk_layouts: List[Tuple[int, int, int, int]],
        self_index: Optional[int] = None,
        anchor_indices: Optional[set] = None,
    ) -> List[Tuple[int, int]]:
        """Select the first-frame anchor, optional special tokens, and recent chunks."""
        recent_start = max(
            0, reference_position - self._recent_history_chunks(reference_position)
        )
        ranges: List[Tuple[int, int]] = []
        for src in history_indices:
            start, end = chunk_boundaries[src]
            video_len, special_len, _, anchor_len = chunk_layouts[src]
            if src >= recent_start:
                ranges.append((start, end))
                continue
            is_anchor = src in (anchor_indices if anchor_indices is not None else {0})
            if is_anchor and anchor_len:
                ranges.append((start, start + anchor_len))
        if self_index is not None:
            ranges.append(chunk_boundaries[self_index])
        return ranges

    def _collect_memory_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_boundaries: List[Tuple[int, int]],
        chunk_layouts: Optional[List[Tuple[int, int, int, int]]],
        chunk_anchor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compact a clean prefill into the context needed by the next chunk."""
        if not self.use_anchor_history or not chunk_layouts:
            return k.detach(), v.detach()
        ranges = self._memory_key_ranges(
            list(range(len(chunk_boundaries))),
            len(chunk_boundaries),
            chunk_boundaries,
            chunk_layouts,
            anchor_indices=(
                set(torch.nonzero(chunk_anchor_mask[0], as_tuple=False).flatten().tolist())
                if chunk_anchor_mask is not None else None
            ),
        )
        compact_k = torch.cat([k[:, start:end] for start, end in ranges], dim=1)
        compact_v = torch.cat([v[:, start:end] for start, end in ranges], dim=1)
        return compact_k.detach(), compact_v.detach()

    def _teacher_forcing_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: torch.Tensor,
        chunk_boundaries: List[Tuple[int, int]],
        dtype: torch.dtype,
        num_clean_chunks: int = 0,
        past_chunk_valid: Optional[torch.Tensor] = None,
        chunk_layouts: Optional[List[Tuple[int, int, int, int]]] = None,
        chunk_anchor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Block-causal attention for teacher forcing.

        Two routing modes:

        (1) Standard block-causal (num_clean_chunks=0).
            Chunk ci attends to all chunks 0..ci. No distinction between past
            and current — past chunks are clean GT (or lightly perturbed) and
            the last chunk is the noisy one being predicted.

        (2) DreamZero-style dual-half routing (num_clean_chunks > 0).
            The sequence holds 2K chunks: the first K are a clean "memory bank"
            (chunks 0..K-1 as clean GT, t=0), the last K are noisy queries
            (each at its own t_k, predicted by the head afterward). RoPE
            positions match — clean chunk i and noisy chunk i share the same
            positional encoding.

            - Clean chunk i (i < K): block-causal among clean chunks 0..i.
            - Noisy chunk i (K <= i < 2K): attends to clean chunks [0..i-K-1]
              ONLY + its own noisy tokens. It never sees other noisy chunks and
              never sees its own clean copy (no answer leak).
        """
        b = q.shape[0]

        if len(chunk_boundaries) == 1:
            # Single chunk: standard bidirectional attention.
            out = attention(
                q.to(dtype), k.to(dtype), v=v.to(dtype),
                k_lens=seq_lens, window_size=self.window_size,
            )
            out = out.to(dtype).flatten(2)
            out = self.o(out)
            return out

        outputs = []

        # past_chunk_valid is [B, K]; real history forms a contiguous suffix.
        # Drop each sample's padded prefix to match history-free inference.
        use_psv = (
            past_chunk_valid is not None
            and past_chunk_valid.ndim == 2
            and bool((past_chunk_valid < 0.5).any().item())
        )

        for ci, (c_start, c_end) in enumerate(chunk_boundaries):
            q_chunk = q[:, c_start:c_end]

            if self.use_anchor_history:
                if chunk_layouts is None:
                    raise ValueError("Anchor history attention requires chunk layouts")
                if num_clean_chunks > 0 and ci >= num_clean_chunks:
                    reference_position = ci - num_clean_chunks
                    history_indices = list(range(reference_position))
                else:
                    reference_position = ci
                    history_indices = list(range(ci))

                per_sample_ranges = []
                for bi in range(b):
                    valid_history = [
                        src for src in history_indices
                        if past_chunk_valid is None
                        or src >= past_chunk_valid.shape[1]
                        or bool(past_chunk_valid[bi, src].item() >= 0.5)
                    ]
                    anchor_indices = (
                        set(torch.nonzero(
                            chunk_anchor_mask[bi], as_tuple=False
                        ).flatten().tolist())
                        if chunk_anchor_mask is not None else None
                    )
                    per_sample_ranges.append(self._memory_key_ranges(
                        valid_history,
                        reference_position,
                        chunk_boundaries,
                        chunk_layouts,
                        self_index=ci,
                        anchor_indices=anchor_indices,
                    ))
                if all(ranges == per_sample_ranges[0] for ranges in per_sample_ranges[1:]):
                    shared_ranges = per_sample_ranges[0]
                    k_ctx = torch.cat(
                        [k[:, start:end] for start, end in shared_ranges], dim=1
                    )
                    v_ctx = torch.cat(
                        [v[:, start:end] for start, end in shared_ranges], dim=1
                    )
                    k_lens_chunk = torch.tensor(
                        [k_ctx.shape[1]] * b, dtype=torch.long, device=q.device
                    )
                    out_chunk = attention(
                        q_chunk.to(dtype), k_ctx.to(dtype), v=v_ctx.to(dtype),
                        k_lens=k_lens_chunk, window_size=self.window_size,
                    )
                else:
                    out_chunk = self._per_sample_attn(
                        q_chunk, k, v, per_sample_ranges, dtype
                    )
                outputs.append(out_chunk)
                continue

            if num_clean_chunks > 0 and ci >= num_clean_chunks:
                clip_pos = ci - num_clean_chunks  # 0..K-1
                # Clean past window: clean chunks 0..clip_pos-1.
                if self.max_attn_chunks == -1 or clip_pos < self.max_attn_chunks:
                    clean_window_start = 0
                else:
                    clean_window_start = clip_pos - self.max_attn_chunks + 1
                # Self noisy tokens (bidirectional within own chunk).
                self_k = k[:, c_start:c_end]
                self_v = v[:, c_start:c_end]

                if use_psv and clip_pos > 0:
                    # Per sample: keep only its REAL clean chunks (suffix of the
                    # window [clean_window_start..clip_pos-1]) + its own self
                    # tokens. The zero-padded clean prefix is dropped per sample.
                    ranges = []
                    for bi in range(b):
                        n_valid = int(
                            past_chunk_valid[bi, clean_window_start:clip_pos].sum().item()
                        )
                        rng = []
                        if n_valid > 0:
                            first_valid = clip_pos - n_valid  # contiguous suffix
                            rng.append((chunk_boundaries[first_valid][0],
                                        chunk_boundaries[clip_pos - 1][1]))
                        rng.append((c_start, c_end))  # self tokens always kept
                        ranges.append(rng)
                    out_chunk = self._per_sample_attn(q_chunk, k, v, ranges, dtype)
                else:
                    if clip_pos == 0:
                        clean_k = k[:, 0:0]
                        clean_v = v[:, 0:0]
                    else:
                        clean_ctx_start = chunk_boundaries[clean_window_start][0]
                        clean_ctx_end = chunk_boundaries[clip_pos - 1][1]
                        clean_k = k[:, clean_ctx_start:clean_ctx_end]
                        clean_v = v[:, clean_ctx_start:clean_ctx_end]
                    k_ctx = torch.cat([clean_k, self_k], dim=1)
                    v_ctx = torch.cat([clean_v, self_v], dim=1)
                    ctx_len = k_ctx.shape[1]
                    k_lens_chunk = torch.tensor([ctx_len] * b, dtype=torch.long, device=q.device)
                    out_chunk = attention(
                        q_chunk.to(dtype), k_ctx.to(dtype), v=v_ctx.to(dtype),
                        k_lens=k_lens_chunk,
                        window_size=self.window_size,
                    )
            else:
                if num_clean_chunks > 0:
                    # Clean half: stay within the clean prefix only.
                    limit = min(ci + 1, num_clean_chunks)
                    if self.max_attn_chunks == -1:
                        window_start = 0
                    else:
                        window_start = max(0, limit - self.max_attn_chunks)
                    ctx_start = chunk_boundaries[window_start][0]
                    ctx_end = chunk_boundaries[limit - 1][1]
                else:
                    if self.max_attn_chunks == -1:
                        window_start = 0
                        ctx_start = 0
                    else:
                        window_start = max(0, ci - self.max_attn_chunks + 1)
                        ctx_start = chunk_boundaries[window_start][0]
                    ctx_end = c_end

                if use_psv and num_clean_chunks > 0:
                    # Drop padded clean history, retaining ci itself; padded outputs are discarded
                    # to prevent phantom history from contaminating later clean K/V.
                    ranges = []
                    for bi in range(b):
                        n_before = int(
                            past_chunk_valid[bi, window_start:ci].sum().item()
                        )
                        first = ci - n_before  # contiguous: [first..ci]
                        ranges.append([(chunk_boundaries[first][0],
                                        chunk_boundaries[ci][1])])
                    out_chunk = self._per_sample_attn(q_chunk, k, v, ranges, dtype)
                elif use_psv:
                    ranges = []
                    for bi in range(b):
                        n_before = int(
                            past_chunk_valid[bi, window_start:ci].sum().item()
                        )
                        sample_ranges = []
                        if n_before > 0:
                            first = ci - n_before
                            sample_ranges.append((chunk_boundaries[first][0], c_start))
                        sample_ranges.append((c_start, c_end))
                        ranges.append(sample_ranges)
                    out_chunk = self._per_sample_attn(q_chunk, k, v, ranges, dtype)
                else:
                    k_ctx = k[:, ctx_start:ctx_end]
                    v_ctx = v[:, ctx_start:ctx_end]
                    ctx_len = ctx_end - ctx_start
                    k_lens_chunk = torch.tensor([ctx_len] * b, dtype=torch.long, device=q.device)
                    out_chunk = attention(
                        q_chunk.to(dtype), k_ctx.to(dtype), v=v_ctx.to(dtype),
                        k_lens=k_lens_chunk,
                        window_size=self.window_size,
                    )

            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1)
        out = out.to(dtype).flatten(2)
        out = self.o(out)
        return out


class CausalChunkAttentionBlock(nn.Module):
    """
    Transformer block using CausalChunkSelfAttention.
    Self-attention, text cross-attention, and feed-forward layers with AdaLN modulation.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        max_attn_chunks: int = -1,
        action_temporal_scale: float = 1.0,
        use_special_memory: bool = False,
        use_continuous_history: bool = False,
        special_memory_recent_chunks: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalChunkSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, max_attn_chunks,
            action_temporal_scale=action_temporal_scale,
            use_special_memory=use_special_memory,
            use_continuous_history=use_continuous_history,
            special_memory_recent_chunks=special_memory_recent_chunks,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        t: int = 0,
        temporal_stride: int = 4,
        chunk_boundaries: Optional[List[Tuple[int, int]]] = None,
        is_tf: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_clean_chunks: int = 0,
        past_chunk_valid: Optional[torch.Tensor] = None,
        collect_kv: bool = False,
        static_cache: bool = False,
        chunk_layouts: Optional[List[Tuple[int, int, int, int]]] = None,
        chunk_time_offsets: Optional[List[float]] = None,
        chunk_anchor_mask: Optional[torch.Tensor] = None,
        decouple_noisy_chunks: bool = False,
        joint_rope_freqs: Optional[torch.Tensor] = None,
        cross_attn_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if e.dim() > 3:
            e_chunks = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e_chunks = [ei.squeeze(2) for ei in e_chunks]
        else:
            e_chunks = (self.modulation + e).chunk(6, dim=1)

        temp_x = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        temp_x = temp_x.to(dtype)
        y, updated_kv_cache = self.self_attn(
            temp_x, seq_lens, grid_sizes, freqs, dtype,
            temporal_stride=temporal_stride,
            chunk_boundaries=chunk_boundaries,
            is_tf=is_tf,
            kv_cache=kv_cache,
            t=t,
            num_clean_chunks=num_clean_chunks,
            past_chunk_valid=past_chunk_valid,
            collect_kv=collect_kv,
            static_cache=static_cache,
            chunk_layouts=chunk_layouts,
            chunk_time_offsets=chunk_time_offsets,
            chunk_anchor_mask=chunk_anchor_mask,
            decouple_noisy_chunks=decouple_noisy_chunks,
            joint_rope_freqs=joint_rope_freqs,
            kv_workspace=kv_workspace,
        )
        x = x + y * e_chunks[2]

        # Cross-attention projections depend only on the instruction and can be
        # reused for every denoise step in an episode.
        normed_x = self.norm3(x)
        if cross_attn_kv is None:
            cross_out = self.cross_attn(
                normed_x, context, context_lens, dtype, t=t
            )
        else:
            cross_out = self.cross_attn.forward_with_kv(
                normed_x, cross_attn_kv, context_lens, dtype, t=t
            )
        x = x + cross_out

        temp_x = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        temp_x = temp_x.to(dtype)
        y = self.ffn(temp_x)
        x = x + y * e_chunks[5]

        return x, updated_kv_cache


class CausalChunkDiTModel(ModelMixin, ConfigMixin):
    """Wan diffusion backbone with causal chunk attention and an action prediction head."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        in_channels: int = 16,
        hidden_size: int = 2048,
        cross_attn_type: Optional[str] = None,
        max_attn_chunks: int = -1,
        action_temporal_scale: float = 1.0,
        action_dim: int = 3,
        traj_head_type: str = "linear",
        traj_head_hidden_dim: int = 768,
        use_special_memory: bool = False,
        use_continuous_history: bool = False,
        num_special_tokens: int = 8,
        special_memory_recent_chunks: int = 2,
        chunk_overlap_frames: int = 0,
    ):
        super().__init__()

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.max_attn_chunks = max_attn_chunks
        self.action_temporal_scale = action_temporal_scale
        self.action_dim = action_dim
        self.traj_head_type = traj_head_type
        self.traj_head_hidden_dim = int(traj_head_hidden_dim)
        if use_special_memory:
            raise ValueError("ScaleNav uses continuous history, without special memory tokens")
        self.use_special_memory = False
        self.use_continuous_history = use_continuous_history
        self.num_special_tokens = 0
        self.special_memory_recent_chunks = special_memory_recent_chunks
        self.chunk_overlap_frames = int(chunk_overlap_frames)
        if self.chunk_overlap_frames < 0:
            raise ValueError("chunk_overlap_frames must be non-negative")
        if self.use_continuous_history and not math.isclose(
            float(self.action_temporal_scale), 1.0
        ):
            raise ValueError(
                "use_continuous_history requires action_temporal_scale=1.0 so "
                "video and action tokens share one physical time axis"
            )

        # Video, text, and action embeddings.
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.traj_embed = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Transformer blocks (causal chunk variant)
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalChunkAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                max_attn_chunks=max_attn_chunks,
                action_temporal_scale=action_temporal_scale,
                use_special_memory=use_special_memory,
                use_continuous_history=use_continuous_history,
                special_memory_recent_chunks=special_memory_recent_chunks,
            )
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)
        if traj_head_type == "deep":
            self.traj_head = DeepActionHead(
                dim,
                action_dim,
                (1, 1, 1),
                eps,
                hidden_dim=self.traj_head_hidden_dim,
            )
        elif traj_head_type == "linear":
            self.traj_head = Head(dim, action_dim, (1, 1, 1), eps)
        else:
            raise ValueError(f"Unknown traj_head_type: {traj_head_type}")
        zero_initialize_layer(self.traj_head.head)

        self.register_parameter("special_tokens", None)
        self.register_parameter("anchor_embedding", None)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False


    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid gradient checkpointing argument")


    def _embed_text_context(self, context: List[torch.Tensor]) -> torch.Tensor:
        return self.text_embedding(torch.stack([
            torch.cat([
                u,
                u.new_zeros(self.text_len - u.size(0), u.size(1)),
            ])
            for u in context
        ]))

    @torch.no_grad()
    def prepare_inference_context(
        self,
        context: List[torch.Tensor],
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, object]:
        """Cache instruction features and per-layer cross-attention K/V."""
        context_emb = self._embed_text_context(context)
        cross_attn_kv = []
        for block in self.blocks:
            precompute = getattr(block.cross_attn, "precompute_kv", None)
            cross_attn_kv.append(
                precompute(context_emb, dtype) if precompute is not None else None
            )
        return {
            "context_emb": context_emb,
            "cross_attn_kv": cross_attn_kv,
        }

    def forward(
        self,
        x_chunks: List[List[torch.Tensor]],
        traj_chunks: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        is_tf: bool = False,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        temporal_stride: int = 4,
        output_all_chunks: bool = False,
        clean_x_chunks: Optional[List[List[torch.Tensor]]] = None,
        clean_traj_chunks: Optional[List[torch.Tensor]] = None,
        past_chunk_valid: Optional[torch.Tensor] = None,
        collect_kv: bool = False,
        static_kv: bool = False,
        prefill_only: bool = False,
        chunk_position_ids: Optional[torch.Tensor] = None,
        inference_context_cache: Optional[Dict[str, object]] = None,
        kv_workspaces: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        precomputed_joint_rope_freqs: Optional[torch.Tensor] = None,
        zero_time_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        """
        Forward pass through causal chunk DiT.

        Args:
            x_chunks: List of K chunks, each a list of per-sample [C_in, F, H, W]
                tensors. When `clean_x_chunks` is provided, these are the NOISY
                copies; otherwise they can be either clean past + noisy current
                when supervising a single target chunk.
            traj_chunks: List of K trajectory tensors, each [B, T, action_dim].
            t: Timesteps. Three accepted shapes:
                 - [B]: same timestep for all tokens in the batch.
                 - [B, noisy_chunk_seq_len]: TI2V / single-noisy-chunk mode.
                   Covers the video + traj tokens of the LAST chunk only; clean
                   past chunks get t=0 automatically.
                 - [B, noisy_seq_len]: in DreamZero mode (clean_x_chunks set),
                   `noisy_seq_len` must equal the token count of the NOISY half
                   (K * per-chunk len). Clean half tokens auto-receive t=0.
            context: List of text embeddings [L, C] per sample.
            seq_len: Per-chunk sequence length (video tokens per chunk).
            is_tf: Teacher forcing mode.
            kv_caches: Per-layer KV caches for inference.
            temporal_stride: VAE temporal compression ratio.
            output_all_chunks: If True, fire video/traj heads on every noisy
                chunk and return concatenated tokens along the sequence axis.
            clean_x_chunks / clean_traj_chunks: DreamZero-style clean context.
                When both are provided (lists of length K, matching x_chunks),
                the model builds a 2K-chunk sequence [clean_0..clean_{K-1},
                noisy_0..noisy_{K-1}]. Clean-half tokens take AdaLN t=0, noisy
                half uses the provided t. Noisy chunk i attends only to clean
                chunks 0..i-1 + its own tokens (never other noisy chunks, never
                its own clean copy). After the transformer blocks, the clean
                half is dropped and the heads fire on the K noisy chunks.

        Returns:
            (video_output, traj_output, updated_kv_caches)
            - Single target (output_all_chunks=False, no clean_x_chunks):
                video=[B, V_last, out_dim], traj=[B, T_last, action_dim].
            - DreamZero / output_all_chunks path:
                video=[B, K*V, out_dim], traj=[B, K*T, action_dim], with the
                caller splitting chunk-wise using known V and T.
        """
        device = self.patch_embedding.weight.device
        dtype = x_chunks[0][0].dtype if isinstance(x_chunks[0], list) else x_chunks[0].dtype

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        num_noisy_chunks = len(x_chunks)
        has_clean_prefix = clean_x_chunks is not None
        if has_clean_prefix:
            assert clean_traj_chunks is not None, (
                "clean_traj_chunks must be provided alongside clean_x_chunks"
            )
            assert len(clean_x_chunks) == num_noisy_chunks, (
                f"clean_x_chunks length {len(clean_x_chunks)} must equal "
                f"x_chunks length {num_noisy_chunks} in DreamZero mode"
            )
            assert len(clean_traj_chunks) == num_noisy_chunks, (
                "clean_traj_chunks and x_chunks length mismatch"
            )
            # Build combined sequence [clean_0..clean_{K-1}, noisy_0..noisy_{K-1}].
            combined_x_chunks = list(clean_x_chunks) + list(x_chunks)
            combined_traj_chunks = list(clean_traj_chunks) + list(traj_chunks)
            num_clean_chunks_for_attn = num_noisy_chunks
        else:
            combined_x_chunks = x_chunks
            combined_traj_chunks = traj_chunks
            num_clean_chunks_for_attn = 0

        B = (
            len(combined_x_chunks[0])
            if isinstance(combined_x_chunks[0], list)
            else combined_x_chunks[0].shape[0]
        )
        if chunk_position_ids is None:
            base_chunk_positions = torch.arange(
                num_noisy_chunks, device=device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1)
        else:
            base_chunk_positions = torch.as_tensor(
                chunk_position_ids, device=device, dtype=torch.float32
            )
            if base_chunk_positions.ndim == 1:
                base_chunk_positions = base_chunk_positions.unsqueeze(0).expand(B, -1)
            if base_chunk_positions.shape != (B, num_noisy_chunks):
                raise ValueError(
                    f"chunk_position_ids shape {tuple(base_chunk_positions.shape)} "
                    f"must be {(B, num_noisy_chunks)}"
                )
        combined_chunk_positions = (
            torch.cat([base_chunk_positions, base_chunk_positions], dim=1)
            if has_clean_prefix else base_chunk_positions
        )

        num_chunks = len(combined_x_chunks)
        # Shadow refs so the downstream patchify loop uses the combined sequence.
        x_chunks = combined_x_chunks
        traj_chunks = combined_traj_chunks
        B = len(x_chunks[0]) if isinstance(x_chunks[0], list) else x_chunks[0].shape[0]

        all_chunk_tokens = []
        all_grid_sizes = []
        chunk_boundaries = []
        traj_lens = []
        video_lens = []
        chunk_layouts: List[Tuple[int, int, int, int]] = []
        latent_advance: Optional[int] = None
        current_pos = 0

        for ci in range(num_chunks):
            chunk_x = x_chunks[ci]
            chunk_traj = traj_chunks[ci]
            traj_len = chunk_traj.shape[1]
            traj_lens.append(traj_len)

            patched = [self.patch_embedding(u.unsqueeze(0)) for u in chunk_x]
            grid_size = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in patched])
            patched = [u.flatten(2).transpose(1, 2) for u in patched]
            video_len = patched[0].shape[1]
            video_lens.append(video_len)

            latent_frames, grid_h, grid_w = grid_size[0].tolist()
            special_len = 0
            # Keep chunk zero's first latent slice as a permanent RGB anchor.
            anchor_len = grid_h * grid_w if self.use_continuous_history else 0
            chunk_layouts.append((video_len, special_len, traj_len, anchor_len))
            this_advance = max(1, latent_frames - 1)
            if latent_advance is None:
                latent_advance = this_advance
            elif latent_advance != this_advance:
                raise ValueError("All chunks must have the same latent temporal length")

            traj_embedding = self.traj_embed(chunk_traj)

            # Concatenate video and trajectory tokens.
            chunk_tokens = []
            for bi, (u, v) in enumerate(zip(patched, traj_embedding)):
                video_tokens = u
                chunk_tokens.append(torch.cat([video_tokens, v.unsqueeze(0)], dim=1))

            chunk_len = chunk_tokens[0].shape[1]
            chunk_boundaries.append((current_pos, current_pos + chunk_len))
            current_pos += chunk_len

            all_chunk_tokens.append(chunk_tokens)
            all_grid_sizes.append(grid_size)

        if self.use_continuous_history:
            if len(set(traj_lens)) != 1:
                raise ValueError(
                    "Continuous history requires equal action lengths in every chunk"
                )
            # Video pretraining intentionally passes zero action tokens. Derive
            # the raw-frame span from the video latent geometry in that case:
            # F_raw = 1 + (F_latent - 1) * temporal_stride (9 -> 3, 5 -> 2).
            raw_frames_per_chunk = 1 + (
                int(all_grid_sizes[0][0, 0].item()) - 1
            ) * temporal_stride
            timing_frames = traj_lens[0] or raw_frames_per_chunk
            raw_chunk_advance = timing_frames - self.chunk_overlap_frames
            if raw_chunk_advance <= 0:
                raise ValueError(
                    "chunk_overlap_frames must be smaller than the chunk frame span; "
                    f"got overlap={self.chunk_overlap_frames}, frames={timing_frames}"
                )
            # Convert raw chunk strides to latent-time RoPE units (9/4 without overlap).
            time_advance = float(raw_chunk_advance) / float(temporal_stride)
            chunk_time_offsets = combined_chunk_positions * time_advance
        else:
            chunk_time_offsets = combined_chunk_positions * float(latent_advance or 1)
        chunk_anchor_mask = combined_chunk_positions.eq(0)

        # Concatenate all chunks into a single sequence
        # all_chunk_tokens[ci] is a list of B tensors, each [1, chunk_len, dim]
        combined_tokens = []
        for bi in range(B):
            sample_tokens = torch.cat(
                [all_chunk_tokens[ci][bi] for ci in range(num_chunks)], dim=1
            )
            combined_tokens.append(sample_tokens)

        total_seq_len = current_pos

        x = torch.cat([
            torch.cat([u, u.new_zeros(1, total_seq_len - u.size(1), u.size(2))], dim=1)
            for u in combined_tokens
        ])  # [B, total_seq_len, dim]

        seq_lens_combined = torch.tensor([total_seq_len] * B, dtype=torch.long)

        # --- Time embeddings ---
        # Four paths:
        #   (a) t.dim() == 1                          → same t for everything
        #   (b) t.size(1) == last_chunk_len           → TI2V / single-noisy-chunk
        #                                               past chunks get t=0
        #   (c) t.size(1) == total_seq_len            → per-token over ALL chunks
        #                                               (Diffusion-Forcing)
        #   (d) DreamZero: clean_x_chunks is set and t covers the NOISY half
        #       only (size K * per_chunk_len). Clean-half tokens auto-receive
        #       t=0 via zero-padding the prefix.

        with amp.autocast(dtype=torch.float32):
            if t.dim() == 1 and zero_time_embedding is not None:
                e, e0 = zero_time_embedding
                if e.shape != (B, self.dim) or e0.shape != (B, 6, self.dim):
                    raise ValueError(
                        "Cached zero-time embedding shape mismatch: "
                        f"e={tuple(e.shape)}, e0={tuple(e0.shape)}"
                    )
                traj_e = e
            elif t.dim() != 1:
                last_start, last_end = chunk_boundaries[-1]
                last_chunk_len = last_end - last_start

                if has_clean_prefix:
                    # Path (d): t must cover the noisy half (chunks K..2K-1).
                    noisy_seq_start = chunk_boundaries[num_clean_chunks_for_attn][0]
                    noisy_seq_len = total_seq_len - noisy_seq_start
                    assert t.size(1) == noisy_seq_len, (
                        f"DreamZero: per-token timestep length {t.size(1)} must "
                        f"equal noisy-half length {noisy_seq_len}."
                    )
                    t_clean = torch.zeros(B, noisy_seq_start, device=t.device, dtype=t.dtype)
                    t_full = torch.cat([t_clean, t], dim=1)
                elif t.size(1) == total_seq_len:
                    # Path (c): per-token timesteps already covering the whole sequence.
                    t_full = t
                elif t.size(1) == last_chunk_len:
                    # Path (b): only last chunk is noisy; pad the clean past with zeros.
                    clean_len = last_start
                    t_clean = torch.zeros(B, clean_len, device=t.device, dtype=t.dtype)
                    t_full = torch.cat([t_clean, t], dim=1)
                else:
                    raise AssertionError(
                        f"Per-token timestep length {t.size(1)} must match either "
                        f"last chunk length ({last_chunk_len}) or total sequence "
                        f"length ({total_seq_len})."
                    )

                bt = t_full.size(0)
                ft = t_full.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, total_seq_len)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))

                # Multi-target training slices these embeddings per chunk below.
                traj_e = e[:, last_end - traj_lens[-1]:last_end]
                video_e = e[:, last_start:last_start + video_lens[-1]]
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                traj_e = e

        if inference_context_cache is None:
            context_emb = self._embed_text_context(context)
            cross_attn_kv = None
        else:
            context_emb = inference_context_cache["context_emb"]
            cross_attn_kv = inference_context_cache["cross_attn_kv"]
            if context_emb.shape[0] != B:
                raise ValueError(
                    f"Inference context batch {context_emb.shape[0]} does not match {B}"
                )
            if len(cross_attn_kv) != self.num_layers:
                raise ValueError(
                    f"Expected {self.num_layers} cross-attention caches, "
                    f"got {len(cross_attn_kv)}"
                )

        if len(all_grid_sizes) > 1:
            for ci in range(1, len(all_grid_sizes)):
                assert (all_grid_sizes[ci] == all_grid_sizes[0]).all(), (
                    f"Chunk {ci} grid_sizes {all_grid_sizes[ci].tolist()} != "
                    f"chunk 0 grid_sizes {all_grid_sizes[0].tolist()}. "
                    f"All chunks must have identical spatial dimensions for RoPE."
                )
        rope_grid_sizes = all_grid_sizes[0]  # [B, 3] — verified same for all chunks

        if kv_workspaces is not None and len(kv_workspaces) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} KV workspaces, got {len(kv_workspaces)}"
            )

        context_lens = None
        updated_kv_caches = [None] * self.num_layers if kv_caches is None else list(kv_caches)

        decouple_noisy_chunks = bool(
            getattr(self, "decouple_future_action", False) and not prefill_only
        )

        for layer_idx, block in enumerate(self.blocks):
            layer_kv = updated_kv_caches[layer_idx] if kv_caches is not None else None
            layer_cross_kv = (
                cross_attn_kv[layer_idx] if cross_attn_kv is not None else None
            )
            layer_workspace = (
                kv_workspaces[layer_idx] if kv_workspaces is not None else None
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing and is_tf:
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # Can't easily checkpoint with kv_cache returns, so only for training
                x_out, kv_out = torch.utils.checkpoint.checkpoint(
                    block, x, e0, seq_lens_combined, rope_grid_sizes, self.freqs,
                    context_emb, context_lens, dtype, 0, temporal_stride,
                    chunk_boundaries, is_tf, layer_kv, num_clean_chunks_for_attn,
                    past_chunk_valid,
                    False, False, chunk_layouts, chunk_time_offsets,
                    chunk_anchor_mask,
                    **ckpt_kwargs,
                )
                x = x_out
            else:
                x, kv_out = block(
                    x, e=e0, seq_lens=seq_lens_combined, grid_sizes=rope_grid_sizes,
                    freqs=self.freqs, context=context_emb, context_lens=context_lens,
                    dtype=dtype, t=0, temporal_stride=temporal_stride,
                    chunk_boundaries=chunk_boundaries, is_tf=is_tf, kv_cache=layer_kv,
                    num_clean_chunks=num_clean_chunks_for_attn,
                    past_chunk_valid=past_chunk_valid,
                    collect_kv=collect_kv,
                    static_cache=static_kv,
                    chunk_layouts=chunk_layouts,
                    chunk_time_offsets=chunk_time_offsets,
                    chunk_anchor_mask=chunk_anchor_mask,
                    decouple_noisy_chunks=decouple_noisy_chunks,
                    joint_rope_freqs=precomputed_joint_rope_freqs,
                    cross_attn_kv=layer_cross_kv,
                    kv_workspace=layer_workspace,
                )
            # static_cache mode returns kv_out=None on purpose — keep the
            # caller-provided cache instead of overwriting it.
            if kv_out is not None or kv_caches is None:
                updated_kv_caches[layer_idx] = kv_out

        # Prefill mode: the caller only wants the per-layer clean-past K/V —
        # skip the output heads entirely.
        if prefill_only:
            return None, None, updated_kv_caches

        # --- Output heads ---
        # When DreamZero's clean prefix is present, the first K chunks are the
        # clean-context "memory bank" and should NOT receive a gradient from
        # the heads. Iterate only over the noisy half.
        if output_all_chunks or has_clean_prefix:
            iter_start = num_clean_chunks_for_attn if has_clean_prefix else 0
            video_outs: List[torch.Tensor] = []
            traj_outs: List[torch.Tensor] = []
            for ci in range(iter_start, len(chunk_boundaries)):
                c_start, c_end = chunk_boundaries[ci]
                chunk_x = x[:, c_start:c_end]
                c_video_len = video_lens[ci]
                c_special_len = chunk_layouts[ci][1]
                c_traj_len = traj_lens[ci]
                c_video_tokens = chunk_x[:, :c_video_len]
                c_traj_start = c_video_len + c_special_len
                c_traj_tokens = chunk_x[:, c_traj_start:c_traj_start + c_traj_len]

                if t.dim() != 1:
                    # Use the per-chunk slice of the full time embedding.
                    c_video_e = e[:, c_start:c_start + c_video_len]
                    c_traj_e = e[:, c_start + c_traj_start:c_start + c_traj_start + c_traj_len]
                    video_outs.append(self.head(c_video_tokens, c_video_e))
                    traj_outs.append(self.traj_head(c_traj_tokens, c_traj_e) if c_traj_len > 0 else c_traj_tokens.new_zeros(B, 0, self.action_dim))
                else:
                    video_outs.append(self.head(c_video_tokens, e))
                    traj_outs.append(self.traj_head(c_traj_tokens, e) if c_traj_len > 0 else c_traj_tokens.new_zeros(B, 0, self.action_dim))

            video_out = torch.cat(video_outs, dim=1)
            traj_out = torch.cat(traj_outs, dim=1)
            return video_out, traj_out, updated_kv_caches

        # Predict the current chunk from its clean history.
        last_start, last_end = chunk_boundaries[-1]
        last_chunk_x = x[:, last_start:last_end]
        last_video_len = video_lens[-1]
        last_special_len = chunk_layouts[-1][1]
        last_traj_len = traj_lens[-1]

        video_tokens = last_chunk_x[:, :last_video_len]
        traj_start = last_video_len + last_special_len
        traj_tokens = last_chunk_x[:, traj_start:traj_start + last_traj_len]

        if t.dim() != 1:
            video_out = self.head(video_tokens, video_e)
        else:
            video_out = self.head(video_tokens, e)
        traj_out = (
            self.traj_head(traj_tokens, traj_e)
            if last_traj_len > 0
            else traj_tokens.new_zeros(B, 0, self.action_dim)
        )

        return video_out, traj_out, updated_kv_caches

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> List[torch.Tensor]:
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


class CausalChunkDiTWrapper(nn.Module):
    """Assemble clean history and current video/action chunks for training and inference."""

    def __init__(self, causal_dit: CausalChunkDiTModel):
        super().__init__()
        self.dit_model = causal_dit
        self.config = causal_dit.config

    @property
    def patch_size(self):
        return self.dit_model.patch_size

    @torch.no_grad()
    def prepare_inference_context(
        self,
        context: List[torch.Tensor],
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, object]:
        return self.dit_model.prepare_inference_context(context, dtype=dtype)

    @torch.no_grad()
    def prepare_joint_kv_workspaces(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        current_token_len: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Allocate persistent [clean prefix | current video/action] K/V buffers."""
        if len(kv_caches) != self.dit_model.num_layers:
            raise ValueError(
                f"Expected {self.dit_model.num_layers} layer caches, got {len(kv_caches)}"
            )
        workspaces = []
        for cached_k, cached_v in kv_caches:
            shape = list(cached_k.shape)
            shape[1] += int(current_token_len)
            full_k = cached_k.new_empty(shape)
            full_v = cached_v.new_empty(shape)
            prefix_len = cached_k.shape[1]
            full_k[:, :prefix_len].copy_(cached_k)
            full_v[:, :prefix_len].copy_(cached_v)
            workspaces.append((full_k, full_v))
        return workspaces


    @torch.no_grad()
    def prepare_joint_schedule_cache(
        self,
        video_latent: torch.Tensor,
        latent_mask: torch.Tensor,
        seq_len: int,
        timesteps: torch.Tensor,
        action_len: int,
        temporal_stride: int = 4,
        action_timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Precompute joint RoPE and exact per-token integer timesteps."""
        if video_latent.ndim != 5 or latent_mask.shape != video_latent.shape:
            raise ValueError(
                "video_latent and latent_mask must share [B,C,T,H,W] shape"
            )
        batch_size = video_latent.shape[0]
        patch_t, patch_h, patch_w = self.patch_size
        grid_sizes = torch.tensor([
            [
                video_latent.shape[2] // patch_t,
                video_latent.shape[3] // patch_h,
                video_latent.shape[4] // patch_w,
            ]
        ] * batch_size, dtype=torch.long)
        actual_seq_len = int(torch.prod(grid_sizes[0]).item())
        if actual_seq_len != int(seq_len):
            raise ValueError(
                f"seq_len={seq_len} does not match latent grid token count "
                f"{actual_seq_len}"
            )

        model = self.dit_model
        device = video_latent.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)
        joint_rope_freqs = build_rope_freqs(
            grid_sizes,
            model.freqs,
            action_len=action_len,
            device=device,
            temporal_stride=temporal_stride,
            action_temporal_scale=model.action_temporal_scale,
        )
        mask_tokens = latent_mask[
            :, 0, ::patch_t, ::patch_h, ::patch_w
        ].flatten(1)[:, :seq_len]
        action_timesteps = (
            timesteps if action_timesteps is None else action_timesteps
        )
        if len(timesteps) != len(action_timesteps):
            raise ValueError("Video and action schedules must have equal length")
        token_timesteps = []
        for video_timestep, action_timestep in zip(
            timesteps, action_timesteps
        ):
            video_scalar = video_timestep.reshape(1).expand(batch_size)
            action_scalar = action_timestep.reshape(1).expand(batch_size)
            t_video = (mask_tokens * video_scalar.float()).long()
            t_action = action_scalar[:, None].expand(-1, action_len).long()
            token_timesteps.append(
                torch.cat([t_video, t_action], dim=1).contiguous()
            )
        return {
            "joint_rope_freqs": joint_rope_freqs,
            "token_timesteps": token_timesteps,
            "zero_time_embedding": self.prepare_zero_time_embedding(
                batch_size, device
            ),
        }


    @torch.no_grad()
    def prepare_zero_time_embedding(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute the t=0 AdaLN input shared by every clean prefill."""
        model = self.dit_model
        t0 = torch.zeros(int(batch_size), device=device)
        with amp.autocast(dtype=torch.float32):
            embedded = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t0).float()
            )
            projected = model.time_projection(embedded).unflatten(
                1, (6, model.dim)
            )
        return embedded, projected

    @torch.no_grad()
    def append_continuous_history_kv(
        self,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        video_latent: torch.Tensor,
        trajectory: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        *,
        chunk_position_id: int,
        total_history_chunks: int,
        temporal_stride: int = 4,
        inference_context_cache: Optional[Dict[str, object]] = None,
        zero_time_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Append one clean chunk to a persistent continuous-history cache.

        ``prefill_past_kv`` intentionally rebuilds a complete clean history.
        That is useful for one-shot evaluation, but a robot rollout changes by
        only one durable GT chunk per request.  This method performs the exact
        block-causal update for that one chunk and then applies the same
        anchor-plus-recent compaction as the full prefill path.

        Special-memory has a different multi-token retention layout and keeps
        using the reference full-prefill implementation for now.
        """
        model = self.dit_model
        if not model.use_continuous_history:
            raise ValueError(
                "incremental history KV currently requires continuous history "
                "without learned special-memory tokens"
            )
        if video_latent.ndim != 5 or trajectory.ndim != 3:
            raise ValueError(
                "video_latent/trajectory must be [B,C,T,H,W] and [B,T,A]"
            )
        batch_size = int(video_latent.shape[0])
        if int(trajectory.shape[0]) != batch_size:
            raise ValueError("video/action batch sizes do not match")
        if kv_caches is None:
            caches = self.prefill_past_kv(
                [video_latent],
                [trajectory],
                context,
                seq_len,
                temporal_stride=temporal_stride,
                past_chunk_position_ids=torch.full(
                    (batch_size, 1),
                    int(chunk_position_id),
                    device=video_latent.device,
                    dtype=torch.long,
                ),
                inference_context_cache=inference_context_cache,
                zero_time_embedding=zero_time_embedding,
            )
            if caches is None:
                raise RuntimeError("single-chunk KV prefill returned no cache")
            return caches

        if len(kv_caches) != model.num_layers:
            raise ValueError(
                f"Expected {model.num_layers} layer caches, got {len(kv_caches)}"
            )
        t0 = torch.zeros(batch_size, device=video_latent.device)
        _, _, caches = model(
            x_chunks=[[video_latent[bi] for bi in range(batch_size)]],
            traj_chunks=[trajectory],
            t=t0,
            context=context,
            seq_len=seq_len,
            is_tf=False,
            kv_caches=kv_caches,
            temporal_stride=temporal_stride,
            collect_kv=True,
            static_kv=True,
            prefill_only=True,
            chunk_position_ids=torch.full(
                (batch_size, 1),
                int(chunk_position_id),
                device=video_latent.device,
                dtype=torch.float32,
            ),
            inference_context_cache=inference_context_cache,
            zero_time_embedding=zero_time_embedding,
        )
        if caches is None or any(cache is None for cache in caches):
            raise RuntimeError("incremental clean-history prefill returned no cache")

        # Full chunk token count is constant for the deployed C5 model.
        patched = model.patch_embedding(video_latent[0:1])
        _, _, latent_frames, grid_h, grid_w = patched.shape
        video_tokens = int(latent_frames * grid_h * grid_w)
        full_chunk_tokens = video_tokens + int(trajectory.shape[1])
        anchor_tokens = int(grid_h * grid_w)
        total_history_chunks = int(total_history_chunks)
        recent_chunks = (
            total_history_chunks
            if int(model.max_attn_chunks) == -1
            else max(0, int(model.max_attn_chunks) - 2)
        )
        if total_history_chunks <= recent_chunks:
            return caches  # type: ignore[return-value]

        keep_recent_tokens = recent_chunks * full_chunk_tokens
        compacted: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for cached_k, cached_v in caches:  # type: ignore[misc]
            if cached_k.shape[1] < anchor_tokens + keep_recent_tokens:
                raise RuntimeError(
                    "incremental KV cache is shorter than its compact layout: "
                    f"tokens={cached_k.shape[1]} anchor={anchor_tokens} "
                    f"recent={keep_recent_tokens}"
                )
            parts_k = [cached_k[:, :anchor_tokens]]
            parts_v = [cached_v[:, :anchor_tokens]]
            if keep_recent_tokens:
                parts_k.append(cached_k[:, -keep_recent_tokens:])
                parts_v.append(cached_v[:, -keep_recent_tokens:])
            compacted.append((
                torch.cat(parts_k, dim=1).detach(),
                torch.cat(parts_v, dim=1).detach(),
            ))
        return compacted

    @torch.no_grad()
    def retain_continuous_history_kv(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        old_positions: List[int],
        retained_positions: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Drop evicted full chunks from an unbounded streaming KV cache.

        This is the standard autoregressive sliding-window operation: retained
        K/V keep the contextualization they had when first produced, while the
        explicitly evicted chunk's keys and values are removed.  It avoids an
        O(history) re-prefill every time the robot's bounded history advances.
        """
        model = self.dit_model
        if not model.use_continuous_history:
            raise ValueError(
                "streaming KV retention requires continuous history without "
                "learned special-memory tokens"
            )
        if int(model.max_attn_chunks) != -1:
            raise ValueError(
                "explicit KV retention is only valid for max_attn_chunks=-1"
            )
        old_positions = [int(position) for position in old_positions]
        retained_positions = [int(position) for position in retained_positions]
        if not old_positions or not retained_positions:
            raise ValueError("old and retained position lists must be non-empty")
        if len(set(old_positions)) != len(old_positions):
            raise ValueError("old_positions contains duplicates")
        old_index = {position: index for index, position in enumerate(old_positions)}
        if any(position not in old_index for position in retained_positions):
            raise ValueError("retained_positions must be a subset of old_positions")
        retained_indices = [old_index[position] for position in retained_positions]
        if retained_indices != sorted(retained_indices):
            raise ValueError("retained_positions must preserve cache order")

        retained: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for cached_k, cached_v in kv_caches:
            if cached_k.shape != cached_v.shape:
                raise ValueError("cached K/V shapes do not match")
            if cached_k.shape[1] % len(old_positions):
                raise ValueError(
                    "cache token length is not divisible by its full chunk count"
                )
            tokens_per_chunk = cached_k.shape[1] // len(old_positions)
            k_parts = [
                cached_k[:, index * tokens_per_chunk:(index + 1) * tokens_per_chunk]
                for index in retained_indices
            ]
            v_parts = [
                cached_v[:, index * tokens_per_chunk:(index + 1) * tokens_per_chunk]
                for index in retained_indices
            ]
            retained.append((
                torch.cat(k_parts, dim=1).detach(),
                torch.cat(v_parts, dim=1).detach(),
            ))
        return retained

    @torch.no_grad()
    def prefill_past_kv(
        self,
        past_video_latents: Optional[List[torch.Tensor]],
        past_trajectories: Optional[List[torch.Tensor]],
        context: List[torch.Tensor],
        seq_len: int,
        temporal_stride: int = 4,
        past_chunk_mask: Optional[torch.Tensor] = None,
        past_chunk_position_ids: Optional[torch.Tensor] = None,
        inference_context_cache: Optional[Dict[str, object]] = None,
        zero_time_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Encode the clean past chunks ONCE and return per-layer post-RoPE (K, V)
        static caches for the denoise loop.

        The past chunks are block-causal among themselves and use AdaLN t=0.
        In either anchor-history mode their post-RoPE K/V are compacted to the
        first-frame anchor and recent full chunks (plus learned tokens only for
        special memory). The clean past never attends the noisy current chunk,
        so these K/V are stable across all denoise steps. Rebuild the cache once
        after the past window changes.

        Returns None when there is no (unmasked) past chunk — the caller should
        then fall back to the plain single-chunk forward.
        """
        if not past_video_latents:
            return None

        x_chunks: List[List[torch.Tensor]] = []
        traj_chunks: List[torch.Tensor] = []
        active_indices: List[int] = []
        for ci, (past_lat, past_traj) in enumerate(
            zip(past_video_latents, past_trajectories)
        ):
            if past_chunk_mask is not None:
                mask_val = past_chunk_mask
                if mask_val.ndim == 2:
                    chunk_active = mask_val[:, ci].sum().item() > 0
                else:
                    chunk_active = mask_val[ci].item() > 0
                if not chunk_active:
                    continue
            x_chunks.append([past_lat[bi] for bi in range(past_lat.shape[0])])
            traj_chunks.append(past_traj)
            active_indices.append(ci)

        if not x_chunks:
            return None

        B = len(x_chunks[0])
        device = x_chunks[0][0].device
        t0 = torch.zeros(B, device=device)

        _, _, caches = self.dit_model(
            x_chunks=x_chunks,
            traj_chunks=traj_chunks,
            t=t0,
            context=context,
            seq_len=seq_len,
            is_tf=False,
            temporal_stride=temporal_stride,
            collect_kv=True,
            prefill_only=True,
            chunk_position_ids=(
                past_chunk_position_ids[:, active_indices]
                if past_chunk_position_ids is not None else None
            ),
            inference_context_cache=inference_context_cache,
            zero_time_embedding=zero_time_embedding,
        )
        # Crop after full prefill so retained K/V preserve their contextualized states.
        if not self.dit_model.use_continuous_history and self.dit_model.max_attn_chunks > 0:
            tokens_per_chunk = int(seq_len) + int(traj_chunks[-1].shape[1])
            max_past_tokens = max(0, self.dit_model.max_attn_chunks - 1) * tokens_per_chunk
            cropped = []
            for cached_k, cached_v in caches:
                if cached_k.shape[1] <= max_past_tokens:
                    cropped.append((cached_k, cached_v))
                elif max_past_tokens == 0:
                    cropped.append((cached_k[:, :0], cached_v[:, :0]))
                else:
                    cropped.append((
                        cached_k[:, -max_past_tokens:],
                        cached_v[:, -max_past_tokens:],
                    ))
            caches = cropped
        return caches

    def forward(
        self,
        x: List[torch.Tensor],
        traj: torch.Tensor,
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        past_video_latents: Optional[List[torch.Tensor]] = None,
        past_trajectories: Optional[List[torch.Tensor]] = None,
        past_chunk_mask: Optional[torch.Tensor] = None,
        past_chunk_position_ids: Optional[torch.Tensor] = None,
        current_chunk_position_id: Optional[torch.Tensor] = None,
        is_training: bool = False,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        temporal_stride: int = 4,
        return_kv_caches: bool = False,
        static_kv_cache: bool = False,
        denoise_all_chunks: bool = False,
        clean_x: Optional[List[torch.Tensor]] = None,
        clean_traj: Optional[torch.Tensor] = None,
        clean_past_video_latents: Optional[List[torch.Tensor]] = None,
        clean_past_trajectories: Optional[List[torch.Tensor]] = None,
        inference_context_cache: Optional[Dict[str, object]] = None,
        kv_workspaces: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        precomputed_joint_rope_freqs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Per-sample current chunk video tensors [C, F, H, W]. In
                `denoise_all_chunks` mode this is the NOISY version of the
                current chunk.
            traj: Current chunk trajectory [B, T, 3] (noisy in DreamZero mode).
            t: Timesteps.
            context: Text embeddings.
            seq_len: Per-chunk video token count.
            past_video_latents: List of N past chunk latents, each
                [B, C_l, T_l, H_l, W_l]. In teacher-forcing mode these are the
                clean past; in DreamZero mode they are the NOISY past versions.
            past_trajectories: Per-chunk past trajectories (noisy in DreamZero).
            past_chunk_mask: [B, N] or [N] mask: 1.0 = real chunk, 0.0 = padded.
            is_training: Training mode flag.
            kv_caches: Per-layer KV caches for inference.
            denoise_all_chunks: DreamZero mode switch. Requires `clean_x` and
                `clean_past_video_latents` (and their traj counterparts) to
                be populated — the model then builds a 2K-chunk sequence where
                the first K are clean GT tokens (context) and the last K are
                noisy copies (prediction targets). Returns one video/traj
                tensor per noisy chunk:
                    video_out : [B, K, out_dim, F, H, W]
                    traj_out  : [B, K, T, action_dim]
            clean_x / clean_past_video_latents / clean_traj / clean_past_trajectories:
                Clean-context counterparts required for DreamZero mode.
        """
        B = len(x)

        def build_chunk_positions(active_past_indices: List[int]) -> Optional[torch.Tensor]:
            if current_chunk_position_id is None:
                return None
            current_pos = torch.as_tensor(
                current_chunk_position_id, device=t.device, dtype=torch.float32
            ).reshape(B, 1)
            if not active_past_indices:
                return current_pos
            if past_chunk_position_ids is not None:
                past_pos = torch.as_tensor(
                    past_chunk_position_ids, device=t.device, dtype=torch.float32
                )[:, active_past_indices]
            else:
                offsets = torch.arange(
                    len(active_past_indices), 0, -1,
                    device=t.device, dtype=torch.float32,
                ).unsqueeze(0)
                past_pos = current_pos - offsets
            return torch.cat([past_pos, current_pos], dim=1)

        updated_caches = None
        if is_training:
            # Teacher forcing: clean past chunks + noisy current chunk.
            # Filter out zero-padded chunks using past_chunk_mask.
            x_chunks = []
            traj_chunks = []
            clean_x_chunks: Optional[List[List[torch.Tensor]]] = None
            clean_traj_chunks: Optional[List[torch.Tensor]] = None
            active_past_indices: List[int] = []

            if denoise_all_chunks:
                assert clean_x is not None and clean_past_video_latents is not None, (
                    "DreamZero mode needs clean_x and clean_past_video_latents"
                )
                assert clean_traj is not None and clean_past_trajectories is not None, (
                    "DreamZero mode needs clean_traj and clean_past_trajectories"
                )
                clean_x_chunks = []
                clean_traj_chunks = []

            if past_video_latents:
                for ci, (past_lat, past_traj) in enumerate(
                    zip(past_video_latents, past_trajectories)
                ):
                    # Skip padded (zero) chunks if mask is available
                    if past_chunk_mask is not None:
                        mask_val = past_chunk_mask
                        if mask_val.ndim == 2:
                            # [B, N] — skip only if ALL samples in batch are padded
                            chunk_active = mask_val[:, ci].sum().item() > 0
                        else:
                            # [N] — shared across batch
                            chunk_active = mask_val[ci].item() > 0
                        if not chunk_active:
                            continue

                    # Past chunks: per-sample list of [C, F, H, W]
                    chunk_x_list = [past_lat[bi] for bi in range(B)]
                    x_chunks.append(chunk_x_list)
                    traj_chunks.append(past_traj)
                    active_past_indices.append(ci)

                    if denoise_all_chunks:
                        clean_past_lat = clean_past_video_latents[ci]
                        clean_past_tr = clean_past_trajectories[ci]
                        clean_x_chunks.append([clean_past_lat[bi] for bi in range(B)])
                        clean_traj_chunks.append(clean_past_tr)

            # Current chunk (noisy in both modes) — always appended last.
            x_chunks.append(x)
            traj_chunks.append(traj)

            if denoise_all_chunks:
                clean_x_chunks.append(clean_x)
                clean_traj_chunks.append(clean_traj)

            # Build [B, K] validity for surviving past chunks plus the real current chunk.
            # Attention drops each sample's padded slots, matching inference.
            _past_chunk_valid = None
            if (
                past_chunk_mask is not None
                and past_chunk_mask.ndim == 2
                and past_video_latents
            ):
                _surv = [ci for ci in range(len(past_video_latents))
                         if past_chunk_mask[:, ci].sum().item() > 0]
                if _surv:
                    _valid_past = past_chunk_mask[:, _surv].to(torch.float32)
                    _cur = torch.ones(
                        B, 1, dtype=torch.float32, device=_valid_past.device
                    )
                    _past_chunk_valid = torch.cat([_valid_past, _cur], dim=1)  # [B, K]

            video_out, traj_out, updated_caches = self.dit_model(
                x_chunks=x_chunks,
                traj_chunks=traj_chunks,
                t=t,
                context=context,
                seq_len=seq_len,
                is_tf=True,
                temporal_stride=temporal_stride,
                output_all_chunks=denoise_all_chunks,
                clean_x_chunks=clean_x_chunks,
                clean_traj_chunks=clean_traj_chunks,
                past_chunk_valid=_past_chunk_valid,
                chunk_position_ids=build_chunk_positions(active_past_indices),
            )
        else:
            x_chunks = [x]
            traj_chunks = [traj]

            # With a static (prefilled) KV cache the clean past is already in
            # the cache — feeding the past chunks again would double-count them.
            use_static = bool(static_kv_cache and kv_caches is not None)
            active_past_count = 0
            active_past_indices: List[int] = []

            if past_video_latents and not use_static:
                for ci, (past_lat, past_traj) in enumerate(
                    zip(past_video_latents, past_trajectories)
                ):
                    if past_chunk_mask is not None:
                        mask_val = past_chunk_mask
                        if mask_val.ndim == 2:
                            chunk_active = mask_val[:, ci].sum().item() > 0
                        else:
                            chunk_active = mask_val[ci].item() > 0
                        if not chunk_active:
                            continue

                    chunk_x_list = [past_lat[bi] for bi in range(B)]
                    x_chunks.insert(-1, chunk_x_list)
                    traj_chunks.insert(-1, past_traj)
                    active_past_count += 1
                    active_past_indices.append(ci)

            if use_static and past_video_latents:
                for ci in range(len(past_video_latents)):
                    if past_chunk_mask is None:
                        active_past_count += 1
                    else:
                        mask_val = past_chunk_mask
                        chunk_active = (
                            mask_val[:, ci].sum().item() > 0
                            if mask_val.ndim == 2 else mask_val[ci].item() > 0
                        )
                        active_past_count += int(chunk_active)
                        if chunk_active:
                            active_past_indices.append(ci)

            chunk_positions = build_chunk_positions(active_past_indices)
            if use_static:
                chunk_positions = chunk_positions[:, -1:] if chunk_positions is not None else None

            video_out, traj_out, updated_caches = self.dit_model(
                x_chunks=x_chunks,
                traj_chunks=traj_chunks,
                t=t,
                context=context,
                seq_len=seq_len,
                is_tf=False,
                kv_caches=kv_caches,
                static_kv=use_static,
                temporal_stride=temporal_stride,
                chunk_position_ids=(
                    chunk_positions
                    if chunk_positions is not None
                    else torch.tensor(
                        [[active_past_count]], device=t.device, dtype=torch.float32
                    ).expand(B, -1) if use_static else None
                ),
                inference_context_cache=inference_context_cache,
                kv_workspaces=kv_workspaces,
                precomputed_joint_rope_freqs=precomputed_joint_rope_freqs,
            )

        # Unpatchify video output.
        # All chunks share the same spatial grid (checked inside the DiT), so
        # the per-sample grid_sizes derived from `x` apply to every chunk.
        grid_sizes = torch.stack([
            torch.tensor(
                self.dit_model.patch_embedding(u.unsqueeze(0)).shape[2:],
                dtype=torch.long,
            )
            for u in x
        ])

        if is_training and denoise_all_chunks:
            # video_out has shape [B, sum_k V_k, out_dim] where V_k == V_0 for
            # every chunk (same grid). Split along the token axis, unpatchify
            # each chunk, and stack back to [B, K, C_out, F, H, W].
            v_per_chunk = int(math.prod(grid_sizes[0].tolist()))
            K = video_out.size(1) // v_per_chunk
            assert video_out.size(1) == K * v_per_chunk, (
                f"video_out token count {video_out.size(1)} must be a multiple "
                f"of per-chunk token count {v_per_chunk}"
            )
            chunk_videos = []
            for ci in range(K):
                seg = video_out[:, ci * v_per_chunk:(ci + 1) * v_per_chunk]
                unpatched = self.dit_model.unpatchify(seg, grid_sizes)
                chunk_videos.append(torch.stack(unpatched))  # [B, C_out, F, H, W]
            video_out = torch.stack(chunk_videos, dim=1)     # [B, K, C_out, F, H, W]

            # Split traj_out [B, sum_k T_k, action_dim] into [B, K, T, action_dim].
            T_per_chunk = traj.shape[1]
            assert traj_out.size(1) == K * T_per_chunk, (
                f"traj_out token count {traj_out.size(1)} != K*T = {K * T_per_chunk}"
            )
            traj_out = traj_out.view(B, K, T_per_chunk, traj_out.size(-1))

            if return_kv_caches:
                return video_out, traj_out, updated_caches
            return video_out, traj_out

        video_out = self.dit_model.unpatchify(video_out, grid_sizes)
        video_out = torch.stack(video_out)

        if return_kv_caches:
            return video_out, traj_out, updated_caches
        return video_out, traj_out


    def save_pretrained(self, save_dir: str, **kwargs) -> None:
        """Save the full causal chunk DiT model."""
        self.dit_model.save_pretrained(save_dir, **kwargs)


    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.dit_model, "enable_gradient_checkpointing"):
            self.dit_model.enable_gradient_checkpointing()
