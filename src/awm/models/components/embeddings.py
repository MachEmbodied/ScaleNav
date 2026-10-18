"""
Embedding and positional encoding modules for Wan Transformer.
"""

from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        dim: Embedding dimension (must be even).
        position: Position indices tensor.

    Returns:
        Sinusoidal embeddings of shape [len(position), dim].
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # Construct the fixed frequency indices directly on the input device. This
    # is numerically identical to a CPU arange followed by ``to(position)`` and
    # avoids an uncapturable host-to-device copy inside CUDA Graph replay.
    frequency_indices = torch.arange(
        half, device=position.device, dtype=position.dtype
    )
    sinusoid = torch.outer(
        position, torch.pow(10000, -frequency_indices.div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """
    Compute rotary position embedding parameters.

    Args:
        max_seq_len: Maximum sequence length.
        dim: Embedding dimension (must be even).
        theta: Base frequency.

    Returns:
        Complex frequency tensor of shape [max_seq_len, dim // 2].
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    temporal_stride: int = 4,
    action_temporal_scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor [B, Seq_Len, Num_Heads, Head_Dim].
        grid_sizes: Grid dimensions [B, 3] -> (frames, height, width).
        freqs: Frequency tensor [Max_Len, Head_Dim // 2].
        temporal_stride: VAE temporal compression ratio.
        action_temporal_scale: Scale factor for action token temporal indices.
            DreamZero uses ~10x to give action tokens an extended temporal range.
            Default 1.0 preserves original behavior.

    Returns:
        Position-encoded tensor.
    """
    n, c = x.size(2), x.size(3) // 2

    # Split frequencies for temporal, height, width
    c_w = c // 3
    c_h = c // 3
    c_t = c - 2 * c_w

    freqs_split = freqs.split([c_t, c_h, c_w], dim=1)
    freqs_t, freqs_h, freqs_w = freqs_split

    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        video_len = f * h * w

        # Video tokens (latent space)
        x_video = torch.view_as_complex(x[i, :video_len].to(torch.float32).reshape(
            video_len, n, -1, 2))

        freqs_video = torch.cat([
            freqs_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(video_len, 1, -1)

        x_video_out = torch.view_as_real(x_video * freqs_video).flatten(2)

        # Action tokens (raw frame space)
        # DreamZero-style: action tokens use an extended temporal frequency
        # range (action_temporal_scale * video range) to capture longer
        # time horizons than the video modality.
        x_action = x[i, video_len:]
        action_len = x_action.size(0)

        if action_len > 0:
            x_action_complex = torch.view_as_complex(x_action.to(torch.float32).reshape(
                action_len, n, -1, 2))

            # Align action time indices to video latent space,
            # then scale to the extended temporal range for actions
            raw_t_indices = torch.arange(action_len, device=x.device, dtype=torch.float32)
            aligned_t_indices = (raw_t_indices / temporal_stride).long()
            aligned_t_indices = torch.clamp(aligned_t_indices, max=f-1)

            # Scale temporal indices for extended range (DreamZero uses ~10x)
            scaled_t_indices = (aligned_t_indices.float() * action_temporal_scale).long()
            max_t_idx = freqs_t.shape[0] - 1
            scaled_t_indices = torch.clamp(scaled_t_indices, max=max_t_idx)

            f_action_t = freqs_t[scaled_t_indices]
            f_action_spatial = torch.ones(
                (action_len, c_h + c_w), device=x.device, dtype=freqs.dtype)

            freqs_action = torch.cat([f_action_t, f_action_spatial], dim=1).reshape(action_len, 1, -1)
            x_action_out = torch.view_as_real(x_action_complex * freqs_action).flatten(2)
        else:
            x_action_out = x_action

        output.append(torch.cat([x_video_out, x_action_out], dim=0))

    return torch.stack(output).to(x.dtype)


def rope_apply_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    temporal_stride: int = 4,
    action_temporal_scale: float = 1.0,
    precomputed_freqs: Optional[torch.Tensor] = None,
) -> tuple:
    """Apply rope to both query and key tensors."""
    if precomputed_freqs is not None:
        return (
            rope_apply_precomputed(q, precomputed_freqs),
            rope_apply_precomputed(k, precomputed_freqs),
        )
    q = rope_apply(q, grid_sizes, freqs, temporal_stride, action_temporal_scale)
    k = rope_apply(k, grid_sizes, freqs, temporal_stride, action_temporal_scale)
    return q, k


@amp.autocast(enabled=False)
@torch.compiler.disable()
def build_rope_freqs(
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    action_len: int,
    device: torch.device,
    temporal_stride: int = 4,
    action_temporal_scale: float = 1.0,
) -> torch.Tensor:
    """Build the standard joint video/action RoPE multiplier once.

    The returned tensor has the same token ordering as :func:`rope_apply`:
    dense video patches followed by raw-rate action tokens. It is independent
    of hidden states and denoising timestep, so a fixed-shape inference loop can
    reuse it in every layer and every denoising step without changing results.
    """
    c = freqs.size(1)
    c_w = c // 3
    c_h = c // 3
    c_t = c - 2 * c_w
    freqs_t, freqs_h, freqs_w = freqs.split([c_t, c_h, c_w], dim=1)

    raw_t_indices = torch.arange(
        action_len, device=device, dtype=torch.float32
    )
    aligned_t_indices = (raw_t_indices / temporal_stride).long()
    action_spatial = torch.ones(
        (action_len, c_h + c_w), device=device, dtype=freqs.dtype
    )

    output = []
    for frames, height, width in grid_sizes.tolist():
        video_freqs = torch.cat([
            freqs_t[:frames].view(frames, 1, 1, -1).expand(
                frames, height, width, -1
            ),
            freqs_h[:height].view(1, height, 1, -1).expand(
                frames, height, width, -1
            ),
            freqs_w[:width].view(1, 1, width, -1).expand(
                frames, height, width, -1
            ),
        ], dim=-1).reshape(frames * height * width, 1, -1)

        sample_indices = torch.clamp(aligned_t_indices, max=frames - 1)
        sample_indices = (
            sample_indices.float() * action_temporal_scale
        ).long()
        sample_indices = torch.clamp(
            sample_indices, max=freqs_t.shape[0] - 1
        )
        if action_len:
            action_freqs = torch.cat([
                freqs_t[sample_indices], action_spatial
            ], dim=1).reshape(action_len, 1, -1)
        else:
            # ``reshape(0, 1, -1)`` is ambiguous in PyTorch. Preserve the
            # expected multimodal RoPE width explicitly for video-only runs.
            action_freqs = video_freqs.new_empty((0, 1, c))
        output.append(torch.cat([video_freqs, action_freqs], dim=0))
    return torch.stack(output)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply_precomputed(
    x: torch.Tensor,
    token_freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply a multiplier returned by :func:`build_rope_freqs`."""
    if x.size(0) != token_freqs.size(0) or x.size(1) != token_freqs.size(1):
        raise ValueError(
            f"Joint RoPE shape mismatch: x={tuple(x.shape)}, "
            f"freqs={tuple(token_freqs.shape)}"
        )
    x_complex = torch.view_as_complex(
        x.to(torch.float32).reshape(x.size(0), x.size(1), x.size(2), -1, 2)
    )
    return torch.view_as_real(x_complex * token_freqs).flatten(3).to(x.dtype)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply_multimodal(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    chunk_boundaries: list,
    chunk_layouts: list,
    chunk_time_offsets: list,
    temporal_stride: int = 4,
) -> torch.Tensor:
    """Apply 3D RoPE to video and continuous-time RoPE to memory/actions.

    ``chunk_layouts`` contains ``(video_len, special_len, action_len,
    anchor_len)`` tuples for token sequences laid out as
    ``[video | special | action]``.  Special tokens are time-major and are
    distributed evenly over the latent video frames.  Unlike ``rope_apply``,
    action positions remain fractional, so 4 Hz actions do not collapse into
    groups with identical temporal positions.
    """
    num_offset_chunks = (
        chunk_time_offsets.shape[1]
        if torch.is_tensor(chunk_time_offsets) else len(chunk_time_offsets)
    )
    if not (
        len(chunk_boundaries) == len(chunk_layouts) == num_offset_chunks
    ):
        raise ValueError("RoPE chunk metadata lengths must match")

    n, c = x.size(2), x.size(3) // 2
    c_w = c // 3
    c_h = c // 3
    c_t = c - 2 * c_w
    freqs_t, freqs_h, freqs_w = freqs.split([c_t, c_h, c_w], dim=1)

    # freqs_t[1] == exp(i * omega).  omega is in [0, 1], so angle() recovers it
    # without phase ambiguity and lets us evaluate RoPE at fractional/global
    # positions beyond the integer lookup table.
    temporal_omega = torch.angle(freqs_t[1]).to(device=x.device, dtype=torch.float32)

    def temporal_freq(position: torch.Tensor) -> torch.Tensor:
        phase = position.to(torch.float32).unsqueeze(-1) * temporal_omega.unsqueeze(0)
        return torch.polar(torch.ones_like(phase), phase)

    outputs = []
    for bi, (_, h, w) in enumerate(grid_sizes.tolist()):
        chunk_outputs = []
        for chunk_idx, ((start, end), layout) in enumerate(zip(
            chunk_boundaries, chunk_layouts
        )):
            if torch.is_tensor(chunk_time_offsets):
                # Keep the position on device so continuous-time RoPE remains
                # CUDA-Graph capturable as the sliding-window offset changes.
                time_offset = chunk_time_offsets[bi, chunk_idx].to(
                    device=x.device, dtype=torch.float32
                )
            else:
                time_offset = float(chunk_time_offsets[chunk_idx])
            video_len, special_len, action_len, _ = layout
            chunk = x[bi, start:end]
            if chunk.size(0) != video_len + special_len + action_len:
                raise ValueError(
                    f"Chunk token length {chunk.size(0)} does not match layout {layout}"
                )
            if video_len % (h * w) != 0:
                raise ValueError("Video token count must be divisible by H*W")
            f = video_len // (h * w)

            video = torch.view_as_complex(
                chunk[:video_len].to(torch.float32).reshape(video_len, n, -1, 2)
            )
            video_t = temporal_freq(
                torch.arange(f, device=x.device, dtype=torch.float32) + time_offset
            )
            video_freq = torch.cat([
                video_t.view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(video_len, 1, -1)
            video_out = torch.view_as_real(video * video_freq).flatten(2)

            special = chunk[video_len:video_len + special_len]
            if special_len:
                if special_len % f != 0:
                    raise ValueError(
                        f"special_len={special_len} must be divisible by latent frames={f}"
                    )
                slots_per_frame = special_len // f
                special_pos = (
                    torch.arange(f, device=x.device, dtype=torch.float32)
                    .repeat_interleave(slots_per_frame)
                    + time_offset
                )
                special_complex = torch.view_as_complex(
                    special.to(torch.float32).reshape(special_len, n, -1, 2)
                )
                special_freq = torch.cat([
                    temporal_freq(special_pos),
                    torch.ones(
                        special_len, c_h + c_w, device=x.device, dtype=freqs.dtype
                    ),
                ], dim=-1).reshape(special_len, 1, -1)
                special_out = torch.view_as_real(
                    special_complex * special_freq
                ).flatten(2)
            else:
                special_out = special

            action = chunk[video_len + special_len:]
            if action_len:
                action_pos = (
                    torch.arange(action_len, device=x.device, dtype=torch.float32)
                    / float(temporal_stride)
                    + time_offset
                )
                action_complex = torch.view_as_complex(
                    action.to(torch.float32).reshape(action_len, n, -1, 2)
                )
                action_freq = torch.cat([
                    temporal_freq(action_pos),
                    torch.ones(
                        action_len, c_h + c_w, device=x.device, dtype=freqs.dtype
                    ),
                ], dim=-1).reshape(action_len, 1, -1)
                action_out = torch.view_as_real(action_complex * action_freq).flatten(2)
            else:
                action_out = action

            chunk_outputs.append(torch.cat([video_out, special_out, action_out], dim=0))
        outputs.append(torch.cat(chunk_outputs, dim=0))

    return torch.stack(outputs).to(x.dtype)


def rope_apply_multimodal_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    chunk_boundaries: list,
    chunk_layouts: list,
    chunk_time_offsets: list,
    temporal_stride: int = 4,
) -> tuple:
    """Apply explicit multimodal RoPE to query and key tensors."""
    kwargs = dict(
        grid_sizes=grid_sizes,
        freqs=freqs,
        chunk_boundaries=chunk_boundaries,
        chunk_layouts=chunk_layouts,
        chunk_time_offsets=chunk_time_offsets,
        temporal_stride=temporal_stride,
    )
    return rope_apply_multimodal(q, **kwargs), rope_apply_multimodal(k, **kwargs)


def zero_initialize_layer(layer: nn.Module) -> None:
    """Initialize layer weights and biases to zero."""
    if isinstance(layer, nn.Linear):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Embedding):
        nn.init.zeros_(layer.weight)
