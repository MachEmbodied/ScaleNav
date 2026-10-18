"""
Attention utility functions supporting Flash Attention and Sage Attention.
"""

import os
import warnings

import torch

# Flash Attention 3
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

# Flash Attention 2
try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# Sage Attention
SAGE_ATTENTION_AVAILABLE = False
sageattn = None
try:
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"{major}.{minor}"
    sage_imports = {
        "8.0": "sageattention_sm80",
        "8.6": "sageattention_sm86",
        "8.9": "sageattention_sm89",
        "9.0": "sageattention_sm90",
    }
    if capability in sage_imports:
        module = __import__(sage_imports[capability], fromlist=['sageattn'])
        sageattn = module.sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major > 9:
        from sageattention_sm120 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        pass




def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor = None,
    k_lens: torch.Tensor = None,
    dropout_p: float = 0.,
    softmax_scale: float = None,
    q_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int = None,
) -> torch.Tensor:
    """
    Flash attention implementation.

    Args:
        q: Query tensor [B, Lq, Nq, C1].
        k: Key tensor [B, Lk, Nk, C1].
        v: Value tensor [B, Lk, Nk, C2].
        q_lens: Query lengths [B].
        k_lens: Key lengths [B].
        dropout_p: Dropout probability.
        softmax_scale: Scaling of QK^T before softmax.
        causal: Whether to apply causal mask.
        window_size: Sliding window attention size.
        deterministic: If True, slightly slower but deterministic.
        dtype: Data type for computation.

    Returns:
        Attention output tensor.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    query_mask = None
    if q_lens is not None:
        query_mask = torch.arange(lq, device=q.device)[None, :] < q_lens.to(q.device)[:, None]

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.full(
            (b,), lq, dtype=torch.int32, device=q.device
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.full(
            (b,), lk, dtype=torch.int32, device=k.device
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 is not available, using flash attention 2.')

    # Apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic
        )[0]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic
        )

    if query_mask is None:
        return x.unflatten(0, (b, lq)).type(out_dtype)
    padded = x.new_zeros((b, lq, *x.shape[1:]))
    padded[query_mask] = x
    return padded.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor = None,
    k_lens: torch.Tensor = None,
    dropout_p: float = 0.,
    softmax_scale: float = None,
    q_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version: int = None,
    attention_type: str = None,
    attn_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unified attention function supporting multiple backends.

    Args:
        q, k, v: Query, key, value tensors.
        q_lens, k_lens: Sequence lengths.
        dropout_p: Dropout probability.
        softmax_scale: Softmax scaling factor.
        causal: Apply causal mask.
        window_size: Sliding window size.
        dtype: Computation dtype.
        attention_type: "FLASH_ATTENTION" or "SAGE_ATTENTION".
        attn_mask: Optional attention mask.

    Returns:
        Attention output tensor.
    """
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION") if attention_type is None else attention_type

    # Sage attention doesn't support gradients
    if torch.is_grad_enabled() and attention_type == "SAGE_ATTENTION":
        attention_type = "FLASH_ATTENTION"

    if (attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE
            and q_lens is None and k_lens is None
            and softmax_scale is None and q_scale is None):
        out = sageattn(q, k, v, attn_mask=attn_mask, tensor_layout="NHD", is_causal=causal, dropout_p=dropout_p)

    elif (
        attention_type == "FLASH_ATTENTION"
        and q.device.type == "cuda"
        and attn_mask is None
        and q.size(-1) <= 256
        and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE)
    ):
        return flash_attention(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        batch, queries, _, _ = q.shape
        keys = k.shape[1]
        valid = torch.ones((batch, 1, 1, keys), dtype=torch.bool, device=q.device)
        if k_lens is not None:
            k_lens = k_lens.to(q.device)
            valid = valid & (torch.arange(keys, device=q.device)[None, None, None, :] < k_lens[:, None, None, None])
        if causal:
            valid = valid & (torch.arange(keys, device=q.device)[None, :] <= torch.arange(queries, device=q.device)[:, None])
        if window_size != (-1, -1):
            positions = torch.arange(keys, device=q.device)[None, :] - torch.arange(queries, device=q.device)[:, None]
            if window_size[0] >= 0:
                valid = valid & (positions >= -window_size[0])
            if window_size[1] >= 0:
                valid = valid & (positions <= window_size[1])
        if attn_mask is None:
            attn_mask = valid
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask & valid
        else:
            attn_mask = attn_mask.masked_fill(~valid, float('-inf'))
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=dropout_p, scale=softmax_scale)
        out = out.transpose(1, 2).contiguous()
        if q_lens is not None:
            q_lens = q_lens.to(q.device)
            query_valid = torch.arange(queries, device=q.device)[None, :] < q_lens[:, None]
            out = out * query_valid[:, :, None, None]

    return out
