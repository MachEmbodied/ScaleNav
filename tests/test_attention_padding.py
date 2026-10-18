import torch
import pytest
from types import SimpleNamespace

from awm.models.components.attention_utils import attention


def test_sdpa_ignores_padded_keys_and_zeroes_padded_queries():
    torch.manual_seed(5)
    q, k, v = torch.randn(2, 4, 2, 8), torch.randn(2, 6, 2, 8), torch.randn(2, 6, 2, 8)
    lengths = torch.tensor([2, 5])
    query_lengths = torch.tensor([3, 4])
    output = attention(q, k, v, k_lens=lengths, q_lens=query_lengths, attention_type='SDPA')
    for i, length in enumerate(lengths):
        reference = attention(q[i:i+1], k[i:i+1, :length], v[i:i+1, :length], attention_type='SDPA')
        torch.testing.assert_close(output[i, :query_lengths[i]], reference[0, :query_lengths[i]])
    assert output[0, 3].count_nonzero() == 0


def test_sdpa_honors_explicit_mask_and_scale():
    q, k, v = torch.randn(1, 3, 2, 8), torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[:, -1] = False
    result = attention(q, k, v, attn_mask=mask, q_scale=.5, softmax_scale=.25, attention_type='SDPA')
    expected = torch.nn.functional.scaled_dot_product_attention(
        (q * .5).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=.25).transpose(1, 2)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Flash packing uses CUDA tensors')
@pytest.mark.parametrize('version', [2, 3])
def test_flash_packed_queries_restore_padding_and_gradients(monkeypatch, version):
    from awm.models.components import attention_utils as backend

    def reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
        parts = []
        for i in range(len(cu_seqlens_q) - 1):
            qs = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].transpose(0, 1)
            ks = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1)
            vs = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1)
            parts.append(torch.nn.functional.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1))
        return torch.cat(parts)

    monkeypatch.setattr(backend, 'FLASH_ATTN_2_AVAILABLE', True)
    monkeypatch.setattr(backend, 'FLASH_ATTN_3_AVAILABLE', version == 3)
    monkeypatch.setattr(backend, 'flash_attn', SimpleNamespace(flash_attn_varlen_func=reference_varlen), raising=False)
    monkeypatch.setattr(backend, 'flash_attn_interface', SimpleNamespace(
        flash_attn_varlen_func=lambda **kwargs: (reference_varlen(**kwargs),)), raising=False)
    torch.manual_seed(9)
    q = torch.randn(2, 4, 2, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 6, 2, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    lengths = dict(q_lens=torch.tensor([2, 4]), k_lens=torch.tensor([3, 5]))
    actual = backend.flash_attention(q, k, v, **lengths, version=version)
    expected = attention(q, k, v, **lengths, attention_type='SDPA')
    torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
    assert actual[0, 2:].count_nonzero() == 0
    actual.sum().backward()
    assert q.grad[0, 2:].count_nonzero() == 0
    assert torch.isfinite(q.grad).all() and q.grad[0, :2].abs().sum() > 0
