from unittest.mock import patch

import pytest
import torch

import awm.models.causal_chunk_ar_model as causal_model_module
from awm.models.causal_chunk_ar_model import CausalChunkDiTModel, CausalChunkDiTWrapper


def _build_model(
    *,
    continuous: bool,
    overlap: int = 0,
    num_layers: int = 1,
    max_attn_chunks: int = 3,
):
    return CausalChunkDiTWrapper(CausalChunkDiTModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=24,
        ffn_dim=48,
        freq_dim=32,
        text_dim=16,
        out_dim=4,
        num_heads=3,
        num_layers=num_layers,
        action_dim=3,
        max_attn_chunks=max_attn_chunks,
        use_continuous_history=continuous,
        chunk_overlap_frames=overlap,
    ))


def _inputs():
    current = [torch.randn(4, 3, 4, 4)]
    trajectory = torch.randn(1, 9, 3)
    context = [torch.randn(3, 16)]
    timesteps = torch.full((1, 12 + 9), 500, dtype=torch.long)
    return current, trajectory, context, timesteps


def test_continuous_history_adds_no_parameters_and_defaults_off():
    torch.manual_seed(0)
    legacy = _build_model(continuous=False)
    continuous = _build_model(continuous=True)

    assert legacy.config.use_continuous_history is False
    assert continuous.config.use_continuous_history is True
    assert legacy.state_dict().keys() == continuous.state_dict().keys()
    assert {
        key: tuple(value.shape) for key, value in legacy.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in continuous.state_dict().items()
    }


def test_continuous_history_uses_raw_chunk_stride_for_global_rope():
    torch.manual_seed(1)
    model = _build_model(continuous=True, overlap=0).eval()
    current, trajectory, context, timesteps = _inputs()
    past_video = [torch.randn(1, 4, 3, 4, 4)]
    past_action = [torch.randn(1, 9, 3)]
    captured_offsets = []
    original = causal_model_module.rope_apply_multimodal_qk

    def capture(*args, **kwargs):
        captured_offsets.append(kwargs["chunk_time_offsets"].detach().cpu())
        return original(*args, **kwargs)

    with patch.object(
        causal_model_module, "rope_apply_multimodal_qk", side_effect=capture
    ):
        with torch.no_grad():
            model(
                x=current,
                traj=trajectory,
                t=timesteps,
                context=context,
                seq_len=12,
                past_video_latents=past_video,
                past_trajectories=past_action,
                past_chunk_position_ids=torch.tensor([[0]]),
                current_chunk_position_id=torch.tensor([1]),
            )

    assert captured_offsets
    # Nine raw frames with no overlap advance by 9 / VAE_stride(4) = 2.25.
    torch.testing.assert_close(
        captured_offsets[0], torch.tensor([[0.0, 2.25]]), atol=0, rtol=0
    )


def test_continuous_history_supports_action_free_video_pretraining():
    torch.manual_seed(11)
    model = _build_model(continuous=True, overlap=0).eval()
    current, _, context, _ = _inputs()
    trajectory = torch.empty(1, 0, 3)
    past_video = [torch.randn(1, 4, 3, 4, 4)]
    past_action = [torch.empty(1, 0, 3)]
    timesteps = torch.full((1, 12), 500, dtype=torch.long)
    captured_offsets = []
    original = causal_model_module.rope_apply_multimodal_qk

    def capture(*args, **kwargs):
        captured_offsets.append(kwargs["chunk_time_offsets"].detach().cpu())
        return original(*args, **kwargs)

    with patch.object(
        causal_model_module, "rope_apply_multimodal_qk", side_effect=capture
    ):
        with torch.no_grad():
            video, action = model(
                x=current,
                traj=trajectory,
                t=timesteps,
                context=context,
                seq_len=12,
                past_video_latents=past_video,
                past_trajectories=past_action,
                past_chunk_position_ids=torch.tensor([[0]]),
                current_chunk_position_id=torch.tensor([1]),
            )

    assert video.shape == (1, 4, 3, 4, 4)
    assert action.shape == (1, 0, 3)
    # With no actions, 3 latent frames still recover the 9-frame raw stride.
    torch.testing.assert_close(
        captured_offsets[0], torch.tensor([[0.0, 2.25]]), atol=0, rtol=0
    )


def test_local_rope_supports_action_free_video_pretraining():
    torch.manual_seed(12)
    model = _build_model(continuous=False, overlap=0).eval()
    current, _, context, _ = _inputs()
    trajectory = torch.empty(1, 0, 3)

    with torch.no_grad():
        video, action = model(
            x=current,
            traj=trajectory,
            t=torch.full((1, 12), 500, dtype=torch.long),
            context=context,
            seq_len=12,
            past_video_latents=[torch.randn(1, 4, 3, 4, 4)],
            past_trajectories=[torch.empty(1, 0, 3)],
            past_chunk_position_ids=torch.tensor([[0]]),
            current_chunk_position_id=torch.tensor([1]),
        )

    assert video.shape == (1, 4, 3, 4, 4)
    assert action.shape == (1, 0, 3)


def test_denoise_all_supports_action_free_video_pretraining():
    torch.manual_seed(13)
    model = _build_model(continuous=True, overlap=0).eval()
    current, _, context, _ = _inputs()
    empty_action = torch.empty(1, 0, 3)
    noisy_past = [torch.randn(1, 4, 3, 4, 4)]
    clean_past = [torch.randn(1, 4, 3, 4, 4)]

    with torch.no_grad():
        video, action = model(
            x=current,
            traj=empty_action,
            t=torch.full((1, 2 * 12), 500, dtype=torch.long),
            context=context,
            seq_len=12,
            past_video_latents=noisy_past,
            past_trajectories=[empty_action],
            past_chunk_position_ids=torch.tensor([[0]]),
            current_chunk_position_id=torch.tensor([1]),
            is_training=True,
            denoise_all_chunks=True,
            clean_x=[torch.randn_like(current[0])],
            clean_traj=empty_action,
            clean_past_video_latents=clean_past,
            clean_past_trajectories=[empty_action],
        )

    assert video.shape == (1, 2, 4, 3, 4, 4)
    assert action.shape == (1, 2, 0, 3)


def test_continuous_history_static_cache_matches_full_history():
    torch.manual_seed(2)
    model = _build_model(
        continuous=True, num_layers=2, max_attn_chunks=4
    ).eval()
    current, trajectory, context, timesteps = _inputs()
    past_video = [torch.randn(1, 4, 3, 4, 4) for _ in range(3)]
    past_action = [torch.randn(1, 9, 3) for _ in range(3)]
    past_positions = torch.tensor([[0, 7, 8]])
    current_position = torch.tensor([9])

    with torch.no_grad():
        full_video, full_action = model(
            x=current,
            traj=trajectory,
            t=timesteps,
            context=context,
            seq_len=12,
            past_video_latents=past_video,
            past_trajectories=past_action,
            past_chunk_position_ids=past_positions,
            current_chunk_position_id=current_position,
        )
        caches = model.prefill_past_kv(
            past_video,
            past_action,
            context,
            seq_len=12,
            past_chunk_position_ids=past_positions,
        )
        cached_video, cached_action = model(
            x=current,
            traj=trajectory,
            t=timesteps,
            context=context,
            seq_len=12,
            past_video_latents=past_video,
            past_trajectories=past_action,
            past_chunk_position_ids=past_positions,
            current_chunk_position_id=current_position,
            kv_caches=caches,
            static_kv_cache=True,
        )

    # Four spatial tokens from anchor chunk 0's first latent slice, plus two
    # complete recent chunks (12 video + 9 action tokens each).
    assert caches[0][0].shape[1] == 4 + 2 * (12 + 9)
    assert model.dit_model.special_tokens is None
    assert model.dit_model.anchor_embedding is None
    torch.testing.assert_close(cached_video, full_video, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(cached_action, full_action, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("max_attn_chunks", [-1, 4])
def test_continuous_history_incremental_kv_matches_full_prefill(max_attn_chunks):
    torch.manual_seed(22)
    model = _build_model(
        continuous=True, num_layers=2, max_attn_chunks=max_attn_chunks
    ).eval()
    context = [torch.randn(3, 16)]
    past_video = [torch.randn(1, 4, 3, 4, 4) for _ in range(6)]
    past_action = [torch.randn(1, 9, 3) for _ in range(6)]
    positions = torch.arange(6, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        reference = model.prefill_past_kv(
            past_video,
            past_action,
            context,
            seq_len=12,
            past_chunk_position_ids=positions,
        )
        incremental = None
        for index, (video, action) in enumerate(zip(past_video, past_action)):
            incremental = model.append_continuous_history_kv(
                incremental,
                video,
                action,
                context,
                seq_len=12,
                chunk_position_id=index,
                total_history_chunks=index + 1,
            )

    assert reference is not None
    assert incremental is not None
    for reference_kv, incremental_kv in zip(reference, incremental):
        torch.testing.assert_close(
            incremental_kv[0], reference_kv[0], atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            incremental_kv[1], reference_kv[1], atol=2e-5, rtol=2e-5
        )


def test_continuous_history_streaming_kv_drops_evicted_full_chunk():
    model = _build_model(continuous=True, max_attn_chunks=-1).eval()
    old_positions = [0, 1, 2, 3]
    retained_positions = [0, 2, 3]
    cached_k = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 24, 1, 1)
    cached_v = cached_k + 100

    retained = model.retain_continuous_history_kv(
        [(cached_k, cached_v)], old_positions, retained_positions
    )

    expected_indices = list(range(0, 6)) + list(range(12, 24))
    torch.testing.assert_close(
        retained[0][0].flatten(), cached_k.flatten()[expected_indices]
    )
    torch.testing.assert_close(
        retained[0][1].flatten(), cached_v.flatten()[expected_indices]
    )


def test_continuous_history_supports_denoise_all_training():
    torch.manual_seed(3)
    model = _build_model(continuous=True)
    current, trajectory, context, _ = _inputs()
    noisy_past = [torch.randn(1, 4, 3, 4, 4)]
    noisy_past_action = [torch.randn(1, 9, 3)]
    clean_current = [torch.randn_like(current[0])]
    clean_trajectory = torch.randn_like(trajectory)
    clean_past = [torch.randn_like(noisy_past[0])]
    clean_past_action = [torch.randn_like(noisy_past_action[0])]
    # Two noisy chunks, each with 12 video + 9 action tokens.
    timesteps = torch.full((1, 2 * (12 + 9)), 500, dtype=torch.long)

    video, action = model(
        x=current,
        traj=trajectory,
        t=timesteps,
        context=context,
        seq_len=12,
        past_video_latents=noisy_past,
        past_trajectories=noisy_past_action,
        past_chunk_position_ids=torch.tensor([[0]]),
        current_chunk_position_id=torch.tensor([1]),
        is_training=True,
        denoise_all_chunks=True,
        clean_x=clean_current,
        clean_traj=clean_trajectory,
        clean_past_video_latents=clean_past,
        clean_past_trajectories=clean_past_action,
    )

    assert video.shape == (1, 2, 4, 3, 4, 4)
    assert action.shape == (1, 2, 9, 3)


def test_continuous_history_teacher_forcing_backward_is_finite_with_mixed_padding():
    torch.manual_seed(4)
    model = _build_model(continuous=True, max_attn_chunks=4).train()
    batch_size = 2
    current = [torch.randn(4, 3, 4, 4) for _ in range(batch_size)]
    trajectory = torch.randn(batch_size, 9, 3)
    context = [torch.randn(3, 16) for _ in range(batch_size)]
    timesteps = torch.full((batch_size, 12 + 9), 500, dtype=torch.long)
    past_video = [
        torch.randn(batch_size, 4, 3, 4, 4) for _ in range(3)
    ]
    past_action = [torch.randn(batch_size, 9, 3) for _ in range(3)]

    video, action = model(
        x=current,
        traj=trajectory,
        t=timesteps,
        context=context,
        seq_len=12,
        past_video_latents=past_video,
        past_trajectories=past_action,
        past_chunk_mask=torch.tensor([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        past_chunk_position_ids=torch.tensor([
            [0, 19, 20],
            [-1, -1, 0],
        ]),
        current_chunk_position_id=torch.tensor([21, 1]),
        is_training=True,
    )
    loss = video.float().square().mean() + action.float().square().mean()
    loss.backward()

    assert torch.isfinite(video).all()
    assert torch.isfinite(action).all()
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_continuous_history_rejects_conflicting_memory_layout():
    with pytest.raises(ValueError, match="without special memory"):
        CausalChunkDiTModel(
            dim=24,
            ffn_dim=48,
            num_heads=3,
            num_layers=1,
            use_special_memory=True,
            use_continuous_history=True,
        )
