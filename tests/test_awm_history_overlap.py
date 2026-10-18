from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from awm.habitat_extensions.awm_agent import AWMAgent


class _LatentDist:
    def sample(self):
        return torch.zeros(1, 4, 1, 1, 1)


class _Encoded:
    latent_dist = _LatentDist()


class _FakeVAE:
    def encode(self, _video):
        return _Encoded()


def _history_agent(overlap: int) -> AWMAgent:
    agent = AWMAgent.__new__(AWMAgent)
    agent.config = SimpleNamespace(
        num_video_frames=3,
        chunk_overlap_frames=overlap,
        ar_num_past_chunks=2,
        use_special_memory=False,
        use_predicted_newest_past=False,
        scale_xy=1.0,
        scale_yaw=1.0,
    )
    agent._gt_frame_buf = deque(
        np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)
    )
    agent._past_video_latents = []
    agent._past_trajectories = []
    agent._past_chunk_position_ids = []
    agent._next_chunk_position_id = 0
    agent._vae = _FakeVAE()
    agent._device = torch.device("cpu")
    agent._dtype = torch.float32
    agent._traj_normalizer_mode = "symlog"
    agent._symlog_c = 0.01
    agent._traj_stats_path = None
    agent.last_predicted_traj = np.ones((3, 3), dtype=np.float32)
    return agent


def test_observed_history_does_not_repeat_boundary_for_overlap_zero():
    agent = _history_agent(overlap=0)
    agent._flush_gt_past_chunk()
    assert list(agent._gt_frame_buf) == []


def test_observed_history_retains_only_configured_boundary_frames():
    agent = _history_agent(overlap=1)
    agent._flush_gt_past_chunk()
    assert len(agent._gt_frame_buf) == 1
    np.testing.assert_array_equal(agent._gt_frame_buf[0], np.full((2, 2, 3), 2))




def test_continuous_history_cap_preserves_anchor_and_latest_chunk():
    agent = _history_agent(overlap=0)
    agent.config.use_continuous_history = True
    agent._past_video_latents = [torch.tensor(i) for i in range(3)]
    agent._past_trajectories = [torch.tensor(i) for i in range(3)]
    agent._past_chunk_position_ids = [0, 1, 2]

    agent._trim_causal_history()

    assert agent._past_chunk_position_ids == [0, 2]






def test_chunk_api_observed_history_keeps_all_nine_frames_in_order():
    import pytest
    from awm.habitat_extensions.awm_agent import AWMAgentConfig
    from awm.scripts.awm_policy_server import PolicyState

    agent = _history_agent(overlap=0)
    agent.config = AWMAgentConfig()
    agent._target_size = (2, 2)
    agent._gt_frame_buf.clear()
    agent._step_count = 0
    agent._postprocess_trajectory = lambda value: value
    agent._predict_causal_observed_chunk = lambda image: np.tile([0.1, 0, 0], (9, 1))
    encoded = []
    def encode(video):
        encoded.append(video.clone())
        return _Encoded()
    agent._vae.encode = encode
    state = PolicyState(agent, dt=0.25, crop_square=False)
    state.have_instruction = True
    state.archive_model_input = lambda *args, **kwargs: None
    state.archive_inference = lambda *args, **kwargs: None
    state._update_viz = lambda *args, **kwargs: None
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(19)]

    assert len(state.act_chunk(frames[0])[1]) == 9
    assert agent._step_count == 9
    with pytest.raises(ValueError, match='Expected 8'):
        state.act_chunk(frames[9])
    assert len(state.act_chunk(frames[9], observations=frames[1:9])[1]) == 9
    assert len(state.act_chunk(frames[18], observations=frames[10:18])[1]) == 9
    assert agent._step_count == 27
    assert len(encoded) == 2
    for index, tensor in enumerate(encoded):
        observed = ((tensor[0, 0, :, 0, 0] + 1) * 127.5).round().numpy()
        np.testing.assert_array_equal(observed, np.arange(index * 9, (index + 1) * 9))
    np.testing.assert_array_equal(agent._gt_frame_buf[0], frames[18])
    np.testing.assert_allclose(agent._past_trajectories[-1][0, :, 0], np.log1p(10), rtol=1e-6)
