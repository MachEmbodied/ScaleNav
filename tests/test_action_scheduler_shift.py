import inspect
from types import SimpleNamespace

import pytest

from awm.habitat_extensions.awm_agent import AWMAgent, AWMAgentConfig
from awm.scripts.inference import (
    clone_noise_scheduler_with_shift,
)


class _FakeScheduler:
    def __init__(self, shift: float = 5.0):
        self.config = SimpleNamespace(shift=shift)
        self.marker = []

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(shift=float(overrides.get("shift", config.shift)))


def test_action_shift_defaults_to_exact_scheduler_clone():
    video = _FakeScheduler(shift=5.0)
    video.marker.append("video")

    action = clone_noise_scheduler_with_shift(video)

    assert action is not video
    assert action.config.shift == 5.0
    assert action.marker == ["video"]


def test_action_shift_rebuilds_scheduler_without_mutating_video():
    video = _FakeScheduler(shift=5.0)

    action = clone_noise_scheduler_with_shift(video, 1.0)

    assert video.config.shift == 5.0
    assert action.config.shift == 1.0


@pytest.mark.parametrize("shift", [0.0, -1.0, float("inf"), float("nan")])
def test_action_shift_rejects_invalid_values(shift):
    with pytest.raises(ValueError, match="finite and positive"):
        clone_noise_scheduler_with_shift(_FakeScheduler(), shift)


def test_closed_loop_uses_modality_specific_timesteps_and_steps():
    source = inspect.getsource(AWMAgent._predict_causal_observed_chunk)
    assert "zip(" in source
    assert "video_scheduler.timesteps" in source
    assert "traj_scheduler.timesteps" in source
    assert "velocity, video_t, x_cond" in source
    assert "traj_pred, action_t, traj_noisy" in source
    assert "bool(has_current_anchor)" in source
