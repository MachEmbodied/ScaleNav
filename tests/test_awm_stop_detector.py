from types import SimpleNamespace

import numpy as np
import pytest
import torch

from awm.habitat_extensions.awm_agent import AWMAgent




def _trajectory(forward):
    result = np.zeros((len(forward), 3), dtype=np.float32)
    result[:, 0] = np.asarray(forward, dtype=np.float32)
    return result








def _continuous_agent(predictions, *, n_frames):
    agent = AWMAgent.__new__(AWMAgent)
    agent.config = SimpleNamespace(
        use_gt_past=False,
        policy_mode="causal_observed_replan",
        min_steps_before_stop=0,
        disable_stop=False,
        auto_stop_on_near_zero=True,
        use_motion_stop_detector=False,
        motion_stop_n_frames=n_frames,
        motion_stop_ignore_prefix_frames=0,
        motion_stop_xy_threshold_m=0.025,
        motion_stop_yaw_threshold_rad=0.025,
        stop_threshold=0.05,
        exec_horizon_frames=0,
        trim_chunk_overshoot=False,
    )
    agent._target_size = (4, 4)
    agent._device = torch.device("cpu")
    agent._dtype = torch.float32
    agent._step_count = 0
    agent._continuous_chunk = None
    agent._continuous_chunk_idx = 0
    agent._continuous_chunk_executable_len = 0
    agent._continuous_stop_pending = False
    agent._continuous_stop_pending_reason = None
    agent._continuous_stop_latched = False
    agent.last_stop_reason = None

    calls = []

    def predict(_):
        calls.append(True)
        return predictions[len(calls) - 1]

    agent._predict_causal_observed_chunk = predict
    agent._postprocess_trajectory = lambda trajectory: trajectory
    return agent, calls


def test_idle_tail_is_executed_before_latched_stop():
    predictions = [
        _trajectory([0.08, 0.07, 0.005, 0.004, 0.003]),
        _trajectory([0.09, 0.08, 0.07, 0.06, 0.05]),
    ]
    agent, calls = _continuous_agent(predictions, n_frames=3)
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}

    np.testing.assert_allclose(agent.act_continuous(observation), [0.08, 0.0, 0.0])
    np.testing.assert_allclose(agent.act_continuous(observation), [0.07, 0.0, 0.0])
    np.testing.assert_allclose(agent.act_continuous(observation), [0.005, 0.0, 0.0])
    np.testing.assert_allclose(agent.act_continuous(observation), [0.004, 0.0, 0.0])
    np.testing.assert_allclose(agent.act_continuous(observation), [0.003, 0.0, 0.0])

    assert agent.act_continuous(observation) is None
    assert agent.last_stop_reason == "near_zero_window"
    assert len(calls) == 1

    # STOP is absorbing until reset, so an OOD post-stop prediction cannot
    # restart the robot.
    assert agent.act_continuous(observation) is None
    assert len(calls) == 1


def test_nine_frame_window_crosses_prediction_chunk_boundary():
    predictions = [
        _trajectory([0.08] * 5 + [0.005] * 4),
        _trajectory([0.005] * 5 + [0.09] * 4),
    ]
    agent, calls = _continuous_agent(predictions, n_frames=9)
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}

    issued = []
    for _ in range(14):
        issued.append(agent.act_continuous(observation))

    assert len(calls) == 2
    assert all(delta is not None for delta in issued)
    np.testing.assert_allclose(
        np.asarray(issued[-9:])[:, 0],
        np.asarray([0.005] * 9),
    )
    assert agent.act_continuous(observation) is None
    assert agent.last_stop_reason == "near_zero_window"
    assert len(calls) == 2


def test_five_action_stop_crosses_chunk_boundary():
    agent, calls = _continuous_agent([
        _trajectory([0.08] * 7 + [0.005] * 2),
        _trajectory([0.005] * 3 + [0.08] * 6),
    ], n_frames=5)
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
    for _ in range(11):
        assert agent.act_continuous(observation) is not None
        assert not agent.stop_requested
    assert agent.act_continuous(observation) is not None
    assert agent.stop_requested
    assert agent.act_continuous(observation) is None
    assert len(calls) == 2


def test_stop_on_last_budgeted_action_does_not_require_extra_action():
    agent, calls = _continuous_agent([_trajectory([0.005] * 9)], n_frames=5)
    agent._step_count = 195
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
    for _ in range(5):
        assert agent.act_continuous(observation) is not None
    assert agent._step_count == 200
    assert agent.stop_requested
    assert len(calls) == 1


def test_rotation_interrupts_near_zero_stop_window():
    prediction = _trajectory([0.005] * 9)
    prediction[4, 2] = 0.04
    agent, _ = _continuous_agent([prediction], n_frames=5)
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
    for _ in range(9):
        assert agent.act_continuous(observation) is not None
    assert not agent.stop_requested


@pytest.mark.parametrize("config_name", [None, "eval_habitatgs.yaml", "eval_ovon.yaml"])
@pytest.mark.parametrize("leading,stop_step", [(0, 9), (13, 22), (17, 26)])
def test_simulation_stops_after_nine_issued_actions(config_name, leading, stop_step):
    from importlib.resources import files
    from awm.habitat_extensions.config import EvalConfig
    from awm.scripts.eval_habitat_gs import _load_eval_yaml

    cfg = EvalConfig() if config_name is None else SimpleNamespace(**_load_eval_yaml(
        str(files("awm").joinpath("configs", config_name))))
    trajectory = _trajectory([0.08] * leading + [0.005] * 27 + [0.08] * 9)
    predictions = [trajectory[start:start + 9] for start in range(0, len(trajectory), 9)]
    agent, calls = _continuous_agent(predictions, n_frames=cfg.motion_stop_n_frames)
    for name in ("min_steps_before_stop", "motion_stop_xy_threshold_m", "motion_stop_yaw_threshold_rad"):
        setattr(agent.config, name, getattr(cfg, name))
    observation = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
    for step in range(1, stop_step):
        assert agent.act_continuous(observation) is not None
        assert agent._step_count == step
        assert not agent.stop_requested
    np.testing.assert_allclose(agent.act_continuous(observation), [0.005, 0, 0])
    assert agent._step_count == stop_step
    assert agent.stop_requested
    calls_at_stop = len(calls)
    assert agent.act_continuous(observation) is None
    assert len(calls) == calls_at_stop
