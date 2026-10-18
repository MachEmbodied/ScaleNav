from types import SimpleNamespace

import numpy as np
import pytest

from awm.habitat_extensions.awm_agent import AWMAgent, AWMAgentConfig
from awm.habitat_extensions.awm_habitat_gs_evaluator import AWMHabitatGSEvaluator
from awm.habitat_extensions.config import EvalConfig


def test_camera_offset_stays_horizontal_after_pitch_and_yaw(monkeypatch, tmp_path):
    import sys
    from scipy.spatial.transform import Rotation
    monkeypatch.setitem(sys.modules, 'magnum', SimpleNamespace(Vector3=lambda *xyz: np.array(xyz)))
    simulator = SimpleNamespace(pathfinder=SimpleNamespace(is_loaded=True))
    captured = {}
    def configure(scene, agents):
        captured['sensor'] = agents[0].sensor_specifications[0]
    evaluator = AWMHabitatGSEvaluator.__new__(AWMHabitatGSEvaluator)
    evaluator.config = EvalConfig()
    evaluator._habitat_sim = SimpleNamespace(
        SimulatorConfiguration=SimpleNamespace, CameraSensorSpec=SimpleNamespace,
        SensorType=SimpleNamespace(COLOR=1), AgentConfiguration=SimpleNamespace,
        Configuration=configure, Simulator=lambda config: simulator)
    evaluator.scene_dataset_config = tmp_path / 'scenes.json'
    evaluator._episode_scene_path = lambda episode: tmp_path / 'scene.gs.ply'
    for pitch in [-15, 1.066, 20]:
        evaluator._build_sim({'cam_forward_m': .35, 'cam_pitch_deg': pitch, 'hfov_deg': 82.6})
        sensor = captured['sensor']
        rotation = Rotation.from_euler('y', 40, degrees=True) * Rotation.from_euler('x', -pitch, degrees=True)
        actual = rotation.apply(sensor.position)
        expected = Rotation.from_euler('y', 40, degrees=True).apply([0, 0, -.35])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        assert sensor.hfov == 82.6
    evaluator._build_sim({})
    np.testing.assert_array_equal(captured['sensor'].position, [0, 0, 0])


@pytest.mark.parametrize('leading,budget,success,steps', [
    (0, 200, True, 18), (191, 200, True, 200),
    (192, 200, False, 200), (0, 17, False, 17),
])
def test_evaluator_counts_zero_actions_and_requires_stop_within_budget(leading, budget, success, steps):
    state = SimpleNamespace(position=np.array([0, 0.35, 0]),
                            rotation=SimpleNamespace(w=1, x=0, y=0, z=0))
    simulator = SimpleNamespace(
        get_agent=lambda index: SimpleNamespace(get_state=lambda: state),
        get_sensor_observations=lambda: {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)},
        pathfinder=SimpleNamespace(snap_point=lambda position: position),
        close=lambda: None,
    )
    agent = AWMAgent.__new__(AWMAgent)
    config = EvalConfig(max_steps=budget)
    agent.config = AWMAgentConfig(
        motion_stop_n_frames=config.motion_stop_n_frames,
        min_steps_before_stop=config.min_steps_before_stop,
        motion_stop_xy_threshold_m=config.motion_stop_xy_threshold_m,
        motion_stop_yaw_threshold_rad=config.motion_stop_yaw_threshold_rad,
    )
    agent._step_count = 0
    agent.reset = lambda instruction: None
    def act(observation):
        delta = [0.02, 0, 0] if agent._step_count < leading else [0, 0, 0]
        agent.record_executed_delta(np.asarray(delta, dtype=np.float32))
        return delta
    agent.act_continuous = act
    evaluator = AWMHabitatGSEvaluator.__new__(AWMHabitatGSEvaluator)
    evaluator.config = config
    evaluator.agent = agent
    evaluator._build_sim = lambda episode: simulator
    evaluator._set_agent_state = lambda sim, position, quat: setattr(state, 'position', position)
    evaluator._reference_distance = lambda *args: 2.0
    evaluator._ndtw = lambda *args: None
    episode = {
        'scene_id': 'test', 'episode_id': 'test', 'instruction': 'Go to the target.',
        'start_xyz': [0, 0.35, 0], 'start_quat_wxyz': [1, 0, 0, 0],
        'goal_xyz': [0, 0, -2], 'cam_height_m': 0.35,
        'action_convention': 'scan2d_body_forward_right_clockwise',
        'target_contract': {'semantic_target_footprint_world_xz':
                            [[-1, 1], [1, 1], [1, -6], [-1, -6]]},
    }
    result = evaluator._run_episode(episode)
    assert result['steps_taken'] == steps
    assert bool(result['success']) is success
    assert result['stopped_via_model'] is success
    if leading == 0:
        assert result['spl'] == float(success)


@pytest.mark.parametrize('ending', ['before_action', 'after_action', 'budget'])
def test_visualization_keeps_initial_and_terminal_observations(ending):
    state = SimpleNamespace(position=np.array([0., .35, 0.]),
                            rotation=SimpleNamespace(w=1, x=0, y=0, z=0))
    buffer = np.zeros((24, 32, 4), dtype=np.uint8)
    def observe():
        buffer[..., 0] = round(-state.position[2] * 100)
        return {'rgb': buffer}
    simulator = SimpleNamespace(
        get_agent=lambda index: SimpleNamespace(get_state=lambda: state),
        get_sensor_observations=observe,
        pathfinder=SimpleNamespace(snap_point=lambda position: position), close=lambda: None)
    agent = SimpleNamespace(reset=lambda instruction: None, stop_requested=False, calls=0)
    def act(obs):
        agent.calls += 1
        if ending == 'before_action' and agent.calls == 2:
            return None
        agent.stop_requested = ending == 'after_action'
        return [.1, 0, 0]
    agent.act_continuous = act
    evaluator = AWMHabitatGSEvaluator.__new__(AWMHabitatGSEvaluator)
    evaluator.config = EvalConfig(max_steps=2, visualize=True)
    evaluator.agent = agent
    evaluator._build_sim = lambda episode: simulator
    evaluator._set_agent_state = lambda sim, position, quat: setattr(state, 'position', position)
    evaluator._reference_distance = lambda *args: 2.
    recorded = {}
    def record(**kwargs):
        recorded.update(kwargs)
        return 'rollout.mp4'
    evaluator._write_rollout_video = record
    episode = {
        'scene_id': 'scene', 'episode_id': 'episode', 'instruction': 'Approach the desk.',
        'start_xyz': [0, .35, 0], 'start_quat_wxyz': [1, 0, 0, 0], 'goal_xyz': [0, 0, -2],
        'action_convention': 'scan2d_body_forward_right_clockwise',
        'target_contract': {'semantic_target_footprint_world_xz': [[-1, -2], [1, -2], [1, -3], [-1, -3]]},
    }
    result = evaluator._run_episode(episode)
    assert result['video_path'] == 'rollout.mp4'
    assert len(recorded['frames']) == len(result['executed_positions_world']) == result['steps_taken'] + 1
    assert recorded['frames'][0][0, 0, 0] == 0
    assert recorded['frames'][-1][0, 0, 0] == round(-state.position[2] * 100)
    assert not np.shares_memory(recorded['frames'][0], buffer)
