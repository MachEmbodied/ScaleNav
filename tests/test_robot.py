import numpy as np
import pytest

from awm.robot import CameraStream, DashboardReporter, Go2Motion, body_velocity, run_episode


class Clock:
    value = 0.0

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class Motion:
    def __init__(self):
        self.events = []

    def move(self, *velocity):
        self.events.append(velocity)

    def stop(self):
        self.events.append('stop')


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def post(self, path, payload):
        self.requests.append((path, payload))
        if path == '/reset':
            return {}
        return next(self.responses)


def test_raw_displacement_to_go2_velocity():
    np.testing.assert_allclose(body_velocity([0.1, 0.02, 0.05]), [0.4, -0.08, -0.2])
    with pytest.raises(ValueError):
        body_velocity([float('nan'), 0, 0])


def test_chunk_execution_returns_intervening_observations_and_halts():
    client = Client([
        {'deltas': [[0.05, 0, 0]] * 9, 'dt': 0.25},
        {'deltas': [[0, 0, 0]] * 5, 'dt': 0.25, 'stop_after_chunk': True},
    ])
    motion, clock = Motion(), Clock()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    result = run_episode(client, lambda: rgb, motion, 'Go forward', sleep=clock.sleep, clock=clock.now)
    assert result == {'termination': 'model_stop', 'steps': 14}
    assert client.requests[1][1]['observations_b64'] == []
    assert len(client.requests[2][1]['observations_b64']) == 8
    assert clock.value == pytest.approx(3.5)
    assert motion.events[-1] == 'stop'
    assert motion.events.count('stop') == 4


def test_budget_does_not_execute_or_count_future_stop():
    client = Client([{'deltas': [[0.05, 0, 0]] * 9, 'dt': 0.25, 'stop_after_chunk': True}])
    motion, clock = Motion(), Clock()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    result = run_episode(client, lambda: rgb, motion, 'Go forward', max_steps=3,
                         sleep=clock.sleep, clock=clock.now)
    assert result == {'termination': 'budget', 'steps': 3}
    assert clock.value == pytest.approx(0.75)


def test_invalid_prediction_stops_without_executing():
    client = Client([{'deltas': [[0.05, 0, 0], [float('inf'), 0, 0]], 'dt': 0.25}])
    motion = Motion()
    with pytest.raises(ValueError):
        run_episode(client, lambda: np.zeros((4, 4, 3), dtype=np.uint8), motion, 'Go forward')
    assert all(event == 'stop' for event in motion.events)


def test_chunk_server_counts_every_issued_action_and_truncates_at_stop():
    from awm.habitat_extensions.awm_agent import AWMAgent, AWMAgentConfig
    from awm.scripts.awm_policy_server import PolicyState
    agent = AWMAgent.__new__(AWMAgent)
    agent.config = AWMAgentConfig()
    agent._step_count = 0
    agent.record_executed_delta(np.zeros(3))
    state = PolicyState.__new__(PolicyState)
    state.agent = agent
    deltas = state._issue_chunk_tail([[0, 0, 0]] * 9)
    assert len(deltas) == 5
    assert agent._step_count == 5
    assert agent.stop_requested


def test_go2_pose_uses_measurement_and_expires(monkeypatch):
    import threading
    from types import SimpleNamespace
    motion = Go2Motion.__new__(Go2Motion)
    motion._pose_lock = threading.Lock()
    motion._pose = None
    motion._pose_at = 0
    clock = Clock()
    monkeypatch.setattr('awm.robot.time.monotonic', clock.now)
    assert motion.pose() is None
    motion._on_state(SimpleNamespace(position=[2, 3, .4], imu_state=SimpleNamespace(rpy=[0, 0, 1.2])))
    assert motion.pose() == [2, 3, 1.2]
    clock.sleep(1.1)
    assert motion.pose() is None
    motion._on_state(SimpleNamespace(position=[float('nan'), 3, .4], imu_state=SimpleNamespace(rpy=[0, 0, 0])))
    assert motion.pose() is None


def test_reporter_sends_final_state_even_if_camera_fails():
    class TelemetryClient:
        def __init__(self):
            self.payloads = []

        def post(self, path, payload):
            assert path == '/telemetry'
            self.payloads.append(payload)

    def broken_camera():
        raise RuntimeError('camera disconnected')

    client = TelemetryClient()
    reporter = DashboardReporter(client, broken_camera)
    reporter.start(7)
    reporter.close('error', 3)
    assert client.payloads[-1] == {'run_id': 7, 'state': 'error', 'steps': 3, 'pose': None}


def test_reporter_records_completed_steps_after_motion_stops():
    class Reporter:
        def __init__(self):
            self.events = []

        def start(self, run_id):
            self.events.append(('start', run_id))

        def update(self, state, steps):
            self.events.append((state, steps))

        def close(self, state, steps):
            assert motion.events[-1] == 'stop'
            self.events.append((state, steps))

    class RunClient(Client):
        def post(self, path, payload):
            return {'run_id': 2} if path == '/reset' else super().post(path, payload)

    client = RunClient([{'deltas': [[.1, 0, 0]] * 3, 'dt': .25}])
    motion, clock, reporter = Motion(), Clock(), Reporter()
    result = run_episode(client, lambda: np.zeros((4, 4, 3), dtype=np.uint8), motion,
                         'test', max_steps=2, sleep=clock.sleep, clock=clock.now, reporter=reporter)
    assert result['termination'] == 'budget'
    assert reporter.events[-1] == ('budget', 2)
    assert clock.value == pytest.approx(.5)


def test_background_camera_reads_rgb_and_rejects_stale_frames(tmp_path):
    import cv2
    video = tmp_path / 'camera.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 4, (32, 24))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((24, 32, 3), [0, 0, 220], dtype=np.uint8))
    writer.release()
    camera = CameraStream(str(video))
    try:
        rgb = camera.read_rgb()
        assert rgb.shape == (288, 512, 3)
        assert rgb[:, :, 0].mean() > 200 and rgb[:, :, 2].mean() < 10
        camera.done.set()
        camera.thread.join(timeout=2)
        camera.updated_at -= 2
        with pytest.raises(RuntimeError, match='stale'):
            camera.read_rgb()
    finally:
        camera.close()
