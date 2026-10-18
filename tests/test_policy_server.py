import base64
import io
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from awm.habitat_extensions.awm_agent import AWMAgentConfig
from awm.scripts.awm_policy_server import PolicyState, make_handler


class Agent:
    def __init__(self):
        self.config = AWMAgentConfig(image_height=24, image_width=32, decode_predicted_video=True)
        self._target_size = (24, 32)
        self.reset("")

    def reset(self, instruction):
        self._instruction = instruction
        self._step_count = 0
        self._continuous_chunk = None
        self._continuous_chunk_idx = 0
        self._continuous_chunk_executable_len = 0
        self._gt_frame_buf = []
        self.stop_requested = False

    def record_executed_delta(self, delta):
        self._step_count += 1

    def act_continuous(self, observation):
        if self._step_count >= 6:
            self.stop_requested = True
            return None
        if self._continuous_chunk is None or self._continuous_chunk_idx >= 3:
            self._continuous_chunk = np.array([[.1, .02, .03]] * 3)
            self._continuous_chunk_idx = 0
            self._continuous_chunk_executable_len = 3
            self.last_generated_video = np.stack([observation['rgb']] * 3)
        delta = self._continuous_chunk[self._continuous_chunk_idx]
        self._continuous_chunk_idx += 1
        self.record_executed_delta(delta)
        return delta


@pytest.fixture
def frame():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 150
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return image, buffer.getvalue()


@pytest.fixture
def server(tmp_path):
    state = PolicyState(Agent(), .25, crop_square=False, record_dir=tmp_path / 'records')
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_port}' + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=3) as response:
            raw = response.read()
            return json.loads(raw) if response.headers['Content-Type'] == 'application/json' else raw

    yield state, request
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_chunk_camera_prediction_history_and_stop_over_http(server, frame):
    state, request = server
    rgb, raw = frame
    encoded = base64.b64encode(raw).decode()
    request('/reset', {'instruction': 'Approach the chair.'})
    first = request('/act_chunk', {'rgb_b64': encoded})
    assert len(first['deltas']) == 3
    snapshot = request('/state.json')
    assert snapshot['step'] == 3 and snapshot['n_infer'] == 1
    assert snapshot['actual_path'] == []
    assert snapshot['predicted_path'][1] == pytest.approx([.1, -.02, -.03])
    assert snapshot['dt'] == .25 and snapshot['pred_video_count'] == 3
    version = snapshot['pred_video_seq']
    assert request(f'/pred_video/0.jpg?v={version}')[:2] == b'\xff\xd8'
    assert request('/latest.jpg')[:2] == b'\xff\xd8'
    with pytest.raises(urllib.error.HTTPError) as error:
        request('/act_chunk', {'rgb_b64': encoded})
    assert error.value.code == 400
    second = request('/act_chunk', {'rgb_b64': encoded, 'observations_b64': [encoded] * 2})
    assert len(second['deltas']) == 3 and len(state.agent._gt_frame_buf) == 2
    with pytest.raises(urllib.error.HTTPError) as error:
        request(f'/pred_video/0.jpg?v={version}')
    assert error.value.code == 404
    stop = request('/act_chunk', {'rgb_b64': encoded, 'observations_b64': [encoded] * 2})
    assert stop['stop'] and stop['deltas'] is None
    snapshot = request('/state.json')
    assert snapshot['stop'] and snapshot['pred_video_count'] == 0
    assert snapshot['predicted_path'] == []
    directory = state.recorder.directory
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    inputs = [event for event in events if event['kind'] == 'input']
    assert len(inputs) == 7
    assert (directory / inputs[0]['image']).read_bytes() == raw
    prediction = next(event for event in events if event.get('prediction'))
    np.testing.assert_array_equal(np.load(directory / prediction['prediction']), np.stack([rgb] * 3))


def test_measured_pose_coordinates_stale_runs_and_reset(server, frame):
    state, request = server
    run = request('/reset', {'instruction': 'first'})['run_id']
    request('/telemetry', {'run_id': run, 'pose': [10, 20, np.pi / 2], 'state': 'waiting'})
    request('/telemetry', {'run_id': run, 'pose': [10, 21, np.pi], 'state': 'executing', 'steps': 2})
    snapshot = request('/state.json')
    np.testing.assert_allclose(snapshot['actual_path'], [[0, 0, 0], [1, 0, np.pi / 2]], atol=1e-12)
    assert snapshot['executed_steps'] == 2
    request('/telemetry', {'run_id': run, 'state': 'model_stop', 'steps': 2})
    assert request('/state.json')['pose_at'] == snapshot['pose_at']
    for invalid in ([1, 2], [1, 2, float('nan')]):
        with pytest.raises(urllib.error.HTTPError) as error:
            request('/telemetry', {'run_id': run, 'pose': invalid})
        assert error.value.code == 400
    old_directory = state.recorder.directory
    request('/reset', {'instruction': 'second'})
    snapshot = request('/state.json')
    assert snapshot['actual_path'] == snapshot['predicted_path'] == []
    assert snapshot['pose_at'] is None and not snapshot['has_frame']
    assert snapshot['step'] == snapshot['n_infer'] == 0
    assert snapshot['executed_steps'] is None and snapshot['last_infer_s'] is None
    assert state.recorder.directory != old_directory
    with pytest.raises(urllib.error.HTTPError) as error:
        request('/telemetry', {'run_id': run, 'pose': [0, 0, 0]})
    assert error.value.code == 400
    assert request('/state.json')['actual_path'] == []


def test_visualization_remains_responsive_during_inference(server, frame):
    state, request = server
    entered, release = threading.Event(), threading.Event()
    original = state.agent.act_continuous

    def slow(observation):
        entered.set()
        assert release.wait(3)
        return original(observation)

    state.agent.act_continuous = slow
    run = request('/reset', {'instruction': 'test'})['run_id']
    errors = []

    def act():
        try:
            request('/act_chunk', {'rgb_b64': base64.b64encode(frame[1]).decode()})
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=act)
    thread.start()
    try:
        assert entered.wait(2)
        assert request('/state.json')['inferencing']
        request('/telemetry', {'run_id': run, 'pose': [0, 0, 0], 'state': 'waiting'})
        assert len(request('/state.json')['actual_path']) == 1
    finally:
        release.set()
        thread.join()
    assert not errors
    assert not request('/state.json')['inferencing']


def test_removed_dashboard_endpoints_return_404(server):
    _, request = server
    for path, body in [('/nav_map.json', None), ('/nav_map', {}), ('/status', {}),
                       ('/recording.json', None), ('/recording/0.jpg', None),
                       ('/input_events.json', None), ('/inference_events.json', None)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(path, body)
        assert error.value.code == 404


def test_disabled_recording_retains_no_input_queue(frame):
    state = PolicyState(Agent(), .25)
    state.reset('test')
    for step in range(5000):
        state.archive_model_input(frame[0], frame[1], source='act', action_step=step)
    assert state.recorder.directory is None
    assert not hasattr(state, '_input_events')


@pytest.mark.parametrize('path', ['/state.json', '/'])
def test_client_disconnect_during_response_keeps_server_healthy(path):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []
    state = PolicyState(Agent(), .25, crop_square=False)

    class Handler(make_handler(state)):
        def end_headers(self):
            entered.set()
            assert release.wait(5)
            super().end_headers()

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            import sys
            errors.append(sys.exc_info()[1])

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                finished.set()

    httpd = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    client = socket.create_connection(httpd.server_address, timeout=3)
    try:
        client.sendall(f'GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n'.encode())
        assert entered.wait(3)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client.close()
        release.set()
        assert finished.wait(3)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{httpd.server_port}/health', timeout=3) as response:
            assert json.load(response)['ok']
        assert errors == []
    finally:
        release.set()
        client.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_per_action_mode_only_updates_prediction_on_new_chunks(frame):
    state = PolicyState(Agent(), .25, crop_square=False)
    state.reset('test')
    for step in range(6):
        phase, delta, chunk, _ = state.act(frame[0], raw_input=frame[1])
        assert phase == 'move' and delta is not None
        assert (chunk is not None) == (step % 3 == 0)
        assert state.viz_state()['n_infer'] == step // 3 + 1
        assert state.viz_state()['step'] == step + 1
    assert state.act(frame[0], raw_input=frame[1])[0] == 'stop'
