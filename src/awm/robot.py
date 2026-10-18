"""Synchronous velocity execution of ScaleNav action chunks."""
import argparse
import base64
import io
import json
import threading
import time
import urllib.request

import numpy as np
from PIL import Image


def body_velocity(delta, dt=0.25):
    action = np.asarray(delta, dtype=float)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("Action must contain three finite displacements")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Action duration must be finite and positive")
    return action * np.asarray([1.0, -1.0, -1.0]) / dt


def encode_frame(rgb):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format="JPEG", quality=95)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class PolicyClient:
    def __init__(self, url, timeout=60):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(self, path, payload):
        request = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)


class Go2Motion:
    def __init__(self, interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        ChannelFactoryInitialize(0, interface)
        self.client = SportClient()
        self.client.SetTimeout(1.0)
        self.client.Init()
        self._pose_lock = threading.Lock()
        self._pose = None
        self._pose_at = 0.0
        self.subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.subscriber.Init(self._on_state, 1)

    def _on_state(self, message):
        pose = [float(message.position[0]), float(message.position[1]), float(message.imu_state.rpy[2])]
        if np.isfinite(pose).all():
            with self._pose_lock:
                self._pose, self._pose_at = pose, time.monotonic()

    def pose(self):
        with self._pose_lock:
            return list(self._pose) if self._pose is not None and time.monotonic() - self._pose_at < 1 else None

    def close(self):
        self.subscriber.Close()

    def move(self, vx, vy, yaw_rate):
        result = self.client.Move(float(vx), float(vy), float(yaw_rate))
        if result not in (None, 0):
            raise RuntimeError(f"Go2 Move failed: {result}")

    def stop(self):
        self.client.StopMove()


class DryRunMotion:
    def move(self, vx, vy, yaw_rate):
        pass

    def stop(self):
        pass


class CameraStream:
    def __init__(self, source):
        import cv2
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Cannot open the rectified RGB stream")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.frame = None
        self.updated_at = 0.0
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            self.close()
            raise RuntimeError("Camera did not provide a frame")

    def _read(self):
        import cv2
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        period = 1 / fps if np.isfinite(fps) and 1 <= fps <= 120 else 1 / 30
        try:
            while not self.done.is_set():
                ok, bgr = self.capture.read()
                if not ok:
                    raise RuntimeError("Camera read failed")
                rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288)), cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame, self.updated_at = rgb, time.monotonic()
                self.ready.set()
                self.done.wait(period)
        except Exception as error:
            self.error = error
            self.ready.set()

    def read_rgb(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Camera stream stopped") from self.error
            if self.frame is None or time.monotonic() - self.updated_at > 1:
                raise RuntimeError("Camera frame is stale")
            return self.frame.copy()

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.capture.release()


class DashboardReporter:
    def __init__(self, client, read_rgb, read_pose=lambda: None):
        self.client, self.read_rgb, self.read_pose = client, read_rgb, read_pose
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.status = {"state": "waiting", "steps": 0}
        self.thread = None

    def update(self, state, steps):
        with self.lock:
            self.status = {"state": state, "steps": steps}

    def publish(self):
        with self.lock:
            payload = {"run_id": self.run_id, **self.status}
        try:
            payload["pose"] = self.read_pose()
        except Exception:
            payload["pose"] = None
        try:
            payload["rgb_b64"] = encode_frame(self.read_rgb())
        except Exception:
            pass
        try:
            self.client.post("/telemetry", payload)
        except Exception:
            pass

    def start(self, run_id):
        self.run_id = run_id
        self.publish()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.done.wait(.25):
            self.publish()

    def close(self, state, steps):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if not self.thread.is_alive():
                self.update(state, steps)
                self.publish()


def run_episode(client, read_rgb, motion, instruction, *, max_steps=200,
                sleep=time.sleep, clock=time.monotonic, reporter=None):
    if max_steps < 1 or max_steps > 200:
        raise ValueError("Execution budget must be between 1 and 200 actions")
    steps = 0
    observations = []
    termination = "error"
    try:
        motion.stop()
        reset = client.post("/reset", {"instruction": instruction})
        if reporter is not None:
            reporter.start(reset["run_id"])
        while steps < max_steps:
            if reporter is not None:
                reporter.update("waiting", steps)
            encoded_observations = [encode_frame(frame) for frame in observations]
            frame = read_rgb()
            response = client.post("/act_chunk", {
                "rgb_b64": encode_frame(frame), "observations_b64": encoded_observations,
            })
            if response.get("stop"):
                termination = "model_stop"
                return {"termination": "model_stop", "steps": steps}
            if response.get("confirm_stop"):
                raise ValueError("Disable confirmation for synchronous chunk execution")
            dt = float(response.get("dt", 0))
            if not np.isclose(dt, 0.25, rtol=0, atol=1e-6):
                raise ValueError("ScaleNav actions represent 0.25 seconds")
            deltas = response.get("deltas")
            if not isinstance(deltas, list) or not 1 <= len(deltas) <= 9:
                raise ValueError("Expected between one and nine actions")
            velocities = [body_velocity(delta, dt) for delta in deltas]
            if reporter is not None:
                reporter.update("executing", steps)
            observations = []
            executed = 0
            for index, velocity in enumerate(velocities):
                if steps >= max_steps:
                    break
                if index:
                    observations.append(np.asarray(read_rgb()).copy())
                deadline = clock() + dt
                while True:
                    motion.move(*velocity)
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    sleep(min(0.05, remaining))
                steps += 1
                executed += 1
                if reporter is not None:
                    reporter.update("executing", steps)
            motion.stop()
            if response.get("stop_after_chunk") and executed == len(deltas):
                termination = "model_stop"
                return {"termination": "model_stop", "steps": steps}
        termination = "budget"
        return {"termination": "budget", "steps": steps}
    finally:
        motion.stop()
        if reporter is not None:
            reporter.close(termination, steps)


def main():
    parser = argparse.ArgumentParser(description="ScaleNav synchronous velocity client")
    parser.add_argument("--server", default="http://127.0.0.1:8620")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--backend", choices=["dry-run", "go2"], default="dry-run")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--camera", default="0", help="Rectified RGB camera or video path")
    parser.add_argument("--image", help="Repeat an image for a hardware-free smoke test")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    if args.image and args.backend != "dry-run":
        parser.error("Repeated-image input is only supported by the dry-run backend")
    capture = None
    if args.image:
        rgb = np.asarray(Image.open(args.image).convert("RGB").resize((512, 288)))
        read_rgb = lambda: rgb.copy()
    else:
        source = int(args.camera) if args.camera.isdigit() else args.camera
        capture = CameraStream(source)
        read_rgb = capture.read_rgb

    motion = None
    try:
        motion = Go2Motion(args.interface) if args.backend == "go2" else DryRunMotion()
        reporter = DashboardReporter(PolicyClient(args.server, timeout=1), read_rgb,
                                     getattr(motion, "pose", lambda: None))
        result = run_episode(PolicyClient(args.server), read_rgb, motion,
                             args.instruction, max_steps=args.max_steps, reporter=reporter)
        print(json.dumps(result))
    finally:
        if capture is not None:
            capture.close()
        if motion is not None and hasattr(motion, "close"):
            motion.close()


if __name__ == "__main__":
    main()
