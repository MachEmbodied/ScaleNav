#!/usr/bin/env python
"""Stateful AWM HTTP policy.

POST /reset sets an instruction; POST /act accepts a base64 RGB image.
POST /act_chunk accepts the intervening observed frames from execution. Responses
contain raw denormalized per-frame displacements and stop/confirmation flags,
not velocities. The visualization sign mapping does not transform API actions.
One active episode/client is supported per process. See docs/INFERENCE.md.

Use the public ``awm serve`` CLI for checkpoint-aware ScaleNav defaults.
"""
import argparse
import base64
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image as PILImage


from awm.habitat_extensions.awm_agent import AWMAgent, AWMAgentConfig
from awm.utils.deploy_contract import DeployContractError, load_deploy_contract
from awm.scripts.visualize_trajectory import integrate_actions

TIMESTEP_PROTOCOL = 1


_CONTRACT_ARG_FIELDS = (
    "num_video_frames",
    "chunk_overlap_frames",
    "ar_num_past_chunks",
    "max_attn_chunks",
    "use_special_memory",
    "use_continuous_history",
    "num_special_tokens",
    "special_memory_recent_chunks",
    "traj_normalizer",
    "symlog_c",
    "image_height",
    "image_width",
    "dtype",
)


def resolve_deploy_args(args: argparse.Namespace) -> dict:
    """Fill checkpoint-bound CLI values and reject explicit mismatches."""
    if args.image_size is not None:
        if args.image_height is None:
            args.image_height = args.image_size
        if args.image_width is None:
            args.image_width = args.image_size

    if args.checkpoint_contract == "auto":
        contract = load_deploy_contract(args.model_path)
        for field in _CONTRACT_ARG_FIELDS:
            expected = contract[field]
            requested = getattr(args, field)
            if requested is not None and requested != expected:
                raise DeployContractError(
                    f"explicit --{field.replace('_', '-')}={requested!r} conflicts "
                    f"with checkpoint value {expected!r}; use the checkpoint value "
                    "and the trained architecture"
                )
            setattr(args, field, expected)
    else:
        manual_defaults = {
            "num_video_frames": 9,
            "chunk_overlap_frames": 0,
            "ar_num_past_chunks": 2,
            "max_attn_chunks": 3,
            "use_special_memory": False,
            "use_continuous_history": False,
            "num_special_tokens": 9,
            "special_memory_recent_chunks": 2,
            "traj_normalizer": "symlog",
            "symlog_c": 0.01,
            "image_height": 288,
            "image_width": 512,
            "dtype": "bf16",
        }
        for field, value in manual_defaults.items():
            if getattr(args, field) is None:
                setattr(args, field, value)
        contract = {field: getattr(args, field) for field in _CONTRACT_ARG_FIELDS}
        contract.update({
            "checkpoint": str(Path(args.model_path).expanduser()),
            "source_files": [],
            "origins": {field: "manual" for field in _CONTRACT_ARG_FIELDS},
        })

    inferred_crop = int(args.image_height) == int(args.image_width)
    if args.crop_square is None:
        args.crop_square = inferred_crop
    if args.crop_square and not inferred_crop:
        raise DeployContractError(
            "square-cropping is incompatible with a non-square checkpoint "
            f"({args.image_width}x{args.image_height}); pass --no-crop-square"
        )
    contract["crop_square"] = bool(args.crop_square)
    contract["control_hz"] = float(args.control_hz)
    contract["num_inference_steps"] = int(args.num_inference_steps)
    contract["guidance_scale"] = float(args.guidance_scale)
    contract["traj_guidance_scale"] = float(args.traj_guidance_scale)
    contract["use_kv_cache"] = bool(args.use_kv_cache)
    contract["use_gt_past"] = bool(args.use_gt_past)
    contract["action_convention"] = args.action_convention
    args.resolved_contract = contract
    return contract


def build_agent(args) -> AWMAgent:
    """Construct AWMAgent mirroring awm_habitat_gs_evaluator._build_agent, but
    driven by CLI flags. Most AWMAgentConfig fields keep their (eval-matched)
    defaults; only paths + the knobs that MUST match the checkpoint are exposed.
    """
    cfg = AWMAgentConfig(
        model_path=args.model_path,
        pretrained_model_path=args.pretrained_model_path,
        config_path=args.config_path,
        policy_mode=args.policy_mode,
        predict_mode=args.predict_mode,
        num_inference_steps=args.num_inference_steps,
        action_shift=args.action_shift,
        num_video_frames=args.num_video_frames,
        chunk_overlap_frames=args.chunk_overlap_frames,
        use_causal_chunk_ar=args.use_causal_chunk_ar,
        ar_num_past_chunks=args.ar_num_past_chunks,
        max_attn_chunks=args.max_attn_chunks,
        exec_horizon_frames=args.exec_horizon_frames,
        traj_normalizer_override=args.traj_normalizer,
        symlog_c_override=args.symlog_c,
        # Use checkpoint image height/width; image-size is the square fallback.
        image_height=args.image_height,
        image_width=args.image_width,
        gpu_memory_mode=args.gpu_memory_mode,
        device=args.device,
        dtype=args.dtype,
        # use_gt_past uses observed FRAMES + model's own chunk (no odometry).
        use_gt_past=args.use_gt_past,
        # Stop-signal policy (faithful to eval defaults; tune per deployment).
        disable_stop=args.disable_stop,
        use_motion_stop_detector=args.use_motion_stop_detector,
        auto_stop_on_near_zero=args.auto_stop_on_near_zero,
        motion_stop_n_frames=args.motion_stop_n_frames,
        motion_stop_ignore_prefix_frames=args.motion_stop_ignore_prefix_frames,
        motion_stop_xy_threshold_m=args.motion_stop_xy_threshold_m,
        motion_stop_yaw_threshold_rad=args.motion_stop_yaw_threshold_rad,
        min_steps_before_stop=args.min_steps_before_stop,
        guidance_scale=args.guidance_scale,
        traj_guidance_scale=args.traj_guidance_scale,
        # Prefill clean history once per control step in causal_observed_replan mode.
        use_kv_cache=args.use_kv_cache,
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_mode=args.cuda_graph_mode,
        decode_predicted_video=args.decode_predicted_video,
        use_special_memory=args.use_special_memory,
        use_continuous_history=args.use_continuous_history,
        num_special_tokens=args.num_special_tokens,
        special_memory_recent_chunks=args.special_memory_recent_chunks,
        # overlap=1 repeats the boundary transition and must trim the final
        # extrapolated action; overlap=0 needs all N actions to reach the next
        # chunk's first frame.
        trim_chunk_overshoot=(args.chunk_overlap_frames > 0),
    )
    print(f"[server] building AWMAgent  model={args.model_path}", flush=True)
    agent = AWMAgent(cfg)
    print("[server] AWMAgent ready.", flush=True)
    return agent


class OfflineRecorder:
    def __init__(self, directory):
        self.root = Path(directory).expanduser() if directory else None
        self.directory = None
        self.lock = threading.Lock()
        self.index = 0

    def start(self, instruction, contract):
        if self.root is None:
            return
        with self.lock:
            self.directory = self.root / f"run-{time.time_ns()}"
            self.directory.mkdir(parents=True, exist_ok=False)
            self.index = 0
            (self.directory / "run.json").write_text(json.dumps(
                {"instruction": instruction, "contract": contract}, ensure_ascii=False, indent=2))

    def save(self, kind, metadata=None, *, rgb=None, raw=None, prediction=None):
        if self.root is None:
            return
        with self.lock:
            if self.directory is None:
                return
            name = f"{self.index:06d}"
            self.index += 1
            event = {"id": name, "kind": kind, "time": time.time(), **(metadata or {})}
            if raw is not None:
                event["image"] = name + "." + _sniff_img_ext(raw)
                (self.directory / event["image"]).write_bytes(raw)
            elif rgb is not None:
                event["image"] = name + ".png"
                PILImage.fromarray(np.asarray(rgb, dtype=np.uint8)).save(self.directory / event["image"])
            if prediction is not None:
                event["prediction"] = name + ".npy"
                np.save(self.directory / event["prediction"], np.asarray(prediction, dtype=np.uint8), allow_pickle=False)
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


class PolicyState:
    """One policy episode, with independent locks for inference and display."""

    _OVERRIDABLE = (
        "motion_stop_n_frames", "motion_stop_ignore_prefix_frames",
        "motion_stop_xy_threshold_m", "motion_stop_yaw_threshold_rad",
        "min_steps_before_stop", "exec_horizon_frames",
    )

    def __init__(self, agent, dt, crop_square=True, confirm_stop=False,
                 deploy_contract=None, action_convention="scan2d_body", record_dir=None):
        self.agent = agent
        self.dt = dt
        self.crop_square = crop_square
        self.deploy_contract = dict(deploy_contract or {})
        self.lock = threading.Lock()
        self._viz_lock = threading.RLock()
        self.have_instruction = False
        self.confirm_stop = confirm_stop
        self.confirm_pending = False
        self._pending_chunk_frames = 0
        self._last_client_frame_seq = None
        self._stop_baseline = {key: getattr(agent.config, key) for key in self._OVERRIDABLE}
        self.recorder = OfflineRecorder(record_dir)
        self._pred_video = []
        self._origin = None
        self._viz = {
            "run_id": 0, "instruction": "", "jpeg": None, "frame_seq": 0,
            "camera_at": None, "frame_width": None, "frame_height": None,
            "deltas": [], "predicted_path": [], "actual_path": [],
            "pose_at": None, "telemetry_at": None, "robot_state": "idle",
            "executed_steps": None, "step": 0, "n_infer": 0,
            "last_infer_s": None, "stop": False, "inferencing": False,
            "pred_video_seq": 0, "prediction_at": None,
            "decode_video": bool(getattr(agent.config, "decode_predicted_video", False)),
        }

    def _encode_jpeg(self, rgb):
        buffer = io.BytesIO()
        PILImage.fromarray(rgb).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def _input_viz_jpeg(self, rgb, raw_input):
        if not self.crop_square and raw_input and raw_input[:2] == b"\xff\xd8":
            return raw_input
        return self._encode_jpeg(rgb)

    def update_preview(self, rgb, raw_input=None):
        jpeg = self._input_viz_jpeg(rgb, raw_input)
        with self._viz_lock:
            self._viz.update(jpeg=jpeg, camera_at=time.time(),
                             frame_width=int(rgb.shape[1]), frame_height=int(rgb.shape[0]))
            self._viz["frame_seq"] += 1

    def _infer(self, rgb):
        with self._viz_lock:
            self._viz["inferencing"] = True
        try:
            return self.agent.act_continuous({"rgb": rgb})
        finally:
            with self._viz_lock:
                self._viz["inferencing"] = False

    def _update_viz(self, deltas, stop, infer_s, video=None):
        path = integrate_actions(deltas) * [1, -1, -1] if deltas else np.empty((0, 3))
        frames = [] if video is None else [self._encode_jpeg(np.asarray(frame, dtype=np.uint8)) for frame in video]
        with self._viz_lock:
            self._viz.update(deltas=deltas or [], predicted_path=path.tolist(), stop=bool(stop),
                             step=int(self.agent._step_count), last_infer_s=float(infer_s),
                             prediction_at=time.time())
            self._viz["n_infer"] += 1
            self._pred_video = frames
            self._viz["pred_video_seq"] += 1

    def pred_video_frame(self, index, version):
        with self._viz_lock:
            if version == self._viz["pred_video_seq"] and 0 <= index < len(self._pred_video):
                return self._pred_video[index]
        return None

    def archive_model_input(self, rgb, raw_input, *, source, action_step, timing=None):
        self.recorder.save("input", {"source": source, "action_step": action_step,
                                     **(timing or {})}, rgb=rgb, raw=raw_input)

    def record_prediction(self, deltas, phase, pred_video=None):
        self.recorder.save("prediction", {"deltas": deltas, "phase": phase,
                                         "action_step": int(self.agent._step_count)},
                           prediction=pred_video)

    def update_telemetry(self, body, rgb=None, raw_input=None):
        run_id = int(body["run_id"])
        pose = body.get("pose")
        if pose is not None:
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("pose must contain finite x, y and counterclockwise yaw")
        status = body.get("state", "idle")
        if status not in {"idle", "waiting", "executing", "model_stop", "budget", "error"}:
            raise ValueError("Invalid robot state")
        steps = int(body.get("steps", 0))
        if steps < 0:
            raise ValueError("steps cannot be negative")
        with self._viz_lock:
            if run_id != self._viz["run_id"]:
                raise ValueError("Telemetry belongs to another episode")
            self._viz.update(robot_state=status, executed_steps=steps, telemetry_at=time.time())
            if pose is not None:
                if self._origin is None:
                    self._origin = pose.copy()
                dx, dy = pose[:2] - self._origin[:2]
                c, s = np.cos(self._origin[2]), np.sin(self._origin[2])
                yaw = float(np.arctan2(np.sin(pose[2] - self._origin[2]), np.cos(pose[2] - self._origin[2])))
                point = [float(c * dx + s * dy), float(-s * dx + c * dy), yaw]
                self._viz["actual_path"].append(point)
                self._viz["actual_path"] = self._viz["actual_path"][-4000:]
                self._viz["pose_at"] = time.time()
            if rgb is not None:
                self.update_preview(rgb, raw_input)
            self.recorder.save("telemetry", {"pose": None if pose is None else pose.tolist(),
                                             "state": status, "steps": steps}, rgb=rgb, raw=raw_input)

    def viz_jpeg(self):
        with self._viz_lock:
            return self._viz["jpeg"]

    def viz_state(self):
        with self._viz_lock:
            result = {key: value for key, value in self._viz.items() if key != "jpeg"}
            result["actual_path"] = list(self._viz["actual_path"])
            result["has_frame"] = self._viz["jpeg"] is not None
            result["pred_video_count"] = len(self._pred_video)
            result["dt"] = self.dt
            return result

    def reset(self, instruction, overrides=None):
        applied = {}
        with self.lock:
            for key, value in self._stop_baseline.items():
                setattr(self.agent.config, key, value)
            for key, value in (overrides or {}).items():
                if key in self._OVERRIDABLE:
                    setattr(self.agent.config, key, type(getattr(self.agent.config, key))(value))
                    applied[key] = getattr(self.agent.config, key)
            self.agent.reset(instruction)
            self.have_instruction = True
            self.confirm_pending = False
            self._pending_chunk_frames = 0
            self._last_client_frame_seq = None
            with self._viz_lock:
                self._viz["run_id"] += 1
                self._viz.update(instruction=instruction, jpeg=None, camera_at=None,
                                 deltas=[], predicted_path=[], actual_path=[], pose_at=None,
                                 telemetry_at=None, robot_state="idle", executed_steps=None,
                                 step=0, n_infer=0, last_infer_s=None, stop=False,
                                 prediction_at=None, inferencing=False)
                self._pred_video = []
                self._viz["pred_video_seq"] += 1
                self._origin = None
                self.recorder.start(instruction, self.deploy_contract)
        return applied

    def _stop_or_confirm(self):
        """Confirm-before-stop: returns ("stop", None) only after TWO
        consecutive stop signals; the first one returns ("confirm", None) so
        the client halts, settles, and sends one fresh frame to re-verify."""
        if self.confirm_pending or not self.confirm_stop:
            self.confirm_pending = False
            return "stop", None
        self.confirm_pending = True
        return "confirm", None

    def act(self, rgb: np.ndarray, client_timing: "dict | None" = None,
            raw_input: "bytes | None" = None):
        with self.lock:
            if not self.have_instruction:
                raise RuntimeError("act before reset — call /reset with an instruction first")
            action_step = int(getattr(self.agent, "_step_count", 0))
            timing = None
            if client_timing is not None:
                sample_step = int(client_timing["sample_step"])
                frame_seq = int(client_timing["frame_seq"])
                frame_monotonic_s = float(client_timing["frame_monotonic_s"])
                if sample_step != action_step:
                    raise ValueError(
                        f"sample_step={sample_step} does not match next "
                        f"action_step={action_step}"
                    )
                if (
                    self._last_client_frame_seq is not None
                    and frame_seq <= self._last_client_frame_seq
                ):
                    raise ValueError(
                        f"frame_seq={frame_seq} is not newer than "
                        f"{self._last_client_frame_seq}"
                    )
                if not math.isfinite(frame_monotonic_s):
                    raise ValueError("frame_monotonic_s must be finite")
                self._last_client_frame_seq = frame_seq
                timing = {
                    "timestep_protocol": TIMESTEP_PROTOCOL,
                    "sample_step": sample_step,
                    "frame_seq": frame_seq,
                    "frame_monotonic_s": frame_monotonic_s,
                    "action_step": None,
                    "chunk_step": None,
                }
            self.update_preview(rgb, raw_input)
            self.archive_model_input(
                rgb,
                raw_input,
                source="act",
                action_step=action_step,
                timing=timing,
            )
            # Refresh prediction visualization only when a new chunk is inferred.
            _chunk = getattr(self.agent, "_continuous_chunk", None)
            _idx = int(getattr(self.agent, "_continuous_chunk_idx", 0) or 0)
            _exec = int(getattr(self.agent, "_continuous_chunk_executable_len", 0) or 0)
            _fresh = _chunk is None or _idx >= _exec
            _t0 = time.monotonic()
            delta = self._infer(rgb)
            _infer_s = time.monotonic() - _t0
            if delta is None:
                phase, out = self._stop_or_confirm()
                self._update_viz([], phase == "stop", _infer_s)
                self.record_prediction([], phase)
                return phase, out, None, timing
            self.confirm_pending = False  # real motion → any pending alarm was false
            chunk_out = None
            if _fresh:
                # Return the full new chunk for path reconstruction without clearing the
                # per-frame action buffer.
                chunk = getattr(self.agent, "_continuous_chunk", None)
                exec_len = int(getattr(self.agent, "_continuous_chunk_executable_len", 0) or 0)
                if chunk is None or exec_len <= 0:
                    deltas = [[float(delta[0]), float(delta[1]), float(delta[2])]]
                else:
                    arr = np.asarray(chunk)[:exec_len]
                    deltas = [[float(a[0]), float(a[1]), float(a[2])] for a in arr]
                chunk_out = deltas
                video = getattr(self.agent, "last_generated_video", None)
                self._update_viz(deltas, False, _infer_s, video)
                self.record_prediction(deltas, "move", video)
            else:
                with self._viz_lock:
                    self._viz["step"] = int(self.agent._step_count)
            if timing is not None:
                timing["action_step"] = action_step
                timing["chunk_step"] = int(
                    getattr(self.agent, "_continuous_chunk_idx", 1) or 1
                ) - 1
        return (
            "move",
            [float(delta[0]), float(delta[1]), float(delta[2])],
            chunk_out,
            timing,
        )

    def act_chunk(self, rgb: np.ndarray, raw_input: "bytes | None" = None,
                  observations: "list[np.ndarray] | None" = None):
        """Return an executable chunk, incorporating observations from the previous chunk."""
        with self.lock:
            if not self.have_instruction:
                raise RuntimeError("act before reset — call /reset with an instruction first")
            cfg = self.agent.config
            frames = observations or []
            if cfg.use_gt_past:
                expected = self._pending_chunk_frames
                if len(frames) != expected:
                    raise ValueError(f"Expected {expected} intervening observations, got {len(frames)}")
                from awm.habitat_extensions.utils import resize_rgb
                resized = [resize_rgb(frame, self.agent._target_size) for frame in frames]
                self.agent._gt_frame_buf.extend(resized)
                self._pending_chunk_frames = 0
            elif frames:
                raise ValueError("Observed frames require observed-history mode")
            for index, frame in enumerate(frames):
                self.archive_model_input(frame, None, source="history",
                                         action_step=int(self.agent._step_count) - len(frames) + index)
            _pm = getattr(cfg, "policy_mode", "")
            if _pm != "causal_observed_replan":
                raise RuntimeError(
                    "/act_chunk requires --policy-mode "
                    f"causal_observed_replan; got policy_mode={_pm!r}."
                )
            self.update_preview(rgb, raw_input)
            self.archive_model_input(
                rgb,
                raw_input,
                source="act_chunk",
                action_step=int(getattr(self.agent, "_step_count", 0)),
            )
            _t0 = time.monotonic()
            first = self._infer(rgb)
            _infer_s = time.monotonic() - _t0
            if first is None:
                phase, _ = self._stop_or_confirm()
                self._update_viz([], phase == "stop", _infer_s)
                self.record_prediction([], phase)
                return phase, None
            self.confirm_pending = False  # real motion → clear pending alarm
            chunk = getattr(self.agent, "_continuous_chunk", None)
            exec_len = int(getattr(self.agent, "_continuous_chunk_executable_len", 0) or 0)
            if chunk is None or exec_len <= 0:
                # Fallback: only the single delta act_continuous surfaced.
                deltas = [[float(first[0]), float(first[1]), float(first[2])]]
            else:
                arr = np.asarray(chunk)[:exec_len]
                deltas = [[float(a[0]), float(a[1]), float(a[2])] for a in arr]
            deltas = self._issue_chunk_tail(deltas)
            if cfg.use_gt_past:
                self._pending_chunk_frames = max(0, len(deltas) - 1)
            # Force a fresh inference on the next call (per-chunk replan).
            self.agent._continuous_chunk = None
            self.agent._continuous_chunk_idx = 0
            self.agent._continuous_chunk_executable_len = 0
            video = getattr(self.agent, "last_generated_video", None)
            self._update_viz(deltas, False, _infer_s, video)
            self.record_prediction(deltas, "move", video)
            return "move", deltas

    def _issue_chunk_tail(self, deltas):
        issued = deltas[:1]
        for delta in deltas[1:]:
            if self.agent.stop_requested:
                break
            self.agent.record_executed_delta(np.asarray(delta, dtype=np.float32))
            issued.append(delta)
        return issued


def _sniff_img_ext(raw: "bytes | None") -> str:
    """Detect the container of the robot's original image bytes (for the
    verbatim archival filename). Defaults to jpg."""
    if not raw:
        return "jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:2] == b"\xff\xd8":
        return "jpg"
    return "bin"


def _decode_rgb_bytes(raw: bytes, crop_square: bool) -> np.ndarray:
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)           # (H, W, 3) RGB
    if crop_square and arr.shape[0] != arr.shape[1]:
        # Match a square checkpoint's field of view before resizing.
        h, w = arr.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        arr = arr[top:top + s, left:left + s]
    return arr




def _decode_uploaded_rgb(
    rgb_b64: str, crop_square: bool
) -> tuple[np.ndarray, bytes]:
    raw = base64.b64decode(rgb_b64)
    return _decode_rgb_bytes(raw, crop_square), raw


_VIZ_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScaleNav · Robot navigation</title>
<style>
:root{color-scheme:dark;--bg:#0c141b;--card:#14212b;--line:#263844;--text:#e5edf2;--muted:#8da4b3;--blue:#72c4f4;--orange:#eab17a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px 24px}header{display:flex;justify-content:space-between;align-items:center;gap:16px}h1{font-size:24px;letter-spacing:-.03em;margin:0}h1 small{font-size:15px;font-weight:400;color:var(--muted);margin-left:12px}.badge{font-size:13px;border:1px solid var(--line);border-radius:20px;padding:5px 12px;color:var(--muted)}.online{color:#8cd8b4}.instruction{font-size:19px;margin:22px 0 16px;overflow-wrap:anywhere}.stats{display:flex;flex-wrap:wrap;gap:12px 30px;padding:14px 0 22px;color:var(--muted)}.stats b{color:var(--text);font-weight:500;margin-left:9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}.panel header{padding:15px 18px}h2{font-size:15px;margin:0;font-weight:600}.hint{font-size:12px;color:var(--muted)}.media{aspect-ratio:16/9;background:#0a1117;position:relative;display:grid;place-items:center}.media img{width:100%;height:100%;object-fit:contain;position:absolute}.media img[hidden]{display:none}.empty{color:var(--muted);font-size:14px}.plot{width:100%;height:300px;display:block}.foot{padding:10px 18px;font-size:12px;color:var(--muted);border-top:1px solid var(--line)}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#8cd8b4;margin-right:5px}.tip{margin-top:20px;color:var(--muted);font-size:12px}@media(max-width:700px){main{padding:20px 14px}.grid{grid-template-columns:1fr}h1 small{display:block;margin:0}header{align-items:flex-start}.stats{gap:8px 20px}.panel header{align-items:center}.instruction{font-size:17px}.plot{height:270px}}
</style></head><body><main>
<header><h1>ScaleNav <small>Robot navigation</small></h1><span id="connection" class="badge">Connecting</span></header>
<p id="instruction" class="instruction">Waiting for an instruction</p>
<div class="stats"><span>Status<b id="status">Idle</b></span><span>Inference time<b id="latency">—</b></span><span>Inferences<b id="inferences">0</b></span><span>Executed actions<b id="steps">—</b></span></div>
<div class="grid">
<section class="panel"><header><h2>Robot camera</h2><span id="camera-state" class="hint">Waiting for a frame</span></header><div class="media"><span id="camera-empty" class="empty">Waiting for the robot camera</span><img id="camera" alt="Current robot view" hidden></div></section>
<section class="panel"><header><h2>Predicted view</h2><span id="prediction-state" class="hint">Waiting for a prediction</span></header><div class="media"><span id="prediction-empty" class="empty">Waiting for a prediction</span><img id="prediction" alt="Predicted view" hidden></div></section>
<section class="panel"><header><h2>Predicted trajectory</h2><span class="hint">Current prediction · m</span></header><canvas id="predicted-path" class="plot" aria-label="Current predicted trajectory"></canvas><div class="foot"><span class="dot"></span>Origin: input pose. Up: forward.</div></section>
<section class="panel"><header><h2>Executed trajectory</h2><span id="pose-state" class="hint">Waiting for odometry</span></header><canvas id="actual-path" class="plot" aria-label="Robot odometry trajectory"></canvas><div class="foot"><span class="dot"></span>Origin: first pose. Up: initial heading. Units: m.</div></section>
</div><p class="tip">Predicted frames follow the action timing and update after each inference. The executed trajectory comes from robot odometry.</p>
<script>
const $=id=>document.getElementById(id);
let state={},cameraSeq=-1,predVersion=-1,predFrames=[],predStarted=0,loadVersion=0;
function draw(id,points,color,empty){
 const canvas=$(id),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
 canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);ctx.scale(ratio,ratio);
 const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.font='13px system-ui';
 if(!points.length){ctx.fillStyle='#8da4b3';ctx.textAlign='center';ctx.fillText(empty,w/2,h/2);return;}
 const xs=points.map(p=>-p[1]),ys=points.map(p=>p[0]);
 const loX=Math.min(-.25,...xs),hiX=Math.max(.25,...xs),loY=Math.min(-.25,...ys),hiY=Math.max(.25,...ys);
 const scale=Math.min((w-80)/(hiX-loX),(h-65)/(hiY-loY));
 const x=v=>w/2+(v-(loX+hiX)/2)*scale,y=v=>h/2-((v-(loY+hiY)/2)*scale);
 const spacing=Math.pow(10,Math.floor(Math.log10(Math.max(hiX-loX,hiY-loY)/3)));
 ctx.strokeStyle='#263844';ctx.lineWidth=1;
 for(let v=Math.ceil(loX/spacing)*spacing;v<=hiX;v+=spacing){ctx.beginPath();ctx.moveTo(x(v),18);ctx.lineTo(x(v),h-25);ctx.stroke();}
 for(let v=Math.ceil(loY/spacing)*spacing;v<=hiY;v+=spacing){ctx.beginPath();ctx.moveTo(20,y(v));ctx.lineTo(w-20,y(v));ctx.stroke();}
 ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(x(-p[1]),y(p[0])):ctx.moveTo(x(-p[1]),y(p[0])));ctx.stroke();
 ctx.fillStyle='#8cd8b4';ctx.beginPath();ctx.arc(x(-points[0][1]),y(points[0][0]),5,0,Math.PI*2);ctx.fill();
 const end=points.at(-1),ex=x(-end[1]),ey=y(end[0]),angle=end[2];ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-14*Math.sin(angle),ey-14*Math.cos(angle));ctx.stroke();
 ctx.fillStyle=color;ctx.beginPath();ctx.arc(ex,ey,4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#8da4b3';ctx.textAlign='right';ctx.fillText('Grid: '+Number(spacing.toPrecision(2))+' m',w-18,h-10);
}
async function loadPrediction(s){
 if(s.pred_video_seq===predVersion)return;predVersion=s.pred_video_seq;const version=++loadVersion;
 predFrames=[];$('prediction').hidden=true;$('prediction-empty').hidden=false;
 if(!s.pred_video_count){$('prediction-state').textContent='No predicted frames';$('prediction-empty').textContent=s.stop?'Model stopped':s.decode_video?'Waiting for a prediction':'Video prediction is disabled';return;}
 $('prediction-empty').textContent='Loading prediction';
 try{
  const frames=await Promise.all(Array.from({length:s.pred_video_count},(_,i)=>new Promise((resolve,reject)=>{
   const img=new Image();img.onload=()=>resolve(img);img.onerror=reject;img.src='/pred_video/'+i+'.jpg?v='+s.pred_video_seq;
  })));
  if(version!==loadVersion)return;predFrames=frames;predStarted=performance.now();$('prediction-empty').hidden=true;$('prediction').hidden=false;playPrediction();
 }catch(e){if(version===loadVersion)$('prediction-empty').textContent='Waiting for the next prediction';}
}
function playPrediction(){
 if(!predFrames.length)return;
 const index=Math.min(predFrames.length-1,Math.floor((performance.now()-predStarted)/((state.dt||.25)*1000)));
 const image=predFrames[index];if($('prediction').src!==image.src)$('prediction').src=image.src;
 $('prediction-state').textContent='Frame '+(index+1)+' / '+predFrames.length;
}
function render(){
 const s=state,now=Date.now()/1000;const names={idle:'Idle',waiting:'Waiting for inference',executing:'Executing',model_stop:'Model STOP',budget:'Action limit reached',error:'Stopped · error'};
 $('instruction').textContent=s.instruction||'Waiting for an instruction';
 $('status').textContent=s.inferencing?'Running inference':s.stop?'Model STOP':names[s.robot_state]||'Idle';
 if(s.telemetry_at&&now-s.telemetry_at>3&&!s.inferencing)$('status').textContent='No recent robot update';
 $('latency').textContent=s.last_infer_s==null?'—':s.last_infer_s.toFixed(2)+' s';$('inferences').textContent=s.n_infer||0;$('steps').textContent=s.executed_steps??'—';
 $('camera-state').textContent=!s.has_frame?'Waiting for a frame':now-s.camera_at>3?'No recent frame':s.frame_width+' × '+s.frame_height;
 $('pose-state').textContent=!s.pose_at?'Waiting for odometry':now-s.pose_at>3?'No recent odometry':'Odometry connected';
 if(!s.has_frame){$('camera').hidden=true;$('camera-empty').hidden=false;}
 else if(s.frame_seq!==cameraSeq){cameraSeq=s.frame_seq;const seq=cameraSeq,image=$('camera');image.onload=()=>{if(state.has_frame&&state.frame_seq===seq){image.hidden=false;$('camera-empty').hidden=true;}};image.src='/latest.jpg?v='+cameraSeq;}
 draw('predicted-path',s.predicted_path||[],'#72c4f4','Waiting for a predicted trajectory');draw('actual-path',s.actual_path||[],'#eab17a','Waiting for robot odometry');loadPrediction(s);
}
async function tick(){
 try{const response=await fetch('/state.json',{cache:'no-store'});if(!response.ok)throw Error();state=await response.json();$('connection').textContent='Connected';$('connection').className='badge online';render();}
 catch(e){$('connection').textContent='Disconnected';$('connection').className='badge';$('status').textContent='Connection lost';}
 setTimeout(tick,400);
}
window.addEventListener('resize',render);setInterval(playPrediction,50);tick();
</script></main></body></html>
""".encode("utf-8")


def make_handler(state: PolicyState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, obj: dict) -> None:
            self._send_bytes(code, json.dumps(obj).encode(), "application/json")

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            if path == "/health":
                self._send(200, {
                    "ok": True,
                    "step": int(state.agent._step_count),
                    "timestep_protocol": TIMESTEP_PROTOCOL,
                    "deploy_contract": state.deploy_contract,
                })
            elif path in ("/", "/viz", "/index.html"):
                self._send_bytes(200, _VIZ_HTML, "text/html; charset=utf-8")
            elif path == "/latest.jpg":
                jpeg = state.viz_jpeg()
                if jpeg is None:
                    self._send(404, {"error": "no frame yet"})
                else:
                    self._send_bytes(200, jpeg, "image/jpeg")
            elif path == "/state.json":
                self._send(200, state.viz_state())
            elif path.startswith("/pred_video/") and path.endswith(".jpg"):
                try:
                    index = int(path[len("/pred_video/"):-4])
                    version = int(parse_qs(parsed.query).get("v", ["-1"])[0])
                except ValueError:
                    index, version = -1, -1
                frame = state.pred_video_frame(index, version)
                if frame is None:
                    self._send(404, {"error": "prediction expired or unavailable"})
                else:
                    self._send_bytes(200, frame, "image/jpeg")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/reset":
                    body = self._read_json()
                    applied = state.reset(str(body.get("instruction", "")),
                                          body.get("overrides"))
                    self._send(200, {"ok": True, "dt": state.dt,
                                     "timestep_protocol": TIMESTEP_PROTOCOL,
                                     "next_action_step": int(state.agent._step_count),
                                     "overrides_applied": applied,
                                     "run_id": state.viz_state()["run_id"]})
                elif self.path == "/act":
                    body = self._read_json()
                    rgb, raw_input = _decode_uploaded_rgb(
                        body["rgb_b64"], state.crop_square
                    )
                    timing_keys = {
                        "timestep_protocol", "sample_step", "frame_seq",
                        "frame_monotonic_s",
                    }
                    supplied = timing_keys.intersection(body)
                    if supplied and supplied != timing_keys:
                        missing = sorted(timing_keys - supplied)
                        raise ValueError(
                            "incomplete timestep metadata; missing " + ", ".join(missing)
                        )
                    client_timing = None
                    if supplied:
                        protocol = int(body["timestep_protocol"])
                        if protocol != TIMESTEP_PROTOCOL:
                            raise ValueError(
                                f"unsupported timestep_protocol={protocol}; "
                                f"expected {TIMESTEP_PROTOCOL}"
                            )
                        client_timing = {key: body[key] for key in timing_keys}
                    phase, delta, chunk, timing = state.act(rgb, client_timing, raw_input=raw_input)
                    # Report stop only after confirmation; an unconfirmed null delta makes clients idle.
                    # Return a whole chunk only on fresh inference.
                    response = {"stop": phase == "stop",
                                "confirm_stop": phase == "confirm",
                                "delta": delta, "chunk": chunk, "dt": state.dt,
                                "stop_after_action": state.agent.stop_requested}
                    if timing is not None:
                        response.update(timing)
                    self._send(200, response)
                elif self.path == "/act_chunk":
                    body = self._read_json()
                    rgb, raw_input = _decode_uploaded_rgb(
                        body["rgb_b64"], state.crop_square
                    )
                    observations = [
                        _decode_uploaded_rgb(frame, state.crop_square)[0]
                        for frame in body.get("observations_b64", [])
                    ]
                    phase, deltas = state.act_chunk(rgb, raw_input=raw_input,
                                                    observations=observations)
                    self._send(200, {"stop": phase == "stop",
                                     "confirm_stop": phase == "confirm",
                                     "deltas": deltas, "dt": state.dt,
                                     "stop_after_chunk": state.agent.stop_requested})
                elif self.path == "/telemetry":
                    body = self._read_json()
                    rgb, raw_input = (None, None)
                    if body.get("rgb_b64"):
                        rgb, raw_input = _decode_uploaded_rgb(body["rgb_b64"], state.crop_square)
                    state.update_telemetry(body, rgb, raw_input)
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 — report to client, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet default access log
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="ScaleNav checkpoint directory")
    ap.add_argument("--pretrained-model-path", required=True, help="Wan2.2-TI2V-5B dir")
    ap.add_argument("--config-path", default="", help="train yaml (for normalizer etc.)")
    ap.add_argument("--record-dir", type=Path, help="Optional offline recording directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8620)
    ap.add_argument("--control-hz", type=float, default=4.0,
                    help="effective per-frame rate; dt=1/hz used by client for delta->velocity")
    # Real-time Go2 defaults. Resolution remains checkpoint-specific and is
    # intentionally configured separately below.
    ap.add_argument("--policy-mode", default="causal_observed_replan", choices=["causal_observed_replan"])
    ap.add_argument("--predict-mode", default="ti2va", choices=["ti2va"])
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--action-shift", type=float, default=1.0,
                    help="Action flow-matching scheduler shift for ScaleNav")
    ap.add_argument("--num-video-frames", type=int, default=None,
                    help="frames per chunk; MUST match the ckpt's training "
                         "video_sample_n_frames.")
    ap.add_argument("--chunk-overlap-frames", type=int, choices=[0, 1], default=None,
                    help="training-time raw-frame overlap. overlap=1 trims the "
                         "duplicated final transition; overlap=0 executes all actions.")
    ap.add_argument("--use-causal-chunk-ar", action="store_true", default=True)
    ap.add_argument("--ar-num-past-chunks", type=int, default=None,
                    help="number of clean history chunks; must match training")
    ap.add_argument("--max-attn-chunks", type=int, default=None,
                    help="attention window including current chunk (2 past + current = 3)")
    ap.add_argument("--exec-horizon-frames", type=int, default=0)
    ap.add_argument("--traj-normalizer", default=None, choices=["symlog"],
                    help="symlog normalization saved by training")
    ap.add_argument("--symlog-c", type=float, default=None,
                    help="symlog scale saved by training (normally inferred)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="square model-input size; used for both axes unless "
                         "--image-height/--image-width are given.")
    ap.add_argument("--image-height", type=int, default=None,
                    help="model-input height (normally inferred from train_meta.pt)")
    ap.add_argument("--image-width", type=int, default=None,
                    help="model-input width (normally inferred from train_meta.pt)")
    ap.add_argument("--gpu-memory-mode", default="model_cpu_offload",
                    choices=["model_full_load", "model_cpu_offload"],
                    help="Model placement strategy.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"],
                    help="AWMAgent dtype key (bf16 for the 4090 deploy).")
    ap.add_argument("--use-gt-past", action="store_true", default=True)
    ap.add_argument("--no-gt-past", dest="use_gt_past", action="store_false")
    ap.add_argument("--disable-stop", action="store_true", default=False)
    ap.add_argument("--auto-stop-on-near-zero", action="store_true", default=True)
    # Stop-detector thresholds. Continuous control uses an issued-action
    # sliding window that crosses chunk boundaries. These remain runtime
    # overridable per episode via /reset without reloading the model.
    ap.add_argument("--motion-stop-n-frames", type=int, default=9)
    ap.add_argument("--motion-stop-xy-threshold-m", type=float, default=0.0325)
    ap.add_argument("--motion-stop-yaw-threshold-rad", type=float, default=0.0325)
    ap.add_argument("--min-steps-before-stop", type=int, default=10)
    # Optional confirmation: halt, collect one fresh frame and re-infer.
    # Only a repeated stop terminates the episode.
    ap.add_argument("--confirm-stop", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--guidance-scale", type=float, default=5.0,
                    help="Video CFG scale. 1.0 skips the unconditional branch; "
                         "")
    ap.add_argument("--traj-guidance-scale", type=float, default=1.0)
    kv_group = ap.add_mutually_exclusive_group()
    kv_group.add_argument("--use-kv-cache", dest="use_kv_cache",
                          action="store_true",
                          help="Enable static history KV cache (default).")
    kv_group.add_argument("--no-kv-cache", dest="use_kv_cache",
                          action="store_false",
                          help="Disable static history KV cache for an A/B check.")
    ap.set_defaults(use_kv_cache=True)
    graph_group = ap.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--use-cuda-graph", dest="use_cuda_graph", action="store_true",
        help="Enable fixed-shape CUDA Graph replay (default).",
    )
    graph_group.add_argument(
        "--no-cuda-graph", dest="use_cuda_graph", action="store_false",
        help="Use optimized eager inference for an A/B check.",
    )
    ap.set_defaults(use_cuda_graph=True)
    ap.add_argument(
        "--cuda-graph-mode", choices=["denoise"], default="denoise",
        help="Capture the joint denoise graph. "
             "",
    )
    decode_group = ap.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--decode-predicted-video",
        dest="decode_predicted_video",
        action="store_true",
        help="Decode predicted video latents for the live debug viewer.",
    )
    decode_group.add_argument(
        "--no-decode-predicted-video",
        dest="decode_predicted_video",
        action="store_false",
        help="Skip visualization-only VAE decode for minimum policy latency.",
    )
    ap.set_defaults(decode_predicted_video=False)
    ap.set_defaults(use_special_memory=None)
    history_group = ap.add_mutually_exclusive_group()
    history_group.add_argument(
        "--use-continuous-history",
        dest="use_continuous_history",
        action="store_true",
        help="Enable first-frame anchor + recent-history absolute-time RoPE.",
    )
    ap.set_defaults(use_continuous_history=None)
    crop_group = ap.add_mutually_exclusive_group()
    crop_group.add_argument("--crop-square", dest="crop_square",
                    action="store_true",
                    help="center-crop non-square frames to 1:1 (match training aspect)")
    crop_group.add_argument("--no-crop-square", dest="crop_square", action="store_false")
    ap.set_defaults(crop_square=None)
    ap.add_argument("--checkpoint-contract", choices=["auto"], default="auto",
                    help="infer and validate checkpoint-bound deployment parameters")
    ap.add_argument("--print-deploy-contract", action="store_true",
                    help="print the resolved contract and exit before loading models")
    ap.add_argument("--action-convention",
                    choices=["scan2d_body"],
                    default="scan2d_body",
                    help="raw action sign convention used by the robot and web viz")
    ap.set_defaults(use_special_memory=None, num_special_tokens=None, special_memory_recent_chunks=None, ahead_no_anchor=False, use_ema_weights=False, stop_threshold=0.02, motion_stop_ignore_prefix_frames=0, use_motion_stop_detector=False, reserve_gpu_gb=0)
    args = ap.parse_args()

    if args.control_hz <= 0:
        ap.error("--control-hz must be positive")
    try:
        contract = resolve_deploy_args(args)
    except DeployContractError as exc:
        ap.error(str(exc))
    print("[server] resolved deploy contract:\n" + json.dumps(
        contract, indent=2, sort_keys=True
    ), flush=True)
    if args.print_deploy_contract:
        return

    agent = build_agent(args)
    state = PolicyState(agent, dt=1.0 / float(args.control_hz),
                        crop_square=args.crop_square,
                        confirm_stop=args.confirm_stop,
                        deploy_contract=contract,
                        action_convention=args.action_convention, record_dir=args.record_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[server] listening on {args.host}:{args.port}  dt={state.dt:.4f}s "
          f"({args.control_hz}Hz)  policy_mode={args.policy_mode} "
          f"kv_cache={args.use_kv_cache} steps={args.num_inference_steps} "
          f"cuda_graph={args.use_cuda_graph}/{args.cuda_graph_mode} "
          f"cfg={args.guidance_scale} past={args.ar_num_past_chunks} "
          f"attn_window={args.max_attn_chunks} "
          f"chunk={args.num_video_frames}/overlap={args.chunk_overlap_frames} "
          f"decode_video={args.decode_predicted_video} "
          f"special={args.use_special_memory} "
          f"continuous_history={args.use_continuous_history}", flush=True)
    print(f"[server]   API:  POST /reset {{instruction}} -> /act_chunk {{rgb_b64}}", flush=True)
    print(f"[server]   VIZ:  http://{args.host}:{args.port}/   "
          f"(live robot-view + predicted trajectory)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down", flush=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
