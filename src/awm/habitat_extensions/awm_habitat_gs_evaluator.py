"""ScaleNav evaluation on Habitat-GS-LN and Short-Horizon OVON episodes."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .awm_agent import AWMAgent, AWMAgentConfig
from .config import EvalConfig
from .eval_seed import EVAL_SEED_MODES, resolve_eval_seed
from .semantic_goal import evaluate_semantic_goal, navigation_success


def _load_nav_wam_episodes(
    data_root: Path,
    split_file: Path,
    text_level: str = "l3",
) -> List[Dict[str, Any]]:
    """Load episode metadata for every "scene/ep" line in split_file.

    Each returned dict has the fields used by the evaluator:
      scene_id, episode_id, ep_dir, start_xyz, goal_xyz, start_quat_wxyz,
      instruction, hfov_deg, cam_height_m, cam_pitch_deg,
      scan_body_radius_m, scan_body_offset_m, gt_geodesic_m, astar_path
    """
    keys: List[str] = []
    with open(split_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                keys.append(line)

    episodes: List[Dict[str, Any]] = []
    for key in keys:
        scene_id, episode_id = key.split("/", 1)
        ep_dir = data_root / scene_id / episode_id
        ep_json = ep_dir / "episode.json"
        if not ep_json.exists():
            raise FileNotFoundError(ep_json)
        with open(ep_json) as fh:
            meta = json.load(fh)

        # Pull instruction in priority order:
        #   1. nav-wam-interior layout — instruction_text_sam2_v2_gemini.json
        #      (hierarchical L1/L2/L3/L4 under instructions.*.canonical)
        #   2. nav-wam-short layout    — goal_text.json["instruction"]
        #      (single short goal-point string, same across all levels)
        # Empty instruction = model gets no language conditioning, which
        # produces near-random rollouts.
        instr_json = ep_dir / "instruction_text_sam2_v2_gemini.json"
        goal_json = ep_dir / "goal_text.json"
        instruction = ""
        goal_meta: Dict[str, Any] = {}
        if goal_json.exists():
            with open(goal_json) as fh:
                goal_meta = json.load(fh)
        if instr_json.exists():
            with open(instr_json) as fh:
                instr = json.load(fh)
            ins_block = instr.get("instructions", {})
            l1 = ins_block.get("l1_object_goal", {}).get("canonical", "")
            l2 = ins_block.get("l2_waypoint_goal", {}).get("canonical", "")
            l3 = ins_block.get("l3_full_chain", {}).get("canonical", "")
            l4_paras = ins_block.get("l3_full_chain", {}).get("paraphrases") or []
            level_map = {"l1": l1, "l2": l2, "l3": l3, "l4": l4_paras[0] if l4_paras else ""}
            instruction = level_map.get(text_level, l3) or l3 or l2 or l1
        elif goal_meta:
            # short-segment layout — single instruction across all 5 levels
            instruction = (goal_meta.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"Missing instruction: {scene_id}/{episode_id}")

        # First-frame orientation from the recorded trajectory.
        traj_npz = ep_dir / "trajectory.npz"
        # Initial agent pose comes from trajectory.npz[0]: it already has
        # camera height baked into y (nav-wam-data does positions[:,1] +=
        # cam_height before saving). episode.json's start_xyz is the
        # floor-level start (y≈0.1m) — using it as the agent position
        # places the camera at the floor and the view is unusably low.
        start_quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        start_xyz_cam: Optional[List[float]] = None
        gt_positions: List[List[float]] = []
        if traj_npz.exists():
            try:
                with np.load(traj_npz) as nz:
                    q = np.asarray(nz["quat_wxyz"], dtype=np.float64)
                    p = np.asarray(nz["position"], dtype=np.float64)
                if len(q) > 0:
                    start_quat_wxyz = q[0].tolist()
                if len(p) > 0:
                    start_xyz_cam = p[0].tolist()
                    # Full GT trajectory (camera-elevated y, same frame
                    # as the executed path) for mini-map overlay.
                    gt_positions = p.tolist()
            except Exception:
                pass

        # A* reference path for NDTW.
        astar_path: List[List[float]] = []
        nm_npz = ep_dir / "navmesh_ref.npz"
        if nm_npz.exists():
            try:
                with np.load(nm_npz) as nz:
                    if "astar_path" in nz.files:
                        astar_path = np.asarray(nz["astar_path"], dtype=np.float64).tolist()
                    elif "smoothed_path" in nz.files:
                        astar_path = np.asarray(nz["smoothed_path"], dtype=np.float64).tolist()
            except Exception:
                pass

        # Current Scan2D releases use cam_height_m/cam_pitch_deg. Keep the
        # robot_cam_* fallbacks for the earlier nav-wam layout.
        cam_height_m = float(
            meta.get("robot_cam_height_m", meta.get("cam_height_m", 1.2))
        )
        cam_pitch_deg = float(
            meta.get("robot_cam_pitch_deg", meta.get("cam_pitch_deg", 0.0))
        )

        # Camera-level start position (y already includes cam_height).
        # Fall back to floor + cam_height if trajectory.npz missing.
        if start_xyz_cam is None:
            sx = list(meta["start_xyz"])
            sx[1] = float(sx[1]) + cam_height_m
            start_xyz_cam = sx
        episodes.append({
            "scene_id": scene_id,
            "episode_id": episode_id,
            "ep_dir": str(ep_dir),
            "start_xyz": start_xyz_cam,
            "start_xyz_floor": list(meta["start_xyz"]),
            "goal_xyz": list(meta["goal_xyz"]),
            "target_mode": (
                meta.get("target_mode")
                or goal_meta.get("target_mode")
                or "unknown"
            ),
            "target_contract": (
                meta.get("target_contract")
                or goal_meta.get("target_contract")
                or {}
            ),
            "gt_positions": gt_positions,
            "start_quat_wxyz": start_quat_wxyz,
            "instruction": instruction,
            "hfov_deg": float(meta.get("hfov_deg", 69.62)),
            "cam_height_m": cam_height_m,
            "cam_pitch_deg": cam_pitch_deg,
            "cam_forward_m": float(meta.get("cam_forward_m", 0.0)),
            "scan_body_radius_m": meta.get("scan_body_radius_m"),
            "scan_body_offset_m": meta.get("scan_body_offset_m"),
            "action_convention": (
                "scan2d_body_forward_right_clockwise"
                if meta.get("action_representation") == "body_delta_se2"
                else "scan2d_body_forward_right_clockwise"
            ),
            "gt_geodesic_m": float(meta.get("geodesic_dist_m", 0.0)),
            "astar_path": astar_path,
        })
    return episodes


def _quat_wxyz_to_yaw(q_wxyz: List[float]) -> float:
    """Yaw around the +Y axis (habitat up) extracted from a wxyz quaternion.

    Identity quaternion [1,0,0,0] must map to yaw=0 (habitat camera at
    identity looks down -Z, which is the convention `_apply_local_delta`
    uses for `forward at yaw=0`).
    """
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_pitch_to_quat_wxyz(yaw: float, pitch_deg: float) -> List[float]:
    """Match the Scan2D camera quaternion while preserving camera pitch."""
    yaw_half = 0.5 * yaw
    pitch_half = 0.5 * math.radians(pitch_deg)
    cy, sy = math.cos(yaw_half), math.sin(yaw_half)
    cp, sp = math.cos(pitch_half), math.sin(pitch_half)
    return [cp * cy, -sp * cy, cp * sy, -sp * sy]


def _model_delta_to_habitat(
    dx: float,
    dy: float,
    dyaw: float,
    action_convention: str,
) -> Tuple[float, float, float]:
    """Convert a checkpoint action label into Habitat forward/left/CCW."""
    if action_convention == "scan2d_body_forward_right_clockwise":
        return dx, -dy, -dyaw
    raise ValueError(f"Unsupported action convention: {action_convention}")


def _apply_local_delta(
    pos: np.ndarray, yaw: float, dx: float, dy: float, dyaw: float,
) -> Tuple[np.ndarray, float]:
    """
    Apply a robot-local (dx forward, dy left, dyaw) step to a habitat
    world-frame (position, yaw) and return the new (position, yaw).

    Habitat axes: +X right, +Y up, +Z toward the camera (i.e. *into* the
    scene at zero yaw is -Z). Robot frame: +x forward, +y left.
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # Forward direction in habitat XZ plane (at zero yaw, robot forward = -Z).
    fwd_x = -sin_y
    fwd_z = -cos_y
    # Left direction (90 ° CCW around +Y from forward).
    left_x = -cos_y
    left_z = sin_y
    new_pos = pos.copy()
    new_pos[0] = pos[0] + dx * fwd_x + dy * left_x
    new_pos[2] = pos[2] + dx * fwd_z + dy * left_z
    # y stays at floor height (caller can snap to navmesh if desired).
    return new_pos, yaw + dyaw


class AWMHabitatGSEvaluator:
    """Closed-loop evaluator on InteriorGS Habitat-GS scenes."""

    def __init__(self, config: EvalConfig, gs_dataset_root: str, nav_wam_data_root: str,
                 split_file: str, text_level: str = "l3"):
        self.config = config
        if self.config.success_mode != "semantic_contract":
            raise ValueError(f"unsupported success_mode: {self.config.success_mode!r}")
        if self.config.eval_seed_mode not in EVAL_SEED_MODES:
            raise ValueError(
                f"unsupported eval_seed_mode: {self.config.eval_seed_mode!r}"
            )
        self.gs_dataset_root = Path(gs_dataset_root)
        self.nav_wam_data_root = Path(nav_wam_data_root)
        self.split_file = Path(split_file)
        self.text_level = text_level

        # Resolve scene_dataset_config.json under gs_dataset_root.
        candidate_configs = list(self.gs_dataset_root.glob("*.scene_dataset_config.json"))
        if config.scene_dataset_config:
            candidate_configs = [Path(config.scene_dataset_config)]
        if not candidate_configs:
            raise FileNotFoundError(
                f"No *.scene_dataset_config.json under {gs_dataset_root}"
            )
        self.scene_dataset_config = candidate_configs[0]

        # Output dir + progress log.
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self._progress_path = Path(self.config.output_dir) / "progress.jsonl"

        # Lazy habitat-sim import.
        self._habitat_sim = None
        self._init_sim_module()

        # Build the AWM agent once.
        self.agent = self._build_agent()

    def _init_sim_module(self):
        try:
            import habitat_sim
            self._habitat_sim = habitat_sim
        except ImportError as e:
            raise ImportError(
                "Install a compatible Habitat-Sim / Habitat-GS runtime for evaluation."
            ) from e

    def _build_agent(self) -> AWMAgent:
        cfg = self.config
        agent_cfg = AWMAgentConfig(
            model_path=cfg.model_path,
            pretrained_model_path=cfg.pretrained_model_path,
            config_path=cfg.config_path,
            policy_mode=cfg.policy_mode,
            predict_mode=cfg.predict_mode,
            num_inference_steps=cfg.num_inference_steps,
            num_video_frames=cfg.num_video_frames,
            action_shift=getattr(cfg, "action_shift", None),
            chunk_overlap_frames=getattr(cfg, "chunk_overlap_frames", 0),
            use_causal_chunk_ar=cfg.use_causal_chunk_ar,
            ar_num_past_chunks=cfg.ar_num_past_chunks,
            max_attn_chunks=getattr(
                cfg, "max_attn_chunks", cfg.ar_num_past_chunks + 1
            ),
            use_special_memory=getattr(cfg, "use_special_memory", False),
            use_continuous_history=getattr(
                cfg, "use_continuous_history", False
            ),
            use_gt_past=getattr(cfg, "use_gt_past", True),
            num_special_tokens=getattr(cfg, "num_special_tokens", 8),
            special_memory_recent_chunks=getattr(
                cfg, "special_memory_recent_chunks", 2
            ),
            horizon_steps=cfg.horizon_steps,
            exec_horizon_frames=int(getattr(cfg, "exec_horizon_frames", 0) or 0),
            scale_xy=cfg.scale_xy,
            scale_yaw=cfg.scale_yaw,
            stop_threshold=cfg.stop_threshold,
            device=cfg.device,
            dtype=cfg.dtype,
            # Forward optional normalizer overrides so a ckpt trained
            # with `TRAJ_NORM=original` can be evaluated even when the
            # YAML pointed at by config_path defaults to symlog.
            traj_normalizer_override=getattr(cfg, "traj_normalizer_override", None),
            symlog_c_override=getattr(cfg, "symlog_c_override", None),
            # Pass gpu_memory_mode straight through so load_models_for_inference
            # knows whether to put aux modules on CPU. Default
            # `model_full_load` (all GPU) is fine on 40 GB+; YAML key
            # `gpu_memory_mode: model_cpu_offload` cuts ~12 GB so the
            # closed loop fits on a 24 GB card.
            gpu_memory_mode=getattr(cfg, "gpu_memory_mode", "model_full_load"),
            # Route sensor resolution to the agent so closed-loop sim frames
            # match the model's training input size. Without this, AWMAgent
            # silently upsampled 384x384 sim renders to 480x640 (its old
            # defaults) — a 1.7x scale mismatch with training that distorted
            # both video-head decoding and trunk attention patterns.
            image_height=cfg.sensor.height,
            image_width=cfg.sensor.width,
            disable_stop=getattr(cfg, "disable_stop", False),
            use_motion_stop_detector=getattr(cfg, "use_motion_stop_detector", False),
            motion_stop_n_frames=cfg.motion_stop_n_frames,
            motion_stop_ignore_prefix_frames=getattr(
                cfg, "motion_stop_ignore_prefix_frames", 0
            ),
            motion_stop_xy_threshold_m=getattr(cfg, "motion_stop_xy_threshold_m", 0.015),
            motion_stop_yaw_threshold_rad=getattr(cfg, "motion_stop_yaw_threshold_rad", 0.015),
            auto_stop_on_near_zero=getattr(cfg, "auto_stop_on_near_zero", False),
            min_steps_before_stop=cfg.min_steps_before_stop,
            # discrete-action mode by default (reverted 2026-05-27 after
            # closed-loop SR regression from the 2026-05-26 continuous
            # flip).  Override with continuous_actions=true in the yaml
            # or --continuous_actions on the CLI when explicitly testing
            # waypoint mode.
            continuous_actions=getattr(cfg, "continuous_actions", False),
            guidance_scale=getattr(cfg, "guidance_scale", 1.0),
            traj_guidance_scale=getattr(cfg, "traj_guidance_scale", 1.0),
            use_kv_cache=getattr(cfg, "use_kv_cache", False),
            use_inference_optimizations=getattr(
                cfg, "use_inference_optimizations", True
            ),
            decode_predicted_video=cfg.decode_predicted_video,
            use_cfg_parallel=cfg.use_cfg_parallel,
            use_cuda_graph=cfg.use_cuda_graph,
            cuda_graph_mode=getattr(cfg, "cuda_graph_mode", "denoise"),
            trim_chunk_overshoot=getattr(cfg, "trim_chunk_overshoot", False),
        )
        return AWMAgent(agent_cfg)

    def _resolve_scene_ply(self, scene_id: str) -> Path:
        ply = self.gs_dataset_root / "scenes" / scene_id / f"{scene_id}.gs.ply"
        if not ply.exists():
            raise FileNotFoundError(f"InteriorGS asset missing: {ply}")
        return ply

    def _episode_scene_path(self, episode):
        if episode.get("scene_path"):
            return self.gs_dataset_root / episode["scene_path"]
        return self._resolve_scene_ply(episode["scene_id"])

    def _reference_distance(self, sim, episode, start, goal):
        if self.config.benchmark != "short_horizon_ovon":
            return self._geodesic(sim, start, goal)
        query = self._habitat_sim.MultiGoalShortestPath()
        ground = np.array(start, dtype=np.float32, copy=True)
        ground[1] -= float(episode.get("cam_height_m", 1.2))
        query.requested_start = sim.pathfinder.snap_point(ground)
        query.requested_ends = np.asarray(episode["reference_goal_positions"], dtype=np.float32)
        if not sim.pathfinder.find_path(query):
            raise ValueError("No navigable target reference points")
        distance = float(query.geodesic_distance)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Invalid reference distance")
        return distance

    def _build_sim(self, episode: Dict[str, Any]):
        """Build a Habitat-GS Simulator for one episode, with per-episode camera params."""
        hab = self._habitat_sim
        cfg = self.config

        stage_path = self._episode_scene_path(episode)

        sim_cfg = hab.SimulatorConfiguration()
        sim_cfg.scene_id = str(stage_path)
        sim_cfg.scene_dataset_config_file = str(self.scene_dataset_config)
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = cfg.simulator.gpu_device_id

        # Sensor: match the data-collection pipeline 1:1.
        rgb = hab.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = hab.SensorType.COLOR
        rgb.resolution = [cfg.sensor.height, cfg.sensor.width]
        rgb.hfov = float(episode.get("hfov_deg", cfg.sensor.hfov))
        # SensorSpec.position must be a Magnum Vector3, not a plain list.
        import magnum as mn  # noqa: WPS433
        forward = float(episode.get("cam_forward_m", 0.0))
        pitch = math.radians(float(episode.get("cam_pitch_deg", 0.0)))
        rgb.position = mn.Vector3(0.0, forward * math.sin(pitch), -forward * math.cos(pitch))

        agent_cfg = hab.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb]
        agent_cfg.height = episode.get("cam_height_m", 1.2)
        agent_cfg.radius = 0.1

        sim = hab.Simulator(hab.Configuration(sim_cfg, [agent_cfg]))
        # Directly selecting a .gs.ply bypasses the scene-instance navmesh
        # lookup in some Habitat-GS builds. Those builds may segfault inside
        # PathFinder.snap_point() when no navmesh is loaded, so attach the
        # scene's navmesh explicitly before any geodesic query.
        if not sim.pathfinder.is_loaded:
            navmesh = (self.gs_dataset_root / episode["navmesh_path"]
                       if episode.get("navmesh_path") else
                       self.gs_dataset_root / "navmeshes" / f"{episode['scene_id']}.navmesh")
            if navmesh.is_file():
                loaded = sim.pathfinder.load_nav_mesh(str(navmesh))
                if not loaded:
                    raise RuntimeError(f"Failed to load navmesh: {navmesh}")
            else:
                raise FileNotFoundError(f"Missing navmesh: {navmesh}")
        return sim

    def _set_agent_state(self, sim, position, quat_wxyz):
        hab = self._habitat_sim
        agent = sim.initialize_agent(0)
        st = hab.AgentState()
        st.position = np.array(position, dtype=np.float32)
        # quaternion from numpy-quaternion (w, x, y, z order)
        try:
            import quaternion as _q
            st.rotation = _q.quaternion(*[float(c) for c in quat_wxyz])
        except ImportError:
            st.rotation = hab.utils.common.quat_from_coeffs(
                np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
            )
        agent.set_state(st)
        return agent

    def _geodesic(self, sim, a, b) -> float:
        """Habitat-GS doesn't expose Simulator.geodesic_distance; use a
        ShortestPath via the pathfinder instead. Fall back to euclidean.

        Both endpoints are snapped to the navmesh (which lives at floor
        level) before find_path is called. Without this, off-mesh inputs
        at camera-elevated y (e.g. 1.3 m) cause find_path to return
        inflated geodesics and can silently prevent the goal-distance
        early-termination rule from firing."""
        try:
            hab = self._habitat_sim
            if not sim.pathfinder.is_loaded:
                raise RuntimeError("pathfinder has no loaded navmesh")
            a_snap = sim.pathfinder.snap_point(np.asarray(a, dtype=np.float32))
            b_snap = sim.pathfinder.snap_point(np.asarray(b, dtype=np.float32))
            if np.any(np.isnan(a_snap)) or np.any(np.isnan(b_snap)):
                raise RuntimeError("snap returned NaN")
            sp = hab.ShortestPath()
            sp.requested_start = np.asarray(a_snap, dtype=np.float32)
            sp.requested_end = np.asarray(b_snap, dtype=np.float32)
            if sim.pathfinder.find_path(sp):
                return float(sp.geodesic_distance)
        except Exception as exc:
            raise RuntimeError("Navmesh distance query failed") from exc
        raise ValueError("No navigable reference path")


    def _run_episode(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from awm.habitat_extensions.benchmark import prepare_episode
        cfg = self.config
        save_video = cfg.save_rollout_video or cfg.visualize
        episode = prepare_episode(episode, cfg.benchmark)
        ep_id = episode["episode_id"]
        scene_id = episode["scene_id"]
        instruction = episode["instruction"]
        start_xyz = np.array(episode["start_xyz"], dtype=np.float32)
        goal_xyz = np.array(episode["goal_xyz"], dtype=np.float32)
        start_quat_wxyz = episode["start_quat_wxyz"]
        cam_height_m = float(episode.get("cam_height_m", 1.2))
        cam_pitch_deg = float(episode.get("cam_pitch_deg", 0.0))
        action_convention = episode.get(
            "action_convention", "scan2d_body_forward_right_clockwise"
        )
        # Lift goal_xyz to the same camera-elevated y so distance
        # comparisons between agent.position and goal_xyz aren't
        # dominated by a 1.2 m vertical offset.
        goal_xyz_floor = goal_xyz.copy()
        goal_xyz = goal_xyz.copy()
        goal_xyz[1] = goal_xyz_floor[1] + cam_height_m

        sim = self._build_sim(episode)
        try:
            self._set_agent_state(sim, start_xyz, start_quat_wxyz)
            self.agent.reset(instruction)

            start_to_goal_geodesic = self._reference_distance(sim, episode, start_xyz, goal_xyz)

            path_length = 0.0
            prev_pos = start_xyz.copy()
            min_dist = start_to_goal_geodesic
            ndtw_positions = [start_xyz.tolist()]
            step = 0
            stopped = False
            termination_reason = "max_steps"
            yaw = _quat_wxyz_to_yaw(start_quat_wxyz)
            goal_evaluation = evaluate_semantic_goal(
                start_xyz,
                yaw,
                episode,
                fallback_radius_m=cfg.success_distance,
                fallback_distance_m=start_to_goal_geodesic,
                require_facing=getattr(
                    cfg, "semantic_success_require_facing", True
                ),
            )
            semantic_region_ever = bool(goal_evaluation["region_success"])
            semantic_contract_ever = bool(goal_evaluation["contract_success"])
            oracle_path_len = 0.0 if semantic_region_ever else None
            frames: List[np.ndarray] = []
            pred_frames: List[Optional[np.ndarray]] = []
            sent_dist_per_step: List[Optional[float]] = []

            while step < cfg.max_steps:
                obs = sim.get_sensor_observations()
                if save_video:
                    frames.append(np.asarray(obs["rgb"], dtype=np.uint8)[..., :3].copy())
                ag_state = sim.get_agent(0).get_state()
                # AWMAgent needs current pose for `causal_observed_replan`
                # past-chunk construction.
                obs["_agent_position"] = np.asarray(ag_state.position, dtype=np.float32)
                rot = ag_state.rotation
                try:
                    obs["_agent_rotation"] = np.asarray(
                        [rot.w, rot.x, rot.y, rot.z], dtype=np.float32,
                    )
                except AttributeError:
                    obs["_agent_rotation"] = np.asarray(
                        getattr(rot, "components", [1, 0, 0, 0]), dtype=np.float32,
                    )

                # Continuous action: AWMAgent.act_continuous returns either
                # a single (dx, dy, dyaw) ndarray for this step, OR None
                # when the model's stop heuristic fires (compute_stop_flag
                # in AWMAgent). Past chunks are buffered internally for
                # the next causal_observed_replan call.
                delta = self.agent.act_continuous(obs)
                if delta is None:
                    stopped = True
                    termination_reason = "model_" + str(
                        getattr(self.agent, "last_stop_reason", None) or "stop"
                    )
                    break
                # Pair the pred (imagined) RGB frame with this step's
                # delta. AWMAgent._continuous_chunk_idx is incremented
                # AFTER returning chunk[idx], so the just-returned delta
                # corresponds to chunk index (idx-1) within the most
                # recent inference's `last_generated_video`.
                pred_frame: Optional[np.ndarray] = None
                if save_video:
                    try:
                        gv = getattr(self.agent, "last_generated_video", None)
                        chunk_local_idx = max(0, getattr(self.agent, "_continuous_chunk_idx", 1) - 1)
                        if gv is not None and 0 <= chunk_local_idx < len(gv):
                            pf = gv[chunk_local_idx]
                            # gv is uint8 [T, H, W, 3] from VAE decode.
                            if pf.dtype != np.uint8:
                                pf = np.clip(pf, 0, 255).astype(np.uint8)
                            pred_frame = pf
                    except Exception:
                        pred_frame = None
                    pred_frames.append(pred_frame)

                    try:
                        dists = getattr(self.agent, "last_sent_dist_per_frame", None)
                        chunk_local_idx = max(0, getattr(self.agent, "_continuous_chunk_idx", 1) - 1)
                        if dists is not None and 0 <= chunk_local_idx < len(dists):
                            sent_dist_per_step.append(float(dists[chunk_local_idx]))
                        else:
                            sent_dist_per_step.append(None)
                    except Exception:
                        sent_dist_per_step.append(None)
                dx, dy, dyaw = float(delta[0]), float(delta[1]), float(delta[2])
                dx_world, dy_world, dyaw_world = _model_delta_to_habitat(
                    dx, dy, dyaw, action_convention,
                )
                if step == 0 or (step + 1) % 20 == 0:
                    print(f"    step {step:3d}: model dx={dx:+.3f}m dy={dy:+.3f}m "
                          f"dyaw={np.degrees(dyaw):+.1f}° → applied dx={dx_world:+.3f}m "
                          f"dyaw={np.degrees(dyaw_world):+.1f}°  "
                          f"|to_goal|={self._reference_distance(sim, episode, ag_state.position, goal_xyz):.2f}m")

                new_pos, yaw = _apply_local_delta(
                    np.asarray(ag_state.position, dtype=np.float32), yaw,
                    dx_world, dy_world, dyaw_world,
                )
                # Soft collision via navmesh check, but ONLY use the
                # snap result's x/z. The agent is at camera-elevated
                # y (~floor + cam_height); the navmesh lives at floor
                # level. If we blindly took snap_point()'s y the
                # camera would collapse to floor level on the first
                # step (the original bug that made the view "very low"
                # in training video vs eval video).
                try:
                    floor_query = np.array(
                        [new_pos[0], new_pos[1] - cam_height_m, new_pos[2]],
                        dtype=np.float32,
                    )
                    snapped = sim.pathfinder.snap_point(floor_query)
                    delta_xz = np.linalg.norm(
                        np.array([snapped[0] - floor_query[0],
                                  snapped[2] - floor_query[2]])
                    )
                    if (not np.any(np.isnan(snapped))) and delta_xz < 0.5:
                        new_pos = np.array(
                            [float(snapped[0]), float(new_pos[1]), float(snapped[2])],
                            dtype=np.float32,
                        )
                    else:
                        new_pos = prev_pos.copy()  # blocked — stay put
                except Exception:
                    pass

                self._set_agent_state(
                    sim, new_pos, _yaw_pitch_to_quat_wxyz(yaw, cam_pitch_deg)
                )
                step += 1

                segment = float(np.linalg.norm(new_pos - prev_pos))
                path_length += segment
                prev_pos = new_pos.copy()
                ndtw_positions.append(new_pos.tolist())

                dist = self._reference_distance(sim, episode, new_pos, goal_xyz)
                min_dist = min(min_dist, dist)
                goal_evaluation = evaluate_semantic_goal(
                    new_pos,
                    yaw,
                    episode,
                    fallback_radius_m=cfg.success_distance,
                    fallback_distance_m=dist,
                    require_facing=getattr(
                        cfg, "semantic_success_require_facing", True
                    ),
                )
                semantic_region_ever = bool(
                    semantic_region_ever or goal_evaluation["region_success"]
                )
                semantic_contract_ever = bool(
                    semantic_contract_ever or goal_evaluation["contract_success"]
                )
                if oracle_path_len is None and goal_evaluation["region_success"]:
                    oracle_path_len = path_length

                if self.agent.stop_requested:
                    stopped = True
                    termination_reason = "model_near_zero_window"
                    break
            if save_video and len(frames) < len(ndtw_positions):
                terminal_rgb = sim.get_sensor_observations()["rgb"]
                frames.append(np.asarray(terminal_rgb, dtype=np.uint8)[..., :3].copy())

            final_pos = np.array(sim.get_agent(0).get_state().position)
            final_dist = self._reference_distance(sim, episode, final_pos, goal_xyz)
            final_goal_evaluation = evaluate_semantic_goal(
                final_pos,
                yaw,
                episode,
                fallback_radius_m=cfg.success_distance,
                fallback_distance_m=final_dist,
                require_facing=getattr(
                    cfg, "semantic_success_require_facing", True
                ),
            )
            stopped_via_model = termination_reason.startswith("model_")
            require_model_stop = bool(getattr(
                cfg, "require_model_stop_for_success", True
            ))
            semantic_success = navigation_success(
                final_goal_evaluation,
                stopped_via_model=stopped_via_model,
                require_model_stop=require_model_stop,
            )
            task_completion_success = bool(
                final_goal_evaluation["contract_success"]
                and stopped_via_model
            )
            semantic_oracle_success = semantic_region_ever
            reached_then_drifted = bool(
                semantic_region_ever and not final_goal_evaluation["region_success"]
            )
            arrived_without_stop = bool(
                final_goal_evaluation["contract_success"] and not stopped_via_model
            )
            stopped_outside_goal = bool(
                stopped_via_model and not final_goal_evaluation["contract_success"]
            )

            success = semantic_success
            oracle_success = semantic_oracle_success
            navigation_error = final_goal_evaluation["navigation_error"]
            primary_success_type = final_goal_evaluation["success_type"]
            primary_oracle_path_len = oracle_path_len

            if start_to_goal_geodesic > 0:
                spl = float(success) * start_to_goal_geodesic / max(
                    path_length, start_to_goal_geodesic
                )
            else:
                spl = 0.0
            oracle_spl = 0.0
            if (
                oracle_success
                and start_to_goal_geodesic > 0
                and primary_oracle_path_len is not None
            ):
                oracle_spl = start_to_goal_geodesic / max(
                    primary_oracle_path_len, start_to_goal_geodesic
                )


            video_path = ""
            if save_video and frames:
                video_path = self._write_rollout_video(
                    frames=frames,
                    pred_frames=pred_frames,
                    ndtw_positions=ndtw_positions,
                    goal_xyz=goal_xyz,
                    sim=sim,
                    scene_id=scene_id,
                    ep_id=ep_id,
                    success=success,
                    instruction=instruction,
                    gt_positions=episode.get("gt_positions") or [],
                    target_footprint_xz=(
                        (episode.get("target_contract") or {}).get(
                            "semantic_target_footprint_world_xz"
                        )
                        or []
                    ),
                    sent_dist_per_step=sent_dist_per_step,
                    stop_threshold=float(getattr(cfg, "stop_threshold", 0.5)),
                )

            return {
                "scene_id": scene_id,
                "episode_id": ep_id,
                "instruction": instruction,
                "target_mode": final_goal_evaluation["target_mode"],
                "success": success,
                "spl": spl,
                "navigation_error": navigation_error,
                "oracle_success": oracle_success,
                "oracle_spl": oracle_spl,
                "semantic_region_success": final_goal_evaluation["region_success"],
                "semantic_contract_success": final_goal_evaluation["contract_success"],
                "task_completion_success": task_completion_success,
                "oracle_semantic_contract_success": semantic_contract_ever,
                "reached_then_drifted": reached_then_drifted,
                "arrived_without_stop": arrived_without_stop,
                "stopped_outside_goal": stopped_outside_goal,
                "semantic_contract_available": final_goal_evaluation[
                    "semantic_contract_available"
                ],
                "success_type": primary_success_type,
                "success_mode": cfg.success_mode,
                "semantic_success": semantic_success,
                "semantic_oracle_success": semantic_oracle_success,
                "success_requires_model_stop": require_model_stop,
                "semantic_goal": final_goal_evaluation,
                "path_length": path_length,
                "steps_taken": step,
                "executed_positions_world": ndtw_positions,
                "final_yaw_rad": float(yaw),
                "stopped": stopped,
                "stopped_via_model": stopped_via_model,
                "termination_reason": termination_reason,
                "start_to_goal_geodesic": start_to_goal_geodesic,
                "gt_geodesic_m": episode.get("gt_geodesic_m"),
                "video_path": video_path,
            }
        finally:
            sim.close()

    def _write_rollout_video(
        self, frames, ndtw_positions, goal_xyz, sim, scene_id, ep_id,
        success, instruction, pred_frames=None, gt_positions=None,
        target_footprint_xz=None, sent_dist_per_step=None, stop_threshold=0.5,
    ) -> str:
        import cv2
        import re
        from awm.scripts.inference import save_rgb_video

        if not frames:
            return ""
        height, width = frames[0].shape[:2]
        include_prediction = pred_frames is not None and any(frame is not None for frame in pred_frames)
        rendered = []
        for index, frame in enumerate(frames):
            panels = [np.asarray(frame, dtype=np.uint8)]
            if include_prediction:
                prediction = pred_frames[index] if index < len(pred_frames) else None
                panels.append(np.zeros_like(frame) if prediction is None else cv2.resize(prediction, (width, height)))
            rendered.append(np.concatenate(panels, axis=1))
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{scene_id}__{ep_id}")
        path = Path(self.config.output_dir) / "videos" / f"{name}.mp4"
        save_rgb_video(np.stack(rendered), path, fps=4)
        return str(path)


    def evaluate(self) -> Dict[str, Any]:
        from awm.habitat_extensions.benchmark import load_manifest, prepare_episode
        if self.config.episode_manifest:
            episodes = load_manifest(self.config.episode_manifest, self.config.benchmark)
        else:
            episodes = _load_nav_wam_episodes(
                self.nav_wam_data_root, self.split_file, self.text_level,
            )
        episodes = [prepare_episode(ep, self.config.benchmark) for ep in episodes]
        for ep in episodes:
            scene = self._episode_scene_path(ep)
            if not scene.is_file():
                raise FileNotFoundError(scene)

        if self.config.num_episodes:
            episodes = episodes[: int(self.config.num_episodes)]
        print(f"[habitat-gs eval] running {len(episodes)} episodes from {self.split_file}")

        results: List[Dict[str, Any]] = []
        # Wipe progress at evaluate() start so each invocation is fresh.
        if self._progress_path.exists():
            self._progress_path.unlink()

        for i, ep in enumerate(episodes):
            print(f"\n[{i + 1}/{len(episodes)}] scene={ep['scene_id']} ep={ep['episode_id']} "
                  f"\"{ep['instruction'][:80]}\"")
            # Deterministic inference RNG for initial latents and VAE sampling.
            # episode mode is invariant to ordering, filtering, and sharding.
            episode_seed = None
            if getattr(self.config, "eval_seed", None) is not None:
                import torch as _torch
                episode_seed = resolve_eval_seed(
                    self.config.eval_seed,
                    self.config.eval_seed_mode,
                    scene_id=ep["scene_id"],
                    episode_id=ep["episode_id"],
                    local_index=i,
                )
                _torch.manual_seed(episode_seed)
                np.random.seed(episode_seed)
            res = self._run_episode(ep)
            if res is None:
                raise RuntimeError(f"Episode returned no result: {ep['episode_id']}")
            res["eval_seed_base"] = self.config.eval_seed
            res["eval_seed"] = episode_seed
            res["eval_seed_mode"] = self.config.eval_seed_mode
            results.append(res)
            with open(self._progress_path, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(
                f"  → success={res['success']} "
                f"region={res['semantic_region_success']} "
                f"stop={res['stopped_via_model']} "
                f"type={res['success_type']} "
                f"ne={res['navigation_error']:.2f}m "
                f"spl={res['spl']:.2f}"
            )

        agg = self._aggregate(results)
        summary_path = Path(self.config.output_dir) / "summary.json"
        with open(summary_path, "w") as fh:
            json.dump({"benchmark": self.config.benchmark, "aggregate": agg, "episodes": results}, fh, indent=2)
        print(f"\n[habitat-gs eval] summary → {summary_path}")
        self._print_metrics(agg, len(results))
        if self.config.visualize:
            from awm.scripts.visualize_evaluation import BENCHMARK_TITLES, build_gallery, read_cases
            output = Path(self.config.output_dir)
            cases = read_cases(results, episodes, output / "videos")
            build_gallery(cases, output / "gallery", title=BENCHMARK_TITLES[self.config.benchmark])
            print(f"[habitat-gs eval] gallery → {output / 'gallery/index.html'}")
        return agg

    @staticmethod
    def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        out: Dict[str, float] = {"n_episodes": len(results)}
        rate_keys = {
            "success",
            "oracle_success",
            "semantic_region_success",
            "semantic_contract_success",
            "task_completion_success",
            "oracle_semantic_contract_success",
            "semantic_contract_available",


            "stopped_via_model",
            "reached_then_drifted",
            "arrived_without_stop",
            "stopped_outside_goal",
        }
        for key in (
            "success", "spl", "navigation_error", "oracle_success",
            "oracle_spl", "semantic_region_success",
            "semantic_contract_success", "task_completion_success",
            "oracle_semantic_contract_success",
            "semantic_contract_available",

            "stopped_via_model", "reached_then_drifted",
            "arrived_without_stop", "stopped_outside_goal",
            "path_length", "steps_taken",
        ):
            vals = [r[key] for r in results if key in r]
            if vals:
                output_key = f"{key}_rate" if key in rate_keys else key
                out[output_key] = (
                    float(np.mean([float(v) for v in vals]))
                )
        return out

    @staticmethod
    def _print_metrics(agg: Dict[str, float], n: int) -> None:
        if not agg:
            print("(no episodes evaluated)")
            return
        print(f"\n=== Habitat-GS eval over {n} episodes ===")
        for k, v in agg.items():
            print(f"  {k:30s}: {v}")
