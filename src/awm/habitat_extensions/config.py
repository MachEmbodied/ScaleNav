"""
Configuration for ScaleNav simulation evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulatorConfig:
    """Habitat simulator configuration."""
    gpu_device_id: int = 0
    allow_sliding: bool = True


@dataclass
class SensorConfig:
    """RGB sensor configuration."""
    height: int = 288
    width: int = 512
    hfov: float = 69.62
    position: list = field(default_factory=lambda: [0.0, 0.35, -0.35])


@dataclass
class ActionConfig:
    """Discrete action space configuration."""
    forward_step_size: float = 0.25
    turn_angle: float = 30.0


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    benchmark: str = "habitatgs_ln"
    episode_manifest: Optional[str] = None
    scene_dataset_config: Optional[str] = None
    max_steps: int = 200
    success_distance: float = 1.25
    success_mode: str = "semantic_contract"

    distributed: bool = False

    output_dir: str = "outputs/evaluation/"
    progress_file: str = "progress.jsonl"
    result_file: str = "summary.json"

    model_path: Optional[str] = None
    pretrained_model_path: str = ""
    config_path: str = ""
    policy_mode: str = "causal_observed_replan"
    predict_mode: str = "ti2va"
    num_inference_steps: int = 4
    num_video_frames: int = 9
    action_shift: Optional[float] = 1.0
    chunk_overlap_frames: int = 0
    use_kv_cache: bool = True
    use_cfg_parallel: bool = True
    use_cuda_graph: bool = True
    decode_predicted_video: bool = False
    eval_seed: Optional[int] = 42
    eval_seed_mode: str = "episode"
    use_causal_chunk_ar: bool = True
    ar_num_past_chunks: int = 3
    max_attn_chunks: int = 4
    use_special_memory: bool = False
    use_continuous_history: bool = True
    use_gt_past: bool = True
    num_special_tokens: int = 8
    special_memory_recent_chunks: int = 2
    horizon_steps: int = 8
    exec_horizon_frames: int = 0
    num_episodes: Optional[int] = None
    scale_xy: float = 1.0
    scale_yaw: float = 1.0
    stop_threshold: float = 0.05
    disable_stop: bool = False
    goal_distance_auto_stop: bool = False
    require_model_stop_for_success: bool = True
    semantic_success_require_facing: bool = False
    use_motion_stop_detector: bool = False
    motion_stop_n_frames: int = 9
    motion_stop_ignore_prefix_frames: int = 0
    motion_stop_xy_threshold_m: float = 0.015
    motion_stop_yaw_threshold_rad: float = 0.015
    min_steps_before_stop: int = 18

    auto_stop_on_near_zero: bool = True

    continuous_actions: bool = True

    guidance_scale: float = 5.0
    traj_guidance_scale: float = 1.0
    trim_chunk_overshoot: bool = False
    device: str = "cuda"
    dtype: str = "bf16"

    save_rollout_video: bool = False
    visualize: bool = False

    gpu_memory_mode: str = "model_cpu_offload"
