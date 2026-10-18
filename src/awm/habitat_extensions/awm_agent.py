"""Observed-history ScaleNav policy for simulation and robot execution."""

import argparse
import copy
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional

import numpy as np
import torch
import yaml

from awm.scripts.inference import (
    clone_noise_scheduler_with_shift,
    encode_joint_observation_anchor,
    frame_to_tensor,
    forward_with_cfg,
    load_models_for_inference,
)

from awm.utils.normalizer import denormalize_trajectory_data, normalize_trajectory_data
from awm.utils.action_cuda_graph import JointVideoActionCudaGraphRunner
from awm.habitat_extensions.utils import habitat_obs_to_rgb, resize_rgb


@dataclass
class AWMAgentConfig:
    """Configuration for AWM agent."""

    model_path: Optional[str] = None
    pretrained_model_path: str = ""
    config_path: str = ""

    policy_mode: str = "causal_observed_replan"

    predict_mode: str = "ti2va"
    num_inference_steps: int = 4
    num_video_frames: int = 9
    action_shift: Optional[float] = 1.0
    chunk_overlap_frames: int = 0

    use_causal_chunk_ar: bool = True
    ar_num_past_chunks: int = 3
    max_attn_chunks: int = 4
    use_special_memory: bool = False
    use_continuous_history: bool = True
    num_special_tokens: int = 8
    special_memory_recent_chunks: int = 2

    guidance_scale: float = 5.0
    traj_guidance_scale: float = 1.0
    use_cfg_parallel: bool = True

    use_kv_cache: bool = True

    use_inference_optimizations: bool = True
    decode_predicted_video: bool = False
    use_cuda_graph: bool = True
    cuda_graph_mode: str = "denoise"

    trim_chunk_overshoot: bool = False

    exec_horizon_frames: int = 0

    horizon_steps: int = 8
    scale_xy: float = 1.0
    scale_yaw: float = 1.0
    stop_threshold: float = 0.05

    disable_stop: bool = False
    no_past_action: bool = False
    use_motion_stop_detector: bool = False
    motion_stop_n_frames: int = 5            # require N consecutive idle frames
    motion_stop_ignore_prefix_frames: int = 0
    motion_stop_xy_threshold_m: float = 0.015  # 1.5cm/frame = 0.06 m/s @ 4fps
    motion_stop_yaw_threshold_rad: float = 0.015  # 0.86 deg/frame = 3.4 deg/s @ 4fps

    auto_stop_on_near_zero: bool = True

    min_steps_before_stop: int = 0
    traj_normalizer_override: Optional[str] = None
    symlog_c_override: Optional[float] = None
    gpu_memory_mode: str = "model_cpu_offload"
    turn_angle_deg: float = 15.0

    continuous_actions: bool = True

    device: str = "cuda"
    dtype: str = "bf16"

    image_height: int = 288
    image_width: int = 512

    use_gt_past: bool = True

    def __post_init__(self):
        if self.policy_mode != "causal_observed_replan" or self.predict_mode != "ti2va":
            raise ValueError("ScaleNav uses observed-history joint video/action prediction")
        if not self.use_causal_chunk_ar or self.use_special_memory:
            raise ValueError("ScaleNav requires the causal model without special memory tokens")
        if not self.auto_stop_on_near_zero or self.use_motion_stop_detector:
            raise ValueError("ScaleNav uses the issued-action near-zero STOP window")
        if not self.continuous_actions or self.cuda_graph_mode != "denoise":
            raise ValueError("ScaleNav uses continuous actions and denoise CUDA graphs")


class AWMAgent:
    """Predict continuous action chunks from RGB observations and an instruction."""

    def __init__(self, config: AWMAgentConfig):
        self.config = config
        if not 0 <= int(config.chunk_overlap_frames) < int(config.num_video_frames):
            raise ValueError(
                "chunk_overlap_frames must be in [0, num_video_frames); "
                f"got overlap={config.chunk_overlap_frames}, "
                f"num_video_frames={config.num_video_frames}"
            )
        expected_window = int(config.ar_num_past_chunks) + 1
        if (
            int(config.max_attn_chunks) != -1
            and int(config.max_attn_chunks) < expected_window
        ):
            raise ValueError(
                "Causal inference history mismatch: max_attn_chunks includes "
                f"the current chunk and must be -1 or >= {expected_window} for "
                f"ar_num_past_chunks={config.ar_num_past_chunks}; got "
                f"{config.max_attn_chunks}"
            )
        if config.use_continuous_history and not math.isclose(
            float(getattr(config, "action_temporal_scale", 1.0)), 1.0
        ):
            raise ValueError(
                "use_continuous_history requires action_temporal_scale=1.0"
            )
        # Pin the CUDA index: simulator initialization can change the default device.
        if config.device == "cuda":
            self._device = torch.device("cuda", 0)
        else:
            self._device = torch.device(config.device)
        _dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self._dtype = _dtype_map[config.dtype]
        self._target_size = (config.image_height, config.image_width)

        self._load_models()

        # Per-episode state
        self._instruction: str = ""
        self._action_queue: List[str] = []
        self._step_count: int = 0
        self._continuous_stop_pending: bool = False
        self._continuous_stop_pending_reason: Optional[str] = None
        self._continuous_stop_latched: bool = False
        self._continuous_motion_window: Deque[bool] = deque(
            maxlen=max(1, int(config.motion_stop_n_frames))
        )
        self.last_stop_reason: Optional[str] = None

        # Causal chunk AR state
        self._ctx_list: Optional[List[torch.Tensor]] = None
        # Unconditional (empty-prompt) context for classifier-free guidance.
        # Built once at reset; nil when guidance_scale <= 1.0.
        self._uncond_ctx_list: Optional[List[torch.Tensor]] = None
        self._inference_context_cache: Optional[Any] = None
        self._uncond_inference_context_cache: Optional[Any] = None
        self._joint_schedule_cache: Optional[Any] = None
        self._joint_cuda_graph_runner: Optional[
            JointVideoActionCudaGraphRunner
        ] = None
        self._joint_cuda_graph_failed: bool = False
        self._past_video_latents: List[torch.Tensor] = []
        self._past_trajectories: List[torch.Tensor] = []
        self._past_chunk_position_ids: List[int] = []
        self._next_chunk_position_id: int = 0
        self._latest_predicted_video_latent: Optional[torch.Tensor] = None
        self._latest_predicted_trajectory: Optional[torch.Tensor] = None
        self._latest_predicted_position_id: Optional[int] = None
        # Cache only fully executed observed chunks persistently. Append predictions
        # temporarily so new observations invalidate only the cached tail.
        self._durable_gt_kv_cond: Optional[Any] = None
        self._durable_gt_kv_uncond: Optional[Any] = None
        self._durable_gt_kv_last_position: int = -1
        self._durable_gt_kv_positions: List[int] = []
        self._persistent_kv_fallbacks: int = 0
        self.last_kv_prefill_s: float = 0.0
        self.last_kv_prefill_mode: str = "none"
        self.last_used_cuda_graph: bool = False

        # Observed visual history is paired with the previous predicted actions.
        self._gt_frame_buf: Deque[np.ndarray] = deque()

        # Optional: cache the most recent denoised video latents decoded to
        # RGB so the rollout driver can render "AWM-imagined" frames next to
        # the real sim frames. None until the first inference fires.
        self.last_generated_video: Optional[np.ndarray] = None  # [T, H, W, 3] uint8

    def _load_models(self):
        """Load AWM model components."""
        cfg = self.config

        with open(cfg.config_path, "r") as f:
            self._train_config = yaml.safe_load(f)

        # Build a minimal argparse namespace for load_models_for_inference
        model_type = self._train_config.get("model_type", "original")
        model_type = "causal_chunk_ar"

        args = argparse.Namespace(
            pretrained_model_path=cfg.pretrained_model_path,
            model_path=cfg.model_path,
            model_type=model_type,
            # Forward gpu_memory_mode so load_models_for_inference puts
            # aux modules (text encoder, VAE) on CPU when set to
            # `model_cpu_offload` — required to fit a 24 GB card.
            gpu_memory_mode=getattr(cfg, "gpu_memory_mode", "model_full_load"),
            chunk_overlap_frames=cfg.chunk_overlap_frames,
            ar_num_past_chunks=cfg.ar_num_past_chunks,
            ar_embed_dim=getattr(cfg, "ar_embed_dim", 1024),
            ar_context_len=getattr(cfg, "ar_context_len", 16),
            ar_num_layers=getattr(cfg, "ar_num_layers", 6),
            ar_model_path=getattr(cfg, "ar_model_path", None),
            max_attn_chunks=cfg.max_attn_chunks,
            action_temporal_scale=getattr(cfg, "action_temporal_scale", 1.0),
            use_special_memory=getattr(cfg, "use_special_memory", False),
            use_continuous_history=getattr(
                cfg, "use_continuous_history", False
            ),
            num_special_tokens=getattr(cfg, "num_special_tokens", 9),
            special_memory_recent_chunks=getattr(
                cfg, "special_memory_recent_chunks", 2
            ),
        )

        models = load_models_for_inference(
            args, self._train_config, self._device, self._dtype
        )
        (
            self._transformer,
            self._vae,
            self._text_encoder,
            self._tokenizer,
            self._noise_scheduler,
        ) = models

        self._action_noise_scheduler = clone_noise_scheduler_with_shift(
            self._noise_scheduler,
            getattr(cfg, "action_shift", None),
        )
        print(
            "Scheduler shifts: "
            f"video={float(self._noise_scheduler.config.shift):g}, "
            f"action={float(self._action_noise_scheduler.config.shift):g}",
            flush=True,
        )

        if not hasattr(self._transformer, 'dit_model'):
            raise ValueError(
                "policy_mode='causal_observed_replan' requires a causal_chunk_ar checkpoint/config."
            )

        # Keep the frequently used VAE on GPU; text encoding runs once per episode on CPU.
        if getattr(cfg, "gpu_memory_mode", "model_full_load") == "model_cpu_offload":
            if self._vae is not None and torch.cuda.is_available():
                self._vae.to(self._device, dtype=self._dtype)
                torch.cuda.empty_cache()

        # Normalization comes from checkpoint/CLI metadata, not model YAML.
        self._traj_normalizer_mode = self._train_config.get("traj_normalizer", "symlog")
        self._symlog_c = float(self._train_config.get("symlog_c", 0.01))
        if cfg.traj_normalizer_override is not None:
            self._traj_normalizer_mode = cfg.traj_normalizer_override
        if self._traj_normalizer_mode != "symlog":
            raise ValueError("ScaleNav uses symlog action normalization")
        if cfg.symlog_c_override is not None:
            self._symlog_c = float(cfg.symlog_c_override)

        _ad = getattr(getattr(self._transformer, "config", None), "action_dim", 3)
        print(
            f"[AWMAgent] action_dim={_ad}  normalizer={self._traj_normalizer_mode}"
            f"(c={self._symlog_c})  disable_stop={cfg.disable_stop}"
            f"  min_steps_before_stop={cfg.min_steps_before_stop}"
            f"  stop_threshold={cfg.stop_threshold}"
            f"  past={cfg.ar_num_past_chunks}"
            f"  attn_window={cfg.max_attn_chunks}"
            f"  chunk={cfg.num_video_frames}/overlap={cfg.chunk_overlap_frames}"
            f"  continuous_history={cfg.use_continuous_history}"
            f"  motion_stop={cfg.motion_stop_n_frames}-frame-sliding",
            flush=True,
        )

    def reset(self, instruction: str):
        """Reset agent state at the beginning of a new episode."""
        self._instruction = instruction
        self._action_queue = []
        self._step_count = 0
        self._ctx_list = self._encode_text(instruction)
        # Encode uncond context too if CFG is enabled.  Done once per
        # episode (the empty-prompt encoding doesn't depend on the
        # instruction, but lives next to _ctx_list so cleanup is symmetric).
        if float(getattr(self.config, "guidance_scale", 1.0)) > 1.0:
            self._uncond_ctx_list = self._encode_text("")
        else:
            self._uncond_ctx_list = None

        self._inference_context_cache = None
        self._uncond_inference_context_cache = None
        if (
            self.config.use_inference_optimizations
            and hasattr(self._transformer, "prepare_inference_context")
        ):
            self._inference_context_cache = (
                self._transformer.prepare_inference_context(
                    self._ctx_list, dtype=self._dtype
                )
            )
            if self._uncond_ctx_list is not None:
                self._uncond_inference_context_cache = (
                    self._transformer.prepare_inference_context(
                        self._uncond_ctx_list, dtype=self._dtype
                    )
                )

        # Reset causal chunk AR state
        self._past_video_latents = []
        self._past_trajectories = []
        self._past_chunk_position_ids = []
        self._next_chunk_position_id = 0
        self._latest_predicted_video_latent = None
        self._latest_predicted_trajectory = None
        self._latest_predicted_position_id = None
        self._durable_gt_kv_cond = None
        self._durable_gt_kv_uncond = None
        self._durable_gt_kv_last_position = -1
        self._durable_gt_kv_positions = []
        self._persistent_kv_fallbacks = 0
        self.last_kv_prefill_s = 0.0
        self.last_kv_prefill_mode = "none"
        self.last_used_cuda_graph = False
        self._gt_frame_buf.clear()

        # Reset gen-video cache
        self.last_generated_video = None
        # Reset debug attrs
        self.last_predicted_traj = None
        self.last_path_2d = None
        self.last_action_names = None
        self.last_stop_reason = None
        self.last_current_frame_anchor = None
        self._continuous_chunk = None
        self._continuous_chunk_idx = 0
        self._continuous_chunk_executable_len = 0
        self._continuous_stop_pending = False
        self._continuous_stop_pending_reason = None
        self._continuous_stop_latched = False
        self._continuous_motion_window = deque(
            maxlen=max(1, int(self.config.motion_stop_n_frames))
        )

    def _encode_text(self, instruction: str) -> List[torch.Tensor]:
        # When gpu_memory_mode=='model_cpu_offload' the text encoder is
        # loaded on CPU (in fp32). Match input device + dtype to where
        # the encoder actually lives; otherwise default to self._device.
        enc_param = next(self._text_encoder.parameters())
        enc_device = enc_param.device
        # Pad only to actual instruction length; masked trailing tokens do not affect T5 outputs.
        text_inputs = self._tokenizer(
            [instruction],
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(enc_device)
        seq_len_ctx = text_inputs.attention_mask[0].gt(0).sum().item()
        ctx = self._text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )[0]
        # Move the resulting context (small, ~seq_len × 4096 floats) onto
        # the inference device + dtype so DiT can consume it.
        ctx = ctx.to(self._device, dtype=self._dtype)
        return [ctx[0][:seq_len_ctx]]


    def _infer_continuous_trajectory(self, rgb: np.ndarray) -> np.ndarray:
        """Run one joint video/action inference and return physical deltas."""
        image_tensor = frame_to_tensor(rgb).to(self._device, self._dtype)
        traj_norm = self._predict_causal_observed_chunk(image_tensor)
        return self._postprocess_trajectory(traj_norm)


    def record_executed_delta(self, delta: np.ndarray) -> np.ndarray:
        """Advance execution state only after a command is issued."""
        action = np.asarray(delta, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] < 3:
            raise ValueError(
                "executed delta must be a 1-D vector with at least 3 values, "
                f"got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("executed delta contains a non-finite value")
        return self._record_continuous_delta(action.copy())


    def act_continuous(self, observation: dict) -> np.ndarray:
        """Return raw body-frame (forward_m, right_m, clockwise_rad)
        delta for the current sim step. Caller integrates the pose manually.

        Behaves like act() for GT-past buffering and inference, but instead
        of running through `trajectory_to_discrete_actions` it walks the
        predicted chunk one delta at a time. A sliding window over issued
        deltas spans prediction chunks and latches STOP after N near-zero
        frames.
        """
        # STOP is absorbing within an episode. In particular, a confirmation
        # request from the robot server must not re-enter inference and resume
        # from the model's unsupervised post-stop trajectory.
        if getattr(self, "_continuous_stop_latched", False):
            self.last_stop_reason = "near_zero_window"
            return None

        rgb_raw = habitat_obs_to_rgb(observation)
        rgb = resize_rgb(rgb_raw, self._target_size)
        if self.config.use_gt_past:
            self._gt_frame_buf.append(rgb)
            if len(self._gt_frame_buf) >= self.config.num_video_frames:
                self._flush_gt_past_chunk()

        chunk = getattr(self, "_continuous_chunk", None)
        idx = getattr(self, "_continuous_chunk_idx", 0)
        cap = getattr(self, "_continuous_chunk_executable_len", None)
        if chunk is not None:
            limit = int(cap) if cap is not None else len(chunk)
            if idx < limit:
                delta = chunk[idx].copy()
                self._continuous_chunk_idx = idx + 1
                return self._record_continuous_delta(delta)

        # A near-zero run may begin after useful leading motion in the same
        # prediction. Execute that prefix first, then surface the stop signal
        # exactly once. The real-robot server can still re-infer on the next
        # fresh frame for its confirm-before-stop check; the simulator treats
        # this first signal as the episode terminator.
        if getattr(self, "_continuous_stop_pending", False):
            self._continuous_stop_pending = False
            self._continuous_chunk = None
            self._continuous_chunk_idx = 0
            self._continuous_chunk_executable_len = 0
            self.last_stop_reason = (
                self._continuous_stop_pending_reason or "stop"
            )
            self._continuous_stop_pending_reason = None
            return None

        image_tensor = frame_to_tensor(rgb).to(self._device, self._dtype)
        traj_norm = self._predict_causal_observed_chunk(image_tensor)
        traj_scaled = self._postprocess_trajectory(traj_norm)  # (T, 3) physical units


        self._continuous_chunk = traj_scaled.astype(np.float32)
        self._continuous_chunk_idx = 1
        n = int(self._continuous_chunk.shape[0])
        if bool(getattr(self.config, "trim_chunk_overshoot", False)) and n > 1:
            self._continuous_chunk_executable_len = n - 1
        else:
            self._continuous_chunk_executable_len = n
        k = int(getattr(self.config, "exec_horizon_frames", 0) or 0)
        if k > 0:
            self._continuous_chunk_executable_len = min(
                self._continuous_chunk_executable_len, k)
        self.last_predicted_traj = traj_scaled.copy()
        return self._record_continuous_delta(self._continuous_chunk[0].copy())

    @property
    def stop_requested(self) -> bool:
        return bool(getattr(self, "_continuous_stop_latched", False))

    def _record_continuous_delta(self, delta: np.ndarray) -> np.ndarray:
        """Record one issued continuous action and update the STOP window.

        Detection intentionally happens after the action is selected for this
        control step. The next call returns STOP only after N consecutive
        near-zero actions have already appeared in the issued stream.
        """
        delta = np.asarray(delta, dtype=np.float32)
        self._step_count += 1

        n = int(self.config.motion_stop_n_frames)
        window = getattr(self, "_continuous_motion_window", None)
        if window is None or window.maxlen != max(1, n):
            previous = list(window or ())
            keep = max(0, n - 1)
            previous = previous[-keep:] if keep else []
            window = deque(previous, maxlen=max(1, n))
            self._continuous_motion_window = window

        xy_mag = float(np.linalg.norm(delta[:2]))
        yaw_mag = float(abs(delta[2]))
        window.append(
            xy_mag < float(self.config.motion_stop_xy_threshold_m)
            and yaw_mag < float(self.config.motion_stop_yaw_threshold_rad)
        )

        if (
            n > 0 and (not self.config.disable_stop) and (self._step_count >= int(self.config.min_steps_before_stop)) and (len(window) == n) and all(window)
        ):
            self._continuous_stop_latched = True
            self.last_stop_reason = "near_zero_window"
        return delta


    def _postprocess_trajectory(self, traj_pred: np.ndarray) -> np.ndarray:
        out = denormalize_trajectory_data(traj_pred, mode="symlog", symlog_c=self._symlog_c).astype(np.float32)
        self.last_sent_dist_per_frame = np.linalg.norm(out[:, :2], axis=1).astype(np.float32)
        return out


    def _current_chunk_has_observation_anchor(self) -> bool:
        return True

    def _effective_causal_history(self):
        return list(self._past_video_latents), list(self._past_trajectories), list(self._past_chunk_position_ids)


    @torch.no_grad()
    def _predict_causal_observed_chunk(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Run one causal chunk, optionally without a post-bootstrap RGB anchor."""
        (
            effective_past_videos,
            effective_past_trajectories,
            effective_past_positions,
        ) = self._effective_causal_history()
        # Diagnostic: log shape of past context + range of last past trajectory
        # delta. This is the cheapest way to spot a body-frame-delta sign or
        # magnitude bug — training-time deltas are typically per-frame
        # forward in [-0.5, +0.5] m and per-frame yaw in [-0.5, +0.5] rad
        # (after normalization, even smaller). Wildly different ranges mean
        # the GT past coming from sim observations doesn't match what the
        # model saw at train time.
        if effective_past_trajectories:
            last_traj = effective_past_trajectories[-1].detach().float().cpu().numpy()
            import logging as _logging
            _logging.debug(
                f"[awm.act] past_chunks={len(effective_past_trajectories)} "
                f"last_traj.shape={tuple(last_traj.shape)} "
                f"fwd[mean,min,max]=[{last_traj[..., 0].mean():.3f},"
                f"{last_traj[..., 0].min():.3f},{last_traj[..., 0].max():.3f}] "
                f"left[mean,min,max]=[{last_traj[..., 1].mean():.3f},"
                f"{last_traj[..., 1].min():.3f},{last_traj[..., 1].max():.3f}] "
                f"yaw[mean,min,max]=[{last_traj[..., 2].mean():.3f},"
                f"{last_traj[..., 2].min():.3f},{last_traj[..., 2].max():.3f}]"
            )

        # The episode bootstrap is always grounded in its real first frame.
        # Joint-ahead inference deliberately does not inject the boundary RGB
        # into later chunks: that frame is unavailable when the next chunk is
        # generated concurrently with execution of the current chunk.
        has_current_anchor = self._current_chunk_has_observation_anchor()
        current_first_frame = image_tensor.unsqueeze(2)  # (1, C, 1, H, W)
        video_n = current_first_frame.expand(
            -1, -1, self.config.num_video_frames, -1, -1
        )
        if not has_current_anchor and effective_past_videos:
            latents_ref = torch.empty_like(effective_past_videos[-1])
            inpaint_latents = torch.zeros_like(latents_ref)
        elif self.config.use_inference_optimizations:
            inpaint_latents = encode_joint_observation_anchor(
                self._vae,
                current_first_frame,
                self.config.num_video_frames,
            ).to(self._dtype)
            latents_ref = inpaint_latents
        else:
            latents_ref = self._vae.encode(
                video_n
            ).latent_dist.sample().to(self._dtype)
        _, c_l, t_l, h_l, w_l = latents_ref.shape

        patch_t, patch_h, patch_w = self._transformer.patch_size
        h_l_trim = (h_l // patch_h) * patch_h
        w_l_trim = (w_l // patch_w) * patch_w
        if h_l_trim != h_l or w_l_trim != w_l:
            latents_ref = latents_ref[:, :, :, :h_l_trim, :w_l_trim]
            h_l, w_l = h_l_trim, w_l_trim

        seq_len = (t_l // patch_t) * (h_l // patch_h) * (w_l // patch_w)

        if not self.config.use_inference_optimizations:
            video_first_only = torch.zeros_like(video_n)
            video_first_only[:, :, 0:1, :, :] = current_first_frame
            inpaint_latents = self._vae.encode(
                video_first_only
            ).latent_dist.sample().to(self._dtype)
        inpaint_latents = inpaint_latents[:, :, :, :h_l, :w_l]

        latent_mask = torch.ones(
            1, c_l, t_l, h_l, w_l,
            device=self._device,
            dtype=self._dtype,
        )
        if has_current_anchor:
            latent_mask[:, :, 0:1, :, :] = 0.0
        self.last_current_frame_anchor = has_current_anchor

        def _blend_and_t(
            latents_noisy: torch.Tensor,
            video_t: torch.Tensor,
            action_t: torch.Tensor,
        ):
            x_cond = (1 - latent_mask) * inpaint_latents + latent_mask * latents_noisy
            mask_tok = latent_mask[:, 0, ::patch_t, ::patch_h, ::patch_w].flatten(1)[:, :seq_len]
            t_video = (mask_tok * video_t.float()).long()
            t_traj = action_t[:, None].expand(-1, self.config.num_video_frames).long()
            return x_cond, torch.cat([t_video, t_traj], dim=1).contiguous()

        _action_dim = getattr(
            getattr(self._transformer, "config", None), "action_dim", 3,
        )
        latents_noisy = torch.randn(1, c_l, t_l, h_l, w_l, device=self._device, dtype=self._dtype)
        traj_noisy = torch.randn(
            1,
            self.config.num_video_frames,
            _action_dim,
            device=self._device,
            dtype=self._dtype,
        )

        video_scheduler = copy.deepcopy(self._noise_scheduler)
        traj_scheduler = copy.deepcopy(self._action_noise_scheduler)
        video_scheduler.set_timesteps(self.config.num_inference_steps, device=self._device)
        traj_scheduler.set_timesteps(self.config.num_inference_steps, device=self._device)

        # Effective past trajectories (shared by prefill and per-step calls).
        past_trajs_eff = (
            [torch.zeros_like(pt) for pt in effective_past_trajectories] if getattr(self.config, 'no_past_action', False) and effective_past_trajectories else effective_past_trajectories
        )
        past_position_ids = (
            torch.tensor(
                effective_past_positions,
                device=self._device,
                dtype=torch.long,
            ).unsqueeze(0)
            if effective_past_positions else None
        )
        current_position_id = torch.tensor(
            [self._next_chunk_position_id], device=self._device, dtype=torch.long
        )

        joint_schedule_cache = None
        if (
            self.config.use_inference_optimizations
            and hasattr(self._transformer, "prepare_joint_schedule_cache")
        ):
            schedule_key = (
                int(seq_len),
                int(traj_noisy.shape[1]),
                int(self.config.num_inference_steps),
                tuple(inpaint_latents.shape),
                float(video_scheduler.config.shift),
                float(traj_scheduler.config.shift),
                str(inpaint_latents.device),
                str(inpaint_latents.dtype),
                bool(has_current_anchor),
            )
            if (
                self._joint_schedule_cache is None
                or self._joint_schedule_cache["key"] != schedule_key
            ):
                prepared = self._transformer.prepare_joint_schedule_cache(
                    inpaint_latents,
                    latent_mask,
                    seq_len,
                    video_scheduler.timesteps,
                    traj_noisy.shape[1],
                    action_timesteps=traj_scheduler.timesteps,
                )
                self._joint_schedule_cache = {
                    "key": schedule_key,
                    **prepared,
                }
            joint_schedule_cache = self._joint_schedule_cache

        # Encode clean history/current observation once. These K/V are
        # immutable across denoising steps.
        static_kv_caches = None
        kv_t0 = time.perf_counter()
        if (
            getattr(self.config, "use_kv_cache", False)
            and effective_past_videos
            and hasattr(self._transformer, "prefill_past_kv")
        ):
            if static_kv_caches is not None:
                self.last_kv_prefill_mode = "persistent"
            else:
                cond_cache = self._transformer.prefill_past_kv(
                    effective_past_videos, past_trajs_eff,
                    self._ctx_list, seq_len,
                    past_chunk_position_ids=past_position_ids,
                    inference_context_cache=self._inference_context_cache,
                    zero_time_embedding=(
                        joint_schedule_cache["zero_time_embedding"]
                        if joint_schedule_cache is not None else None
                    ),
                )
                uncond_cache = None
                if (
                    float(self.config.guidance_scale) > 1.0
                    and self._uncond_ctx_list is not None
                ):
                    uncond_cache = self._transformer.prefill_past_kv(
                        effective_past_videos, past_trajs_eff,
                        self._uncond_ctx_list, seq_len,
                        past_chunk_position_ids=past_position_ids,
                        inference_context_cache=self._uncond_inference_context_cache,
                        zero_time_embedding=(
                            joint_schedule_cache["zero_time_embedding"]
                            if joint_schedule_cache is not None else None
                        ),
                    )
                if cond_cache is not None:
                    static_kv_caches = {
                        "cond": cond_cache,
                        "uncond": uncond_cache,
                    }
                self.last_kv_prefill_mode = "full"
        else:
            self.last_kv_prefill_mode = "none"
        self.last_kv_prefill_s = time.perf_counter() - kv_t0

        used_cuda_graph = False
        joint_graph_eligible = bool(
            self.config.use_inference_optimizations
            and self.config.use_cuda_graph
            and self.config.cuda_graph_mode == "denoise"
            and not self._joint_cuda_graph_failed
            and self._device.type == "cuda"
            and (
                float(self.config.guidance_scale) <= 1.0
                or (
                    bool(getattr(self.config, "use_cfg_parallel", True))
                    and self._uncond_ctx_list is not None
                    and self._uncond_inference_context_cache is not None
                    and isinstance(static_kv_caches, dict)
                    and static_kv_caches.get("uncond") is not None
                )
            )
            and not bool(getattr(self.config, "use_special_memory", False))
            and self._inference_context_cache is not None
            and joint_schedule_cache is not None
            and static_kv_caches is not None
            and int(self.config.ar_num_past_chunks) > 0
            and len(effective_past_videos)
            >= int(self.config.ar_num_past_chunks)
            and all(
                layer_kv is not None
                for layer_kv in self._inference_context_cache["cross_attn_kv"]
            )
        )

        static_kv_workspaces = None
        if (
            self.config.use_inference_optimizations
            and not joint_graph_eligible
            and static_kv_caches is not None
            # A persistent history already keeps cond+uncond K/V resident.
            # Materializing another full per-layer [history | current]
            # workspace pair makes memory scale by a third copy and exhausts
            # a shared 48-GB GPU near 14 C5 chunks.  The attention fallback
            # concatenates one layer at a time, keeping the same values with a
            # much smaller peak allocation.
            and self.last_kv_prefill_mode != "persistent"
            and hasattr(self._transformer, "prepare_joint_kv_workspaces")
        ):
            current_token_len = int(seq_len) + int(traj_noisy.shape[1])
            cond_workspace = self._transformer.prepare_joint_kv_workspaces(
                static_kv_caches["cond"], current_token_len
            )
            uncond_workspace = None
            if static_kv_caches.get("uncond") is not None:
                uncond_workspace = self._transformer.prepare_joint_kv_workspaces(
                    static_kv_caches["uncond"], current_token_len
                )
            static_kv_workspaces = {
                "cond": cond_workspace,
                "uncond": uncond_workspace,
            }

        if joint_graph_eligible:
            try:
                runner = self._joint_cuda_graph_runner
                if runner is None or not runner.matches(
                    static_kv_caches["cond"],
                    latents_noisy,
                    traj_noisy,
                    seq_len,
                    len(effective_past_videos),
                    uncond_kv_caches=static_kv_caches.get("uncond"),
                ):
                    print(
                        "[AWMAgent] Capturing joint video-action denoise CUDA "
                        "graph...",
                        flush=True,
                    )
                    runner = JointVideoActionCudaGraphRunner(
                        wrapper=self._transformer,
                        kv_caches=static_kv_caches["cond"],
                        context=self._ctx_list,
                        inference_context_cache=self._inference_context_cache,
                        schedule_cache=joint_schedule_cache,
                        inpaint_latent=inpaint_latents,
                        latent_mask=latent_mask,
                        video_template=latents_noisy,
                        trajectory_template=traj_noisy,
                        seq_len=seq_len,
                        scheduler=video_scheduler,
                        trajectory_scheduler=traj_scheduler,
                        num_inference_steps=self.config.num_inference_steps,
                        num_past_chunks=len(effective_past_videos),
                        uncond_kv_caches=(
                            static_kv_caches.get("uncond")
                            if isinstance(static_kv_caches, dict) else None
                        ),
                        uncond_context=self._uncond_ctx_list,
                        uncond_inference_context_cache=(
                            self._uncond_inference_context_cache
                        ),
                        guidance_scale=float(self.config.guidance_scale),
                        traj_guidance_scale=float(self.config.traj_guidance_scale),
                    )
                    self._joint_cuda_graph_runner = runner
                latents_noisy, traj_noisy = runner(
                    static_kv_caches["cond"],
                    self._inference_context_cache,
                    inpaint_latents,
                    latents_noisy,
                    traj_noisy,
                    current_position_id,
                    uncond_kv_caches=(
                        static_kv_caches.get("uncond")
                        if isinstance(static_kv_caches, dict) else None
                    ),
                    uncond_inference_context_cache=(
                        self._uncond_inference_context_cache
                    ),
                )
                used_cuda_graph = True
            except Exception as exc:
                self._joint_cuda_graph_failed = True
                self._joint_cuda_graph_runner = None
                print(
                    "[AWMAgent] Joint CUDA graph unavailable; falling back to "
                    f"equivalent eager execution: {exc}",
                    flush=True,
                )
        self.last_used_cuda_graph = bool(used_cuda_graph)

        if not used_cuda_graph:
            for denoise_index, (video_t, action_t) in enumerate(zip(
                video_scheduler.timesteps,
                traj_scheduler.timesteps,
            )):
                x_cond, t_tokens = _blend_and_t(
                    latents_noisy,
                    video_t.unsqueeze(0),
                    action_t.unsqueeze(0),
                )
                expected_t_len = seq_len + traj_noisy.shape[1]
                assert t_tokens.size(1) == expected_t_len, (
                    f"Causal Habitat timestep length {t_tokens.size(1)} "
                    f"!= expected {expected_t_len}."
                )

                video_pred, traj_pred = forward_with_cfg(
                    transformer=self._transformer,
                    guidance_scale=float(self.config.guidance_scale),
                    traj_guidance_scale=float(self.config.traj_guidance_scale),
                    cfg_parallel=bool(
                        getattr(self.config, "use_cfg_parallel", True)
                    ),
                    context=self._ctx_list,
                    uncond_context=self._uncond_ctx_list,
                    static_kv_caches=static_kv_caches,
                    static_inference_contexts={
                        "cond": self._inference_context_cache,
                        "uncond": self._uncond_inference_context_cache,
                    },
                    static_kv_workspaces=static_kv_workspaces,
                    x=[x_cond[0]],
                    traj=traj_noisy,
                    t=t_tokens,
                    seq_len=seq_len,
                    past_video_latents=effective_past_videos or None,
                    past_trajectories=past_trajs_eff,
                    past_chunk_position_ids=past_position_ids,
                    current_chunk_position_id=current_position_id,
                    is_training=False,
                    precomputed_joint_rope_freqs=(
                        joint_schedule_cache["joint_rope_freqs"]
                        if joint_schedule_cache is not None else None
                    ),
                )
                velocity = video_pred[:, :c_l]
                prev = video_scheduler.step(
                    velocity, video_t, x_cond
                ).prev_sample.to(self._dtype)
                latents_noisy = (
                    (1 - latent_mask) * inpaint_latents + latent_mask * prev
                )
                traj_noisy = traj_scheduler.step(
                    traj_pred, action_t, traj_noisy
                ).prev_sample.to(self._dtype)

        elif not self.config.use_gt_past:
            self._past_video_latents.append(latents_noisy.detach())
            self._past_trajectories.append(traj_noisy.detach())
            self._past_chunk_position_ids.append(self._next_chunk_position_id)
            self._next_chunk_position_id += 1
            self._trim_causal_history()

        # Decode the predicted video latents to RGB for visualization. This
        # adds one VAE forward per replan but is essential for the gen-vs-sim
        # side-by-side video. Stored as uint8 [T, H, W, 3].
        if not self.config.decode_predicted_video:
            self.last_generated_video = None
        else:
            try:
                decoded = self._vae.decode(latents_noisy).sample
                decoded = (decoded.clamp(-1, 1) + 1.0) * 127.5
                decoded = decoded.to(torch.uint8)
                self.last_generated_video = (
                    decoded[0].permute(1, 2, 3, 0).cpu().numpy()
                )
            except Exception:
                # Don't let a visualization-only feature break the rollout.
                self.last_generated_video = None

        return traj_noisy[0].cpu().float().numpy()

    def _trim_causal_history(self) -> None:
        """Apply the history cap while preserving an enabled episode anchor."""
        cap = max(0, int(self.config.ar_num_past_chunks))
        while len(self._past_video_latents) > cap:
            # Both anchor modes treat the first observed chunk as permanent.
            # The attention layers later reduce an old anchor chunk to its first
            # latent temporal slice, while these lists reserve the source chunk.
            evict_idx = (
                1 if getattr(self.config, 'use_continuous_history', False) and len(self._past_video_latents) > 1 else 0
            )
            self._past_video_latents.pop(evict_idx)
            self._past_trajectories.pop(evict_idx)
            self._past_chunk_position_ids.pop(evict_idx)

    def _flush_gt_past_chunk(self) -> None:
        """Encode observed frames and retain the previous predicted actions as history."""
        n_frames = self.config.num_video_frames
        if len(self._gt_frame_buf) < n_frames:
            return

        # Take the first n_frames from the buffer.
        frames = [self._gt_frame_buf.popleft() for _ in range(n_frames)]

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # [-1, 1]
        # [T, H, W, 3] -> [1, 3, T, H, W]
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(
            self._device, self._dtype,
        )
        with torch.no_grad():
            past_latent = self._vae.encode(video_tensor).latent_dist.sample().to(self._dtype)

        last_chunk = getattr(self, "last_predicted_traj", None)
        if last_chunk is None:
            last_chunk = getattr(self, "_continuous_chunk", None)
        if last_chunk is not None and len(last_chunk) >= n_frames:
            chunk_np = np.asarray(last_chunk[:n_frames], dtype=np.float32)
            normalized = normalize_trajectory_data(
                chunk_np,
                mode=self._traj_normalizer_mode,
                symlog_c=self._symlog_c,
            )
            past_traj = torch.from_numpy(normalized).unsqueeze(0).float()
            past_traj = past_traj.to(self._device, self._dtype)
            past_source = "model"
        else:
            raise RuntimeError("Observed history requires a complete predicted action chunk")

        self._past_video_latents.append(past_latent)
        self._past_trajectories.append(past_traj)
        self._past_chunk_position_ids.append(self._next_chunk_position_id)
        self._next_chunk_position_id += 1
        self._trim_causal_history()

        # Diagnostic — print AFTER append + pop so n_past reflects the real
        # cap, not the +1 hack.  Also report ar_num_past_chunks so any
        # train-inference mismatch is visible.
        try:
            last_d = past_traj[0, n_frames - 1].detach().float().cpu().numpy()
            print(
                f"    [past_flush] source={past_source}  "
                f"past_traj[{n_frames-1}]=({last_d[0]:+.3f}, {last_d[1]:+.3f}, "
                f"{last_d[2]:+.3f})  n_past={len(self._past_trajectories)}"
                f"(cap={self.config.ar_num_past_chunks})",
                flush=True,
            )
        except Exception:
            pass

        # Retain only the boundary observations explicitly shared by adjacent
        # training chunks. With overlap=0 the next chunk starts at the next
        # fresh observation instead of repeating this chunk's last frame.
        overlap = int(getattr(self.config, "chunk_overlap_frames", 0))
        if overlap:
            self._gt_frame_buf.extendleft(reversed(frames[-overlap:]))
