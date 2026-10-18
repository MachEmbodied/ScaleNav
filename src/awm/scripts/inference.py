"""Joint video/action sampling and single-observation inference."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer
from awm.models import AutoencoderKLWan3_8, WanT5EncoderModel
from awm.utils.misc import reencode_h264


def _aux_module_placement(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.device, torch.dtype]:
    gpu_memory_mode = getattr(args, "gpu_memory_mode", "model_full_load")
    if device.type == "cuda" and gpu_memory_mode == "model_cpu_offload":
        return torch.device("cpu"), torch.float32
    return device, dtype


def _validate_causal_checkpoint_contract(args: argparse.Namespace) -> None:
    """Reject silent architecture mismatches when loading a causal checkpoint."""
    if not getattr(args, "model_path", None):
        return
    config_path = Path(args.model_path) / "transformer" / "config.json"
    if not config_path.is_file():
        return
    stored = json.loads(config_path.read_text())
    requested_memory = bool(getattr(args, "use_special_memory", False))
    if "use_special_memory" in stored:
        stored_memory = bool(stored["use_special_memory"])
        if stored_memory != requested_memory:
            raise ValueError(
                "Checkpoint special-memory mismatch: "
                f"checkpoint={stored_memory}, requested={requested_memory}. "
                "Use the eval/inference settings saved for this checkpoint."
            )
    requested_continuous = bool(
        getattr(args, "use_continuous_history", False)
    )
    stored_continuous = bool(stored.get("use_continuous_history", False))
    if stored_continuous != requested_continuous:
        raise ValueError(
            "Checkpoint continuous-history mismatch: "
            f"checkpoint={stored_continuous}, requested={requested_continuous}. "
            "Use the eval/inference settings saved for this checkpoint."
        )
    if requested_memory:
        expected = {
            "num_special_tokens": int(getattr(args, "num_special_tokens", 9)),
            "special_memory_recent_chunks": int(
                getattr(args, "special_memory_recent_chunks", 2)
            ),
        }
        for key, value in expected.items():
            if key in stored and int(stored[key]) != value:
                raise ValueError(
                    f"Checkpoint {key} mismatch: checkpoint={stored[key]}, "
                    f"requested={value}"
                )
    if "max_attn_chunks" in stored:
        requested_window = int(getattr(args, "max_attn_chunks", -1))
        if int(stored["max_attn_chunks"]) != requested_window:
            raise ValueError(
                "Checkpoint max_attn_chunks mismatch: "
                f"checkpoint={stored['max_attn_chunks']}, requested={requested_window}"
            )
    if "chunk_overlap_frames" in stored:
        requested_overlap = int(getattr(args, "chunk_overlap_frames", 0))
        if int(stored["chunk_overlap_frames"]) != requested_overlap:
            raise ValueError(
                "Checkpoint chunk_overlap_frames mismatch: "
                f"checkpoint={stored['chunk_overlap_frames']}, "
                f"requested={requested_overlap}"
            )


def load_models_for_inference(
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Returns (transformer, vae, text_encoder, tokenizer, noise_scheduler).

    VAE / text encoder / tokenizer always come from --pretrained_model_path.
    Transformer comes from --model_path/transformer/ when provided, otherwise
    from --pretrained_model_path.
    """
    if not args.model_path:
        raise ValueError("A trained ScaleNav checkpoint is required")
    base_path = args.pretrained_model_path
    if getattr(args, 'gpu_memory_mode', 'model_full_load') not in {'model_full_load', 'model_cpu_offload'}:
        raise ValueError('Supported memory modes are model_full_load and model_cpu_offload')
    print("Loading models...")
    _validate_causal_checkpoint_contract(args)

    sched_cfg = config.get("scheduler_kwargs", {})
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        shift=sched_cfg.get("shift", 5.0),
    )

    te_cfg = config.get("text_encoder_kwargs", {})
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_path, te_cfg.get("tokenizer_subpath", "google/umt5-xxl"))
    )

    aux_device, aux_dtype = _aux_module_placement(args, device, dtype)
    gpu_memory_mode = getattr(args, "gpu_memory_mode", "model_full_load")

    print("  Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_path, te_cfg.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=te_cfg,
        low_cpu_mem_usage=True,
        torch_dtype=aux_dtype,
    )
    text_encoder.eval().requires_grad_(False).to(aux_device, dtype=aux_dtype)
    print(f"    text encoder -> {aux_device} ({aux_dtype})")

    print("  Loading VAE...")
    vae_cfg = config.get("vae_kwargs", {})
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(base_path, vae_cfg.get("vae_subpath", "vae")),
        additional_kwargs=vae_cfg,
    )
    vae.eval().requires_grad_(False).to(aux_device, dtype=aux_dtype)
    print(f"    VAE -> {aux_device} ({aux_dtype})")
    if device.type == "cuda" and gpu_memory_mode == "model_cpu_offload":
        torch.cuda.empty_cache()

    from awm.utils.checkpoint import load_causal_checkpoint
    transformer = load_causal_checkpoint(args.model_path, device, dtype)
    return transformer, vae, text_encoder, tokenizer, noise_scheduler


def clone_noise_scheduler_with_shift(
    noise_scheduler: Any,
    shift: Optional[float] = None,
) -> Any:
    """Clone a flow-matching scheduler with an optional modality shift.

    ``shift=None`` preserves the checkpoint/config scheduler exactly.  A new
    scheduler instance is required when the action path uses a different
    shift because ``set_shift`` only changes the runtime attribute; it does
    not rebuild the training sigma table created by the constructor.
    """
    if shift is None:
        return copy.deepcopy(noise_scheduler)

    resolved_shift = float(shift)
    if not math.isfinite(resolved_shift) or resolved_shift <= 0.0:
        raise ValueError(f"action shift must be finite and positive, got {shift!r}")

    configured_shift = float(getattr(noise_scheduler.config, "shift", 1.0))
    if math.isclose(resolved_shift, configured_shift, rel_tol=0.0, abs_tol=1e-12):
        return copy.deepcopy(noise_scheduler)

    return type(noise_scheduler).from_config(
        noise_scheduler.config,
        shift=resolved_shift,
    )


def frame_to_tensor(frame_rgb: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 RGB → (1,C,H,W) float in [-1,1]."""
    arr = frame_rgb.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def encode_joint_observation_anchor(
    vae: torch.nn.Module,
    first_frame_video: torch.Tensor,
    num_video_frames: int,
    generator: Optional[torch.Generator] = None,
    temporal_compression_ratio: int = 4,
) -> torch.Tensor:
    """Encode the observation anchor with the publication sampling sequence."""
    if first_frame_video.ndim != 5 or first_frame_video.shape[2] != 1:
        raise ValueError(
            "first_frame_video must have shape [B,C,1,H,W], got "
            f"{tuple(first_frame_video.shape)}"
        )
    if int(num_video_frames) < 1 or int(temporal_compression_ratio) < 1:
        raise ValueError(
            "num_video_frames and temporal_compression_ratio must be positive"
        )
    latent_frames = 1 + (int(num_video_frames) - 1) // int(
        temporal_compression_ratio
    )
    posterior = vae.encode(first_frame_video).latent_dist
    full_shape = list(posterior.mean.shape)
    full_shape[2] = latent_frames

    # The first draw corresponds to the discarded repeated-video posterior.
    _ = randn_tensor(
        tuple(full_shape),
        generator=generator,
        device=posterior.mean.device,
        dtype=posterior.mean.dtype,
    )
    noise = randn_tensor(
        tuple(full_shape),
        generator=generator,
        device=posterior.mean.device,
        dtype=posterior.mean.dtype,
    )
    anchor = posterior.mean + posterior.std * noise[:, :, :1]
    inpaint_latent = anchor.new_zeros(full_shape)
    inpaint_latent[:, :, :1] = anchor
    return inpaint_latent


def forward_with_cfg(
    transformer: torch.nn.Module,
    guidance_scale: float,
    context: List[torch.Tensor],
    uncond_context: Optional[List[torch.Tensor]] = None,
    return_kv_caches: bool = False,
    kv_caches: Optional[Any] = None,
    traj_guidance_scale: Optional[float] = None,
    cfg_parallel: bool = False,
    static_kv_caches: Optional[Any] = None,
    static_inference_contexts: Optional[Any] = None,
    static_kv_workspaces: Optional[Any] = None,
    **kwargs: Any,
):
    """Run one transformer step with optional classifier-free guidance.

    Args:
        traj_guidance_scale: Separate CFG scale for trajectory. If None,
            falls back to ``guidance_scale``.
        cfg_parallel: Batch the uncond+cond branches into ONE forward
            (batch dim doubles) instead of two sequential calls. Row order is
            [uncond..., cond...]. When kv_caches is provided it must already
            be batched the same way (e.g. prefilled with duplicated inputs) —
            the per-branch {"cond": ..., "uncond": ...} dict is not used.
    """
    traj_gs = traj_guidance_scale if traj_guidance_scale is not None else guidance_scale
    if isinstance(static_inference_contexts, dict) and (
        "cond" in static_inference_contexts or "uncond" in static_inference_contexts
    ):
        cond_context_cache = static_inference_contexts.get("cond")
        uncond_context_cache = static_inference_contexts.get("uncond")
    else:
        cond_context_cache, uncond_context_cache = static_inference_contexts, None
    if isinstance(static_kv_workspaces, dict):
        cond_workspace = static_kv_workspaces.get("cond")
        uncond_workspace = static_kv_workspaces.get("uncond")
    else:
        cond_workspace, uncond_workspace = static_kv_workspaces, None

    if static_kv_caches is not None:
        # Read-only prefilled clean-past caches (per CFG branch). The caches
        # were built once per control step via wrapper.prefill_past_kv();
        # each denoise step only forwards the current chunk's tokens.
        if isinstance(static_kv_caches, dict):
            cond_cache = static_kv_caches.get("cond")
            uncond_cache = static_kv_caches.get("uncond")
        else:
            cond_cache, uncond_cache = static_kv_caches, None
        if guidance_scale <= 1.0 or uncond_context is None:
            return transformer(
                context=context, kv_caches=cond_cache,
                static_kv_cache=True, **kwargs,
                inference_context_cache=cond_context_cache,
                kv_workspaces=cond_workspace,
            )

        # Persistent-KV CFG path.  The cache itself remains branch-specific,
        # but the two branches share one transformer invocation with batch
        # order [unconditional, conditional].  This is numerically the same
        # classifier-free guidance computation while avoiding two sequential
        # launches of every DiT block.
        if cfg_parallel:
            if cond_cache is None or uncond_cache is None:
                raise ValueError(
                    "cfg_parallel static inference requires cond+uncond KV caches"
                )
            if len(cond_cache) != len(uncond_cache):
                raise ValueError("cond/uncond KV cache layer counts differ")

            kw = dict(kwargs)
            x = kw.pop("x")
            traj = kw.pop("traj")
            t = kw.pop("t")
            past_v = kw.pop("past_video_latents", None)
            past_t = kw.pop("past_trajectories", None)
            batch = len(x)
            x2 = list(x) + list(x)
            traj2 = torch.cat([traj, traj], dim=0)
            t2 = torch.cat([t, t], dim=0)
            context2 = list(uncond_context) + list(context)

            def _duplicate_history(items):
                if items is None:
                    return None
                return [torch.cat([item, item], dim=0) for item in items]

            def _duplicate_position_ids(value):
                if value is None:
                    return None
                tensor = torch.as_tensor(value)
                return torch.cat([tensor, tensor], dim=0)

            kv2 = [
                (
                    torch.cat([unc_k, con_k], dim=0),
                    torch.cat([unc_v, con_v], dim=0),
                )
                for (con_k, con_v), (unc_k, unc_v)
                in zip(cond_cache, uncond_cache)
            ]

            # Static KV workspaces are optional.  If present, duplicate them
            # in the same branch order; persistent history normally leaves
            # them disabled to avoid a third full cache copy.
            ws2 = None
            if cond_workspace is not None or uncond_workspace is not None:
                if cond_workspace is None or uncond_workspace is None:
                    raise ValueError("cond/uncond KV workspace presence differs")
                ws2 = [
                    (
                        torch.cat([unc_k, con_k], dim=0),
                        torch.cat([unc_v, con_v], dim=0),
                    )
                    for (con_k, con_v), (unc_k, unc_v)
                    in zip(cond_workspace, uncond_workspace)
                ]

            cond_ctx_cache = cond_context_cache
            uncond_ctx_cache = uncond_context_cache
            if cond_ctx_cache is None or uncond_ctx_cache is None:
                raise ValueError(
                    "cfg_parallel static inference requires cached contexts"
                )
            context_cache2 = {
                "context_emb": torch.cat([
                    uncond_ctx_cache["context_emb"],
                    cond_ctx_cache["context_emb"],
                ], dim=0),
                "cross_attn_kv": [
                    tuple(torch.cat([u, c], dim=0) for u, c in zip(unc, con))
                    for unc, con in zip(
                        uncond_ctx_cache["cross_attn_kv"],
                        cond_ctx_cache["cross_attn_kv"],
                    )
                ],
            }
            kw["x"] = x2
            kw["traj"] = traj2
            kw["t"] = t2
            if past_v is not None:
                kw["past_video_latents"] = _duplicate_history(past_v)
            if past_t is not None:
                kw["past_trajectories"] = _duplicate_history(past_t)
            if "past_chunk_position_ids" in kw:
                kw["past_chunk_position_ids"] = _duplicate_position_ids(
                    kw["past_chunk_position_ids"]
                )
            if "current_chunk_position_id" in kw:
                kw["current_chunk_position_id"] = _duplicate_position_ids(
                    kw["current_chunk_position_id"]
                )
            if kw.get("precomputed_joint_rope_freqs") is not None:
                kw["precomputed_joint_rope_freqs"] = torch.cat(
                    [
                        kw["precomputed_joint_rope_freqs"],
                        kw["precomputed_joint_rope_freqs"],
                    ],
                    dim=0,
                )

            video2, traj2_out = transformer(
                context=context2,
                kv_caches=kv2,
                static_kv_cache=True,
                inference_context_cache=context_cache2,
                kv_workspaces=ws2,
                **kw,
            )
            video_uncond, video_cond = video2[:batch], video2[batch:]
            traj_uncond, traj_cond = traj2_out[:batch], traj2_out[batch:]
            guided_video = video_uncond + guidance_scale * (
                video_cond - video_uncond
            )
            guided_traj = traj_uncond + traj_gs * (traj_cond - traj_uncond)
            return guided_video, guided_traj

        video_uncond, traj_uncond = transformer(
            context=uncond_context, kv_caches=uncond_cache,
            static_kv_cache=True, **kwargs,
            inference_context_cache=uncond_context_cache,
            kv_workspaces=uncond_workspace,
        )
        video_cond, traj_cond = transformer(
            context=context, kv_caches=cond_cache,
            static_kv_cache=True, **kwargs,
            inference_context_cache=cond_context_cache,
            kv_workspaces=cond_workspace,
        )
        guided_video = video_uncond + guidance_scale * (video_cond - video_uncond)
        guided_traj = traj_uncond + traj_gs * (traj_cond - traj_uncond)
        return guided_video, guided_traj

    if cfg_parallel and guidance_scale > 1.0 and uncond_context is not None:
        kw = dict(kwargs)
        x = kw.pop("x")
        traj = kw.pop("traj")
        t = kw.pop("t")
        past_v = kw.pop("past_video_latents", None)
        past_t = kw.pop("past_trajectories", None)
        B = len(x)

        x2 = list(x) + list(x)
        traj2 = torch.cat([traj, traj], dim=0)
        t2 = torch.cat([t, t], dim=0)
        ctx2 = list(uncond_context) + list(context)
        pv2 = [torch.cat([p, p], dim=0) for p in past_v] if past_v else past_v
        pt2 = [torch.cat([p, p], dim=0) for p in past_t] if past_t else past_t
        for position_key in ("past_chunk_position_ids", "current_chunk_position_id"):
            if kw.get(position_key) is not None:
                position_ids = torch.as_tensor(kw[position_key])
                kw[position_key] = torch.cat([position_ids, position_ids], dim=0)
        if kw.get("precomputed_joint_rope_freqs") is not None:
            rope_freqs = kw["precomputed_joint_rope_freqs"]
            kw["precomputed_joint_rope_freqs"] = torch.cat(
                [rope_freqs, rope_freqs], dim=0
            )

        out = transformer(
            x=x2, traj=traj2, t=t2, context=ctx2,
            past_video_latents=pv2, past_trajectories=pt2,
            kv_caches=kv_caches, return_kv_caches=return_kv_caches, **kw,
        )
        if return_kv_caches:
            video2, traj_o2, caches = out
        else:
            video2, traj_o2 = out
            caches = None

        video_uncond, video_cond = video2[:B], video2[B:]
        traj_uncond, traj_cond = traj_o2[:B], traj_o2[B:]
        guided_video = video_uncond + guidance_scale * (video_cond - video_uncond)
        guided_traj = traj_uncond + traj_gs * (traj_cond - traj_uncond)
        if return_kv_caches:
            return guided_video, guided_traj, caches
        return guided_video, guided_traj

    if guidance_scale <= 1.0 or uncond_context is None:
        if return_kv_caches:
            return transformer(
                context=context,
                kv_caches=kv_caches,
                return_kv_caches=True,
                inference_context_cache=cond_context_cache,
                **kwargs,
            )
        return transformer(
            context=context,
            inference_context_cache=cond_context_cache,
            **kwargs,
        )

    if return_kv_caches:
        cond_cache = kv_caches.get("cond") if isinstance(kv_caches, dict) else None
        uncond_cache = kv_caches.get("uncond") if isinstance(kv_caches, dict) else None

        video_uncond, traj_uncond, uncond_cache_out = transformer(
            context=uncond_context,
            kv_caches=uncond_cache,
            return_kv_caches=True,
            inference_context_cache=uncond_context_cache,
            **kwargs,
        )
        video_cond, traj_cond, cond_cache_out = transformer(
            context=context,
            kv_caches=cond_cache,
            return_kv_caches=True,
            inference_context_cache=cond_context_cache,
            **kwargs,
        )
        guided_video = video_uncond + guidance_scale * (video_cond - video_uncond)
        guided_traj = traj_uncond + traj_gs * (traj_cond - traj_uncond)
        return guided_video, guided_traj, {"cond": cond_cache_out, "uncond": uncond_cache_out}

    video_uncond, traj_uncond = transformer(
        context=uncond_context,
        inference_context_cache=uncond_context_cache,
        **kwargs,
    )
    video_cond, traj_cond = transformer(
        context=context,
        inference_context_cache=cond_context_cache,
        **kwargs,
    )
    guided_video = video_uncond + guidance_scale * (video_cond - video_uncond)
    guided_traj = traj_uncond + traj_gs * (traj_cond - traj_uncond)
    return guided_video, guided_traj


def save_rgb_video(frames: np.ndarray, path: Path, fps: int) -> None:
    """Save uint8 RGB frames without overlays for quantitative evaluation."""
    if len(frames) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    reencode_h264(str(path))
    print(f"  Saved raw video → {path}")


def main():
    from awm.habitat_extensions.awm_agent import AWMAgent, AWMAgentConfig
    from awm.utils.deploy_contract import load_deploy_contract
    from awm.scripts.visualize_trajectory import plot_trajectory

    parser = argparse.ArgumentParser(description="ScaleNav image and instruction inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--traj_guidance_scale", type=float, default=1.0)
    parser.add_argument("--action_shift", type=float, default=1.0)
    parser.add_argument("--gpu_memory_mode", default="model_cpu_offload",
                        choices=["model_full_load", "model_cpu_offload"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-video", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    contract = load_deploy_contract(args.model_path)
    rgb = cv2.imread(args.input_image)
    if rgb is None:
        raise ValueError(f"Cannot read image: {args.input_image}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (contract['image_width'], contract['image_height']))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = AWMAgentConfig(
        model_path=args.model_path, pretrained_model_path=args.pretrained_model_path,
        config_path=args.config, device=args.device, dtype=contract['dtype'],
        image_height=contract['image_height'], image_width=contract['image_width'],
        num_video_frames=contract['num_video_frames'],
        ar_num_past_chunks=contract['ar_num_past_chunks'],
        max_attn_chunks=contract['max_attn_chunks'],
        chunk_overlap_frames=contract['chunk_overlap_frames'],
        use_continuous_history=contract['use_continuous_history'],
        traj_normalizer_override=contract['traj_normalizer'], symlog_c_override=contract['symlog_c'],
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        traj_guidance_scale=args.traj_guidance_scale, action_shift=args.action_shift,
        gpu_memory_mode=args.gpu_memory_mode, continuous_actions=True,
        decode_predicted_video=args.decode_video, use_cuda_graph=False,
    )
    agent = AWMAgent(config)
    agent.reset(args.prompt)
    with torch.inference_mode():
        trajectory = agent._infer_continuous_trajectory(rgb)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_image).stem
    np.save(output / f"{stem}_trajectory.npy", trajectory)
    plot_trajectory(trajectory, output / f"{stem}_trajectory.png", rgb)
    if args.decode_video and agent.last_generated_video is not None:
        save_rgb_video(agent.last_generated_video, output / f"{stem}_generated_raw.mp4", fps=4)


if __name__ == "__main__":
    main()
