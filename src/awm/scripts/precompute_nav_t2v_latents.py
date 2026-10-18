#!/usr/bin/env python
"""Precompute aligned SpatialVID VAE chunks for both Phase-P sub-stages.

Chunk starts are exactly ``chunk_step * n``. With 9 frames, stride 1, and no
overlap, both chunk length and start interval are the requested 9 frames.

The script supports a single process or ``torchrun --nproc_per_node=N``. Each
rank writes disjoint sample directories and a shard manifest; rank zero merges
them into ``manifest.json`` only after every rank completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


from awm.data.datasets.nav_t2v import NavT2VDataset
from awm.models import AutoencoderKLWan3_8
from awm.utils.cache_fingerprint import encoder_fingerprint, records_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_sample_n_frames", type=int, default=9)
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--video_sample_size", nargs=2, type=int, default=[288, 512])
    parser.add_argument("--chunk_overlap_frames", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode_batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional deterministic source-index cap for a cache smoke test.",
    )
    return parser.parse_args()


def distributed_context(device_arg: str) -> Tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_arg)
    return rank, world_size, device


def load_config(path: str) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_source_dataset(args: argparse.Namespace, config: Dict) -> NavT2VDataset:
    source_cfg = next(
        (dict(item) for item in config.get("datasets", []) if item.get("type") == "nav_t2v"),
        None,
    )
    if source_cfg is None:
        raise ValueError("Precompute config must contain one nav_t2v dataset")
    source_cfg.pop("type", None)
    source_cfg.pop("weight", None)
    meta_path = source_cfg.pop("meta_path", "<nav_t2v>")
    source_cfg["shuffle_index"] = False
    if args.max_samples is not None:
        source_cfg["max_samples"] = args.max_samples
    return NavT2VDataset(
        meta_path=meta_path,
        video_sample_size=tuple(args.video_sample_size),
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        chunk_overlap_frames=args.chunk_overlap_frames,
        text_drop_ratio=0.0,
        action_dim=3,
        num_past_chunks=0,
        preserve_first_chunk=False,
        **source_cfg,
    )


def load_vae(
    args: argparse.Namespace, config: Dict, device: torch.device
) -> Tuple[AutoencoderKLWan3_8, torch.dtype]:
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    vae_cfg = config.get("vae_kwargs", {})
    vae_path = str(
        Path(args.pretrained_model_path) / vae_cfg.get("vae_subpath", "Wan2.2_VAE.pth")
    )
    vae = AutoencoderKLWan3_8.from_pretrained(
        vae_path, additional_kwargs=vae_cfg
    )
    vae.eval().requires_grad_(False).to(device, dtype=dtype)
    return vae, dtype


def preprocess_video(frames, size: Tuple[int, int]) -> torch.Tensor:
    """Resize with preserved aspect ratio, center-crop, and normalize to [-1, 1]."""
    target_h, target_w = size
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    _, _, source_h, source_w = tensor.shape
    if target_h / source_h > target_w / source_w:
        resize = (target_h, round(source_w * target_h / source_h))
    else:
        resize = (round(source_h * target_w / source_w), target_w)
    tensor = TF.resize(tensor, resize, InterpolationMode.BILINEAR)
    tensor = TF.center_crop(tensor, [target_h, target_w])
    tensor = tensor * 2.0 - 1.0
    return tensor.permute(1, 0, 2, 3).contiguous()


def atomic_torch_save(payload: Dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def ensure_cache_spec(
    args: argparse.Namespace,
    dataset: NavT2VDataset,
    output_dir: Path,
    rank: int,
    world_size: int,
) -> None:
    """Prevent resume from silently reusing chunks with different geometry."""
    spec = {
        "version": 2,
        "source_records_sha256": records_fingerprint(dataset.samples),
        "encoder": encoder_fingerprint(args.pretrained_model_path, load_config(args.config), args.dtype),
        "source_config": str(Path(args.config).resolve()),
        "source_config_sha256": hashlib.sha256(
            Path(args.config).read_bytes()
        ).hexdigest(),
        "video_sample_n_frames": args.video_sample_n_frames,
        "video_sample_stride": args.video_sample_stride,
        "video_sample_size": list(args.video_sample_size),
        "chunk_overlap_frames": args.chunk_overlap_frames,
        "chunk_raw": dataset.required_raw_frames,
        "chunk_step": dataset.chunk_step_raw_frames,
    }
    spec_path = output_dir / "cache_spec.json"
    if spec_path.is_file():
        existing = json.loads(spec_path.read_text())
        if existing != spec:
            raise RuntimeError(
                f"Cache geometry/source mismatch at {spec_path}. "
                "Use a new empty LATENT_DIR for the 9-frame cache."
            )
    else:
        has_old_artifacts = (
            (output_dir / "samples").exists()
            or (output_dir / "manifest.json").exists()
            or any(output_dir.glob("manifest.rank*.json"))
        )
        if has_old_artifacts:
            raise RuntimeError(
                f"Cache artifacts exist without a geometry spec under {output_dir}. "
                "Use a new empty LATENT_DIR instead of reusing this directory."
            )
        if rank == 0:
            temporary = output_dir / "cache_spec.json.tmp"
            temporary.write_text(json.dumps(spec, indent=2))
            temporary.replace(spec_path)
    if world_size > 1:
        dist.barrier()


def encode_sample(
    args: argparse.Namespace,
    dataset: NavT2VDataset,
    sample_index: int,
    vae: AutoencoderKLWan3_8,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Dict | None:
    sample = dataset.samples[sample_index]
    frames = dataset._load_window(
        sample["video"], sample["t0"], sample["t_end"]
    )
    chunk_raw = dataset.required_raw_frames
    chunk_step = dataset.chunk_step_raw_frames
    n_chunks = 0 if len(frames) < chunk_raw else (len(frames) - chunk_raw) // chunk_step + 1
    if n_chunks < 1:
        return None

    relative_dir = Path("samples") / f"{sample_index:08d}"
    sample_dir = output_dir / relative_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = [sample_dir / f"chunk_{idx:04d}.pt" for idx in range(n_chunks)]

    def valid_chunk(path):
        if not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            return payload['latent'].ndim == 4 and payload['inpaint_latent'].shape[1] == 1
        except (OSError, RuntimeError, KeyError, EOFError):
            return False

    if not all(valid_chunk(path) for path in chunk_paths):
        videos: List[torch.Tensor] = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_step
            raw_window = frames[start:start + chunk_raw]
            ids = torch.linspace(
                0, len(raw_window) - 1, args.video_sample_n_frames
            ).round().long().numpy()
            videos.append(
                preprocess_video(raw_window[ids], tuple(args.video_sample_size))
            )
        for start in range(0, len(videos), args.encode_batch_size):
            video = torch.stack(videos[start:start + args.encode_batch_size]).to(device=device, dtype=dtype)
            first_only = torch.zeros_like(video)
            first_only[:, :, :1] = video[:, :, :1]
            with torch.inference_mode():
                latent = vae.encode(video).latent_dist.sample().to(dtype)
                inpaint = vae.encode(first_only).latent_dist.sample().to(dtype)
            for offset in range(len(video)):
                chunk_idx = start + offset
                atomic_torch_save({
                    'latent': latent[offset].cpu(),
                    'inpaint_latent': inpaint[offset, :, :1].cpu(),
                    'chunk_idx': chunk_idx,
                    'start_frame': chunk_idx * chunk_step,
                }, chunk_paths[chunk_idx])

    return {
        "source_index": sample_index,
        "relative_dir": relative_dir.as_posix(),
        "text": sample["text"],
        "n_chunks": n_chunks,
        "total_frames": len(frames),
        "source_video": sample["video"],
        "source_t0": int(sample["t0"]),
        "source_t_end": int(sample["t_end"]),
    }


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    rank, world_size, device = distributed_context(args.device)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_source_dataset(args, config)
    ensure_cache_spec(args, dataset, output_dir, rank, world_size)
    vae, dtype = load_vae(args, config, device)
    indices = list(range(rank, len(dataset), world_size))
    entries = []
    iterator = tqdm(indices, desc=f"VAE cache rank {rank}", disable=rank != 0)
    for sample_index in iterator:
        try:
            entry = encode_sample(
                args, dataset, sample_index, vae, device, dtype, output_dir
            )
        except Exception as error:
            print(f"[rank {rank}] skip sample {sample_index}: {error}", flush=True)
            continue
        if entry is not None:
            entries.append(entry)

    shard_path = output_dir / f"manifest.rank{rank:03d}.json"
    shard_path.write_text(json.dumps(entries, indent=2))
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged = []
        for shard_rank in range(world_size):
            path = output_dir / f"manifest.rank{shard_rank:03d}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing cache shard manifest: {path}")
            merged.extend(json.loads(path.read_text()))
        merged.sort(key=lambda item: item["source_index"])
        saved_spec = json.loads((output_dir / 'cache_spec.json').read_text())
        manifest = {
            **saved_spec,
            "complete": len(merged) == len(dataset),
            "source_config": str(Path(args.config).resolve()),
            "video_sample_n_frames": args.video_sample_n_frames,
            "video_sample_stride": args.video_sample_stride,
            "video_sample_size": list(args.video_sample_size),
            "chunk_overlap_frames": args.chunk_overlap_frames,
            "chunk_raw": dataset.required_raw_frames,
            "chunk_step": dataset.chunk_step_raw_frames,
            "source_samples": len(dataset),
            "cached_samples": len(merged),
            "cached_chunks": sum(item["n_chunks"] for item in merged),
            "samples": merged,
        }
        temporary = output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(output_dir / "manifest.json")
        print(
            f"[DONE] cached {len(merged)}/{len(dataset)} samples, "
            f"{manifest['cached_chunks']} chunks at step "
            f"{dataset.chunk_step_raw_frames}: {output_dir}"
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    manifest = json.loads((output_dir / 'manifest.json').read_text())
    if not manifest['complete']:
        raise RuntimeError('Cache is incomplete; correct the reported input/encoding errors and rerun')


if __name__ == "__main__":
    main()
