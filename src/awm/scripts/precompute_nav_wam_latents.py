#!/usr/bin/env python
"""Build a strict, resumable VAE cache for simulated navigation episodes.

Every episode is split on the same 9*n chunk grid used by training and robot
inference. Each chunk stores both the full video latent and the exact latent of
an RGB tensor where only the first frame is present. The latter is required for
TI2V conditioning; slicing temporal slot zero from the full latent is not
equivalent because Wan's VAE is temporally coupled.

The script supports either plain Python or torchrun. A cache is marked complete
only after every source episode has been encoded without error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


from awm.data.datasets.base import BaseVideoDataset
from awm.models import AutoencoderKLWan3_8
from awm.utils.cache_fingerprint import encoder_fingerprint, records_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/inference.yaml"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--video_sample_n_frames", type=int, default=9)
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--video_sample_size", nargs=2, type=int, default=[288, 512])
    parser.add_argument("--chunk_mode", choices=["floor", "ceil"], default="ceil")
    parser.add_argument("--chunk_overlap_frames", type=int, default=0)
    parser.add_argument("--cam_pitch_default", type=float, default=0.0)
    parser.add_argument("--encode_batch_size", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def distributed_context(device_arg: str) -> Tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device(device_arg)
    return rank, world_size, device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(meta_path: Path, split_file: Path | None) -> List[Dict[str, Any]]:
    wanted = None
    if split_file is not None:
        wanted = {line.strip() for line in split_file.read_text().splitlines() if line.strip()}
    rows: List[Dict[str, Any]] = []
    seen = set()
    for line_number, line in enumerate(meta_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if not key:
            raise ValueError(f"{meta_path}:{line_number}: missing key")
        if key in seen:
            raise ValueError(f"{meta_path}:{line_number}: duplicate key {key!r}")
        seen.add(key)
        if wanted is not None and key not in wanted:
            continue
        if not row.get("ok_canonical_ends_with_goal", True):
            continue
        for field in ("scene", "episode", "video"):
            if not row.get(field):
                raise ValueError(f"{meta_path}:{line_number}: {key!r} missing {field}")
        if not row.get("actions") and not row.get("trajectory"):
            raise ValueError(
                f"{meta_path}:{line_number}: {key!r} needs actions or trajectory"
            )
        rows.append(row)
    if wanted is not None:
        selected = {row["key"] for row in rows}
        missing = wanted - selected
        if missing:
            raise ValueError(
                f"split {split_file} contains {len(missing)} unavailable keys; "
                f"examples={sorted(missing)[:5]}"
            )
    if not rows:
        raise ValueError("No usable episodes were selected")
    cache_locations: Dict[Tuple[str, str], str] = {}
    for row in rows:
        location = (str(row["scene"]), str(row["episode"]))
        previous = cache_locations.setdefault(location, str(row["key"]))
        if previous != row["key"]:
            raise ValueError(
                "cache path collision for scene/episode {}: {!r} and {!r}".format(
                    location, previous, row["key"]
                )
            )
    return rows


def load_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    return np.stack(frames)


def load_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    with np.load(path) as payload:
        if "position" not in payload or "quat_wxyz" not in payload:
            raise ValueError(f"trajectory lacks position/quat_wxyz: {path}")
        position = np.asarray(payload["position"], dtype=np.float64)
        quaternion = np.asarray(payload["quat_wxyz"], dtype=np.float64)
        actions = (
            np.asarray(payload["action_delta_body"], dtype=np.float32)
            if "action_delta_body" in payload
            else None
        )
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"invalid position shape {position.shape}: {path}")
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"invalid quat_wxyz shape {quaternion.shape}: {path}")
    if len(position) != len(quaternion):
        raise ValueError(f"pose array length mismatch: {path}")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError(f"non-finite pose values: {path}")
    if actions is not None:
        if actions.shape != (len(position) - 1, 3):
            raise ValueError(
                f"action_delta_body must have shape {(len(position) - 1, 3)}, "
                f"got {actions.shape}: {path}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"non-finite action_delta_body values: {path}")
    return np.concatenate([position, quaternion], axis=1), actions


def load_actions(path: Path) -> np.ndarray:
    actions = np.asarray(np.load(path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"actions must have shape [N,3], got {actions.shape}: {path}")
    if not np.isfinite(actions).all():
        raise ValueError(f"non-finite actions: {path}")
    return actions


def preprocess_video(frames: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
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


def load_pitch(
    row: Dict[str, Any], data_root: Path, video_path: Path, default: float
) -> float:
    candidates = (
        data_root / row["scene"] / row["episode"] / "episode.json",
        video_path.parent / "episode.json",
    )
    for path in candidates:
        try:
            metadata = json.loads(path.read_text())
            pitch = metadata.get(
                "robot_cam_pitch_deg", metadata.get("cam_pitch_deg", default)
            )
            return float(pitch)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return default


class _TrajectoryHelper(BaseVideoDataset):
    def __len__(self) -> int:
        return 0

    def get_batch(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


def trajectory_helper(n_frames: int, stride: int) -> _TrajectoryHelper:
    helper = _TrajectoryHelper.__new__(_TrajectoryHelper)
    helper.video_sample_n_frames = n_frames
    helper.video_sample_stride = stride
    return helper


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def valid_cached_chunk(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return all(
            key in payload
            for key in (
                "latent",
                "inpaint_latent",
                "trajectory",
                "real_len_in_output",
                "is_last_chunk",
                "chunk_idx",
            )
        ) and payload["inpaint_latent"].shape[1] == 1
    except Exception:
        return False


def load_vae(
    args: argparse.Namespace, config: Dict[str, Any], device: torch.device
) -> Tuple[AutoencoderKLWan3_8, torch.dtype]:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    vae_config = config.get("vae_kwargs", {})
    path = str(
        Path(args.pretrained_model_path)
        / vae_config.get("vae_subpath", "Wan2.2_VAE.pth")
    )
    vae = AutoencoderKLWan3_8.from_pretrained(path, additional_kwargs=vae_config)
    vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return vae, dtype


def cache_spec(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_raw = (args.video_sample_n_frames - 1) * args.video_sample_stride + 1
    chunk_step = chunk_raw - args.chunk_overlap_frames
    return {
        "version": 3,
        "source_records_sha256": records_fingerprint(rows),
        "encoder": encoder_fingerprint(args.pretrained_model_path,
            OmegaConf.to_container(OmegaConf.load(args.config), resolve=True), args.dtype),
        "cam_pitch_default": args.cam_pitch_default,
        "source_meta": str(Path(args.meta_path).resolve()),
        "source_meta_sha256": sha256_file(Path(args.meta_path)),
        "source_split": str(Path(args.split_file).resolve()) if args.split_file else None,
        "source_split_sha256": sha256_file(Path(args.split_file)) if args.split_file else None,
        "data_root": str(Path(args.data_root).resolve()),
        "source_episodes": len(rows),
        "pretrained_model_path": str(Path(args.pretrained_model_path).resolve()),
        "video_sample_n_frames": args.video_sample_n_frames,
        "video_sample_stride": args.video_sample_stride,
        "video_sample_size": list(args.video_sample_size),
        "chunk_mode": args.chunk_mode,
        "chunk_overlap_frames": args.chunk_overlap_frames,
        "chunk_raw": chunk_raw,
        "chunk_step": chunk_step,
    }


def ensure_spec(
    spec: Dict[str, Any], output_dir: Path, rank: int, world_size: int
) -> None:
    path = output_dir / "cache_spec.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != spec:
            raise RuntimeError(
                f"Cache source/geometry mismatch at {path}; use a new output directory"
            )
    elif rank == 0:
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"Cache artifacts exist without cache_spec.json under {output_dir}"
            )
        temporary = output_dir / "cache_spec.json.tmp"
        temporary.write_text(json.dumps(spec, indent=2))
        temporary.replace(path)
    if world_size > 1:
        dist.barrier(device_ids=[rank])


def complete_cache_matches(
    output_dir: Path, spec: Dict[str, Any], rows: List[Dict[str, Any]]
) -> bool:
    """Fast-path a completed cache without loading one VAE per rank."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete", False):
            return False
        if any(manifest.get(key) != value for key, value in spec.items()):
            return False
        episodes = manifest.get("episodes", [])
        if [item.get("key") for item in episodes] != [row.get("key") for row in rows]:
            return False
        for episode in episodes:
            episode_dir = output_dir / episode["scene"] / episode["episode"]
            if any(
                not (episode_dir / f"chunk_{index:04d}.pt").is_file()
                for index in range(int(episode["n_chunks"]))
            ):
                return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def chunk_count(total: int, chunk_raw: int, chunk_step: int, mode: str) -> int:
    if total < 1:
        return 0
    if mode == "floor":
        return max(1, (total - chunk_raw) // chunk_step + 1)
    return max(1, math.ceil(max(0, total - chunk_raw) / chunk_step) + 1)


def encode_episode(
    args: argparse.Namespace,
    row: Dict[str, Any],
    source_index: int,
    vae: AutoencoderKLWan3_8,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    data_root = Path(args.data_root)
    video_path = data_root / row["video"]
    frames = load_video(video_path)
    actions = None
    poses = None
    if row.get("actions"):
        actions = load_actions(data_root / row["actions"])
        if len(actions) not in (len(frames) - 1, len(frames)):
            raise ValueError(
                f"video/action length mismatch for {row['key']}: "
                f"{len(frames)} vs {len(actions)}"
            )
        if len(actions) == len(frames) and not np.allclose(actions[-1], 0, atol=1e-7):
            raise ValueError('An N-row action file for N frames must end with a zero stop action')
    else:
        trajectory_path = data_root / row["trajectory"]
        poses, actions = load_trajectory(trajectory_path)
        if len(frames) != len(poses):
            raise ValueError(
                f"video/trajectory length mismatch for {row['key']}: "
                f"{len(frames)} vs {len(poses)}"
            )
    if actions is not None and args.video_sample_stride != 1:
        raise ValueError('Explicit body-frame deltas require video_sample_stride=1')
    if len(frames) < 2:
        raise ValueError(
            f"episode {row['key']} has {len(frames)} frames, needs at least "
            "2 frames for an observed transition"
        )

    chunk_raw = (args.video_sample_n_frames - 1) * args.video_sample_stride + 1
    chunk_step = chunk_raw - args.chunk_overlap_frames
    n_chunks = chunk_count(len(frames), chunk_raw, chunk_step, args.chunk_mode)
    episode_dir = output_dir / row["scene"] / row["episode"]
    episode_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = [episode_dir / f"chunk_{index:04d}.pt" for index in range(n_chunks)]

    helper = trajectory_helper(args.video_sample_n_frames, args.video_sample_stride)
    pitch = load_pitch(row, data_root, video_path, args.cam_pitch_default)
    pending: List[Tuple[int, torch.Tensor, torch.Tensor, int]] = []

    def flush() -> None:
        if not pending:
            return
        video = torch.stack([item[1] for item in pending]).to(device=device, dtype=dtype)
        first_only = torch.zeros_like(video)
        first_only[:, :, 0:1] = video[:, :, 0:1]
        with torch.inference_mode():
            latents = vae.encode(video).latent_dist.sample().to(dtype)
            inpaint = vae.encode(first_only).latent_dist.sample().to(dtype)
        for batch_index, (chunk_index, _, trajectory, real_len) in enumerate(pending):
            atomic_torch_save(
                {
                    "latent": latents[batch_index].cpu(),
                    "inpaint_latent": inpaint[batch_index, :, 0:1].cpu(),
                    "trajectory": trajectory,
                    "real_len_in_output": int(real_len),
                    "is_last_chunk": bool(chunk_index == n_chunks - 1),
                    "chunk_idx": int(chunk_index),
                    "start_frame": int(chunk_index * chunk_step),
                },
                chunk_paths[chunk_index],
            )
        pending.clear()

    for chunk_index, path in enumerate(chunk_paths):
        if valid_cached_chunk(path):
            continue
        start = chunk_index * chunk_step
        raw = frames[start : start + chunk_raw]
        real_raw = len(raw)
        if real_raw < chunk_raw:
            if args.chunk_mode != "ceil":
                start = max(0, len(frames) - chunk_raw)
                raw = frames[start : start + chunk_raw]
                real_raw = len(raw)
            else:
                raw = np.concatenate(
                    [raw, np.repeat(raw[-1:], chunk_raw - real_raw, axis=0)], axis=0
                )
        frame_ids = np.arange(0, chunk_raw, args.video_sample_stride)[
            : args.video_sample_n_frames
        ]
        sampled = raw[frame_ids]
        real_len = int(sum(index < real_raw for index in frame_ids))
        if actions is not None:
            trajectory = torch.zeros(
                args.video_sample_n_frames, 3, dtype=torch.float32
            )
            source_ids = start + frame_ids
            valid = source_ids < len(actions)
            if valid.any():
                trajectory[torch.from_numpy(valid)] = torch.from_numpy(
                    actions[source_ids[valid]]
                )
        else:
            trajectory = helper._compute_trajectory(
                poses, start_idx=start, clip_length=chunk_raw, pitch=pitch
            )
        pending.append(
            (
                chunk_index,
                preprocess_video(sampled, tuple(args.video_sample_size)),
                trajectory,
                real_len,
            )
        )
        if len(pending) >= args.encode_batch_size:
            flush()
    flush()

    if not all(valid_cached_chunk(path) for path in chunk_paths):
        raise RuntimeError(f"cache verification failed for {row['key']}")
    return {
        "source_index": source_index,
        "key": row["key"],
        "scene": row["scene"],
        "episode": row["episode"],
        "n_chunks": n_chunks,
        "total_frames": len(frames),
        "source_video": row["video"],
        "source_trajectory": row.get("trajectory"),
        "source_actions": row.get("actions"),
        "canonical": row.get("canonical", ""),
        "l1_canonical": row.get("l1_canonical", ""),
        "l2_canonical": row.get("l2_canonical", ""),
        "l3_canonical": row.get("l3_canonical", ""),
        "l4_paraphrase": row.get("l4_paraphrase", ""),
    }


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1:
        raise ValueError("--encode_batch_size must be positive")
    chunk_raw = (args.video_sample_n_frames - 1) * args.video_sample_stride + 1
    if not 0 <= args.chunk_overlap_frames < chunk_raw:
        raise ValueError("chunk_overlap_frames must be smaller than the raw chunk span")

    rank, world_size, device = distributed_context(args.device)
    rows = read_rows(
        Path(args.meta_path), Path(args.split_file) if args.split_file else None
    )
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    spec = cache_spec(args, rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_spec(spec, output_dir, rank, world_size)

    cache_complete = torch.zeros(1, dtype=torch.uint8, device=device)
    if rank == 0 and complete_cache_matches(output_dir, spec, rows):
        cache_complete.fill_(1)
    if world_size > 1:
        dist.broadcast(cache_complete, src=0)
    if cache_complete.item():
        if rank == 0:
            print(f"[SKIP] completed cache already matches request: {output_dir}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    if rank == 0:
        incomplete = {**spec, "complete": False, "episodes": []}
        (output_dir / "manifest.json").write_text(json.dumps(incomplete, indent=2))
    if world_size > 1:
        dist.barrier(device_ids=[rank])

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    vae, dtype = load_vae(args, config, device)
    entries = []
    errors = []
    indices = list(range(rank, len(rows), world_size))
    iterator = tqdm(indices, desc=f"action VAE cache rank {rank}", disable=rank != 0)
    for source_index in iterator:
        try:
            entries.append(
                encode_episode(
                    args,
                    rows[source_index],
                    source_index,
                    vae,
                    dtype,
                    device,
                    output_dir,
                )
            )
        except Exception as error:
            record = {
                "source_index": source_index,
                "key": rows[source_index].get("key"),
                "error": f"{type(error).__name__}: {error}",
            }
            errors.append(record)
            print(f"[rank {rank}] ERROR {record}", flush=True)

    shard = {"entries": entries, "errors": errors}
    (output_dir / f"manifest.rank{rank:03d}.json").write_text(
        json.dumps(shard, indent=2)
    )
    error_count = torch.tensor([len(errors)], device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(error_count, op=dist.ReduceOp.SUM)
        dist.barrier(device_ids=[rank])

    if rank == 0:
        merged_entries = []
        merged_errors = []
        for shard_rank in range(world_size):
            payload = json.loads(
                (output_dir / f"manifest.rank{shard_rank:03d}.json").read_text()
            )
            merged_entries.extend(payload["entries"])
            merged_errors.extend(payload["errors"])
        merged_entries.sort(key=lambda item: item["source_index"])
        manifest = {
            **spec,
            "complete": not merged_errors and len(merged_entries) == len(rows),
            "cached_episodes": len(merged_entries),
            "cached_chunks": sum(item["n_chunks"] for item in merged_entries),
            "episodes": merged_entries,
            "errors": merged_errors,
        }
        temporary = output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(output_dir / "manifest.json")
        print(
            f"[DONE] complete={manifest['complete']} episodes="
            f"{len(merged_entries)}/{len(rows)} chunks={manifest['cached_chunks']} "
            f"output={output_dir}"
        )

    if world_size > 1:
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()
    if int(error_count.item()) or (rank == 0 and len(entries) != len(rows) and world_size == 1):
        raise SystemExit("Action VAE cache is incomplete; inspect manifest.json errors")


if __name__ == "__main__":
    main()
