"""Precomputed-latent dataset for action-free causal video pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

from .base import BaseVideoDataset


class NavT2VCachedDataset(BaseVideoDataset):
    """Load aligned NavT2V chunks produced by precompute_nav_t2v_latents.py.

    The causal phase samples any current chunk n >= 0. Available preceding
    chunks are clean context; at n=0 the past slots are masked padding, so no
    history tokens enter attention. Action tensors remain layout-only zeros.
    """

    def __init__(
        self,
        meta_path: str,
        latent_dir: str,
        action_dim: int = 3,
        max_samples: int | None = None,
        uniform_chunk_sampling: bool = False,
        **base_kwargs: Any,
    ):
        super().__init__(meta_path=meta_path, **base_kwargs)

        if action_dim != 3:
            raise ValueError(f"NavT2VCachedDataset requires action_dim=3, got {action_dim}")
        self.action_dim = int(action_dim)
        self.uniform_chunk_sampling = bool(uniform_chunk_sampling)
        self.latent_dir = Path(latent_dir)
        manifest_path = self.latent_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"NavT2V cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"NavT2V cache is incomplete: {manifest_path}")

        expected = {
            "video_sample_n_frames": self.video_sample_n_frames,
            "video_sample_stride": self.video_sample_stride,
            "video_sample_size": list(self.video_sample_size),
            "chunk_overlap_frames": self.chunk_overlap_frames,
            "chunk_step": self.chunk_step_raw_frames,
        }
        for key, value in expected.items():
            cached = self.manifest.get(key)
            if cached != value:
                raise ValueError(
                    f"NavT2V cache geometry mismatch: manifest {key}={cached}, "
                    f"runtime={value}"
                )

        self.samples: List[Dict[str, Any]] = []
        for sample in self.manifest.get("samples", []):
            n_chunks = int(sample["n_chunks"])
            sample_dir = self.latent_dir / sample["relative_dir"]
            if not sample_dir.is_dir():
                continue
            if not all(
                (sample_dir / f"chunk_{idx:04d}.pt").is_file()
                for idx in range(n_chunks)
            ):
                continue
            self.samples.append({**sample, "sample_dir": sample_dir})
            if max_samples and len(self.samples) >= int(max_samples):
                break

        if not self.samples:
            raise RuntimeError(
                f"NavT2VCachedDataset found no usable samples under {self.latent_dir}"
            )

        self._chunk_index = []
        if self.uniform_chunk_sampling:
            self._chunk_index = [
                (sample_index, chunk_index)
                for sample_index, sample in enumerate(self.samples)
                for chunk_index in range(int(sample["n_chunks"]))
            ]

        print(
            f"[NavT2VCachedDataset] {len(self.samples)} samples/"
            f"{sum(int(s['n_chunks']) for s in self.samples)} chunks, "
            f"step={self.chunk_step_raw_frames}, "
            f"uniform_chunk_sampling={self.uniform_chunk_sampling}, "
            "zero-history samples enabled"
        )

    def __len__(self) -> int:
        return len(self._chunk_index) if self.uniform_chunk_sampling else len(self.samples)


    def _load_chunk(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        return torch.load(
            sample["sample_dir"] / f"chunk_{index:04d}.pt",
            map_location="cpu",
            weights_only=False,
        )

    def _instruction(self, sample: Dict[str, Any]) -> str:
        text = sample["text"]
        if self.text_drop_ratio > 0 and random.random() < self.text_drop_ratio:
            return ""
        return text

    def _zero_trajectory(self) -> torch.Tensor:
        return torch.zeros(
            self.video_sample_n_frames, self.action_dim, dtype=torch.float32
        )


    def _current_index(self, sample: Dict[str, Any]) -> int:
        return random.randint(0, int(sample['n_chunks']) - 1)

    def _sample_and_current_index(self, idx: int) -> tuple[Dict[str, Any], int]:
        if self.uniform_chunk_sampling:
            sample_index, current_index = self._chunk_index[idx]
            return self.samples[sample_index], current_index
        sample = self.samples[idx]
        return sample, self._current_index(sample)

    def _current_dict(
        self, sample: Dict[str, Any], current: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        n = self.video_sample_n_frames
        trajectory = self._zero_trajectory()
        traj_mask = torch.zeros(n, dtype=torch.float32)
        return {
            "latents": current["latent"].float(),
            "inpaint_latent": current["inpaint_latent"].float(),
            "text": self._instruction(sample),
            "trajectory": trajectory,
            "idx": idx,
            "data_type": "latent",
            "is_precomputed": True,
            "has_traj": False,
            "trajectory_loss_mask": traj_mask,
            "video_frame_loss_mask": torch.ones(n, dtype=torch.float32),
        }

    def get_batch(self, idx: int) -> Dict[str, Any]:
        sample, current_idx = self._sample_and_current_index(idx)
        return self._current_dict(sample, self._load_chunk(sample, current_idx), idx)

    def get_multi_chunk_batch(
        self, idx: int, num_past_chunks: int
    ) -> Dict[str, Any]:
        sample, current_idx = self._sample_and_current_index(idx)
        current = self._load_chunk(sample, current_idx)

        n_real = min(current_idx, num_past_chunks)
        n_pad = num_past_chunks - n_real
        past_indices = self._select_past_chunk_indices(
            current_idx, num_past_chunks
        )
        zero_latent = torch.zeros_like(current["latent"], dtype=torch.float32)
        zero_inpaint = torch.zeros_like(
            current["inpaint_latent"], dtype=torch.float32
        )

        past_latents = [zero_latent.clone() for _ in range(n_pad)]
        past_inpaint_latents = [zero_inpaint.clone() for _ in range(n_pad)]
        past_trajectories = [self._zero_trajectory() for _ in range(n_pad)]
        for chunk_idx in past_indices:
            past_chunk = self._load_chunk(sample, chunk_idx)
            past_latents.append(past_chunk["latent"].float())
            past_inpaint_latents.append(past_chunk["inpaint_latent"].float())
            past_trajectories.append(self._zero_trajectory())
        past_mask = torch.zeros(num_past_chunks, dtype=torch.float32)
        past_mask[n_pad:] = 1.0
        past_positions = torch.full(
            (num_past_chunks,), -1, dtype=torch.long
        )
        if past_indices:
            past_positions[n_pad:] = torch.tensor(past_indices, dtype=torch.long)

        result = self._current_dict(sample, current, idx)
        result.update({
            "past_latents": past_latents,
            "past_inpaint_latents": past_inpaint_latents,
            "past_trajectories": past_trajectories,
            "past_chunk_mask": past_mask,
            "past_chunk_position_ids": past_positions,
            "current_chunk_position_id": torch.tensor(
                current_idx, dtype=torch.long
            ),
        })
        return result
