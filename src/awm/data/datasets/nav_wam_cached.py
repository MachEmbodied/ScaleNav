"""Load prepared video/action latents with episode instructions and padding masks."""
from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BaseVideoDataset


class NavWamCachedDataset(BaseVideoDataset):
    """Sample aligned navigation chunks and their observed history."""

    _LEVEL_FIELDS = {
        "l1": "l1_canonical",
        "l2": "l2_canonical",
        "l3": "l3_canonical",
        "l4": "l4_paraphrase",
        "canonical": "canonical",
    }

    def __init__(
        self,
        meta_path: str,                 # original meta.jsonl (for text + filtering)
        latent_dir: str,                # cache directory containing manifest.json + per-chunk .pt
        split_file: Optional[str] = None,
        text_level: str = "canonical",
        action_dim: int = 3,
        chunk_mode: str = "ceil",
        max_samples: Optional[int] = None,
        uniform_chunk_sampling: bool = False,
        **base_kwargs: Any,
    ):
        super().__init__(meta_path=meta_path, **base_kwargs)

        if action_dim != 3:
            raise ValueError(f'action_dim must be 3, got {action_dim}')
        self.action_dim = action_dim
        if chunk_mode not in ("floor", "ceil"):
            raise ValueError(f"chunk_mode must be 'floor' or 'ceil', got {chunk_mode}")
        self.chunk_mode = chunk_mode

        self.uniform_chunk_sampling = bool(uniform_chunk_sampling)

        if text_level == "random":
            self._text_levels = ["l1", "l2", "l3", "l4"]
        elif text_level in self._LEVEL_FIELDS:
            self._text_levels = [text_level]
        else:
            raise ValueError(f"text_level must be 'random' or one of "
                             f"{list(self._LEVEL_FIELDS)}; got {text_level!r}")

        # Load manifest produced by precompute_nav_wam_latents.py
        self.latent_dir = Path(latent_dir)
        manifest_path = self.latent_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"NavWam cache is incomplete: {manifest_path}")
        if int(self.manifest.get("version", 0)) < 2:
            raise RuntimeError(
                f"NavWam cache at {manifest_path} predates exact TI2V inpaint "
                "latents. Rebuild it with scripts/precompute_nav_wam_latents.py."
            )

        # Confirm cache geometry matches CLI knobs
        for k in ("video_sample_n_frames", "video_sample_stride",
                  "chunk_mode", "chunk_overlap_frames"):
            cached = self.manifest.get(k)
            current = getattr(self, k, None)
            # Some live attrs come from base; only verify the cached value
            # against the CLI/script-provided values via base kwargs.
            if cached is not None and current is not None and cached != current:
                raise ValueError(
                    f"NavWam cache geometry mismatch: manifest {k}={cached}, "
                    f"runtime={current}; rebuild the cache with "
                    "scripts/precompute_nav_wam_latents.py under the current config"
                )

        # Read meta.jsonl to assemble (text, episode) lookup + apply split
        wanted_keys = None
        if split_file:
            with open(split_file) as f:
                wanted_keys = {ln.strip() for ln in f if ln.strip()}
        text_map: Dict[str, Dict[str, str]] = {}
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("key")
                if wanted_keys is not None and key not in wanted_keys:
                    continue
                text_map[key] = {
                    "canonical": row.get("canonical", ""),
                    "l1_canonical": row.get("l1_canonical", ""),
                    "l2_canonical": row.get("l2_canonical", ""),
                    "l3_canonical": row.get("l3_canonical", ""),
                    "l4_paraphrase": row.get("l4_paraphrase", ""),
                }

        # Drop episodes whose cache directory is missing
        self.samples: List[Dict[str, Any]] = []
        for ep in self.manifest["episodes"]:
            if wanted_keys is not None and ep["key"] not in wanted_keys:
                continue
            ep_dir = self.latent_dir / ep["scene"] / ep["episode"]
            if not ep_dir.exists():
                continue
            text = text_map.get(ep["key"]) or {"canonical": ep.get("canonical", "")}
            self.samples.append({
                "key": ep["key"],
                "scene": ep["scene"],
                "episode": ep["episode"],
                "n_chunks": int(ep["n_chunks"]),
                "ep_dir": ep_dir,
                **text,
            })

        if not self.samples:
            raise RuntimeError(
                f"NavWamCachedDataset: 0 episodes from latent_dir={latent_dir} "
                f"meta_path={meta_path} split_file={split_file}"
            )

        # --- Cache-coverage check -------------------------------------------
        # A split key with no cached latent is silently skipped above (it is
        # never added to self.samples). If the precompute was incomplete OR run
        # against a different meta/collection than this split, training quietly
        # uses far less data than intended with no other signal. (Real incident:
        # v13_latent was precomputed from a different collection; only 376/1864
        # split keys matched -> the run trained on 20% of the data unnoticed.)
        # Warn loudly here; set NAV_WAM_CACHE_STRICT=1 to make it fatal.
        if wanted_keys is not None:
            got_keys = {s["key"] for s in self.samples}
            missing = wanted_keys - got_keys
            if missing:
                frac = 100.0 * len(missing) / len(wanted_keys)
                msg = (
                    f"[NavWamCachedDataset] CACHE COVERAGE GAP: split "
                    f"'{split_file}' requests {len(wanted_keys)} keys but only "
                    f"{len(got_keys)} have cached latents in '{latent_dir}' -- "
                    f"{len(missing)} MISSING ({frac:.1f}%). Training will "
                    f"SILENTLY use less data than intended. Likely cause: the "
                    f"precompute was incomplete or built from a different "
                    f"meta/collection than this split. Examples of missing keys: "
                    f"{sorted(missing)[:5]}. Re-run "
                    f"scripts/precompute_nav_wam_latents.py with the SAME "
                    f"--meta_path as this split. Set NAV_WAM_CACHE_STRICT=1 to "
                    f"make this fatal."
                )
                warnings.warn(msg)
                print(msg, flush=True)
                if os.environ.get("NAV_WAM_CACHE_STRICT", "0") == "1":
                    raise RuntimeError(msg)

        if max_samples:
            self.samples = self.samples[: int(max_samples)]

        # Index chunks directly so longer episodes contribute proportionally.
        self._chunk_index: List[Tuple[int, int]] = []
        if self.uniform_chunk_sampling:
            self._chunk_index = [
                (sample_index, chunk_index)
                for sample_index, sample in enumerate(self.samples)
                for chunk_index in range(int(sample["n_chunks"]))
            ]

        print(
            f"[NavWamCachedDataset] latent_dir={latent_dir} split={split_file or '<all>'} "
            f"text_level={text_level} action_dim={action_dim} "
            f"=> {len(self.samples)} episodes/{sum(int(s['n_chunks']) for s in self.samples)} chunks "
            f"uniform_chunk_sampling={self.uniform_chunk_sampling}"
        )

    def __len__(self) -> int:
        return len(self._chunk_index) if self.uniform_chunk_sampling else len(self.samples)


    def _instruction_text(self, sample: Dict[str, Any]) -> str:
        level = random.choice(self._text_levels)
        field = self._LEVEL_FIELDS[level]
        text = (sample.get(field) or "").strip()
        if not text:
            text = (sample.get("canonical")
                    or sample.get("l3_canonical") or "").strip()
        return text

    def _load_chunk(self, ep_dir: Path, k: int) -> Dict[str, Any]:
        path = ep_dir / f"chunk_{k:04d}.pt"
        # weights_only=False because we saved a dict of mixed types
        # (tensors + python int/bool/str)
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        if "inpaint_latent" not in chunk:
            raise RuntimeError(
                f"Cached chunk lacks exact inpaint_latent: {path}. Rebuild the cache."
            )
        return chunk

    def _current_index(self, n_chunks: int) -> int:
        return random.randint(0, n_chunks - 1)

    def _sample_and_current_index(self, idx: int) -> Tuple[Dict[str, Any], int]:
        if self.uniform_chunk_sampling:
            sample_index, current_index = self._chunk_index[idx]
            return self.samples[sample_index], current_index
        sample = self.samples[idx]
        return sample, self._current_index(int(sample["n_chunks"]))

    def _build_stop_and_mask(
        self,
        trajectory: torch.Tensor,
        is_last_chunk: bool,
        real_len_out: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask repeated video padding; keep zero-action stop supervision by default."""
        n_out = trajectory.shape[0]


        is_padded_tail = is_last_chunk and 0 <= real_len_out < n_out

        # Video mask always excludes ceil-padded repeats.
        video_mask = torch.ones(n_out, dtype=torch.float32)
        if is_padded_tail:
            video_mask[real_len_out:] = 0.0

        traj_mask = torch.ones(n_out, dtype=torch.float32)
        return trajectory, traj_mask, video_mask

    def get_batch(self, idx: int) -> Dict[str, Any]:
        sample, chunk_idx = self._sample_and_current_index(idx)

        cd = self._load_chunk(sample["ep_dir"], chunk_idx)
        trajectory = cd["trajectory"]
        traj_full, traj_mask, video_mask = self._build_stop_and_mask(
            trajectory, cd["is_last_chunk"], cd["real_len_in_output"],
        )
        instruction = self._instruction_text(sample)
        if self.text_drop_ratio > 0 and random.random() < self.text_drop_ratio:
            instruction = ""

        return {
            "latents": cd["latent"].float(),
            "inpaint_latent": cd["inpaint_latent"].float(),
            "text": instruction,
            "trajectory": traj_full,
            "idx": idx,
            "data_type": "latent",
            "is_precomputed": True,
            "has_traj": True,
            "trajectory_loss_mask": traj_mask,
            "video_frame_loss_mask": video_mask,
        }


    def get_multi_chunk_batch(
        self, idx: int, num_past_chunks: int,
    ) -> Dict[str, Any]:
        sample, curr_chunk_idx = self._sample_and_current_index(idx)
        n_real_past = min(curr_chunk_idx, num_past_chunks)
        n_zero_pad = num_past_chunks - n_real_past
        past_chunk_indices = self._select_past_chunk_indices(
            curr_chunk_idx, num_past_chunks
        )

        # Load current + real past chunks
        chunk_records = []
        for k in past_chunk_indices + [curr_chunk_idx]:
            cd = self._load_chunk(sample["ep_dir"], k)
            traj_full, traj_mask, video_mask = self._build_stop_and_mask(
                cd["trajectory"], cd["is_last_chunk"], cd["real_len_in_output"],
            )
            chunk_records.append({
                "latent": cd["latent"].float(),
                "inpaint_latent": cd["inpaint_latent"].float(),
                "trajectory": traj_full,
                "traj_mask": traj_mask,
                "video_mask": video_mask,
            })

        current = chunk_records[-1]
        zero_latent = torch.zeros_like(current["latent"])
        zero_traj = torch.zeros_like(current["trajectory"])
        past_latents: List[torch.Tensor] = []
        past_inpaint_latents: List[torch.Tensor] = []
        past_trajectories: List[torch.Tensor] = []
        for _ in range(n_zero_pad):
            past_latents.append(zero_latent.clone())
            past_inpaint_latents.append(
                torch.zeros_like(current["inpaint_latent"])
            )
            past_trajectories.append(zero_traj.clone())
        for ci in range(n_real_past):
            past_latents.append(chunk_records[ci]["latent"])
            past_inpaint_latents.append(chunk_records[ci]["inpaint_latent"])
            past_trajectories.append(chunk_records[ci]["trajectory"])

        past_chunk_mask = torch.zeros(num_past_chunks, dtype=torch.float32)
        past_chunk_mask[n_zero_pad:] = 1.0
        past_chunk_position_ids = torch.full(
            (num_past_chunks,), -1, dtype=torch.long
        )
        if past_chunk_indices:
            past_chunk_position_ids[n_zero_pad:] = torch.tensor(
                past_chunk_indices, dtype=torch.long
            )

        instruction = self._instruction_text(sample)
        if self.text_drop_ratio > 0 and random.random() < self.text_drop_ratio:
            instruction = ""

        return {
            "latents": current["latent"],
            "inpaint_latent": current["inpaint_latent"],
            "text": instruction,
            "trajectory": current["trajectory"],
            "idx": idx,
            "data_type": "latent",
            "is_precomputed": True,
            "has_traj": True,
            "past_latents": past_latents,
            "past_inpaint_latents": past_inpaint_latents,
            "past_trajectories": past_trajectories,
            "past_chunk_mask": past_chunk_mask,
            "past_chunk_position_ids": past_chunk_position_ids,
            "current_chunk_position_id": torch.tensor(curr_chunk_idx, dtype=torch.long),
            "trajectory_loss_mask": current["traj_mask"],
            "video_frame_loss_mask": current["video_mask"],
        }
