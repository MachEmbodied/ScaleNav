"""Read instruction-aligned source video segments for latent preparation."""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .base import BaseVideoDataset


class NavT2VDataset(BaseVideoDataset):
    def __init__(
        self,
        domains: Optional[List[Dict[str, str]]] = None,
        perclip_dir: Optional[str] = None,
        pool_root: Optional[str] = None,
        meta_jsonl: Optional[str] = None,
        data_root: Optional[str] = None,
        meta_path: str = "",
        action_dim: int = 3,
        min_confidence: str = "B",       # keep confidence >= this (A best, B ok)
        max_samples: Optional[int] = None,
        shuffle_index: bool = True,
        index_seed: int = 0,
        aligned_sampling: bool = False,
        **base_kwargs,
    ):
        super().__init__(meta_path=meta_path or "<nav_t2v>", **base_kwargs)

        self.action_dim = int(action_dim)
        self.aligned_sampling = bool(aligned_sampling)
        self._conf_rank = {"A": 2, "B": 1, "C": 0}
        min_rank = self._conf_rank.get(min_confidence.upper(), 1)

        if domains is None:
            if perclip_dir and pool_root:
                domains = [{"perclip_dir": perclip_dir, "pool_root": pool_root}]
            elif meta_jsonl:
                domains = [{"meta_jsonl": meta_jsonl, "data_root": data_root}]
            else:
                raise AssertionError(
                    "NavT2VDataset needs `domains`, or (perclip_dir, pool_root), "
                    "or (meta_jsonl, data_root)"
                )

        self.samples: List[Dict[str, Any]] = []
        self._need = self.min_required_frames
        for dom in domains:
            if dom.get("meta_jsonl"):
                self._index_meta_jsonl(dom, max_samples)
            else:
                self._index_perclip(dom, min_rank, shuffle_index, index_seed, max_samples)
            if max_samples and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(
                "NavT2VDataset built an empty index — check the domain paths "
                f"(perclip_dir/pool_root or meta_jsonl/data_root). domains={domains}"
            )


    def _index_perclip(self, dom, min_rank, shuffle_index, index_seed, max_samples):
        """Index a perclip harvest domain {perclip_dir, pool_root}: one json per
        source clip, each holding a LIST of go-to segments."""
        pdir = dom["perclip_dir"]
        vids = os.path.join(dom["pool_root"], "videos")
        files = sorted(glob.glob(os.path.join(pdir, "*.json")))
        if shuffle_index:
            random.Random(index_seed).shuffle(files)
        for jf in files:
            try:
                segs = json.load(open(jf))
            except Exception:
                continue
            for s in segs:
                if self._conf_rank.get(str(s.get("confidence", "B")).upper(), 1) < min_rank:
                    continue
                t0, t_end = int(s.get("t0", 0)), int(s.get("t_end", 0))
                if t_end - t0 < self._need:
                    continue
                mp4 = os.path.join(vids, f"{s.get('clip')}.mp4")
                if not os.path.exists(mp4):
                    continue
                self.samples.append({"video": mp4, "t0": t0, "t_end": t_end,
                                     "text": s.get("instruction", ""),
                                     "segment_id": s.get("segment_id")})
                if max_samples and len(self.samples) >= max_samples:
                    return
            if max_samples and len(self.samples) >= max_samples:
                return

    def _index_meta_jsonl(self, dom, max_samples):
        """Index a per-episode meta.jsonl domain {meta_jsonl, data_root}: one
        (video, instruction) row per line (SVN nav-episode layout). Each episode
        video IS one go-to segment, so the whole clip is the window [0, n_frames)."""
        mpath = dom["meta_jsonl"]
        droot = dom.get("data_root") or os.path.dirname(os.path.abspath(mpath))
        with open(mpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rel = row.get("video")
                if not rel:
                    continue
                mp4 = rel if os.path.isabs(rel) else os.path.join(droot, rel)
                # Drop segments too short for a FIXED-stride chunk (need
                # required_raw_frames, not just n output frames) so get_batch
                # never has to shrink the stride. n_frames is at the CUT fps.
                n = int(row.get("n_frames", 0))
                if n and n < self.required_raw_frames:
                    continue
                if not os.path.exists(mp4):
                    continue
                text = row.get("instruction") or row.get("canonical") or ""
                self.samples.append({"video": mp4, "t0": 0, "t_end": n, "text": text})
                if max_samples and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def _load_window(self, path: str, t0: int, t_end: int) -> np.ndarray:
        """Decode frames [t0, t_end) of a clip → (n, H, W, C) uint8 RGB."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        # Seek to t0 (fall back to sequential if the codec ignores the seek).
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, t0)))
        frames: List[np.ndarray] = []
        want = t_end - t0
        for _ in range(want):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < self.min_required_frames:
            # Seek unreliable / clip shorter than annotated — read all & slice.
            cap = cv2.VideoCapture(path)
            allf: List[np.ndarray] = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                allf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            sl = allf[t0:t_end] if t_end <= len(allf) else allf[t0:]
            frames = sl
        if len(frames) < self.min_required_frames:
            raise ValueError(
                f"window [{t0}:{t_end}] of {path} has {len(frames)} < "
                f"{self.min_required_frames} frames"
            )
        return np.stack(frames, axis=0)

    def get_batch(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        frames = self._load_window(sample["video"], sample["t0"], sample["t_end"])

        # Keep the sampling stride fixed to preserve the training frame rate.
        n = self.video_sample_n_frames
        total = len(frames)
        raw = self.required_raw_frames
        if total < raw:
            raise ValueError(
                f"{sample['video']} decoded {total} < {raw} frames for fixed "
                f"stride {self.video_sample_stride} — skip (real-time consistency)"
            )
        if self.aligned_sampling:
            n_chunks = (total - raw) // self.chunk_step_raw_frames + 1
            start_idx = random.randint(0, n_chunks - 1) * self.chunk_step_raw_frames
        else:
            start_idx = random.randint(0, total - raw)
        ids = np.linspace(start_idx, start_idx + raw - 1, n).round().astype(int)
        pixel_values = frames[ids]  # (n, H, W, C) uint8 @ exactly src_fps/stride
        trajectory = torch.zeros(n, self.action_dim, dtype=torch.float32)
        traj_mask = torch.zeros(n, dtype=torch.float32)
        video_mask = torch.ones(n, dtype=torch.float32)

        instruction = sample["text"]
        if self.text_drop_ratio > 0 and random.random() < self.text_drop_ratio:
            instruction = ""

        return {
            "pixel_values": pixel_values,
            "text": instruction,
            "trajectory": trajectory,
            "idx": idx,
            "data_type": "video",
            "has_traj": False,
            "trajectory_loss_mask": traj_mask,
            "video_frame_loss_mask": video_mask,
        }
