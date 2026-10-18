"""Shared chunk geometry, history selection, and trajectory transforms."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseVideoDataset(Dataset, ABC):
    """Common geometry for source videos and cached navigation chunks."""

    def __init__(
        self,
        meta_path: str,
        video_sample_size: Union[int, Tuple[int, int]] = 256,
        video_sample_stride: int = 4,
        video_sample_n_frames: int = 17,
        text_drop_ratio: float = 0.0,
        num_past_chunks: int = 0,
        chunk_overlap_frames: int = 0,
        preserve_first_chunk: bool = False,
    ):
        super().__init__()
        self.meta_path = meta_path
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.required_raw_frames = (video_sample_n_frames - 1) * video_sample_stride + 1
        self.min_required_frames = video_sample_n_frames
        self.text_drop_ratio = text_drop_ratio
        self.num_past_chunks = num_past_chunks
        self.chunk_overlap_frames = int(chunk_overlap_frames)
        self.preserve_first_chunk = bool(preserve_first_chunk)

        if isinstance(video_sample_size, int):
            self.video_sample_size = (video_sample_size, video_sample_size)
        else:
            self.video_sample_size = tuple(video_sample_size)

        if self.chunk_overlap_frames < 0 or self.chunk_overlap_frames >= self.required_raw_frames:
            raise ValueError(
                f"chunk_overlap_frames must be in [0, {self.required_raw_frames - 1}], "
                f"got {self.chunk_overlap_frames}"
            )
        # Raw-frame shift between adjacent chunks in multi-chunk sampling.
        self.chunk_step_raw_frames = self.required_raw_frames - self.chunk_overlap_frames


    def _select_past_chunk_indices(
        self, current_chunk_idx: int, num_past_chunks: int
    ) -> List[int]:
        """Choose causal history, optionally reserving slot zero for the anchor."""
        n_real = min(current_chunk_idx, num_past_chunks)
        if (
            self.preserve_first_chunk
            and current_chunk_idx > num_past_chunks
            and num_past_chunks > 0
        ):
            indices = [0]
            if num_past_chunks > 1:
                indices.extend(range(
                    current_chunk_idx - (num_past_chunks - 1), current_chunk_idx
                ))
            return indices
        return list(range(current_chunk_idx - n_real, current_chunk_idx))


    @abstractmethod
    def get_batch(self, idx: int) -> Dict[str, Any]:
        """Load and return a single sample dict."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_retries = 10
        for _ in range(max_retries):
            try:
                if self.num_past_chunks > 0:
                    return self.get_multi_chunk_batch(idx, self.num_past_chunks)
                return self.get_batch(idx)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error on sample {idx}: {e}")
                idx = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[{self.__class__.__name__}] Failed to load a sample after {max_retries} retries"
        )


    def get_multi_chunk_batch(self, idx: int, num_past_chunks: int) -> Dict[str, Any]:
        raise NotImplementedError('History sampling requires a prepared latent dataset')

    def _compute_trajectory(
        self,
        poses: np.ndarray,
        start_idx: int,
        clip_length: int,
        pitch: float = 0.0,
    ) -> torch.Tensor:
        """
        One action per sampled frame: (dx, dy, dyaw) from frame i → frame i+1.

        Args:
            poses     : (N, 4, 4) camera-to-world matrices for the full video.
            start_idx : First frame index in the original (un-strided) video.
            clip_length: Number of consecutive original frames in the clip.
            pitch     : Camera pitch angle in degrees.

        Returns:
            Tensor of shape (num_sampled_frames, 3).
        """
        if poses.ndim == 2 and poses.shape[1] == 7:
            poses = self._pose_vec_to_matrix(poses)

        # Match the frame indices used to sample the video.
        n_target = self.video_sample_n_frames
        stride = max(1, (clip_length - 1) // (n_target - 1)) if n_target > 1 else 1
        frame_ids = np.arange(start_idx, start_idx + (n_target - 1) * stride + 1, stride)
        n = len(frame_ids)
        # Include the transition after the final sampled frame when available.
        next_ids = np.empty(n, dtype=int)
        next_ids[:-1] = frame_ids[1:]
        if n > 0:
            next_ids[-1] = min(int(frame_ids[-1]) + stride, len(poses) - 1)
        deltas = []

        for i in range(n):
            cur = min(int(frame_ids[i]), len(poses) - 1)
            nxt = min(int(next_ids[i]),  len(poses) - 1)
            if cur == nxt:
                deltas.append(np.zeros(3, dtype=np.float32))
                continue
            rel = self._get_trajectory_relative_to_frame(poses[cur:nxt + 1], pitch)
            delta = rel[-1, :3].astype(np.float32) if len(rel) > 1 else np.zeros(3, dtype=np.float32)
            deltas.append(delta)

        traj = np.array(deltas, dtype=np.float32)   # (T_frames, 3)
        return torch.tensor(traj, dtype=torch.float32)

    def _pose_vec_to_matrix(self, poses: np.ndarray) -> np.ndarray:
        """Convert (N, 7) pose vectors [x,y,z,qw,qx,qy,qz] → (N, 4, 4)."""
        from scipy.spatial.transform import Rotation

        matrices = []
        for pose in poses:
            pos = pose[:3]
            quat = pose[3:]  # [qw, qx, qy, qz]
            rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy [x,y,z,w]
            mat = np.eye(4)
            mat[:3, :3] = rot.as_matrix()
            mat[:3, 3] = pos
            matrices.append(mat)
        return np.array(matrices)

    def _get_trajectory_relative_to_frame(
        self,
        extrinsics: np.ndarray,
        camera_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Transform camera c2w matrices to robot-frame (x, y, yaw) relative to
        the first element (current frame).
        """
        T_camera2robot = np.array([
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [1.0,  0.0,  0.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ])
        T_robot2camera = np.linalg.inv(T_camera2robot)

        if camera_deg != 0:
            rad = np.radians(camera_deg)
            T_deg = np.array([
                [1.0,           0.0,            0.0, 0.0],
                [0.0,  np.cos(-rad), -np.sin(-rad), 0.0],
                [0.0,  np.sin(-rad),  np.cos(-rad), 0.0],
                [0.0,           0.0,            0.0, 1.0],
            ])
            T_robot2camera = T_robot2camera @ T_deg
            T_camera2robot = np.linalg.inv(T_robot2camera)

        extrinsics_robot = extrinsics @ T_camera2robot
        T_ref_inv = np.linalg.inv(extrinsics_robot[0])
        relative = T_ref_inv @ extrinsics_robot

        translations = relative[:, :2, 3]
        yaws = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        return np.column_stack([translations, yaws])
