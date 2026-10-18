"""CUDA Graph runtime for joint video/action denoising."""

import copy
import warnings
from contextlib import contextmanager
import weakref
from typing import Dict, List, Optional, Tuple

import torch


@contextmanager
def _capture(graph, device):
    with torch.cuda.device(device), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph):
            yield
    for warning in caught:
        if "CUDA Graph is empty" in str(warning.message):
            raise RuntimeError("CUDA graph captured no kernels")
        warnings.warn(str(warning.message), warning.category, stacklevel=2)


def _concat_kv_branches(
    uncond: List[Tuple[torch.Tensor, torch.Tensor]],
    cond: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate CFG branches in [unconditional, conditional] order."""
    if len(uncond) != len(cond):
        raise ValueError("conditional/unconditional KV layer counts differ")
    return [
        (torch.cat([u_k, c_k], dim=0), torch.cat([u_v, c_v], dim=0))
        for (u_k, u_v), (c_k, c_v) in zip(uncond, cond)
    ]

def _concat_context_caches(
    uncond: Dict[str, object],
    cond: Dict[str, object],
) -> Dict[str, object]:
    """Concatenate cached text/cross-attention features for CFG batching."""
    unc_layers = uncond["cross_attn_kv"]
    cond_layers = cond["cross_attn_kv"]
    if len(unc_layers) != len(cond_layers):
        raise ValueError("conditional/unconditional context layer counts differ")
    return {
        "context_emb": torch.cat(
            [uncond["context_emb"], cond["context_emb"]], dim=0
        ),
        "cross_attn_kv": [
            tuple(torch.cat([u, c], dim=0) for u, c in zip(unc, con))
            for unc, con in zip(unc_layers, cond_layers)
        ],
    }


class JointVideoActionCudaGraphRunner:
    """Replay the joint DiT forward while preserving the denoising loop.

    Clean-history K/V is prepared outside the graph. One fixed-shape joint DiT
    call is captured and replayed per denoise step; both scheduler updates stay
    eager so their original state transitions and numerical behavior are kept.
    """

    def __init__(
        self,
        wrapper: torch.nn.Module,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        context: List[torch.Tensor],
        inference_context_cache: Dict[str, object],
        schedule_cache: Dict[str, object],
        inpaint_latent: torch.Tensor,
        latent_mask: torch.Tensor,
        video_template: torch.Tensor,
        trajectory_template: torch.Tensor,
        seq_len: int,
        scheduler: object,
        trajectory_scheduler: Optional[object],
        num_inference_steps: int,
        num_past_chunks: int,
        uncond_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        uncond_context: Optional[List[torch.Tensor]] = None,
        uncond_inference_context_cache: Optional[Dict[str, object]] = None,
        guidance_scale: float = 1.0,
        traj_guidance_scale: Optional[float] = None,
        warmup_iterations: int = 2,
    ) -> None:
        if not torch.cuda.is_available() or video_template.device.type != "cuda":
            raise ValueError("JointVideoActionCudaGraphRunner requires CUDA tensors")
        if not kv_caches:
            raise ValueError("Joint CUDA graph requires static K/V caches")
        if any(
            layer_kv is None
            for layer_kv in inference_context_cache["cross_attn_kv"]
        ):
            raise ValueError("Every cross-attention layer must expose precomputed K/V")

        self.wrapper = wrapper
        self.guidance_scale = float(guidance_scale)
        self.traj_guidance_scale = float(
            self.guidance_scale
            if traj_guidance_scale is None else traj_guidance_scale
        )
        self.cfg_parallel = self.guidance_scale > 1.0
        if self.cfg_parallel:
            if uncond_kv_caches is None or uncond_context is None:
                raise ValueError(
                    "CFG CUDA graph requires unconditional KV/context inputs"
                )
            if uncond_inference_context_cache is None:
                raise ValueError(
                    "CFG CUDA graph requires unconditional context cache"
                )
            graph_kv_caches = _concat_kv_branches(uncond_kv_caches, kv_caches)
            graph_context = list(uncond_context) + list(context)
            graph_context_cache = _concat_context_caches(
                uncond_inference_context_cache, inference_context_cache
            )
        else:
            graph_kv_caches = kv_caches
            graph_context = list(context)
            graph_context_cache = inference_context_cache
        self.seq_len = int(seq_len)
        self.num_past_chunks = int(num_past_chunks)
        self.prefix_lengths = [cached_k.shape[1] for cached_k, _ in graph_kv_caches]
        self.context = [tensor.clone() for tensor in graph_context]
        self.inference_context_cache = _clone_context_cache(
            graph_context_cache
        )
        self._source_context_ref = None

        current_token_len = self.seq_len + trajectory_template.shape[1]
        self.kv_workspaces = wrapper.prepare_joint_kv_workspaces(
            graph_kv_caches, current_token_len
        )
        self.kv_caches = [
            (full_k[:, :prefix_len], full_v[:, :prefix_len])
            for (full_k, full_v), prefix_len in zip(
                self.kv_workspaces, self.prefix_lengths
            )
        ]

        batch_multiplier = 2 if self.cfg_parallel else 1
        self.inpaint_latent = inpaint_latent.repeat(batch_multiplier, 1, 1, 1, 1)
        self.latent_mask = latent_mask.repeat(batch_multiplier, 1, 1, 1, 1)
        self.video_input = video_template.repeat(batch_multiplier, 1, 1, 1, 1)
        self.trajectory_input = trajectory_template.repeat(batch_multiplier, 1, 1)
        self.current_position = torch.zeros(
            trajectory_template.shape[0] * batch_multiplier,
            device=trajectory_template.device,
            dtype=torch.long,
        )
        self.past_video_placeholders = [
            video_template.repeat(batch_multiplier, 1, 1, 1, 1)
            for _ in range(self.num_past_chunks)
        ]
        self.past_action_placeholders = [
            trajectory_template.repeat(batch_multiplier, 1, 1)
            for _ in range(self.num_past_chunks)
        ]
        self.joint_rope_freqs = schedule_cache["joint_rope_freqs"].repeat(
            batch_multiplier, *([1] * (schedule_cache["joint_rope_freqs"].ndim - 1))
        )
        self.t_tokens = [
            tensor.repeat(batch_multiplier, 1).clone()
            for tensor in schedule_cache["token_timesteps"]
        ]
        self.t_tokens_input = torch.empty_like(self.t_tokens[0])

        self.video_scheduler = copy.deepcopy(scheduler)
        self.trajectory_scheduler = copy.deepcopy(
            trajectory_scheduler if trajectory_scheduler is not None else scheduler
        )
        if not (
            len(self.video_scheduler.timesteps)
            == len(self.trajectory_scheduler.timesteps)
            == int(num_inference_steps)
        ):
            raise ValueError(
                "Joint CUDA graph requires already-initialized aligned schedulers"
            )
        self._load_kv_caches(graph_kv_caches)
        self._load_context(graph_context_cache)

        def forward_body() -> Tuple[torch.Tensor, torch.Tensor]:
            x_cond = (
                (1 - self.latent_mask) * self.inpaint_latent
                + self.latent_mask * self.video_input
            )
            video_out, traj_out = self.wrapper(
                x=[x_cond[index] for index in range(x_cond.shape[0])],
                traj=self.trajectory_input,
                t=self.t_tokens_input,
                context=self.context,
                seq_len=self.seq_len,
                past_video_latents=self.past_video_placeholders,
                past_trajectories=self.past_action_placeholders,
                current_chunk_position_id=self.current_position,
                kv_caches=self.kv_caches,
                static_kv_cache=True,
                is_training=False,
                inference_context_cache=self.inference_context_cache,
                kv_workspaces=self.kv_workspaces,
                precomputed_joint_rope_freqs=self.joint_rope_freqs,
            )
            if self.cfg_parallel:
                video_uncond, video_cond = video_out.chunk(2, dim=0)
                traj_uncond, traj_cond = traj_out.chunk(2, dim=0)
                video_out = video_uncond + self.guidance_scale * (
                    video_cond - video_uncond
                )
                traj_out = traj_uncond + self.traj_guidance_scale * (
                    traj_cond - traj_uncond
                )
            return video_out, traj_out

        device = video_template.device
        self.t_tokens_input.copy_(self.t_tokens[0])
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(1, int(warmup_iterations))):
                warm_video, warm_trajectory = forward_body()
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        self.video_output = torch.empty_like(warm_video)
        self.trajectory_output = torch.empty_like(warm_trajectory)
        self.graph = torch.cuda.CUDAGraph()
        with _capture(self.graph, device):
            graph_video, graph_trajectory = forward_body()
            self.video_output.copy_(graph_video)
            self.trajectory_output.copy_(graph_trajectory)

    @property
    def cache_signature(self) -> Tuple[Tuple[torch.Size, torch.Size], ...]:
        return tuple(
            (cached_k.shape, cached_v.shape)
            for cached_k, cached_v in self.kv_caches
        )

    def matches(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        video: torch.Tensor,
        trajectory: torch.Tensor,
        seq_len: int,
        num_past_chunks: int,
        uncond_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> bool:
        expected_caches = kv_caches
        if self.cfg_parallel:
            if uncond_kv_caches is None:
                return False
            expected_caches = _concat_kv_branches(uncond_kv_caches, kv_caches)
        return (
            int(seq_len) == self.seq_len
            and int(num_past_chunks) == self.num_past_chunks
            and video.shape[0] * (2 if self.cfg_parallel else 1)
            == self.video_input.shape[0]
            and trajectory.shape[0] * (2 if self.cfg_parallel else 1)
            == self.trajectory_input.shape[0]
            and video.shape[1:] == self.video_input.shape[1:]
            and trajectory.shape[1:] == self.trajectory_input.shape[1:]
            and tuple(
                (cached_k.shape, cached_v.shape)
                for cached_k, cached_v in expected_caches
            ) == self.cache_signature
        )

    def _load_kv_caches(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for (cached_k, cached_v), (graph_k, graph_v) in zip(
            kv_caches, self.kv_caches
        ):
            graph_k.copy_(cached_k)
            graph_v.copy_(cached_v)

    def _load_context(
        self,
        inference_context_cache: Dict[str, object],
    ) -> None:
        _copy_context_cache(inference_context_cache, self.inference_context_cache)
        self._source_context_ref = weakref.ref(
            inference_context_cache["context_emb"]
        )

    def __call__(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        inference_context_cache: Dict[str, object],
        inpaint_latent: torch.Tensor,
        video: torch.Tensor,
        trajectory: torch.Tensor,
        current_position: torch.Tensor,
        uncond_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        uncond_inference_context_cache: Optional[Dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg_parallel:
            if uncond_kv_caches is None or uncond_inference_context_cache is None:
                raise ValueError("CFG CUDA graph replay needs unconditional inputs")
            replay_kv_caches = _concat_kv_branches(uncond_kv_caches, kv_caches)
            replay_context_cache = _concat_context_caches(
                uncond_inference_context_cache, inference_context_cache
            )
            replay_position = torch.cat([current_position, current_position], dim=0)
            replay_video = torch.cat([video, video], dim=0)
            replay_trajectory = torch.cat([trajectory, trajectory], dim=0)
            replay_inpaint = torch.cat([inpaint_latent, inpaint_latent], dim=0)
        else:
            replay_kv_caches = kv_caches
            replay_context_cache = inference_context_cache
            replay_position = current_position
            replay_video = video
            replay_trajectory = trajectory
            replay_inpaint = inpaint_latent
        if not self.matches(
            kv_caches, video, trajectory, self.seq_len, self.num_past_chunks,
            uncond_kv_caches=uncond_kv_caches,
        ):
            raise ValueError("Joint CUDA graph inputs no longer match captured shapes")
        self._load_kv_caches(replay_kv_caches)
        if (
            self._source_context_ref is None
            or self._source_context_ref()
            is not replay_context_cache["context_emb"]
        ):
            self._load_context(replay_context_cache)
        self.inpaint_latent.copy_(replay_inpaint)
        self.video_input.copy_(replay_video)
        self.trajectory_input.copy_(replay_trajectory)
        self.current_position.copy_(replay_position)
        self.video_scheduler._step_index = 0
        self.trajectory_scheduler._step_index = 0
        current_video = video
        current_trajectory = trajectory
        base_mask = self.latent_mask[: video.shape[0]]
        base_inpaint = self.inpaint_latent[: video.shape[0]]
        for index, (video_timestep, action_timestep) in enumerate(zip(
            self.video_scheduler.timesteps,
            self.trajectory_scheduler.timesteps,
        )):
            if self.cfg_parallel:
                self.video_input.copy_(torch.cat([current_video, current_video], dim=0))
                self.trajectory_input.copy_(
                    torch.cat([current_trajectory, current_trajectory], dim=0)
                )
            else:
                self.video_input.copy_(current_video)
                self.trajectory_input.copy_(current_trajectory)
            self.t_tokens_input.copy_(self.t_tokens[index])
            self.graph.replay()

            x_cond = (
                (1 - base_mask) * base_inpaint
                + base_mask * current_video
            )
            previous = self.video_scheduler.step(
                self.video_output[:, :current_video.shape[1]],
                video_timestep,
                x_cond,
            ).prev_sample.to(current_video.dtype)
            current_video = (
                (1 - base_mask) * base_inpaint
                + base_mask * previous
            )
            current_trajectory = self.trajectory_scheduler.step(
                self.trajectory_output,
                action_timestep,
                current_trajectory,
            ).prev_sample.to(current_trajectory.dtype)
        return current_video, current_trajectory


def _clone_context_cache(
    inference_context_cache: Dict[str, object],
) -> Dict[str, object]:
    return {
        "context_emb": torch.empty_like(inference_context_cache["context_emb"]),
        "cross_attn_kv": [
            tuple(torch.empty_like(tensor) for tensor in layer_kv)
            for layer_kv in inference_context_cache["cross_attn_kv"]
        ],
    }


def _copy_context_cache(
    source: Dict[str, object],
    target: Dict[str, object],
) -> None:
    target["context_emb"].copy_(source["context_emb"])
    for source_layer, target_layer in zip(
        source["cross_attn_kv"], target["cross_attn_kv"]
    ):
        for source_tensor, target_tensor in zip(source_layer, target_layer):
            target_tensor.copy_(source_tensor)
