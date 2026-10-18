import copy

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from awm.utils.action_cuda_graph import JointVideoActionCudaGraphRunner


class JointForward:
    def prepare_joint_kv_workspaces(self, caches, current_token_len):
        return [tuple(torch.cat([value, torch.zeros_like(value[:, :1]).expand(
            -1, current_token_len, -1)], dim=1) for value in pair) for pair in caches]

    def __call__(self, *, x, traj, t, inference_context_cache, kv_caches, **kwargs):
        video = torch.stack(x)
        context = inference_context_cache['context_emb'].mean(dim=1)
        history = kv_caches[0][0].mean(dim=(1, 2))
        bias = context + history
        return (0.1 * video + bias[:, None, None, None, None] + t[:, :1, None, None, None] / 1000,
                0.2 * traj + bias[:, None, None] + t[:, -1:, None] / 1000)


def context_cache(value):
    value = torch.tensor([[value]], device='cuda')
    return {'context_emb': value, 'cross_attn_kv': [(value, value)]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires a GPU')
def test_cfg_graph_replay_matches_separate_branches_and_modality_schedules():
    wrapper = JointForward()
    video = torch.ones((1, 1, 1, 1, 2), device='cuda')
    trajectory = torch.ones((1, 2, 3), device='cuda')
    anchor = torch.full_like(video, 0.5)
    mask = torch.tensor([0., 1.], device='cuda').reshape_as(video)
    cond_kv = [(torch.ones(1, 2, 1, device='cuda'), torch.zeros(1, 2, 1, device='cuda'))]
    unc_kv = [(torch.zeros(1, 2, 1, device='cuda'), torch.zeros(1, 2, 1, device='cuda'))]
    cond, unc = context_cache(2.), context_cache(0.)
    video_scheduler = FlowMatchEulerDiscreteScheduler(shift=5)
    action_scheduler = FlowMatchEulerDiscreteScheduler(shift=1)
    for scheduler in (video_scheduler, action_scheduler):
        scheduler.set_timesteps(4, device='cuda')
    tokens = [torch.cat([v.repeat(2), a.repeat(2)])[None] for v, a in zip(
        video_scheduler.timesteps, action_scheduler.timesteps)]
    runner = JointVideoActionCudaGraphRunner(
        wrapper, cond_kv, [cond['context_emb']], cond,
        {'joint_rope_freqs': torch.zeros(1, 1, 1, device='cuda'), 'token_timesteps': tokens},
        anchor, mask, video, trajectory, 2, video_scheduler, action_scheduler, 4, 3,
        uncond_kv_caches=unc_kv, uncond_context=[unc['context_emb']],
        uncond_inference_context_cache=unc, guidance_scale=5, traj_guidance_scale=1,
    )
    position = torch.tensor([3], device='cuda')
    for offset in (0., 0.7):
        cond['context_emb'].fill_(2 + offset)
        unc_kv[0][0].fill_(offset)
        source_video, source_traj = video + offset, trajectory + offset
        v, a = source_video.clone(), source_traj.clone()
        vs, ats = copy.deepcopy(video_scheduler), copy.deepcopy(action_scheduler)
        for index, (vt, at) in enumerate(zip(vs.timesteps, ats.timesteps)):
            conditioned = (1 - mask) * anchor + mask * v
            uv, ua = wrapper(x=list(conditioned), traj=a, t=tokens[index],
                             inference_context_cache=unc, kv_caches=unc_kv)
            cv, ca = wrapper(x=list(conditioned), traj=a, t=tokens[index],
                             inference_context_cache=cond, kv_caches=cond_kv)
            v = (1 - mask) * anchor + mask * vs.step(uv + 5 * (cv - uv), vt, conditioned).prev_sample
            a = ats.step(ca, at, a).prev_sample
        assert runner.matches(cond_kv, source_video, source_traj, 2, 3, uncond_kv_caches=unc_kv)
        actual_v, actual_a = runner(cond_kv, cond, anchor, source_video, source_traj, position,
                                   uncond_kv_caches=unc_kv, uncond_inference_context_cache=unc)
        torch.testing.assert_close(actual_v, v)
        torch.testing.assert_close(actual_a, a)
