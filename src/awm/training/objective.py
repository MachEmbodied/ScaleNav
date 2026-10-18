"""Teacher-forced video/action flow matching."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def set_trainable(model, phase):
    if phase not in {'video_pretrain', 'action_alignment', 'joint_finetune'}:
        raise ValueError(f'Unknown phase: {phase}')
    for name, parameter in model.dit_model.named_parameters():
        action = name.startswith(('traj_embed.', 'traj_head.'))
        parameter.requires_grad_(
            not action if phase == 'video_pretrain' else
            action or '.self_attn.' in name if phase == 'action_alignment' else True
        )
    return [p for p in model.parameters() if p.requires_grad]


def frame_mask_to_latent_mask(mask, length):
    if mask.ndim != 2 or length < 1 or mask.shape[1] < 1:
        raise ValueError('Expected a nonempty [batch, frames] validity mask')
    width = (mask.shape[1] + length - 1) // length
    padded = F.pad(mask, (0, length * width - mask.shape[1]), mode='replicate')
    return padded.reshape(mask.shape[0], length, width).amax(-1)


def masked_mse(prediction, target, mask):
    error = (prediction.float() - target.float()).square()
    mask = mask.to(error).expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def balanced_loss(video, action):
    denominator = (video.detach() + action.detach()).clamp_min(1e-8)
    action_weight = video.detach() / denominator
    return (1 - action_weight) * video + action_weight * action


def sample_noise_levels(batch_size, device, shift, count=1000):
    sigma = (torch.randint(0, count, (batch_size,), device=device) + 1).float() / count
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma, sigma * count


def _select(batch, selection):
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[selection]
        elif isinstance(value, list):
            result[key] = [item[selection] for item in value]
    return result


class FlowObjective(nn.Module):
    def __init__(self, model, recipe):
        super().__init__()
        self.model = model
        self.recipe = recipe

    def forward(self, batch, contexts):
        action_flags = batch['has_traj'].bool()
        if self.recipe['phase'] == 'video_pretrain' and action_flags.any():
            raise ValueError('Video pretraining requires action-free samples')
        total = batch['latents'].new_zeros((), dtype=torch.float32)
        metrics = {'video_loss': total.detach().clone(), 'action_loss': total.detach().clone()}
        for has_actions in (False, True):
            selection = action_flags == has_actions
            indices = selection.nonzero().flatten().tolist()
            if not indices:
                continue
            subset = _select(batch, selection)
            video, action = self._group(subset, [contexts[i] for i in indices], has_actions)
            fraction = len(indices) / len(action_flags)
            future_valid = frame_mask_to_latent_mask(subset['video_frame_loss_mask'], subset['latents'].shape[2])[:, 1:]
            if has_actions:
                group_loss = balanced_loss(video, action) if future_valid.any() else action
            else:
                group_loss = video
            total = total + fraction * group_loss
            metrics['video_loss'] = metrics['video_loss'] + fraction * video.detach()
            metrics['action_loss'] = metrics['action_loss'] + fraction * action.detach()
        if not action_flags.any() and self.recipe['phase'] != 'video_pretrain':
            # Flush accumulated action gradients even when the last microbatch is video-only.
            for module in (self.model.dit_model.traj_embed, self.model.dit_model.traj_head):
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        total = total + parameter.reshape(-1)[0] * 0
        metrics['replay_fraction'] = (~action_flags).float().mean()
        return total, metrics

    def _group(self, batch, contexts, has_actions):
        recipe = self.recipe
        latents = batch['latents']
        batch_size, channels, temporal, height, width = latents.shape
        device = latents.device
        actions = batch['trajectory']
        actions = actions.sign() * torch.log1p(actions.abs() / recipe['symlog_c'])
        if not has_actions:
            actions = actions[:, :0]
        past_video = batch.get('past_latents')
        past_actions = batch.get('past_trajectories')
        if past_actions is not None:
            past_actions = [a.sign() * torch.log1p(a.abs() / recipe['symlog_c']) for a in past_actions]
            if not has_actions:
                past_actions = [a[:, :0] for a in past_actions]
        sv, tv = sample_noise_levels(batch_size, device, recipe['video_shift'], recipe['noise_timesteps'])
        sa, ta = sample_noise_levels(batch_size, device, recipe['action_shift'], recipe['noise_timesteps'])
        noise_v, noise_a = torch.randn_like(latents), torch.randn_like(actions)
        noisy_v = (1 - sv[:, None, None, None, None]) * latents + sv[:, None, None, None, None] * noise_v
        noisy_a = (1 - sa[:, None, None]) * actions + sa[:, None, None] * noise_a
        noisy_v[:, :, :1] = batch['inpaint_latents']
        pt, ph, pw = self.model.config.patch_size
        sequence = (temporal // pt) * (height // ph) * (width // pw)
        video_times = tv[:, None].expand(-1, sequence).clone()
        video_times[:, :(height // ph) * (width // pw)] = 0
        times = torch.cat((video_times, ta[:, None].expand(-1, actions.shape[1])), dim=1)
        video_prediction, action_prediction = self.model(
            x=list(noisy_v.unbind()), traj=noisy_a, t=times,
            context=contexts, seq_len=sequence,
            past_video_latents=past_video, past_trajectories=past_actions,
            past_chunk_mask=batch.get('past_chunk_mask'),
            past_chunk_position_ids=batch.get('past_chunk_position_ids'),
            current_chunk_position_id=batch.get('current_chunk_position_id'),
            is_training=True,
        )
        if isinstance(video_prediction, list):
            video_prediction = torch.stack(video_prediction)
        valid = frame_mask_to_latent_mask(batch['video_frame_loss_mask'], temporal)
        valid[:, 0] = 0
        video_loss = masked_mse(video_prediction[:, :channels], noise_v - latents,
                               valid[:, None, :, None, None])
        action_loss = video_loss.new_zeros(())
        if has_actions:
            action_loss = masked_mse(action_prediction, noise_a - actions,
                                    batch['trajectory_loss_mask'][:, :, None])
        return video_loss, action_loss
