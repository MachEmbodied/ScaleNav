import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from awm.models.causal_chunk_ar_model import CausalChunkDiTModel, CausalChunkDiTWrapper
from awm.training.data import UpdateBatchSampler, collate_samples
from awm.training.objective import FlowObjective, balanced_loss, frame_mask_to_latent_mask, masked_mse, set_trainable
from awm.training.recipe import accumulation_steps, load_recipe, lr_multiplier
from awm.training.runner import deployment_metadata, load_model, save_checkpoint
from awm.utils.checkpoint import checkpoint_weight_files, load_causal_checkpoint, save_causal_weights


def tiny_model(stage='joint'):
    recipe = load_recipe(stage)
    return CausalChunkDiTWrapper(CausalChunkDiTModel(
        model_type='t2v', patch_size=(1, 2, 2), text_len=8,
        in_dim=4, dim=24, ffn_dim=48, freq_dim=32, text_dim=16, out_dim=4,
        num_heads=3, num_layers=1, action_dim=3, max_attn_chunks=recipe['max_attn_chunks'],
        use_continuous_history=bool(recipe['past_chunks']),
        traj_head_type='deep', traj_head_hidden_dim=16))


def batch(has_actions=(False, True), history=True):
    count = len(has_actions)
    result = dict(latents=torch.randn(count, 4, 3, 4, 4),
                  inpaint_latents=torch.randn(count, 4, 1, 4, 4),
                  trajectory=torch.randn(count, 9, 3) * .01,
                  trajectory_loss_mask=torch.ones(count, 9),
                  video_frame_loss_mask=torch.ones(count, 9),
                  has_traj=torch.tensor(has_actions))
    if history:
        result.update(past_latents=[torch.randn(count, 4, 3, 4, 4) for _ in range(3)],
                      past_trajectories=[torch.randn(count, 9, 3) * .01 for _ in range(3)],
                      past_chunk_mask=torch.tensor([[0, 0, 0]] + [[1, 1, 1]] * (count - 1)),
                      past_chunk_position_ids=torch.tensor([[-1, -1, -1]] + [[0, 2, 3]] * (count - 1)),
                      current_chunk_position_id=torch.tensor([0] + [4] * (count - 1)))
    return result


@pytest.mark.parametrize('stage', ['p0', 'p1', 'align', 'joint'])
def test_real_model_training_step_respects_frozen_parameters(stage):
    torch.manual_seed(12)
    model, recipe = tiny_model(stage), load_recipe(stage)
    parameters = set_trainable(model, recipe['phase'])
    before = {name: value.clone() for name, value in model.named_parameters()}
    model.train()
    model.enable_gradient_checkpointing()
    objective = FlowObjective(model, recipe)
    data = batch((False, False) if stage in ('p0', 'p1') else (False, True), stage != 'p0')
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss, metrics = objective(data, [torch.randn(3, 16), torch.randn(5, 16)])
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        elif not torch.equal(parameter, before[name]):
            changed.append(name)
    assert changed
    if stage == 'align':
        assert all('.self_attn.' in name or '.traj_' in name for name in changed)
    if stage in ('p0', 'p1'):
        assert metrics['action_loss'] == 0
        assert not any('.traj_' in name for name in changed)


def test_action_free_training_has_no_action_tokens_and_ignores_dummy_values():
    model = tiny_model('p1')
    objective = FlowObjective(model, load_recipe('p1'))
    data = batch((False,), True)
    context = [torch.randn(3, 16)]
    lengths = []
    handle = model.dit_model.traj_embed.register_forward_pre_hook(lambda module, values: lengths.append(values[0].shape[1]))
    torch.manual_seed(2)
    first, _ = objective(data, context)
    data['trajectory'].fill_(999)
    for past in data['past_trajectories']:
        past.fill_(999)
    torch.manual_seed(2)
    second, _ = objective(data, context)
    handle.remove()
    assert all(length == 0 for length in lengths)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_balanced_loss_stops_gradient_through_weights():
    video, action = torch.tensor(4., requires_grad=True), torch.tensor(1., requires_grad=True)
    loss = balanced_loss(video, action)
    loss.backward()
    assert loss.item() == pytest.approx(1.6)
    assert video.grad.item() == pytest.approx(.2)
    assert action.grad.item() == pytest.approx(.8)


def test_replay_has_full_video_loss_in_a_mixed_batch(monkeypatch):
    objective = FlowObjective(tiny_model(), load_recipe('joint'))
    video = torch.tensor(4., requires_grad=True)
    action = torch.tensor(1., requires_grad=True)
    replay = torch.tensor(6., requires_grad=True)
    def fake_group(data, context, has_actions):
        return (video, action) if has_actions else (replay, replay.new_zeros(()))
    monkeypatch.setattr(objective, '_group', fake_group)
    loss, _ = objective(batch(), [torch.zeros(1), torch.zeros(1)])
    loss.backward()
    assert loss.item() == pytest.approx(.5 * 1.6 + .5 * 6)
    assert replay.grad.item() == .5


def test_temporal_padding_mask_and_zero_action_supervision():
    mask = frame_mask_to_latent_mask(torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0.]]), 3)
    assert mask.tolist() == [[1., 1., 0.]]
    prediction = torch.ones(1, 1, 3, 1, 1, requires_grad=True)
    masked_mse(prediction, torch.zeros_like(prediction), mask[:, None, :, None, None]).backward()
    assert prediction.grad.flatten().tolist() == [1., 1., 0.]
    action = torch.ones(1, 9, 3, requires_grad=True)
    masked_mse(action, torch.zeros_like(action), torch.ones(1, 9, 1)).backward()
    assert action.grad[:, -1].abs().sum() > 0


def test_paper_update_counts_batches_and_schedule_boundaries():
    assert [load_recipe(s)['updates'] for s in ('p0', 'p1', 'align', 'joint')] == [2000, 8000, 2500, 6500]
    joint = load_recipe('joint')
    for step, wanted in [(0, 0), (250, .5), (500, 1), (4000, 1), (4750, .55), (5500, .1), (6500, .1)]:
        assert lr_multiplier(step, joint) == pytest.approx(wanted)
    assert accumulation_steps(joint, 2, 1) == 16
    assert accumulation_steps(load_recipe('align'), 8, 8) == 1
    with pytest.raises(ValueError):
        accumulation_steps(joint, 3, 1)


def test_sampler_resume_preserves_exact_remaining_samples():
    kwargs = dict(lengths=[73, 11], replay_probability=.2, micro_batch=4,
                  total_micro_batches=500, seed=42, rank=1, world_size=2)
    all_batches = list(UpdateBatchSampler(**kwargs))
    assert list(UpdateBatchSampler(**kwargs, start_micro_batch=17)) == all_batches[17:]
    probability = sum(item[0] == 0 for group in all_batches for item in group) / 2000
    assert .17 < probability < .23
    assert all_batches != list(UpdateBatchSampler(**{**kwargs, 'rank': 0}))


def test_sharded_training_checkpoint_loads_in_inference(tmp_path):
    model = tiny_model()
    state = model.dit_model.state_dict()
    directory = tmp_path / 'transformer'
    directory.mkdir()
    model.dit_model.save_config(directory)
    names = list(state)
    mapping = {}
    for number, part in enumerate((names[::2], names[1::2])):
        filename = f'weights-{number}.safetensors'
        save_file({key: state[key] for key in part}, directory / filename)
        mapping.update({key: filename for key in part})
    (directory / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    loaded = load_causal_checkpoint(tmp_path, 'cpu', torch.float32)
    for key, tensor in state.items():
        torch.testing.assert_close(loaded.dit_model.state_dict()[key], tensor, rtol=0, atol=0)
    mapping[names[0]] = 'weights-1.safetensors'
    (directory / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    with pytest.raises(ValueError, match='index'):
        load_causal_checkpoint(tmp_path, 'cpu', torch.float32)


def test_native_wan_initialization_loads_every_backbone_tensor(tmp_path):
    original = tiny_model('p0').dit_model
    config = dict(original.config)
    config.pop('_use_default_values', None)
    config['model_type'] = 'ti2v'
    config.pop('cross_attn_type', None)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    weights = {name: tensor for name, tensor in original.state_dict().items()
               if not name.startswith(('traj_embed.', 'traj_head.'))}
    save_file(weights, tmp_path / 'diffusion_pytorch_model.safetensors')
    recipe = load_recipe('p0')
    recipe['traj_head_hidden_dim'] = 16
    loaded = load_model(tmp_path, recipe)
    for name, tensor in weights.items():
        torch.testing.assert_close(loaded.dit_model.state_dict()[name], tensor, rtol=0, atol=0)
    assert loaded.config.cross_attn_type == 'cross_attn'
    assert not any('k_img' in name for name in loaded.state_dict())
    del weights[next(iter(weights))]
    save_file(weights, tmp_path / 'diffusion_pytorch_model.safetensors')
    with pytest.raises(ValueError, match='Incomplete Wan backbone'):
        load_model(tmp_path, recipe)


def test_current_observation_is_not_a_denoising_target():
    model = tiny_model('p0')
    objective = FlowObjective(model, load_recipe('p0'))
    captured = []
    def capture(module, inputs, output):
        output[0].retain_grad()
        captured.append(output[0])
    handle = model.register_forward_hook(capture)
    loss, _ = objective(batch((False,), False), [torch.randn(3, 16)])
    loss.backward()
    handle.remove()
    assert captured[0].grad[:, :, 0].count_nonzero() == 0
    assert captured[0].grad[:, :, 1:].abs().sum() > 0


def test_terminal_actions_remain_supervised_without_future_video():
    model = tiny_model('joint')
    objective = FlowObjective(model, load_recipe('joint'))
    data = batch((True,), False)
    data['video_frame_loss_mask'].zero_()
    data['video_frame_loss_mask'][:, 0] = 1
    loss, metrics = objective(data, [torch.randn(3, 16)])
    assert metrics['video_loss'] == 0
    torch.testing.assert_close(loss.detach(), metrics['action_loss'])
    loss.backward()
    assert model.dit_model.traj_head.head.weight.grad.abs().sum() > 0


def test_saved_state_restores_optimizer_scheduler_and_rng(tmp_path, fresh_accelerator):
    from accelerate import Accelerator
    torch.manual_seed(6)
    accelerator = Accelerator(cpu=True)
    recipe = load_recipe('joint')
    model = tiny_model()
    params = set_trainable(model, recipe['phase'])
    objective = FlowObjective(model, recipe)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    objective, optimizer = accelerator.prepare(objective, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .95 ** step)
    accelerator.register_for_checkpointing(scheduler)
    data, context = batch(), [torch.randn(3, 16), torch.randn(3, 16)]
    loss, _ = objective(data, context)
    accelerator.backward(loss)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    path = save_checkpoint(accelerator, objective, tmp_path, recipe, 1, {'test': True})
    for filename in checkpoint_weight_files(path / 'transformer'):
        assert all(tensor.dtype == torch.bfloat16 for tensor in load_file(filename).values())
    expected_weights = {n: p.clone() for n, p in objective.named_parameters()}
    expected_rng = torch.rand(3)
    saved_step = scheduler.state_dict()['last_epoch']
    loss, _ = objective(data, context)
    accelerator.backward(loss)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    accelerator.load_state(path / 'state')
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    assert scheduler.state_dict()['last_epoch'] == saved_step
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    for name, parameter in objective.named_parameters():
        torch.testing.assert_close(parameter, expected_weights[name], rtol=0, atol=0)
    loaded = load_causal_checkpoint(path, 'cpu', torch.float32)
    assert loaded.dit_model.config.use_continuous_history
    metadata = torch.load(path / 'train_meta.pt', weights_only=True)
    assert metadata['args']['traj_normalizer'] == 'symlog'
    assert (path / 'training.json').is_file()
    assert not path.with_name(path.name + '.partial').exists()


def test_export_shards_use_supplied_bfloat16_weights(tmp_path):
    model = tiny_model().dit_model
    state = {name: tensor.to(torch.bfloat16) for name, tensor in model.state_dict().items()}
    save_causal_weights(model, state, tmp_path / 'transformer', max_shard_size=16384)
    assert len(checkpoint_weight_files(tmp_path / 'transformer')) > 1
    loaded = load_causal_checkpoint(tmp_path, 'cpu', torch.bfloat16)
    for name, tensor in loaded.dit_model.state_dict().items():
        assert tensor.dtype == torch.bfloat16
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    assert all(tensor.dtype == torch.float32 for tensor in model.state_dict().values())
