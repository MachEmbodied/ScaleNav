import copy
import json
from types import SimpleNamespace

import pytest
import torch

from awm.training import runner
from awm.training.data import SeededSources
from awm.training.recipe import load_recipe
from test_training import tiny_model


class FixedVideoSource:
    num_past_chunks = 0

    def __len__(self):
        return 3

    def get_batch(self, index):
        return {'is_precomputed': True, 'text': 'Go to the chair.', 'has_traj': False,
                'latents': torch.full((4, 3, 4, 4), index / 10),
                'inpaint_latent': torch.full((4, 1, 4, 4), index / 10),
                'trajectory': torch.zeros(9, 3), 'trajectory_loss_mask': torch.zeros(9),
                'video_frame_loss_mask': torch.ones(9)}


def test_parameter_groups_preserve_adam_updates():
    flat = [torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) + i)
            for i in range(3)]
    grouped = [torch.nn.Parameter(value.detach().clone()) for value in flat]
    groups = runner.parameter_groups(grouped, max_elements=16)
    assert len(groups) == 3
    optimizers = [torch.optim.AdamW(flat, lr=1e-3), torch.optim.AdamW(groups, lr=1e-3)]
    for step in range(2):
        for values, optimizer in zip((flat, grouped), optimizers):
            for value in values:
                value.grad = torch.full_like(value, 0.1 * (step + 1))
            optimizer.step()
    for first, second in zip(flat, grouped):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('metadata', [
    {'stage': 'p1', 'global_step': 9999},
    {'stage': 'joint', 'global_step': 10000},
])
def test_posttraining_rejects_wrong_pretraining_checkpoint(tmp_path, metadata):
    torch.save(metadata, tmp_path / 'train_meta.pt')
    with pytest.raises(ValueError):
        runner.validate_initialization(tmp_path, 'align')


@pytest.mark.parametrize('complete,state', [(False, True), (True, False)])
def test_causal_pretraining_requires_completed_optimizer_state(tmp_path, complete, state):
    torch.save({'stage': 'p0', 'global_step': 2000}, tmp_path / 'train_meta.pt')
    (tmp_path / 'training.json').write_text(json.dumps({'complete': complete}))
    if state:
        (tmp_path / 'state').mkdir()
    with pytest.raises(ValueError, match='optimizer state'):
        runner.validate_initialization(tmp_path, 'p1')


def test_training_loop_resume_matches_uninterrupted_updates(tmp_path, monkeypatch, fresh_accelerator):
    from awm.training import data
    monkeypatch.setattr(runner, 'require_training_device', lambda accelerator: None)
    monkeypatch.setattr(runner, 'load_model', lambda *values: tiny_model('p0'))
    monkeypatch.setattr(runner, 'load_text_encoder', lambda *values: (None, None))
    monkeypatch.setattr(runner, 'encode_text', lambda text, *values: [torch.ones(3, 16) for _ in text])
    monkeypatch.setattr(data, 'build_sources', lambda *values: (SeededSources(FixedVideoSource()), {'test': 'fixed'}))
    recipe = {**load_recipe('p0'), 'updates': 2, 'global_batch': 4, 'warmup_updates': 0}
    args = SimpleNamespace(micro_batch=1, deepspeed=False, check_data=False, seed=42,
                           resume=None, max_updates=2, workers=0, output=str(tmp_path / 'continuous'),
                           initialize_from=None, base_model='unused', save_every=1)
    assert runner.train(args, recipe) == 0
    split = copy.copy(args)
    split.output, split.max_updates = str(tmp_path / 'resumed'), 1
    assert runner.train(split, recipe) == 0
    split.resume = str(tmp_path / 'resumed/checkpoint-1')
    split.max_updates = 2
    assert runner.train(split, recipe) == 0
    from safetensors.torch import load_file
    continuous = load_file(tmp_path / 'continuous/checkpoint-2/state/model.safetensors')
    resumed = load_file(tmp_path / 'resumed/checkpoint-2/state/model.safetensors')
    assert continuous.keys() == resumed.keys()
    for key in continuous:
        torch.testing.assert_close(continuous[key], resumed[key], rtol=0, atol=0)
    split.resume = str(tmp_path / 'resumed/checkpoint-2')
    monkeypatch.setattr(runner, 'load_model', lambda *values: pytest.fail('Completed phase reloaded its model'))
    assert runner.train(split, recipe) == 0
