import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from awm.training import pipeline, recipe, runner
from test_training import tiny_model


@pytest.fixture
def small_pipeline(tmp_path, monkeypatch):
    torch.manual_seed(7)
    config = yaml.safe_load(recipe.CONFIG.read_text())
    config['common']['traj_head_hidden_dim'] = 16
    for stage in config['stages'].values():
        stage.update(updates=2, global_batch=2, warmup_updates=1)
    config_path = tmp_path / 'training.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(recipe, 'CONFIG', config_path)
    monkeypatch.setenv('AWM_TEST_RECIPE', str(config_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')

    base = tmp_path / 'base model'
    base.mkdir()
    model = tiny_model('p0').dit_model
    model.save_config(base)
    save_file({name: value for name, value in model.state_dict().items()
               if not name.startswith(('traj_embed.', 'traj_head.'))},
              base / 'diffusion_pytorch_model.safetensors')
    for kind in ('video', 'action'):
        cache = tmp_path / kind
        (cache / 'scene/episode').mkdir(parents=True)
        for index in range(4):
            torch.save({'latent': torch.full((4, 3, 4, 4), index / 10),
                        'inpaint_latent': torch.full((4, 1, 4, 4), index / 10),
                        'trajectory': torch.full((9, 3), .01), 'is_last_chunk': index == 3,
                        'real_len_in_output': 9}, cache / f'scene/episode/chunk_{index:04d}.pt')
    meta, split = tmp_path / 'meta.jsonl', tmp_path / 'train.txt'
    meta.write_text(json.dumps({'key': 'scene/episode', 'canonical': 'Approach the chair.'}) + '\n')
    split.write_text('scene/episode\n')
    geometry = dict(complete=True, video_sample_n_frames=9, video_sample_stride=1,
                    video_sample_size=[288, 512], chunk_overlap_frames=0, chunk_step=9)
    (tmp_path / 'video/manifest.json').write_text(json.dumps({**geometry, 'samples': [
        {'relative_dir': 'scene/episode', 'n_chunks': 4, 'text': 'Approach the chair.'}]}))
    (tmp_path / 'action/manifest.json').write_text(json.dumps({**geometry, 'version': 2,
        'source_meta_sha256': hashlib.sha256(meta.read_bytes()).hexdigest(), 'episodes': [
            {'key': 'scene/episode', 'scene': 'scene', 'episode': 'episode', 'n_chunks': 4}]}))

    launch = subprocess.run
    calls = []

    def worker(command):
        calls.append(command)
        position = command.index('--module')
        command = [*command[:position], str(Path(__file__).with_name('training_pipeline_worker.py')),
                   *command[position + 2:]]
        result = launch(command, capture_output=True, text=True, timeout=120)
        (tmp_path / f'worker-{len(calls)}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout + result.stderr)
        return result

    monkeypatch.setattr(pipeline.subprocess, 'run', worker)

    def run(stage, output, extra=()):
        values = ['--stage', stage, '--gpus', '1', '--base-model', str(base),
                  '--video-cache', str(tmp_path / 'video'), '--output', str(tmp_path / output),
                  '--action-cache', str(tmp_path / 'action'), '--action-meta', str(meta),
                  '--action-split', str(split), '--workers', '0', '--text-device', 'cpu',
                  '--save-every', '1', *map(str, extra)]
        return pipeline.run(runner.parser().parse_args(values))

    return run, calls


def assert_same_state(first, second):
    for name in ('model.safetensors',):
        left, right = load_file(first / name), load_file(second / name)
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    left, right = (torch.load(path / 'optimizer.bin', weights_only=True) for path in (first, second))
    assert left['param_groups'] == right['param_groups']
    assert left['state'].keys() == right['state'].keys()
    for key in left['state']:
        for field in left['state'][key]:
            torch.testing.assert_close(left['state'][key][field], right['state'][key][field], rtol=0, atol=0)


def test_real_training_crosses_all_phase_boundaries_and_resumes(tmp_path, small_pipeline):
    run, calls = small_pipeline
    assert run('pretrain', 'pretrain') == 0
    assert len(calls) == 2
    pretrained = tmp_path / 'pretrain/final'
    assert pretrained.resolve() == tmp_path / 'pretrain/p1/checkpoint-4'
    state = torch.load(pretrained / 'state/optimizer.bin', weights_only=True)
    assert {int(value['step']) for value in state['state'].values()} == {4}
    scheduler = torch.load(pretrained / 'state/custom_checkpoint_0.pkl', weights_only=True)
    assert scheduler['last_epoch'] == 4
    assert run('posttrain', 'posttrain', ['--initialize-from', pretrained]) == 0
    assert len(calls) == 4
    for stage in ('align', 'joint'):
        state = torch.load(tmp_path / f'posttrain/{stage}/checkpoint-2/state/optimizer.bin', weights_only=True)
        assert {int(value['step']) for value in state['state'].values()} == {2}
    final = tmp_path / 'posttrain/final'
    assert final.resolve() == tmp_path / 'posttrain/joint/checkpoint-2'
    before = len((tmp_path / 'posttrain/joint/metrics.jsonl').read_text().splitlines())
    assert run('posttrain', 'posttrain', ['--resume', final]) == 0
    assert len((tmp_path / 'posttrain/joint/metrics.jsonl').read_text().splitlines()) == before

    assert run('pretrain', 'resumed', ['--max-updates', '1']) == 0
    assert not (tmp_path / 'resumed/final').exists()
    assert run('pretrain', 'resumed', ['--resume', tmp_path / 'resumed/p0/checkpoint-1']) == 0
    assert_same_state(pretrained / 'state', tmp_path / 'resumed/final/state')

    assert run('posttrain', 'resumed_post', ['--initialize-from', pretrained, '--max-updates', '1']) == 0
    assert run('posttrain', 'resumed_post', ['--resume', tmp_path / 'resumed_post/align/checkpoint-1']) == 0
    assert_same_state(final / 'state', tmp_path / 'resumed_post/final/state')


def test_real_boundary_resume_after_failure_before_first_checkpoint(tmp_path, monkeypatch, small_pipeline):
    run, calls = small_pipeline
    monkeypatch.setenv('AWM_TEST_FAIL_STAGE', 'p1')
    assert run('pretrain', 'run') != 0
    assert len(calls) == 2
    assert not (tmp_path / 'run/final').exists()
    assert not list((tmp_path / 'run/p1').glob('checkpoint-*'))
    assert (tmp_path / 'run/p1/metrics.jsonl').is_file()
    monkeypatch.delenv('AWM_TEST_FAIL_STAGE')
    assert run('pretrain', 'run', ['--resume', tmp_path / 'run/p0/checkpoint-2']) == 0
    assert len(calls) == 4
    assert (tmp_path / 'run/p1.interrupted-1/metrics.jsonl').is_file()
    assert (tmp_path / 'run/final').resolve() == tmp_path / 'run/p1/checkpoint-4'
    assert len((tmp_path / 'run/p0/metrics.jsonl').read_text().splitlines()) == 2
    state = torch.load(tmp_path / 'run/final/state/optimizer.bin', weights_only=True)
    assert {int(value['step']) for value in state['state'].values()} == {4}
