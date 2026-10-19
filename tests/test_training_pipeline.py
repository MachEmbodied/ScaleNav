import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from awm.training import pipeline, recipe, runner


def arguments(tmp_path, stage='pretrain', extra=()):
    values = ['--stage', stage, '--gpus', '8', '--base-model', 'base model',
              '--video-cache', 'video cache', '--output', str(tmp_path / stage)]
    if stage == 'posttrain':
        values += ['--initialize-from', 'pretrained/final', '--action-cache', 'action cache',
                   '--action-meta', 'actions.jsonl', '--action-split', 'train.txt']
    return runner.parser().parse_args([*values, *extra])


def checkpoint(entry, step=None):
    path = Path(entry['checkpoint'])
    (path / 'state').mkdir(parents=True)
    (path / 'state/optimizer.bin').write_bytes(b'optimizer')
    (path / 'transformer').mkdir()
    (path / 'transformer/config.json').write_text('{}')
    (path / 'transformer/diffusion_pytorch_model.safetensors').write_bytes(b'weights')
    (path / 'train_meta.pt').write_bytes(b'metadata')
    (path / 'training.json').write_text(json.dumps({
        'complete': True, 'step': entry['recipe']['updates'] if step is None else step,
        'contract': {'recipe': entry['recipe']}}))
    return path


@pytest.mark.parametrize('stage,phases', [('pretrain', ['p0', 'p1']), ('posttrain', ['align', 'joint'])])
def test_pipeline_commands_pass_the_worker_parser(tmp_path, monkeypatch, capsys, stage, phases):
    args = arguments(tmp_path, stage, ['--deepspeed', '--text-device', 'cpu'])
    entries = pipeline.plan(args)
    assert [entry['stage'] for entry in entries] == phases
    with monkeypatch.context() as scoped:
        scoped.setenv('WORLD_SIZE', '8')
        for entry in entries:
            command = entry['command']
            assert '--nproc-per-node=8' in command and '--deepspeed' in command
            assert command[command.index('--base-model') + 1] == 'base model'
            worker_args = command[command.index('--module') + 2:]
            assert runner.main([*worker_args, '--dry-run']) == 0
            report = json.loads(capsys.readouterr().out)
            assert report['recipe'] == entry['recipe']
            assert report['gradient_accumulation_steps'] == entry['gradient_accumulation_steps']
    assert entries[1]['command'][entries[1]['command'].index('--initialize-from') + 1] == entries[0]['checkpoint']


@pytest.mark.parametrize('stage', ['pretrain', 'posttrain'])
def test_successful_pipeline_passes_checkpoint_and_publishes_final(tmp_path, monkeypatch, stage):
    args = arguments(tmp_path, stage)
    entries = pipeline.plan(args)
    calls = []

    def launch(command):
        entry = entries[len(calls)]
        if calls:
            assert Path(command[command.index('--initialize-from') + 1]).is_dir()
        assert command == entry['command']
        calls.append(command)
        checkpoint(entry)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', launch)
    assert pipeline.run(args) == 0
    assert len(calls) == 2
    assert (Path(args.output) / 'final').resolve() == Path(entries[-1]['checkpoint']).resolve()


def test_failed_worker_never_starts_the_next_phase(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command: (
        calls.append(command) or SimpleNamespace(returncode=7)))
    assert pipeline.run(args) == 7
    assert len(calls) == 1
    assert not Path(args.output).exists()


@pytest.mark.parametrize('failure', ['partial', 'missing_state', 'missing_weights'])
def test_incomplete_checkpoint_cannot_advance_pipeline(tmp_path, monkeypatch, failure):
    args = arguments(tmp_path)
    entry = pipeline.plan(args)[0]
    calls = []

    def launch(command):
        calls.append(command)
        path = checkpoint(entry, 1 if failure == 'partial' else None)
        if failure == 'missing_state':
            (path / 'state/optimizer.bin').unlink()
        if failure == 'missing_weights':
            (path / 'transformer/diffusion_pytorch_model.safetensors').unlink()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', launch)
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.run(args)
    assert len(calls) == 1
    assert not (Path(args.output) / 'final').exists()


@pytest.mark.parametrize('stage', ['pretrain', 'posttrain'])
def test_resume_later_phase_does_not_restart_earlier_phase(tmp_path, stage):
    args = arguments(tmp_path, stage)
    last = pipeline.plan(args)[-1]
    args.resume = str(checkpoint(last, 1))
    args.initialize_from = None
    entries = pipeline.plan(args)
    assert len(entries) == 1
    assert entries[0]['stage'] == last['stage']
    assert '--resume' in entries[0]['command']
    assert '--initialize-from' not in entries[0]['command']


def test_resume_completed_first_phase_continues_next_phase(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    entries = pipeline.plan(args)
    args.resume = str(checkpoint(entries[0]))
    calls = []

    def launch(command):
        calls.append(command)
        if len(calls) == 1:
            assert command[command.index('--resume') + 1] == args.resume
        else:
            assert command[command.index('--initialize-from') + 1] == args.resume
            checkpoint(entries[1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', launch)
    assert pipeline.run(args) == 0
    assert len(calls) == 2


def test_smoke_limit_does_not_advance_or_publish_final(tmp_path, monkeypatch):
    args = arguments(tmp_path, extra=['--max-updates', '1'])
    entries = pipeline.plan(args)
    assert len(entries) == 1 and not entries[0]['completes_phase']
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command: SimpleNamespace(returncode=0))
    assert pipeline.run(args) == 0
    assert not (Path(args.output) / 'final').exists()


def test_check_data_checks_both_phases_without_publishing(tmp_path, monkeypatch):
    args = arguments(tmp_path, extra=['--check-data'])
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command: (
        calls.append(command) or SimpleNamespace(returncode=0)))
    assert pipeline.run(args) == 0
    assert len(calls) == 2 and all('--check-data' in c for c in calls)
    assert not Path(args.output).exists()


def test_posttrain_data_check_needs_no_initial_checkpoint(tmp_path, monkeypatch):
    args = arguments(tmp_path, 'posttrain', ['--check-data'])
    args.initialize_from = None
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command: (
        calls.append(command) or SimpleNamespace(returncode=0)))
    assert pipeline.run(args) == 0
    assert len(calls) == 2 and '--initialize-from' not in calls[0]


def test_invalid_launch_settings_fail_before_starting_workers(tmp_path, monkeypatch):
    args = arguments(tmp_path, extra=['--gpus', '3'])
    with pytest.raises(ValueError, match='divide'):
        pipeline.plan(args)
    monkeypatch.setenv('LOCAL_RANK', '0')
    with pytest.raises(ValueError, match='without torchrun'):
        pipeline.plan(args)


def test_wrong_resume_pipeline_or_output_is_rejected(tmp_path):
    args = arguments(tmp_path)
    args.resume = str(checkpoint(pipeline.plan(args)[0]))
    args.output = str(tmp_path / 'different')
    with pytest.raises(ValueError, match='output directory'):
        pipeline.plan(args)
    args.stage = 'posttrain'
    args.action_cache, args.action_meta, args.action_split = 'a', 'b', 'c'
    with pytest.raises(ValueError, match='does not belong'):
        pipeline.plan(args)


def test_configured_phase_lengths_control_checkpoint_handoff(tmp_path, monkeypatch):
    import yaml
    config = yaml.safe_load(recipe.CONFIG.read_text())
    config['stages']['p0']['updates'] = 3
    config['stages']['p1']['updates'] = 4
    path = tmp_path / 'training.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(recipe, 'CONFIG', path)
    entries = pipeline.plan(arguments(tmp_path))
    assert entries[0]['checkpoint'].endswith('/p0/checkpoint-3')
    assert entries[1]['checkpoint'].endswith('/p1/checkpoint-7')
    assert runner.deployment_metadata(entries[1]['recipe'], 2)['global_step'] == 5
