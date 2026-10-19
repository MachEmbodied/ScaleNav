import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

from awm.cli import CONFIG_DIR, build_inference_command, main


def test_doctor_fails_when_data_tools_are_missing(monkeypatch, capsys):
    monkeypatch.setattr('awm.cli.shutil.which', lambda name: None)
    assert main(['doctor', '--data-tools']) == 1
    report = json.loads(capsys.readouterr().out)
    assert any('ffprobe' in error for error in report['errors'])
    assert any('ffmpeg' in error for error in report['errors'])


def test_doctor_reports_simulator_loader_failure(monkeypatch, capsys):
    import awm.cli as cli
    original = cli.importlib.import_module

    def load(name, *args, **kwargs):
        if name == 'habitat_sim':
            raise OSError('libcudart.so missing')
        if name == 'magnum':
            raise ImportError('No module named magnum')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, 'import_module', load)
    assert main(['doctor', '--simulator']) == 1
    report = json.loads(capsys.readouterr().out)
    assert any('libcudart.so missing' in error for error in report['errors'])
    assert any('magnum' in error for error in report['errors'])


def test_doctor_reports_cuda_initialization_failure(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)

    def fail(_):
        raise RuntimeError('CUDA device unavailable')

    monkeypatch.setattr(torch.cuda, 'get_device_name', fail)
    assert main(['doctor', '--require-cuda']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['cuda_available'] is False
    assert any('CUDA device unavailable' in error for error in report['errors'])


@pytest.mark.parametrize('result', ['success', 'native_abort', 'timeout'])
def test_doctor_checks_simulator_rendering(monkeypatch, capsys, result):
    import awm.cli as cli
    original = cli.importlib.import_module

    def load(name, *args, **kwargs):
        if name in ('habitat_sim', 'magnum'):
            return SimpleNamespace(__file__=name, cuda_enabled=True)
        return original(name, *args, **kwargs)

    def render(command, **kwargs):
        assert command[:2] == [sys.executable, '-c']
        if result == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, -6 if result == 'native_abort' else 0,
                                           stdout='', stderr='OpenGL context creation failed')

    monkeypatch.setattr(cli.importlib, 'import_module', load)
    monkeypatch.setattr(cli.subprocess, 'run', render)
    assert main(['doctor', '--simulator']) == int(result != 'success')
    report = json.loads(capsys.readouterr().out)
    assert report['simulator_render'] is (result == 'success')
    if result == 'native_abort':
        assert any('OpenGL context creation failed' in error for error in report['errors'])
    if result == 'timeout':
        assert any('timed out' in error for error in report['errors'])


@pytest.mark.parametrize('failure', ['allocation', 'synchronize', None])
def test_doctor_checks_cuda_computation(monkeypatch, capsys, failure):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'test GPU')
    calls = []
    cpu_ones = torch.ones

    def allocate(size, *, device):
        calls.append(('allocate', device))
        if failure == 'allocation':
            raise RuntimeError('CUDA allocation failed')
        return cpu_ones(size)

    def synchronize(device):
        calls.append(('synchronize', device))
        if failure == 'synchronize':
            raise RuntimeError('CUDA kernel failed')

    monkeypatch.setattr(torch, 'ones', allocate)
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    assert main(['doctor', '--require-cuda']) == int(failure is not None)
    report = json.loads(capsys.readouterr().out)
    assert report['cuda_available'] is True
    assert report['cuda_compute'] is (failure is None)
    assert calls[0] == ('allocate', 'cuda:0')
    if failure != 'allocation':
        assert calls[-1] == ('synchronize', 0)
    if failure:
        assert any('CUDA computation failed' in error for error in report['errors'])


@pytest.fixture
def assets(tmp_path):
    checkpoint = tmp_path / 'checkpoint with spaces'
    (checkpoint / 'transformer').mkdir(parents=True)
    (checkpoint / 'transformer/config.json').write_text(json.dumps({
        'max_attn_chunks': 4, 'use_continuous_history': True,
        'use_special_memory': False, 'chunk_overlap_frames': 0,
        'action_dim': 3, 'action_temporal_scale': 1.0,
        'num_special_tokens': 8, 'special_memory_recent_chunks': 2,
    }))
    torch.save({'global_step': 6500, 'args': {
        'video_sample_size': [288, 512], 'video_sample_n_frames': 9,
        'video_sample_stride': 1, 'num_train_past_chunks': 3,
        'max_attn_chunks': 4, 'chunk_overlap_frames': 0,
        'use_continuous_history': True, 'use_special_memory': False,
        'traj_normalizer': 'symlog', 'symlog_c': 0.01, 'mixed_precision': 'bf16',
    }}, checkpoint / 'train_meta.pt')
    (checkpoint / 'ar_model.pt').write_bytes(b'fixture')
    (checkpoint / 'transformer/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'Wan2.2_VAE.pth').touch()
    (base / 'models_t5_umt5-xxl-enc-bf16.pth').touch()
    (base / 'google/umt5-xxl').mkdir(parents=True)
    image = tmp_path / 'image.png'
    Image.new('RGB', (64, 64)).save(image)
    return checkpoint, base, image


def test_cli_help_works_without_loading_model_stack(tmp_path):
    code = "import sys; from awm.cli import main; assert 'torch' not in sys.modules; main(['--help'])"
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert 'infer' in result.stdout


@pytest.mark.parametrize('decode_video', [False, True])
def test_dry_run_uses_checkpoint_and_preserves_literal_prompt(assets, capsys, decode_video):
    checkpoint, base, image = assets
    prompt = 'Go to "chair"; $(touch NOT_EXECUTED)'
    assert main(['infer', '--checkpoint', str(checkpoint), '--base-model', str(base),
                 '--image', str(image), '--prompt', prompt, '--dry-run']
                + (['--decode-video'] if decode_video else [])) == 0
    report = json.loads(capsys.readouterr().out)
    command = report['command']
    assert report['contract']['image_height'] == 288
    assert report['contract']['ar_num_past_chunks'] == 3
    assert report['contract']['max_attn_chunks'] == 4
    assert report['contract']['use_continuous_history']
    assert '--use_special_memory' not in command
    assert command[command.index('--prompt')+1] == prompt
    assert command[command.index('--action_shift')+1] == '1.0'
    assert ('--decode-video' if decode_video else '--no-decode-video') in command


def test_missing_base_component_is_reported(assets, capsys):
    checkpoint, base, image = assets
    (base / 'Wan2.2_VAE.pth').unlink()
    with pytest.raises(SystemExit) as error:
        main(['infer', '--checkpoint', str(checkpoint), '--base-model', str(base),
              '--image', str(image), '--prompt', 'Move', '--dry-run'])
    assert error.value.code == 2
    assert 'Wan2.2_VAE.pth' in capsys.readouterr().err


def test_unimplemented_sequential_offload_is_rejected_before_model_loading(capsys):
    with pytest.raises(SystemExit) as error:
        main(['infer', '--checkpoint', 'x', '--base-model', 'x', '--image', 'x',
              '--prompt', 'Move', '--memory-mode', 'sequential_cpu_offload'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_checksum_rejects_modified_weights(assets, capsys):
    checkpoint, _, _ = assets
    with pytest.raises(SystemExit) as error:
        main(['check', '--checkpoint', str(checkpoint), '--sha256'])
    assert error.value.code == 2
    assert 'SHA-256 mismatch' in capsys.readouterr().err


def test_server_defaults_are_local_and_history_mode_explicit(assets, capsys):
    checkpoint, base, _ = assets
    assert main(['serve', '--checkpoint', str(checkpoint), '--base-model', str(base),
                 '--history', 'predicted', '--dry-run']) == 0
    command = json.loads(capsys.readouterr().out)['command']
    assert command[command.index('--host')+1] == '127.0.0.1'
    assert '--no-gt-past' in command
    assert '--use-cuda-graph' in command
    assert '--no-decode-predicted-video' in command
    assert '--auto-stop-on-near-zero' in command
    assert command[command.index('--motion-stop-n-frames')+1] == '9'
    assert command[command.index('--min-steps-before-stop')+1] == '10'
    assert command[command.index('--motion-stop-xy-threshold-m')+1] == '0.0325'
    assert command[command.index('--motion-stop-yaw-threshold-rad')+1] == '0.0325'
    assert '--no-confirm-stop' in command
    assert command[command.index('--action-shift')+1] == '1.0'


def test_server_offline_recording_is_opt_in(assets, capsys, tmp_path):
    checkpoint, base, _ = assets
    args = ['serve', '--checkpoint', str(checkpoint), '--base-model', str(base), '--dry-run']
    assert main(args) == 0
    assert '--record-dir' not in json.loads(capsys.readouterr().out)['command']
    assert main(args + ['--record-dir', str(tmp_path / 'records')]) == 0
    command = json.loads(capsys.readouterr().out)['command']
    assert command[command.index('--record-dir') + 1] == str(tmp_path / 'records')


@pytest.mark.parametrize('value', ['0', '-1'])
def test_zero_or_negative_steps_rejected(value):
    with pytest.raises(SystemExit) as error:
        main(['infer', '--checkpoint', 'x', '--base-model', 'x', '--image', 'x', '--prompt', 'x', '--steps', value])
    assert error.value.code == 2


@pytest.mark.parametrize('stop_args,expected_stop', [
    ([], (9, 10, 0.0325, 0.0325)),
    (['--motion-stop-n-frames', '13', '--min-steps-before-stop', '24',
      '--motion-stop-xy-threshold-m', '0.04', '--motion-stop-yaw-threshold-rad', '0.05'],
     (13, 24, 0.04, 0.05)),
])
def test_server_parses_action_and_stop_policy(assets, monkeypatch, stop_args, expected_stop):
    from types import SimpleNamespace
    from awm.scripts import awm_policy_server as server
    checkpoint, base, _ = assets
    captured = {}

    def fake_agent(config):
        captured['config'] = config
        return SimpleNamespace(config=config, _step_count=0)

    class FakeHTTPServer:
        def __init__(self, address, handler):
            captured['address'] = address

        def serve_forever(self):
            pass

    monkeypatch.setattr(server, 'AWMAgent', fake_agent)
    monkeypatch.setattr(server, 'ThreadingHTTPServer', FakeHTTPServer)
    monkeypatch.setattr(sys, 'argv', ['server', '--model-path', str(checkpoint),
                                     '--pretrained-model-path', str(base),
                                     '--action-shift', '1.25', '--host', '127.0.0.1'] + stop_args)
    server.main()
    config = captured['config']
    assert (config.motion_stop_n_frames, config.min_steps_before_stop,
            config.motion_stop_xy_threshold_m, config.motion_stop_yaw_threshold_rad) == expected_stop
    assert config.auto_stop_on_near_zero
    assert not config.disable_stop
    assert captured['config'].action_shift == 1.25
    assert captured['config'].use_continuous_history is True
    assert captured['config'].ar_num_past_chunks == 3
    assert captured['address'][0] == '127.0.0.1'


def test_export_removes_private_metadata_and_preserves_contract(assets, tmp_path):
    from awm.release import FILES, export_checkpoint, file_sha256
    from awm.utils.deploy_contract import load_deploy_contract
    checkpoint, _, _ = assets
    metadata = torch.load(checkpoint / 'train_meta.pt', weights_only=True)
    metadata['args']['data_root'] = '/private/company/data'
    metadata['args']['wandb_project'] = 'internal-project'
    torch.save(metadata, checkpoint / 'train_meta.pt')
    expected = {name: file_sha256(checkpoint / name) for name in FILES}
    output = tmp_path / 'public'
    report = export_checkpoint(checkpoint, output, expected)
    public = torch.load(output / 'train_meta.pt', weights_only=True)
    assert 'data_root' not in public['args']
    assert 'wandb_project' not in public['args']
    assert public['global_step'] == 6500
    before, after = load_deploy_contract(checkpoint), load_deploy_contract(output)
    for key in set(before) - {'checkpoint', 'source_files', 'origins'}:
        assert before[key] == after[key]
    assert report['files']['transformer/diffusion_pytorch_model.safetensors'] == expected['transformer/diffusion_pytorch_model.safetensors']
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, output, expected)
    assert torch.load(checkpoint / 'train_meta.pt', weights_only=True)['args']['data_root'] == '/private/company/data'


def test_export_rejects_corruption_without_leaving_partial_directory(assets, tmp_path):
    from awm.release import FILES, export_checkpoint, file_sha256
    checkpoint, _, _ = assets
    expected = {name: file_sha256(checkpoint / name) for name in FILES}
    (checkpoint / 'transformer/diffusion_pytorch_model.safetensors').write_bytes(b'corrupt')
    output = tmp_path / 'public'
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        export_checkpoint(checkpoint, output, expected)
    assert not output.exists()


def test_visualization_overrides_do_not_change_stop_policy(assets, capsys):
    checkpoint, base, _ = assets
    main(['serve', '--checkpoint', str(checkpoint), '--base-model', str(base),
          '--no-cuda-graph', '--decode-video', '--dry-run'])
    command = json.loads(capsys.readouterr().out)['command']
    assert '--no-cuda-graph' in command
    assert '--decode-predicted-video' in command
    assert '--auto-stop-on-near-zero' in command


@pytest.mark.parametrize('decode_video', [False, True])
def test_image_entrypoint_uses_saved_geometry_and_writes_physical_actions(assets, tmp_path, monkeypatch, decode_video):
    import numpy as np
    from awm.scripts import inference
    from awm.habitat_extensions import awm_agent
    checkpoint, base, image = assets
    captured = {}
    expected = np.tile(np.array([0.1, -0.02, 0.03], dtype=np.float32), (9, 1))

    class Agent:
        def __init__(self, config):
            captured['config'] = config
            self.last_generated_video = np.zeros((9, 288, 512, 3), dtype=np.uint8)
        def reset(self, prompt):
            captured['prompt'] = prompt
        def _infer_continuous_trajectory(self, rgb):
            captured['shape'] = rgb.shape
            return expected

    monkeypatch.setattr(awm_agent, 'AWMAgent', Agent)
    monkeypatch.setattr(inference, 'save_rgb_video', lambda frames, path, fps: captured.update(video=(frames.shape, path.name, fps)))
    output = tmp_path / 'prediction'
    monkeypatch.setattr(sys, 'argv', ['inference', '--model_path', str(checkpoint),
        '--pretrained_model_path', str(base), '--config', str(CONFIG_DIR / 'inference.yaml'),
        '--input_image', str(image), '--prompt', 'Go to the chair.', '--output_dir', str(output), '--device', 'cpu']
        + (['--decode-video'] if decode_video else []))
    inference.main()
    assert captured['shape'] == (288, 512, 3)
    assert captured['config'].ar_num_past_chunks == 3
    assert captured['config'].use_continuous_history
    assert captured['config'].action_shift == 1.0
    assert captured['config'].decode_predicted_video is decode_video
    assert captured['prompt'] == 'Go to the chair.'
    np.testing.assert_array_equal(np.load(output / 'image_trajectory.npy'), expected)
    with Image.open(output / 'image_trajectory.png') as plot:
        assert plot.size == (1620, 594)
        assert len(plot.convert('RGB').getcolors(plot.width * plot.height)) > 10
    if decode_video:
        assert captured['video'] == ((9, 288, 512, 3), 'image_generated_raw.mp4', 4)
    else:
        assert 'video' not in captured
