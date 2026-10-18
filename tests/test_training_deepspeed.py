import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def run_training_process(command, environment, timeout=1800):
    with subprocess.Popen(command, env=environment, start_new_session=True) as process:
        try:
            returncode = process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                import psutil
                children = psutil.Process(process.pid).children(recursive=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(children, timeout=10)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


@pytest.mark.parametrize('stage', ['p0', 'align'])
def test_deepspeed_updates_and_resume_match_uninterrupted_training(tmp_path, stage):
    if not torch.cuda.is_available():
        pytest.skip('DeepSpeed training requires CUDA')
    pytest.importorskip('deepspeed')
    environment = {**os.environ, 'OMP_NUM_THREADS': '2'}
    for mode in ('continuous', 'first', 'resume'):
        run_training_process(
            [sys.executable, '-m', 'torch.distributed.run', '--standalone',
             '--nproc_per_node=1', str(Path(__file__).resolve()), mode, str(tmp_path), stage],
            environment)
    from safetensors.torch import load_file
    states = [load_file(tmp_path / name / 'checkpoint-2/transformer/diffusion_pytorch_model.safetensors')
              for name in ('continuous', 'resumed')]
    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    first = load_file(tmp_path / 'resumed/checkpoint-1/transformer/diffusion_pytorch_model.safetensors')
    assert any(not torch.equal(first[name], states[1][name]) for name in first)


@pytest.mark.parametrize('last_source', ['video', 'action'])
def test_deepspeed_mixed_accumulation_updates_action_head(tmp_path, last_source):
    if not torch.cuda.is_available():
        pytest.skip('DeepSpeed training requires CUDA')
    pytest.importorskip('deepspeed')
    run_training_process(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node=1', str(Path(__file__).resolve()),
         f'mixed-{last_source}', str(tmp_path)],
        {**os.environ, 'OMP_NUM_THREADS': '2'})
    from safetensors.torch import load_file
    state = load_file(tmp_path / 'mixed/checkpoint-1/transformer/diffusion_pytorch_model.safetensors')
    assert torch.count_nonzero(state['traj_head.head.weight']) > 0


def run_worker(mode, directory, stage='p0'):
    from test_training import tiny_model
    from test_training_runner import FixedVideoSource
    from awm.training import data, runner
    from awm.training.data import SeededSources
    from awm.training.recipe import load_recipe

    class FixedActionSource(FixedVideoSource):
        def get_batch(self, index):
            sample = super().get_batch(index)
            sample.update(has_traj=True, trajectory_loss_mask=torch.ones(9))
            return sample

    if mode.startswith('mixed-'):
        class MixedSampler:
            def __init__(self, *args):
                pass

            def __iter__(self):
                order = [1, 1, 1, 0] if mode == 'mixed-video' else [0, 1, 1, 1]
                for index, source in enumerate(order):
                    yield [(source, index % 3, index)]

        runner.load_model = lambda *args: tiny_model('align')
        runner.load_text_encoder = lambda *args: (None, None)
        runner.encode_text = lambda text, *args: [torch.ones(3, 16) for _ in text]
        data.build_sources = lambda *args: (SeededSources(FixedVideoSource(), FixedActionSource()), {'test': 'mixed'})
        data.UpdateBatchSampler = MixedSampler
        recipe = {**load_recipe('align'), 'updates': 1, 'global_batch': 4}
        args = SimpleNamespace(
            micro_batch=1, deepspeed=True, check_data=False, seed=42,
            resume=None, max_updates=1, workers=0, output=str(Path(directory) / 'mixed'),
            initialize_from=None, base_model='unused', save_every=1, text_device='cpu')
        assert runner.train(args, recipe) == 0
        return

    runner.load_model = lambda *args: tiny_model(stage)
    runner.load_text_encoder = lambda *args: (None, None)
    runner.encode_text = lambda text, *args: [torch.ones(3, 16) for _ in text]
    action_source = FixedActionSource() if stage == 'align' else None
    data.build_sources = lambda *args: (SeededSources(FixedVideoSource(), action_source), {'test': 'fixed'})
    recipe = {**load_recipe(stage), 'updates': 2, 'global_batch': 4, 'warmup_updates': 0}
    output = Path(directory) / ('continuous' if mode == 'continuous' else 'resumed')
    args = SimpleNamespace(
        micro_batch=1, deepspeed=True, check_data=False, seed=42,
        resume=str(output / 'checkpoint-1') if mode == 'resume' else None,
        max_updates=1 if mode == 'first' else 2, workers=0, output=str(output),
        initialize_from=None, base_model='unused', save_every=1, text_device='cpu')
    assert runner.train(args, recipe) == 0


if __name__ == '__main__':
    run_worker(*sys.argv[1:])
