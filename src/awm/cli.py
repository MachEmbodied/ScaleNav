"""Portable entry points; importing the CLI does not load the model stack."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from awm import __version__

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
DEFAULT_CONFIG = CONFIG_DIR / 'inference.yaml'
BASE_FILES = ('Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/umt5-xxl')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    for relative in ('transformer/config.json', 'train_meta.pt'):
        if not (checkpoint / relative).is_file():
            raise ValueError(f'Missing checkpoint file: {checkpoint / relative}')
    from awm.utils.checkpoint import checkpoint_weight_files
    checkpoint_weight_files(checkpoint / 'transformer')
    from awm.utils.deploy_contract import load_deploy_contract
    return load_deploy_contract(checkpoint)


def check_base(base):
    base = Path(base).expanduser().resolve()
    missing = [name for name in BASE_FILES if not (base / name).exists()]
    if missing:
        raise ValueError(f'Missing Wan base components in {base}: {", ".join(missing)}')
    return base


def build_inference_command(args, contract):
    return [sys.executable, '-m', 'awm.scripts.inference',
            '--model_path', str(Path(args.checkpoint).expanduser().resolve()),
            '--pretrained_model_path', str(Path(args.base_model).expanduser().resolve()),
            '--config', str(DEFAULT_CONFIG), '--input_image', str(Path(args.image).expanduser().resolve()),
            '--prompt', args.prompt, '--output_dir', str(Path(args.output).expanduser().resolve()),
            '--num_inference_steps', str(args.steps), '--guidance_scale', str(args.guidance),
            '--traj_guidance_scale', str(args.trajectory_guidance), '--action_shift', str(args.action_shift),
            '--gpu_memory_mode', args.memory_mode, '--seed', str(args.seed),
            '--decode-video' if args.decode_video else '--no-decode-video']


def build_server_command(args):
    command = [sys.executable, '-m', 'awm.scripts.awm_policy_server',
               '--model-path', str(Path(args.checkpoint).expanduser().resolve()),
               '--pretrained-model-path', str(Path(args.base_model).expanduser().resolve()),
               '--config-path', str(DEFAULT_CONFIG), '--checkpoint-contract', 'auto',
               '--host', args.host, '--port', str(args.port),
               '--num-inference-steps', str(args.steps), '--guidance-scale', str(args.guidance),
               '--traj-guidance-scale', str(args.trajectory_guidance), '--action-shift', str(args.action_shift),
               '--gpu-memory-mode', args.memory_mode,
               '--auto-stop-on-near-zero', '--motion-stop-n-frames', '9',
               '--min-steps-before-stop', '10',
               '--motion-stop-xy-threshold-m', '0.0325',
               '--motion-stop-yaw-threshold-rad', '0.0325', '--no-confirm-stop',
               '--decode-predicted-video' if args.decode_video else '--no-decode-predicted-video',
               '--use-cuda-graph' if args.cuda_graph else '--no-cuda-graph']
    if args.history == 'predicted':
        command.append('--no-gt-past')
    if args.record_dir:
        command.extend(['--record-dir', str(Path(args.record_dir).expanduser().resolve())])
    return command


def doctor(args):
    errors = []
    report = {'awm': __version__, 'python': sys.version.split()[0], 'ffmpeg': shutil.which('ffmpeg'),
              'config_available': DEFAULT_CONFIG.is_file()}
    try:
        import torch
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        report['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        report.update(cuda_available=False, inference_dependencies='Install .[inference]')
    except RuntimeError as exc:
        report.update(cuda_available=False)
        errors.append(f'CUDA initialization failed: {exc}')
    for package in ('diffusers', 'transformers', 'safetensors'):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
            if args.require_cuda:
                errors.append(f'Missing {package}; install .[inference]')
    if args.require_cuda and not report['cuda_available']:
        errors.append('CUDA is unavailable; check the NVIDIA driver and PyTorch build')
    if args.data_tools:
        report['ffprobe'] = shutil.which('ffprobe')
        for name in ('ffmpeg', 'ffprobe'):
            if not report[name]:
                errors.append(f'Missing {name}; install FFmpeg and add its bin directory to PATH')
    if args.simulator:
        for name in ('habitat_sim', 'magnum'):
            try:
                module = importlib.import_module(name)
                report[name] = module.__file__
                if name == 'habitat_sim' and not module.cuda_enabled:
                    errors.append('Habitat-GS lacks CUDA support; rebuild with HABITAT_WITH_CUDA=ON')
            except (ImportError, OSError) as exc:
                errors.append(f'{name}: {exc}; see docs/EVALUATION.md#simulator-installation')
    if args.require_cuda and report['cuda_available']:
        try:
            probe = torch.ones(1, device='cuda:0')
            probe.add_(1)
            torch.cuda.synchronize(0)
            report['cuda_compute'] = True
        except RuntimeError as exc:
            report['cuda_compute'] = False
            errors.append(f'CUDA computation failed: {exc}')
    if args.checkpoint:
        report['contract'] = checkpoint_contract(args.checkpoint)
    if args.base_model:
        report['base_model'] = str(check_base(args.base_model))
    report['errors'] = errors
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'train':
        from awm.training.runner import main as train_main
        return train_main(argv[1:])
    parser = argparse.ArgumentParser(prog='awm', description='ScaleNav: language-conditioned local navigation')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help='Run video pretraining or action post-training (awm train --help)')
    doc = sub.add_parser('doctor', help='Inspect installation, GPU, and optional model paths')
    doc.add_argument('--checkpoint')
    doc.add_argument('--base-model')
    doc.add_argument('--require-cuda', action='store_true')
    doc.add_argument('--data-tools', action='store_true', help='Require FFmpeg and ffprobe for data preparation')
    doc.add_argument('--simulator', action='store_true', help='Require the CUDA Habitat-GS simulator and Magnum')
    check = sub.add_parser('check', help='Validate the checkpoint contract and optionally hashes')
    check.add_argument('--checkpoint', required=True)
    check.add_argument('--sha256', action='store_true', help='Read approximately 10 GB and verify ScaleNav hashes')
    export = sub.add_parser('export-checkpoint', help='Copy approved weights with deployment-only metadata')
    export.add_argument('--checkpoint', required=True, help='ScaleNav source checkpoint')
    export.add_argument('--output', required=True, help='New directory; must not already exist')
    for name in ('infer', 'serve'):
        command = sub.add_parser(name, help='Run single-image inference' if name == 'infer' else 'Start the HTTP policy')
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--base-model', required=True)
        command.add_argument('--steps', type=positive_int, default=4)
        command.add_argument('--guidance', type=positive_float, default=5.0)
        command.add_argument('--trajectory-guidance', type=positive_float, default=1.0)
        command.add_argument('--action-shift', type=positive_float, default=1.0)
        command.add_argument('--memory-mode', choices=['model_full_load', 'model_cpu_offload'], default='model_cpu_offload')
        command.add_argument('--dry-run', action='store_true', help='Validate paths/contract and print command without loading weights')
        command.add_argument('--decode-video', action=argparse.BooleanOptionalAction, default=False,
                             help='Also decode the predicted RGB video')
        if name == 'infer':
            command.add_argument('--image', required=True)
            command.add_argument('--prompt', required=True)
            command.add_argument('--output', default='outputs/example')
            command.add_argument('--seed', type=int, default=42)
        else:
            command.add_argument('--host', default='127.0.0.1')
            command.add_argument('--port', type=int, default=8620)
            command.add_argument('--record-dir', help='Save camera, predictions and telemetry locally')
            command.add_argument('--history', choices=['observed', 'predicted'], default='observed')
            command.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            return doctor(args)
        contract = checkpoint_contract(args.checkpoint)
        if args.command == 'export-checkpoint':
            from awm.release import export_checkpoint
            expected = json.loads((CONFIG_DIR / 'checkpoint_sha256.json').read_text())
            print(json.dumps(export_checkpoint(args.checkpoint, args.output, expected), indent=2))
            return 0
        if args.command == 'check':
            if args.sha256:
                manifest = Path(args.checkpoint).expanduser() / 'SHA256SUMS.json'
                if not manifest.is_file():
                    manifest = CONFIG_DIR / 'checkpoint_sha256.json'
                hashes = json.loads(manifest.read_text())
                known = json.loads((CONFIG_DIR / 'checkpoint_sha256.json').read_text())
                if not isinstance(hashes, dict) or set(hashes) != set(known):
                    raise ValueError('Checksum manifest must contain exactly the four checkpoint files')
                # Exported metadata may differ; tensor weights/config must remain the released model files.
                for relative in set(known) - {'train_meta.pt'}:
                    if hashes[relative] != known[relative]:
                        raise ValueError(f'Unrecognized ScaleNav checksum: {relative}')
                for relative, expected in hashes.items():
                    digest = hashlib.sha256()
                    with (Path(args.checkpoint).expanduser() / relative).open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != expected:
                        raise ValueError(f'SHA-256 mismatch: {relative}')
            print(json.dumps({'contract': contract, 'sha256_verified': args.sha256}, indent=2))
            return 0
        check_base(args.base_model)
        if args.command == 'infer':
            if not args.prompt.strip():
                raise ValueError('Prompt must not be empty')
            from PIL import Image
            with Image.open(Path(args.image).expanduser()) as image:
                image.verify()
            command = build_inference_command(args, contract)
        else:
            if not 1 <= args.port <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            command = build_server_command(args)
        if args.dry_run:
            print(json.dumps({'command': command, 'contract': contract}, indent=2))
            return 0
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; run awm doctor and install a driver-compatible PyTorch build')
        return subprocess.call(command)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
