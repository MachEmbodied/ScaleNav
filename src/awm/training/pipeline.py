"""Launch each training phase in a fresh distributed process group."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .recipe import accumulation_steps, checkpoint_step, load_pipeline, load_recipe


def recover_next_phase(entry, prior_contract):
    output = Path(entry['checkpoint']).parent
    run = output / 'run.json'
    if not run.is_file():
        return
    contract = {**prior_contract, 'recipe': entry['recipe']}
    if json.loads(run.read_text()) != contract:
        raise ValueError(f'Existing phase configuration/data differ from the resumed run: {output}')
    checkpoints = []
    for path in output.glob('checkpoint-*'):
        if not path.is_dir() or path.name.endswith('.partial'):
            continue
        info = json.loads((path / 'training.json').read_text())
        step = info.get('step', 0)
        if (not info.get('complete') or info.get('contract') != contract
                or not 1 <= step <= entry['recipe']['updates']
                or path.name != f"checkpoint-{checkpoint_step(entry['recipe'], step)}"):
            raise ValueError(f'Invalid recovery checkpoint: {path}')
        checkpoints.append((step, path))
    if checkpoints:
        command = entry['command']
        position = command.index('--initialize-from')
        command[position:position + 2] = ['--resume', str(max(checkpoints)[1])]
    else:
        number = 1
        while True:
            archive = output.with_name(f'{output.name}.interrupted-{number}')
            if not archive.exists() and not archive.is_symlink():
                entry['archive_interrupted_output'] = str(archive)
                break
            number += 1


def plan(args):
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run pretrain/posttrain directly with --gpus, without torchrun')
    if args.micro_batch < 1 or args.workers < 0 or (args.save_every is not None and args.save_every < 1):
        raise ValueError('Invalid micro-batch, worker or checkpoint interval')
    stages = load_pipeline(args.stage)
    if args.stage == 'pretrain' and args.initialize_from:
        raise ValueError('Pretraining initializes from --base-model; use --resume to continue it')
    if args.stage == 'posttrain':
        if not (args.initialize_from or args.resume or args.check_data):
            raise ValueError('Post-training requires --initialize-from PRETRAIN/final or --resume')
        if not all((args.action_cache, args.action_meta, args.action_split)):
            raise ValueError('Post-training requires --action-cache, --action-meta and --action-split')
    output = Path(args.output)
    start = 0
    if args.resume:
        checkpoint = Path(args.resume)
        info = json.loads((checkpoint / 'training.json').read_text())
        stage = info.get('contract', {}).get('recipe', {}).get('stage')
        if stage not in stages:
            raise ValueError(f'Resume checkpoint does not belong to {args.stage}')
        if checkpoint.resolve().parent != (output / stage).resolve():
            raise ValueError('--resume must belong to the selected pipeline output directory')
        start = stages.index(stage)
    entries = []
    initialization = args.initialize_from
    for index in range(start, len(stages)):
        stage = stages[index]
        recipe = load_recipe(stage)
        if index and recipe['initialize_from'] != stages[index - 1]:
            raise ValueError(f'Invalid phase transition: {stages[index - 1]} to {stage}')
        maximum = args.max_updates if index == start else None
        if maximum is not None and not 1 <= maximum <= recipe['updates']:
            raise ValueError('--max-updates must be between 1 and the current phase length')
        phase_output = output / stage
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={args.gpus}', '--module', 'awm.training.runner',
                   '--stage', stage, '--base-model', args.base_model,
                   '--video-cache', args.video_cache, '--output', str(phase_output),
                   '--micro-batch', str(args.micro_batch), '--workers', str(args.workers),
                   '--seed', str(args.seed), '--save-every', str(args.save_every or recipe['updates']),
                   '--text-device', args.text_device]
        for option in ('action_cache', 'action_meta', 'action_split'):
            if getattr(args, option):
                command.extend(['--' + option.replace('_', '-'), getattr(args, option)])
        if index == start and args.resume:
            command.extend(['--resume', args.resume])
        elif initialization:
            command.extend(['--initialize-from', initialization])
        if maximum is not None:
            command.extend(['--max-updates', str(maximum)])
        for option in ('deepspeed', 'check_data'):
            if getattr(args, option):
                command.append('--' + option.replace('_', '-'))
        checkpoint = phase_output / f'checkpoint-{checkpoint_step(recipe)}'
        entries.append({'stage': stage, 'recipe': recipe, 'command': command,
                        'checkpoint': str(checkpoint),
                        'gradient_accumulation_steps': accumulation_steps(recipe, args.gpus, args.micro_batch),
                        'completes_phase': maximum is None or maximum == recipe['updates']})
        if args.resume and index > start and not args.check_data:
            recover_next_phase(entries[-1], info['contract'])
        initialization = str(checkpoint)
        if not entries[-1]['completes_phase']:
            break
    return entries


def require_completed_phase(entry):
    checkpoint = Path(entry['checkpoint'])
    info = json.loads((checkpoint / 'training.json').read_text())
    if (not info.get('complete') or info.get('step') != entry['recipe']['updates']
            or info.get('contract', {}).get('recipe') != entry['recipe']):
        raise ValueError(f'Phase did not produce a completed checkpoint: {checkpoint}')
    if not (checkpoint / 'state').is_dir() or not any((checkpoint / 'state').iterdir()):
        raise ValueError(f'Training state is missing: {checkpoint}')
    from awm.utils.checkpoint import checkpoint_weight_files
    checkpoint_weight_files(checkpoint / 'transformer')
    for name in ('train_meta.pt', 'transformer/config.json'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Missing checkpoint file: {checkpoint / name}')


def run(args):
    entries = plan(args)
    final = Path(args.output) / 'final'
    target = Path(entries[-1]['checkpoint'])
    if args.dry_run:
        print(json.dumps({'pipeline': args.stage, 'gpus': args.gpus,
                          'phases': entries, 'final': str(final) if not args.check_data
                          and all(entry['completes_phase'] for entry in entries) else None}, indent=2))
        return 0
    if final.exists() or final.is_symlink():
        if not final.is_symlink() or final.resolve() != target.resolve():
            raise FileExistsError(f'Final checkpoint path already exists: {final}')
    for entry in entries:
        if 'archive_interrupted_output' in entry:
            archive = Path(entry['archive_interrupted_output'])
            if archive.exists() or archive.is_symlink():
                raise FileExistsError(f'Interrupted-run archive already exists: {archive}')
            Path(entry['checkpoint']).parent.rename(archive)
            print(f'Preserved interrupted phase: {archive}', flush=True)
        print(f"Starting {args.stage}: {entry['stage']}", flush=True)
        result = subprocess.run(entry['command'])
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
        if args.check_data:
            continue
        if not entry['completes_phase']:
            return 0
        require_completed_phase(entry)
    if not args.check_data:
        if not final.is_symlink():
            final.symlink_to(os.path.relpath(target.resolve(), final.parent.resolve()), target_is_directory=True)
        print(f'Final checkpoint: {final}', flush=True)
    return 0
