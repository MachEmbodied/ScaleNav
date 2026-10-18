"""Portable training entry point with explicit stage initialization and resume."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .recipe import accumulation_steps, load_recipe, lr_multiplier, write_json


def parser():
    result = argparse.ArgumentParser(description='Train the ScaleNav paper recipe')
    result.add_argument('--stage', choices=['p0', 'p1', 'align', 'joint'], required=True)
    result.add_argument('--base-model', required=True)
    result.add_argument('--video-cache', required=True)
    result.add_argument('--action-cache')
    result.add_argument('--action-meta')
    result.add_argument('--action-split')
    result.add_argument('--output', required=True)
    source = result.add_mutually_exclusive_group()
    source.add_argument('--initialize-from', help='Previous stage checkpoint; P0-to-P1 also restores optimizer state')
    source.add_argument('--resume', help='Complete checkpoint from this stage, including training state')
    result.add_argument('--micro-batch', type=int, default=1)
    result.add_argument('--workers', type=int, default=4)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--save-every', type=int, default=500)
    result.add_argument('--max-updates', type=int, help='Stop early for a smoke run; does not change the LR schedule')
    result.add_argument('--deepspeed', action='store_true', help='ZeRO-2 with CPU optimizer offload')
    result.add_argument('--text-device', choices=['cuda', 'cpu'], default='cuda',
                        help='Device for the frozen text encoder')
    result.add_argument('--dry-run', action='store_true', help='Print resolved parameters without loading models')
    result.add_argument('--check-data', action='store_true', help='Validate data and sample a batch, then exit')
    return result


def load_model(base_model, recipe, initialization=None):
    import torch
    from safetensors import safe_open
    from awm.models.causal_chunk_ar_model import CausalChunkDiTModel, CausalChunkDiTWrapper
    from awm.utils.checkpoint import checkpoint_weight_files, load_causal_checkpoint

    overrides = dict(max_attn_chunks=recipe['max_attn_chunks'], action_temporal_scale=1.0,
                     use_continuous_history=bool(recipe['past_chunks']), use_special_memory=False,
                     chunk_overlap_frames=0, action_dim=3, traj_head_type=recipe['traj_head_type'],
                     traj_head_hidden_dim=recipe['traj_head_hidden_dim'])
    if initialization:
        return load_causal_checkpoint(initialization, 'cpu', torch.float32, overrides)
    config = json.loads((Path(base_model) / 'config.json').read_text())
    config.update(overrides)
    config.update(cross_attn_type='cross_attn', in_channels=config['in_dim'], hidden_size=config['dim'])
    model = CausalChunkDiTModel.from_config(config)
    expected, found = model.state_dict(), set()
    with torch.no_grad():
        for path in checkpoint_weight_files(base_model):
            with safe_open(str(path), framework='pt', device='cpu') as weights:
                for name in weights.keys():
                    if name in found or name not in expected:
                        raise ValueError(f'Unexpected or duplicate Wan tensor: {name}')
                    if tuple(weights.get_slice(name).get_shape()) != tuple(expected[name].shape):
                        raise ValueError(f'Incompatible Wan tensor shape: {name}')
                    expected[name].copy_(weights.get_tensor(name))
                    found.add(name)
    missing = set(expected) - found
    if any(not name.startswith(('traj_embed.', 'traj_head.')) for name in missing):
        raise ValueError(f'Incomplete Wan backbone: {sorted(missing)}')
    return CausalChunkDiTWrapper(model)


def load_text_encoder(base_model, device):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from awm.models import WanT5EncoderModel

    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'inference.yaml'
    config = yaml.safe_load(config_path.read_text())['text_encoder_kwargs']
    tokenizer = AutoTokenizer.from_pretrained(Path(base_model) / config['tokenizer_subpath'])
    encoder = WanT5EncoderModel.from_pretrained(
        str(Path(base_model) / config['text_encoder_subpath']), additional_kwargs=config,
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    ).eval().requires_grad_(False).to(device)
    return tokenizer, encoder


def encode_text(text, tokenizer, encoder, device):
    import torch
    with torch.no_grad():
        tokens = tokenizer(text, padding='longest', max_length=512, truncation=True,
                           add_special_tokens=True, return_tensors='pt').to(device)
        embeddings = encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
        lengths = tokens.attention_mask.sum(-1).tolist()
    return [embedding[:length] for embedding, length in zip(embeddings, lengths)]


def deployment_metadata(recipe, step):
    return {'global_step': step + (2000 if recipe['stage'] == 'p1' else 0), 'stage': recipe['stage'],
            'stage_step': step, 'args': {
                'video_sample_size': [recipe['height'], recipe['width']],
                'video_sample_n_frames': recipe['frames'], 'video_sample_stride': 1,
                'num_train_past_chunks': recipe['past_chunks'], 'max_attn_chunks': recipe['max_attn_chunks'],
                'use_continuous_history': bool(recipe['past_chunks']), 'use_special_memory': False,
                'chunk_overlap_frames': 0, 'action_dim': 3, 'action_temporal_scale': 1.0,
                'traj_normalizer': 'symlog', 'symlog_c': recipe['symlog_c'], 'mixed_precision': 'bf16',
            }}


def validate_initialization(path, stage):
    import torch
    previous, expected_step = {'p1': ('p0', 2000), 'align': ('p1', 10000),
                               'joint': ('align', 2500)}[stage]
    metadata = torch.load(Path(path) / 'train_meta.pt', map_location='cpu', weights_only=True)
    if metadata.get('global_step') != expected_step:
        raise ValueError(f'{stage} requires the {previous} checkpoint at update {expected_step}')
    if metadata.get('stage', previous) != previous:
        raise ValueError(f'{stage} cannot initialize from stage {metadata.get("stage")}')
    if stage == 'p1':
        info_path = Path(path) / 'training.json'
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        if not info.get('complete') or not (Path(path) / 'state').is_dir():
            raise ValueError('P1 requires the completed P0 checkpoint with optimizer state')


def save_checkpoint(accelerator, objective, output, recipe, step, contract):
    import torch
    from awm.utils.checkpoint import save_causal_weights
    name = f"checkpoint-{step + (2000 if recipe['stage'] == 'p1' else 0)}"
    final = Path(output) / name
    staging = final.with_name(name + '.partial')
    if accelerator.is_main_process:
        if final.exists() or staging.exists():
            raise FileExistsError(f'Checkpoint destination already exists: {final}')
        staging.mkdir(parents=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(staging / 'state'))
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(objective).model.dit_model
        # DDP and ZeRO-2 keep full model parameters on each rank.
        state = {name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                 for name, value in model.state_dict().items()}
        save_causal_weights(model, state, staging / 'transformer')
        torch.save({}, staging / 'ar_model.pt')
        torch.save(deployment_metadata(recipe, step), staging / 'train_meta.pt')
        write_json(staging / 'training.json', {'step': step, 'contract': contract, 'complete': True})
        staging.rename(final)
    accelerator.wait_for_everyone()
    return final


def require_training_device(accelerator):
    if accelerator.device.type != 'cuda':
        raise ValueError('Full model training requires CUDA')


def parameter_groups(parameters, max_elements=250_000_000):
    """Bound DeepSpeed's temporary contiguous parameter buffers."""
    groups, current, size = [], [], 0
    for parameter in parameters:
        if current and size + parameter.numel() > max_elements:
            groups.append({'params': current})
            current, size = [], 0
        current.append(parameter)
        size += parameter.numel()
    if current:
        groups.append({'params': current})
    return groups


def train(args, recipe):
    import torch
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DistributedDataParallelKwargs, send_to_device, set_seed
    from torch.utils.data import DataLoader
    from .data import UpdateBatchSampler, build_sources, collate_samples
    from .objective import FlowObjective, set_trainable

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    accumulation = accumulation_steps(recipe, world_size, args.micro_batch)
    plugin = None
    if args.deepspeed:
        plugin = DeepSpeedPlugin(hf_ds_config={
            'train_micro_batch_size_per_gpu': args.micro_batch,
            'gradient_accumulation_steps': accumulation,
            'train_batch_size': recipe['global_batch'],
            'gradient_clipping': recipe['max_grad_norm'],
            'bf16': {'enabled': False},
            'torch_autocast': {'enabled': True, 'dtype': 'bfloat16'},
            'zero_optimization': {'stage': 2, 'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                                  'ignore_unused_parameters': True,
                                  'reduce_bucket_size': 50_000_000, 'allgather_bucket_size': 50_000_000},
            'zero_allow_untested_optimizer': True,
        })
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=accumulation,
                              deepspeed_plugin=plugin, step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.num_processes != world_size:
        raise ValueError('Launcher world size does not match the distributed training environment')
    if not args.check_data:
        require_training_device(accelerator)
    set_seed(args.seed, device_specific=True)
    sources, fingerprints = build_sources(args, recipe)
    text_device = torch.device('cpu') if getattr(args, 'text_device', 'cuda') == 'cpu' else accelerator.device
    contract = {'recipe': recipe, 'data': fingerprints, 'world_size': world_size,
                'micro_batch': args.micro_batch, 'seed': args.seed, 'deepspeed': args.deepspeed,
                'text_device': text_device.type}
    start_step = 0
    if args.resume:
        info = json.loads((Path(args.resume) / 'training.json').read_text())
        if not info.get('complete') or not (Path(args.resume) / 'state').is_dir():
            raise ValueError('Resume needs a complete training checkpoint with optimizer and RNG state')
        if info['contract'] != contract:
            raise ValueError('Resume configuration/data differ from the saved training contract')
        start_step = info['step']
    maximum = args.max_updates or recipe['updates']
    if start_step >= maximum:
        raise ValueError('Checkpoint has already reached the requested update count')
    sampling_offset = (2000 if recipe['stage'] == 'p1' else 0) * accumulation
    sampler = UpdateBatchSampler([len(source) if source is not None else 0 for source in sources.sources],
                                 recipe['replay_probability'], args.micro_batch,
                                 sampling_offset + maximum * accumulation,
                                 args.seed, accelerator.process_index, world_size,
                                 sampling_offset + start_step * accumulation)
    loader = DataLoader(sources, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_samples, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    if args.check_data:
        batch = next(iter(loader))
        print(json.dumps({'data_checked': True, 'sources': [len(s) if s is not None else 0 for s in sources.sources],
                          'latent_shape': list(batch['latents'].shape), 'contract': contract}, indent=2))
        return 0
    if args.initialize_from:
        validate_initialization(args.initialize_from, recipe['stage'])
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        previous = output / 'run.json'
        if previous.exists() and not args.resume:
            raise FileExistsError('Output already contains a run; use --resume or a new --output')
        write_json(previous, contract)
    accelerator.wait_for_everyone()
    model = load_model(args.base_model, recipe, args.resume or args.initialize_from)
    model.train()
    model.enable_gradient_checkpointing()
    parameters = set_trainable(model, recipe['phase'])
    objective = FlowObjective(model, recipe)
    optimizer = torch.optim.AdamW(parameter_groups(parameters) if args.deepspeed else parameters,
                                 lr=recipe['learning_rate'], betas=recipe['adam_betas'],
                                 eps=recipe['adam_epsilon'], weight_decay=recipe['weight_decay'])
    objective, optimizer = accelerator.prepare(objective, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: lr_multiplier(update, recipe))
    accelerator.register_for_checkpointing(scheduler)
    tokenizer, text_encoder = load_text_encoder(args.base_model, text_device)
    if args.resume:
        accelerator.load_state(str(Path(args.resume) / 'state'))
    elif recipe['stage'] == 'p1':
        source = Path(args.initialize_from)
        info = json.loads((source / 'training.json').read_text()) if (source / 'training.json').is_file() else {}
        if info.get('complete') and info.get('step') == 2000 and (source / 'state').is_dir():
            prior = info['contract']
            for key in ('world_size', 'micro_batch', 'seed', 'deepspeed', 'data', 'text_device'):
                if prior[key] != contract[key]:
                    raise ValueError(f'P0-to-P1 optimizer continuation requires matching {key}')
            accelerator.load_state(str(source / 'state'))
        else:
            raise ValueError('P1 continuation requires the completed P0 checkpoint including state; rerun P0 to reproduce training')
    step = start_step
    log_path = output / 'metrics.jsonl'
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        text = batch.pop('text')
        batch = send_to_device(batch, accelerator.device)
        contexts = [context.to(accelerator.device) for context in
                    encode_text(text, tokenizer, text_encoder, text_device)]
        with accelerator.accumulate(objective):
            with accelerator.autocast():
                loss, metrics = objective(batch, contexts)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at update {step}')
            accelerator.backward(loss)
            if accelerator.sync_gradients and not args.deepspeed:
                norm = accelerator.clip_grad_norm_(parameters, recipe['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'Nonfinite gradients at update {step}')
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients:
            step += 1
            report = {'step': step, 'loss': float(loss.detach()), 'lr': scheduler.get_last_lr()[0],
                      **{key: float(value) for key, value in metrics.items()}}
            if accelerator.is_main_process:
                with log_path.open('a') as stream:
                    stream.write(json.dumps(report) + '\n')
                if step == start_step + 1 or step % 10 == 0:
                    print(json.dumps(report), flush=True)
            if step % args.save_every == 0 or step == maximum:
                save_checkpoint(accelerator, objective, output, recipe, step, contract)
    accelerator.end_training()
    return 0


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        recipe = load_recipe(args.stage)
        if args.micro_batch < 1 or args.workers < 0 or args.save_every < 1:
            raise ValueError('Invalid micro-batch, worker or checkpoint interval')
        if args.max_updates is not None and not 1 <= args.max_updates <= recipe['updates']:
            raise ValueError('--max-updates must be between 1 and the paper stage length')
        if args.stage == 'p0' and args.initialize_from:
            raise ValueError('P0 initializes from --base-model; use --resume to continue P0')
        if args.stage != 'p0' and not (args.initialize_from or args.resume):
            raise ValueError(f"{args.stage} requires --initialize-from {recipe['initialize_from']} checkpoint or --resume")
        if args.dry_run:
            world = int(os.environ.get('WORLD_SIZE', '1'))
            print(json.dumps({'recipe': recipe, 'world_size': world,
                              'gradient_accumulation_steps': accumulation_steps(recipe, world, args.micro_batch),
                              'initialization': args.resume or args.initialize_from or args.base_model}, indent=2))
            return 0
        return train(args, recipe)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        arguments.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
