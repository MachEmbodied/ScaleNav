"""Export a portable checkpoint with deployment-only metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


FILES = ('ar_model.pt', 'train_meta.pt', 'transformer/config.json',
         'transformer/diffusion_pytorch_model.safetensors')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(source, destination, expected_hashes):
    """Preserve weight bytes, whitelist metadata, and refuse existing destinations.

    expected_hashes is a trusted source manifest, not derived from the source
    being verified. A failed export removes only the newly created destination.
    """
    import torch
    from awm.utils.deploy_contract import load_deploy_contract

    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Destination must be outside the source checkpoint')
    contract = load_deploy_contract(source)
    if contract['traj_normalizer'] != 'symlog':
        raise ValueError('This release exporter supports the ScaleNav symlog contract only')
    args = {key: contract[key] for key in (
        'max_attn_chunks', 'chunk_overlap_frames', 'use_special_memory',
        'use_continuous_history', 'num_special_tokens', 'special_memory_recent_chunks',
        'action_dim', 'action_temporal_scale', 'traj_normalizer', 'symlog_c')}
    args.update(video_sample_size=[contract['image_height'], contract['image_width']],
                video_sample_n_frames=contract['num_video_frames'],
                video_sample_stride=contract['video_sample_stride'],
                num_train_past_chunks=contract['ar_num_past_chunks'],
                mixed_precision=contract['dtype'])
    # Check the original metadata before creating a new destination.
    if file_sha256(source / 'train_meta.pt') != expected_hashes['train_meta.pt']:
        raise ValueError('Source SHA-256 mismatch: train_meta.pt')
    destination.mkdir(parents=True, exist_ok=False)
    try:
        hashes = {}
        for relative in FILES:
            if relative == 'train_meta.pt':
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with (source / relative).open('rb') as incoming, target.open('xb') as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                    outgoing.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected_hashes[relative]:
                raise ValueError(f'Source SHA-256 mismatch: {relative}')
            hashes[relative] = digest.hexdigest()
        torch.save({'global_step': contract['global_step'], 'args': args}, destination / 'train_meta.pt')
        hashes['train_meta.pt'] = file_sha256(destination / 'train_meta.pt')
        recovered = load_deploy_contract(destination)
        for key in set(contract) - {'checkpoint', 'source_files', 'origins'}:
            if recovered.get(key) != contract[key]:
                raise ValueError(f'Export changed deployment contract field: {key}')
        (destination / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
        return {'files': hashes, 'global_step': contract['global_step'],
                'metadata': 'deployment fields only', 'weight_bytes_unchanged': True}
    except BaseException:
        shutil.rmtree(destination)
        raise
