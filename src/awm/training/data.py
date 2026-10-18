"""Cached navigation data and reproducible distributed batch sampling."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def collate_samples(examples):
    if not examples or not all(ex.get('is_precomputed') for ex in examples):
        raise ValueError('Training requires prepared navigation latent caches')
    batch = {'text': [ex['text'] for ex in examples],
             'has_traj': torch.tensor([ex['has_traj'] for ex in examples], dtype=torch.bool)}
    fields = ('latents', 'trajectory', 'trajectory_loss_mask', 'video_frame_loss_mask')
    for key in fields:
        batch[key] = torch.stack([torch.as_tensor(ex[key]).float() for ex in examples])
    if any(ex.get('inpaint_latent') is None for ex in examples):
        raise ValueError('Cache is missing exact first-observation inpaint latents; rebuild it')
    batch['inpaint_latents'] = torch.stack([ex['inpaint_latent'].float() for ex in examples])
    for key in ('past_latents', 'past_trajectories'):
        lengths = {len(ex.get(key, [])) for ex in examples}
        if len(lengths) != 1:
            raise ValueError(f'Inconsistent history slots for {key}')
        if next(iter(lengths)):
            batch[key] = [torch.stack([torch.as_tensor(ex[key][slot]).float() for ex in examples])
                          for slot in range(next(iter(lengths)))]
    for key in ('past_chunk_mask', 'past_chunk_position_ids', 'current_chunk_position_id'):
        if key in examples[0]:
            batch[key] = torch.stack([torch.as_tensor(ex[key]) for ex in examples])
    return batch


class SeededSources(Dataset):
    def __init__(self, video, action=None):
        self.sources = [video, action]

    def __len__(self):
        return sum(len(source) for source in self.sources if source is not None)

    def __getitem__(self, item):
        source_id, index, seed = item
        source = self.sources[source_id]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            if source.num_past_chunks:
                return source.get_multi_chunk_batch(index, source.num_past_chunks)
            return source.get_batch(index)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class UpdateBatchSampler(Sampler):
    def __init__(self, lengths, replay_probability, micro_batch, total_micro_batches,
                 seed, rank=0, world_size=1, start_micro_batch=0):
        self.lengths = lengths
        self.probability = replay_probability
        self.micro_batch = micro_batch
        self.total = total_micro_batches
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.start = start_micro_batch

    def __iter__(self):
        for micro in range(self.start, self.total):
            batch = []
            for offset in range(self.micro_batch):
                sample_number = (micro * self.world_size + self.rank) * self.micro_batch + offset
                digest = hashlib.blake2b(f'{self.seed}:{sample_number}'.encode(), digest_size=8).digest()
                seed = int.from_bytes(digest, 'little')
                rng = random.Random(seed)
                source = 0 if rng.random() < self.probability else 1
                batch.append((source, rng.randrange(self.lengths[source]), rng.getrandbits(63)))
            yield batch

    def __len__(self):
        return self.total - self.start


def build_sources(args, recipe):
    from awm.data.datasets.nav_t2v_cached import NavT2VCachedDataset
    from awm.data.datasets.nav_wam_cached import NavWamCachedDataset

    common = dict(video_sample_size=(recipe['height'], recipe['width']),
                  video_sample_n_frames=recipe['frames'], video_sample_stride=1,
                  chunk_overlap_frames=0, text_drop_ratio=recipe['text_dropout'],
                  num_past_chunks=recipe['past_chunks'], preserve_first_chunk=bool(recipe['past_chunks']),
                  action_dim=3)
    video = NavT2VCachedDataset(meta_path='<video-cache>', latent_dir=args.video_cache,
                                uniform_chunk_sampling=True, **common)
    expected_video = len(video.manifest['samples'])
    if len(video.samples) != expected_video:
        raise ValueError(f'Video cache is missing samples: {len(video.samples)}/{expected_video}')
    action = None
    if recipe['phase'] != 'video_pretrain':
        if not all((args.action_cache, args.action_meta, args.action_split)):
            raise ValueError('Action stages require --action-cache, --action-meta and --action-split')
        action = NavWamCachedDataset(meta_path=args.action_meta, latent_dir=args.action_cache,
                                    split_file=args.action_split, text_level='canonical',
                                    chunk_mode='ceil',
                                    uniform_chunk_sampling=True, **common)
        wanted = {line.strip() for line in Path(args.action_split).read_text().splitlines() if line.strip()}
        if len(action.samples) != len(wanted):
            raise ValueError(f'Action cache/meta misses split episodes: {len(action.samples)}/{len(wanted)}')
        expected_hash = action.manifest.get('source_meta_sha256')
        actual_hash = hashlib.sha256(Path(args.action_meta).read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError('Action cache was built from a different metadata file')
        for name, value in [('video_sample_size', [recipe['height'], recipe['width']])]:
            if action.manifest.get(name) != value:
                raise ValueError(f'Action cache has incompatible {name}')
    paths = [Path(args.video_cache) / 'manifest.json']
    if action is not None:
        paths += [Path(args.action_cache) / 'manifest.json', Path(args.action_meta), Path(args.action_split)]
    fingerprints = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return SeededSources(video, action), fingerprints
