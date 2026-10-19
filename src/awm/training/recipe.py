"""Paper configuration and optimizer-update schedules."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'training.yaml'


def load_pipeline(name):
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    stages = config['pipelines'][name]
    if not stages or len(stages) != len(set(stages)) or any(s not in config['stages'] for s in stages):
        raise ValueError(f'Invalid training pipeline: {name}')
    return stages


def load_recipe(stage):
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    if stage not in config['stages']:
        raise ValueError(f'Unknown stage: {stage}')
    recipe = copy.deepcopy(config['common'])
    recipe.update(config['stages'][stage])
    recipe['stage'] = stage
    return recipe


def checkpoint_step(recipe, step=None):
    offset = load_recipe('p0')['updates'] if recipe['stage'] == 'p1' else 0
    return offset + (recipe['updates'] if step is None else step)


def accumulation_steps(recipe, world_size, micro_batch):
    denominator = world_size * micro_batch
    if min(world_size, micro_batch) < 1 or recipe['global_batch'] % denominator:
        raise ValueError('world_size × micro_batch must divide the paper global batch size')
    return recipe['global_batch'] // denominator


def lr_multiplier(step, recipe):
    warmup = recipe['warmup_updates']
    if warmup and step < warmup:
        return step / warmup
    start, end = recipe.get('decay_start'), recipe.get('decay_end')
    if start is None or step <= start:
        return 1.0
    floor = recipe['min_lr'] / recipe['learning_rate']
    if step >= end:
        return floor
    return floor + (1 - floor) * (1 + math.cos(math.pi * (step - start) / (end - start))) / 2


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
