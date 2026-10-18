"""Strict, low-CPU-memory loading for the released causal transformer."""
import json
from pathlib import Path



def save_causal_weights(model, state, directory, max_shard_size='5GB'):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_config(directory)
    shards = split_torch_state_dict_into_shards(
        state, max_shard_size=max_shard_size,
        filename_pattern='diffusion_pytorch_model{suffix}.safetensors')
    for filename, names in shards.filename_to_tensors.items():
        save_file({name: state[name] for name in names}, directory / filename, metadata={'format': 'pt'})
    if shards.is_sharded:
        index = {'metadata': shards.metadata, 'weight_map': shards.tensor_to_filename}
        (directory / 'diffusion_pytorch_model.safetensors.index.json').write_text(
            json.dumps(index, indent=2, sort_keys=True) + '\n')


def checkpoint_weight_files(directory):
    directory = Path(directory)
    index = directory / 'diffusion_pytorch_model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text()).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Invalid checkpoint shard index')
        names = sorted(set(mapping.values()))
        if any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
            raise ValueError('Unsafe checkpoint shard path')
    else:
        names = ['diffusion_pytorch_model.safetensors']
    paths = [directory / name for name in names]
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing checkpoint weight file: {path}')
    return paths


def load_causal_checkpoint(checkpoint, device, dtype, config_overrides=None):
    """Materialize one tensor at a time, avoiding an intermediate full Wan model."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from awm.models.causal_chunk_ar_model import CausalChunkDiTModel, CausalChunkDiTWrapper

    directory = Path(checkpoint) / 'transformer'
    config = json.loads((directory / 'config.json').read_text())
    if config.get('_class_name') != 'CausalChunkDiTModel':
        raise ValueError('Direct loading requires a CausalChunkDiTModel checkpoint')
    config.update(config_overrides or {})
    with init_empty_weights():
        model = CausalChunkDiTModel.from_config(config)
    expected = model.state_dict()
    paths = checkpoint_weight_files(directory)
    found = {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as weights:
            for name in weights.keys():
                if name in found:
                    raise ValueError(f'Duplicate checkpoint tensor: {name}')
                found[name] = (path, tuple(weights.get_slice(name).get_shape()))
    missing, unexpected = set(expected) - set(found), set(found) - set(expected)
    if missing or unexpected:
        raise ValueError(f'Checkpoint tensor names mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for name, target in expected.items():
        if tuple(target.shape) != found[name][1]:
            raise ValueError(f'Checkpoint shape mismatch for {name}: expected {tuple(target.shape)}, got {found[name][1]}')
    index_path = directory / 'diffusion_pytorch_model.safetensors.index.json'
    if index_path.is_file():
        mapping = json.loads(index_path.read_text())['weight_map']
        if set(mapping) != set(found) or any(mapping[name] != found[name][0].name for name in found):
            raise ValueError('Checkpoint shard index does not match tensor files')
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as weights:
            for name in weights.keys():
                set_module_tensor_to_device(model, name, device, value=weights.get_tensor(name), dtype=dtype)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Checkpoint left unmaterialized model parameters')
    model.eval().requires_grad_(False)
    print(f'Loaded causal checkpoint directly: {len(expected)} tensors; no missing or unexpected keys.')
    return CausalChunkDiTWrapper(model).eval().requires_grad_(False)
