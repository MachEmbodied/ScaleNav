import pytest
import torch
from safetensors.torch import save_file

from awm.models.causal_chunk_ar_model import CausalChunkDiTModel, CausalChunkDiTWrapper
from awm.utils.checkpoint import load_causal_checkpoint


def tiny_checkpoint(tmp_path):
    model = CausalChunkDiTModel(model_type='t2v', patch_size=(1, 2, 2), text_len=8,
        in_dim=4, dim=24, ffn_dim=48, freq_dim=32, text_dim=16, out_dim=4,
        num_heads=3, num_layers=1, action_dim=3, max_attn_chunks=4,
        use_continuous_history=True, traj_head_type='deep', traj_head_hidden_dim=16)
    torch.nn.init.normal_(model.traj_head.head.weight, std=0.02)
    directory = tmp_path / 'transformer'
    directory.mkdir()
    model.save_config(directory)
    save_file(model.state_dict(), directory / 'diffusion_pytorch_model.safetensors')
    return model


def test_direct_loader_preserves_weights_and_forward_output(tmp_path):
    torch.manual_seed(7)
    original = tiny_checkpoint(tmp_path).eval()
    loaded = load_causal_checkpoint(tmp_path, torch.device('cpu'), torch.float32)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.dit_model.state_dict()[name], value, rtol=0, atol=0)
    reference = CausalChunkDiTWrapper(original)
    inputs = dict(x=[torch.randn(4, 3, 4, 4)], traj=torch.randn(1, 9, 3),
                  t=torch.full((1, 21), 500, dtype=torch.long),
                  context=[torch.randn(3, 16)], seq_len=12)
    with torch.no_grad():
        expected_video, expected_action = reference(**inputs)
        actual_video, actual_action = loaded(**inputs)
    for expected, actual in zip(expected_video, actual_video):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'shape'])
def test_direct_loader_rejects_partial_or_incompatible_weights(tmp_path, corruption):
    model = tiny_checkpoint(tmp_path)
    state = model.state_dict()
    key = next(iter(state))
    if corruption == 'missing':
        del state[key]
    elif corruption == 'extra':
        state['unknown_tensor'] = torch.zeros(1)
    else:
        state[key] = torch.zeros(1)
    save_file(state, tmp_path / 'transformer/diffusion_pytorch_model.safetensors')
    with pytest.raises(ValueError, match='mismatch'):
        load_causal_checkpoint(tmp_path, torch.device('cpu'), torch.float32)
