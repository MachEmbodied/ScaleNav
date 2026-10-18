import json

import pytest
import torch

from awm.utils.deploy_contract import DeployContractError, load_deploy_contract


def _checkpoint(tmp_path, *, special=False, continuous=False):
    run = tmp_path / "run"
    checkpoint = run / "joint" / "checkpoint-4000"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(json.dumps({
        "_class_name": "CausalChunkDiTModel",
        "action_dim": 3,
        "max_attn_chunks": 4 if special else 3,
        "use_special_memory": special,
        "use_continuous_history": continuous,
        "num_special_tokens": 9 if special else 8,
        "special_memory_recent_chunks": 2,
        "action_temporal_scale": 1.0,
    }))
    torch.save({
        "global_step": 4000,
        "args": {
            "video_sample_size": [288, 512],
            "video_sample_n_frames": 9,
            "video_sample_stride": 1,
            "chunk_overlap_frames": 0,
            "num_train_past_chunks": 3 if special else 2,
            # Deliberately different: deploy recovery must not use this field.
            "ar_num_past_chunks": 4,
            "max_attn_chunks": 4 if special else 3,
            "use_special_memory": special,
            "use_continuous_history": continuous,
            "num_special_tokens": 9 if special else 8,
            "special_memory_recent_chunks": 2,
            "traj_normalizer": "symlog",
            "symlog_c": 0.01,
            "action_dim": 3,
            "mixed_precision": "bf16",
        },
    }, checkpoint / "train_meta.pt")
    geometry = {
        "frames": 9,
        "stride": 1,
        "height": 288,
        "width": 512,
        "chunk_overlap": 0,
        "past_chunks": 3 if special else 2,
        "max_attn_chunks": 4 if special else 3,
        "use_continuous_history": continuous,
    }
    if special:
        geometry.update({
            "use_special_memory": True,
            "num_special_tokens": 9,
            "special_memory_recent_chunks": 2,
        })
    (run / "experiment.json").write_text(json.dumps({"geometry": geometry}))
    return checkpoint


def test_recovers_completed_video_action_checkpoint_contract(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    contract = load_deploy_contract(checkpoint)

    assert contract["num_video_frames"] == 9
    assert (contract["image_height"], contract["image_width"]) == (288, 512)
    assert contract["ar_num_past_chunks"] == 2
    assert contract["max_attn_chunks"] == 3
    assert contract["traj_normalizer"] == "symlog"
    assert contract["symlog_c"] == 0.01
    assert contract["use_special_memory"] is False
    assert contract["use_continuous_history"] is False
    assert contract["crop_square"] is False


def test_special_memory_checkpoint_is_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path, special=True)
    with pytest.raises(DeployContractError, match="without special memory"):
        load_deploy_contract(checkpoint)


def test_recovers_continuous_history_contract(tmp_path):
    contract = load_deploy_contract(_checkpoint(tmp_path, continuous=True))

    assert contract["use_special_memory"] is False
    assert contract["use_continuous_history"] is True


def test_rejects_experiment_and_checkpoint_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    experiment_path = checkpoint.parents[1] / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    experiment["geometry"]["frames"] = 13
    experiment_path.write_text(json.dumps(experiment))

    with pytest.raises(DeployContractError, match="num_video_frames"):
        load_deploy_contract(checkpoint)
