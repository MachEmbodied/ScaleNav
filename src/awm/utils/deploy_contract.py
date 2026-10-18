"""Recover the inference contract saved with an AWM training checkpoint.

The transformer config stores architectural switches, while ``train_meta.pt``
stores the data geometry and normalization used by the training process.  A
nearby ``experiment.json`` is an additional, human-readable cross-check.  Real
robot launchers should consume these files instead of duplicating their values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class DeployContractError(ValueError):
    """Raised when checkpoint metadata is missing or internally inconsistent."""


_FALLBACKS: Dict[str, Any] = {
    "num_video_frames": 9,
    "video_sample_stride": 1,
    "image_height": 288,
    "image_width": 512,
    "chunk_overlap_frames": 0,
    "ar_num_past_chunks": 2,
    "max_attn_chunks": 3,
    "use_special_memory": False,
    "use_continuous_history": False,
    "num_special_tokens": 9,
    "special_memory_recent_chunks": 2,
    "traj_normalizer": "symlog",
    "symlog_c": 0.01,
    "action_dim": 3,
    "action_temporal_scale": 1.0,
    "dtype": "bf16",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployContractError(f"cannot read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DeployContractError(f"checkpoint metadata must be an object: {path}")
    return payload


def _read_train_meta(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before the weights_only argument.
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - surface the checkpoint path.
        raise DeployContractError(f"cannot read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DeployContractError(f"train_meta.pt must contain a mapping: {path}")
    return payload


def _find_experiment(checkpoint: Path) -> Optional[Path]:
    for directory in (checkpoint, *checkpoint.parents[:4]):
        candidate = directory / "experiment.json"
        if candidate.is_file():
            return candidate
    return None


def _normalise_value(key: str, value: Any) -> Any:
    if key in {
        "num_video_frames",
        "video_sample_stride",
        "image_height",
        "image_width",
        "chunk_overlap_frames",
        "ar_num_past_chunks",
        "max_attn_chunks",
        "num_special_tokens",
        "special_memory_recent_chunks",
        "action_dim",
    }:
        return int(value)
    if key in {"symlog_c", "action_temporal_scale"}:
        return float(value)
    if key in {"use_special_memory", "use_continuous_history"}:
        return bool(value)
    return value


def _record(
    values: Dict[str, Any],
    origins: Dict[str, str],
    key: str,
    value: Any,
    source: Path,
) -> None:
    if value is None:
        return
    value = _normalise_value(key, value)
    if key in values and values[key] != value:
        raise DeployContractError(
            f"checkpoint contract mismatch for {key}: {values[key]!r} from "
            f"{origins[key]} vs {value!r} from {source}"
        )
    values[key] = value
    origins[key] = str(source)


def _validate(values: Mapping[str, Any]) -> None:
    frames = int(values["num_video_frames"])
    height = int(values["image_height"])
    width = int(values["image_width"])
    overlap = int(values["chunk_overlap_frames"])
    past = int(values["ar_num_past_chunks"])
    window = int(values["max_attn_chunks"])
    if frames <= 0:
        raise DeployContractError("num_video_frames must be positive")
    if height <= 0 or width <= 0:
        raise DeployContractError("checkpoint image dimensions must be positive")
    if not 0 <= overlap < frames:
        raise DeployContractError(
            f"chunk_overlap_frames={overlap} is invalid for {frames} frames"
        )
    if past < 0:
        raise DeployContractError("ar_num_past_chunks must be non-negative")
    if window != -1 and window < past + 1:
        raise DeployContractError(
            f"max_attn_chunks={window} cannot hold {past} past chunks plus current"
        )
    if int(values["action_dim"]) != 3:
        raise DeployContractError(
            "the Go2 continuous controller requires action_dim=3 "
            f"(got {values['action_dim']})"
        )
    if bool(values["use_special_memory"]):
        raise DeployContractError("ScaleNav requires continuous history without special memory tokens")
    if values["traj_normalizer"] != "symlog":
        raise DeployContractError("ScaleNav requires symlog action normalization")
    if bool(values["use_continuous_history"]) and float(
        values["action_temporal_scale"]
    ) != 1.0:
        raise DeployContractError(
            "continuous history requires action_temporal_scale=1.0"
        )


def load_deploy_contract(model_path: str | Path) -> Dict[str, Any]:
    """Load and cross-check all deployment metadata available for a checkpoint."""
    checkpoint = Path(model_path).expanduser()
    if not checkpoint.is_dir():
        raise DeployContractError(f"checkpoint directory does not exist: {checkpoint}")

    transformer_path = checkpoint / "transformer" / "config.json"
    if not transformer_path.is_file():
        raise DeployContractError(f"missing transformer config: {transformer_path}")

    values: Dict[str, Any] = {}
    origins: Dict[str, str] = {}
    source_files = []

    transformer = _read_json(transformer_path)
    source_files.append(str(transformer_path))
    for key in (
        "max_attn_chunks",
        "chunk_overlap_frames",
        "use_special_memory",
        "use_continuous_history",
        "num_special_tokens",
        "special_memory_recent_chunks",
        "action_dim",
        "action_temporal_scale",
    ):
        _record(values, origins, key, transformer.get(key), transformer_path)

    meta_path = checkpoint / "train_meta.pt"
    if meta_path.is_file():
        meta = _read_train_meta(meta_path)
        args = meta.get("args", {})
        if not isinstance(args, Mapping):
            raise DeployContractError(f"train_meta args must be a mapping: {meta_path}")
        source_files.append(str(meta_path))
        size = args.get("video_sample_size")
        if size is not None:
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise DeployContractError(
                    f"video_sample_size must be [height, width]: {meta_path}"
                )
            _record(values, origins, "image_height", size[0], meta_path)
            _record(values, origins, "image_width", size[1], meta_path)
        mapping = {
            "video_sample_n_frames": "num_video_frames",
            "video_sample_stride": "video_sample_stride",
            "chunk_overlap_frames": "chunk_overlap_frames",
            # This is the number actually sampled by the causal dataset. The
            # older ar_num_past_chunks CLI field is unrelated and may be 4.
            "num_train_past_chunks": "ar_num_past_chunks",
            "max_attn_chunks": "max_attn_chunks",
            "use_special_memory": "use_special_memory",
            "use_continuous_history": "use_continuous_history",
            "num_special_tokens": "num_special_tokens",
            "special_memory_recent_chunks": "special_memory_recent_chunks",
            "traj_normalizer": "traj_normalizer",
            "symlog_c": "symlog_c",
            "action_dim": "action_dim",
            "action_temporal_scale": "action_temporal_scale",
            "mixed_precision": "dtype",
        }
        for source_key, contract_key in mapping.items():
            _record(values, origins, contract_key, args.get(source_key), meta_path)
        if meta.get("global_step") is not None:
            values["global_step"] = int(meta["global_step"])

    experiment_path = _find_experiment(checkpoint)
    if experiment_path is not None:
        experiment = _read_json(experiment_path)
        geometry = experiment.get("geometry", {})
        if not isinstance(geometry, Mapping):
            raise DeployContractError(
                f"experiment geometry must be an object: {experiment_path}"
            )
        source_files.append(str(experiment_path))
        mapping = {
            "frames": "num_video_frames",
            "stride": "video_sample_stride",
            "height": "image_height",
            "width": "image_width",
            "chunk_overlap": "chunk_overlap_frames",
            "past_chunks": "ar_num_past_chunks",
            "max_attn_chunks": "max_attn_chunks",
            "use_special_memory": "use_special_memory",
            "use_continuous_history": "use_continuous_history",
            "num_special_tokens": "num_special_tokens",
            "special_memory_recent_chunks": "special_memory_recent_chunks",
        }
        for source_key, contract_key in mapping.items():
            _record(
                values,
                origins,
                contract_key,
                geometry.get(source_key),
                experiment_path,
            )

    # A copied checkpoint may not include its parent experiment manifest. Fill
    # only genuinely absent values; saved metadata always wins and is checked.
    for key, value in _FALLBACKS.items():
        if key not in values:
            values[key] = value
            origins[key] = "go2_9f_fallback"

    # Pretraining checkpoints may store the causal window only in the model config.
    if origins["ar_num_past_chunks"] == "go2_9f_fallback":
        window = int(values["max_attn_chunks"])
        if window > 0:
            values["ar_num_past_chunks"] = window - 1
            origins["ar_num_past_chunks"] = origins["max_attn_chunks"]

    _validate(values)
    values["crop_square"] = int(values["image_height"]) == int(values["image_width"])
    values["checkpoint"] = str(checkpoint)
    values["source_files"] = source_files
    values["origins"] = origins
    return values
