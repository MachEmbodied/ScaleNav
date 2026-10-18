"""Evaluate ScaleNav on HabitatGS-LN and Short-Horizon OVON."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from awm.habitat_extensions.config import ActionConfig, EvalConfig, SensorConfig, SimulatorConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_config", required=True)
    for name in ("episode_manifest", "scene_dataset_config", "model_path", "pretrained_model_path",
                 "config", "output_dir", "gs_dataset_root", "data_root", "split_file", "text_level", "device"):
        parser.add_argument("--" + name)
    parser.add_argument("--simulator_gpu_device_id", type=int)
    parser.add_argument("--num_episodes", type=int)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--save_rollout_video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None,
                        help="Save every rollout and build a video/trajectory gallery")
    return parser.parse_args()


def _load_eval_yaml(yaml_path: str, seen=None) -> dict:
    """Load an eval YAML with optional relative ``base_config`` inheritance."""
    path = Path(yaml_path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"Cyclic eval base_config reference: {path}")
    seen.add(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base_config", None)
    if not base:
        return raw
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = _load_eval_yaml(str(base_path), seen)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _build_config(yaml_path, args):
    raw = _load_eval_yaml(yaml_path)
    dataset = raw.pop("dataset", {}) or {}
    gs_root = args.gs_dataset_root or raw.pop("gs_dataset_root", None)
    raw.pop("gs_dataset_root", None)
    cfg = EvalConfig(
        simulator=SimulatorConfig(**raw.pop("simulator", {})),
        sensor=SensorConfig(**raw.pop("sensor", {})),
        action=ActionConfig(**raw.pop("action", {})), **raw)
    for name in ("episode_manifest", "scene_dataset_config", "model_path", "pretrained_model_path",
                 "output_dir", "device", "num_episodes", "eval_seed", "save_rollout_video", "visualize"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.config:
        cfg.config_path = args.config
    if args.simulator_gpu_device_id is not None:
        cfg.simulator.gpu_device_id = args.simulator_gpu_device_id
    data_root = args.data_root or dataset.get("data_root")
    split_file = args.split_file or dataset.get("split_file")
    if cfg.episode_manifest:
        data_root = data_root or str(Path(cfg.episode_manifest).parent)
        split_file = cfg.episode_manifest
    if not gs_root or not data_root or not split_file:
        raise ValueError("Supply the scene root and an episode manifest, or data_root and split_file")
    if not cfg.model_path or not cfg.pretrained_model_path:
        raise ValueError("Supply model_path and pretrained_model_path")
    if not cfg.config_path or cfg.config_path == "configs/inference.yaml":
        from awm.cli import DEFAULT_CONFIG
        cfg.config_path = str(DEFAULT_CONFIG)
    return cfg, gs_root, data_root, split_file, args.text_level or dataset.get("text_level", "l3")


def main():
    args = parse_args()
    cfg, gs_root, data_root, split_file, text_level = _build_config(args.eval_config, args)
    from awm.habitat_extensions.awm_habitat_gs_evaluator import AWMHabitatGSEvaluator
    AWMHabitatGSEvaluator(cfg, gs_root, data_root, split_file, text_level).evaluate()


if __name__ == "__main__":
    main()
