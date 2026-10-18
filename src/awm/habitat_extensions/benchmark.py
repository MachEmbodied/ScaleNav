"""Episode contracts for ScaleNav's two simulation benchmarks."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from awm.habitat_extensions.semantic_goal import _target_footprints


def prepare_episode(episode, benchmark):
    if benchmark not in {"habitatgs_ln", "short_horizon_ovon"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    result = deepcopy(episode)
    for field in ("scene_id", "episode_id", "instruction"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"Missing episode field: {field}")
    for field, size in (("start_xyz", 3), ("goal_xyz", 3), ("start_quat_wxyz", 4)):
        value = np.asarray(result.get(field), dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid episode field: {field}")
    if np.linalg.norm(result["start_quat_wxyz"]) < 1e-8:
        raise ValueError("Invalid start quaternion")
    contract = result.get("target_contract") or {}
    doorway = (benchmark == "habitatgs_ln" and result.get("target_mode") == "doorway"
               and contract.get("success_condition", {}).get("type") == "distance_to_navigation_goal")
    if not _target_footprints(contract) and not doorway:
        raise ValueError("Target footprint polygons are required")
    if result.get("action_convention") != "scan2d_body_forward_right_clockwise":
        raise ValueError("Episode action convention must be scan2d_body_forward_right_clockwise")
    contract.setdefault("success_condition", {})["max_distance_m"] = 1.25
    result.update(target_mode="doorway" if doorway else "object", target_contract=contract,
                  scan_body_radius_m=0.25, scan_body_offset_m=0.18)
    if benchmark == "short_horizon_ovon":
        points = np.asarray(result.get("reference_goal_positions"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("OVON requires navigable reference_goal_positions near valid targets")
        if not result.get("scene_path"):
            raise ValueError("OVON requires the HM3D scene_path")
    return result


def load_manifest(path, benchmark):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("benchmark") != benchmark:
        raise ValueError("Manifest benchmark does not match the evaluation configuration")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest must contain a nonempty episodes list")
    result = [prepare_episode(ep, benchmark) for ep in episodes]
    keys = [(ep["scene_id"], ep["episode_id"]) for ep in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate episode identifiers in manifest")
    return result
