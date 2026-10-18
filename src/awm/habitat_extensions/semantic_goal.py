"""Semantic goal-region evaluation for local navigation rollouts."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _world_xz(value: Any) -> Optional[np.ndarray]:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size >= 3:
        point = point[[0, 2]]
    elif point.size != 2:
        return None
    if not np.isfinite(point).all():
        return None
    return point


def _first_world_xz(values: Iterable[Any]) -> Optional[np.ndarray]:
    for value in values:
        point = _world_xz(value)
        if point is not None:
            return point
    return None


def _angle_error_deg(position_xz: np.ndarray, yaw: float, target_xz: np.ndarray) -> float:
    direction = target_xz - position_xz
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 180.0
    direction /= norm
    # Habitat yaw=0 looks along -Z. This matches the evaluator pose update.
    forward = np.asarray([-math.sin(yaw), -math.cos(yaw)], dtype=np.float64)
    cosine = float(np.clip(np.dot(forward, direction), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _polygon_from_contract(contract: Dict[str, Any]) -> Optional[np.ndarray]:
    region = contract.get("region") or {}
    success = contract.get("success_condition") or {}
    candidates = (
        success.get("region_polygon_xz"),
        success.get("polygon_xz"),
        success.get("region_profile_xz"),
        region.get("polygon_xz"),
        region.get("profile_xz"),
        region.get("profile"),
    )
    for candidate in candidates:
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 3:
            continue
        points = []
        for value in candidate:
            point = _world_xz(value)
            if point is None:
                points = []
                break
            points.append(point)
        if len(points) >= 3:
            return np.asarray(points, dtype=np.float64)
    return None


def _coerce_polygon(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points = []
    for item in value:
        point = _world_xz(item)
        if point is None:
            return None
        points.append(point)
    return np.asarray(points, dtype=np.float64)


def _target_footprints(contract: Dict[str, Any]) -> list[np.ndarray]:
    footprint = _coerce_polygon(
        contract.get("semantic_target_footprint_world_xz")
    )
    if footprint is not None:
        return [footprint]

    success = contract.get("success_condition") or {}
    selected_ids = {
        str(value) for value in (success.get("support_obj_ids") or [])
    }
    footprints = []
    for support in contract.get("supporting_instances") or []:
        if selected_ids and str(support.get("obj_id")) not in selected_ids:
            continue
        footprint = _coerce_polygon(support.get("footprint_world_xz"))
        if footprint is not None:
            footprints.append(footprint)
    return footprints


def _footprint_distance_threshold(
    contract: Dict[str, Any], fallback_radius_m: float,
) -> float:
    success = contract.get("success_condition") or {}
    value = success.get("max_distance_m")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return max(0.0, float(value))
    thresholds = ((contract.get("arrival") or {}).get("thresholds") or {})
    value = thresholds.get("footprint_distance_max_m")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return max(0.0, float(value))
    legacy_band = thresholds.get("surface_standoff_m")
    if isinstance(legacy_band, (list, tuple)) and len(legacy_band) == 2:
        try:
            return max(0.0, float(legacy_band[1]))
        except (TypeError, ValueError):
            pass
    return max(0.0, float(fallback_radius_m))


def _point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray,
) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    epsilon = 1e-9
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    offset = second_start - first_start
    denominator = _cross_2d(first_delta, second_delta)

    if abs(denominator) <= epsilon:
        if abs(_cross_2d(offset, first_delta)) > epsilon:
            return False
        first_length_sq = float(np.dot(first_delta, first_delta))
        if first_length_sq <= epsilon:
            return (
                _point_to_segment_distance(
                    first_start, second_start, second_end
                )
                <= epsilon
            )
        first_t = float(np.dot(second_start - first_start, first_delta))
        second_t = float(np.dot(second_end - first_start, first_delta))
        lower, upper = sorted((first_t, second_t))
        return lower <= first_length_sq + epsilon and upper >= -epsilon

    first_t = _cross_2d(offset, second_delta) / denominator
    second_t = _cross_2d(offset, first_delta) / denominator
    return (
        -epsilon <= first_t <= 1.0 + epsilon
        and -epsilon <= second_t <= 1.0 + epsilon
    )


def _segment_to_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    if _segments_intersect(
        first_start, first_end, second_start, second_end
    ):
        return 0.0
    return min(
        _point_to_segment_distance(first_start, second_start, second_end),
        _point_to_segment_distance(first_end, second_start, second_end),
        _point_to_segment_distance(second_start, first_start, first_end),
        _point_to_segment_distance(second_end, first_start, first_end),
    )


def _polygon_membership(point: np.ndarray, polygon: np.ndarray) -> Tuple[bool, float]:
    boundary_distance = min(
        _point_to_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )
    if boundary_distance <= 1e-6:
        return True, 0.0

    x, z = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        crosses = (zi > z) != (zj > z)
        if crosses:
            x_cross = (xj - xi) * (z - zi) / (zj - zi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside, 0.0 if inside else boundary_distance


def _segment_to_polygon_distance(
    start: np.ndarray, end: np.ndarray, polygon: np.ndarray,
) -> float:
    start_inside, _start_distance = _polygon_membership(start, polygon)
    end_inside, _end_distance = _polygon_membership(end, polygon)
    if start_inside or end_inside:
        return 0.0
    return min(
        _segment_to_segment_distance(
            start, end, polygon[index], polygon[(index + 1) % len(polygon)]
        )
        for index in range(len(polygon))
    )


def _nonnegative_episode_value(
    episode: Dict[str, Any], key: str,
) -> float:
    value = episode.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, float(value))


def _robot_footprint_distance(
    position_xz: np.ndarray,
    yaw: float,
    target: np.ndarray,
    *,
    body_radius_m: float,
    body_offset_m: float,
) -> Tuple[float, float]:
    forward = np.asarray(
        [-math.sin(float(yaw)), -math.cos(float(yaw))], dtype=np.float64
    )
    spine_start = position_xz - body_offset_m * forward
    spine_end = position_xz + body_offset_m * forward
    spine_distance = _segment_to_polygon_distance(
        spine_start, spine_end, target
    )
    return max(0.0, spine_distance - body_radius_m), spine_distance


def _surface_band(contract: Dict[str, Any]) -> Tuple[float, float]:
    arrival = contract.get("arrival") or {}
    thresholds = arrival.get("thresholds") or {}
    band = thresholds.get("surface_standoff_m")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        try:
            lower, upper = float(band[0]), float(band[1])
            if 0 <= lower <= upper:
                return lower, upper
        except (TypeError, ValueError):
            pass

    desired = contract.get("desired_surface_standoff_m")
    if isinstance(desired, (int, float)) and math.isfinite(float(desired)):
        desired = float(desired)
        return max(0.0, desired - 0.4), desired + 0.6
    return 0.25, 1.25


def _facing_limit(contract: Dict[str, Any]) -> float:
    value = (
        ((contract.get("arrival") or {}).get("thresholds") or {})
        .get("facing_max_deg", 45.0)
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return 45.0


def evaluate_semantic_goal(
    position: Sequence[float],
    yaw: float,
    episode: Dict[str, Any],
    *,
    fallback_radius_m: float,
    fallback_distance_m: Optional[float] = None,
    require_facing: bool = True,
) -> Dict[str, Any]:
    """Evaluate one pose against the episode's semantic success set.

    Object and near targets use their stored ground-plane footprint when
    available. Episodes with Scan2D body dimensions measure the closest
    distance from the oriented Go2 capsule to that target footprint; older
    episodes fall back to the agent-center distance. There is intentionally no
    minimum distance or facing gate. Region targets use a stored polygon.
    Other/legacy targets retain explicit fallback behavior.
    """
    position_xz = _world_xz(position)
    if position_xz is None:
        raise ValueError("position must be a finite 2D/3D world point")

    contract = episode.get("target_contract") or {}
    floor_filter = contract.get("floor_filter")
    if floor_filter:
        xyz = np.asarray(position, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Floor-filtered targets require a finite 3D camera position")
        floor_y = float(xyz[1]) - float(episode["cam_height_m"])
        supports = contract.get("supporting_instances") or []
        same_floor = [support for support in supports
                      if support["ymin"] <= floor_y + floor_filter["above_m"]
                      and support["ymax"] >= floor_y - floor_filter["below_m"]]
        contract = {**contract, "supporting_instances": same_floor or supports}
    target_mode = str(
        episode.get("target_mode") or contract.get("target_mode") or "unknown"
    )
    target_kind = str(contract.get("kind") or target_mode)
    success = contract.get("success_condition") or {}
    if target_mode == "doorway" and success.get("type") == "distance_to_navigation_goal":
        if fallback_distance_m is None or not math.isfinite(float(fallback_distance_m)) or fallback_distance_m < 0:
            raise ValueError("Doorway evaluation requires a finite navigation-goal geodesic distance")
        radius = float(success["max_distance_m"])
        reached = float(fallback_distance_m) < radius
        return {
            "target_mode": target_mode, "success_type": "navigation_goal_geodesic_distance",
            "semantic_contract_available": True, "region_success": reached, "contract_success": reached,
            "navigation_error": max(0.0, float(fallback_distance_m) - radius),
            "point_distance_m": float(fallback_distance_m), "point_radius_m": radius,
            "facing_success": None, "facing_required": False,
        }

    if target_mode in ("object", "near"):
        footprints = _target_footprints(contract)
        if footprints:
            memberships = [
                _polygon_membership(position_xz, footprint)
                for footprint in footprints
            ]
            inside_any = any(inside for inside, _distance in memberships)
            center_footprint_distance = min(
                distance for _inside, distance in memberships
            )
            body_radius_m = _nonnegative_episode_value(
                episode, "scan_body_radius_m"
            )
            body_offset_m = _nonnegative_episode_value(
                episode, "scan_body_offset_m"
            )
            robot_distances = [
                _robot_footprint_distance(
                    position_xz,
                    float(yaw),
                    footprint,
                    body_radius_m=body_radius_m,
                    body_offset_m=body_offset_m,
                )
                for footprint in footprints
            ]
            footprint_distance, spine_distance = min(
                robot_distances, key=lambda item: item[0]
            )
            robot_footprint_applied = (
                body_radius_m > 0.0 or body_offset_m > 0.0
            )
            threshold = _footprint_distance_threshold(
                contract, fallback_radius_m
            )
            region_success = (footprint_distance < threshold if success.get("strict", False)
                              else footprint_distance <= threshold)
            center_candidates = [contract.get("semantic_target_world")]
            center_candidates.extend(
                support.get("world")
                for support in (contract.get("supporting_instances") or [])
            )
            center_xz = _first_world_xz(center_candidates)
            facing_error = (
                _angle_error_deg(position_xz, float(yaw), center_xz)
                if center_xz is not None else None
            )
            return {
                "target_mode": target_mode,
                "success_type": (
                    "semantic_target_robot_footprint_distance"
                    if robot_footprint_applied
                    else "semantic_target_footprint_distance"
                ),
                "semantic_contract_available": True,
                "region_success": bool(region_success),
                "contract_success": bool(region_success),
                "navigation_error": max(
                    0.0, float(footprint_distance) - threshold
                ),
                "target_footprint_distance_m": float(footprint_distance),
                "robot_footprint_distance_m": float(footprint_distance),
                "agent_center_to_target_footprint_distance_m": float(
                    center_footprint_distance
                ),
                "robot_spine_to_target_footprint_distance_m": float(
                    spine_distance
                ),
                "robot_footprint_body_radius_m": body_radius_m,
                "robot_footprint_body_offset_m": body_offset_m,
                "robot_footprint_applied": robot_footprint_applied,
                "robot_footprint_overlaps_target": bool(
                    footprint_distance <= 1e-6
                ),
                "target_footprint_max_distance_m": threshold,
                "target_footprint_count": len(footprints),
                "inside_target_footprint": bool(inside_any),
                "facing_error_deg": facing_error,
                "facing_success": None,
                "facing_required": False,
            }

    if target_kind == "object" or target_mode == "object":
        arrival = contract.get("arrival") or {}
        supports = contract.get("supporting_instances") or []
        support_world = supports[0].get("world") if supports else None
        center_xz = _first_world_xz((
            contract.get("semantic_target_world"),
            arrival.get("target_center_world"),
            support_world,
        ))
        if center_xz is not None:
            radius_value = contract.get(
                "semantic_target_radius_m", arrival.get("target_planar_radius_m", 0.0)
            )
            try:
                object_radius = max(0.0, float(radius_value))
            except (TypeError, ValueError):
                object_radius = 0.0
            center_distance = float(np.linalg.norm(position_xz - center_xz))
            surface_standoff = center_distance - object_radius
            _lower, upper = _surface_band(contract)
            region_success = surface_standoff < upper
            navigation_error = max(0.0, surface_standoff - upper)
            facing_error = _angle_error_deg(position_xz, float(yaw), center_xz)
            facing_limit = _facing_limit(contract)
            facing_success = facing_error <= facing_limit
            return {
                "target_mode": target_mode,
                "success_type": "semantic_object_surface_distance_fallback",
                "semantic_contract_available": True,
                "region_success": bool(region_success),
                "contract_success": bool(
                    region_success and (facing_success or not require_facing)
                ),
                "navigation_error": float(navigation_error),
                "target_center_distance_m": center_distance,
                "target_planar_radius_m": object_radius,
                "surface_standoff_m": surface_standoff,
                "surface_standoff_range_m": [None, upper],
                "facing_error_deg": facing_error,
                "facing_limit_deg": facing_limit,
                "facing_success": bool(facing_success),
                "facing_required": bool(require_facing),
            }

    if target_kind == "semantic_region" or target_mode == "semantic_region":
        polygon = _polygon_from_contract(contract)
        if polygon is not None:
            region_success, navigation_error = _polygon_membership(position_xz, polygon)
            return {
                "target_mode": target_mode,
                "success_type": "semantic_region_polygon",
                "semantic_contract_available": True,
                "region_success": bool(region_success),
                "contract_success": bool(region_success),
                "navigation_error": float(navigation_error),
                "facing_success": None,
                "facing_required": False,
            }

    goal_xz = _first_world_xz((
        contract.get("navigation_goal_world"),
        episode.get("goal_xyz"),
    ))
    if goal_xz is None:
        raise ValueError("episode has neither semantic target nor navigation goal")
    planar_distance = float(np.linalg.norm(position_xz - goal_xz))
    distance = (
        float(fallback_distance_m)
        if fallback_distance_m is not None and math.isfinite(float(fallback_distance_m))
        else planar_distance
    )
    radius = max(0.0, float(fallback_radius_m))
    region_success = distance < radius
    return {
        "target_mode": target_mode,
        "success_type": "point_radius_fallback",
        "semantic_contract_available": False,
        "semantic_fallback_reason": (
            "region_geometry_missing"
            if target_mode == "semantic_region" else "target_mode_has_no_region_contract"
        ),
        "region_success": bool(region_success),
        "contract_success": bool(region_success),
        "navigation_error": max(0.0, distance - radius),
        "point_distance_m": distance,
        "point_radius_m": radius,
        "facing_success": None,
        "facing_required": False,
    }


def navigation_success(
    goal_evaluation: Dict[str, Any],
    *,
    stopped_via_model: bool,
    require_model_stop: bool,
) -> bool:
    """Combine final semantic-goal satisfaction with the model STOP decision."""
    return bool(
        goal_evaluation.get("contract_success", False)
        and (stopped_via_model or not require_model_stop)
    )
