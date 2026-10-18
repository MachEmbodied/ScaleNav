import json

import pytest

from awm.habitat_extensions.benchmark import load_manifest, prepare_episode
from awm.habitat_extensions.semantic_goal import evaluate_semantic_goal, navigation_success


def episode():
    return {
        "scene_id": "scene", "episode_id": "episode", "instruction": "Go to the desk.",
        "start_xyz": [0, 0.35, 0], "goal_xyz": [0, 0, -2],
        "start_quat_wxyz": [1, 0, 0, 0],
        "action_convention": "scan2d_body_forward_right_clockwise",
        "target_contract": {"semantic_target_footprint_world_xz":
                            [[-0.5, -3], [0.5, -3], [0.5, -2], [-0.5, -2]]},
    }


def test_success_requires_stop_and_capsule_distance():
    ep = prepare_episode(episode(), "habitatgs_ln")
    goal = evaluate_semantic_goal([0, 0, -0.5], 0, ep, fallback_radius_m=1.25)
    assert goal["robot_footprint_distance_m"] == pytest.approx(1.07)
    assert goal["contract_success"]
    assert not navigation_success(goal, stopped_via_model=False, require_model_stop=True)
    assert navigation_success(goal, stopped_via_model=True, require_model_stop=True)


def test_success_includes_distance_boundary_without_facing_gate():
    ep = prepare_episode(episode(), "habitatgs_ln")
    goal = evaluate_semantic_goal([0, 0, -0.32], 0, ep, fallback_radius_m=0.1)
    assert goal["robot_footprint_distance_m"] == pytest.approx(1.25)
    assert goal["contract_success"]


def test_paper_geometry_overrides_legacy_episode_thresholds_without_mutation():
    original = episode()
    original['target_contract']['success_condition'] = {'max_distance_m': 0.5}
    original['scan_body_radius_m'] = 0
    ep = prepare_episode(original, 'habitatgs_ln')
    assert ep['scan_body_radius_m'] == 0.25
    assert ep['scan_body_offset_m'] == 0.18
    assert ep['target_contract']['success_condition']['max_distance_m'] == 1.25
    assert original['target_contract']['success_condition']['max_distance_m'] == 0.5


def test_missing_footprints_cannot_fall_back_to_point_goal():
    ep = episode()
    ep['target_contract'] = {}
    with pytest.raises(ValueError, match='footprint'):
        prepare_episode(ep, 'habitatgs_ln')


def test_doorway_uses_explicit_point_goal_and_requires_geodesic_distance():
    ep = episode()
    ep['target_mode'] = 'doorway'
    ep['target_contract'] = {'success_condition': {'type': 'distance_to_navigation_goal'}}
    ep = prepare_episode(ep, 'habitatgs_ln')
    with pytest.raises(ValueError, match='geodesic distance'):
        evaluate_semantic_goal([0, 0, -2], 0, ep, fallback_radius_m=1.25)
    for distance, success in [(1.24, True), (1.25, False), (2.0, False)]:
        goal = evaluate_semantic_goal([0, 0, -2], 0, ep, fallback_radius_m=1.25, fallback_distance_m=distance)
        assert goal['contract_success'] is success
        assert not navigation_success(goal, stopped_via_model=False, require_model_stop=True)
    with pytest.raises(ValueError, match='footprint'):
        prepare_episode(ep, 'short_horizon_ovon')


def test_ovon_requires_navigable_reference_points():
    ep = episode()
    ep['scene_path'] = 'scene.glb'
    with pytest.raises(ValueError, match='reference_goal_positions'):
        prepare_episode(ep, 'short_horizon_ovon')
    ep['reference_goal_positions'] = [[0, 0, -1]]
    assert prepare_episode(ep, 'short_horizon_ovon')['target_mode'] == 'object'


def test_manifest_rejects_duplicate_episodes(tmp_path):
    path = tmp_path / 'episodes.json'
    path.write_text(json.dumps({'benchmark': 'habitatgs_ln', 'episodes': [episode(), episode()]}))
    with pytest.raises(ValueError, match='Duplicate'):
        load_manifest(path, 'habitatgs_ln')
