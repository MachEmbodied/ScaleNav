from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from awm.habitat_extensions.semantic_goal import evaluate_semantic_goal

spec = importlib.util.spec_from_file_location("prepare_ovon", Path(__file__).resolve().parents[1] / "scripts/prepare_ovon.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_release_labels_contain_fixed_selection_without_upstream_geometry():
    annotation = json.loads(prepare.ANNOTATIONS.read_text())
    rows = annotation["episodes"]
    assert len(rows) == 524
    assert len({row["scene_id"] for row in rows}) == 36
    assert len({(row["scene_id"], row["episode_id"]) for row in rows}) == 524
    assert all(set(row) == {"scene_id", "episode_id", "instruction", "support_obj_ids"} for row in rows)
    assert all(row["instruction"] and row["support_obj_ids"] for row in rows)
    assert all(len(row["support_obj_ids"]) == len(set(row["support_obj_ids"])) for row in rows)
    assert annotation["subsets"]["test_subset"] == ['00832-qyAac8rV8Zk/traj_1067', '00827-BAbdmeyTvMZ/traj_3148', '00810-CrMo8WxCyVb/traj_4159', '00800-TEEsavR23oF/traj_2411', '00808-y9hTuugGdiq/traj_381', '00878-XB4GS9ShBRE/traj_3370', '00847-bCPU9suPUw9/traj_2404']


def test_official_archive_extracts_only_selected_episode_files_and_checks_hash(tmp_path):
    poses = b"[]"
    episode = gzip.compress(b'{"episodes": []}')
    (tmp_path / "poses.json").write_bytes(poses)
    with tarfile.open(tmp_path / "episodes.tar.gz", "w:gz") as archive:
        for name, payload in [("hm3d/val_unseen/content/scene.json.gz", episode), ("../outside", b"ignored")]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    metadata = {"start_poses": {"filename": "poses.json", "sha256": hashlib.sha256(poses).hexdigest()},
                "episodes": {"filename": "episodes.tar.gz", "sha256": prepare.sha256(tmp_path / "episodes.tar.gz")},
                "episode_files": {"scene.json.gz": hashlib.sha256(episode).hexdigest()}}
    records, content = prepare.prepare_upstream(tmp_path, metadata, False)
    assert records == []
    assert (content / "scene.json.gz").read_bytes() == episode
    assert not (tmp_path.parent / "outside").exists()
    (content / "scene.json.gz").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="episode checksum"):
        prepare.prepare_upstream(tmp_path, metadata, False)


def test_subset_preserves_selection_order_without_changing_full_benchmark():
    annotation = json.loads(prepare.ANNOTATIONS.read_text())
    rows = annotation["episodes"]
    selected = [f"{row['scene_id']}/{row['episode_id']}" for row in (rows[20], rows[0])]
    annotation["subsets"] = {"test_subset": selected}
    assert prepare.evaluation_rows(annotation, "test_subset") == [rows[20], rows[0]]
    assert prepare.evaluation_rows(annotation, "full") == rows
    for invalid in ([], selected + selected[:1], ["missing/episode"]):
        annotation["subsets"]["test_subset"] = invalid
        with pytest.raises(ValueError, match="Invalid OVON subset"):
            prepare.evaluation_rows(annotation, "test_subset")


def test_interrupted_episode_extraction_is_retryable(tmp_path, monkeypatch):
    (tmp_path / "poses.json").write_text("[]")
    payload = gzip.compress(b'{"episodes": []}')
    with tarfile.open(tmp_path / "episodes.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("hm3d/val_unseen/content/scene.json.gz")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    metadata = {"start_poses": {"filename": "poses.json", "sha256": prepare.sha256(tmp_path / "poses.json")},
                "episodes": {"filename": "episodes.tar.gz", "sha256": prepare.sha256(tmp_path / "episodes.tar.gz")},
                "episode_files": {"scene.json.gz": hashlib.sha256(payload).hexdigest()}}
    original = prepare.shutil.copyfileobj
    def interrupted(incoming, outgoing):
        outgoing.write(incoming.read(2))
        raise OSError("Interrupted extraction")
    monkeypatch.setattr(prepare.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="Interrupted extraction"):
        prepare.prepare_upstream(tmp_path, metadata, False)
    target = tmp_path / "episodes/hm3d/val_unseen/content/scene.json.gz"
    assert not target.exists()
    monkeypatch.setattr(prepare.shutil, "copyfileobj", original)
    prepare.prepare_upstream(tmp_path, metadata, False)
    assert target.read_bytes() == payload


def scene_archives(root):
    checksums = {}
    for archive_name, suffixes in (
        ("hm3d-val-habitat-v0.2.tar", ("basis.glb", "basis.navmesh")),
        ("hm3d-val-semantic-annots-v0.2.tar", ("semantic.glb", "semantic.txt")),
    ):
        with tarfile.open(root / archive_name, "w") as archive:
            for scene in ("00800-first", "00801-second"):
                for suffix in suffixes:
                    name = f"{scene}/{scene.split('-', 1)[1]}.{suffix}"
                    payload = name.encode()
                    member = tarfile.TarInfo("./" + name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                    checksums["val/" + name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            member = tarfile.TarInfo("../../outside")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bad"))
    return checksums


def test_scene_archives_extract_only_selection_and_reuse_verified_files(tmp_path):
    checksums = scene_archives(tmp_path)
    destination = tmp_path / "scenes"
    prepare.extract_scene_archives(tmp_path, destination, {"00801-second"}, checksums)
    assert sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()) == sorted(
        name for name in checksums if "00801-second" in name)
    assert not (tmp_path / "outside").exists()
    prepare.validate_scenes(destination, tmp_path / "prepared", ["00801-second"], checksums)
    prepare.extract_scene_archives(tmp_path / "no-archives-needed", destination, {"00801-second"}, checksums)
    prepare.extract_scene_archives(tmp_path, destination, {"00800-first", "00801-second"}, checksums)
    assert len(list(destination.rglob("*.*"))) == 8


def test_scene_archive_extraction_interruption_is_retryable(tmp_path, monkeypatch):
    checksums = scene_archives(tmp_path)
    original = prepare.shutil.copyfileobj
    def interrupted(incoming, outgoing):
        outgoing.write(incoming.read(2))
        raise OSError("Interrupted")
    monkeypatch.setattr(prepare.shutil, "copyfileobj", interrupted)
    destination = tmp_path / "scenes"
    with pytest.raises(OSError, match="Interrupted"):
        prepare.extract_scene_archives(tmp_path, destination, {"00800-first"}, checksums)
    assert not any(p.is_file() for p in destination.rglob("*"))
    monkeypatch.setattr(prepare.shutil, "copyfileobj", original)
    prepare.extract_scene_archives(tmp_path, destination, {"00800-first"}, checksums)
    prepare.validate_scenes(destination, tmp_path / "prepared", ["00800-first"], checksums)


def test_scene_archive_checksum_failure_does_not_publish_asset(tmp_path):
    checksums = scene_archives(tmp_path)
    checksums["val/00800-first/first.basis.glb"]["sha256"] = "0" * 64
    destination = tmp_path / "scenes"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.extract_scene_archives(tmp_path, destination, {"00800-first"}, checksums)
    assert not any(p.is_file() for p in destination.rglob("*"))


def test_scene_archive_missing_member_fails(tmp_path):
    checksums = scene_archives(tmp_path)
    with tarfile.open(tmp_path / "hm3d-val-habitat-v0.2.tar", "w"):
        pass
    with pytest.raises(FileNotFoundError, match="Missing HM3D assets"):
        prepare.extract_scene_archives(tmp_path, tmp_path / "scenes", {"00800-first"}, checksums)


def test_scene_assets_require_exact_checksums(tmp_path):
    scene = "00800-scene"
    source, output = tmp_path / "source", tmp_path / "output"
    directory = source / "val" / scene
    directory.mkdir(parents=True)
    checksums = {}
    for suffix in ("basis.glb", "basis.navmesh", "semantic.glb", "semantic.txt"):
        path = directory / f"scene.{suffix}"
        path.write_bytes(suffix.encode())
        checksums[str(path.relative_to(source))] = {"size": path.stat().st_size, "sha256": prepare.sha256(path)}
    prepare.link_scenes(prepare.validate_scenes(source, output, [scene], checksums))
    assert (output / "scenes/val" / scene).resolve() == directory.resolve()
    (directory / "scene.basis.navmesh").write_bytes(b"changed navmesh")
    with pytest.raises(ValueError, match="scene checksum"):
        prepare.validate_scenes(source, output, [scene], checksums)


def test_scene_conflict_does_not_create_links_for_earlier_scenes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    checksums = {}
    scenes = ["00800-first", "00801-second"]
    for scene in scenes:
        directory = source / "val" / scene
        directory.mkdir(parents=True)
        for suffix in ("basis.glb", "basis.navmesh", "semantic.glb", "semantic.txt"):
            path = directory / f"{scene.split('-', 1)[1]}.{suffix}"
            path.write_bytes(suffix.encode())
            checksums[str(path.relative_to(source))] = {"size": path.stat().st_size, "sha256": prepare.sha256(path)}
    conflict = output / "scenes/val" / scenes[1]
    conflict.parent.mkdir(parents=True)
    conflict.symlink_to(tmp_path / "other-source")
    with pytest.raises(ValueError, match="different source"):
        prepare.validate_scenes(source, output, scenes, checksums)
    assert not (output / "scenes/val" / scenes[0]).exists()
    assert not (output / "preparation.json").exists()
    assert conflict.is_symlink()


def test_missing_scenes_can_be_corrected_without_poisoning_output(tmp_path, monkeypatch):
    import sys

    source, scenes, output = tmp_path / "upstream", tmp_path / "hm3d", tmp_path / "prepared"
    scene = "00800-scene"
    directory = scenes / "val" / scene
    directory.mkdir(parents=True)
    checksums = {}
    for suffix in ("basis.glb", "basis.navmesh", "semantic.glb", "semantic.txt"):
        path = directory / f"scene.{suffix}"
        path.write_bytes(suffix.encode())
        checksums[str(path.relative_to(scenes))] = {"size": path.stat().st_size, "sha256": prepare.sha256(path)}
    source.mkdir()
    poses = [{"scene_id": scene, "traj_id": "traj_1", "habitat_start_position": [2, 0.45, 0],
              "habitat_quaternion_wxyz": [1, 0, 0, 0]}]
    pose_path = source / "poses.json"
    pose_path.write_text(json.dumps(poses))
    content = source / "episodes/hm3d/val_unseen/content"
    content.mkdir(parents=True)
    task = {"episodes": [{"episode_id": 1, "object_category": "chair"}],
            "goals_by_category": {"scene.basis.glb_chair": [{"object_id": "chair_1", "position": [0, 0.5, 0]}]}}
    episode_path = content / "scene.json.gz"
    episode_path.write_bytes(gzip.compress(json.dumps(task).encode()))
    annotation = json.loads(prepare.ANNOTATIONS.read_text())
    annotation["episodes"] = [{"scene_id": scene, "episode_id": "traj_1", "instruction": "Approach the chair.", "support_obj_ids": [1]}]
    annotation["upstream"] = {"start_poses": {"filename": "poses.json", "sha256": prepare.sha256(pose_path)},
                              "episode_files": {"scene.json.gz": prepare.sha256(episode_path), "unused.json.gz": "unused"}, "scene_files": checksums}
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotation))
    transform = np.eye(4)
    transform[1, 3] = 0.5
    obj = SimpleNamespace(semantic_id=1, obb=SimpleNamespace(half_extents=[0.5, 0.5, 0.5], local_to_world=transform, center=[0, 0.5, 0]))
    def find_path(query):
        query.geodesic_distance = 1.0
        return True
    pathfinder = SimpleNamespace(is_loaded=True, snap_point=lambda point: np.asarray(point),
                                 get_island=lambda point: 0, find_path=find_path)
    simulator = SimpleNamespace(pathfinder=pathfinder, semantic_scene=SimpleNamespace(objects=[obj]), close=lambda: None)
    habitat = SimpleNamespace(SimulatorConfiguration=SimpleNamespace, Configuration=lambda *args: None,
                              Simulator=lambda config: simulator, MultiGoalShortestPath=SimpleNamespace,
                              agent=SimpleNamespace(AgentConfiguration=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "habitat_sim", habitat)
    args = SimpleNamespace(source=source, scenes=tmp_path / "wrong-scene-root", output=output,
                           annotations=annotation_path, download=False)
    with pytest.raises(FileNotFoundError, match="HM3D"):
        prepare.prepare(args)
    assert not output.exists()
    args.scenes = scenes
    result = prepare.prepare(args)
    assert len(json.loads(result.read_text())["episodes"]) == 1
    assert json.loads((output / "preparation.json").read_text())["scenes"] == str(scenes.resolve())
    assert (output / "scenes/val" / scene).resolve() == directory.resolve()


def test_obb_footprint_uses_full_world_transform_and_vertical_extent():
    transform = np.eye(4)
    transform[:3, 3] = [2, 3, 4]
    obj = SimpleNamespace(semantic_id=7, obb=SimpleNamespace(half_extents=[1, 2, 0.5],
                                                          local_to_world=transform, center=[2, 3, 4]))
    footprint = prepare.object_footprint(obj, "table")
    assert footprint["footprint_world_xz"] == [[1, 3.5], [3, 3.5], [3, 4.5], [1, 4.5]]
    assert (footprint["ymin"], footprint["ymax"]) == (1, 5)
    np.testing.assert_allclose(prepare.polygon_distances(np.array([[2, 4], [4, 4], [4, 5.5]]),
                                                       footprint["footprint_world_xz"]), [0, 1, np.sqrt(2)])


def test_reference_points_keep_first_nonempty_band_island_and_order():
    sampling = {"bands_m": [0.35, 0.5, 0.75], "grid_step_m": 0.1, "grid_margin_m": 1.3,
                "snap_max_planar_m": 0.3, "max_reference_points": 11}
    instances = [{"footprint_world_xz": [[0, 0], [1, 0], [1, 1], [0, 1]], "ymin": 0, "ymax": 1}]
    pathfinder = SimpleNamespace(snap_point=lambda point: point,
                                 get_island=lambda point: 0 if point[0] >= 0 else 1)
    points, band = prepare.reference_points(pathfinder, [0.5, 0, 0.5], instances, sampling,
                                            {"floor_above_m": 2.2, "floor_below_m": 0.5})
    assert band == 0.35 and len(points) == 11
    assert np.all(points[:, 0] >= 0)
    assert np.all(prepare.polygon_distances(points[:, [0, 2]], instances[0]["footprint_world_xz"]) <= 0.350001)
    assert list(map(tuple, points[:, [0, 2]])) == sorted(map(tuple, points[:, [0, 2]]))


def test_reference_points_expand_band_only_when_previous_band_is_empty():
    sampling = {"bands_m": [0.35, 0.5, 0.75], "grid_step_m": 0.1, "grid_margin_m": 1.3,
                "snap_max_planar_m": 0.3, "max_reference_points": 4000}
    instances = [{"footprint_world_xz": [[0, 0], [1, 0], [1, 1], [0, 1]], "ymin": 0, "ymax": 1}]
    def snap(point):
        return point if point[0] > 1.39 else np.full(3, np.nan)
    points, band = prepare.reference_points(SimpleNamespace(snap_point=snap, get_island=lambda point: 0),
                                            [2, 0, 0.5], instances, sampling,
                                            {"floor_above_m": 2.2, "floor_below_m": 0.5})
    assert band == 0.5
    assert np.all(points[:, 0] > 1.39)


def floor_episode():
    def instance(name, x, ymin, ymax):
        return {"obj_id": name, "footprint_world_xz": [[x, -0.5], [x + 1, -0.5], [x + 1, 0.5], [x, 0.5]],
                "ymin": ymin, "ymax": ymax}
    return {"target_mode": "object", "cam_height_m": 0.352846,
            "target_contract": {"kind": "object", "supporting_instances": [instance("lower", 3, 0, 1), instance("upper", 0, 4, 5)],
                                "floor_filter": {"below_m": 0.5, "above_m": 2.2},
                                "success_condition": {"max_distance_m": 1.25, "strict": True}}}


def test_floor_filter_tracks_current_camera_pose_and_preserves_unmarked_contracts():
    episode = floor_episode()
    before = deepcopy(episode)
    lower = evaluate_semantic_goal([0, 0.352846, 0], 0, episode, fallback_radius_m=1.25)
    upper = evaluate_semantic_goal([0, 4.352846, 0], 0, episode, fallback_radius_m=1.25)
    assert not lower["contract_success"] and lower["target_footprint_count"] == 1
    assert upper["contract_success"] and upper["target_footprint_count"] == 1
    assert episode == before
    del episode["target_contract"]["floor_filter"]
    unfiltered = evaluate_semantic_goal([0, 0.352846, 0], 0, episode, fallback_radius_m=1.25)
    assert unfiltered["contract_success"] and unfiltered["target_footprint_count"] == 2


@pytest.mark.parametrize("ymin,ymax,included", [(2.2, 3, True), (2.200001, 3, False),
                                              (-1, -0.5, True), (-1, -0.500001, False)])
def test_floor_filter_includes_touching_vertical_boundaries(ymin, ymax, included):
    episode = floor_episode()
    near = episode["target_contract"]["supporting_instances"][1]
    near.update(ymin=ymin, ymax=ymax)
    result = evaluate_semantic_goal([0, 0.352846, 0], 0, episode, fallback_radius_m=1.25)
    assert result["contract_success"] is included


def test_floor_filter_falls_back_to_all_when_no_instance_matches():
    result = evaluate_semantic_goal([0, 20.352846, 0], 0, floor_episode(), fallback_radius_m=1.25)
    assert result["contract_success"] and result["target_footprint_count"] == 2


def test_ovon_strict_distance_does_not_change_other_benchmarks_boundary():
    episode = floor_episode()
    episode["target_contract"].pop("floor_filter")
    episode["target_contract"]["supporting_instances"] = [{"footprint_world_xz": [[1.25, -1], [2, -1], [2, 1], [1.25, 1]]}]
    strict = evaluate_semantic_goal([0, 0, 0], 0, episode, fallback_radius_m=1.25)
    assert not strict["contract_success"]
    episode["target_contract"]["success_condition"].pop("strict")
    assert evaluate_semantic_goal([0, 0, 0], 0, episode, fallback_radius_m=1.25)["contract_success"]


def test_runtime_reference_queries_follow_current_ground_pose_without_mutating_camera_pose():
    from awm.habitat_extensions.awm_habitat_gs_evaluator import AWMHabitatGSEvaluator

    queried = []
    def snap(point):
        queried.append(point.copy())
        return point
    pathfinder = SimpleNamespace(snap_point=snap, find_path=lambda query: True)
    evaluator = AWMHabitatGSEvaluator.__new__(AWMHabitatGSEvaluator)
    evaluator.config = SimpleNamespace(benchmark="short_horizon_ovon")
    evaluator._habitat_sim = SimpleNamespace(MultiGoalShortestPath=lambda: SimpleNamespace(geodesic_distance=2.0))
    episode = {"cam_height_m": 0.35, "start_xyz_floor": [0, 0, 0], "reference_goal_positions": [[0, 0, -2]]}
    camera = np.array([0, 0.35, 0], dtype=np.float32)
    evaluator._reference_distance(SimpleNamespace(pathfinder=pathfinder), episode, camera, [0, 0, -2])
    evaluator._reference_distance(SimpleNamespace(pathfinder=pathfinder), episode, [1, 2.35, 3], [0, 0, -2])
    np.testing.assert_allclose(queried, [[0, 0, 0], [1, 2, 3]], atol=1e-6)
    np.testing.assert_allclose(camera, [0, 0.35, 0])
    evaluator.config.benchmark = "habitatgs_ln"
    evaluator._geodesic = lambda sim, start, goal: np.asarray(start).tolist()
    assert evaluator._reference_distance(None, episode, [1, 2.35, 3], [0, 0, -2]) == [1, 2.35, 3]
