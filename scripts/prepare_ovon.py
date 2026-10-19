"""Prepare Short-Horizon OVON from official tasks and licensed HM3D scenes."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np

ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations/evaluation/short_horizon_ovon.json"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def upstream_file(spec, source, download):
    path = source / spec["filename"]
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {path}; obtain the official data or pass --download")
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(spec["repository"], spec["filename"], repo_type="dataset",
                                   revision=spec["revision"], local_dir=source,
                                   endpoint="https://huggingface.co"))
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Upstream checksum mismatch: {path}")
    return path


def prepare_upstream(source, metadata, download):
    source.mkdir(parents=True, exist_ok=True)
    poses = upstream_file(metadata["start_poses"], source, download)
    content = source / "episodes/hm3d/val_unseen/content"
    missing = [name for name in metadata["episode_files"] if not (content / name).is_file()]
    if missing:
        archive = upstream_file(metadata["episodes"], source, download)
        with tarfile.open(archive, "r:gz") as stream:
            for name in missing:
                member = stream.getmember(f"hm3d/val_unseen/content/{name}")
                if not member.isfile():
                    raise ValueError(f"Expected an episode file in the official archive: {name}")
                target = content / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with stream.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha256(temporary) != metadata["episode_files"][name]:
                    raise ValueError(f"Upstream episode checksum mismatch: {name}")
                temporary.replace(target)
    for name, expected in metadata["episode_files"].items():
        if sha256(content / name) != expected:
            raise ValueError(f"Upstream episode checksum mismatch: {name}")
    return json.loads(poses.read_text()), content


def scene_config(scenes):
    return {"stages": {
        "default_attributes": {
            "shader_type": "flat", "up": [0, 0, 1], "front": [0, 1, 0], "origin": [0, 0, 0],
            "semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.txt",
            "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.glb",
            "has_semantic_textures": True,
        },
        "paths": {".glb": [f"scenes/val/{scene}/*.basis.glb" for scene in sorted(scenes)]},
    }, "scene_instances": {"default_attributes": {"default_lighting": "no_lights"}}}


def extract_scene_archives(archives, destination, scenes, checksums):
    selected = {name: info for name, info in checksums.items() if name.split("/")[1] in scenes}
    for archive_name, suffixes in (
        ("hm3d-val-habitat-v0.2.tar", (".basis.glb", ".basis.navmesh")),
        ("hm3d-val-semantic-annots-v0.2.tar", (".semantic.glb", ".semantic.txt")),
    ):
        missing = {}
        for name, info in selected.items():
            if not name.endswith(suffixes):
                continue
            target = destination / name
            if target.is_file():
                if target.stat().st_size != info["size"] or sha256(target) != info["sha256"]:
                    raise ValueError(f"HM3D scene checksum mismatch: {target}")
            else:
                missing[name.removeprefix("val/")] = (target, info)
        if not missing:
            continue
        with tarfile.open(Path(archives) / archive_name, "r:") as stream:
            for member in stream:
                name = member.name.removeprefix("./")
                if name not in missing:
                    continue
                target, info = missing[name]
                if not member.isfile() or member.size != info["size"]:
                    raise ValueError(f"Invalid HM3D archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                try:
                    with stream.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    if sha256(temporary) != info["sha256"]:
                        raise ValueError(f"HM3D scene checksum mismatch: {member.name}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                del missing[name]
                if not missing:
                    break
        if missing:
            raise FileNotFoundError(f"Missing HM3D assets in {archive_name}: {', '.join(missing)}")


def validate_scenes(source, output, scenes, checksums):
    links = []
    for scene in sorted(scenes):
        short = scene.split("-", 1)[1]
        directory = source / "val" / scene
        for suffix in ("basis.glb", "basis.navmesh", "semantic.glb", "semantic.txt"):
            path = directory / f"{short}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing licensed HM3D v0.2 asset: {path}")
            expected = checksums[str(path.relative_to(source))]
            if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"HM3D scene checksum mismatch: {path}")
        target = output / "scenes/val" / scene
        if target.exists() or target.is_symlink():
            if target.resolve() != directory.resolve():
                raise ValueError(f"Existing scene points to a different source: {target}")
        links.append((target, directory.resolve()))
    return links


def link_scenes(links):
    for target, directory in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != directory:
                raise ValueError(f"Existing scene points to a different source: {target}")
        else:
            target.symlink_to(directory, target_is_directory=True)


def object_footprint(obj, label):
    half = np.asarray(obj.obb.half_extents, dtype=float)
    transform = np.asarray(obj.obb.local_to_world, dtype=float).reshape(4, 4)
    corners = np.asarray([(transform @ [x * half[0], y * half[1], z * half[2], 1])[:3]
                          for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    points = sorted({(round(x, 4), round(z, 4)) for x, z in corners[:, [0, 2]]})
    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    lower, upper = [], []
    for chain, values in ((lower, points), (upper, reversed(points))):
        for point in values:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError(f"Selected object has a degenerate footprint: {obj.semantic_id}")
    return {"obj_id": str(obj.semantic_id), "label": label,
            "world": np.asarray(obj.obb.center, dtype=float).tolist(),
            "footprint_world_xz": [list(point) for point in hull],
            "ymin": float(corners[:, 1].min()), "ymax": float(corners[:, 1].max())}


def polygon_distances(points, polygon):
    polygon = np.asarray(polygon, dtype=float)
    inside = np.ones(len(points), dtype=bool)
    distances = np.full(len(points), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        offset = points - a
        t = np.clip((offset @ edge) / max(np.dot(edge, edge), 1e-9), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(points - (a + t[:, None] * edge), axis=1))
        inside &= edge[0] * offset[:, 1] - edge[1] * offset[:, 0] >= 0
    distances[inside] = 0
    return distances


def reference_points(pathfinder, start, instances, sampling, success):
    start = np.asarray(start, dtype=np.float32)
    island = pathfinder.get_island(start)
    candidates = []
    for instance in instances:
        if not (instance["ymin"] <= start[1] + success["floor_above_m"]
                and instance["ymax"] >= start[1] - success["floor_below_m"]):
            continue
        polygon = np.asarray(instance["footprint_world_xz"])
        low, high = polygon.min(axis=0) - sampling["grid_margin_m"], polygon.max(axis=0) + sampling["grid_margin_m"]
        x, z = np.meshgrid(np.arange(low[0], high[0], sampling["grid_step_m"]),
                           np.arange(low[1], high[1], sampling["grid_step_m"]), indexing="ij")
        points = np.column_stack((x.ravel(), z.ravel()))
        candidates.append((instance, points, polygon_distances(points, polygon)))
    for band in sampling["bands_m"]:
        goals = []
        for instance, points, distances in candidates:
            for x, z in points[distances <= band]:
                point = np.asarray(pathfinder.snap_point(np.array([x, start[1], z], dtype=np.float32)), dtype=np.float32)
                if (np.isfinite(point).all() and pathfinder.get_island(point) == island
                        and (instance["ymin"] - 0.6 <= float(point[1]) <= instance["ymax"] + 0.3
                             or abs(float(point[1]) - float(start[1])) < 0.5)
                        and math.hypot(float(point[0]) - x, float(point[2]) - z) < sampling["snap_max_planar_m"]):
                    goals.append(point)
        if goals:
            return np.asarray(goals[:sampling["max_reference_points"]], dtype=np.float32), band
    raise ValueError("No same-island reference points near the selected target instances")


def start_pose(record, pathfinder, camera):
    ground = np.asarray(record["habitat_start_position"], dtype=float)
    ground[1] -= camera["upstream_height_m"]
    ground = np.asarray(pathfinder.snap_point(ground), dtype=float)
    if not np.isfinite(ground).all():
        raise ValueError("The official start pose cannot be snapped to the scene navmesh")
    w, x, y, z = record["habitat_quaternion_wxyz"]
    yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(math.radians(camera["cam_pitch_deg"]) / 2), math.sin(math.radians(camera["cam_pitch_deg"]) / 2)
    return ground, [cp * cy, -sp * cy, cp * sy, -sp * sy]


def evaluation_rows(annotation, subset):
    rows = annotation["episodes"]
    keys = [f"{row['scene_id']}/{row['episode_id']}" for row in rows]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicated OVON episode selection")
    if subset != "full":
        selected = annotation.get("subsets", {}).get(subset, [])
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(keys):
            raise ValueError(f"Invalid OVON subset: {subset}")
        indexed = dict(zip(keys, rows))
        rows = [indexed[key] for key in selected]
    return rows


def prepare(args):
    import habitat_sim

    annotation_path = Path(getattr(args, "annotations", None) or ANNOTATIONS)
    annotation = json.loads(annotation_path.read_text())
    if annotation.get("benchmark") != "short_horizon_ovon":
        raise ValueError("Expected Short-Horizon OVON annotations")
    source, scenes, output = Path(args.source), Path(args.scenes), Path(args.output)
    subset = getattr(args, "subset", "full")
    rows = evaluation_rows(annotation, subset)
    settings = {"annotations_sha256": sha256(annotation_path), "source": str(source.resolve()),
                "scenes": str(scenes.resolve()), "benchmark": annotation["benchmark"], "subset": subset}
    binding = output / "preparation.json"
    if binding.exists() and json.loads(binding.read_text()) != settings:
        raise ValueError("Output belongs to another preparation; choose a new --output")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    if getattr(args, "hm3d_archives", None) is not None:
        extract_scene_archives(args.hm3d_archives, scenes, grouped, annotation["upstream"]["scene_files"])
    links = validate_scenes(scenes, output, grouped, annotation["upstream"]["scene_files"])
    metadata = dict(annotation["upstream"])
    metadata["episode_files"] = {f"{scene.split('-', 1)[1]}.json.gz":
                                 metadata["episode_files"][f"{scene.split('-', 1)[1]}.json.gz"]
                                 for scene in grouped}
    poses, content = prepare_upstream(source, metadata, args.download)
    poses = {(row["scene_id"], row["traj_id"]): row for row in poses}
    write_json(binding, settings)
    link_scenes(links)
    config = output / "scenes.scene_dataset_config.json"
    write_json(config, scene_config(grouped))
    camera, success = annotation["camera"], annotation["success"]
    episodes = []
    for scene, selection in sorted(grouped.items()):
        short = scene.split("-", 1)[1]
        scene_path = f"scenes/val/{scene}/{short}.basis.glb"
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str((output / scene_path).resolve())
        cfg.scene_dataset_config_file = str(config.resolve())
        cfg.enable_physics = False
        cfg.create_renderer = False
        cfg.load_semantic_mesh = True
        cfg.use_semantic_textures = False
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration(sensor_specifications=[])]))
        try:
            if not sim.pathfinder.is_loaded:
                raise ValueError(f"Scene navmesh did not load: {scene}")
            objects = {obj.semantic_id: obj for obj in sim.semantic_scene.objects}
            with gzip.open(content / f"{short}.json.gz", "rt") as stream:
                upstream = json.load(stream)
            tasks = {"traj_" + str(row["episode_id"]): row for row in upstream["episodes"]}
            for row in selection:
                key = (scene, row["episode_id"])
                task = tasks[row["episode_id"]]
                category = task["object_category"]
                goals = upstream["goals_by_category"][f"{short}.basis.glb_{category}"]
                upstream_ids = {int(str(goal["object_id"]).rsplit("_", 1)[1]) for goal in goals}
                if not set(row["support_obj_ids"]).issubset(upstream_ids):
                    raise ValueError(f"Selected object IDs differ from official goals: {key}")
                instances = [object_footprint(objects[sid], category) for sid in row["support_obj_ids"]]
                ground, quaternion = start_pose(poses[key], sim.pathfinder, camera)
                points, band = reference_points(sim.pathfinder, ground, instances, annotation["reference_sampling"], success)
                query = habitat_sim.MultiGoalShortestPath()
                query.requested_start = ground.astype(np.float32)
                query.requested_ends = points
                if not sim.pathfinder.find_path(query) or not math.isfinite(query.geodesic_distance):
                    raise ValueError(f"Reference path is not reachable: {key}")
                start = ground.copy()
                start[1] += camera["cam_height_m"]
                goal = min((np.asarray(goal["position"], dtype=float) for goal in goals), key=lambda point: np.linalg.norm(point - ground))
                episode = {"scene_id": scene, "episode_id": row["episode_id"], "instruction": row["instruction"],
                           "start_xyz": start.tolist(), "start_xyz_floor": ground.tolist(), "start_quat_wxyz": quaternion,
                           "goal_xyz": goal.tolist(), "scene_path": scene_path,
                           "navmesh_path": f"scenes/val/{scene}/{short}.basis.navmesh",
                           "action_convention": "scan2d_body_forward_right_clockwise", "action_dt_s": 0.25,
                           "target_mode": "object", "object_category": category,
                           "scan_body_radius_m": success["scan_body_radius_m"], "scan_body_offset_m": success["scan_body_offset_m"],
                           "target_contract": {"kind": "object", "supporting_instances": instances,
                                               "floor_filter": {"below_m": success["floor_below_m"], "above_m": success["floor_above_m"]},
                                               "success_condition": {"type": "distance_to_target_footprint", "max_distance_m": success["max_distance_m"], "strict": True}},
                           "reference_goal_positions": points.tolist(), "reference_band_m": band,
                           "reference_distance_m": float(query.geodesic_distance)}
                episode.update({name: camera[name] for name in ("hfov_deg", "cam_height_m", "cam_pitch_deg", "cam_forward_m")})
                episodes.append(episode)
        finally:
            sim.close()
        print(f"Prepared {scene}: {len(selection)} episodes", flush=True)
    order = {(row["scene_id"], row["episode_id"]): i for i, row in enumerate(rows)}
    episodes.sort(key=lambda row: order[(row["scene_id"], row["episode_id"])])
    write_json(output / "episodes.json", {"benchmark": annotation["benchmark"], "episodes": episodes})
    print(f"Prepared {len(episodes)} episodes in {len(grouped)} scenes: {output / 'episodes.json'}")
    return output / "episodes.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--hm3d-archives", type=Path,
                        help="Extract only selected scene assets from official HM3D v0.2 TAR files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--subset", choices=["test_subset", "full"], default="full")
    parser.add_argument("--download", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
