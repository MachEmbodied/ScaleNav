"""Prepare ScaleNav training data and separately obtained evaluation scenes."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / 'annotations'
INTERNET_FILTER = 'fps=4:eof_action=pass,scale=512:288:flags=lanczos,setsar=1'
EVALUATION = ANNOTATIONS / 'evaluation/habitatgs_ln.json'
INTERNET_MEDIA = ANNOTATIONS / 'internet-media.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_annotations(kind):
    manifest = json.loads((ANNOTATIONS / 'manifest.json').read_text())
    names = [f'{kind}.jsonl.gz'] + (['simulation.npz'] if kind == 'simulation' else [])
    for name in names:
        if sha256(ANNOTATIONS / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Annotation checksum mismatch: {name}')
    with gzip.open(ANNOTATIONS / f'{kind}.jsonl.gz', 'rt') as stream:
        return manifest[kind], [json.loads(line) for line in stream]


def upstream_file(repo, revision, filename, source, download):
    path = source / filename
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f'{path}; obtain the data or pass --download after obtaining access')
        from huggingface_hub import hf_hub_download, hf_hub_url
        from huggingface_hub.utils import LocalEntryNotFoundError, build_hf_headers
        try:
            path = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=revision,
                                        local_dir=source, endpoint='https://huggingface.co'))
        except LocalEntryNotFoundError:
            # Some proxies reject Hub metadata HEAD requests but allow authenticated GET.
            import requests
            url = hf_hub_url(repo, filename, repo_type='dataset', revision=revision,
                             endpoint='https://huggingface.co')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.partial')
            with requests.get(url, headers=build_hf_headers(), stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1 << 20):
                        stream.write(block)
                length = response.headers.get('Content-Length')
                if length and temporary.stat().st_size != int(length):
                    raise IOError(f'Incomplete upstream download: {filename}')
            temporary.replace(path)
    return path


def bind_output(output, settings):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'preparation.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Output belongs to a different preparation; choose a new --output directory')
    write_json(path, settings)


def select_rows(rows, args, kind=None):
    if getattr(args, 'subset', 'full') != 'full':
        manifest = json.loads((ANNOTATIONS / 'manifest.json').read_text())
        ids = manifest['subsets'][args.subset][kind]
        field = 'segment_id' if kind == 'internet' else 'key'
        selected_ids = set(ids)
        rows = [r for r in rows if r[field] in selected_ids]
        if len(rows) != len(selected_ids) or len(ids) != len(selected_ids):
            raise ValueError(f'Invalid {args.subset} subset for {kind}')
    selected = [r for r in rows if args.split == 'all' or r['split'] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('Empty selection; the smoke subset contains training samples only')
    if any(args.instructions not in r['instructions'] for r in selected):
        raise ValueError(f'Instruction set {args.instructions!r} is unavailable for this collection')
    return selected


def subset_settings(rows, args):
    if getattr(args, 'subset', 'full') == 'full':
        return {}
    return {'subset': args.subset, 'selected_ids': [r.get('segment_id', r.get('key')) for r in rows]}


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                             '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    numerator, denominator = map(float, stream['avg_frame_rate'].split('/'))
    return stream['width'], stream['height'], numerator / denominator, int(stream['nb_read_frames'])


def check_video(path, minimum, exact=False):
    width, height, fps, frames = probe_video(path)
    if (width, height) != (512, 288) or abs(fps - 4) > 1e-6 or frames < minimum or (exact and frames != minimum):
        raise ValueError(f'Unexpected video geometry or frame count: {path}: {(width, height, fps, frames)}')


def convert_internet_video(source, destination, minimum):
    temporary = destination.with_suffix('.partial.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(source), '-vf',
                    INTERNET_FILTER, '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
                    '-pix_fmt', 'yuv420p', '-threads', '2', str(temporary)], check=True)
    check_video(temporary, minimum)
    temporary.replace(destination)


def extract_clips(archive, requested, output):
    missing = set(requested)
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = member.name.removeprefix('./').removeprefix('SpatialVID/')
            if name not in missing:
                continue
            if not member.isfile():
                raise ValueError(f'Not a regular video: {name}')
            target = output / Path(name).name
            temporary = target.with_suffix('.partial')
            with stream.extractfile(member) as source, temporary.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
            missing.remove(name)
            if not missing:
                break
    if missing:
        raise ValueError(f'Archive lacks {len(missing)} requested clips: {archive}')


def verified_media_file(manifest, entry, source, download):
    path = upstream_file(manifest['repo_id'], manifest['revision'], entry['path'], source, download)
    if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError(f'Media checksum mismatch: {path}; remove this cached file and retry')
    return path


def extract_media(archive, requested, videos):
    remaining = set(requested)
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            if member.name in ('LICENSE', 'SOURCE.txt'):
                if not member.isfile() or member.size > 100_000:
                    raise ValueError(f'Invalid media notice: {member.name}')
                (videos.parent / member.name).write_bytes(stream.extractfile(member).read())
                continue
            if member.name not in requested:
                continue
            entry = requested[member.name]
            if member.name not in remaining or not member.isfile() or member.size != entry['bytes']:
                raise ValueError(f'Invalid media member: {member.name}')
            destination = videos / Path(member.name).name
            temporary = destination.with_suffix('.partial.mp4')
            with stream.extractfile(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            if sha256(temporary) != entry['sha256']:
                temporary.unlink()
                raise ValueError(f'Video checksum mismatch: {member.name}')
            temporary.replace(destination)
            remaining.remove(member.name)
    if remaining:
        raise ValueError(f'Media archive lacks {len(remaining)} selected videos: {archive}')


def prepare_released_internet(args, rows, videos):
    manifest = json.loads(INTERNET_MEDIA.read_text())
    if manifest['annotations_sha256'] != sha256(ANNOTATIONS / 'internet.jsonl.gz'):
        raise ValueError('Internet media and annotations belong to different releases')
    index_path = verified_media_file(manifest, manifest['index'], args.source, args.download)
    with gzip.open(index_path, 'rt') as stream:
        entries = [json.loads(line) for line in stream]
    index = {entry['clip']: entry for entry in entries}
    if len(index) != len(entries):
        raise ValueError('Duplicate clip in Internet media index')
    files = {entry['path']: entry for entry in manifest['files']}
    archives = defaultdict(dict)
    for row in rows:
        entry = index.get(row['clip'])
        if (entry is None or any(entry[k] != row[k] for k in ('split', 'archive', 'source_video'))
                or (entry['width'], entry['height'], entry['fps']) != (512, 288, 4)
                or not 0 <= row['t0'] < row['t_end'] <= entry['num_frames']):
            raise ValueError(f'Media does not match annotation: {row["segment_id"]}')
        archives[entry['file']][row['clip']] = entry
    if args.subset == 'smoke':
        archives = {manifest['smoke_file']: {clip: entry for group in archives.values()
                                           for clip, entry in group.items()}}
    for name, clips in sorted(archives.items()):
        pending = {}
        for clip, entry in clips.items():
            path = videos / f'{clip}.mp4'
            if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                pending[f'videos/{clip}.mp4'] = entry
        if pending:
            archive = verified_media_file(manifest, files[name], args.source, args.download)
            extract_media(archive, pending, videos)
        print(f'Prepared {name}: {len(clips)} clips', flush=True)
    shutil.copyfile(ANNOTATIONS / 'LICENSE', args.output / 'LICENSE')
    shutil.copyfile(ANNOTATIONS / 'SPATIALVID_NOTICE.txt', args.output / 'SOURCE.txt')
    return manifest


def prepare_upstream_internet(args, info, rows, videos, ends):
    archives = defaultdict(dict)
    for row in rows:
        archives[row['archive']][row['source_video']] = row['clip']
    scratch = args.output / 'source-clips'
    scratch.mkdir(exist_ok=True)
    for archive, clips in sorted(archives.items()):
        pending = {name: clip for name, clip in clips.items() if not (videos / f'{clip}.mp4').exists()}
        missing = {name: clip for name, clip in pending.items() if not (scratch / f'{clip}.mp4').exists()}
        if missing:
            source = upstream_file(info['repo_id'], info['revision'], archive, args.source, args.download)
            extract_clips(source, missing, scratch)
        for clip in pending.values():
            raw = scratch / f'{clip}.mp4'
            convert_internet_video(raw, videos / f'{clip}.mp4', ends[clip])
            raw.unlink()
        print(f'Prepared {archive}: {len(clips)} clips', flush=True)


def prepare_internet(args):
    info, rows = load_annotations('internet')
    rows = select_rows(rows, args, 'internet')
    upstream = getattr(args, 'from_upstream', False)
    settings = {'collection': 'internet', 'annotations_sha256': sha256(ANNOTATIONS / 'internet.jsonl.gz'),
                'instructions': args.instructions, 'split': args.split, 'limit': args.limit,
                'video_filter': INTERNET_FILTER, **subset_settings(rows, args)}
    if not upstream:
        settings['media_manifest_sha256'] = sha256(INTERNET_MEDIA)
    bind_output(args.output, settings)
    videos = args.output / 'videos'
    videos.mkdir(exist_ok=True)
    ends = defaultdict(int)
    for row in rows:
        ends[row['clip']] = max(ends[row['clip']], row['t_end'])
    if upstream:
        prepare_upstream_internet(args, info, rows, videos, ends)
    else:
        prepare_released_internet(args, rows, videos)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['split'], row['clip'])].append({
            'clip': row['clip'], 'segment_id': row['segment_id'], 't0': row['t0'], 't_end': row['t_end'],
            'instruction': row['instructions'][args.instructions], 'confidence': 'A'})
    for (split, clip), labels in grouped.items():
        if upstream:
            check_video(videos / f'{clip}.mp4', ends[clip])
        write_json(args.output / split / f'{clip}.json', labels)
    for split in sorted({r['split'] for r in rows}):
        config = {'vae_kwargs': {'vae_subpath': 'Wan2.2_VAE.pth'}, 'datasets': [{'type': 'nav_t2v',
                  'meta_path': 'video_segments.jsonl', 'domains': [{'perclip_dir': str(args.output / split),
                                                                     'pool_root': str(args.output)}]}]}
        write_json(args.output / f'{split}.yaml', config)


def camera_positions(position, quat, forward):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat.T
    yaw = np.arctan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
    result = np.array(position, copy=True)
    result[:, 0] -= np.sin(yaw).astype(result.dtype) * forward
    result[:, 2] -= np.cos(yaw).astype(result.dtype) * forward
    return result


def scene_config():
    return {'stages': {'default_attributes': {'render_asset_type': 'gaussian_splatting', 'units_to_meters': 1.0,
            'up': [0, 1, 0], 'front': [0, 0, -1], 'origin': [0, 0, 0], 'frustum_culling': False}},
            'scene_instances': {}, 'navmesh_instances': {}}


def make_sim(scene, config, hfov, gpu):
    import habitat_sim as hab
    import magnum as mn
    sim = hab.SimulatorConfiguration()
    sim.scene_id = str(scene)
    sim.scene_dataset_config_file = str(config)
    sim.enable_physics = False
    sim.gpu_device_id = gpu
    sensor = hab.CameraSensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = hab.SensorType.COLOR
    sensor.resolution = [288, 512]
    sensor.hfov = hfov
    sensor.position = mn.Vector3(0, 0, 0)
    agent = hab.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    return hab.Simulator(hab.Configuration(sim, [agent]))


def render_video(scene, config, row, positions, quats, destination, gpu):
    import imageio.v2 as imageio
    import quaternion
    sim = make_sim(scene, config, row['hfov_deg'], gpu)
    temporary = destination.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(str(temporary), fps=4, codec='libx264', quality=8,
                               pixelformat='yuv420p', macro_block_size=1) as writer:
            agent = sim.get_agent(0)
            for position, quat in zip(camera_positions(positions, quats, row['cam_forward_m']), quats):
                state = agent.get_state()
                state.position = position
                state.rotation = quaternion.quaternion(*map(float, quat))
                agent.set_state(state, infer_sensor_states=True)
                writer.append_data(sim.get_sensor_observations()['rgb'][..., :3])
        check_video(temporary, row['num_frames'], exact=True)
        temporary.replace(destination)
    finally:
        sim.close()


def prepare_simulation(args):
    info, rows = load_annotations('simulation')
    rows = select_rows(rows, args, 'simulation')
    if any(not (args.scenes / r['scene'] / f"{r['scene']}.gs.ply").is_file() for r in rows):
        require_tools(args.converter)
    bind_output(args.output, {'collection': 'simulation', 'annotations_sha256': sha256(ANNOTATIONS / 'simulation.jsonl.gz'),
                              'trajectories_sha256': sha256(ANNOTATIONS / 'simulation.npz'),
                              'instructions': args.instructions, 'split': args.split, 'limit': args.limit,
                              **subset_settings(rows, args)})
    config = args.output / 'scenes.scene_dataset_config.json'
    write_json(config, scene_config())
    meta = []
    with np.load(ANNOTATIONS / 'simulation.npz', allow_pickle=False) as data:
        data = {name: data[name] for name in data.files}
        for row in rows:
            scene = args.scenes / row['scene'] / f"{row['scene']}.gs.ply"
            if not scene.exists():
                raw = upstream_file(info['repo_id'], info['revision'], f"{row['scene']}/3dgs_compressed.ply",
                                    args.source, args.download)
                scene.parent.mkdir(parents=True, exist_ok=True)
                temporary = scene.with_name('converted.partial.ply')
                # SplatTransform's PLY convention maps +90 here to (X, Z, -Y).
                subprocess.run([args.converter, str(raw), '--rotate', '90,0,0', str(temporary)], check=True)
                temporary.replace(scene)
            folder = args.output / row['key']
            folder.mkdir(parents=True, exist_ok=True)
            p, a, n = row['pose_offset'], row['action_offset'], row['num_frames']
            position, quat = data['position'][p:p+n], data['quat_wxyz'][p:p+n]
            actions = data['action_delta_body'][a:a+n-1]
            if len(position) != n or len(quat) != n or len(actions) != n-1:
                raise ValueError(f"Truncated trajectory: {row['key']}")
            if not (folder / 'video.mp4').exists():
                render_video(scene, config, row, position, quat, folder / 'video.mp4', args.gpu)
            check_video(folder / 'video.mp4', n, exact=True)
            np.save(folder / 'actions.npy', actions, allow_pickle=False)
            np.savez_compressed(folder / 'trajectory.npz', position=position, quat_wxyz=quat,
                                action_delta_body=actions, fps=np.float32(4), action_dt=np.float32(.25))
            instruction = row['instructions'][args.instructions]
            meta.append({'key': row['key'], 'scene': row['scene'], 'episode': row['episode'],
                         'video': f"{row['key']}/video.mp4", 'actions': f"{row['key']}/actions.npy",
                         'canonical': instruction, 'l3_canonical': instruction, 'num_frames': n})
            print(f"Prepared {row['key']}", flush=True)
    (args.output / 'awm_meta.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in meta))
    for split in ['train', 'val']:
        (args.output / f'{split}.txt').write_text(''.join(r['key'] + '\n' for r in rows if r['split'] == split))


def target_footprint(labels, instance_id):
    from scipy.spatial import ConvexHull, QhullError
    matches = [x for x in labels if str(x['ins_id']) == str(instance_id)]
    if len(matches) != 1:
        raise ValueError(f'Expected one instance {instance_id}, found {len(matches)}')
    points = np.array([[p['x'], -p['y']] for p in matches[0]['bounding_box']], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite target footprint')
    points = np.unique(points, axis=0)
    try:
        return points[ConvexHull(points).vertices].tolist()
    except QhullError as exc:
        raise ValueError('Degenerate target footprint') from exc


def evaluation_rows(manifest, subset):
    rows = manifest['episodes']
    keys = [f"{r['scene_id']}/{r['episode_id']}" for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate evaluation episode identifiers')
    if subset != 'full':
        selected = manifest['subsets'][subset]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(keys):
            raise ValueError(f'Invalid evaluation subset: {subset}')
        indexed = dict(zip(keys, rows))
        rows = [indexed[key] for key in selected]
    return rows


def prepare_evaluation(args):
    manifest = json.loads(EVALUATION.read_text())
    rows = evaluation_rows(manifest, args.subset)
    scene_ids = sorted({row['scene_id'] for row in rows})
    local_scenes = getattr(args, 'scenes', None) or args.source / 'scenes'
    upstream_paths = manifest['upstream']['scenes']['paths']
    reconstruct = [scene for scene in scene_ids if scene not in upstream_paths and
                   not (local_scenes / scene / f'{scene}.gs.ply').is_file()]
    if reconstruct:
        require_tools(getattr(args, 'converter', 'splat-transform'))
    navmeshes = {}
    upstream = manifest['upstream']['navmeshes']
    for scene in scene_ids:
        path = local_scenes / scene / f'{scene}.navmesh'
        if path.is_file():
            expected = manifest['assets'][scene]['navmesh']
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Evaluation checksum mismatch: {path}')
            navmeshes[scene] = path
    for scene in scene_ids:
        if scene in navmeshes:
            continue
        path = upstream_file(upstream['repo_id'], upstream['revision'],
                             f"{upstream['prefix']}/{scene}.navmesh", args.source, args.download)
        expected = manifest['assets'][scene]['navmesh']
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Evaluation checksum mismatch: {path}')
        navmeshes[scene] = path
    bind_output(args.output, {'collection': manifest['benchmark'], 'subset': args.subset,
                              'annotations_sha256': sha256(EVALUATION)})
    scenes = {}
    for scene in scene_ids:
        assets = {'navmesh': navmeshes[scene].resolve()}
        for suffix in ('labels.json', 'gs.ply'):
            labels = suffix == 'labels.json'
            upstream = manifest['upstream']['labels' if labels else 'scenes']
            path = local_scenes / scene / ('labels.json' if labels else f'{scene}.{suffix}')
            if suffix == 'gs.ply' and scene in reconstruct:
                official = manifest['upstream']['labels']
                raw = upstream_file(official['repo_id'], official['revision'], f'{scene}/3dgs_compressed.ply',
                                    args.source, args.download)
                spec = importlib.util.spec_from_file_location('prepare_gs_assets', ROOT / 'scripts/prepare_gs_assets.py')
                converter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(converter)
                expected = manifest['assets'][scene]['gs.ply']
                converter.reconstruct_runtime_gs(raw, path, expected_sha256=expected['sha256'],
                                                 expected_bytes=expected['bytes'],
                                                 converter=getattr(args, 'converter', 'splat-transform'))
            elif not path.is_file():
                filename = f'{scene}/labels.json' if labels else f'{upstream_paths[scene]}/interior_{scene}.{suffix}'
                path = upstream_file(upstream['repo_id'], upstream['revision'], filename, args.source, args.download)
            if sha256(path) != manifest['assets'][scene][suffix]['sha256']:
                raise ValueError(f'Evaluation checksum mismatch: {path}')
            assets[suffix] = path.resolve()
        scenes[scene] = assets
        print(f'Checked scene {scene}', flush=True)
    episodes = []
    for row in rows:
        assets = scenes[row['scene_id']]
        episode = {k: v for k, v in row.items() if k not in {'target_instance_id', 'target_label'}}
        episode['start_xyz'] = list(row['start_xyz'])
        if manifest['start_position_reference'] == 'floor':
            episode['start_xyz'][1] += episode['cam_height_m']
        episode.update(scene_path=str(assets['gs.ply']), navmesh_path=str(assets['navmesh']))
        if row['target_mode'] == 'doorway':
            episode['target_contract'] = {'success_condition': {'type': 'distance_to_navigation_goal'}}
        else:
            episode['target_contract'] = {'semantic_target_footprint_world_xz': target_footprint(
                json.loads(assets['labels.json'].read_text()), row['target_instance_id'])}
        episodes.append(episode)
    from awm.habitat_extensions.benchmark import prepare_episode
    episodes = [prepare_episode(episode, manifest['benchmark']) for episode in episodes]
    write_json(args.output / 'episodes.json', {'benchmark': manifest['benchmark'], 'episodes': episodes})
    write_json(args.output / 'scenes.scene_dataset_config.json', scene_config())
    print(f"Prepared {len(episodes)} episodes in {len(scenes)} scenes: {args.output / 'episodes.json'}")


def prepare_ovon(args):
    spec = importlib.util.spec_from_file_location('prepare_ovon', ROOT / 'scripts/prepare_ovon.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(args)


def prepare_example(args):
    summary_path = args.source / 'summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        matches = [row for row in summary['episodes']
                   if f"{row['scene_id']}/{row['episode_id']}" == args.episode]
        if len(matches) != 1:
            raise ValueError(f'Expected one evaluation episode {args.episode}, found {len(matches)}')
        record = matches[0]
        if not record.get('video_path'):
            raise ValueError('Evaluation has no video; run evaluation with --visualize')
        video = args.source / 'videos' / Path(record['video_path']).name
        check_video(video, record['steps_taken'] + 1, exact=True)
        instruction = record['instruction']
        provenance = {
            'benchmark': summary['benchmark'], 'instruction_variant': 'evaluation',
            'eval_seed': record.get('eval_seed'), 'summary_sha256': sha256(summary_path)}
    else:
        _, annotations = load_annotations('simulation')
        labels = [row for row in annotations if row['key'] == args.episode]
        if len(labels) != 1:
            raise ValueError(f'Unknown simulation episode: {args.episode}')
        label = labels[0]
        with (args.source / 'awm_meta.jsonl').open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        matches = [row for row in records if row['key'] == args.episode]
        if len(matches) != 1:
            raise ValueError(f'Expected one prepared episode {args.episode}, found {len(matches)}')
        record = matches[0]
        if (record['scene'] != label['scene'] or record['episode'] != label['episode']
                or record['num_frames'] != label['num_frames']
                or record['video'] != f"{label['key']}/video.mp4"):
            raise ValueError(f'Prepared episode differs from released annotations: {args.episode}')
        video = args.source / record['video']
        check_video(video, label['num_frames'], exact=True)
        instruction = label['instructions']['diverse_iv']
        provenance = {
            'instruction_variant': 'diverse_iv',
            'annotations_sha256': sha256(ANNOTATIONS / 'simulation.jsonl.gz')}
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.output / 'observation.png'
    temporary = image.with_suffix('.partial.png')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(video),
                    '-frames:v', '1', '-update', '1', str(temporary)], check=True)
    temporary.replace(image)
    (args.output / 'instruction.txt').write_text(instruction + '\n')
    write_json(args.output / 'source.json', {
        **provenance, 'episode': args.episode, 'frame_index': 0,
        'instruction': instruction, 'video_sha256': sha256(video),
        'image_sha256': sha256(image)})
    print(f"Prepared {args.episode} frame 0: {image}")
    print(instruction)


def require_tools(*names):
    missing = [name for name in names if not shutil.which(name)]
    if missing:
        raise ValueError(f'Missing executable(s): {", ".join(missing)}; see docs/DATA.md for installation')


def preflight(command):
    if command in ('internet', 'simulation', 'example'):
        require_tools('ffmpeg', 'ffprobe')
    if command == 'simulation':
        for name in ('habitat_sim', 'magnum'):
            try:
                module = importlib.import_module(name)
            except (ImportError, OSError) as exc:
                raise ValueError(f'Cannot load {name}: {exc}; see docs/EVALUATION.md#simulator-installation') from exc
            if name == 'habitat_sim' and not module.cuda_enabled:
                raise ValueError('Habitat-GS requires HABITAT_WITH_CUDA=ON; see docs/EVALUATION.md#simulator-installation')


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    example = commands.add_parser('example', help='Extract a paired first observation and original instruction')
    example.set_defaults(function=prepare_example)
    example.add_argument('--source', type=Path, required=True,
                         help='Prepared simulation directory or evaluation output with videos')
    example.add_argument('--episode', required=True,
                         help='Released simulation key or evaluation scene_id/episode_id')
    example.add_argument('--output', type=Path, required=True)
    for name, function in [('internet', prepare_internet), ('simulation', prepare_simulation), ('evaluation', prepare_evaluation), ('ovon', prepare_ovon)]:
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        sub.add_argument('--source', type=Path, required=True, help='Downloaded data cache directory')
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--download', action='store_true', help='Fetch missing files with your Hugging Face access')
        if name == 'internet':
            sub.add_argument('--from-upstream', action='store_true',
                             help='Rebuild from original SpatialVID shards instead of the compact release (up to 1.07 TB)')
        if name == 'ovon':
            sub.add_argument('--scenes', type=Path, required=True, help='Official HM3D assets extracted locally')
            sub.add_argument('--hm3d-archives', type=Path,
                             help='Extract only selected scene assets from official HM3D v0.2 TAR files')
            sub.add_argument('--subset', choices=['test_subset', 'full'], default='full')
        elif name == 'evaluation':
            sub.add_argument('--subset', choices=['test_subset', 'full'], default='test_subset')
            sub.add_argument('--scenes', type=Path, help='Optional local scene assets and original navigation meshes')
            sub.add_argument('--converter', default='splat-transform')
        else:
            sub.add_argument('--instructions', choices=['goto', 'diverse', 'diverse_iv'], default='diverse_iv')
            sub.add_argument('--split', choices=['train', 'val', 'all'], default='all')
            sub.add_argument('--subset', choices=['full', 'smoke'], default='full',
                             help='smoke: fixed training samples from one Internet shard or one simulation scene')
            sub.add_argument('--limit', type=int, help='Prepare only this many episodes for a small test')
        if name == 'simulation':
            sub.add_argument('--scenes', type=Path, required=True, help='Converted Habitat Y-up scene directory')
            sub.add_argument('--converter', default='splat-transform')
            sub.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if getattr(args, 'limit', None) is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    args.source, args.output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if getattr(args, 'scenes', None) is not None:
        args.scenes = args.scenes.expanduser().resolve()
    try:
        preflight(args.command)
        args.function(args)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
