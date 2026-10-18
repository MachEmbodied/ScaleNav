import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_data', ROOT / 'scripts/prepare_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_missing_tools_fail_before_downloading_or_writing(monkeypatch, tmp_path):
    import sys
    output = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare_data.py', 'internet', '--source', str(tmp_path / 'source'),
                                     '--output', str(output), '--download'])
    monkeypatch.setattr(prepare.shutil, 'which', lambda name: None)
    monkeypatch.setattr(prepare, 'upstream_file', lambda *args: pytest.fail('Download started before preflight'))
    with pytest.raises(SystemExit) as exc:
        prepare.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_simulator_preflight_rejects_cpu_only_build(monkeypatch):
    monkeypatch.setattr(prepare.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(prepare.importlib, 'import_module', lambda name: SimpleNamespace(cuda_enabled=False))
    with pytest.raises(ValueError, match='HABITAT_WITH_CUDA=ON'):
        prepare.preflight('simulation')


def test_released_annotations_cover_fixed_splits_and_aligned_actions():
    info, internet = prepare.load_annotations('internet')
    assert len(internet) == info['segments'] == 18249
    assert len({r['clip'] for r in internet}) == 13647
    assert sum(r['split'] == 'train' for r in internet) == 18184
    assert len({r['segment_id'] for r in internet}) == len(internet)
    train = {r['clip'] for r in internet if r['split'] == 'train'}
    val = {r['clip'] for r in internet if r['split'] == 'val'}
    assert not train & val
    for row in internet:
        assert row['t0'] >= 0 and row['t_end'] - row['t0'] >= 9 and row['fps'] == 4
        assert set(row['instructions']) == {'goto', 'diverse_iv'}
        assert row['source_video'].endswith(row['clip'] + '.mp4')
        assert not Path(row['source_video']).is_absolute()
    info, simulation = prepare.load_annotations('simulation')
    assert len(simulation) == info['episodes'] == 2600
    assert len({r['key'] for r in simulation}) == len(simulation)
    assert sum(r['split'] == 'train' for r in simulation) == 2460
    train = {r['scene'] for r in simulation if r['split'] == 'train'}
    val = {r['scene'] for r in simulation if r['split'] == 'val'}
    assert len(train) == 618 and len(val) == 32 and not train & val
    p = a = 0
    for row in simulation:
        assert (row['pose_offset'], row['action_offset']) == (p, a)
        assert set(row['instructions']) == {'goto', 'diverse', 'diverse_iv'}
        assert row['cam_forward_m'] >= 0 and 0 < row['hfov_deg'] < 180
        p += row['num_frames']
        a += row['num_frames'] - 1
    with np.load(prepare.ANNOTATIONS / 'simulation.npz', allow_pickle=False) as data:
        assert set(data.files) == {'position', 'quat_wxyz', 'action_delta_body'}
        assert data['position'].shape == (p, 3) == (103501, 3)
        assert data['quat_wxyz'].shape == (p, 4)
        assert data['action_delta_body'].shape == (a, 3) == (100901, 3)
        for name in data.files:
            assert data[name].dtype == np.float32 and np.isfinite(data[name]).all()
        np.testing.assert_allclose(np.linalg.norm(data['quat_wxyz'], axis=1), 1, atol=1e-6)


def test_inference_example_pairs_first_frame_with_original_diverse_instruction(tmp_path, monkeypatch):
    if not all(shutil.which(tool) for tool in ['ffmpeg', 'ffprobe']):
        pytest.skip('FFmpeg is required for the video extraction check')
    from PIL import Image
    import sys

    _, labels = prepare.load_annotations('simulation')
    label = next(row for row in labels if row['key'] == 'far_v070/0003_839989/seg_00000')
    source, output = tmp_path / 'simulation', tmp_path / 'example'
    video = source / label['key'] / 'video.mp4'
    video.parent.mkdir(parents=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'color=c=red:s=512x288:r=4:d=0.25',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=512x288:r=4:d=15.25',
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0',
                    '-c:v', 'libx264', '-threads', '2', str(video)], check=True)
    record = {name: label[name] for name in ['key', 'scene', 'episode', 'num_frames']}
    record.update(video=f"{label['key']}/video.mp4", canonical='go to the sofa')
    other = dict(record, key='other/episode', canonical='Wrong sample instruction')
    (source / 'awm_meta.jsonl').write_text(json.dumps(other) + '\n' + json.dumps(record) + '\n')
    monkeypatch.setattr(sys, 'argv', ['prepare_data.py', 'example', '--source', str(source),
                                     '--episode', label['key'], '--output', str(output)])
    prepare.main()
    assert (output / 'instruction.txt').read_text() == 'Please move to the sofa visible in your initial view\n'
    with Image.open(output / 'observation.png') as image:
        assert image.size == (512, 288)
        red, green, blue = image.convert('RGB').getpixel((256, 144))
        assert red > 240 and green < 10 and blue < 10
    provenance = json.loads((output / 'source.json').read_text())
    assert provenance['episode'] == label['key']
    assert provenance['frame_index'] == 0
    assert provenance['video_sha256'] == prepare.sha256(video)
    assert provenance['image_sha256'] == prepare.sha256(output / 'observation.png')


@pytest.mark.parametrize('invalid', ['unknown', 'missing', 'mismatched'])
def test_inference_example_rejects_unpaired_inputs(tmp_path, invalid):
    _, labels = prepare.load_annotations('simulation')
    label = labels[0]
    record = {name: label[name] for name in ['key', 'scene', 'episode', 'num_frames']}
    record['video'] = 'different/episode/video.mp4'
    (tmp_path / 'awm_meta.jsonl').write_text('' if invalid == 'missing' else json.dumps(record) + '\n')
    args = SimpleNamespace(source=tmp_path, output=tmp_path / 'example',
                           episode='unknown/episode' if invalid == 'unknown' else label['key'])
    with pytest.raises(ValueError, match='Unknown simulation episode|Expected one prepared episode|differs'):
        prepare.prepare_example(args)
    assert not args.output.exists()


def test_evaluation_example_keeps_first_frame_and_matching_scene_instruction(tmp_path):
    if not all(shutil.which(tool) for tool in ['ffmpeg', 'ffprobe']):
        pytest.skip('FFmpeg is required for the video extraction check')
    from PIL import Image

    video = tmp_path / 'videos' / 'scene__episode.mp4'
    video.parent.mkdir()
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'color=c=red:s=512x288:r=4:d=0.25',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=512x288:r=4:d=0.25',
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0',
                    '-c:v', 'libx264', '-threads', '2', str(video)], check=True)
    row = {'scene_id': 'scene', 'episode_id': 'episode', 'steps_taken': 1,
           'instruction': 'Walk to the coffee machine visible in your initial view.',
           'video_path': '/previous-server/videos/scene__episode.mp4', 'eval_seed': 42}
    summary = {'benchmark': 'short_horizon_ovon', 'episodes': [
        dict(row, scene_id='other', instruction='Different scene'), row]}
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps(summary))
    args = SimpleNamespace(source=tmp_path, episode='scene/episode', output=tmp_path / 'example')
    prepare.prepare_example(args)
    assert (args.output / 'instruction.txt').read_text().strip() == row['instruction']
    with Image.open(args.output / 'observation.png') as image:
        red, green, blue = image.convert('RGB').getpixel((256, 144))
        assert red > 240 and green < 10 and blue < 10
    provenance = json.loads((args.output / 'source.json').read_text())
    assert provenance['episode'] == 'scene/episode' and provenance['frame_index'] == 0
    assert provenance['eval_seed'] == 42
    assert provenance['summary_sha256'] == prepare.sha256(summary_path)
    assert provenance['video_sha256'] == prepare.sha256(video)
    summary['episodes'].append(row)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Expected one evaluation episode'):
        prepare.prepare_example(args)


def test_camera_offset_matches_world_forward_for_tilted_camera():
    rotations = Rotation.from_euler('xyz', [[12, 45, 6], [-3, -60, 2]], degrees=True)
    xyzw = rotations.as_quat()
    poses = np.array([[1., .35, 2.], [2., .4, 1.]], dtype=np.float32)
    forward = rotations.apply([0, 0, -1])
    forward[:, 1] = 0
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    actual = prepare.camera_positions(poses, xyzw[:, [3, 0, 1, 2]], .35)
    np.testing.assert_allclose(actual, poses + .35 * forward, atol=1e-6)
    np.testing.assert_array_equal(poses[:, 1], actual[:, 1])


@pytest.mark.parametrize('kind,count', [('internet', 17), ('simulation', 4)])
def test_smoke_subset_is_small_disjoint_and_reproducible(kind, count, tmp_path):
    _, rows = prepare.load_annotations(kind)
    args = SimpleNamespace(split='train', limit=None, instructions='goto', subset='smoke')
    selected = prepare.select_rows(rows, args, kind)
    assert len(selected) == count
    assert {r['split'] for r in selected} == {'train'}
    if kind == 'internet':
        assert {r['archive'] for r in selected} == {'videos/group_0074.tar.gz'}
        assert len({r['clip'] for r in selected}) == 14
        assert not {r['clip'] for r in selected} & {r['clip'] for r in rows if r['split'] == 'val'}
    else:
        assert {r['scene'] for r in selected} == {'0003_839989'}
        assert sum(r['num_frames'] for r in selected) == 170
        assert not {r['scene'] for r in selected} & {r['scene'] for r in rows if r['split'] == 'val'}
        evaluation = json.loads(prepare.EVALUATION.read_text())
        assert selected[0]['scene'] not in {r['scene_id'] for r in prepare.evaluation_rows(evaluation, 'test_subset')}
    settings = prepare.subset_settings(selected, args)
    prepare.bind_output(tmp_path, settings)
    prepare.bind_output(tmp_path, settings)
    args.limit = 1
    limited = prepare.select_rows(rows, args, kind)
    assert limited == selected[:1]
    with pytest.raises(ValueError, match='different preparation'):
        prepare.bind_output(tmp_path, prepare.subset_settings(limited, args))
    args.split = 'val'
    with pytest.raises(ValueError, match='training samples only'):
        prepare.select_rows(rows, args, kind)
    args.subset, args.limit = 'full', None
    assert prepare.select_rows(rows, args, kind) == [r for r in rows if r['split'] == 'val']
    assert prepare.subset_settings(rows, args) == {}


def test_archive_only_extracts_requested_regular_videos(tmp_path):
    archive = tmp_path / 'shard.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ['../../outside.mp4', 'SpatialVID/videos/group_0001/wanted.mp4', 'other.mp4']:
            entry = tarfile.TarInfo(name)
            entry.size = 5
            tar.addfile(entry, io.BytesIO(b'video'))
    dest = tmp_path / 'clips'
    dest.mkdir()
    prepare.extract_clips(archive, ['videos/group_0001/wanted.mp4'], dest)
    assert [p.name for p in dest.iterdir()] == ['wanted.mp4']
    assert (dest / 'wanted.mp4').read_bytes() == b'video'
    with pytest.raises(ValueError, match='lacks'):
        prepare.extract_clips(archive, ['videos/missing.mp4'], dest)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_resampling_keeps_the_terminal_observation(tmp_path):
    source, destination = tmp_path / 'source.mp4', tmp_path / 'prepared.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=512x288:rate=30', '-frames:v', '62', '-c:v', 'libx264',
                    '-threads', '2', str(source)], check=True)
    prepare.convert_internet_video(source, destination, 9)
    assert prepare.probe_video(destination) == (512, 288, 4, 9)


def test_instruction_changes_require_separate_preparation(tmp_path):
    prepare.bind_output(tmp_path, {'instructions': 'goto'})
    with pytest.raises(ValueError, match='different preparation'):
        prepare.bind_output(tmp_path, {'instructions': 'diverse_iv'})
    assert json.loads((tmp_path / 'preparation.json').read_text())['instructions'] == 'goto'
    args = SimpleNamespace(split='all', limit=None, instructions='diverse')
    with pytest.raises(ValueError, match='unavailable'):
        prepare.select_rows([{'split': 'train', 'instructions': {'goto': 'Go to the chair.'}}], args)


def test_evaluation_subsets_reuse_assets_and_preserve_episode_contracts(tmp_path, monkeypatch):
    from awm.habitat_extensions.benchmark import load_manifest
    labels = [{'ins_id': '80', 'bounding_box': [
        {'x': x, 'y': y, 'z': z} for x in [1., 2.] for y in [-4., -2.] for z in [0., 1.]]}]
    source = tmp_path / 'source'
    folder = source / '0022_840117'
    folder.mkdir(parents=True)
    (folder / 'labels.json').write_text(json.dumps(labels))
    assets = source / 'train/interior_0022_840117'
    assets.mkdir(parents=True)
    for suffix in ['gs.ply', 'navmesh']:
        (assets / f'interior_0022_840117.{suffix}').write_text(suffix)
    released = source / 'evaluation/habitatgs_ln/navmeshes/0022_840117.navmesh'
    released.parent.mkdir(parents=True)
    shutil.copyfile(assets / 'interior_0022_840117.navmesh', released)
    hashes = {'labels.json': prepare.sha256(folder / 'labels.json')}
    hashes.update({suffix: prepare.sha256(assets / f'interior_0022_840117.{suffix}') for suffix in ['gs.ply', 'navmesh']})
    manifest = json.loads(prepare.EVALUATION.read_text())
    row = manifest['episodes'][0]
    assert row['scene_id'] == '0022_840117' and row['target_instance_id'] == '80'
    doorway = dict(row, episode_id='doorway', target_mode='doorway', target_label='doorway')
    doorway.pop('target_instance_id')
    manifest['episodes'] = [row, doorway]
    manifest['subsets']['test_subset'] = [f"{row['scene_id']}/{row['episode_id']}"]
    manifest['assets'] = {row['scene_id']: {k: {'sha256': v} for k, v in hashes.items()}}
    manifest['assets'][row['scene_id']]['navmesh']['bytes'] = released.stat().st_size
    path = tmp_path / 'evaluation.json'
    prepare.write_json(path, manifest)
    monkeypatch.setattr(prepare, 'EVALUATION', path)
    args = SimpleNamespace(source=source, output=tmp_path / 'test_subset', download=False, subset='test_subset')
    prepare.prepare_evaluation(args)
    episode = load_manifest(args.output / 'episodes.json', 'habitatgs_ln')[0]
    assert episode['target_contract']['semantic_target_footprint_world_xz'] == [[1, 2], [2, 2], [2, 4], [1, 4]]
    assert episode['start_xyz'][1] == pytest.approx(.552846)
    assert episode['start_quat_wxyz'] != [1, 0, 0, 0]
    assert episode['cam_forward_m'] == .35
    args.subset = 'full'
    with pytest.raises(ValueError, match='different preparation'):
        prepare.prepare_evaluation(args)
    args.output = tmp_path / 'full'
    prepare.prepare_evaluation(args)
    full = load_manifest(args.output / 'episodes.json', 'habitatgs_ln')
    assert len(full) == 2 and full[0] == episode
    assert full[1]['target_mode'] == 'doorway'
    assert full[1]['scene_path'] == episode['scene_path']
    released.write_text('corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare.prepare_evaluation(args)


def test_rotated_target_retains_its_oriented_footprint():
    labels = [{'ins_id': '1', 'bounding_box': [
        {'x': x, 'y': -y, 'z': z} for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)] for z in [0, 2]]}]
    polygon = prepare.target_footprint(labels, '1')
    assert {tuple(p) for p in polygon} == {(0, 1), (1, 0), (0, -1), (-1, 0)}
    with pytest.raises(ValueError, match='Expected one instance'):
        prepare.target_footprint(labels, '2')


def test_small_subset_and_full_release_have_fixed_distinct_episode_ids():
    manifest = json.loads(prepare.EVALUATION.read_text())
    full, test_subset = [prepare.evaluation_rows(manifest, subset) for subset in ['full', 'test_subset']]
    assert len(full) == 157 and len({r['scene_id'] for r in full}) == 45
    assert len(test_subset) == 16 and len({r['scene_id'] for r in test_subset}) == 13
    assert manifest['subsets']['test_subset'][:6] == ['0022_840117/base_mixed_v4__seg_10000', '0022_840117/object_dense_v4__v03_gpu0__seg_00004', '0184_840116/near_dense_v3__v02_gpu0__seg_00000', '0184_840116/object_scale_v3__v02_gpu0__seg_00001', '0402_840115/object_dense_v5__v02_gpu1__seg_00004', '0184_840116/base_mixed_v4__seg_10000']
    assert '0022_840117/object_dense_v4__v03_gpu0__seg_00005' not in manifest['subsets']['test_subset']
    assert {r['scene_id'] for r in test_subset} <= manifest['assets'].keys()
    assert all(row in full for row in test_subset)
    _, training = prepare.load_annotations('simulation')
    assert not {r['scene_id'] for r in full} & {r['scene'] for r in training if r['split'] == 'train'}
    assert sum(row['target_mode'] == 'doorway' for row in full) == 3
    assert all('target_contract' not in row for row in full)
    manifest['subsets']['test_subset'].append('missing/episode')
    with pytest.raises(ValueError, match='Invalid evaluation subset'):
        prepare.evaluation_rows(manifest, 'test_subset')


def test_missing_converter_fails_before_full_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, 'upstream_file', lambda *args: pytest.fail('Download before scene preflight'))
    monkeypatch.setattr(prepare.shutil, 'which', lambda name: None)
    args = SimpleNamespace(source=tmp_path / 'source', output=tmp_path / 'full',
                           download=True, subset='full', scenes=None)
    with pytest.raises(ValueError, match='Missing executable'):
        prepare.prepare_evaluation(args)
    assert not args.output.exists()


def test_wrong_original_navmeshes_fail_before_reconstruction_or_downloads(tmp_path, monkeypatch):
    manifest = json.loads(prepare.EVALUATION.read_text())
    local = tmp_path / 'scenes'
    for scene in manifest['assets'].keys() - manifest['upstream']['scenes']['paths'].keys():
        directory = local / scene
        directory.mkdir(parents=True)
        (directory / f'{scene}.navmesh').write_bytes(b'not the original navmesh')
    monkeypatch.setattr(prepare, 'require_tools', lambda *args: None)
    monkeypatch.setattr(prepare, 'upstream_file', lambda *args: pytest.fail('Download before navmesh validation'))
    args = SimpleNamespace(source=tmp_path / 'source', output=tmp_path / 'full',
                           download=True, subset='full', scenes=local)
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare.prepare_evaluation(args)
    assert not args.output.exists()


def test_full_evaluation_fetches_original_navmesh_without_local_scene_argument(tmp_path, monkeypatch):
    import huggingface_hub

    manifest = json.loads(prepare.EVALUATION.read_text())
    scene = '0025_840019'
    row = next(r for r in manifest['episodes'] if r['scene_id'] == scene)
    manifest['episodes'] = [row]
    source = tmp_path / 'source'
    folder = source / 'scenes' / scene
    folder.mkdir(parents=True)
    (folder / f'{scene}.gs.ply').write_bytes(b'visual asset')
    (folder / 'labels.json').write_text(json.dumps([{'ins_id': row['target_instance_id'], 'bounding_box': [
        {'x': x, 'y': y, 'z': 0.} for x, y in [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]]}]))
    original = tmp_path / 'original.navmesh'
    original.write_bytes(b'original navigation mesh')
    files = {'gs.ply': folder / f'{scene}.gs.ply', 'labels.json': folder / 'labels.json', 'navmesh': original}
    manifest['assets'] = {scene: {name: {'bytes': path.stat().st_size, 'sha256': prepare.sha256(path)}
                                   for name, path in files.items()}}
    path = tmp_path / 'evaluation.json'
    prepare.write_json(path, manifest)
    monkeypatch.setattr(prepare, 'EVALUATION', path)
    calls = []

    def download(repo_id, filename, **kwargs):
        upstream = manifest['upstream']['navmeshes']
        assert repo_id == upstream['repo_id']
        assert kwargs['revision'] == upstream['revision']
        assert filename == f"{upstream['prefix']}/{scene}.navmesh"
        destination = Path(kwargs['local_dir']) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        calls.append(filename)
        return str(destination)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    args = SimpleNamespace(source=source, output=tmp_path / 'full', download=True, subset='full', scenes=None)
    prepare.prepare_evaluation(args)
    episodes = json.loads((args.output / 'episodes.json').read_text())['episodes']
    assert len(episodes) == 1
    navmesh = Path(episodes[0]['navmesh_path'])
    assert navmesh.read_bytes() == original.read_bytes()
    assert Path(episodes[0]['scene_path']) == (folder / f'{scene}.gs.ply').resolve()
    args.download = False
    prepare.prepare_evaluation(args)
    assert len(calls) == 1
    navmesh.write_bytes(b'corrupt downloaded navigation mesh')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare.prepare_evaluation(args)
