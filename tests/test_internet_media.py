import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_media', ROOT / 'scripts/prepare_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def media_release(tmp_path, monkeypatch):
    if not all(shutil.which(name) for name in ('ffmpeg', 'ffprobe')):
        pytest.skip('FFmpeg required')
    annotations = tmp_path / 'annotations'
    annotations.mkdir()
    for name in ('LICENSE', 'SPATIALVID_NOTICE.txt'):
        shutil.copyfile(ROOT / 'annotations' / name, annotations / name)
    rows = [{'clip': clip, 'segment_id': clip, 'split': split, 't0': 0, 't_end': 9, 'fps': 4,
             'instructions': {'goto': 'Go to the chair.', 'diverse_iv': 'Approach the chair in your initial view.'},
             'archive': 'videos/group_0001.tar.gz', 'source_video': f'videos/group_0001/{clip}.mp4'}
            for clip, split in [('a', 'train'), ('b', 'train'), ('c', 'val')]]
    label_path = annotations / 'internet.jsonl.gz'
    label_path.write_bytes(gzip.compress(''.join(json.dumps(r) + '\n' for r in rows).encode()))
    prepare.write_json(annotations / 'manifest.json', {
        'files': {'internet.jsonl.gz': {'sha256': prepare.sha256(label_path)}},
        'internet': {'repo_id': 'upstream/SpatialVID', 'revision': 'original'},
        'subsets': {'smoke': {'internet': ['a']}}})
    video = tmp_path / 'sample.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=512x288:rate=4', '-frames:v', '12', '-c:v', 'libx264',
                    '-threads', '1', str(video)], check=True)
    source = tmp_path / 'cache'
    folder = source / 'internet'
    folder.mkdir(parents=True)
    files = []
    for name, clips in [('train.tar', ['a', 'b']), ('val.tar', ['c']), ('train-smoke.tar', ['a'])]:
        path = folder / name
        with tarfile.open(path, 'w') as archive:
            for clip in clips:
                archive.add(video, arcname=f'videos/{clip}.mp4')
        files.append({'path': 'internet/' + name, 'sha256': prepare.sha256(path), 'bytes': path.stat().st_size})
    index = [dict(clip=r['clip'], split=r['split'], archive=r['archive'], source_video=r['source_video'],
                  num_frames=12, width=512, height=288, fps=4,
                  bytes=video.stat().st_size, sha256=prepare.sha256(video),
                  file=f'internet/{r["split"]}.tar') for r in rows]
    path = folder / 'videos.jsonl.gz'
    path.write_bytes(gzip.compress(''.join(json.dumps(r) + '\n' for r in index).encode()))
    manifest = {'repo_id': 'ScaleNav/Data', 'revision': 'fixed-release',
                'annotations_sha256': prepare.sha256(label_path), 'files': files,
                'index': {'path': 'internet/videos.jsonl.gz', 'bytes': path.stat().st_size,
                          'sha256': prepare.sha256(path)}, 'smoke_file': 'internet/train-smoke.tar'}
    media = annotations / 'internet-media.json'
    prepare.write_json(media, manifest)
    monkeypatch.setattr(prepare, 'ANNOTATIONS', annotations)
    monkeypatch.setattr(prepare, 'INTERNET_MEDIA', media)
    args = SimpleNamespace(source=source, output=tmp_path / 'output', download=False,
                           instructions='diverse_iv', split='all', limit=None, subset='full', from_upstream=False)
    calls = []
    real_file = prepare.upstream_file

    def track(repo, revision, filename, source, download):
        assert repo == 'ScaleNav/Data' and revision == 'fixed-release'
        calls.append(filename)
        return real_file(repo, revision, filename, source, download)

    monkeypatch.setattr(prepare, 'upstream_file', track)
    monkeypatch.setattr(prepare, 'prepare_upstream_internet', lambda *args: pytest.fail('Original shards requested'))
    return args, calls


def test_small_subset_uses_only_small_archive_and_preserves_instructions(media_release):
    args, calls = media_release
    args.subset, args.split = 'smoke', 'train'
    prepare.prepare_internet(args)
    assert calls == ['internet/videos.jsonl.gz', 'internet/train-smoke.tar']
    assert sorted(p.name for p in (args.output / 'videos').iterdir()) == ['a.mp4']
    assert not (args.output / 'val').exists()
    labels = json.loads((args.output / 'train/a.json').read_text())
    assert labels[0]['instruction'] == 'Approach the chair in your initial view.'
    assert (labels[0]['t0'], labels[0]['t_end']) == (0, 9)
    assert 'SpatialVID-HQ' in (args.output / 'SOURCE.txt').read_text()
    calls.clear()
    prepare.prepare_internet(args)
    assert calls == ['internet/videos.jsonl.gz']


def test_validation_download_is_independent_and_corrupt_video_is_repaired(media_release):
    args, calls = media_release
    args.split = 'val'
    prepare.prepare_internet(args)
    assert calls == ['internet/videos.jsonl.gz', 'internet/val.tar']
    path = args.output / 'videos/c.mp4'
    expected = prepare.sha256(path)
    path.write_bytes(b'corrupt')
    calls.clear()
    prepare.prepare_internet(args)
    assert prepare.sha256(path) == expected
    assert calls == ['internet/videos.jsonl.gz', 'internet/val.tar']


@pytest.mark.parametrize('corruption', ['archive', 'index', 'annotations'])
def test_media_mismatch_fails_without_upstream_fallback(media_release, corruption):
    args, _ = media_release
    args.split = 'val'
    if corruption == 'annotations':
        path = prepare.INTERNET_MEDIA
        manifest = json.loads(path.read_text())
        manifest['annotations_sha256'] = 'wrong'
        prepare.write_json(path, manifest)
    else:
        path = args.source / 'internet' / ('val.tar' if corruption == 'archive' else 'videos.jsonl.gz')
        path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch|different releases'):
        prepare.prepare_internet(args)
    assert not list((args.output / 'videos').glob('*.mp4'))


def test_media_extractor_rejects_links_and_ignores_unrequested_paths(tmp_path):
    archive_path = tmp_path / 'unsafe.tar'
    with tarfile.open(archive_path, 'w') as archive:
        member = tarfile.TarInfo('../outside')
        member.size = 4
        archive.addfile(member, io.BytesIO(b'data'))
        member = tarfile.TarInfo('videos/a.mp4')
        member.type, member.linkname = tarfile.SYMTYPE, '../outside'
        archive.addfile(member)
    videos = tmp_path / 'videos'
    videos.mkdir()
    with pytest.raises(ValueError, match='Invalid media member'):
        prepare.extract_media(archive_path, {'videos/a.mp4': {'bytes': 4, 'sha256': 'unused'}}, videos)
    assert not (tmp_path.parent / 'outside').exists()
