import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("checkpoint_download", Path(__file__).resolve().parents[1] / "scripts/download_checkpoint.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def recipe(payloads):
    full = b"".join(payloads)
    return {"path": "transformer/weights.bin", "size": len(full),
            "sha256": hashlib.sha256(full).hexdigest()}


def test_mismatching_existing_file_is_preserved(tmp_path):
    entry = recipe([b"right"])
    target = tmp_path / entry["path"]
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Existing file differs"):
        download.download_huggingface_file(entry, tmp_path, "a" * 40)
    assert target.read_bytes() == b"wrong"


def manifest_for(payloads):
    import json
    paths = [entry["path"] for entry in json.loads(download.MANIFEST.read_text())["files"]]
    entries = []
    for path, payload in zip(paths, payloads):
        entry = recipe([payload])
        entry["path"] = path
        entries.append(entry)
    return {"huggingface": {"repository": download.HF_REPOSITORY, "revision": "a" * 40},
            "files": entries}


def hub_transport(monkeypatch, payloads, calls):
    import huggingface_hub
    def fetch(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[kwargs["filename"]])
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)


def test_huggingface_default_downloads_only_manifest_files_and_reuses_offline(tmp_path, monkeypatch):
    payloads = [b"ar", b"meta", b"config", b"weights", b"checksums"]
    manifest = manifest_for(payloads)
    expected = {entry["path"]: body for entry, body in zip(manifest["files"], payloads)}
    calls = []
    hub_transport(monkeypatch, expected, calls)
    download.download_checkpoint(manifest, tmp_path)
    assert {call["filename"] for call in calls} == set(expected)
    assert len(calls) == 5
    assert all(call["repo_id"] == download.HF_REPOSITORY and call["revision"] == "a" * 40 for call in calls)
    assert all(Path(call["local_dir"]).is_relative_to(tmp_path) for call in calls)
    for name, body in expected.items():
        assert (tmp_path / name).read_bytes() == body
        assert not (tmp_path / ".huggingface" / name).exists()
    download.download_checkpoint(manifest, tmp_path)
    assert len(calls) == 5


def test_huggingface_preflight_preserves_mismatch_before_any_download(tmp_path, monkeypatch):
    manifest = manifest_for([b"first", b"right"])
    target = tmp_path / manifest["files"][1]["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"wrong")
    calls = []
    hub_transport(monkeypatch, {}, calls)
    with pytest.raises(ValueError, match="Existing file differs"):
        download.download_checkpoint(manifest, tmp_path)
    assert target.read_bytes() == b"wrong"
    assert not (tmp_path / manifest["files"][0]["path"]).exists()
    assert not calls


def test_huggingface_corrupt_download_is_not_installed_and_can_retry(tmp_path, monkeypatch):
    manifest = manifest_for([b"right"])
    name = manifest["files"][0]["path"]
    calls = []
    payloads = {name: b"wrong"}
    hub_transport(monkeypatch, payloads, calls)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download.download_checkpoint(manifest, tmp_path)
    assert not (tmp_path / name).exists()
    assert not (tmp_path / ".huggingface" / name).exists()
    payloads[name] = b"right"
    download.download_checkpoint(manifest, tmp_path)
    assert (tmp_path / name).read_bytes() == b"right"


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_huggingface_requires_immutable_revision(tmp_path, monkeypatch, revision):
    manifest = manifest_for([b"weights"])
    manifest["huggingface"]["revision"] = revision
    calls = []
    hub_transport(monkeypatch, {}, calls)
    with pytest.raises(ValueError, match="pinned Hugging Face revision"):
        download.download_checkpoint(manifest, tmp_path)
    assert not calls


@pytest.mark.parametrize('field, value', [
    ('path', '../weights'), ('path', '/weights'), ('path', '.'),
    ('size', 0), ('size', -1), ('sha256', 'invalid'),
])
def test_invalid_manifest_fails_before_download(tmp_path, monkeypatch, field, value):
    manifest = manifest_for([b'weights'])
    manifest['files'][0][field] = value
    calls = []
    hub_transport(monkeypatch, {}, calls)
    with pytest.raises(ValueError, match='Invalid'):
        download.download_checkpoint(manifest, tmp_path)
    assert not calls


@pytest.mark.parametrize("status", [401, 403, 404])
def test_huggingface_access_error_names_correct_login(tmp_path, monkeypatch, status):
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    from requests import Response
    response = Response()
    response.status_code = status
    def denied(**kwargs):
        error = RepositoryNotFoundError if status == 404 else HfHubHTTPError
        raise error("Access denied", response=response)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", denied)
    with pytest.raises(RuntimeError, match="hf auth login"):
        download.download_checkpoint(manifest_for([b"weights"]), tmp_path)


def test_huggingface_never_overwrites_file_created_during_download(tmp_path, monkeypatch):
    import huggingface_hub
    manifest = manifest_for([b"right"])
    name = manifest["files"][0]["path"]
    def concurrent_write(**kwargs):
        path = Path(kwargs["local_dir"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"right")
        (tmp_path / name).write_bytes(b"other")
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", concurrent_write)
    with pytest.raises(ValueError, match="Existing file differs"):
        download.download_checkpoint(manifest, tmp_path)
    assert (tmp_path / name).read_bytes() == b"other"


def test_huggingface_recovers_complete_cache_with_missing_staging_parent(tmp_path, monkeypatch):
    entry = recipe([b"weights"])
    cached = tmp_path / ".huggingface/.cache/huggingface/download/transformer/hash.incomplete"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"weights")
    calls = []
    hub_transport(monkeypatch, {}, calls)
    download.download_huggingface_file(entry, tmp_path, "a" * 40)
    assert (tmp_path / entry["path"]).read_bytes() == b"weights"
    assert not cached.exists()
    assert not calls


def test_huggingface_recovers_if_parent_disappears_after_download(tmp_path, monkeypatch):
    import huggingface_hub
    import shutil
    entry = recipe([b"weights"])
    def lost_directory(**kwargs):
        staging = Path(kwargs["local_dir"])
        cached = staging / ".cache/huggingface/download/transformer/hash.incomplete"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"weights")
        shutil.rmtree((staging / kwargs["filename"]).parent)
        raise FileNotFoundError(str(staging / kwargs["filename"]))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lost_directory)
    download.download_huggingface_file(entry, tmp_path, "a" * 40)
    assert (tmp_path / entry["path"]).read_bytes() == b"weights"


def disable_no_replace_rename(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(download.ctypes, "CDLL", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(download.os, "link", lambda *a, **k: pytest.fail("Hard links must not be required"))


def test_install_falls_back_to_exclusive_copy_without_hardlinks(tmp_path, monkeypatch):
    disable_no_replace_rename(monkeypatch)
    entry = recipe([b"weights"])
    partial = tmp_path / "verified.download"
    partial.write_bytes(b"weights")
    download.install_verified_file(entry, partial, tmp_path)
    assert (tmp_path / entry["path"]).read_bytes() == b"weights"
    assert not partial.exists()


def test_exclusive_copy_preserves_existing_file_and_staged_data(tmp_path, monkeypatch):
    disable_no_replace_rename(monkeypatch)
    entry = recipe([b"weights"])
    partial = tmp_path / "verified.download"
    partial.write_bytes(b"weights")
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="Existing file differs"):
        download.install_verified_file(entry, partial, tmp_path)
    assert target.read_bytes() == b"existing"
    assert partial.read_bytes() == b"weights"


def test_interrupted_copy_preserves_verified_staging_for_retry(tmp_path, monkeypatch):
    disable_no_replace_rename(monkeypatch)
    entry = recipe([b"weights"])
    partial = tmp_path / "verified.download"
    partial.write_bytes(b"weights")
    def interrupted(source, target, block):
        target.write(source.read(2))
        raise OSError("Disconnected filesystem")
    monkeypatch.setattr(download.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="Disconnected filesystem"):
        download.install_verified_file(entry, partial, tmp_path)
    assert not (tmp_path / entry["path"]).exists()
    assert partial.read_bytes() == b"weights"


def test_huggingface_does_not_install_corrupt_cache_after_failed_finalize(tmp_path, monkeypatch):
    import huggingface_hub
    entry = recipe([b"weights"])
    cached = tmp_path / ".huggingface/.cache/huggingface/download/transformer/hash.incomplete"
    def failed_finalize(**kwargs):
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt")
        raise FileNotFoundError(str(Path(kwargs["local_dir"]) / kwargs["filename"]))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failed_finalize)
    with pytest.raises(FileNotFoundError):
        download.download_huggingface_file(entry, tmp_path, "a" * 40)
    assert not (tmp_path / entry["path"]).exists()
    assert cached.read_bytes() == b"corrupt"


def test_two_processes_share_output_without_racing_staged_payload(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import time

    payload = b"verified checkpoint"
    manifest = manifest_for([payload])
    output = tmp_path / "checkpoint"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    worker = r"""
import importlib.util
import json
from pathlib import Path
import sys
import time
import huggingface_hub

script, root, role = sys.argv[1:]
root = Path(root)
spec = importlib.util.spec_from_file_location('download', script)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
manifest = json.loads((root / 'manifest.json').read_text())
def fetch(**kwargs):
    with (root / 'fetches').open('a') as log:
        log.write(role + '\n')
    staged = Path(kwargs['local_dir']) / kwargs['filename']
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b'verified checkpoint')
    if role == 'first':
        (root / 'first-ready').touch()
        deadline = time.monotonic() + 20
        while not (root / 'release-first').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for test release')
            time.sleep(0.01)
    return str(staged)
huggingface_hub.hf_hub_download = fetch
(root / (role + '-started')).touch()
download.download_checkpoint(manifest, root / 'checkpoint')
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    processes = []
    def start(role):
        process = subprocess.Popen([sys.executable, "-c", worker, str(download.MANIFEST.parent.parent / "scripts/download_checkpoint.py"),
                                    str(tmp_path), role], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=environment)
        processes.append(process)
        return process
    def wait_for(path):
        deadline = time.monotonic() + 10
        while not path.exists():
            assert all(process.poll() is None for process in processes)
            assert time.monotonic() < deadline, f"Timed out waiting for {path.name}"
            time.sleep(0.01)
    try:
        start("first")
        wait_for(tmp_path / "first-ready")
        second = start("second")
        wait_for(tmp_path / "second-started")
        time.sleep(0.15)
        assert second.poll() is None
        assert not (output / manifest["files"][0]["path"]).exists()
        (tmp_path / "release-first").touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert (output / manifest["files"][0]["path"]).read_bytes() == payload
    assert (tmp_path / "fetches").read_text().splitlines() == ["first"]
