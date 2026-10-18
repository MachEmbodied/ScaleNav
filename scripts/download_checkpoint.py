"""Download and verify the ScaleNav model weights."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import sys

HF_REPOSITORY = "Yangyihui/ScaleNav"
MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "checkpoint-download-manifest.json"
BLOCK = 8 * 1024 * 1024


def digest(stream, size):
    result = hashlib.sha256()
    remaining = size
    while remaining:
        data = stream.read(min(BLOCK, remaining))
        if not data:
            raise ValueError("Unexpected end of file")
        result.update(data)
        remaining -= len(data)
    return result.hexdigest()


def verified_existing(entry, output):
    target = output / entry["path"]
    if not target.exists() and not target.is_symlink():
        return False
    with target.open("rb") as stream:
        valid = target.stat().st_size == entry["size"] and digest(stream, entry["size"]) == entry["sha256"]
    if not valid:
        raise ValueError(f"Existing file differs from the release: {target}; move it aside first")
    print(f"Verified existing {entry['path']}", flush=True)
    return True


def move_without_overwrite(source, target):
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is not None:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
            return
        error = ctypes.get_errno()
        if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EXDEV}:
            raise OSError(error, os.strerror(error), str(target))
    created = None
    try:
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            status = os.fstat(outgoing.fileno())
            created = (status.st_dev, status.st_ino)
            shutil.copyfileobj(incoming, outgoing, BLOCK)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except BaseException:
        if created is not None and target.exists():
            status = target.stat()
            if (status.st_dev, status.st_ino) == created:
                target.unlink()
        raise
    source.unlink()


def install_verified_file(entry, partial, output):
    target = output / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        move_without_overwrite(partial, target)
    except FileExistsError:
        verified_existing(entry, output)
        partial.unlink()
    print(f"Verified {entry['path']}", flush=True)


def verified_huggingface_staging(entry, staging):
    relative = Path(entry["path"])
    candidates = [staging / relative]
    cache = staging / ".cache" / "huggingface" / "download" / relative.parent
    candidates.extend(sorted(cache.glob("*.incomplete")))
    for path in candidates:
        if not path.is_file() or path.stat().st_size != entry["size"]:
            continue
        with path.open("rb") as stream:
            if digest(stream, entry["size"]) == entry["sha256"]:
                print(f"Verified staged {entry['path']}", flush=True)
                return path
    return None


def download_huggingface_file(entry, output, revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    if verified_existing(entry, output):
        return
    staging = output / ".huggingface"
    partial = verified_huggingface_staging(entry, staging)
    if partial is not None:
        install_verified_file(entry, partial, output)
        return
    parent = (staging / entry["path"]).parent
    parent.mkdir(parents=True, exist_ok=True)
    (parent / ".keep").touch(exist_ok=True)
    try:
        partial = Path(hf_hub_download(
            repo_id=HF_REPOSITORY, revision=revision, filename=entry["path"],
            local_dir=staging, endpoint="https://huggingface.co",
        ))
    except FileNotFoundError:
        partial = verified_huggingface_staging(entry, staging)
        if partial is None:
            raise
        install_verified_file(entry, partial, output)
        return
    except HfHubHTTPError as exc:
        if isinstance(exc, RepositoryNotFoundError) or (exc.response is not None and exc.response.status_code in (401, 403)):
            raise RuntimeError("Hugging Face access denied. Run `hf auth login` with an authorized account") from exc
        raise RuntimeError(f"Hugging Face download failed: {entry['path']}: {exc}") from exc
    with partial.open("rb") as stream:
        valid = partial.stat().st_size == entry["size"] and digest(stream, entry["size"]) == entry["sha256"]
    if not valid:
        partial.unlink()
        raise ValueError(f"Checksum mismatch: {entry['path']}")
    install_verified_file(entry, partial, output)


def download_checkpoint(manifest, output):
    from filelock import FileLock

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".download.lock")):
        _download_checkpoint(manifest, output)


def _download_checkpoint(manifest, output):
    paths = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative == Path(".") or entry["path"] in paths:
            raise ValueError("Invalid path in checkpoint manifest")
        paths.add(entry["path"])
        if (not isinstance(entry["size"], int) or entry["size"] <= 0
                or not isinstance(entry["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("Invalid file metadata in checkpoint manifest")
    pending = [entry for entry in manifest["files"] if not verified_existing(entry, output)]
    if pending:
        hub = manifest.get("huggingface", {})
        revision = hub.get("revision")
        if hub.get("repository") != HF_REPOSITORY or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Checkpoint manifest requires a pinned Hugging Face revision")
        for entry in pending:
            download_huggingface_file(entry, output, revision)
    print(f"Checkpoint ready: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/scalenav"))
    args = parser.parse_args()
    download_checkpoint(json.loads(MANIFEST.read_text()), args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
