"""Video encoding and model argument utilities."""

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type



def _resolve_ffmpeg() -> Optional[str]:
    """Find ffmpeg even when a launcher supplies Python without its PATH."""
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    candidate = Path(sys.executable).with_name("ffmpeg")
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    return None


def reencode_h264(path: str) -> None:
    """Re-encode a video file to H.264/yuv420p for broad compatibility.

    Replaces the file in-place using system FFmpeg or imageio's bundled binary.
    """
    ffmpeg_bin = _resolve_ffmpeg()
    if ffmpeg_bin is None:
        print(
            f"[reencode_h264] ffmpeg not found — '{path}' kept as-is "
            "(may not play in VSCode / PowerPoint)."
        )
        return

    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                path,
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp, path)
    except FileNotFoundError:
        print(
            f"[reencode_h264] ffmpeg not found — '{path}' kept as-is "
            "(may not play in VSCode / PowerPoint)."
        )
        if os.path.exists(tmp):
            os.remove(tmp)
    except subprocess.CalledProcessError as e:
        print(f"[reencode_h264] ffmpeg failed for '{path}': {e.stderr.decode()}")
        if os.path.exists(tmp):
            os.remove(tmp)


def filter_kwargs(cls: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep arguments accepted by the model constructor."""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}
