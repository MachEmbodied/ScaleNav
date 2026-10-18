from pathlib import Path

import awm.utils.misc as misc


def test_resolve_packaged_ffmpeg_without_system_binary(monkeypatch, tmp_path):
    import imageio_ffmpeg
    monkeypatch.delenv('FFMPEG_BIN', raising=False)
    monkeypatch.setattr(misc.shutil, 'which', lambda _: None)
    monkeypatch.setattr(misc.sys, 'executable', str(tmp_path / 'python'))
    binary = misc._resolve_ffmpeg()
    assert binary == imageio_ffmpeg.get_ffmpeg_exe()
    result = misc.subprocess.run([binary, '-version'], capture_output=True, text=True, check=True)
    assert result.stdout.startswith('ffmpeg version')


def test_resolve_ffmpeg_next_to_active_python(monkeypatch, tmp_path):
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    ffmpeg = bin_dir / "ffmpeg"
    python.touch()
    ffmpeg.touch(mode=0o755)

    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr(misc.shutil, "which", lambda _: None)
    monkeypatch.setattr(misc.sys, "executable", str(python))

    assert misc._resolve_ffmpeg() == str(ffmpeg)


def test_reencode_h264_writes_browser_compatible_mp4(monkeypatch, tmp_path):
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"mpeg4")
    command = []

    def fake_run(args, **kwargs):
        command.extend(args)
        Path(args[-1]).write_bytes(b"h264")

    monkeypatch.setattr(misc, "_resolve_ffmpeg", lambda: "/env/bin/ffmpeg")
    monkeypatch.setattr(misc.subprocess, "run", fake_run)

    misc.reencode_h264(str(video))

    assert video.read_bytes() == b"h264"
    assert command[:2] == ["/env/bin/ffmpeg", "-nostdin"]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-movflags") + 1] == "+faststart"
