"""Tests for the shared ``ffmpeg_clip`` helper.

``ffmpeg_clip`` is a dependency-light module-level function that shells out
to ``ffmpeg`` to make a short, time-bounded clip of a video. It replaces the
inline DeepLabCut ``VideoWriter.shorten`` call previously used in tutorial 24.
"""

import shutil

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_ffmpeg_clip_builds_expected_command(tmp_path, monkeypatch):
    """The helper builds a time-based ffmpeg command and returns the path."""
    import importlib

    mv = importlib.import_module("spyglass.position.utils.make_video")

    src = tmp_path / "cam1_reach.mp4"
    src.touch()
    dest = tmp_path / "clips"

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # ffmpeg would create the output; emulate that so the helper's
        # existence check passes.
        (dest / f"{src.stem}short{src.suffix}").touch()

        class _Ret:
            returncode = 0
            stderr = ""

        return _Ret()

    monkeypatch.setattr(mv.subprocess, "run", fake_run)

    out = mv.ffmpeg_clip(src, dest)

    # Output name preserves the source stem tokens (e.g. "cam1").
    assert out == dest / "cam1_reachshort.mp4"
    assert "cam1" in out.name

    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    # Non-overwrite default uses -n and time-based -ss/-to selection.
    assert "-n" in cmd
    assert cmd[cmd.index("-ss") + 1] == "00:00:00"
    assert cmd[cmd.index("-to") + 1] == "00:00:01"
    assert cmd[cmd.index("-i") + 1] == str(src)
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert str(out) in cmd


def test_ffmpeg_clip_custom_bounds_and_overwrite(tmp_path, monkeypatch):
    """Custom start/end/suffix and overwrite flag are honored."""
    import importlib

    mv = importlib.import_module("spyglass.position.utils.make_video")

    src = tmp_path / "vid.mov"
    src.touch()
    dest = tmp_path / "clips"

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        (dest / f"{src.stem}mini{src.suffix}").touch()

        class _Ret:
            returncode = 0
            stderr = ""

        return _Ret()

    monkeypatch.setattr(mv.subprocess, "run", fake_run)

    out = mv.ffmpeg_clip(
        src,
        dest,
        start="00:00:02",
        end="00:00:05",
        suffix="mini",
        overwrite=True,
    )

    assert out == dest / "vidmini.mov"
    cmd = captured["cmd"]
    assert "-y" in cmd and "-n" not in cmd
    assert cmd[cmd.index("-ss") + 1] == "00:00:02"
    assert cmd[cmd.index("-to") + 1] == "00:00:05"


def test_ffmpeg_clip_raises_on_failure(tmp_path, monkeypatch):
    """A non-zero ffmpeg exit raises RuntimeError."""
    import importlib

    mv = importlib.import_module("spyglass.position.utils.make_video")

    src = tmp_path / "vid.mp4"
    src.touch()

    def fake_run(cmd, **kwargs):
        class _Ret:
            returncode = 1
            stderr = "boom"

        return _Ret()

    monkeypatch.setattr(mv.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        mv.ffmpeg_clip(src, tmp_path / "clips")


def test_ffmpeg_clip_skips_existing(tmp_path, monkeypatch):
    """Without overwrite, a pre-existing clip is returned without ffmpeg."""
    import importlib

    mv = importlib.import_module("spyglass.position.utils.make_video")

    src = tmp_path / "cam2_x.mp4"
    src.touch()
    dest = tmp_path / "clips"
    dest.mkdir()
    existing = dest / "cam2_xshort.mp4"
    existing.touch()

    def fail_run(cmd, **kwargs):  # pragma: no cover
        raise AssertionError("ffmpeg should not be invoked for existing clip")

    monkeypatch.setattr(mv.subprocess, "run", fail_run)

    out = mv.ffmpeg_clip(src, dest)
    assert out == existing


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_ffmpeg_clip_real_clip(tmp_path):
    """End-to-end: generate a tiny video and clip one second from it."""
    import importlib
    import subprocess

    mv = importlib.import_module("spyglass.position.utils.make_video")

    src = tmp_path / "cam1_src.mp4"
    # 2-second synthetic test pattern at 10 fps.
    gen = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x48:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(src),
            "-hide_banner",
            "-loglevel",
            "error",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert gen.returncode == 0, gen.stderr

    out = mv.ffmpeg_clip(src, tmp_path / "clips", end="00:00:01")
    assert out.exists()
    assert out.stat().st_size > 0
    assert "cam1" in out.name
