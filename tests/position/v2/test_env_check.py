"""Unit tests for the Position V2 pre-flight environment check.

Pure: no database, no real package imports (metadata lookups are monkeypatched).
"""

import pytest


def _fake_installed(versions):
    """Return a stand-in for env_check._installed backed by a dict."""

    def _installed(name):
        return versions.get(name)

    return _installed


def _patch(monkeypatch, versions, ffmpeg_present=True):
    from spyglass.position.v2 import env_check

    monkeypatch.setattr(env_check, "_installed", _fake_installed(versions))
    monkeypatch.setattr(
        env_check.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if ffmpeg_present else None,
    )
    return env_check


class TestCheckEnvironment:
    def test_clean_env_reports_no_problems(self, monkeypatch):
        env_check = _patch(
            monkeypatch, {"jax": "0.5.0", "torch": "2.10.0", "deeplabcut": "3"}
        )
        assert env_check.check_environment(verbose=False) == []

    def test_tensorflow_plus_jax_flagged(self, monkeypatch):
        env_check = _patch(
            monkeypatch,
            {
                "tensorflow": "2.15.1",
                "jax": "0.4.18",
                "keras": "2.15.0",
                "torch": "2.10.0",
            },
        )
        problems = env_check.check_environment(verbose=False)
        assert len(problems) == 1
        msg = problems[0]
        assert "TensorFlow" in msg and "jax" in msg
        # names concrete packages to uninstall (only those present)
        assert "pip uninstall" in msg and "tensorflow" in msg
        assert "keras" in msg
        # does not name a TF package that isn't installed
        assert "tensorpack" not in msg

    def test_jax_only_is_clean(self, monkeypatch):
        """The correct V2 state (jax, no TensorFlow) is not flagged."""
        env_check = _patch(monkeypatch, {"jax": "0.5.0", "torch": "2.10.0"})
        assert env_check.check_environment(verbose=False) == []

    def test_pose_tool_without_torch_flagged(self, monkeypatch):
        env_check = _patch(monkeypatch, {"deeplabcut": "3.0.0"})
        problems = env_check.check_environment(verbose=False)
        # Only the missing-PyTorch problem fires (no tf+jax conflict here).
        assert len(problems) == 1 and "PyTorch" in problems[0]

    def test_missing_ffmpeg_flagged(self, monkeypatch):
        env_check = _patch(
            monkeypatch,
            {"jax": "0.5.0", "torch": "2.10.0"},
            ffmpeg_present=False,
        )
        problems = env_check.check_environment(verbose=False)
        assert len(problems) == 1 and "ffmpeg" in problems[0]

    def test_raise_on_error(self, monkeypatch):
        env_check = _patch(
            monkeypatch, {"tensorflow": "2.15.1", "jax": "0.4.18"}
        )
        with pytest.raises(RuntimeError, match="TensorFlow"):
            env_check.check_environment(raise_on_error=True, verbose=False)
