"""Unit tests for the Position V2 pre-flight environment check.

Pure: no database, no real package imports (metadata lookups are monkeypatched).
"""

import pytest


def _fake_installed(versions):
    """Return a stand-in for env_check._installed backed by a dict."""

    def _installed(name):
        return versions.get(name)

    return _installed


def _patch(
    monkeypatch,
    versions,
    ffmpeg_present=True,
    blas_error=None,
    qt_problem=None,
):
    from spyglass.position.v2 import env_check

    monkeypatch.setattr(env_check, "_installed", _fake_installed(versions))
    monkeypatch.setattr(
        env_check.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if ffmpeg_present else None,
    )
    monkeypatch.setattr(env_check, "_blas_lapack_error", lambda: blas_error)
    # Pinned so a result never depends on the session running the suite: the
    # real probe reads WAYLAND_DISPLAY and the installed Qt plugins.
    monkeypatch.setattr(env_check, "_qt_platform_problem", lambda: qt_problem)
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

    def test_broken_blas_flagged(self, monkeypatch):
        env_check = _patch(
            monkeypatch,
            {"jax": "0.5.0", "torch": "2.10.0"},
            blas_error="liwork=1",
        )
        problems = env_check.check_environment(verbose=False)
        assert len(problems) == 1
        assert "BLAS/LAPACK" in problems[0] and "liwork=1" in problems[0]

    def test_raise_on_error(self, monkeypatch):
        env_check = _patch(
            monkeypatch, {"tensorflow": "2.15.1", "jax": "0.4.18"}
        )
        with pytest.raises(RuntimeError, match="TensorFlow"):
            env_check.check_environment(raise_on_error=True, verbose=False)


class TestQtPlatformProbe:
    """The Wayland platform-plugin probe, driven by env vars only."""

    @staticmethod
    def _probe(monkeypatch, *, wayland, display, requested, plugin):
        from spyglass.position.v2 import env_check

        for name, value in (
            ("WAYLAND_DISPLAY", wayland),
            ("DISPLAY", display),
            ("QT_QPA_PLATFORM", requested),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        monkeypatch.setattr(
            env_check,
            "_qt_plugin_dirs",
            lambda: [_FakeDir(["libqwayland.so"])] if plugin else [],
        )
        return env_check._qt_platform_problem()

    def test_headless_is_not_a_problem(self, monkeypatch):
        """A compute node has no display and needs none."""
        assert (
            self._probe(
                monkeypatch,
                wayland=None,
                display=None,
                requested=None,
                plugin=False,
            )
            is None
        )

    def test_x11_session_is_not_a_problem(self, monkeypatch):
        """The xcb plugin ships with Qt, so an X11 session is fine."""
        assert (
            self._probe(
                monkeypatch,
                wayland=None,
                display=":0",
                requested=None,
                plugin=False,
            )
            is None
        )

    def test_explicit_platform_is_honoured(self, monkeypatch):
        """A caller naming xcb is not second-guessed."""
        assert (
            self._probe(
                monkeypatch,
                wayland="wayland-0",
                display=":0",
                requested="xcb",
                plugin=False,
            )
            is None
        )

    def test_wayland_with_plugin_is_not_a_problem(self, monkeypatch):
        assert (
            self._probe(
                monkeypatch,
                wayland="wayland-0",
                display=None,
                requested=None,
                plugin=True,
            )
            is None
        )

    def test_present_dir_without_the_plugin_reports(self, monkeypatch):
        """The case the probe could never report.

        A directory that exists but holds no `libqwayland*.so` is every
        machine with Qt and no Wayland plugin, which is the reported bug.
        """
        from spyglass.position.v2 import env_check

        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.setattr(
            env_check,
            "_qt_plugin_dirs",
            lambda: [_FakeDir(["libqxcb.so", "libqminimal.so"])],
        )
        msg = env_check._qt_platform_problem()
        assert msg is not None, "a dir with no wayland plugin must report"
        assert "QT_QPA_PLATFORM=xcb" in msg

    def test_wayland_without_plugin_reports_both_fixes(self, monkeypatch):
        """The one failing case names the env var and the package."""
        msg = self._probe(
            monkeypatch,
            wayland="wayland-0",
            display=None,
            requested=None,
            plugin=False,
        )
        assert msg is not None
        assert "QT_QPA_PLATFORM=xcb" in msg
        assert "conda install -c conda-forge qt-wayland" in msg
        assert 'Qt platform plugin "wayland"' in msg

    def test_problem_reaches_check_environment(self, monkeypatch):
        """The probe's result is one of check_environment's problems."""
        env_check = _patch(
            monkeypatch, {"torch": "2.10.0"}, qt_problem="wayland is unhappy"
        )
        problems = env_check.check_environment(verbose=False)
        assert problems == ["wayland is unhappy"]


class _FakeDir:
    """Stands in for a plugin directory, answering `glob`.

    `glob` returns an iterator, as `Path.glob` does. Returning a list made
    the empty case falsy, which hid that the probe was testing generator
    objects rather than their contents.
    """

    def __init__(self, names):
        self._names = names

    def glob(self, pattern):
        import fnmatch

        return iter(n for n in self._names if fnmatch.fnmatch(n, pattern))

    def __str__(self):
        return "/fake/plugins/platforms"
