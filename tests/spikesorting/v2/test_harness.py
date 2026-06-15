"""Regression tests for pytest-harness friction in the shared test config.

These exercise the *harness itself* (``tests/conftest.py`` teardown, the shared
``DataDownloader``), not any Spyglass pipeline, so they run DB-free. Run with
``--no-docker`` to skip the MySQL container the root ``pytest_configure`` would
otherwise build at session start.
"""

from __future__ import annotations

import subprocess
import sys


def test_unconfigure_tolerates_unbound_server():
    """``pytest_unconfigure`` must not raise when ``SERVER`` is unset.

    ``pytest_configure`` sets ``TEARDOWN`` before it binds ``SERVER`` (the latter
    only after building the Docker MySQL manager). If configure raises in between
    -- e.g. Docker is unavailable -- pytest still runs ``pytest_unconfigure`` from
    ``wrap_session``'s ``finally``. Pre-guard, teardown's ``SERVER.stop()`` then
    raised a second traceback (``NameError`` in production, where the name was
    never bound; ``AttributeError`` here, where we bind the module default
    ``None``) that buried the real configuration error. The module-level
    ``SERVER = None`` default plus the ``SERVER is not None`` teardown guard make
    the real error surface instead.
    """
    import tests.conftest as root_conftest

    saved = {
        name: getattr(root_conftest, name, _UNSET)
        for name in ("TEARDOWN", "SERVER", "TMP_BASE_DIR")
    }
    try:
        # Simulate a configure that set TEARDOWN but bailed before binding a real
        # SERVER, so SERVER holds its module-level default of None.
        root_conftest.TEARDOWN = True
        root_conftest.SERVER = None
        root_conftest.TMP_BASE_DIR = None

        # Must not raise. Pre-fix this raised AttributeError on ``None.stop()``;
        # in production the unbound name raised NameError.
        root_conftest.pytest_unconfigure(_DummyConfig())
    finally:
        for name, value in saved.items():
            if value is _UNSET:
                if hasattr(root_conftest, name):
                    delattr(root_conftest, name)
            else:
                setattr(root_conftest, name, value)


class _DummyConfig:
    """``pytest_unconfigure`` ignores its ``config`` argument."""


_UNSET = object()


class _FakePopen:
    """Minimal stand-in for ``subprocess.Popen`` used by ``wait_for``."""

    stdout = None
    stderr = None

    def poll(self):
        return 0  # finished successfully, immediately


def test_data_downloader_no_download_on_construction(tmp_path, monkeypatch):
    """Constructing the shared ``DataDownloader`` must not launch a download.

    The root ``pytest_configure`` builds a ``DataDownloader`` against a fresh,
    empty temp ``base_dir`` on every session -- including pure-helper runs that
    touch no DB and no sample data. Eagerly resolving ``file_downloads`` in
    ``__init__`` spawned a ``curl`` per absent file (minirec + videos) for tests
    that never consume them. The download must instead fire lazily, on the first
    ``wait_for`` / ``move_dlc_items`` call.
    """
    import tests.data_downloader as dd

    launched = []
    monkeypatch.setattr(
        dd, "Popen", lambda cmd, **_: launched.append(cmd) or _FakePopen()
    )

    downloader = dd.DataDownloader(
        base_dir=tmp_path, download_dlc=False, verbose=False
    )

    assert launched == [], f"download launched on construction: {launched!r}"
    # The cached_property must not have been computed yet.
    assert "file_downloads" not in downloader.__dict__


def test_data_downloader_downloads_lazily_on_wait_for(tmp_path, monkeypatch):
    """The deferred download still fires when a consumer calls ``wait_for``.

    Guards against the fix over-correcting into a silent no-download: tests that
    genuinely need minirec/video must still get them.
    """
    import tests.data_downloader as dd

    launched = []
    monkeypatch.setattr(
        dd, "Popen", lambda cmd, **_: launched.append(cmd) or _FakePopen()
    )

    downloader = dd.DataDownloader(
        base_dir=tmp_path, download_dlc=False, verbose=False
    )
    target = dd.FILE_PATHS[0]["target_name"]
    downloader.wait_for(target, timeout=5, interval=1)

    assert any(target in " ".join(map(str, cmd)) for cmd in launched), (
        f"wait_for did not trigger the deferred download for {target!r}"
    )


# --------------------------------------------------------------------------
# Item B path 1: the v2 smoke fixture is fetched only when a collected test
# needs the DB, not unconditionally at session start (which downloaded a 57MB
# fixture even for pure-helper unit runs that consume nothing).
# --------------------------------------------------------------------------


class _FakeItem:
    def __init__(self, fixturenames):
        self.fixturenames = fixturenames


def test_eager_fetch_names_empty_for_pure_helper_run(monkeypatch):
    """With neither env var set, session start downloads nothing eagerly.

    This is the fix: a pure-helper run that requires no fixture and opts into no
    full fetch triggers no download at session start.
    """
    from tests.spikesorting.v2.conftest import _eager_fetch_names

    monkeypatch.delenv("SPYGLASS_V2_REQUIRE_FIXTURES", raising=False)
    monkeypatch.delenv("SPYGLASS_V2_FETCH_FULL", raising=False)

    assert _eager_fetch_names() == []


def test_eager_fetch_names_fetches_required_set(monkeypatch):
    """The honest-green-gated fixtures are pre-fetched so the gate finds them."""
    from tests.spikesorting.v2.conftest import _eager_fetch_names

    monkeypatch.setenv("SPYGLASS_V2_REQUIRE_FIXTURES", "mearec_polymer_smoke")
    monkeypatch.delenv("SPYGLASS_V2_FETCH_FULL", raising=False)

    assert _eager_fetch_names() == ["mearec_polymer_smoke"]


def test_eager_fetch_names_full_opt_in(monkeypatch):
    """``SPYGLASS_V2_FETCH_FULL=1`` still pulls every configured fixture."""
    from tests.spikesorting.v2.conftest import _eager_fetch_names
    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    monkeypatch.setenv("SPYGLASS_V2_FETCH_FULL", "1")

    assert set(_eager_fetch_names()) == set(FIXTURE_URLS)


def test_eager_fetch_names_ignores_unknown_required(monkeypatch):
    """A required fixture with no download URL is gate-checked, not fetched."""
    from tests.spikesorting.v2.conftest import _eager_fetch_names

    monkeypatch.setenv("SPYGLASS_V2_REQUIRE_FIXTURES", "no_such_fixture_xyz")
    monkeypatch.delenv("SPYGLASS_V2_FETCH_FULL", raising=False)

    assert _eager_fetch_names() == []


def test_session_needs_db_detects_dj_conn():
    """The lazy smoke fetch fires only when a collected test requests the DB."""
    from tests.spikesorting.v2.conftest import _session_needs_db

    assert not _session_needs_db([_FakeItem(["tmp_path"]), _FakeItem([])])
    assert _session_needs_db([_FakeItem(["tmp_path"]), _FakeItem(["dj_conn"])])


def test_require_fixtures_gate_still_exits_nonzero():
    """The honest-green gate must fail loudly when a required fixture is absent.

    The Item B change must not weaken this into a silent skip: CI relies on it to
    catch a fixture whose download failed. Run a child pytest that requires a
    genuinely-absent fixture and assert it exits non-zero with a pointed message.
    """
    # Target this very module: the gate fires in pytest_sessionstart regardless
    # of which path is collected, and a self-reference can't break for an
    # unrelated sibling rename.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/spikesorting/v2/test_harness.py",
            "--collect-only",
            "--no-docker",
            "-p",
            "no:xvfb",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-o",
            "addopts=-p no:warnings",
        ],
        env={
            **_clean_env(),
            "SPYGLASS_V2_REQUIRE_FIXTURES": "no_such_fixture_xyz",
        },
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0, (
        "gate did not fail on a missing required fixture:\n" + proc.stdout
    )
    assert "no_such_fixture_xyz" in (proc.stdout + proc.stderr)


def _clean_env():
    """Child-process env without the fixture-control vars the test sets itself."""
    import os

    env = dict(os.environ)
    env.pop("SPYGLASS_V2_REQUIRE_FIXTURES", None)
    env.pop("SPYGLASS_V2_FETCH_FULL", None)
    return env


# --------------------------------------------------------------------------
# Item C: every category named in ``filterwarnings`` must stay resolvable, so
# toggling ``-p no:warnings`` (a developer wanting to see warnings) does not
# break collection. pytest resolves a bare category against ``builtins``; a bare
# *custom* category therefore fails with AttributeError.
# --------------------------------------------------------------------------


def test_filterwarnings_categories_are_resolvable():
    """Each ``filterwarnings`` category resolves the way pytest resolves it.

    A bare name must be a builtin warning; anything else must be fully qualified
    (``module.path.Category``) and importable. A bare custom category -- the bug
    this guards -- crashes collection once the warnings plugin is active.
    """
    import builtins
    import importlib
    import tomllib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    config = tomllib.loads((repo_root / "pyproject.toml").read_text())
    filters = config["tool"]["pytest"]["ini_options"]["filterwarnings"]

    for entry in filters:
        # filterwarnings format: action:message:category:module:lineno
        parts = entry.split(":")
        category = parts[2] if len(parts) > 2 else ""
        if not category:
            continue  # no category named (e.g. "ignore::ResourceWarning" -> set)
        if "." in category:
            module_path, _, klass = category.rpartition(".")
            module = importlib.import_module(module_path)
            resolved = getattr(module, klass)
        else:
            assert hasattr(builtins, category), (
                f"filterwarnings category {category!r} is a bare name but not a "
                "builtin warning; pytest resolves it against builtins and "
                "crashes. Qualify it as module.path.Category."
            )
            resolved = getattr(builtins, category)
        assert isinstance(resolved, type) and issubclass(resolved, Warning), (
            f"filterwarnings category {category!r} is not a Warning subclass"
        )
