"""Tests for the shared pytest harness: teardown, fixture fetching, warnings.

These exercise the *harness itself* (``tests/conftest.py`` teardown, the shared
``DataDownloader``), not any Spyglass pipeline, so they run DB-free. Run with
``--no-docker`` to skip the MySQL container the root ``pytest_configure`` would
otherwise build at session start.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_unconfigure_tolerates_unbound_server():
    """``pytest_unconfigure`` must not raise when ``SERVER`` is unset.

    ``pytest_configure`` sets ``TEARDOWN`` before it binds ``SERVER`` (the latter
    only after building the Docker MySQL manager). If configure raises in between
    -- e.g. Docker is unavailable -- pytest still runs ``pytest_unconfigure`` from
    ``wrap_session``'s ``finally``. An unguarded ``SERVER.stop()`` would then
    raise a second traceback (``NameError`` where the name was never bound;
    ``AttributeError`` here, where we bind the module default ``None``) that
    buries the real configuration error. The module-level ``SERVER = None``
    default plus the ``SERVER is not None`` teardown guard make the real error
    surface instead.
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

        # Must not raise. Unguarded, this would raise AttributeError on
        # ``None.stop()`` (NameError where the name is unbound).
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
    touch no DB and no sample data. Resolving ``file_downloads`` eagerly in
    ``__init__`` would spawn a ``curl`` per absent file (minirec + videos) for
    tests that never consume them, so the download fires lazily, on the first
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

    Deferring the download must not turn into never downloading: tests that
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

    assert any(
        target in " ".join(map(str, cmd)) for cmd in launched
    ), f"wait_for did not trigger the deferred download for {target!r}"


# --------------------------------------------------------------------------
# The v2 smoke fixture is fetched only when a collected test needs the DB, not
# unconditionally at session start, so pure-helper unit runs that consume
# nothing do not download the 57MB fixture.
# --------------------------------------------------------------------------


def test_eager_fetch_names_empty_for_pure_helper_run(monkeypatch):
    """With neither env var set, session start downloads nothing eagerly.

    A pure-helper run that requires no fixture and opts into no full fetch
    triggers no download at session start.
    """
    from tests.spikesorting.v2.conftest import _eager_fetch_names

    monkeypatch.delenv("SPYGLASS_V2_REQUIRE_FIXTURES", raising=False)
    monkeypatch.delenv("SPYGLASS_V2_FETCH_FULL", raising=False)

    assert _eager_fetch_names() == []


def test_eager_fetch_names_fetches_required_set(monkeypatch):
    """Required fixtures are pre-fetched so the required-fixture check finds
    them."""
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
    """A required fixture with no download URL is checked for, not fetched."""
    from tests.spikesorting.v2.conftest import _eager_fetch_names

    monkeypatch.setenv("SPYGLASS_V2_REQUIRE_FIXTURES", "no_such_fixture_xyz")
    monkeypatch.delenv("SPYGLASS_V2_FETCH_FULL", raising=False)

    assert _eager_fetch_names() == []


def test_smoke_fixture_fetches_missing_verified_artifact(tmp_path, monkeypatch):
    """The explicit dependency downloads before a consumer's existence check."""
    import hashlib

    from tests.spikesorting.v2.conftest import smoke_nwb
    from tests.spikesorting.v2.fixtures import _fetch

    payload = b"canonical smoke fixture"
    expected = tmp_path / "mearec_polymer_smoke.nwb"
    monkeypatch.setattr(_fetch, "_THIS_DIR", tmp_path)
    monkeypatch.setattr(
        _fetch,
        "_manifest_nwb_sha256",
        lambda name: hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        _fetch, "_download_http", lambda url, path: path.write_bytes(payload)
    )
    assert not expected.exists()
    assert smoke_nwb.__wrapped__() == expected
    assert expected.read_bytes() == payload


def test_smoke_fixture_download_failure_is_an_error(tmp_path, monkeypatch):
    """A missing required input cannot silently skip a targeted regression."""
    from tests.spikesorting.v2.conftest import smoke_nwb
    from tests.spikesorting.v2.fixtures import _fetch

    monkeypatch.setattr(_fetch, "_THIS_DIR", tmp_path)
    monkeypatch.setitem(_fetch.FIXTURE_URLS, "mearec_polymer_smoke", None)
    with pytest.raises(_fetch.FixtureFetchError, match="no download URL"):
        smoke_nwb.__wrapped__()


@pytest.mark.parametrize(
    "fixture_name",
    ["populated_sorting", "planted_two_unit_sort", "planted_three_unit_sort"],
)
def test_shared_sort_fixtures_require_smoke_nwb(request, fixture_name):
    """Pytest schedules the download even when the test names no NWB path."""
    fixture_defs = request._fixturemanager.getfixturedefs(
        fixture_name, request.node
    )
    assert fixture_defs
    assert "smoke_nwb" in fixture_defs[-1].argnames


def test_require_fixtures_gate_ignores_stale_ingested_copies(
    tmp_path, monkeypatch
):
    """A leftover ingest copy must not satisfy a downloaded fixture's check.

    ``copy_and_insert_nwb`` copies every ingested fixture into the shared raw
    data directory under its own stem, so that directory accumulates files
    named exactly like the v2 fixtures. The check exists to prove THIS run
    downloaded and verified the fixture, so only
    ``tests/spikesorting/v2/fixtures/<name>.nwb`` counts for a name
    ``_fetch.py`` knows.

    The real recorded session (``minirec20230622``) has no entry there -- CI
    curls it straight into the raw data directory -- so for that name, and
    only that name, the raw directory is where the check looks.
    """
    import tests.spikesorting.v2.conftest as v2_conftest
    from tests.spikesorting.v2.conftest import _missing_required_fixtures
    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    # Mirror the real layout: the helper resolves both directories from its
    # own module path (``tests/spikesorting/v2/conftest.py`` ->
    # ``./fixtures`` and ``../../_data/raw``).
    conftest_path = tmp_path / "tests" / "spikesorting" / "v2" / "conftest.py"
    fixtures_dir = conftest_path.parent / "fixtures"
    raw_dir = tmp_path / "tests" / "_data" / "raw"
    fixtures_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(v2_conftest, "__file__", str(conftest_path))

    assert "mearec_polymer_smoke" in FIXTURE_URLS
    assert "minirec20230622" not in FIXTURE_URLS
    required = ["mearec_polymer_smoke", "minirec20230622"]

    # Nothing anywhere: both are missing.
    assert _missing_required_fixtures(required) == required

    # A stale ingest copy of the DOWNLOADED fixture sits in the raw directory,
    # and the real session sits there legitimately.
    (raw_dir / "mearec_polymer_smoke.nwb").write_bytes(b"stale ingest copy")
    (raw_dir / "minirec20230622.nwb").write_bytes(b"curled by CI")
    assert _missing_required_fixtures(required) == ["mearec_polymer_smoke"], (
        "a leftover copy in the raw data directory satisfied a downloaded "
        "fixture's check, so a failed download would look green"
    )

    # The verified download itself is what clears it.
    (fixtures_dir / "mearec_polymer_smoke.nwb").write_bytes(b"downloaded")
    assert _missing_required_fixtures(required) == []

    # ...and the fixtures directory is NOT where the real session is looked
    # for, so a file of that name there does not clear it.
    (raw_dir / "minirec20230622.nwb").unlink()
    (fixtures_dir / "minirec20230622.nwb").write_bytes(b"wrong home")
    assert _missing_required_fixtures(required) == ["minirec20230622"]


def test_missing_fixture_message_separates_unhosted_from_failed(monkeypatch):
    """The check's exit message says WHY each required fixture is absent.

    A fixture with no download URL is not hosted, so no re-run can fix it and
    the message must point at the hosting instructions. A fixture that has a
    URL, or that ``_fetch.py`` does not know (the curled real session), is
    absent because a download failed or a link went stale. Each stem must be
    named only under its own reason.
    """
    from tests.spikesorting.v2.conftest import _missing_fixtures_message
    from tests.spikesorting.v2.fixtures import _fetch

    no_url, has_url, unknown = "no_url_xyz", "has_url_xyz", "minirec20230622"
    monkeypatch.setitem(_fetch.FIXTURE_URLS, no_url, None)
    monkeypatch.setitem(_fetch.FIXTURE_URLS, has_url, "https://example.invalid")
    assert unknown not in _fetch.FIXTURE_URLS

    lines = _missing_fixtures_message([no_url, has_url, unknown]).splitlines()
    assert len(lines) == 2, lines
    unhosted_line, failed_line = lines

    assert no_url in unhosted_line
    assert has_url not in unhosted_line and unknown not in unhosted_line
    assert "No download URL is configured" in unhosted_line
    assert "the fixture is not hosted" in unhosted_line
    assert "tests/spikesorting/v2/fixtures/README.md" in unhosted_line
    assert "download step failed" not in unhosted_line

    assert has_url in failed_line and unknown in failed_line
    assert no_url not in failed_line
    assert "The download step failed or a Box link is stale" in failed_line
    assert "No download URL is configured" not in failed_line

    # Only the reasons that apply are reported.
    assert "No download URL" not in _missing_fixtures_message([has_url])
    assert "download step failed" not in _missing_fixtures_message([no_url])


def test_require_fixtures_gate_still_exits_nonzero():
    """The required-fixture check fails loudly when a required fixture is
    absent.

    Fetching fixtures lazily must not weaken this into a silent skip: CI relies
    on it to catch a fixture whose download failed. Run a child pytest that
    requires a genuinely-absent fixture and assert it exits non-zero with a
    pointed message.
    """
    # Target this very module by its absolute path (``__file__``) so the
    # reference can never rot when the file is renamed. The check fires in the
    # package ``pytest_sessionstart`` regardless of which path is collected;
    # using a real, collectable module guarantees the ONLY reason for a
    # non-zero exit is the check -- a bogus path would exit non-zero on its own
    # and mask a weakened check (false green).
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "--collect-only",
            "--no-docker",
            # The child shares this session's tests/_data; without this its
            # teardown deletes the parent's tmp/ and analysis/ mid-session.
            "--no-teardown",
            "-p",
            "no:xvfb",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-o",
            "addopts=",
        ],
        env={
            **_clean_env(),
            "SPYGLASS_V2_REQUIRE_FIXTURES": "no_such_fixture_xyz",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "check did not fail on a missing required fixture:\n" + out
    )
    # The non-zero exit must come from the check, not a file-not-found /
    # collection error -- otherwise this test would stay green even if the check
    # stopped firing.
    assert "Required v2 fixtures are absent" in out, (
        "exit was not the required-fixture check:\n" + out
    )
    assert "no_such_fixture_xyz" in out


def _clean_env():
    """Child-process env without the fixture-control vars the test sets itself."""
    import os

    env = dict(os.environ)
    env.pop("SPYGLASS_V2_REQUIRE_FIXTURES", None)
    env.pop("SPYGLASS_V2_FETCH_FULL", None)
    return env


# --------------------------------------------------------------------------
# Every category named in ``filterwarnings`` must stay resolvable, so toggling
# ``-p no:warnings`` (a developer wanting to see warnings) does not break
# collection. pytest resolves a bare category against ``builtins``; a bare
# *custom* category therefore fails with AttributeError.
# --------------------------------------------------------------------------


def test_filterwarnings_categories_are_resolvable():
    """Each ``filterwarnings`` category resolves the way pytest resolves it.

    A bare name must be a builtin warning; anything else must be fully qualified
    (``module.path.Category``) and importable. A bare custom category crashes
    collection once the warnings plugin is active.
    """
    import builtins
    import importlib
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

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
        assert isinstance(resolved, type) and issubclass(
            resolved, Warning
        ), f"filterwarnings category {category!r} is not a Warning subclass"
