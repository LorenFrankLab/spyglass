"""Tests for the dirty-install and unification across conda calls.

Split into two groups:
  Mock-based (no DB): CondaEnvCache and SpyglassInstall, including warnings
  DB-required:        UserEnvironment dirty-column integration

Run with: conda run -n dirty pytest tests/common/test_dirty.py --no-teardown
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ─── Shared mock data ─────────────────────────────────────────────────────────

_CONDA_STDOUT = yaml.dump(
    {
        "name": "test_env",
        "prefix": "/opt/conda/envs/test_env",
        "channels": ["defaults"],
        "dependencies": [
            "numpy=1.24.0=py310h",
            {"pip": ["scipy==1.11.0", "spyglass==0.5.6"]},
        ],
    }
)

_PIP_STDOUT = "scipy==1.11.0\nspyglass==0.5.6\n"


def _as_install(**facts):
    """Patch the process-wide install to report these facts.

    Each fact is a `cached_property`, so priming the instance dict is how they
    are set, and a test need only name what it exercises. Returns a context
    manager, so a test reads as "with this install, ...".
    """
    from spyglass.utils.git_utils import spyglass_install

    return patch.dict(spyglass_install.__dict__, facts)


def _proc(stdout="", returncode=0) -> CompletedProcess:
    return CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def _warn_with(facts: dict) -> None:
    """Warn as an install with these facts, bypassing real inspection."""
    from spyglass.utils.git_utils import SpyglassInstall

    inst = SpyglassInstall()
    inst.__dict__.update(facts)  # prime the cached properties
    inst.warn()


_OFFICIAL_REMOTE_V = (
    "sg\thttps://github.com/LorenFrankLab/spyglass (fetch)\n"
    "sg\thttps://github.com/LorenFrankLab/spyglass (push)\n"
)


def _git_mock(
    rev_parse="abc1234\n",
    diff="",
    log="4102444800",
    branch="  sg/master\n",
    remote=_OFFICIAL_REMOTE_V,
    for_each_ref="abc1234 commit\trefs/remotes/sg/master\n",
):
    """Return a ``git_utils.sub_run`` side effect keyed on the git subcommand.

    Dispatching on the command rather than on call order: which git commands
    ``SpyglassInstall`` runs, and in what sequence, is an implementation
    detail, and a positional ``side_effect`` list has to be re-shuffled every
    time it moves. Each argument takes a string (stdout, returncode 0) or a
    ``CompletedProcess`` for a non-zero exit.
    """
    by_subcommand = {
        "rev-parse": rev_parse,
        "diff": diff,
        "log": log,
        # Defaults: official remote configured, refs fetched, HEAD on a branch
        "branch": branch,
        "remote": remote,
        "for-each-ref": for_each_ref,
    }

    def _dispatch(cmd, *_, **__):
        # The subcommand only: a tmp_path can contain a word like "branch"
        name = cmd[3] if len(cmd) > 3 and cmd[1] == "-C" else ""
        for subcommand, result in by_subcommand.items():
            if subcommand == name:
                return (
                    result
                    if isinstance(result, CompletedProcess)
                    else _proc(result)
                )
        return _proc("")  # an unstubbed git call reads as empty, not an error

    return _dispatch


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def conda_env_cache_cls():
    """Return the CondaEnvCache class (deferred import)."""
    from spyglass.utils.env_cache import CondaEnvCache

    return CondaEnvCache


@pytest.fixture(scope="module")
def not_computed():
    """Return the _NOT_COMPUTED sentinel (deferred import)."""
    from spyglass.utils.env_cache import _NOT_COMPUTED

    return _NOT_COMPUTED


@pytest.fixture(scope="module")
def warn_dirty_days():
    """Return the WARN_DIRTY_ENV_DAYS constant (deferred import)."""
    from spyglass.utils.git_utils import WARN_DIRTY_ENV_DAYS

    return WARN_DIRTY_ENV_DAYS


@pytest.fixture(scope="module")
def admin_trust_days():
    """Return the WARN_ADMIN_TRUST_DAYS constant (deferred import)."""
    from spyglass.utils.git_utils import WARN_ADMIN_TRUST_DAYS

    return WARN_ADMIN_TRUST_DAYS


@pytest.fixture(scope="module")
def compute_install_info_fn():
    """Inspect the install afresh, bypassing the process-wide instance.

    Every fact is read before returning. They are cached properties, so a
    lazily-inspected instance would read whatever the working tree looks like
    at the assertion rather than what the test had patched.
    """
    from spyglass.utils.git_utils import SpyglassInstall

    facts = (
        "is_dev",
        "commit_hash",
        "install_path",
        "has_local_changes",
        "is_stale",
        "is_official",
    )

    def _inspect():
        fresh = SpyglassInstall()
        for fact in facts:
            getattr(fresh, fact)
        return fresh

    return _inspect


@pytest.fixture
def fake_clone():
    """Build a spyglass clone layout and return its ``__init__.py`` path.

    ``install_path`` is only accepted when ``<root>/src/spyglass/__init__.py``
    resolves to ``spyglass.__file__``, so a bare ``<root>/spyglass/`` or a
    pointer at this test file is rejected, so tests build the real shape.
    """

    def _factory(tmp_path):
        (tmp_path / ".git").mkdir()
        pkg_init = tmp_path / "src" / "spyglass" / "__init__.py"
        pkg_init.parent.mkdir(parents=True)
        pkg_init.write_text("")
        return pkg_init

    return _factory


@pytest.fixture
def make_cache(conda_env_cache_cls, not_computed):
    """Fixture factory returning isolated CondaEnvCache instances."""

    def _factory(tmp_path, *, cache_exists=False, history_mtime=None):
        """Create a fresh CondaEnvCache scoped to tmp_path.

        Both staleness signals are redirected into ``tmp_path``, so a
        verdict does not depend on whatever was last installed on this
        machine. They start well in the past, so a test must move one
        deliberately to induce staleness.
        """
        cache = conda_env_cache_cls()
        cache_path = tmp_path / ".spyglass_conda_test_prefix.yaml"
        history_path = tmp_path / "history"
        site_path = tmp_path / "site-packages"
        history_path.write_text("# conda history\n")
        site_path.mkdir()

        past = time.time() - 100
        for p in (history_path, site_path):
            os.utime(p, (past, past))
        if history_mtime is not None:
            os.utime(history_path, (history_mtime, history_mtime))
        if cache_exists:
            old_time = time.time() - 10
            cache_path.write_text(
                yaml.dump(
                    {
                        "conda": {
                            "channels": ["defaults"],
                            "dependencies": [],
                        },
                        "pip": [],
                    }
                )
            )
            os.utime(cache_path, (old_time, old_time))
            os.utime(history_path, (time.time(), time.time()))
        cache.__dict__["_cache_path"] = cache_path
        cache.__dict__["_history_path"] = history_path
        cache.__dict__["_site_packages_paths"] = (site_path,)
        return cache, cache_path

    return _factory


@pytest.fixture(autouse=True, scope="module")
def _reset_user_env_cache(user_env_tbl):
    """Don't leak cached_property state to other test modules.

    Tests here poke cached_property attrs on the session-scoped
    ``user_env_tbl``. If this module ends with ``this_env`` cached but
    ``matching_env_id`` popped, a later module's ``del ...matching_env_id``
    raises ``AttributeError``. Clear all of them together on teardown.
    """
    yield
    for attr in ("env", "env_hash", "matching_env_id", "this_env"):
        user_env_tbl.__dict__.pop(attr, None)


@pytest.fixture(autouse=True)
def _clear_dirty_marker_each_test():
    """Keep the local dirty marker from leaking across tests."""
    from spyglass.utils import git_utils

    git_utils.spyglass_install.marker.clear()
    yield
    git_utils.spyglass_install.marker.clear()


# ═════════════════════════════════════════════════════════════════════════════
#     CondaEnvCache (mock-based, no DB)
# ═════════════════════════════════════════════════════════════════════════════


def test_cache_hit(tmp_path, make_cache):
    """Conda-only get() runs conda once; the second call is fully cached."""
    cache, _ = make_cache(tmp_path)
    side_effects = [_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)]

    with patch(
        "spyglass.utils.env_cache.sub_run", side_effect=side_effects
    ) as mock_run:
        first = cache.get()
        second = cache.get()

    assert mock_run.call_count == 1  # conda only; pip freeze skipped
    assert first is second  # same in-process dict object


def test_conda_only_skips_pip_freeze(tmp_path, make_cache):
    """get() (conda-only) runs conda env export but never pip freeze."""
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ) as mock_run:
        cache.get()

    assert [c.args[0][0] for c in mock_run.call_args_list] == ["conda"]
    # Disk holds conda only, with no partial pip list
    assert "pip" not in yaml.safe_load(cache_path.read_text())


def test_with_pip_after_conda_only_fetches_pip(tmp_path, make_cache):
    """A with_pip call after a conda-only call runs pip freeze and persists it."""
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ) as mock_run:
        cache.get()  # conda-only: conda export
        cache.get(with_pip=True)  # now needs pip: pip freeze

    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert cmds[0][0] == "conda"
    # pip runs via the current interpreter, not PATH's pip
    assert cmds[1] == [sys.executable, "-m", "pip", "freeze"]
    assert "pip" in yaml.safe_load(cache_path.read_text())  # disk completed


def test_cache_with_pip_hit(tmp_path, make_cache):
    """get(with_pip=True) also uses in-process cache after the first call."""
    cache, _ = make_cache(tmp_path)
    side_effects = [_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)]

    with patch("spyglass.utils.env_cache.sub_run", side_effect=side_effects):
        merged1 = cache.get(with_pip=True)
        merged2 = cache.get(with_pip=True)

    assert merged1 is merged2


def test_merge_leaves_cached_pip_intact(tmp_path, make_cache, not_computed):
    """A merge must not strip the pip list from the cached conda export.

    The conda export is shallow-copied before merging, so the embedded pip
    dict is shared with the cache. A pop there empties the cached export, so
    every later conda-only reader records no pip packages: the analysis NWB
    ``source_script`` and the export ``environment.yml``.
    """
    cache, _ = make_cache(tmp_path)

    def pip_entries(env):
        return next(
            (d.get("pip") for d in env["dependencies"] if isinstance(d, dict)),
            None,
        )

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get(with_pip=True)
        conda_only = cache.get()
        cache._merged = not_computed  # force a second merge of the same raw
        merged_again = cache.get(with_pip=True)

    assert pip_entries(conda_only) == ["scipy==1.11.0", "spyglass==0.5.6"]
    assert pip_entries(merged_again)  # re-merge still sees conda's pip list


def test_get_conda_only_keys(tmp_path, make_cache):
    """get() returns only conda-export keys; no custom or pip-freeze artifacts."""
    cache, _ = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        result = cache.get()

    assert "channels" in result
    assert "dependencies" in result
    assert "custom" not in result
    assert "raw pip freeze" not in result


def test_get_with_pip_merges(tmp_path, make_cache):
    """get(with_pip=True) merges pip freeze into the dependencies structure."""
    pip_out = "scipy==1.11.0\nspyglass==0.5.6\nnewpkg==2.0.0\n"
    cache, _ = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(pip_out)],
    ):
        merged = cache.get(with_pip=True)

    # Merged dict has same conda top-level keys; may also have 'custom'
    assert "channels" in merged
    assert "dependencies" in merged
    assert "raw pip freeze" not in merged

    # newpkg is pip-only, should appear in the pip sub-list
    pip_deps = next(
        (
            d["pip"]
            for d in merged["dependencies"]
            if isinstance(d, dict) and "pip" in d
        ),
        [],
    )
    assert any("newpkg" in entry for entry in pip_deps)


def test_disk_cache_format(tmp_path, make_cache):
    """Disk cache stores raw conda and pip under separate top-level keys."""
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get(with_pip=True)

    assert cache_path.exists()
    on_disk = yaml.safe_load(cache_path.read_text())
    assert set(on_disk.keys()) == {"conda", "pip"}
    assert isinstance(on_disk["conda"], dict)
    assert isinstance(on_disk["pip"], list)
    assert "raw pip freeze" not in on_disk


def test_cache_stale_on_history_change(tmp_path, make_cache, not_computed):
    """A newer conda-meta/history makes a fresh cache stale."""
    cache, cache_path = make_cache(tmp_path)
    side_effects_round1 = [_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)]
    side_effects_round2 = [_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)]

    with patch(
        "spyglass.utils.env_cache.sub_run", side_effect=side_effects_round1
    ):
        cache.get()

    assert cache_path.exists()

    # Simulate conda install: advance history mtime past cache mtime
    future = time.time() + 10
    os.utime(cache.__dict__["_history_path"], (future, future))

    # Reset in-process state to simulate a new import
    cache._cached_conda = not_computed
    cache._cached_pip = not_computed
    cache._merged = not_computed

    with patch(
        "spyglass.utils.env_cache.sub_run", side_effect=side_effects_round2
    ) as mock_run:
        cache.get()

    assert mock_run.call_count == 1  # conda re-fetched after staleness


def test_cache_loads_from_disk(tmp_path, make_cache):
    """A fresh disk cache avoids the subprocess entirely."""
    cache, cache_path = make_cache(tmp_path)

    cache_data = {
        "conda": {"channels": ["defaults"], "dependencies": []},
        "pip": [],
    }
    cache_path.write_text(yaml.dump(cache_data))
    old_time = time.time() - 100
    os.utime(cache.__dict__["_history_path"], (old_time, old_time))

    with patch("spyglass.utils.env_cache.sub_run") as mock_run:
        result = cache.get()

    mock_run.assert_not_called()
    assert result == {"channels": ["defaults"], "dependencies": []}


def test_cache_no_temp_dir(conda_env_cache_cls):
    """When _cache_path is None, get() still returns a result (no crash)."""
    cache = conda_env_cache_cls()
    cache.__dict__["_cache_path"] = None
    cache.__dict__["_history_path"] = None

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        result = cache.get()

    assert isinstance(result, dict)
    assert "channels" in result


def test_cache_no_conda_prefix(tmp_path, make_cache):
    """A pip-only venv uses site-packages alone, rather than always re-exporting.

    With no ``CONDA_PREFIX`` there is no conda history to compare against. The
    cache calls itself stale only when a signal moves, so a uv or pip venv
    no longer re-exports on every import and in every spawn worker.
    """
    cache, cache_path = make_cache(tmp_path)
    cache.__dict__["_history_path"] = None
    cache_path.write_text(yaml.dump({"conda": {"dependencies": []}}))

    assert cache.is_stale() is False

    future = time.time() + 10  # pip install
    os.utime(cache.__dict__["_site_packages_paths"][0], (future, future))

    assert cache.is_stale() is True


def test_cache_no_signals_is_stale(tmp_path, make_cache):
    """With neither conda history nor site-packages, assume the env changed."""
    cache, cache_path = make_cache(tmp_path)
    cache.__dict__["_history_path"] = None
    cache.__dict__["_site_packages_paths"] = ()
    cache_path.write_text(yaml.dump({"conda": {"dependencies": []}}))

    assert cache.is_stale() is True


def test_cache_stale_on_pip_install(tmp_path, make_cache, not_computed):
    """A pip install with no conda transaction invalidates the cache.

    pip adds or removes a ``*.dist-info`` directory in site-packages, so the
    directory mtime moves. ``conda-meta/history`` does not, which is why it
    cannot be the only signal.
    """
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get()
    assert cache_path.exists()
    assert cache.is_stale() is False

    future = time.time() + 10  # pip install, no conda transaction
    os.utime(cache.__dict__["_site_packages_paths"][0], (future, future))
    assert cache.is_stale() is True

    cache._cached_conda = not_computed
    cache._cached_pip = not_computed
    cache._merged = not_computed

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ) as mock_run:
        cache.get()

    assert mock_run.call_count == 1  # re-exported


def test_cache_hit_refreshes_mtime(tmp_path, make_cache, not_computed):
    """A cache hit keeps the file off the weekly temp_dir sweep.

    `cleanup.py` deletes anything untouched for seven days, and a hit does not
    rewrite the file.
    """
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get()

    # Old enough for the sweep, newer than every staleness signal
    day = 60 * 60 * 24
    for sig in (
        cache.__dict__["_history_path"],
        cache.__dict__["_site_packages_paths"][0],
    ):
        os.utime(sig, (time.time() - 10 * day,) * 2)
    stale_time = time.time() - 8 * day
    os.utime(cache_path, (stale_time, stale_time))

    cache._cached_conda = not_computed  # a new session
    cache._merged = not_computed
    with patch("spyglass.utils.env_cache.sub_run") as mock_run:
        cache.get()

    mock_run.assert_not_called()  # served from disk
    assert cache_path.stat().st_mtime > stale_time  # and kept alive


def test_cache_not_touched_while_stale(tmp_path, make_cache, not_computed):
    """A stale cache is re-exported, not touched into looking fresh."""
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get()

    future = time.time() + 10  # a pip install after the cache was written
    os.utime(cache.__dict__["_site_packages_paths"][0], (future, future))
    assert cache.is_stale() is True

    cache._cached_conda = not_computed
    cache._merged = not_computed
    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ) as mock_run:
        cache.get()

    assert mock_run.call_count == 1  # re-exported rather than touched


def test_cache_equal_mtime_is_stale(tmp_path, make_cache):
    """An mtime equal to the cache's counts as stale.

    Filesystem timestamp granularity can place a change in the same tick as
    the cache write; treating equal as fresh would lose it.
    """
    cache, cache_path = make_cache(tmp_path)
    cache_path.write_text(yaml.dump({"conda": {"dependencies": []}}))

    same = cache_path.stat().st_mtime
    os.utime(cache.__dict__["_site_packages_paths"][0], (same, same))

    assert cache.is_stale() is True


def test_cache_scoped_by_user(tmp_path, monkeypatch, conda_env_cache_cls):
    """Two users sharing temp_dir get different cache file paths."""
    monkeypatch.setattr("spyglass.settings.temp_dir", str(tmp_path))

    cache_a = conda_env_cache_cls()
    cache_b = conda_env_cache_cls()

    with patch(
        "spyglass.utils.env_cache.getpass.getuser", return_value="alice"
    ):
        path_a = cache_a._cache_path

    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="bob"):
        path_b = cache_b._cache_path

    assert path_a != path_b


def test_cache_scoped_by_env(tmp_path, monkeypatch, conda_env_cache_cls):
    """Two conda environments for the same user get different cache paths.

    Scoped on ``sys.prefix`` rather than ``CONDA_DEFAULT_ENV``: two hosts can
    each hold an env of the same name, and on a shared ``temp_dir`` the name
    alone put them on one cache file.
    """
    monkeypatch.setattr("spyglass.settings.temp_dir", str(tmp_path))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "same_name")

    monkeypatch.setattr(
        "spyglass.utils.env_cache.sys_prefix", "/opt/envs/alpha"
    )
    cache_a = conda_env_cache_cls()
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_a = cache_a._cache_path

    monkeypatch.setattr("spyglass.utils.env_cache.sys_prefix", "/opt/envs/beta")
    cache_b = conda_env_cache_cls()
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_b = cache_b._cache_path

    assert path_a != path_b


def test_cache_scoped_by_activated_env(
    tmp_path, monkeypatch, conda_env_cache_cls
):
    """One interpreter under two activated envs gets two cache paths.

    `conda env export` reads `CONDA_PREFIX` while `pip freeze` reads
    `sys.prefix`. Keyed on the latter alone, a script started by path from
    another activated env stored that env's export under this env's name, and
    a correctly activated session then read it and ran no export at all.
    """
    monkeypatch.setattr("spyglass.settings.temp_dir", str(tmp_path))
    monkeypatch.setattr(
        "spyglass.utils.env_cache.sys_prefix", "/opt/envs/legacy"
    )

    monkeypatch.setenv("CONDA_PREFIX", "/opt/envs/base")
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_unactivated = conda_env_cache_cls()._cache_path

    monkeypatch.setenv("CONDA_PREFIX", "/opt/envs/legacy")
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_activated = conda_env_cache_cls()._cache_path

    assert path_unactivated != path_activated


def test_failed_pip_freeze_stores_nothing(tmp_path, make_cache):
    """A failed `pip freeze` yields no environment, not a trusted one.

    Substituting an empty pip list left a full export with `has_editable=0`,
    which `UserEnvironment.is_trusted` reads as examined and clean, so a venv
    without pip could log a row that gates recompute open.
    """
    cache, _ = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc("", returncode=1)],
    ):
        merged = cache.get(with_pip=True)

    assert merged == {}
    assert cache.get() != {}  # the conda half is still usable on its own


def test_atomic_write(tmp_path, make_cache):
    """save() produces a valid YAML file; a second save() overwrites cleanly."""
    cache, cache_path = make_cache(tmp_path)

    with patch(
        "spyglass.utils.env_cache.sub_run",
        side_effect=[_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)],
    ):
        cache.get()

    assert cache_path.exists()
    on_disk = yaml.safe_load(cache_path.read_text())
    assert isinstance(on_disk, dict)  # not a torn/partial write
    assert "conda" in on_disk

    # Second save() (same or different PID) also produces valid YAML
    cache.save()
    on_disk2 = yaml.safe_load(cache_path.read_text())
    assert isinstance(on_disk2, dict)


def test_load_bad_format(tmp_path, make_cache):
    """load() returns False when the on-disk file is missing 'conda' key."""
    cache, cache_path = make_cache(tmp_path)
    cache_path.write_text(yaml.dump({"bad_key": []}))

    assert cache.load() is False


def test_preload_already_loaded(tmp_path, make_cache):
    """preload() is a no-op when _cached_conda is already populated."""
    cache, _ = make_cache(tmp_path)
    cache._cached_conda = {"channels": [], "dependencies": []}

    with patch("spyglass.utils.env_cache.sub_run") as mock_run:
        cache.preload()

    mock_run.assert_not_called()


def test_preload_fresh_disk(tmp_path, make_cache):
    """preload() loads from a fresh disk cache without calling subprocess."""
    cache, cache_path = make_cache(tmp_path)
    cache_path.write_text(
        yaml.dump(
            {"conda": {"channels": ["defaults"], "dependencies": []}, "pip": []}
        )
    )
    old_time = time.time() - 100
    os.utime(cache.__dict__["_history_path"], (old_time, old_time))

    with patch("spyglass.utils.env_cache.sub_run") as mock_run:
        cache.preload()

    mock_run.assert_not_called()
    assert cache._cached_conda == {"channels": ["defaults"], "dependencies": []}


# ─── _parse_pip_line branch coverage ──────────────────────────────────────────


def test_parse_pip_line_editable_git(conda_env_cache_cls):
    """'-e git+...#egg=name' is stored as custom install; sets has_editable."""
    cache = conda_env_cache_cls()
    line = "-e git+https://github.com/user/mypkg.git#egg=my_pkg"
    assert cache._parse_pip_line(line) is True
    assert "my-pkg" in cache._pip_custom
    assert cache.has_editable is True


def test_parse_pip_line_editable_git_spyglass(conda_env_cache_cls):
    """'-e git+...#egg=spyglass' does NOT set has_editable."""
    cache = conda_env_cache_cls()
    line = "-e git+https://github.com/user/spyglass.git#egg=spyglass"
    assert cache._parse_pip_line(line) is True
    assert cache.has_editable is False


def test_parse_pip_line_custom_at(conda_env_cache_cls):
    """'name @ file://...' is recorded in _pip_custom."""
    cache = conda_env_cache_cls()
    line = "mypkg @ file:///home/user/mypkg"
    assert cache._parse_pip_line(line) is True
    assert "mypkg" in cache._pip_custom


def test_parse_pip_line_conda_conflict(conda_env_cache_cls):
    """When conda and pip disagree on a version, conflict is recorded."""
    cache = conda_env_cache_cls()
    cache._conda_pip_dict["scipy"] = "scipy==1.11.0"
    line = "scipy==2.0.0"
    assert cache._parse_pip_line(line) is True
    assert "scipy" in cache._conda_conflicts
    assert cache._conda_pip_dict["scipy"] == "scipy==2.0.0"  # pip wins


def test_parse_pip_line_comment(conda_env_cache_cls):
    """A '# (pkg==ver)' comment line is stored in _freeze_comments."""
    cache = conda_env_cache_cls()
    line = "# (some-package==1.2.3)"
    assert cache._parse_pip_line(line) is True
    assert cache._freeze_comments.get("some-package") == "1.2.3"


def test_parse_pip_line_editable_path(conda_env_cache_cls):
    """'-e /path' with a preceding comment is stored as a custom install."""
    cache = conda_env_cache_cls()
    cache._parse_pip_line("# (some-package==1.2.3)")
    assert cache._parse_pip_line("-e /path/to/some-package") is True
    assert "some-package" in cache._pip_custom


def test_parse_pip_line_editable_path_unrelated_dirname(conda_env_cache_cls):
    """The comment names the package, so the directory need not.

    A git worktree is named for its branch, so an editable Spyglass commonly
    sits in a directory sharing no name with the distribution. The package is
    read from the comment, which also keeps Spyglass exempt from
    ``has_editable``.
    """
    cache = conda_env_cache_cls()
    cache._parse_pip_line("# (spyglass-neuro==0.5.5a2.dev86+gf82724895)")

    assert cache._parse_pip_line("-e /home/user/wrk/spyglass/dirty") is True
    assert "spyglass-neuro" in cache._pip_custom
    assert cache.has_editable is False  # spyglass is exempt, by any dirname


def test_parse_pip_line_editable_comment_is_consumed(conda_env_cache_cls):
    """One comment pairs with one '-e' line and is not reused by the next."""
    cache = conda_env_cache_cls()
    cache._parse_pip_line("# (some-package==1.2.3)")
    cache._parse_pip_line("-e /path/to/whatever")

    with pytest.raises(ValueError, match="preceding comment"):
        cache._parse_pip_line("-e /path/to/second-editable")


def test_parse_pip_line_editable_path_no_comment(conda_env_cache_cls):
    """'-e /path' without a preceding comment raises ValueError."""
    cache = conda_env_cache_cls()
    with pytest.raises(ValueError, match="preceding comment"):
        cache._parse_pip_line("-e /path/to/missing-comment")


def test_parse_pip_line_parse_fail(conda_env_cache_cls):
    """An unrecognised line returns False."""
    cache = conda_env_cache_cls()
    assert cache._parse_pip_line("this is not a valid pip line") is False


def test_warn_if_custom_pip_emits_warning(tmp_path, make_cache, caplog):
    """_warn_if_custom_or_conflict warns when non-spyglass custom pkgs exist."""
    cache, _ = make_cache(tmp_path)
    cache._pip_custom["otherpkg"] = "otherpkg @ file:///home/user/otherpkg"

    with caplog.at_level("WARNING"):
        cache._warn_if_custom_or_conflict()

    assert any("Custom pip installs" in r.message for r in caplog.records)


def test_delim_split_list():
    """_delim_split on a list processes elements and returns a dict."""
    from spyglass.utils.env_cache import _delim_split

    result = _delim_split(["pkg1==1.0", "pkg2==2.0"], as_dict=True)
    assert result == {"pkg1": "1.0", "pkg2": "2.0"}


def test_delim_split_none():
    """_delim_split with None returns ('', None) tuple."""
    from spyglass.utils.env_cache import _delim_split

    assert _delim_split(None) == ("", None)
    assert _delim_split(None, as_dict=True) == {"": None}


# ═════════════════════════════════════════════════════════════════════════════
#     SpyglassInstall (mock-based, no DB)
# ═════════════════════════════════════════════════════════════════════════════


def test_pip_install_no_git(tmp_path, compute_install_info_fn):
    """Standard pip install: no .git ancestor → install_path None, no changes."""
    fake_pkg = tmp_path / "site-packages" / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    with patch("spyglass.__version__", "0.5.6"):
        with patch("spyglass.__file__", str(fake_pkg)):
            result = compute_install_info_fn()

    assert result.install_path is None
    assert result.has_local_changes is False
    assert result.is_official is True


def test_dev_version_is_dev(tmp_path, fake_clone, compute_install_info_fn):
    """A '.dev' version sets is_dev=True; local edits make it non-official."""
    # Mocked so the result does not depend on this working tree
    with patch("spyglass.__version__", "0.5.6.dev42+gb3db6d09"):
        with patch("spyglass.__file__", str(fake_clone(tmp_path))):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(diff="src/x.py\n"),
            ):
                result = compute_install_info_fn()

    assert result.is_dev is True
    assert result.has_local_changes is True
    assert result.is_official is False


def test_release_version_not_dev(tmp_path, compute_install_info_fn):
    """A clean release version string gives is_dev=False."""
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    with patch("spyglass.__version__", "0.5.6"):
        with patch("spyglass.__file__", str(fake_pkg)):
            result = compute_install_info_fn()

    assert result.is_dev is False


def test_commit_hash_parsed_from_version(
    tmp_path, fake_clone, compute_install_info_fn
):
    """The +g<hash> suffix in __version__ parses into commit_hash.

    Mocked rev-parse, because this test file lives inside a clone: left
    unmocked, ``install_path`` resolves to that clone and HEAD supplies the
    real hash, since HEAD now outranks ``__version__``.
    """
    with patch("spyglass.__version__", "0.5.6.dev1+gabcd1234"):
        with patch("spyglass.__file__", str(fake_clone(tmp_path))):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(rev_parse=_proc("", returncode=128)),
            ):
                result = compute_install_info_fn()

    assert result.commit_hash == "abcd1234"


def test_git_dirty_sets_has_local_changes(
    tmp_path, fake_clone, compute_install_info_fn
):
    """git diff returning output means has_local_changes=True."""
    dirty_proc = _proc("src/spyglass/something.py\n")

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_clone(tmp_path))):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(diff=dirty_proc.stdout),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is True
    assert result.is_official is False


def test_git_clean_no_changes(tmp_path, fake_clone, compute_install_info_fn):
    """git diff returning empty output means has_local_changes=False."""
    clean_proc = _proc("")

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_clone(tmp_path))):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(diff=clean_proc.stdout),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_git_unavailable_no_crash(
    tmp_path, fake_clone, compute_install_info_fn
):
    """FileNotFoundError from git is swallowed; has_local_changes=False."""
    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_clone(tmp_path))):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=FileNotFoundError("git not found"),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_git_short_hash_fallback(tmp_path, fake_clone, compute_install_info_fn):
    """When version has no +g hash, git rev-parse is called as a fallback."""
    fake_pkg = fake_clone(tmp_path)

    hash_proc = _proc("abc1234\n")
    diff_proc = _proc("")

    with patch("spyglass.__version__", "0.5.6.dev1"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(
                    rev_parse=hash_proc.stdout, diff=diff_proc.stdout
                ),
            ):
                result = compute_install_info_fn()

    assert result.commit_hash == "abc1234"


def test_official_remote_found_by_url(
    tmp_path, fake_clone, compute_install_info_fn
):
    """The official remote is identified by URL, whatever it is named.

    Remote names are a local convention: a clone may call it `sg`, `upstream`
    or `origin`, and may hold several forks alongside it.
    """
    from spyglass.utils.git_utils import SpyglassInstall

    fake_pkg = fake_clone(tmp_path)
    remotes = (
        "me\thttps://github.com/cbroz1/spyglass (fetch)\n"
        "me\thttps://github.com/cbroz1/spyglass (push)\n"
        "whatever\tgit@github.com:LorenFrankLab/spyglass.git (fetch)\n"
        "whatever\tgit@github.com:LorenFrankLab/spyglass.git (push)\n"
    )

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(remote=remotes),
            ):
                assert SpyglassInstall().official_remote == "whatever"


def test_no_official_remote_is_not_published(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A clone with no official remote cannot be reached from upstream."""
    fake_pkg = fake_clone(tmp_path)
    fork_only = (
        "me\thttps://github.com/cbroz1/spyglass (fetch)\n"
        "me\thttps://github.com/cbroz1/spyglass (push)\n"
    )

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(remote=fork_only),
            ):
                result = compute_install_info_fn()

    assert result.install_path is not None  # it is a clone
    assert result.has_local_changes is True


def test_unpushed_head_is_not_official(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A commit no official branch contains is a local change.

    `git diff` is empty once the work is committed, so committing was enough to
    read as clean. Pushing to a fork is likewise not enough: the code is still
    unobtainable from the official repository, and both cases are the one
    condition of no fetched official branch containing HEAD.
    """
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                # No official branch contains HEAD, though refs were fetched.
                side_effect=_git_mock(branch=""),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is True
    assert result.is_official is False


def test_published_head_with_clean_diff_is_official(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A pushed HEAD and a clean diff is still official."""
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(diff="", branch="  sg/master\n"),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False
    assert result.is_official is True


def test_official_feature_branch_is_published(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A clean checkout of an official branch other than master is official.

    Obtainable, not reviewed, is the test: anyone can fetch
    `spikesorting-v2` from the official remote, so a user testing it before it
    merges is reproducible and owed no warning, no email and no partial row.
    """
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(branch="  sg/spikesorting-v2\n"),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False
    assert result.is_official is True


def test_unfetched_clone_does_not_flag(
    tmp_path, fake_clone, compute_install_info_fn
):
    """With no official ref on disk there is nothing to compare against.

    A shallow or freshly-cloned-by-tag checkout has the remote configured but
    no remote-tracking branches, so the containment test would answer "no"
    about every commit.
    """
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(for_each_ref="", branch=""),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_official_remote_matched_case_insensitively(
    tmp_path, fake_clone, compute_install_info_fn
):
    """GitHub resolves `lorenfranklab/spyglass` to the same repository."""
    from spyglass.utils.git_utils import SpyglassInstall

    fake_pkg = fake_clone(tmp_path)
    lowercase = (
        "origin\thttps://github.com/lorenfranklab/Spyglass.git (fetch)\n"
        "origin\thttps://github.com/lorenfranklab/Spyglass.git (push)\n"
    )

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(remote=lowercase),
            ):
                assert SpyglassInstall().official_remote == "origin"


def test_git_branch_failure_does_not_flag(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A failing `git branch` leaves the install unflagged, not flagged."""
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(branch=_proc("", returncode=129)),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_wheel_built_from_dirty_tree_is_flagged(compute_install_info_fn):
    """A `.dYYYYMMDD` version flags an install with no repo to inspect.

    hatch-vcs adds the suffix when the tree it built from had uncommitted
    changes. `_VCS_HASH_RE` is anchored, so the suffix also prevents the commit
    being parsed, and a non-editable wheel has no `.git`, leaving the version
    string as the only trace.
    """
    with patch("spyglass.__version__", "0.6.1.dev1+gabcdef0.d20260101"):
        with patch("spyglass.__file__", "/opt/env/lib/spyglass/__init__.py"):
            result = compute_install_info_fn()

    assert result.install_path is None
    assert result.has_local_changes is True
    assert result.is_official is False


def test_clean_wheel_is_not_flagged(compute_install_info_fn):
    """A plain `+g` version with no dirty suffix stays official."""
    with patch("spyglass.__version__", "0.6.1.dev1+gabcdef0"):
        with patch("spyglass.__file__", "/opt/env/lib/spyglass/__init__.py"):
            result = compute_install_info_fn()

    assert result.has_local_changes is False
    assert result.is_official is True


def test_cleaned_clone_keeps_its_dirty_version(
    tmp_path, fake_clone, compute_install_info_fn
):
    """In a clone, git decides, not the `.dYYYYMMDD` suffix.

    hatch-vcs writes `_version.py` at install time and nothing rewrites it
    afterwards, so an editable install made while the tree had edits carries
    the suffix for the life of the install. Reading the suffix first flagged
    such a clone however clean it later became, and told its owner to reset a
    repository that was already on an official commit.
    """
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev2203+g34e554ce9.d20261008"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(),  # clean diff, published HEAD
            ):
                result = compute_install_info_fn()

    assert result.install_path is not None  # it is a clone
    assert result.has_local_changes is False
    assert result.is_official is True


def test_venv_inside_another_repo_is_not_a_clone(
    tmp_path, compute_install_info_fn
):
    """A release install in a venv inside someone else's repo is not a clone.

    uv puts `.venv` in the project directory by default, and an analysis
    project is often a git repo. Walking up to the first `.git` adopted that
    repo's commit and dirty state, so a user's own edits flagged spyglass as
    dirty and `dirty_path` named their analysis repo.
    """
    (tmp_path / ".git").mkdir()  # the user's analysis repo
    pkg_init = (
        tmp_path
        / ".venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "spyglass"
        / "__init__.py"
    )
    pkg_init.parent.mkdir(parents=True)
    pkg_init.write_text("")

    with patch("spyglass.__version__", "0.6.1"):
        with patch("spyglass.__file__", str(pkg_init)):
            result = compute_install_info_fn()

    assert result.install_path is None
    assert result.commit_hash is None
    assert result.has_local_changes is False
    assert result.is_official is True


def test_sibling_src_spyglass_is_not_this_install(
    tmp_path, compute_install_info_fn
):
    """A repo that merely contains a `src/spyglass` is not accepted either.

    The check compares resolved paths, not the layout alone, so vendoring or a
    second checkout cannot be mistaken for the running install.
    """
    (tmp_path / ".git").mkdir()
    decoy = tmp_path / "src" / "spyglass" / "__init__.py"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("")

    elsewhere = tmp_path / ".venv" / "spyglass" / "__init__.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("")

    with patch("spyglass.__version__", "0.6.1"):
        with patch("spyglass.__file__", str(elsewhere)):
            result = compute_install_info_fn()

    assert result.install_path is None


def test_src_layout_clone_is_accepted(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A real clone at <root>/src/spyglass still resolves to its root."""
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.6.1.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(rev_parse="abc1234\n"),
            ):
                result = compute_install_info_fn()

    assert result.install_path == str(tmp_path)
    assert result.commit_hash == "abc1234"


def test_head_outranks_version_hash(
    tmp_path, fake_clone, compute_install_info_fn
):
    """In a clone, HEAD supplies commit_hash even when __version__ carries one.

    hatch-vcs writes ``_version.py`` at ``pip install -e .`` and a ``git pull``
    does not rewrite it, so ``__version__`` keeps naming the commit the clone
    was installed at. Recompute would then treat results from new code as
    results from the old environment.
    """
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.5.6.dev1+gaaaaaaa"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(rev_parse="bbbbbbb\n"),
            ):
                result = compute_install_info_fn()

    assert result.commit_hash == "bbbbbbb"


def test_version_hash_used_without_a_clone(compute_install_info_fn):
    """With no repo, commit_hash still comes from the __version__ suffix."""
    with patch("spyglass.__version__", "0.5.6.dev1+gaaaaaaa"):
        with patch("spyglass.__file__", "/opt/env/lib/spyglass/__init__.py"):
            result = compute_install_info_fn()

    assert result.install_path is None
    assert result.commit_hash == "aaaaaaa"


def test_git_diff_nonzero_exit(tmp_path, fake_clone, compute_install_info_fn):
    """git diff non-zero exit code is handled gracefully; has_local_changes=False."""
    fake_pkg = fake_clone(tmp_path)

    error_proc = _proc("", returncode=128)

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(diff=error_proc),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_git_diff_generic_exception(
    tmp_path, fake_clone, compute_install_info_fn
):
    """Generic exception from git diff is swallowed; has_local_changes=False."""
    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=RuntimeError("unexpected"),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False


def test_git_commit_date_reads_head(tmp_path, fake_clone):
    """commit_date is HEAD's committer date, as a date."""
    from spyglass.utils.git_utils import SpyglassInstall

    fake_pkg = fake_clone(tmp_path)
    stamp = int(
        datetime.combine(date(2024, 3, 7), datetime.min.time()).timestamp()
    )

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(log=str(stamp)),
            ):
                assert SpyglassInstall().commit_date == date(2024, 3, 7)


def test_git_commit_date_none_off_a_clone():
    """A release install has no clone, so no date to store."""
    from spyglass.utils.git_utils import SpyglassInstall

    inst = SpyglassInstall()
    inst.__dict__["install_path"] = None

    assert inst.commit_date is None


def test_git_commit_date_unreadable(tmp_path, fake_clone):
    """Unparsable git output yields no date rather than raising."""
    from spyglass.utils.git_utils import SpyglassInstall

    fake_pkg = fake_clone(tmp_path)

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(log="not-a-timestamp"),
            ):
                assert SpyglassInstall().commit_date is None


@pytest.mark.parametrize(
    "age_days, expected",
    [(200, True), (5, False)],
    ids=["months_old", "recent"],
)
def test_stale_is_measured_from_commit_date(age_days, expected):
    """is_stale reads commit_date, the one date the row also stores."""
    from spyglass.utils.git_utils import SpyglassInstall

    inst = SpyglassInstall()
    inst.__dict__["install_path"] = "/home/user/spyglass"
    inst.__dict__["commit_date"] = date.today() - timedelta(days=age_days)

    assert inst.is_stale is expected


def test_stale_false_without_a_commit_date():
    """No date, no staleness claim."""
    from spyglass.utils.git_utils import SpyglassInstall

    inst = SpyglassInstall()
    inst.__dict__["install_path"] = None
    inst.__dict__["commit_date"] = None

    assert inst.is_stale is False


def test_git_stale_old_head(tmp_path, fake_clone, compute_install_info_fn):
    """A clean clone with a months-old HEAD is is_stale=True, not official."""
    fake_pkg = fake_clone(tmp_path)

    clean_proc = _proc("")  # no local changes
    old_proc = _proc("100000000")  # 1973, well past the threshold

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(
                    diff=clean_proc.stdout, log=old_proc.stdout
                ),
            ):
                result = compute_install_info_fn()

    assert result.has_local_changes is False
    assert result.is_stale is True
    assert result.is_official is False


def test_clean_current_clone_is_official(
    tmp_path, fake_clone, compute_install_info_fn
):
    """A clean, recently-fetched dev clone is official despite is_dev=True."""
    fake_pkg = fake_clone(tmp_path)

    clean_proc = _proc("")  # no local changes
    recent_proc = _proc("4102444800")  # year 2100, not stale

    with patch("spyglass.__version__", "0.5.6.dev42+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=_git_mock(
                    diff=clean_proc.stdout, log=recent_proc.stdout
                ),
            ):
                result = compute_install_info_fn()

    assert result.is_dev is True
    assert result.has_local_changes is False
    assert result.is_stale is False
    assert result.is_official is True  # being a few commits behind is fine


# ═════════════════════════════════════════════════════════════════════════════
#     SpyglassInstall.warn (no DB)
# ═════════════════════════════════════════════════════════════════════════════


def test_warn_official_install_silent(caplog):
    """An official install emits no warning."""
    info = {
        "is_official": True,
        "is_dev": False,
        "has_local_changes": False,
        "install_path": None,
        "commit_hash": None,
    }
    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert not any("repository" in r.message for r in caplog.records)


def test_warn_dirty_emits_warning(caplog, warn_dirty_days):
    """A dirty install (has_local_changes) emits the countdown warning."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False

    mock_env = MagicMock()
    mock_env.__and__ = MagicMock(return_value=mock_env)
    mock_env.__bool__ = MagicMock(return_value=False)

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with patch.object(_user_mod, "UserEnvironment", mock_env):
            with caplog.at_level("WARNING"):
                _warn_with(info)

    assert any("official commit" in r.message for r in caplog.records)
    assert any(f"{warn_dirty_days} days" in r.message for r in caplog.records)


_FORK_ONLY_INFO = {
    "is_official": False,
    "is_dev": True,
    "has_local_changes": True,
    "is_stale": False,
    "install_path": "/home/user/spyglass",
    "commit_hash": "abc1234",
    "_remotes": (
        "me\thttps://github.com/CBroz1/spyglass (fetch)\n"
        "me\thttps://github.com/CBroz1/spyglass (push)\n"
    ),
    "official_remote": None,
}


def test_warn_fork_only_names_the_remote_command(caplog):
    """A clone with no official remote is told how to add one.

    The countdown warning asks the user to fetch from the official repository,
    which a fork-only clone has no remote for.
    """
    with caplog.at_level("WARNING"):
        _warn_with(dict(_FORK_ONLY_INFO))

    assert any("git remote add upstream" in r.message for r in caplog.records)
    assert any("official commit" in r.message for r in caplog.records)


def test_warn_official_remote_omits_the_remote_command(caplog):
    """A clone that already has the official remote is not told to add it."""
    info = dict(_FORK_ONLY_INFO, official_remote="sg")

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert not any(
        "git remote add" in r.message for r in caplog.records
    )  # the fetch advice in the countdown is actionable here


def test_warn_unusable_git_omits_the_remote_command(caplog):
    """Git giving no answer is unknown, not a missing remote."""
    info = dict(_FORK_ONLY_INFO, _remotes=None)

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert not any("git remote add" in r.message for r in caplog.records)


_ADMIN_INFO = {
    "is_official": False,
    "is_dev": True,
    "has_local_changes": True,
    "is_stale": True,  # an admin's branch is behind master as well as edited
    "install_path": "/home/admin/spyglass",
    "commit_hash": "abc1234",
}


def _cache_admin_check(is_admin=True, days_ago=0):
    """Record an admin check in the marker, as `log_install` would."""
    from spyglass.utils.git_utils import spyglass_install

    checked = date.today() - timedelta(days=days_ago)
    spyglass_install.marker.write(
        is_admin=is_admin, admin_checked=checked.isoformat()
    )


def test_warn_cached_admin_is_exempt(caplog):
    """A cached admin check silences both messages.

    Admins run modified code deliberately, so neither the countdown nor the
    stale advisory is news to them.
    """
    _cache_admin_check()

    with caplog.at_level("WARNING"):
        _warn_with(_ADMIN_INFO)

    assert caplog.text == ""


def test_warn_admin_warned_until_the_check_is_cached(caplog):
    """An admin's first flagged session is warned.

    The exemption is a query, so the import-time warning can only honour it
    once an insert has cached the answer.
    """
    with caplog.at_level("WARNING"):
        _warn_with(_ADMIN_INFO)

    assert "not on an official commit" in caplog.text


def test_warn_cached_admin_check_expires(caplog, admin_trust_days):
    """An aged admin check stops exempting, so a demotion takes effect."""
    _cache_admin_check(days_ago=admin_trust_days + 1)

    with caplog.at_level("WARNING"):
        _warn_with(_ADMIN_INFO)

    assert "not on an official commit" in caplog.text


def test_warn_cached_non_admin_is_warned(caplog):
    """A cached negative is an answer, not a missing one."""
    _cache_admin_check(is_admin=False)

    with caplog.at_level("WARNING"):
        _warn_with(_ADMIN_INFO)

    assert "not on an official commit" in caplog.text


def test_warn_dev_but_clean_emits_stale_warning(caplog):
    """A clean-but-stale clone emits the softer 'older commit' advisory."""
    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": False,
        "is_stale": True,
        "install_path": "/home/user/spyglass",
        "commit_hash": "stale123",
    }

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert "older commit" in caplog.text
    assert "not on an official commit" not in caplog.text  # no countdown


def test_warn_touches_no_database(caplog):
    """Warning opens no connection and imports nothing from common.

    This is what the split buys: a bare `import spyglass` neither waits on a
    connection nor prompts for a password.
    """
    import datajoint as dj

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    with patch.object(dj, "conn", side_effect=AssertionError("connected!")):
        with caplog.at_level("WARNING"):
            _warn_with(info)

    assert "not on an official commit" in caplog.text


def test_warn_countdown_runs_from_marker(caplog, warn_dirty_days):
    """The countdown counts from when this clone was first seen dirty."""
    import datetime

    from spyglass.utils import git_utils

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    ten_days_ago = (
        datetime.date.today() - datetime.timedelta(days=10)
    ).isoformat()
    git_utils.spyglass_install.marker.write(
        commit="abc1234", first_seen=ten_days_ago
    )

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert f"within {warn_dirty_days - 10} days" in caplog.text


def test_warn_countdown_survives_a_new_commit(caplog, warn_dirty_days):
    """Committing while dirty does not restart the clock.

    A user who commits every day keeps one countdown, rather than a fresh 30
    days each time, and so reaches the escalation.
    """
    import datetime

    from spyglass.utils import git_utils

    ten_days_ago = (
        datetime.date.today() - datetime.timedelta(days=10)
    ).isoformat()
    git_utils.spyglass_install.marker.write(
        commit="old_commit", first_seen=ten_days_ago
    )

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "new_commit",  # moved on since first flagged
    }
    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert f"within {warn_dirty_days - 10} days" in caplog.text
    assert (
        git_utils.spyglass_install.marker.read()["first_seen"] == ten_days_ago
    )


def test_warn_first_sight_starts_the_clock(caplog, warn_dirty_days):
    """With no marker, the countdown is the full window and is recorded."""
    import datetime

    from spyglass.utils import git_utils

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert f"within {warn_dirty_days} days" in caplog.text
    marker = git_utils.spyglass_install.marker.read()
    assert marker["first_seen"] == datetime.date.today().isoformat()
    assert marker["commit"] == "abc1234"


def test_warn_clean_is_silent(caplog):
    """An official install warns about nothing."""
    info = {
        "is_official": True,
        "is_dev": False,
        "has_local_changes": False,
        "is_stale": False,
        "install_path": None,
        "commit_hash": None,
    }

    with caplog.at_level("WARNING"):
        _warn_with(info)

    assert caplog.text == ""


# ─── deferred logging: the database half ──────────────────────────────────────


def test_mixin_logs_install_once_per_session():
    """The first insert logs; later inserts, on any table, do not.

    The gate is ``BaseMixin._logged_install_mode``, bound on ``BaseMixin``
    itself: an instance attribute would shadow it and leave every other table
    unguarded.
    """
    import spyglass.common.common_user as _user_mod
    from spyglass.utils.mixins.base import BaseMixin

    class TableA(BaseMixin):
        _test_mode = False
        _logger = MagicMock()

    class TableB(BaseMixin):
        _test_mode = False
        _logger = MagicMock()

    calls = []
    BaseMixin._logged_install_mode = False
    try:
        with patch.object(
            _user_mod.UserEnvironment,
            "log_install",
            lambda self: calls.append(1),
        ):
            TableA()._log_install_once()
            TableB()._log_install_once()
            TableA()._log_install_once()
    finally:
        BaseMixin._logged_install_mode = True  # leave the session gated

    assert len(calls) == 1


def _patched_env(insert_return=None, row_date=None):
    """Patch UserEnvironment's collaborators, keeping the real log_install.

    The method under test lives on the table now, so the class itself must not
    be replaced, only what it calls out to.
    """
    import spyglass.common.common_user as _user_mod

    inserted = MagicMock(
        return_value=insert_return or {"env_id": "user_env_00"}
    )
    return (
        _user_mod.UserEnvironment,
        inserted,
        patch.multiple(
            _user_mod.UserEnvironment,
            insert_current_env=inserted,
            _row_first_seen=MagicMock(return_value=row_date),
        ),
    )


_DIRTY_INFO = {
    "is_official": False,
    "is_dev": True,
    "has_local_changes": True,
    "is_stale": False,
    "install_path": "/home/user/spyglass",
    "commit_hash": "abc1234",
}

_CLEAN_INFO = {
    "is_official": True,
    "is_dev": False,
    "has_local_changes": False,
    "is_stale": False,
    "install_path": None,
    "commit_hash": None,
}


def test_log_install_writes_the_record():
    """A dirty install is recorded on the session's first write."""
    import spyglass.common.common_lab as _lab_mod

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, inserted, patches = _patched_env()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    inserted.assert_called_once()


def test_log_install_adopts_the_rows_date():
    """The marker's countdown defers to the logged row's own date.

    The warning reads local state and the email reads the database, so without
    this they drift: clearing `temp_dir` restarts the local clock while the
    email keeps counting, and a user can be emailed the same day they are told
    they have the full window left.
    """
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils import git_utils

    git_utils.spyglass_install.marker.clear()  # as a sweep would
    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, _, patches = _patched_env(row_date="2026-10-01")

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    marker = git_utils.spyglass_install.marker.read()
    assert marker["first_seen"] == "2026-10-01"


def test_log_install_keeps_local_date_without_a_row():
    """With nothing readable in the database, the local date stands.

    The first warning of an episode fires before any row exists, so the marker
    has to be able to start the clock on its own.
    """
    import datetime

    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils import git_utils

    today = datetime.date.today().isoformat()
    git_utils.spyglass_install.marker.write(commit="abc1234", first_seen=today)
    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, _, patches = _patched_env(row_date=None)

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    marker = git_utils.spyglass_install.marker.read()
    assert marker["first_seen"] == today


def test_log_install_skips_second_call_same_day():
    """A dirty state already recorded today needs no second write."""
    import datetime

    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils import git_utils

    git_utils.spyglass_install.marker.write(
        commit="abc1234",
        first_seen="2026-01-01",
        last_logged=datetime.date.today().isoformat(),
    )
    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, inserted, patches = _patched_env()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    inserted.assert_not_called()


def test_log_install_admin_not_logged():
    """An admin is not recorded, and so never emailed."""
    import spyglass.common.common_lab as _lab_mod

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = True
    tbl, inserted, patches = _patched_env()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    inserted.assert_not_called()


def test_log_install_caches_the_admin_check():
    """The admin answer is left in the marker for the warning to read."""
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils.git_utils import spyglass_install

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = True
    tbl, _, patches = _patched_env()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

        assert spyglass_install.cached_is_admin is True
    assert spyglass_install.marker.read()["admin_checked"] == (
        date.today().isoformat()
    )


def test_log_install_trusts_the_cached_admin_check():
    """A cached exemption spares the query on later sessions."""
    import spyglass.common.common_lab as _lab_mod

    mock_lab = MagicMock()
    tbl, inserted, patches = _patched_env()
    _cache_admin_check()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    mock_lab.assert_not_called()
    inserted.assert_not_called()


def test_log_install_clean_admin_clears_the_marker():
    """An admin who resolves a dirty state returns to the fast path.

    Nothing was recorded for them, so there is no resolution to log. The
    marker is cleared so its cached exemption does not outlive the episode.
    """
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils.git_utils import spyglass_install

    mock_lab = MagicMock()
    tbl, inserted, patches = _patched_env()
    _cache_admin_check()

    with _as_install(**_CLEAN_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    inserted.assert_not_called()
    assert not spyglass_install.marker.exists()


def test_log_install_clean_no_marker_is_noop():
    """A clean install never flagged here queries nothing at all."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod

    mock_lab = MagicMock()
    with _as_install(**_CLEAN_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            _user_mod.UserEnvironment().log_install()

    mock_lab.assert_not_called()


def test_log_install_clean_with_marker_records_resolution():
    """Clean with a marker logs the full row and clears the marker."""
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils import git_utils

    git_utils.spyglass_install.marker.write(
        commit="abc1234", first_seen="2026-01-01"
    )
    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, inserted, patches = _patched_env()

    with _as_install(**_CLEAN_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

    inserted.assert_called_once()
    assert git_utils.spyglass_install.marker.exists() is False


def test_log_install_stale_keeps_the_daily_gate():
    """A stale clean install logs once a day, not once a session.

    The marker was cleared on every clean path, stale included, so
    `last_logged` was never written and each session re-ran the admin query,
    the environment read and two more queries, against a docstring promising
    "at most once a day". The cached admin answer went with it.
    """
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils import git_utils

    stale = dict(_CLEAN_INFO, is_stale=True, is_official=False)
    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False
    tbl, inserted, patches = _patched_env()

    with _as_install(**stale):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()  # first session
                marker = git_utils.spyglass_install.marker.read()
                assert marker["last_logged"] == date.today().isoformat()
                assert marker["is_admin"] is False  # answer kept, not lost

                tbl().log_install()  # second session, same day

    inserted.assert_called_once()  # gated, not repeated


def test_log_install_does_not_cache_a_failed_admin_check():
    """A check that could not run is not stored as "not admin".

    Storing the failure left an admin warned, and logged rows for them, for
    WARN_ADMIN_TRUST_DAYS, against the comment that said it would ask again
    next session.
    """
    import spyglass.common.common_lab as _lab_mod
    from spyglass.utils.git_utils import spyglass_install

    mock_lab = MagicMock(side_effect=RuntimeError("database down"))
    tbl, inserted, patches = _patched_env()

    with _as_install(**_DIRTY_INFO):
        with patch.object(_lab_mod, "LabMember", mock_lab):
            with patches:
                tbl().log_install()

        assert spyglass_install.cached_is_admin is None  # nothing cached
    assert "is_admin" not in spyglass_install.marker.read()
    inserted.assert_called_once()  # treated as non-admin for this session


def test_warn_once_is_idempotent(caplog):
    """warn_once warns on the first call only.

    No retry path: the warning needs no database, so there is no "not ready
    yet" state for a later call to recover from.
    """
    from spyglass.utils import git_utils

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    git_utils.spyglass_install.warned = False
    try:
        with _as_install(**info):
            with caplog.at_level("WARNING"):
                git_utils.spyglass_install.warn_once()
                first = caplog.text.count("not on an official commit")
                git_utils.spyglass_install.warn_once()
                second = caplog.text.count("not on an official commit")
    finally:
        # leave the session as the import found it
        git_utils.spyglass_install.warned = True

    assert first == 1
    assert second == 1


# ═════════════════════════════════════════════════════════════════════════════
#     UserEnvironment integration (requires DB)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def clean_install_info():
    """Mock install info for a clean/official install."""
    return {
        "is_dev": False,
        "commit_hash": None,
        "install_path": None,
        "has_local_changes": False,
        "is_stale": False,
        "is_official": True,
        "commit_date": None,  # a release install has no clone to date
    }


@pytest.fixture(scope="module")
def dirty_install_info():
    """Mock install info for a dirty/editable install (local edits)."""
    return {
        "is_dev": True,
        "commit_hash": "abcd1234",
        "install_path": "/home/user/spyglass",
        "has_local_changes": True,
        "is_stale": False,
        "is_official": False,
        "commit_date": date.today() - timedelta(days=5),
    }


@pytest.fixture(scope="module")
def stale_install_info():
    """Mock install info for a clean-but-stale install (months behind)."""
    return {
        "is_dev": True,
        "commit_hash": "stale123",
        "install_path": "/home/user/spyglass",
        "has_local_changes": False,
        "is_stale": True,
        "is_official": False,
        "commit_date": date.today() - timedelta(days=200),
    }


def test_clean_install_null_fields(user_env_tbl, clean_install_info):
    """A clean install stores NULL for dirty_path and spyglass_commit."""
    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    row = (user_env_tbl & f'env_id="{env_id}"').fetch1()
    assert row["dirty_path"] is None
    assert row["spyglass_commit"] is None


def test_dirty_path_populated(user_env_tbl, dirty_install_info):
    """A dirty install stores dirty_path and spyglass_commit."""
    with _as_install(**dirty_install_info):
        # Clear cached properties so insert_current_env re-runs
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    row = (user_env_tbl & f'env_id="{env_id}"').fetch1()
    assert row["dirty_path"] == dirty_install_info["install_path"]
    assert row["spyglass_commit"] == dirty_install_info["commit_hash"]


def test_dirty_hash_differs_from_clean(
    user_env_tbl, clean_install_info, dirty_install_info
):
    """Same conda env + different spyglass_commit → different env_hash."""
    from spyglass.utils.env_cache import _env_cache

    _ = _env_cache.get(with_pip=True)

    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id"):
            user_env_tbl.__dict__.pop(attr, None)
        clean_hash = user_env_tbl.env_hash

    with _as_install(**dirty_install_info):
        for attr in ("env", "env_hash", "matching_env_id"):
            user_env_tbl.__dict__.pop(attr, None)
        dirty_hash = user_env_tbl.env_hash

    assert clean_hash != dirty_hash


def test_dirty_idempotent(user_env_tbl, dirty_install_info):
    """Calling insert_current_env twice with the same dirty commit is idempotent."""
    with _as_install(**dirty_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        first = user_env_tbl.insert_current_env()

        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        second = user_env_tbl.insert_current_env()

    assert first["env_id"] == second["env_id"]
    dirty_rows = user_env_tbl & "dirty_path IS NOT NULL"
    assert len(dirty_rows) == 1


def test_dirty_installs_query(user_env_tbl, dirty_install_info):
    """dirty_installs() returns a DataJoint expression with days_since_warn."""
    with _as_install(**dirty_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        user_env_tbl.insert_current_env()

    result = user_env_tbl.dirty_installs()
    assert hasattr(
        result, "fetch"
    ), "dirty_installs() should return a DJ expression"

    rows = result.fetch(as_dict=True)
    assert len(rows) >= 1
    assert "days_since_warn" in rows[0]
    assert isinstance(rows[0]["days_since_warn"], int)


@pytest.fixture(scope="module")
def same_commit_pair():
    """Clean and dirty install info differing *only* in local edits.

    The clean/dirty fixtures elsewhere also differ in ``commit_hash``, so a
    test using them passes on the commit difference and says nothing about
    whether dirty state itself is recorded.
    """
    shared = dict(is_dev=True, commit_hash="cafe1234", is_stale=False)
    return (
        dict(
            **shared,
            install_path=None,
            has_local_changes=False,
            is_official=True,
        ),
        dict(
            **shared,
            install_path="/home/user/spyglass",
            has_local_changes=True,
            is_official=False,
        ),
    )


def _log_as(user_env_tbl, info):
    """Clear cached state and log the environment under a given install info."""
    with _as_install(**info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        return user_env_tbl.insert_current_env()


def test_clean_then_dirty_at_one_commit(user_env_tbl, same_commit_pair):
    """Going dirty at an already-logged commit records a distinct row.

    The clean and dirty states hash identically, so a table keyed on
    ``env_hash`` alone could not represent the change: the dirty episode was
    silently dropped and the user was never emailed.
    """
    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    clean_id = _log_as(user_env_tbl, clean)["env_id"]
    dirty_id = _log_as(user_env_tbl, dirty)["env_id"]

    assert clean_id != dirty_id
    clean_row = (user_env_tbl & f'env_id="{clean_id}"').fetch1()
    dirty_row = (user_env_tbl & f'env_id="{dirty_id}"').fetch1()

    assert clean_row["env_hash"] is not None and clean_row["env"] is not None
    assert dirty_row["env_hash"] is None and dirty_row["env"] is None
    assert dirty_row["spyglass_commit"] == clean_row["spyglass_commit"]
    assert dirty_id in list(user_env_tbl.dirty_installs().fetch("env_id"))


def test_this_env_routes_a_dirty_install_off_a_clean_row(
    user_env_tbl, same_commit_pair
):
    """`this_env` must not hand a dirty session a clean row's id.

    The hash covers the commit, not the edits on top of it, so a clean row
    logged at this commit matched and was returned: the session recorded no
    dirty state and its files were attributed to a reproducible environment.
    """
    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    clean_id = _log_as(user_env_tbl, clean)["env_id"]

    with _as_install(**dirty):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        key = user_env_tbl.this_env

    assert key["env_id"] != clean_id
    row = (user_env_tbl & f"env_id=\"{key['env_id']}\"").fetch1()
    assert row["env_hash"] is None  # a partial row, as a dirty install gets


def test_dirty_then_clean_at_one_commit_resolves(
    user_env_tbl, same_commit_pair
):
    """Discarding edits at the same commit clears the escalation.

    Without a recordable resolution the user kept receiving "not on an official
    commit" email after they had already fixed it.
    """
    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    dirty_id = _log_as(user_env_tbl, dirty)["env_id"]
    assert dirty_id in list(user_env_tbl.dirty_installs().fetch("env_id"))

    _log_as(user_env_tbl, clean)
    assert list(user_env_tbl.dirty_installs().fetch("env_id")) == []

    import io

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        user_env_tbl.write_dirty_notifications()
    assert buf.getvalue() == ""


def test_dirty_path_runs_no_subprocess(user_env_tbl, same_commit_pair):
    """A dirty install costs no conda export, no pip freeze and no hashing."""
    _, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    with patch("spyglass.utils.env_cache.sub_run") as mock_run:
        result = _log_as(user_env_tbl, dirty)

    mock_run.assert_not_called()
    assert result["env_id"] is not None


def test_dirty_episode_logged_once(user_env_tbl, same_commit_pair):
    """A second session refreshes last_seen rather than adding a row."""
    _, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    first = _log_as(user_env_tbl, dirty)["env_id"]
    before = (user_env_tbl & f'env_id="{first}"').fetch1("last_seen")

    second = _log_as(user_env_tbl, dirty)["env_id"]
    row = (user_env_tbl & f'env_id="{first}"').fetch1()

    assert second == first
    assert len(user_env_tbl & "env_hash IS NULL") == 1
    assert row["last_seen"] >= before  # refreshed
    assert row["timestamp"] <= row["last_seen"]  # first appearance held


def test_dirty_episode_survives_a_dependency_change(
    user_env_tbl, same_commit_pair
):
    """A dependency update does not restart the 30-day clock.

    The partial row holds no environment, so there is nothing for a
    dependency change to alter, and the dedupe matches on the user's
    unresolved row rather than on a hash. Day 0 therefore survives both a
    dependency update and a new commit.
    """
    _, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    first = _log_as(user_env_tbl, dirty)["env_id"]
    before = (user_env_tbl & f'env_id="{first}"').fetch1("timestamp")

    moved_on = dict(dirty, commit_hash="deadbeef")
    second = _log_as(user_env_tbl, moved_on)["env_id"]
    row = (user_env_tbl & f'env_id="{second}"').fetch1()

    assert second == first  # same episode
    assert row["timestamp"] == before  # same day 0
    assert row["spyglass_commit"] == "deadbeef"  # but the commit is current


def test_dirty_episode_takes_the_oldest_row(user_env_tbl, same_commit_pair):
    """With two unresolved rows, the countdown uses the older.

    Concurrent sessions can both find no row and both insert, since MySQL
    permits repeated nulls under the unique index. `fetch` has no guaranteed
    order, so the oldest has to be chosen explicitly.
    """
    from datetime import datetime, timedelta

    import datajoint as dj

    _, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    # The dedupe is scoped to this user's env_ids, so use the real prefix
    user = dj.config.get("database.user", "")
    older_id, newer_id = f"{user}_race_00", f"{user}_race_01"

    older = datetime.now() - timedelta(days=20)
    newer = datetime.now() - timedelta(days=2)
    user_env_tbl.insert(
        [
            {
                "env_id": newer_id,  # inserted second, newer
                "env_hash": None,
                "env": None,
                "has_editable": None,
                "dirty_path": "/home/user/spyglass",
                "spyglass_commit": "cafe1234",
                "timestamp": newer,
                "last_seen": newer,
            },
            {
                "env_id": older_id,
                "env_hash": None,
                "env": None,
                "has_editable": None,
                "dirty_path": "/home/user/spyglass",
                "spyglass_commit": "cafe1234",
                "timestamp": older,
                "last_seen": older,
            },
        ],
        skip_duplicates=True,
    )

    try:
        with patch.object(type(user_env_tbl), "_increment_id") as never:
            chosen = _log_as(user_env_tbl, dirty)["env_id"]
            never.assert_not_called()  # reused, did not insert
        assert chosen == older_id  # the older episode
    finally:
        for eid in (older_id, newer_id):
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def test_user_scoping_does_not_reach_another_user(
    user_env_tbl, same_commit_pair
):
    """A username that prefixes another's must not touch their dirty row.

    `_` is a single-character wildcard in LIKE, so `root_%` also matched
    `rootx_base_00`: this user's session adopted the other user's episode,
    refreshed it, added no row of its own, and a later clean session resolved
    it, restarting the other user's countdown and delaying their email.
    """
    from datetime import datetime, timedelta

    import datajoint as dj

    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    # One character longer than this user, so the unescaped pattern matches.
    user = dj.config.get("database.user", "")
    other_id = f"{user}x_base_00"
    stamped = datetime.now() - timedelta(days=20)
    user_env_tbl.insert1(
        {
            "env_id": other_id,
            "env_hash": None,
            "env": None,
            "has_editable": None,
            "dirty_path": "/home/other/spyglass",
            "spyglass_commit": "0ther123",
            "timestamp": stamped,
            "last_seen": stamped,
        }
    )

    mine = None
    try:
        # MySQL rounds a `timestamp` to the nearest second; `replace` truncates
        before = (user_env_tbl & f'env_id="{other_id}"').fetch1("last_seen")

        mine = _log_as(user_env_tbl, dirty)["env_id"]
        assert mine != other_id  # its own episode, not the other user's

        _log_as(user_env_tbl, clean)  # resolution
        other = (user_env_tbl & f'env_id="{other_id}"').fetch1()

        assert other["resolved_at"] is None  # not this user's to resolve
        assert other["spyglass_commit"] == "0ther123"  # never refreshed
        assert other["last_seen"] == before
        assert other_id in list(user_env_tbl.dirty_installs().fetch("env_id"))
    finally:
        for eid in filter(None, (other_id, mine)):
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def test_episode_after_resolution_starts_fresh(user_env_tbl, same_commit_pair):
    """A new dirty episode gets its own first-appeared date.

    Deduping against every past dirty row matches the resolved one and
    inherits its months-old timestamp, emailing the user on day one.
    """
    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)

    first = _log_as(user_env_tbl, dirty)["env_id"]
    _log_as(user_env_tbl, clean)  # resolution
    second = _log_as(user_env_tbl, dirty)["env_id"]

    assert second != first
    assert len(user_env_tbl & "env_hash IS NULL") == 2
    assert list(user_env_tbl.dirty_installs().fetch("env_id")) == [second]


@pytest.mark.parametrize(
    "info_name", ["dirty_install_info", "stale_install_info"]
)
def test_insert_current_env_survives_its_own_hook(
    user_env_tbl, request, info_name
):
    """A direct call logs one row, though its own insert re-enters the hook.

    `insert1` calls `BaseMixin.insert`, which logs the install, which calls
    this method again. The inner call inserted first, so the outer one landed
    on it: a primary-key `DuplicateError` for a flagged install taking the
    default id, an `env_hash` collision for a clean-but-stale one, and two
    partial rows for one episode when the ids differed. `Recompute.md` teaches
    this call, so it is reached directly and not only through the hook.
    """
    from spyglass.utils.mixins.base import BaseMixin

    info = request.getfixturevalue(info_name)
    user_env_tbl.delete(safemode=False)
    _cache_admin_check(is_admin=False)  # the test user is an admin

    # test_mode gates the hook off, so ungate this one instance
    user_env_tbl.__dict__["_test_mode"] = False
    BaseMixin._logged_install_mode = False
    try:
        with _as_install(**info):
            for attr in ("env", "env_hash", "matching_env_id", "this_env"):
                user_env_tbl.__dict__.pop(attr, None)
            key = user_env_tbl.insert_current_env()
    finally:
        BaseMixin._logged_install_mode = True  # leave the session gated
        user_env_tbl.__dict__.pop("_test_mode", None)

    assert key["env_id"] is not None
    assert len(user_env_tbl) == 1


def test_resolution_survives_a_failed_capture(user_env_tbl, same_commit_pair):
    """A clean session that cannot capture its env still stops the clock.

    Resolution ran after the environment check, so a `pip freeze` line the
    parser rejects left the row open, and `log_install` then cleared the
    marker so no later session tried again. The user kept receiving email
    after discarding their edits.
    """
    clean, dirty = same_commit_pair
    user_env_tbl.delete(safemode=False)
    dirty_id = _log_as(user_env_tbl, dirty)["env_id"]

    with _as_install(**clean):
        for attr in ("env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.__dict__["env"] = {}  # capture failed
        key = user_env_tbl.insert_current_env()
    user_env_tbl.__dict__.pop("env", None)

    assert key == {"env_id": None}
    assert (user_env_tbl & f'env_id="{dirty_id}"').fetch1("resolved_at")
    assert not user_env_tbl.unresolved_dirty()


def test_resolve_survives_a_table_without_resolved_at(user_env_tbl):
    """Resolution must not propagate a pre-alter restriction error.

    It now runs on the way to every full row, including on an instance whose
    table predates the release-note `alter()`, where restricting on
    `resolved_at` raises and would take the caller's own populate with it.
    """
    import datajoint as dj

    with patch.object(
        type(user_env_tbl),
        "unresolved_dirty",
        side_effect=dj.DataJointError(
            "Attribute `resolved_at` is not found in query"
        ),
    ):
        user_env_tbl._mark_dirty_resolved()  # must not raise


def test_stale_install_is_not_dirty(user_env_tbl, stale_install_info):
    """A clean-but-stale install gets a full row and is never escalated.

    An older official commit is reproducible, so its environment is worth
    storing and the user is owed the soft "consider updating" advisory rather
    than the 30-day dirty-install email.
    """
    with _as_install(**stale_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    row = (user_env_tbl & f'env_id="{env_id}"').fetch1()
    assert row["dirty_path"] is None
    assert row["env_hash"] is not None  # a full row
    assert row["env"] is not None
    assert row["spyglass_commit"] == stale_install_info["commit_hash"]
    assert env_id not in list(user_env_tbl.dirty_installs().fetch("env_id"))


def test_clean_install_not_in_dirty_installs(user_env_tbl, clean_install_info):
    """A clean install is NOT surfaced by dirty_installs()."""
    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    dirty_ids = user_env_tbl.dirty_installs().fetch("env_id")
    assert env_id not in dirty_ids


def test_dirty_installs_excludes_inactive(user_env_tbl):
    """A dirty row not seen within 2x the warn window is dropped.

    The window tests ``last_seen``, not ``timestamp``: both rows below first
    went dirty at the same long-ago moment, so keying on first-appearance would
    drop them both and stop emailing a user who is dirty and still working.
    """
    from datetime import datetime, timedelta

    from spyglass.utils.git_utils import WARN_DIRTY_ENV_DAYS

    user_env_tbl.delete(safemode=False)
    old_ts = datetime.now() - timedelta(days=2 * WARN_DIRTY_ENV_DAYS + 5)
    active_ts = datetime.now() - timedelta(days=WARN_DIRTY_ENV_DAYS + 5)

    # Partial rows: env_hash IS NULL marks a dirty install
    user_env_tbl.insert(
        [
            {
                "env_id": "ghost_envstale_00",
                "env_hash": None,
                "env": None,
                "has_editable": None,
                "dirty_path": "/old",
                "spyglass_commit": "stale01",
                "timestamp": old_ts,
                "last_seen": old_ts,
            },
            {
                "env_id": "ghost_envactive_00",
                "env_hash": None,
                "env": None,
                "has_editable": None,
                "dirty_path": "/recent",
                "spyglass_commit": "active1",
                "timestamp": old_ts,  # same first-appearance as the other
                "last_seen": active_ts,  # but seen recently
            },
        ],
        skip_duplicates=True,
    )
    try:
        dirty_ids = list(user_env_tbl.dirty_installs().fetch("env_id"))
        assert "ghost_envstale_00" not in dirty_ids  # aged out (inactive)
        assert "ghost_envactive_00" in dirty_ids  # seen within 2x window
    finally:
        for eid in ("ghost_envstale_00", "ghost_envactive_00"):
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def _server_today(user_env_tbl):
    """Today by the database's clock, which `DATEDIFF` measures against."""
    return user_env_tbl.connection.query("SELECT CURDATE()").fetchone()[0]


def _insert_dated_rows(user_env_tbl, rows):
    """Insert ghost rows with explicit commit_date/last_seen, and clean up."""
    user_env_tbl.insert(rows, skip_duplicates=True)
    return [r["env_id"] for r in rows]


def test_stale_installs_surfaces_old_full_rows(user_env_tbl):
    """A reproducible row whose code is months old is surfaced, with its age.

    ``days_stale`` is measured from ``commit_date``, so it reports the age of
    the code and not of the row: both rows below were logged today.
    """
    from spyglass.utils.git_utils import WARN_STALE_DAYS

    # Server dates: `days_stale` is a server-side DATEDIFF, and the two clocks
    # can disagree on today
    server_today = _server_today(user_env_tbl)
    old = server_today - timedelta(days=WARN_STALE_DAYS + 40)
    fresh = server_today - timedelta(days=5)
    rows = [
        {
            "env_id": "ghost_stale_00",
            "env_hash": "a" * 32,
            "env": {"dependencies": []},
            "has_editable": False,
            "spyglass_commit": "old0001",
            "commit_date": old,
        },
        {
            "env_id": "ghost_fresh_00",
            "env_hash": "b" * 32,
            "env": {"dependencies": []},
            "has_editable": False,
            "spyglass_commit": "new0001",
            "commit_date": fresh,
        },
    ]
    ids = _insert_dated_rows(user_env_tbl, rows)
    try:
        got = user_env_tbl.stale_installs().fetch("env_id", "days_stale")
        surfaced = dict(zip(*got))

        assert "ghost_stale_00" in surfaced
        assert "ghost_fresh_00" not in surfaced
        assert surfaced["ghost_stale_00"] == WARN_STALE_DAYS + 40
    finally:
        for eid in ids:
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def test_stale_installs_excludes_dirty_and_undated(user_env_tbl):
    """A dirty row and an undated row are both out of scope.

    A dirty install runs the shorter 30-day clock through ``dirty_installs``,
    and a release install stores no date to measure.
    """
    from spyglass.utils.git_utils import WARN_STALE_DAYS

    old = date.today() - timedelta(days=WARN_STALE_DAYS + 40)
    rows = [
        {  # dirty: env_hash null, dated and old
            "env_id": "ghost_dirtydated_00",
            "env_hash": None,
            "env": None,
            "has_editable": None,
            "dirty_path": "/home/user/spyglass",
            "spyglass_commit": "dirty01",
            "commit_date": old,
        },
        {  # release: reproducible, but no clone to date
            "env_id": "ghost_undated_00",
            "env_hash": "c" * 32,
            "env": {"dependencies": []},
            "has_editable": False,
            "dirty_path": None,
            "spyglass_commit": None,
            "commit_date": None,
        },
    ]
    ids = _insert_dated_rows(user_env_tbl, rows)
    try:
        stale_ids = list(user_env_tbl.stale_installs().fetch("env_id"))

        assert "ghost_dirtydated_00" not in stale_ids
        assert "ghost_undated_00" not in stale_ids
    finally:
        for eid in ids:
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def test_stale_installs_excludes_abandoned_clone(user_env_tbl):
    """An old clone nobody has used in two stale windows ages out."""
    from spyglass.utils.git_utils import WARN_STALE_DAYS

    old = date.today() - timedelta(days=WARN_STALE_DAYS + 40)
    gone = datetime.now() - timedelta(days=2 * WARN_STALE_DAYS + 5)
    rows = [
        {
            "env_id": "ghost_abandoned_00",
            "env_hash": "d" * 32,
            "env": {"dependencies": []},
            "has_editable": False,
            "spyglass_commit": "aband01",
            "commit_date": old,
            "last_seen": gone,
        }
    ]
    ids = _insert_dated_rows(user_env_tbl, rows)
    try:
        stale_ids = list(user_env_tbl.stale_installs().fetch("env_id"))

        assert "ghost_abandoned_00" not in stale_ids
    finally:
        for eid in ids:
            (user_env_tbl & f'env_id="{eid}"').delete(safemode=False)


def test_stale_row_in_daily_use_keeps_its_place(user_env_tbl):
    """Using a full row refreshes `last_seen`, so it does not age out.

    Only the dirty path wrote `last_seen`, so a full row kept its insert time
    however often it was used, and `stale_installs` dropped exactly the users
    who had been stale the longest.
    """
    from spyglass.utils.git_utils import WARN_STALE_DAYS

    user_env_tbl.delete(safemode=False)
    old_commit = date.today() - timedelta(days=WARN_STALE_DAYS + 40)
    long_ago = datetime.now() - timedelta(days=2 * WARN_STALE_DAYS + 5)

    clean = dict(_CLEAN_INFO, is_stale=True, is_official=False)
    with _as_install(**clean):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        live_hash = user_env_tbl.env_hash  # so the session matches this row

        user_env_tbl.insert1(
            {
                "env_id": "ghost_daily_00",
                "env_hash": live_hash,
                "env": {"dependencies": []},
                "has_editable": False,
                "spyglass_commit": "daily01",
                "commit_date": old_commit,
                "timestamp": long_ago,
                "last_seen": long_ago,
            }
        )
        try:
            # MySQL rounds a `timestamp`; a truncated expectation is a coin flip
            before = (user_env_tbl & 'env_id="ghost_daily_00"').fetch1(
                "timestamp", "last_seen"
            )
            before_ts, before_seen = before

            # Aged out: unused for two stale windows, by `last_seen`.
            assert "ghost_daily_00" not in list(
                user_env_tbl.stale_installs().fetch("env_id")
            )

            user_env_tbl.__dict__.pop("matching_env_id", None)
            key = user_env_tbl.insert_current_env()
            row = (user_env_tbl & 'env_id="ghost_daily_00"').fetch1()

            assert key == {"env_id": "ghost_daily_00"}  # matched, not inserted
            assert row["last_seen"] > before_seen  # refreshed
            assert row["timestamp"] == before_ts  # the clock stays put
            assert "ghost_daily_00" in list(
                user_env_tbl.stale_installs().fetch("env_id")
            )
        finally:
            (user_env_tbl & 'env_id="ghost_daily_00"').delete(safemode=False)


def test_stale_row_stores_its_commit_date(user_env_tbl, stale_install_info):
    """A full row keeps the date staleness is measured from."""
    with _as_install(**stale_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    row = (user_env_tbl & f'env_id="{result["env_id"]}"').fetch1()
    assert row["commit_date"] == stale_install_info["commit_date"]


def test_dirty_row_stores_its_commit_date(user_env_tbl, dirty_install_info):
    """A partial row stores the date too, though it is not what escalates it."""
    with _as_install(**dirty_install_info):
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    row = (user_env_tbl & f'env_id="{result["env_id"]}"').fetch1()
    assert row["env_hash"] is None  # a partial row
    assert row["commit_date"] == dirty_install_info["commit_date"]


def test_has_matching_env_same_hash(user_env_tbl, clean_install_info):
    """has_matching_env returns True when the stored hash matches current env.

    Patched clean: a dirty install logs a partial row with no environment, and
    the test suite itself usually runs from a working tree with local edits.
    """
    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        env_id = user_env_tbl.this_env["env_id"]
        assert user_env_tbl.has_matching_env(env_id=env_id) is True


def test_has_matching_env_missing_id(user_env_tbl):
    """has_matching_env returns False when env_id is not in the table."""
    assert (
        user_env_tbl.has_matching_env(env_id="nonexistent_env_xyz_999") is False
    )


def test_get_dep_version_returns_dict(user_env_tbl):
    """get_dep_version returns {env_id: version_str} for a known package."""
    for attr in ("env", "env_hash", "matching_env_id", "this_env"):
        user_env_tbl.__dict__.pop(attr, None)
    env_id = user_env_tbl.this_env["env_id"]
    result = user_env_tbl.get_dep_version("python", env_id)
    assert isinstance(result, dict)
    assert env_id in result


def test_get_dep_version_missing_package(user_env_tbl):
    """get_dep_version returns empty string for an absent package."""
    for attr in ("env", "env_hash", "matching_env_id", "this_env"):
        user_env_tbl.__dict__.pop(attr, None)
    env_id = user_env_tbl.this_env["env_id"]
    result = user_env_tbl.get_dep_version("nonexistent_pkg_xyz_abc", env_id)
    assert result[env_id] == ""


def test_write_env_yaml(user_env_tbl, tmp_path, clean_install_info):
    """write_env_yaml writes a YAML file named after the env_id.

    Patched clean, as above: a partial row has no environment to write.
    """
    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        env_id = user_env_tbl.this_env["env_id"]
    user_env_tbl.write_env_yaml(env_id=env_id, dest_path=str(tmp_path))

    yaml_file = tmp_path / f"{env_id}.yaml"
    assert yaml_file.exists()
    data = yaml.safe_load(yaml_file.read_text())
    assert data.get("name") == env_id
    assert "dependencies" in data


def test_write_env_yaml_missing_id(user_env_tbl, tmp_path):
    """write_env_yaml logs error and returns when env_id is absent."""
    user_env_tbl.write_env_yaml(
        env_id="nonexistent_env_xyz_abc", dest_path=str(tmp_path)
    )
    assert not (tmp_path / "nonexistent_env_xyz_abc.yaml").exists()


def test_parse_env_dict_invalid_input(user_env_tbl):
    """parse_env_dict returns empty dict for non-dict input."""
    assert user_env_tbl.parse_env_dict("not a dict") == {}


def test_has_matching_env_mismatch(user_env_tbl):
    """has_matching_env returns False and logs diff when stored env differs."""
    fake_env_id = "test_mismatch_env_99"
    fake_hash = "aabbccddeeff00112233445566778899"
    fake_env = {"channels": ["fake"], "dependencies": ["fakepkg=1.0=py"]}

    user_env_tbl.insert1(
        {
            "env_id": fake_env_id,
            "env_hash": fake_hash,
            "env": fake_env,
            "has_editable": False,
        },
        skip_duplicates=True,
    )
    try:
        for attr in ("env", "env_hash"):
            user_env_tbl.__dict__.pop(attr, None)
        result = user_env_tbl.has_matching_env(env_id=fake_env_id)
    finally:
        (user_env_tbl & f'env_id="{fake_env_id}"').delete(safemode=False)

    assert result is False


def test_get_dep_version_multi_row(user_env_tbl, dirty_install_info):
    """get_dep_version iterates all rows when env_id is omitted and len > 1."""
    for attr in ("env", "env_hash", "matching_env_id", "this_env"):
        user_env_tbl.__dict__.pop(attr, None)
    # Ensure at least one clean row exists
    user_env_tbl.this_env

    # Insert a second row with a different hash
    with _as_install(**dirty_install_info):
        for attr in ("env", "env_hash", "matching_env_id"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.insert_current_env()

    if len(user_env_tbl) > 1:
        result = user_env_tbl.get_dep_version("python")
        assert isinstance(result, dict)
        assert len(result) >= 1


def test_write_dirty_notifications_no_overdue(user_env_tbl):
    """write_dirty_notifications returns early when no overdue rows exist."""
    from spyglass.utils.git_utils import WARN_DIRTY_ENV_DAYS

    # Patch dirty_installs to return a query that yields no overdue rows
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=[])
    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        user_env_tbl.write_dirty_notifications()  # should return silently


def test_write_dirty_notifications_writes_tsv(user_env_tbl, tmp_path):
    """Overdue rows are written as TSV; the longest matching prefix wins."""
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "alice_env_00",
            "days_since_warn": 45,
            "spyglass_commit": "aaa1111",
            "dirty_path": "/home/alice/spyglass",
        },
        {
            "env_id": "alice_bob_env_00",
            "days_since_warn": 31,
            "spyglass_commit": "bbb2222",
            "dirty_path": "/home/alice_bob/sg",
        },
        {
            "env_id": "carol_env_00",  # below threshold, filtered out
            "days_since_warn": 10,
            "spyglass_commit": "ccc3333",
            "dirty_path": "/home/carol/sg",
        },
    ]
    member_rows = [
        {
            "datajoint_user_name": "alice",
            "google_user_name": "alice@lab.org",
            "admin": 0,
        },
        {
            "datajoint_user_name": "alice_bob",
            "google_user_name": "ab@lab.org",
            "admin": 0,
        },
        {
            "datajoint_user_name": "carol",
            "google_user_name": "carol@lab.org",
            "admin": 0,
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo,
            "fetch",
            return_value=member_rows,
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    lines = [ln for ln in out_file.read_text().splitlines() if ln]
    assert len(lines) == 2  # carol filtered out (10 < 30 days)
    assert "alice@lab.org\t45\taaa1111\t/home/alice/spyglass" in lines
    # Longest-prefix match: alice_bob_env_00 -> alice_bob, NOT alice.
    assert "ab@lab.org\t31\tbbb2222\t/home/alice_bob/sg" in lines
    assert not any("carol" in ln for ln in lines)


def test_write_dirty_notifications_one_line_per_user(user_env_tbl, tmp_path):
    """Several overdue rows for one user collapse to a single line.

    `run_jobs.sh` sends one email per line on each cron run, so a user with a
    row per commit was emailed once per row, every day from day 30 to day 60,
    with the admin copied on each.
    """
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "alice_env_00",
            "days_since_warn": 31,
            "spyglass_commit": "aaa1111",
            "dirty_path": "/home/alice/spyglass",
        },
        {
            "env_id": "alice_env_01",  # same clone, later commit
            "days_since_warn": 45,
            "spyglass_commit": "bbb2222",
            "dirty_path": "/home/alice/spyglass",
        },
        {
            "env_id": "alice_env_02",  # a second clone
            "days_since_warn": 33,
            "spyglass_commit": "ccc3333",
            "dirty_path": "/home/alice/other",
        },
        {
            "env_id": "bob_env_00",
            "days_since_warn": 40,
            "spyglass_commit": "ddd4444",
            "dirty_path": "/home/bob/sg",
        },
    ]
    member_rows = [
        {
            "datajoint_user_name": "alice",
            "google_user_name": "alice@lab.org",
            "admin": 0,
        },
        {
            "datajoint_user_name": "bob",
            "google_user_name": "bob@lab.org",
            "admin": 0,
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo,
            "fetch",
            return_value=member_rows,
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    lines = [ln for ln in out_file.read_text().splitlines() if ln]
    assert len(lines) == 2  # one per user, not one per row

    by_email = {ln.split("\t")[0]: ln.split("\t") for ln in lines}
    alice = by_email["alice@lab.org"]
    assert alice[1] == "45"  # oldest episode, not the first row seen
    # Each clone at its newest commit: day 31 wins over day 45
    assert alice[3] == "/home/alice/other /home/alice/spyglass"
    assert alice[2] == "ccc3333 aaa1111"  # paired with the paths, in order
    assert len(alice) == 4  # run_jobs.sh reads exactly four fields

    assert by_email["bob@lab.org"][1] == "40"


def test_write_dirty_notifications_reports_user_once(
    user_env_tbl, tmp_path, capsys
):
    """An unmatched user is reported once, not once per row."""
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "ghost_env_00",
            "days_since_warn": 40,
            "spyglass_commit": "aaa1111",
            "dirty_path": "/home/ghost/a",
        },
        {
            "env_id": "ghost_env_01",
            "days_since_warn": 50,
            "spyglass_commit": "bbb2222",
            "dirty_path": "/home/ghost/b",
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo, "fetch", return_value=[]
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    assert out_file.read_text().strip() == ""
    stderr = capsys.readouterr().err
    assert stderr.count("No LabMemberInfo email") == 2  # distinct env ids


def test_write_dirty_notifications_skips_unknown_user(
    user_env_tbl, tmp_path, capsys
):
    """A row with no matching LabMemberInfo email is skipped (stderr note)."""
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "ghost_env_00",
            "days_since_warn": 60,
            "spyglass_commit": "ddd4444",
            "dirty_path": "/home/ghost/sg",
        },
    ]
    member_rows = [
        {
            "datajoint_user_name": "alice",
            "google_user_name": "alice@lab.org",
            "admin": 0,
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo,
            "fetch",
            return_value=member_rows,
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    assert out_file.read_text().strip() == ""  # nothing written
    assert "ghost_env_00" in capsys.readouterr().err


def test_write_dirty_notifications_skips_admins(user_env_tbl, tmp_path):
    """An admin's overdue row is not emailed.

    `log_install` exempts admins, but `insert_current_env` has no admin check
    and recompute reaches it through `RecordingRecomputeSelection.env_dict` on
    every spike sorting make. An admin on a dirty clone therefore accumulates
    rows that no clean session of theirs resolves, and they were emailed about
    them from day 30, with a copy to the admin address on every cron run.
    """
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "root_env_00",
            "days_since_warn": 45,
            "spyglass_commit": "aaa1111",
            "dirty_path": "/home/root/spyglass",
        },
        {
            "env_id": "dave_env_00",
            "days_since_warn": 45,
            "spyglass_commit": "ddd4444",
            "dirty_path": "/home/dave/spyglass",
        },
    ]
    member_rows = [
        {
            "datajoint_user_name": "root",
            "google_user_name": "root@lab.org",
            "admin": 1,
        },
        {
            "datajoint_user_name": "dave",
            "google_user_name": "dave@lab.org",
            "admin": 0,
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo,
            "fetch",
            return_value=member_rows,
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    lines = [ln for ln in out_file.read_text().splitlines() if ln]
    assert len(lines) == 1
    assert lines[0].startswith("dave@lab.org")
    assert "root" not in out_file.read_text()


def test_write_dirty_notifications_names_missing_fields(user_env_tbl, tmp_path):
    """A dirty wheel has no clone path and no commit, so both read `unknown`.

    Left empty, the email said "repository at  is not on an official commit
    ()", and `run_jobs.sh` reads the line with `IFS=$'\\t'`, where bash
    collapses the adjacent tabs and shifts the path into the commit field.
    """
    import spyglass.common.common_lab as _lab_mod

    overdue_rows = [
        {
            "env_id": "erin_env_00",
            "days_since_warn": 40,
            "spyglass_commit": None,
            "dirty_path": None,
        },
    ]
    member_rows = [
        {
            "datajoint_user_name": "erin",
            "google_user_name": "erin@lab.org",
            "admin": 0,
        },
    ]
    mock_result = MagicMock()
    mock_result.fetch = MagicMock(return_value=overdue_rows)
    out_file = tmp_path / "dirty.tsv"

    with patch.object(user_env_tbl, "dirty_installs", return_value=mock_result):
        with patch.object(
            _lab_mod.LabMember.LabMemberInfo,
            "fetch",
            return_value=member_rows,
        ):
            user_env_tbl.write_dirty_notifications(str(out_file))

    line = out_file.read_text().strip()
    assert line == "erin@lab.org\t40\tunknown\tunknown"
    assert line.split("\t") == ["erin@lab.org", "40", "unknown", "unknown"]


def test_env_hash_clean_backward_compat(user_env_tbl, clean_install_info):
    """Clean install (commit_hash=None) yields the pre-PR hash (dict untouched).

    Protects existing recompute rows: folding spyglass_commit into the hash
    must be a no-op when there is no commit, so a clean install hashes exactly
    as it did before the column was added.
    """
    from hashlib import md5
    from json import dumps as json_dumps

    with _as_install(**clean_install_info):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        env_dict = user_env_tbl.parse_env_dict(user_env_tbl.env)
        expected = md5(
            json_dumps(env_dict, sort_keys=True).encode()
        ).hexdigest()
        got = user_env_tbl.env_hash

    assert got == expected  # no spyglass_commit folded in for a clean install
