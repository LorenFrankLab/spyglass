"""Tests for the dirty-install and unification across conda calls.

Split into two groups:
  Mock-based (no DB): CondaEnvCache, get_install_info, warn_if_dirty logic
  DB-required:        UserEnvironment dirty-column integration

Run with: conda run -n dirty pytest tests/common/test_dirty.py --no-teardown
"""

import os
import time
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


def _proc(stdout="", returncode=0) -> CompletedProcess:
    return CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


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
def compute_install_info_fn():
    """Return the _compute_install_info function (deferred import)."""
    from spyglass.utils.git_utils import _compute_install_info

    return _compute_install_info


@pytest.fixture
def make_cache(conda_env_cache_cls, not_computed):
    """Fixture factory — returns isolated CondaEnvCache instances."""

    def _factory(tmp_path, *, cache_exists=False, history_mtime=None):
        """Create a fresh CondaEnvCache scoped to tmp_path."""
        cache = conda_env_cache_cls()
        cache_path = tmp_path / ".spyglass_conda_test_base.yaml"
        history_path = tmp_path / "history"
        history_path.write_text("# conda history\n")
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

    assert mock_run.call_count == 1  # conda only — pip freeze skipped
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
    # Disk holds conda only — no partial pip list a with_pip reader could
    # mistake for "no pip packages".
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

    assert [c.args[0][0] for c in mock_run.call_args_list] == ["conda", "pip"]
    assert "pip" in yaml.safe_load(cache_path.read_text())  # disk completed


def test_cache_with_pip_hit(tmp_path, make_cache):
    """get(with_pip=True) also uses in-process cache after the first call."""
    cache, _ = make_cache(tmp_path)
    side_effects = [_proc(_CONDA_STDOUT), _proc(_PIP_STDOUT)]

    with patch("spyglass.utils.env_cache.sub_run", side_effect=side_effects):
        merged1 = cache.get(with_pip=True)
        merged2 = cache.get(with_pip=True)

    assert merged1 is merged2


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
    """A newer conda-meta/history makes a previously-fresh cache stale."""
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
    """With _history_path=None, is_stale() always returns True (no crash)."""
    cache, _ = make_cache(tmp_path)
    cache.__dict__["_history_path"] = None

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
    """Two conda environments for the same user get different cache paths."""
    monkeypatch.setattr("spyglass.settings.temp_dir", str(tmp_path))

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "env_a")
    cache_a = conda_env_cache_cls()
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_a = cache_a._cache_path

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "env_b")
    cache_b = conda_env_cache_cls()
    with patch("spyglass.utils.env_cache.getpass.getuser", return_value="user"):
        path_b = cache_b._cache_path

    assert path_a != path_b


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
#     get_install_info / _compute_install_info (mock-based, no DB)
# ═════════════════════════════════════════════════════════════════════════════


def test_pip_install_no_git(tmp_path, compute_install_info_fn):
    """Standard pip install: no .git ancestor → install_path None, no changes."""
    fake_pkg = tmp_path / "site-packages" / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    with patch("spyglass.__version__", "0.5.6"):
        with patch("spyglass.__file__", str(fake_pkg)):
            result = compute_install_info_fn()

    assert result["install_path"] is None
    assert result["has_local_changes"] is False
    assert result["is_official"] is True


def test_dev_version_is_dev(compute_install_info_fn):
    """A '.dev' version sets is_dev=True; local edits make it non-official."""
    # Mocked so the result does not depend on this checkout's working tree:
    # a dirty diff + a recent HEAD date. is_official derives from edits/stale,
    # not is_dev.
    with patch("spyglass.__version__", "0.5.6.dev42+gb3db6d09"):
        with patch("spyglass.__file__", __file__):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[_proc("src/x.py\n"), _proc("4102444800")],
            ):
                result = compute_install_info_fn()

    assert result["is_dev"] is True
    assert result["has_local_changes"] is True
    assert result["is_official"] is False


def test_release_version_not_dev(tmp_path, compute_install_info_fn):
    """A clean release version string gives is_dev=False."""
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    with patch("spyglass.__version__", "0.5.6"):
        with patch("spyglass.__file__", str(fake_pkg)):
            result = compute_install_info_fn()

    assert result["is_dev"] is False


def test_commit_hash_from_version(compute_install_info_fn):
    """Commit hash is parsed from the +g<hash> suffix in __version__."""
    with patch("spyglass.__version__", "0.5.6.dev1+gabcd1234"):
        with patch("spyglass.__file__", __file__):
            result = compute_install_info_fn()

    assert result["commit_hash"] == "abcd1234"


def test_git_dirty_sets_has_local_changes(compute_install_info_fn):
    """git diff returning output means has_local_changes=True."""
    dirty_proc = _proc("src/spyglass/something.py\n")

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", __file__):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[dirty_proc, _proc("4102444800")],
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is True
    assert result["is_official"] is False


def test_git_clean_no_changes(compute_install_info_fn):
    """git diff returning empty output means has_local_changes=False."""
    clean_proc = _proc("")

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", __file__):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[clean_proc, _proc("4102444800")],
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is False


def test_git_unavailable_no_crash(compute_install_info_fn):
    """FileNotFoundError from git is swallowed; has_local_changes=False."""
    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", __file__):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=FileNotFoundError("git not found"),
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is False


def test_git_short_hash_fallback(tmp_path, compute_install_info_fn):
    """When version has no +g hash, git rev-parse is called as a fallback."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    hash_proc = _proc("abc1234\n")
    diff_proc = _proc("")

    with patch("spyglass.__version__", "0.5.6.dev1"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[hash_proc, diff_proc, _proc("4102444800")],
            ):
                result = compute_install_info_fn()

    assert result["commit_hash"] == "abc1234"


def test_git_diff_nonzero_exit(tmp_path, compute_install_info_fn):
    """git diff non-zero exit code is handled gracefully; has_local_changes=False."""
    (tmp_path / ".git").mkdir()
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    error_proc = _proc("", returncode=128)

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[error_proc, _proc("4102444800")],
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is False


def test_git_diff_generic_exception(tmp_path, compute_install_info_fn):
    """Generic exception from git diff is swallowed; has_local_changes=False."""
    (tmp_path / ".git").mkdir()
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=RuntimeError("unexpected"),
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is False


def test_git_stale_old_head(tmp_path, compute_install_info_fn):
    """A clean clone with a months-old HEAD is is_stale=True, not official."""
    (tmp_path / ".git").mkdir()
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    clean_proc = _proc("")  # no local changes
    old_proc = _proc("100000000")  # 1973 — well past the staleness threshold

    with patch("spyglass.__version__", "0.5.6.dev1+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[clean_proc, old_proc],
            ):
                result = compute_install_info_fn()

    assert result["has_local_changes"] is False
    assert result["is_stale"] is True
    assert result["is_official"] is False


def test_clean_current_clone_is_official(tmp_path, compute_install_info_fn):
    """A clean, recently-fetched dev clone is official despite is_dev=True."""
    (tmp_path / ".git").mkdir()
    fake_pkg = tmp_path / "spyglass" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    clean_proc = _proc("")  # no local changes
    recent_proc = _proc("4102444800")  # year 2100 — not stale

    with patch("spyglass.__version__", "0.5.6.dev42+gabc1234"):
        with patch("spyglass.__file__", str(fake_pkg)):
            with patch(
                "spyglass.utils.git_utils.sub_run",
                side_effect=[clean_proc, recent_proc],
            ):
                result = compute_install_info_fn()

    assert result["is_dev"] is True
    assert result["has_local_changes"] is False
    assert result["is_stale"] is False
    assert result["is_official"] is True  # being a few commits behind is fine


# ═════════════════════════════════════════════════════════════════════════════
#     warn_if_dirty (mock DB, no real connection needed)
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
    from spyglass.utils.git_utils import warn_if_dirty

    with caplog.at_level("WARNING"):
        warn_if_dirty(info)

    assert not any("repository" in r.message for r in caplog.records)


def test_warn_dirty_emits_warning(caplog, warn_dirty_days):
    """A dirty install (has_local_changes) emits the countdown warning."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod
    from spyglass.utils.git_utils import warn_if_dirty

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
                warn_if_dirty(info)

    assert any("official commit" in r.message for r in caplog.records)
    assert any(f"{warn_dirty_days} days" in r.message for r in caplog.records)


def test_warn_admin_skipped(caplog):
    """Admin users receive no warning."""
    import spyglass.common.common_lab as _lab_mod

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "install_path": "/home/admin/spyglass",
        "commit_hash": "abc1234",
    }
    from spyglass.utils.git_utils import warn_if_dirty

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = True

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with caplog.at_level("WARNING"):
            warn_if_dirty(info)

    assert not any("official commit" in r.message for r in caplog.records)


def test_warn_dev_but_clean_emits_stale_warning(caplog):
    """A clean-but-stale clone emits the softer 'older commit' advisory."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": False,
        "is_stale": True,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }
    from spyglass.utils.git_utils import warn_if_dirty

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False

    mock_env = MagicMock()
    mock_env.__and__ = MagicMock(return_value=mock_env)
    mock_env.__bool__ = MagicMock(return_value=False)

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with patch.object(_user_mod, "UserEnvironment", mock_env):
            with caplog.at_level("WARNING"):
                warn_if_dirty(info)

    assert any("older commit" in r.message for r in caplog.records)
    assert not any("official commit" in r.message for r in caplog.records)


def test_warn_countdown_uses_days_since_first_flag(caplog, warn_dirty_days):
    """N counts down from WARN_DIRTY_ENV_DAYS based on the oldest dirty row."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod
    from spyglass.utils.git_utils import warn_if_dirty

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False

    # warn_if_dirty now computes the day count DB-side:
    # dirty.proj(_d=DIRTY_DAYS_SQL).fetch("_d"); oldest row -> largest count.
    mock_proj = MagicMock()
    mock_proj.fetch = MagicMock(return_value=[5])  # days since first flag

    mock_dirty = MagicMock()
    mock_dirty.__bool__ = MagicMock(return_value=True)
    mock_dirty.__and__ = MagicMock(return_value=mock_dirty)
    mock_dirty.proj = MagicMock(return_value=mock_proj)

    mock_env = MagicMock()
    mock_env.__and__ = MagicMock(return_value=mock_dirty)

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with patch.object(_user_mod, "UserEnvironment", mock_env):
            with caplog.at_level("WARNING"):
                warn_if_dirty(info)

    expected_n = warn_dirty_days - 5
    assert any(f"{expected_n} days" in r.message for r in caplog.records)


def test_warn_logs_dirty_install(caplog):
    """A flagged non-admin install logs itself so the 30-day clock starts."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod
    from spyglass.utils.git_utils import warn_if_dirty

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/user/spyglass",
        "commit_hash": "abc1234",
    }

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = False

    mock_inst = MagicMock()  # the UserEnvironment() instance
    mock_env = MagicMock(return_value=mock_inst)  # UserEnvironment() -> inst
    mock_env.__and__ = MagicMock(return_value=mock_env)
    mock_env.__bool__ = MagicMock(return_value=False)

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with patch.object(_user_mod, "UserEnvironment", mock_env):
            warn_if_dirty(info)

    mock_inst.insert_current_env.assert_called_once()


def test_warn_admin_not_logged(caplog):
    """An admin install is neither warned nor logged."""
    import spyglass.common.common_lab as _lab_mod
    import spyglass.common.common_user as _user_mod
    from spyglass.utils.git_utils import warn_if_dirty

    info = {
        "is_official": False,
        "is_dev": True,
        "has_local_changes": True,
        "is_stale": False,
        "install_path": "/home/admin/spyglass",
        "commit_hash": "abc1234",
    }

    mock_lab = MagicMock()
    mock_lab.return_value.user_is_admin = True

    mock_inst = MagicMock()
    mock_env = MagicMock(return_value=mock_inst)

    with patch.object(_lab_mod, "LabMember", mock_lab):
        with patch.object(_user_mod, "UserEnvironment", mock_env):
            warn_if_dirty(info)

    mock_inst.insert_current_env.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
#     _warn_once import-time gating (mock-based, no DB)
# ═════════════════════════════════════════════════════════════════════════════
#
# _warn_once() is the synchronous, once-per-process warning hook called from
# spyglass/__init__.py (and lazily by UserEnvironment.this_env). These tests
# cover its gating; the CondaEnvCache is_stale() gating is covered by the
# test_cache_stale_* / test_cache_no_conda_prefix tests above.


def test_warn_once_idempotent():
    """_warn_once calls warn_if_dirty only once per process."""
    import spyglass.utils.git_utils as gu

    gu._warned = False
    calls = []
    try:
        with patch.object(gu, "get_install_info", return_value={}):
            with patch.object(
                gu,
                "warn_if_dirty",
                side_effect=lambda info: calls.append(info) or True,
            ):
                gu._warn_once()
                gu._warn_once()
        assert len(calls) == 1  # second call short-circuited by the flag
        assert gu._warned is True
    finally:
        gu._warned = False


def test_warn_once_retries_when_not_ready():
    """A False result (DB not ready) leaves the flag unset for a later retry."""
    import spyglass.utils.git_utils as gu

    gu._warned = False
    outcomes = iter([False, True])
    calls = []

    def fake_warn(info):
        calls.append(info)
        return next(outcomes)

    try:
        with patch.object(gu, "get_install_info", return_value={}):
            with patch.object(gu, "warn_if_dirty", side_effect=fake_warn):
                gu._warn_once()  # returns False -> not decisive
                assert gu._warned is False
                gu._warn_once()  # returns True -> decisive, flag set
                assert gu._warned is True
        assert len(calls) == 2
    finally:
        gu._warned = False


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
    }


def test_clean_install_null_fields(user_env_tbl, clean_install_info):
    """A clean install stores NULL for dirty_path and spyglass_commit."""
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=clean_install_info,
    ):
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
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=dirty_install_info,
    ):
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

    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=clean_install_info,
    ):
        for attr in ("env", "env_hash", "matching_env_id"):
            user_env_tbl.__dict__.pop(attr, None)
        clean_hash = user_env_tbl.env_hash

    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=dirty_install_info,
    ):
        for attr in ("env", "env_hash", "matching_env_id"):
            user_env_tbl.__dict__.pop(attr, None)
        dirty_hash = user_env_tbl.env_hash

    assert clean_hash != dirty_hash


def test_dirty_idempotent(user_env_tbl, dirty_install_info):
    """Calling insert_current_env twice with the same dirty commit is idempotent."""
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=dirty_install_info,
    ):
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
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=dirty_install_info,
    ):
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


def test_stale_install_flagged(user_env_tbl, stale_install_info):
    """A clean-but-stale install (no edits) still populates dirty_path."""
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=stale_install_info,
    ):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    row = (user_env_tbl & f'env_id="{env_id}"').fetch1()
    assert row["dirty_path"] == stale_install_info["install_path"]
    assert row["spyglass_commit"] == stale_install_info["commit_hash"]


def test_clean_install_not_in_dirty_installs(user_env_tbl, clean_install_info):
    """A clean install is NOT surfaced by dirty_installs()."""
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=clean_install_info,
    ):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        user_env_tbl.delete(safemode=False)
        result = user_env_tbl.insert_current_env()

    env_id = result["env_id"]
    dirty_ids = user_env_tbl.dirty_installs().fetch("env_id")
    assert env_id not in dirty_ids


def test_has_matching_env_same_hash(user_env_tbl):
    """has_matching_env returns True when the stored hash matches current env."""
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


def test_write_env_yaml(user_env_tbl, tmp_path):
    """write_env_yaml writes a YAML file named after the env_id."""
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
    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=dirty_install_info,
    ):
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
            "env_id": "carol_env_00",  # below threshold — filtered out
            "days_since_warn": 10,
            "spyglass_commit": "ccc3333",
            "dirty_path": "/home/carol/sg",
        },
    ]
    member_rows = [
        {"datajoint_user_name": "alice", "google_user_name": "alice@lab.org"},
        {"datajoint_user_name": "alice_bob", "google_user_name": "ab@lab.org"},
        {"datajoint_user_name": "carol", "google_user_name": "carol@lab.org"},
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
        {"datajoint_user_name": "alice", "google_user_name": "alice@lab.org"},
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


def test_env_hash_clean_backward_compat(user_env_tbl, clean_install_info):
    """Clean install (commit_hash=None) yields the pre-PR hash (dict untouched).

    Protects existing recompute rows: folding spyglass_commit into the hash
    must be a no-op when there is no commit, so a clean install hashes exactly
    as it did before the column was added.
    """
    from hashlib import md5
    from json import dumps as json_dumps

    with patch(
        "spyglass.common.common_user.get_install_info",
        return_value=clean_install_info,
    ):
        for attr in ("env", "env_hash", "matching_env_id", "this_env"):
            user_env_tbl.__dict__.pop(attr, None)
        env_dict = user_env_tbl.parse_env_dict(user_env_tbl.env)
        expected = md5(
            json_dumps(env_dict, sort_keys=True).encode()
        ).hexdigest()
        got = user_env_tbl.env_hash

    assert got == expected  # no spyglass_commit folded in for a clean install
