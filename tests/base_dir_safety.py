"""Pure helpers for pytest base-dir safety (issue #1573).

These live in their own module rather than ``conftest.py`` so tests can
import them unambiguously. ``import conftest`` is unreliable: the test suite
has many ``conftest.py`` files and that name resolves inconsistently in a
full-suite run.
"""

import json
import os
import tempfile
import warnings
from pathlib import Path
from shutil import rmtree as shutil_rmtree

import pytest

TEST_ROOT_SENTINEL = ".spyglass-test-root"


def _derive_dir_env_vars():
    """Return every `{PREFIX}_{KEY}_DIR` env var listed in directory_schema.json.

    Uses the same directory schema that ``SpyglassConfig`` uses to populate
    relative directories. Used to scrub stale env vars before tests run, so
    a shell-exported `DLC_VIDEO_DIR=/shared/prod/video` (or similar) cannot
    leak into spyglass.settings after we override SPYGLASS_BASE_DIR.
    """
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "spyglass"
        / "directory_schema.json"
    )
    try:
        schema = json.loads(schema_path.read_text())["directory_schema"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise pytest.UsageError(
            "Could not derive Spyglass test directory env vars from "
            f"{schema_path}. Refusing to run tests without a complete scrub "
            "list for path-related environment variables."
        ) from exc
    return [
        f"{prefix.upper()}_{key.upper()}_DIR"
        for prefix, dirs in schema.items()
        for key in dirs
    ]


def _resolved_path(path: str | Path) -> Path:
    """Return the resolved absolute form of a filesystem path."""
    return Path(path).expanduser().resolve()


def _sentinel_path(base_dir: str | Path) -> Path:
    """Return the sandbox sentinel path for a test base directory."""
    return _resolved_path(base_dir) / TEST_ROOT_SENTINEL


def _write_test_root_sentinel(base_dir: str | Path) -> None:
    """Mark a pytest-created temp base directory as safe for test cleanup."""
    _sentinel_path(base_dir).write_text(
        "Spyglass pytest sandbox. Files here may be deleted by tests.\n"
    )


def _require_test_root_sentinel(base_dir: str | Path, source: str) -> None:
    """Require a sentinel before using a persistent test base directory."""
    sentinel = _sentinel_path(base_dir)
    if sentinel.is_file():
        return

    raise pytest.UsageError(
        f"{source} resolved to {sentinel.parent}, but that directory is not "
        "marked as a Spyglass test sandbox. Omit --base-dir to let pytest "
        "create a fresh temp sandbox automatically, or choose a dedicated "
        "test-only directory that already contains the sentinel file. Never "
        "place this sentinel in a shared or production data root; it is a "
        "durable opt-in that allows destructive tests to use the directory "
        "(issue #1573)."
    )


def _resolve_base_dir(config) -> tuple[str, str | None]:
    """Resolve the pytest base dir and allocated temp dir, if any."""
    env_base = os.environ.get("SPYGLASS_BASE_DIR")
    if config.option.base_dir:
        _require_test_root_sentinel(config.option.base_dir, "--base-dir")
        return config.option.base_dir, None

    if config.option.use_env_base_dir and env_base:
        _require_test_root_sentinel(env_base, "SPYGLASS_BASE_DIR")
        return env_base, None

    # --no-teardown is meaningless with the temp-dir default: a fresh
    # mkdtemp() is allocated each session, so DB rows from a previous
    # run would point at a temp path that no longer exists on the next
    # run. Require an explicit --base-dir (or --use-env-base-dir with
    # SPYGLASS_BASE_DIR set) for persistent-state workflows.
    if config.option.no_teardown:
        raise pytest.UsageError(
            "--no-teardown requires an explicit --base-dir (or "
            "--use-env-base-dir with SPYGLASS_BASE_DIR set). The "
            "default temp-dir base is created fresh each session, so "
            "the preserved database would point at files that no "
            "longer exist. Try: pytest --no-teardown --base-dir "
            "./tests/_data/"
        )

    # RuntimeWarning, not the default UserWarning: conftest installs a
    # module-level `filterwarnings("ignore", category=UserWarning)`, which
    # would silently swallow these notices in a real run.
    if config.option.use_env_base_dir and not env_base:
        warnings.warn(
            "--use-env-base-dir was passed but SPYGLASS_BASE_DIR is not set; "
            "falling back to a temp dir.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif env_base:
        warnings.warn(
            f"Ignoring SPYGLASS_BASE_DIR={env_base!r}; pass "
            "--use-env-base-dir to honor it, or --base-dir to set an "
            "explicit path. See issue #1573.",
            RuntimeWarning,
            stacklevel=2,
        )

    tmp_base_dir = tempfile.mkdtemp(prefix="spyglass_test_")
    _write_test_root_sentinel(tmp_base_dir)
    warnings.warn(
        f"Using temp base_dir for tests: {tmp_base_dir}",
        RuntimeWarning,
        stacklevel=2,
    )
    return tmp_base_dir, tmp_base_dir


def _warn_cleanup_failure(
    action: str, path: str | Path, exc: Exception
) -> None:
    """Warn that best-effort pytest cleanup could not remove a path."""
    warnings.warn(
        f"Could not {action} during pytest cleanup: {path} ({exc})",
        RuntimeWarning,
        stacklevel=2,
    )


def _rmtree_warn(path: str | Path) -> None:
    """Remove a directory tree, warning instead of silently swallowing errors."""
    try:
        shutil_rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        _warn_cleanup_failure("remove directory tree", path, exc)


def unconfigure_cleanup(
    server, tmp_base_dir, teardown, close_nwb_files
) -> None:
    """Run pytest session teardown cleanup.

    Extracted from the ``pytest_unconfigure`` hook as a pure function so it
    can be tested without the pytest hook machinery or conftest globals.

    Parameters
    ----------
    server : object or None
        The Docker MySQL manager, or None if ``pytest_configure`` failed
        before it was created.
    tmp_base_dir : str or None
        Path of the per-session temp base dir, or None if a user-supplied
        ``--base-dir`` was used.
    teardown : bool
        Whether to stop the database server.
    close_nwb_files : callable
        Callback that closes any open NWB files. Only invoked when ``server``
        is set, so the spyglass import it needs can stay lazy.
    """
    if server is None:
        if tmp_base_dir is not None:
            _rmtree_warn(tmp_base_dir)
        return

    close_nwb_files()
    if teardown:
        server.stop()

    # Remove any temp base_dir we allocated. It's under $TMPDIR, was created by
    # this run, and has no reuse value. User-supplied persistent base_dirs are
    # intentionally not cleaned here: the durable sentinel is enough to opt into
    # running tests there, but not enough to prove pytest may delete old files in
    # that directory.
    if tmp_base_dir is not None:
        _rmtree_warn(tmp_base_dir)
