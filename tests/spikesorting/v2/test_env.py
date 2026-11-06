"""Isolated test-environment bootstrap for standalone modern-spike-sorting scripts.

The pytest suite isolates the database and analysis directories through
``tests/conftest.py``: it launches a Docker MySQL server, writes a local
DataJoint config, sets ``database.prefix = "pytests"``, and redirects
``SPYGLASS_BASE_DIR`` to a temporary directory. Standalone scripts -- fixture
generation and v1 baseline capture -- do not inherit that fixture setup, so they
must call :func:`bootstrap_v2_test_environment` before importing Spyglass or
touching the database.

The bootstrap is deliberately strict: it refuses to run against a production-like
database prefix or a base directory that looks like shared lab storage, so a
script can never write to (or clean up) production by accident.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import datajoint as dj

# Substrings that indicate shared lab storage rather than a test directory.
# A resolved base directory containing any of these is rejected outright.
_SHARED_STORAGE_MARKERS = ("stelmo",)

# A database prefix is accepted only if it is exactly this or starts with
# "test"; everything else is treated as production-like.
_TEST_PREFIX = "pytests"

# Environment variable that must be set for production-connected reads.
_PRODUCTION_SMOKE_ENV = "SPYGLASS_ALLOW_PRODUCTION_SMOKE"


def _repo_root() -> Path:
    """Return the repository root (four levels above this file)."""
    return Path(__file__).resolve().parents[3]


def _is_within(child: Path, parent: Path) -> bool:
    """Return True if ``child`` is ``parent`` or nested under it."""
    return child == parent or parent in child.parents


def _assert_safe_base_dir(base_dir: Path | str) -> Path:
    """Resolve ``base_dir`` and reject shared-storage or out-of-bounds paths.

    Parameters
    ----------
    base_dir : pathlib.Path or str
        Requested Spyglass base directory.

    Returns
    -------
    pathlib.Path
        The resolved, absolute base directory.

    Raises
    ------
    RuntimeError
        If the resolved directory contains a shared-storage marker or does not
        sit under ``tests/_data/`` or a temporary directory.
    """
    resolved = Path(base_dir).expanduser().resolve()
    lowered = str(resolved).lower()
    for marker in _SHARED_STORAGE_MARKERS:
        if marker in lowered:
            raise RuntimeError(
                f"Refusing base_dir {resolved}: it contains the shared-storage "
                f"marker {marker!r}. Modern-spike-sorting validation scripts "
                "must write under tests/_data/ or a temporary directory, never "
                "shared lab storage."
            )

    allowed_roots = {
        (_repo_root() / "tests" / "_data").resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
    }
    if not any(_is_within(resolved, root) for root in allowed_roots):
        raise RuntimeError(
            f"Refusing base_dir {resolved}: it is outside the permitted test "
            "locations. Pass a directory under tests/_data/ or a temporary "
            f"directory (allowed roots: {sorted(map(str, allowed_roots))})."
        )
    return resolved


def _assert_safe_prefix(database_prefix: str) -> str:
    """Validate that ``database_prefix`` is a test prefix, not production.

    Parameters
    ----------
    database_prefix : str
        Requested DataJoint schema prefix.

    Returns
    -------
    str
        The validated prefix.

    Raises
    ------
    RuntimeError
        If the prefix is empty or not recognizably a test prefix.
    """
    prefix = (database_prefix or "").strip()
    if not prefix:
        raise RuntimeError(
            "database_prefix is empty; refusing to connect. Pass 'pytests' or "
            "a 'test'-prefixed schema name."
        )
    if prefix != _TEST_PREFIX and not prefix.startswith("test"):
        raise RuntimeError(
            f"database_prefix {prefix!r} is not a recognized test prefix. "
            "Modern-spike-sorting validation writes only to 'pytests' or a "
            "'test'-prefixed schema; production prefixes are not permitted."
        )
    return prefix


def _docker_test_server_credentials() -> dict:
    """Start or reuse the Docker MySQL test server and return its credentials.

    Mirrors the server the pytest suite uses (``tests/container.py``). The
    server is reused if already running and is left running on exit so a script
    run can be followed by ``pytest`` without a restart.
    """
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tests.container import DockerMySQLManager

    server = DockerMySQLManager(restart=False, shutdown=False, verbose=False)
    server.wait()
    return server.credentials


def bootstrap_v2_test_environment(
    base_dir: Path | str,
    database_prefix: str = _TEST_PREFIX,
    *,
    dj_config: Path | str | None = None,
    allow_production_smoke: bool = False,
) -> None:
    """Prepare an isolated database + directory environment for v2 scripts.

    Call this once, before importing any Spyglass module that reads config.
    It validates the requested base directory and database prefix, scrubs
    Spyglass directory environment variables that might point at shared
    storage, loads DataJoint credentials, and pins the schema prefix.

    Parameters
    ----------
    base_dir : pathlib.Path or str
        Spyglass base directory. Must resolve under ``tests/_data/`` or a
        temporary directory.
    database_prefix : str, optional
        DataJoint schema prefix. Must be ``"pytests"`` or ``"test"``-prefixed.
        Defaults to ``"pytests"``.
    dj_config : pathlib.Path or str, optional
        Path to an existing DataJoint config file. If given, it is loaded
        instead of starting the Docker test server.
    allow_production_smoke : bool, optional
        If True, the caller intends read-only production-connected lookups.
        This requires ``SPYGLASS_ALLOW_PRODUCTION_SMOKE=1`` in the environment;
        all writes still target ``database_prefix`` and ``base_dir``.

    Raises
    ------
    RuntimeError
        If the base directory or prefix is unsafe, or if production smoke is
        requested without the environment gate.
    FileNotFoundError
        If ``dj_config`` is given but does not exist.
    """
    resolved_base = _assert_safe_base_dir(base_dir)
    prefix = _assert_safe_prefix(database_prefix)

    if allow_production_smoke and os.environ.get(_PRODUCTION_SMOKE_ENV) != "1":
        raise RuntimeError(
            f"allow_production_smoke=True requires {_PRODUCTION_SMOKE_ENV}=1 in "
            "the environment. Production-connected lookups are read-only and "
            f"all writes still target the {prefix!r} prefix."
        )

    resolved_base.mkdir(parents=True, exist_ok=True)

    # Scrub any exported Spyglass directory variables (including a production
    # SPYGLASS_BASE_DIR) so settings.py resolves directories from our base dir.
    for var in [
        name
        for name in os.environ
        if name.startswith("SPYGLASS_") and name.endswith("_DIR")
    ]:
        os.environ.pop(var, None)
    os.environ["SPYGLASS_BASE_DIR"] = str(resolved_base)

    if dj_config is not None:
        config_path = Path(dj_config)
        if not config_path.exists():
            raise FileNotFoundError(
                f"--dj-config {config_path} does not exist."
            )
        dj.config.load(str(config_path))
    else:
        dj.config.update(_docker_test_server_credentials())

    dj.config["database.prefix"] = prefix

    # Reset directory overrides from any globally saved config so DataJoint and
    # the env-var fallback both resolve to our base dir (see issue #1573).
    custom = dj.config.setdefault("custom", {})
    for key in ("dlc_dirs", "moseq_dirs", "kachery_dirs"):
        custom.pop(key, None)
    custom["spyglass_dirs"] = {"base": str(resolved_base)}

    if allow_production_smoke:
        warnings.warn(
            "Production-connected metadata reads are enabled. All writes still "
            f"target the {prefix!r} prefix and {resolved_base}.",
            stacklevel=2,
        )
