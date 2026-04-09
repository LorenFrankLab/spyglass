"""Pytest configuration for setup tests.

These tests don't require database connections, so we override the
session-scoped fixtures from the parent conftest.py to avoid triggering
spyglass imports that connect to MySQL.
"""

import sys
from pathlib import Path

import pytest

# Suppress non-essential install.py console output in-process without
# propagating to subprocess test runs (env vars propagate; class patch does not).
try:
    _sp = str(Path(__file__).parent.parent.parent / "scripts")
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
    import install as _install

    _install.Console._quiet = True
    del _sp, _install
except Exception:
    pass

# Override ALL session-scoped fixtures that might trigger database connections
# or spyglass imports during collection


@pytest.fixture(scope="session")
def dj_conn():
    """Override dj_conn to avoid database connection requirement."""
    yield None


@pytest.fixture(scope="session")
def server():
    """Override server to avoid Docker database requirement."""
    yield None


@pytest.fixture(scope="session")
def mini_insert():
    """Override mini_insert to avoid spyglass imports."""
    yield None


@pytest.fixture(scope="session")
def mini_path():
    """Override mini_path to avoid data downloads."""
    yield None


@pytest.fixture(scope="session")
def mini_content():
    """Override mini_content to avoid NWB file loading."""
    yield None


@pytest.fixture(scope="session")
def mini_copy_name():
    """Override mini_copy_name."""
    yield None


@pytest.fixture(scope="session")
def mini_dict():
    """Override mini_dict."""
    yield None


@pytest.fixture(scope="session")
def mini_restr():
    """Override mini_restr."""
    yield None


@pytest.fixture(scope="session")
def teardown():
    """Override teardown."""
    yield True


@pytest.fixture(scope="session")
def server_credentials():
    """Override server_credentials."""
    yield {}
