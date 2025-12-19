"""Pytest configuration for setup tests.

These tests don't require database connections, so we override the
session-scoped fixtures from the parent conftest.py to avoid triggering
spyglass imports that connect to MySQL.
"""

import pytest

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
