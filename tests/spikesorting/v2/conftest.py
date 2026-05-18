"""Fixtures for the modern spike sorting test suite.

The repository-wide ``tests/conftest.py`` defines a session-scoped, autouse
``mini_insert`` fixture that starts a Docker MySQL server and ingests the
shared sample NWB file. The scaffold and pure-helper tests in this package
exercise Pydantic models, path helpers, and static schema validation only --
they need neither a database connection nor sample data. This module overrides
``mini_insert`` with a no-op so those tests run without Docker. Tests that need
a live database request the ``dj_conn`` / ``server`` fixtures explicitly.
"""

import copy

import datajoint as dj
import pytest


@pytest.fixture(scope="session", autouse=True)
def mini_insert():
    """No-op override of the repository-wide sample-data ingestion fixture.

    Spike sorting tests in this package that require a database request
    ``dj_conn`` directly; the scaffold and helper tests do not touch
    DataJoint at all, so no server or sample data is needed.
    """
    yield


@pytest.fixture
def restore_custom_config():
    """Snapshot and restore ``dj.config['custom']`` around a test.

    Also guarantees ``dj.config['custom']`` is a dict before the test runs,
    so a test that assigns into it is self-contained regardless of how
    pytest is invoked.
    """
    original = copy.deepcopy(dict(dj.config.get("custom") or {}))
    dj.config["custom"] = copy.deepcopy(original)
    yield
    dj.config["custom"] = copy.deepcopy(original)
