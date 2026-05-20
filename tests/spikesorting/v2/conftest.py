"""Fixtures for the modern spike sorting test suite.

The repository-wide ``tests/conftest.py`` defines a session-scoped, autouse
``mini_insert`` fixture that starts a Docker MySQL server and ingests the
shared sample NWB file. The scaffold and pure-helper tests in this package
exercise Pydantic models, path helpers, and static schema validation only --
they need neither a database connection nor sample data. This module overrides
``mini_insert`` with a no-op so those tests run without Docker. Database-tier
tests in this package (the AnalysisNwbfile-backed hash determinism check and
the MEArec fixture ingestion round-trip) request the DB-tier fixtures defined
below directly, and they expect Docker to be available.

The fixture/baseline generator scripts and the standalone bootstrap helper
are filtered out of collection so pytest does not try to import them as test
modules (their ``test_`` prefixes are coincidental component names).
"""

import copy
from pathlib import Path

import datajoint as dj
import pytest

# These files are scripts and helper modules, not pytest test modules; the
# leading ``test_`` is part of the component name (the standalone test
# environment bootstrap). Excluding them keeps ``--doctest-modules`` from
# importing them outside the bootstrap.
collect_ignore = [
    "test_env.py",
    "baseline_capture.py",
    "fixtures/generate_mearec.py",
]


@pytest.fixture(scope="session", autouse=True)
def mini_insert():
    """No-op override of the repository-wide sample-data ingestion fixture.

    Spike sorting tests in this package that require a database request the
    ``analysis_nwbfile_for_hash`` fixture (or ``dj_conn`` directly); the
    scaffold and helper tests do not touch DataJoint at all, so no server or
    sample data is needed.
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


@pytest.fixture(scope="session")
def analysis_nwbfile_for_hash(dj_conn, mini_path):
    """Create an ``AnalysisNwbfile`` row anchored to the bundled mini NWB.

    Ingests ``minirec20230622.nwb`` into the isolated test database if it is
    not already present, then creates a fresh ``AnalysisNwbfile`` linked to
    it. Yields the analysis file name -- the argument
    ``_hash_nwb_recording`` expects.

    Skips the test cleanly if Docker is unavailable or sample data is missing.
    """
    if not Path(mini_path).exists():
        pytest.skip(f"Sample NWB not found at {mini_path}.")

    from spyglass.common import Nwbfile
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.data_import import insert_sessions

    mini_dict = {"nwb_file_name": f"{mini_path.stem}_.nwb"}
    if not (Nwbfile & mini_dict):
        insert_sessions(mini_path.name, raise_err=True, reinsert=True)
    parent = (Nwbfile & f"nwb_file_name LIKE '{mini_path.stem}%'").fetch1(
        "nwb_file_name"
    )

    analysis_file_name = AnalysisNwbfile().create(parent)
    AnalysisNwbfile().add(parent, analysis_file_name)
    yield analysis_file_name
