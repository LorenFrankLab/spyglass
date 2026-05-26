"""Fixtures for the modern spike sorting test suite.

The repository-wide ``tests/conftest.py`` defines a session-scoped, autouse
``mini_insert`` fixture that starts a Docker MySQL server and ingests the
shared sample NWB file. The scaffold and pure-helper tests in this package
exercise Pydantic models and pure helpers only -- they need neither a
database connection nor sample data. This module overrides ``mini_insert``
with a no-op so those tests run without Docker. Database-tier tests
(the MEArec fixture ingestion round-trip) request ``dj_conn`` directly
and expect Docker to be available.

The fixture/baseline generator scripts and the standalone bootstrap helper
are filtered out of collection so pytest does not try to import them as test
modules (their ``test_`` prefixes are coincidental component names).
"""

import copy

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
    "_fixtures/phase1_baseline.py",
]


@pytest.fixture(scope="session", autouse=True)
def mini_insert():
    """No-op override of the repository-wide sample-data ingestion fixture.

    Tests that need the database request ``dj_conn`` directly; the scaffold
    and helper tests do not touch DataJoint, so no server or sample data is
    needed.
    """
    yield


@pytest.fixture(autouse=True)
def _disable_datajoint_safemode(request):
    """Force ``dj.config['safemode']`` to boolean False for v2 tests.

    The Docker MySQL credentials configured in ``tests/container.py`` set
    ``safemode`` to the *string* ``"false"``, which Python evaluates as
    truthy. DataJoint then prompts ``"Commit deletes? [yes, No]"`` on every
    non-empty delete; under pytest's captured stdin that becomes an
    ``EOFError`` which surfaces as a misleading "Delete cannot use a
    transaction within an ongoing transaction" error on the next call.

    Forcing the boolean before each test is the smallest fix that keeps
    the v2 cleanup pattern (``super_delete`` between tests) working
    without per-call ``safemode=False`` boilerplate. Per-test scope
    (rather than session) because the session-scoped ``dj_conn`` fixture
    calls ``dj.config.load(...)`` which overwrites our setting; depending
    on ``dj_conn`` directly would force every v2 test to require Docker
    even when none of its assertions touch the DB.
    """
    dj.config["safemode"] = False
    yield


@pytest.fixture(scope="session")
def phase1_baseline_artifacts():
    """Load the on-disk pre-refactor baseline bundle for regression
    tests.

    The bundle is generated on **unmodified pre-refactor code** by
    the standalone test ``test_phase1_baseline_regen.py``. After
    the v2 refactors land, validation tests load this fixture to
    check that bit-equivalence holds on deterministic paths (the
    ``Recording`` artifact and ``clusterless_thresholder`` spike
    samples).

    This fixture intentionally does **not** depend on ``dj_conn`` or on
    the raw 60s polymer NWB fixture -- it only reads the captured npz /
    pickle / json bundle. Tests that need both the baseline and a live
    populate request ``dj_conn`` and ``polymer_60s_session`` alongside
    this fixture; this one is the cheap-load layer.

    **Local / manual gate, not a CI gate.** Only the lightweight
    ``MANIFEST.json`` + ``stage_metrics.json`` are committed; the
    heavy ``.npz`` / ``.pkl`` payloads are gitignored. CI runs
    therefore skip every test that depends on this fixture (the
    bundle is "not present"). To exercise the gate locally,
    regenerate the bundle on pre-refactor tip via
    ``pytest tests/spikesorting/v2/test_phase1_baseline_regen.py
    -m regenerate`` (the ``regenerate`` marker is excluded by
    default in ``pyproject.toml``).

    Skipped with a clear pointer to the regen workflow if the bundle is
    missing or if the captured SI / NumPy / pynwb versions diverge from
    the current environment. Version drift would invalidate the
    bit-equivalence comparisons, so we fail loudly rather than silently
    relaxing the gate.
    """
    from tests.spikesorting.v2._fixtures import phase1_baseline as _baseline

    if not _baseline.baseline_present():
        pytest.skip(
            "Pre-refactor baseline bundle missing from "
            f"{_baseline.PHASE1_BASELINE_DIR}. Regenerate via "
            "`pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q` "
            "from a clean checkout of the pre-refactor tip."
        )
    bundle = _baseline.load()
    mismatches = _baseline.verify_manifest_compatible(bundle.manifest)
    if mismatches:
        pytest.skip(
            "Pre-refactor baseline bundle env mismatch: "
            + "; ".join(mismatches)
            + ". Regenerate or pin the dependency versions to the baseline."
        )
    yield bundle


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
