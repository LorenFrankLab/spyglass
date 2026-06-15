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


# The per-PR smoke fixture. Fetched lazily -- only when a collected test needs
# the database (see ``pytest_collection_modifyitems``), never unconditionally.
_SMOKE_FIXTURE = "mearec_polymer_smoke"


def _eager_fetch_names():
    """Fixture stems to download at *session start*, before collection.

    Only fixtures the honest-green gate will require (so they are present when
    the gate checks below) -- plus every fixture when ``SPYGLASS_V2_FETCH_FULL=1``,
    an explicit developer opt-in. The per-PR smoke fixture is deliberately NOT in
    this set: a run that collects no database test (a pure-helper unit run) needs
    no fixture, so fetching one unconditionally here starts a spurious download.
    The smoke fixture is fetched lazily at collection time for runs that do need
    the DB (see ``pytest_collection_modifyitems``).
    """
    import os

    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    if os.environ.get("SPYGLASS_V2_FETCH_FULL") == "1":
        return list(FIXTURE_URLS)
    required = os.environ.get("SPYGLASS_V2_REQUIRE_FIXTURES", "").split()
    return [n for n in required if n in FIXTURE_URLS]


# Shared fixtures (defined below) that ingest the smoke NWB; a test using one is
# a consumer even when its own module never names the fixture.
_SMOKE_CONSUMING_FIXTURES = frozenset(
    {"populated_sorting", "populated_sorting_with_curation"}
)

_MODULE_SOURCE_CACHE: dict[str, str] = {}


def _module_references_smoke(item):
    """True if the collected item's test module mentions the smoke fixture
    anywhere in its source (module-level ``_FIXTURE_PATH`` constant or an
    in-function ``copy_and_insert_nwb`` path). Cheap, cached per module file."""
    path = getattr(getattr(item, "module", None), "__file__", None)
    if not path:
        return False
    if path not in _MODULE_SOURCE_CACHE:
        try:
            _MODULE_SOURCE_CACHE[path] = Path(path).read_text()
        except OSError:
            _MODULE_SOURCE_CACHE[path] = ""
    return _SMOKE_FIXTURE in _MODULE_SOURCE_CACHE[path]


def _item_consumes_smoke_fixture(item):
    """True only if a collected item actually ingests the smoke MEArec fixture.

    A consumer either depends on a shared smoke-ingesting fixture, or requests
    ``dj_conn`` *and* references the fixture in its module source. Requiring both
    signals keeps two non-consumers out: a DB test that only declares tables
    (``dj_conn`` but no fixture reference) and a pure-helper test that merely
    names the fixture in an assertion (a reference but no ``dj_conn``). Neither
    should trigger a 55MB download."""
    fixtures = set(getattr(item, "fixturenames", ()))
    if fixtures & _SMOKE_CONSUMING_FIXTURES:
        return True
    return "dj_conn" in fixtures and _module_references_smoke(item)


def pytest_sessionstart(session):
    """Pre-fetch only the fixtures this session is configured to require.

    Downloads the honest-green-gated set (``SPYGLASS_V2_REQUIRE_FIXTURES``, or
    every fixture under ``SPYGLASS_V2_FETCH_FULL=1``) and verifies the gate. The
    per-PR smoke fixture is fetched lazily at collection time instead (see
    ``pytest_collection_modifyitems``), so a pure-helper run -- which collects no
    DB test -- starts no download. ``ensure_fixture`` is a no-op when the file is
    already present (e.g. generated locally) or when no URL is configured.
    """
    import os
    import warnings

    from tests.spikesorting.v2.fixtures._fetch import (
        FixtureFetchError,
        ensure_fixture,
    )

    for name in _eager_fetch_names():
        try:
            ensure_fixture(name, required=False)
        except FixtureFetchError as exc:
            # Don't abort the whole session on a fetch failure -- dependent
            # tests skip via their own ``_PATH.exists()`` guard. Surface it.
            warnings.warn(f"[v2 fixtures] could not fetch {name}: {exc}")

    # Honest-green gate: any fixture named in SPYGLASS_V2_REQUIRE_FIXTURES MUST
    # be present, or its test would silently skip and the run would look green
    # without exercising the gate. CI sets this per tier to exactly the fixtures
    # it downloaded (per-PR: smoke; nightly: + 60s polymer; manual dispatch:
    # + scenario fixtures). Unset locally, so absent fixtures skip as before.
    required = os.environ.get("SPYGLASS_V2_REQUIRE_FIXTURES", "").split()
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    missing = [n for n in required if not (fixtures_dir / f"{n}.nwb").exists()]
    if missing:
        pytest.exit(
            "Required v2 fixtures are absent, so their gates would silently "
            "skip: " + ", ".join(missing) + ". The download step failed or a "
            "Box link is stale -- see tests/spikesorting/v2/fixtures/_fetch.py.",
            returncode=1,
        )


def pytest_collection_modifyitems(session, config, items):
    """Fetch the per-PR smoke fixture lazily -- only when the collected tests
    actually need the database.

    Runs after collection, where the collected ``items`` (and their fixture
    closures) are known, so only a run that actually ingests the fixture fetches
    it -- a pure-helper unit run, or a DB run that just declares tables, triggers
    no download. ``ensure_fixture`` is a no-op when the file is already present,
    so this only reaches the network when the smoke fixture is genuinely missing
    and a collected test will consume it.
    """
    import warnings

    from tests.spikesorting.v2.fixtures._fetch import (
        FixtureFetchError,
        ensure_fixture,
    )

    if not any(_item_consumes_smoke_fixture(item) for item in items):
        return
    try:
        ensure_fixture(_SMOKE_FIXTURE, required=False)
    except FixtureFetchError as exc:
        warnings.warn(f"[v2 fixtures] could not fetch {_SMOKE_FIXTURE}: {exc}")


# Same fixture the lazy fetch above produces; named once so the two can't drift.
_DOWNSTREAM_FIXTURE_NAME = _SMOKE_FIXTURE
_DOWNSTREAM_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / f"{_DOWNSTREAM_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="package")
def populated_sorting(dj_conn):
    """Populate Recording -> ArtifactDetection -> Sorting for the smoke
    fixture. Package-scoped so the heavy populate is paid once and shared
    across ``test_downstream_consumers.py`` and ``test_integrity.py``.

    Lives in conftest (rather than in one module with a cross-module
    import) so the integrity tests resolve it directly: an import from a
    sibling test module passes vacuously when a CI shard split collects
    the two modules separately, leaving the integrity loops iterating
    over an empty DB.
    """
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _DOWNSTREAM_FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_DOWNSTREAM_FIXTURE_PATH.name} "
            "not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    # Ingest under a session name UNIQUE to this shared fixture. Other v2
    # test modules ingest the same smoke fixture under its own basename and
    # run ``_clean_session_v2`` / ``reinsert`` on that session; isolating this
    # one keeps the package-scoped rows (cached as ``sort_pk`` below) from
    # being cascade-deleted out from under the downstream/integrity tests.
    nwb_file_name = copy_and_insert_nwb(
        _DOWNSTREAM_FIXTURE_PATH, dest_name="mearec_downstream_smoke.nwb"
    )
    session_key = {"nwb_file_name": nwb_file_name}

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 downstream"},
        skip_duplicates=True,
    )

    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
            "artifact_id": art_pk["artifact_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)
    yield sort_pk


def _clear_curations_for(sorting_key):
    """Drop a sorting's CurationV2 rows + merge masters (shared helper)."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(sorting_key)


@pytest.fixture
def populated_sorting_with_curation(populated_sorting):
    """A root ``CurationV2`` over the populated smoke sort.

    Builds on the package-scoped ``populated_sorting`` and inserts one
    root curation (``parent_curation_id=-1``, no labels, no merges) so the
    curation-side read tests (``get_unit_brain_regions`` on ``CurationV2``,
    ``get_merged_sorting`` early returns) have a known master to query.

    Function-scoped and self-cleaning: the persistent test DB carries rows
    across runs, so the fixture clears any pre-existing curations for this
    sorting first, then removes the ones it created on teardown. Yields the
    ``{"sorting_id", "curation_id"}`` PK dict.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations_for(populated_sorting)
    curation_key = CurationV2.insert_curation(sorting_key=populated_sorting)
    yield curation_key
    _clear_curations_for(populated_sorting)


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
