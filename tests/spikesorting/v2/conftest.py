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


_DOWNSTREAM_FIXTURE_NAME = "mearec_polymer_smoke"
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
    """Delete every CurationV2 row for a sorting plus its merge masters.

    DataJoint refuses to drop a part row whose master is still present, so
    walk from the ``SpikeSortingOutput`` merge master down before dropping
    the ``CurationV2`` rows. Mirrors the test-module helper of the same
    shape; kept in conftest so the shared curation fixture is
    self-contained (no cross-module import of a test helper).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    for mid in (SpikeSortingOutput.CurationV2 & sorting_key).fetch("merge_id"):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & sorting_key).super_delete(warn=False)


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
