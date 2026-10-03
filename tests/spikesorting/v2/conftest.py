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

from tests.spikesorting.v2._ingest_helpers import (
    clear_curations_for as _clear_curations_for,
)
from tests.spikesorting.v2._motion_db_helpers import (
    CONCAT_GROUP,
    DRIFT_DURATION_S,
    DRIFT_NWB,
    MEMBER_A_INTERVAL,
    MEMBER_B_INTERVAL,
    MOTION_TEAM,
    drop_motion_selections,
    session_start_s,
)
from tests.spikesorting.v2._sorter_stub import plant_sorter

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


@pytest.fixture(autouse=True)
def _isolate_si_metric_defaults():
    """Keep SpikeInterface's metric defaults from leaking between tests.

    A direct ``compute_quality_metrics`` / ``compute_template_metrics`` call
    merges its ``metric_params`` into the metric classes' shared default
    dicts, so a kwarg one test sets would silently apply to every later
    compute in the process, and a test reading SI's defaults would depend on
    test order. Each test sees (and may change) a copy of every quality-
    and template-metric class's defaults, and the original dicts are
    restored afterwards.

    The copies are made without ``isolated_si_metric_defaults``'s lock:
    holding that lock on the main thread for a whole test would block every
    worker thread the test starts that computes metrics through Spyglass.

    No-op under SpikeInterface < 0.101. The legacy (v0/v1) CI job installs
    SI 0.99 and collects one file from this directory,
    ``test_clusterless_waveform_features.py`` (for its v0/v1 guard test), so
    this autouse fixture also runs there. ``_si_metric_patches`` imports
    ``spikeinterface.metrics`` at module scope, which does not exist before
    SI 0.101, and legacy tests never call SI's quality/template metric
    computes, so skipping the copy changes nothing they exercise.
    """
    import spikeinterface as si
    from packaging.version import Version

    if Version(si.__version__) < Version("0.101"):
        yield
        return

    from spyglass.spikesorting.v2._si_metric_patches import (
        _si_metric_param_copies,
    )

    with _si_metric_param_copies():
        yield


# The per-PR smoke fixture. Fetched lazily -- only when a collected test needs
# the database (see ``pytest_collection_modifyitems``), never unconditionally.
_SMOKE_FIXTURE = "mearec_polymer_smoke"


def _eager_fetch_names():
    """Fixture stems to download at *session start*, before collection.

    Only fixtures the required-fixture check will require (so they are present
    when that check runs below) -- plus every fixture when
    ``SPYGLASS_V2_FETCH_FULL=1``, an explicit developer opt-in. The per-PR smoke
    fixture is deliberately NOT in this set: a run that collects no database
    test (a pure-helper unit run) needs no fixture, so fetching one
    unconditionally here starts a spurious download. The smoke fixture is
    fetched lazily at collection time for runs that do need the DB (see
    ``pytest_collection_modifyitems``).
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


def _missing_required_fixtures(required):
    """Names from ``SPYGLASS_V2_REQUIRE_FIXTURES`` that no file satisfies.

    A name ``_fetch.py`` knows is satisfied ONLY by
    ``tests/spikesorting/v2/fixtures/<name>.nwb`` -- the file it downloads and
    sha256-verifies. The shared raw data directory must not count for those:
    ``copy_and_insert_nwb`` copies every ingested fixture into it under its
    own stem, so a leftover copy from a previous run would satisfy a check
    whose whole purpose is to prove THIS run's download happened.

    A name ``_fetch.py`` does not know has no home in ``fixtures/``. That is
    the real recorded session (``minirec20230622``) the workflow curls
    straight into the raw data directory, so for those -- and only those --
    the raw data directory is where the check looks.

    Parameters
    ----------
    required : sequence of str
        Fixture stems (no ``.nwb``) the run declares it must exercise.

    Returns
    -------
    list of str
        The names with no satisfying file, in the given order.
    """
    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    here = Path(__file__).resolve()
    fixtures_dir = here.parent / "fixtures"
    raw_dir = here.parents[2] / "_data" / "raw"
    missing = []
    for name in required:
        home = fixtures_dir if name in FIXTURE_URLS else raw_dir
        if not (home / f"{name}.nwb").exists():
            missing.append(name)
    return missing


def _missing_fixtures_message(missing: list[str]) -> str:
    """Session-exit message for required fixtures that are absent.

    Separates the two reasons a required fixture can be absent, because they
    need different fixes: a stem whose ``FIXTURE_URLS`` entry is empty has no
    hosted copy to download (someone must upload it and set its URL), while
    any other stem had a download that failed or a link that went stale (a
    known stem) or a workflow download step that failed (a stem ``_fetch.py``
    does not know, such as ``minirec20230622``).

    Parameters
    ----------
    missing : sequence of str
        Fixture stems (no ``.nwb``) returned by ``_missing_required_fixtures``.

    Returns
    -------
    str
        One line per reason that applies, each naming its stems.
    """
    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    unhosted = [n for n in missing if n in FIXTURE_URLS and not FIXTURE_URLS[n]]
    failed = [n for n in missing if n not in unhosted]
    lines = []
    if unhosted:
        lines.append(
            "Required v2 fixtures are absent, so the tests that need them "
            "would silently skip: "
            + ", ".join(unhosted)
            + ". No download URL is configured "
            "-- the fixture is not hosted; see "
            "tests/spikesorting/v2/fixtures/README.md."
        )
    if failed:
        lines.append(
            "Required v2 fixtures are absent, so the tests that need them "
            "would silently skip: "
            + ", ".join(failed)
            + ". The download step failed or a "
            "Box link is stale -- see tests/spikesorting/v2/fixtures/_fetch.py."
        )
    return "\n".join(lines)


def pytest_sessionstart(session):
    """Pre-fetch only the fixtures this session is configured to require.

    Downloads the required set (``SPYGLASS_V2_REQUIRE_FIXTURES``, or every
    fixture under ``SPYGLASS_V2_FETCH_FULL=1``) and checks it is present. The
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

    # Required-fixture check: any fixture named in SPYGLASS_V2_REQUIRE_FIXTURES
    # MUST be present, or its test would silently skip and the run would look
    # green without exercising it. The CI workflow sets a job-wide list per
    # trigger (its "Select required fixtures for this run" step), and steps
    # that need other fixtures (the acceptance probes, the two-session matcher
    # gate, the motion benchmark) set their own list on their pytest command
    # only. Unset locally, so absent fixtures skip as before.
    required = os.environ.get("SPYGLASS_V2_REQUIRE_FIXTURES", "").split()
    missing = _missing_required_fixtures(required)
    if missing:
        pytest.exit(_missing_fixtures_message(missing), returncode=1)


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
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
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
    AnalyzerWaveformParameters.insert_default()
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
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)
    yield sort_pk


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


#: Smoke fixture used by the planted-two-unit sort below.
_PREVIEW_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


@pytest.fixture(scope="package")
def planted_two_unit_sort(dj_conn):
    """A populated Sorting with two planted units (so a merge group exists).

    Shared in conftest (rather than a single test module) so the merge-aware
    tests across modules -- preview-merge warnings and the merged-parent guard
    -- resolve the SAME populated sort without a fragile
    cross-module import. The smoke sort yields only one MEArec unit, so this
    plants two units on the real recording with ``plant_sorter``. Tests clear curations around themselves for isolation.
    """
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        copy_and_insert_nwb,
    )

    if not _PREVIEW_FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_PREVIEW_FIXTURE_PATH.name} not found.")

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = copy_and_insert_nwb(
        _PREVIEW_FIXTURE_PATH, dest_name="mearec_preview.nwb"
    )
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 preview"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.array(
            [500, 1500, 2500, 3500, 4500, 600, 1600, 2600, 3600, 4600],
            dtype=np.int64,
        )
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if len(Sorting.Unit & sort_pk) < 2:
        pytest.skip("planted sort did not yield >=2 units")
    yield sort_pk
    _clear_curations_for(sort_pk)
    _clean_session_v2(session)


@pytest.fixture(scope="package")
def planted_three_unit_sort(dj_conn):
    """A populated Sorting with three planted units (0, 1, 2).

    The merged-parent composition tests need a parent curation that keeps at
    least TWO units after a merge, one of which is a FRESH merged id absent
    from ``Sorting.Unit`` -- so the child genuinely composes from the parent
    namespace rather than the raw sort. Two raw units collapse to a single
    merged unit, which is not enough; three raw units let a parent merge two
    of them and still expose a second unit to merge/label in the child.

    Modeled on ``planted_two_unit_sort`` (the smoke sort yields one MEArec
    unit, so ``plant_sorter`` plants the three units on the real
    recording). The three units' spikes are spaced ~200
    frames apart so no cross-unit coincidence is removed by the 0.4 ms merge
    dedup -- a manual merge of any two then conserves spikes exactly, keeping
    the conservation assertions clean. Tests clear curations around themselves
    for isolation.
    """
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        copy_and_insert_nwb,
    )

    if not _PREVIEW_FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_PREVIEW_FIXTURE_PATH.name} not found.")

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = copy_and_insert_nwb(
        _PREVIEW_FIXTURE_PATH, dest_name="mearec_three_unit.nwb"
    )
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 three-unit"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.array(
            [
                500,
                1500,
                2500,
                3500,  # unit 0
                700,
                1700,
                2700,
                3700,  # unit 1
                900,
                1900,
                2900,
                3900,  # unit 2
            ],
            dtype=np.int64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if len(Sorting.Unit & sort_pk) < 3:
        pytest.skip("planted sort did not yield >=3 units")
    yield sort_pk
    _clear_curations_for(sort_pk)
    _clean_session_v2(session)


@pytest.fixture
def curation_evaluation_defaults(dj_conn):
    """Ensure the default metric/auto-curation/waveform Lookup rows exist.

    The Lookup rows a ``CurationEvaluation`` needs ("minimal" metrics, "none"
    auto-curation rules, default waveform window) -- the curation-eval subset of
    ``initialize_v2_defaults``. Shared here so every eval-based test (curation,
    UnitMatch, concat) reuses it instead of re-seeding defaults in-body.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    AnalyzerWaveformParameters.insert_default()


#: ``session_group_owner`` LabTeam used by the chronic concat fixture and its
#: tests. Named once so the fixture's setup/teardown cleanup and the tests that
#: create groups under it cannot drift.
CHRONIC_OWNER_TEAM = "chronic_concat_owner"


@pytest.fixture(scope="package")
def chronic_2_session_minirec(dj_conn, tmp_path_factory):
    """Two same-day + one next-day single-tetrode synthetic sessions.

    Synthesizes three short 4-channel recordings with byte-identical channel
    positions (the fixed ``tetrode_probe_layout``) but different planted-spike
    seeds, writes each into a Frank-lab-style NWB, ingests all three, sets up a
    one-tetrode sort group per session, and populates the per-member
    ``Recording`` cache for the two SAME-DAY members under the ``"default"``
    preprocessing recipe. The third session shares neither a Recording nor a
    date with the first two; it exists so the ``SessionGroup`` same-day check
    has a second date to reject.

    Package-scoped read substrate: tests build their OWN ``SessionGroup`` /
    ``ConcatenatedRecording`` rows on top and tear them down. The fixture
    clears any leftover groups for ``CHRONIC_OWNER_TEAM`` on entry (the
    persistent test DB carries rows across runs) and again on teardown, then
    drops the three sessions.

    Yields
    ------
    dict
        ``owner`` (the ``session_group_owner`` team), ``preprocessing_params_name``,
        ``same_day_members`` (two member dicts, both dated day 1, with populated
        ``Recording`` rows and a ``recording_id``), ``next_day_member`` (one
        member dict dated day 2, no populated ``Recording``), and ``recording_pks``
        (the two same-day recording PK dicts).
    """
    import datetime as dt

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
        synthesize_minirec_nwb,
    )

    tmp_dir = tmp_path_factory.mktemp("chronic_minirec")
    day1 = dt.datetime(2023, 6, 22, 12, 0, 0, tzinfo=dt.timezone.utc)
    specs = [
        ("a", "chronic_minirec_a.nwb", day1, 11),
        ("b", "chronic_minirec_b.nwb", day1.replace(hour=15, minute=30), 22),
        (
            "c",
            "chronic_minirec_c.nwb",
            day1 + dt.timedelta(days=1),
            33,
        ),
    ]

    clean_session_groups_for_owner(CHRONIC_OWNER_TEAM)

    members: list[dict] = []
    nwb_file_names: list[str] = []
    for tag, dest_name, start, seed in specs:
        src = synthesize_minirec_nwb(
            tmp_dir / dest_name,
            session_start=start,
            fixture_name=f"chronic_minirec_{tag}",
            seed=seed,
        )
        nwb_file_name = copy_and_insert_nwb(src, dest_name=dest_name)
        nwb_file_names.append(nwb_file_name)
        run = configure_v2_run_inputs(nwb_file_name, CHRONIC_OWNER_TEAM)
        members.append(
            {
                "nwb_file_name": run["nwb_file_name"],
                "sort_group_id": run["sort_group_id"],
                "interval_list_name": run["interval_list_name"],
            }
        )

    same_day_members = members[:2]
    next_day_member = members[2]

    # ``recording_pks[i]`` is the populated Recording PK for ``same_day_members[i]``;
    # the member dicts stay CLEAN (only Member columns) so they can be passed
    # straight to ``SessionGroup.create_group`` without leaking a recording_id
    # into the Member insert.
    recording_pks = []
    for member in same_day_members:
        rec_pk = RecordingSelection.insert_selection(
            {
                **member,
                "preprocessing_params_name": "default",
                "team_name": CHRONIC_OWNER_TEAM,
            }
        )
        if not (Recording & rec_pk):
            Recording.populate(rec_pk, reserve_jobs=False)
        recording_pks.append(rec_pk)

    yield {
        "owner": CHRONIC_OWNER_TEAM,
        "preprocessing_params_name": "default",
        "same_day_members": same_day_members,
        "next_day_member": next_day_member,
        "recording_pks": recording_pks,
    }

    clean_session_groups_for_owner(CHRONIC_OWNER_TEAM)
    for nwb_file_name in nwb_file_names:
        _clean_session_v2({"nwb_file_name": nwb_file_name})


#: Two intervals cut from one session per day for the daily concatenations
#: the UnitMatch input tests match (seconds after the session's first sample;
#: each longer than ``Recording``'s 1 s minimum segment). Both START in one
#: float64 binade ([2, 4) s): SpikeInterface estimates a persisted recording's
#: sampling rate from its first 1000 timestamps, and a concatenation requires
#: its members' rates to agree to 1e-9 Hz, which timestamps rounded in
#: different binades miss.
DAILY_CONCAT_INTERVALS = {
    "first": (2.02, 3.22),
    "second": (3.3, 4.9),
}


#: Planted unit ids that differ from the default 0, so the two daily
#: concatenations' units are distinguishable.
DAILY_CONCAT_UNIT_IDS = {"concat_day2": 3}


def _plant_spread_unit(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    *,
    job_kwargs=None,
    execution_params=None,
    statistics_spans=None,
    unit_id=0,
):
    """One planted unit, ``unit_id``, firing every 5000 frames throughout."""
    import numpy as np
    import spikeinterface as si

    del sorter, sorter_params, sorting_id, job_kwargs, execution_params
    samples = np.arange(1000, recording.get_num_samples() - 1000, 5000)
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples.astype(np.int64)],
        labels_list=[np.full(len(samples), unit_id, dtype=np.int32)],
        sampling_frequency=recording.get_sampling_frequency(),
    )


@pytest.fixture(scope="module")
def daily_concat_match_inputs(chronic_2_session_minirec):
    """Curated sorts of single recordings and of same-day concatenations.

    On the chronic minirec sessions (``a`` and ``b`` on day 1, ``c`` on day
    2): session ``a`` and session ``c`` are each cut into two intervals
    (``DAILY_CONCAT_INTERVALS``) and concatenated into one daily
    concatenation per day; a third, multi-day concatenation joins ``b``'s
    and ``c``'s first intervals. Every sort is planted (one unit every
    5000 frames, via ``plant_sorter``; its unit id is
    ``DAILY_CONCAT_UNIT_IDS`` or 0) and root-curated:
    single-recording sorts of ``a``, ``b`` and ``a``'s first interval, and
    one sort of each concatenation.

    Yields
    ------
    dict
        ``curations`` (name -> ``{"sorting_id", "curation_id"}``; names
        ``single_a``, ``single_b``, ``single_a_first``, ``concat_day1``,
        ``concat_day2``, ``concat_multi_day``), ``concat_keys`` (name ->
        ``{"concat_recording_id"}``), ``recording_keys`` (name ->
        ``{"recording_id"}`` for ``a``, ``b``, ``a_first``, ``a_second``,
        ``b_first``, ``c_first``, ``c_second``) and ``nwb_file_names``
        (``a``, ``b``, ``c``).
    """
    import functools
    import math

    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters
    from tests.spikesorting.v2._concat_helpers import select_unmasked_concat
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
        clear_curations_for,
        configure_v2_run_inputs,
        drop_unitmatch_selections_for,
    )

    sub = chronic_2_session_minirec
    owner = sub["owner"]
    preprocessing = sub["preprocessing_params_name"]
    nwb_a = sub["same_day_members"][0]["nwb_file_name"]
    nwb_b = sub["same_day_members"][1]["nwb_file_name"]
    nwb_c = sub["next_day_member"]["nwb_file_name"]
    SorterParameters.insert_default()
    MatcherParameters.insert_default()
    clean_session_groups_for_owner(owner)

    recording_keys = {
        "a": sub["recording_pks"][0],
        "b": sub["recording_pks"][1],
    }
    members = {}
    intervals = [
        (tag, nwb_file_name, part)
        for tag, nwb_file_name in (("a", nwb_a), ("c", nwb_c))
        for part in DAILY_CONCAT_INTERVALS
    ] + [("b", nwb_b, "first")]
    for tag, nwb_file_name, part in intervals:
        t0 = float(
            (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": "raw data valid times",
                }
            ).fetch1("valid_times")[0][0]
        )
        start, stop = DAILY_CONCAT_INTERVALS[part]
        # Every interval of every session starts in one binade (see the
        # constant).
        assert math.frexp(t0 + DAILY_CONCAT_INTERVALS["first"][0])[1] == (
            math.frexp(t0 + DAILY_CONCAT_INTERVALS["second"][0])[1]
        )
        name = f"unitmatch_daily_{part}"
        IntervalList.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": name,
                "valid_times": np.asarray([[t0 + start, t0 + stop]]),
                "pipeline": "unitmatch_daily_concat_test",
            },
            skip_duplicates=True,
        )
        member = configure_v2_run_inputs(
            nwb_file_name, owner, interval_list_name=name
        )
        members[f"{tag}_{part}"] = member
        key = RecordingSelection.insert_selection(
            {**member, "preprocessing_params_name": preprocessing}
        )
        if not (Recording & key):
            Recording.populate(key, reserve_jobs=False)
        recording_keys[f"{tag}_{part}"] = key

    groups = {
        "concat_day1": ([members["a_first"], members["a_second"]], False),
        "concat_day2": ([members["c_first"], members["c_second"]], False),
        "concat_multi_day": ([members["b_first"], members["c_first"]], True),
    }
    concat_keys = {}
    for name, (group_members, multi_day) in groups.items():
        SessionGroup.create_group(
            owner,
            f"unitmatch_{name}",
            group_members,
            allow_multi_day=multi_day,
        )
        concat_keys[name] = select_unmasked_concat(
            {
                "session_group_owner": owner,
                "session_group_name": f"unitmatch_{name}",
                "preprocessing_params_name": preprocessing,
            }
        )
        ConcatenatedRecording.populate(concat_keys[name], reserve_jobs=False)

    sorter = {
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }
    sources = {
        "single_a": recording_keys["a"],
        "single_b": recording_keys["b"],
        "single_a_first": recording_keys["a_first"],
        **concat_keys,
    }
    sort_keys = {}
    curations = {}
    patch = pytest.MonkeyPatch()
    try:
        for name, source in sources.items():
            plant_sorter(
                patch,
                functools.partial(
                    _plant_spread_unit,
                    unit_id=DAILY_CONCAT_UNIT_IDS.get(name, 0),
                ),
            )
            sort_key = SortingSelection.insert_selection({**source, **sorter})
            if not (Sorting & sort_key):
                Sorting.populate(sort_key, reserve_jobs=False)
            clear_curations_for(sort_key)
            curation = CurationV2.insert_curation(sorting_key=sort_key)
            sort_keys[name] = sort_key
            curations[name] = {
                "sorting_id": curation["sorting_id"],
                "curation_id": curation["curation_id"],
            }
    finally:
        patch.undo()

    yield {
        "curations": curations,
        "concat_keys": concat_keys,
        "recording_keys": recording_keys,
        "nwb_file_names": {"a": nwb_a, "b": nwb_b, "c": nwb_c},
    }

    # Selections pin these curations; drop them before the curations go.
    drop_unitmatch_selections_for(
        [{"sorting_id": key["sorting_id"]} for key in sort_keys.values()]
    )
    clean_session_groups_for_owner(owner)
    for name in ("single_a", "single_b", "single_a_first"):
        if name in sort_keys:
            clear_curations_for(sort_keys[name])
            (Sorting & sort_keys[name]).super_delete(warn=False)
            (SortingSelection & sort_keys[name]).super_delete(warn=False)
    for name in ("a_first", "a_second", "b_first", "c_first", "c_second"):
        (RecordingSelection & recording_keys[name]).super_delete(warn=False)


# ---- motion correction: the planted-drift polymer session -------------------


@pytest.fixture(scope="module")
def drift_recording(dj_conn, tmp_path_factory):
    """A populated ``Recording`` of the planted-drift polymer session."""
    import datetime as dt

    from spyglass.spikesorting.v2.motion import MotionEstimationParameters
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        write_drifting_polymer_nwb,
    )

    src = write_drifting_polymer_nwb(
        tmp_path_factory.mktemp("motion") / DRIFT_NWB,
        session_start=dt.datetime(2023, 6, 22, 12, tzinfo=dt.timezone.utc),
        fixture_name="motion_drift_polymer",
        seed=0,
        duration_s=DRIFT_DURATION_S,
    )
    nwb_file_name = copy_and_insert_nwb(src, dest_name=DRIFT_NWB)
    run = configure_v2_run_inputs(nwb_file_name, MOTION_TEAM)
    recording_key = RecordingSelection.insert_selection(
        {**run, "preprocessing_params_name": "default"}
    )
    drop_motion_selections(recording_key)
    if not (Recording & recording_key):
        Recording.populate(recording_key, reserve_jobs=False)
    MotionEstimationParameters.insert_default()

    yield {"recording_key": recording_key, "nwb_file_name": nwb_file_name}

    drop_motion_selections(recording_key)
    _clean_session_v2({"nwb_file_name": nwb_file_name})


@pytest.fixture(scope="module")
def discontinuous_sources(drift_recording):
    """A gapped ``Recording`` and a two-member ``ConcatenatedRecording``."""
    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from tests.spikesorting.v2._concat_helpers import select_unmasked_concat
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
    )

    nwb_file_name = drift_recording["nwb_file_name"]
    valid = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0, t_end = float(valid[0][0]), float(valid[-1][1])
    intervals = {
        MEMBER_A_INTERVAL: [[t0 + 16.0, t0 + 20.0]],
        MEMBER_B_INTERVAL: [[t0 + 23.0, t0 + 26.0], [t0 + 27.0, t_end]],
    }
    recording_keys = {}
    members = []
    for name, times in intervals.items():
        IntervalList.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": name,
                "valid_times": np.asarray(times, dtype=float),
                "pipeline": "motion_estimate_test",
            },
            skip_duplicates=True,
        )
        run = configure_v2_run_inputs(
            nwb_file_name, MOTION_TEAM, interval_list_name=name
        )
        members.append(run)
        recording_keys[name] = RecordingSelection.insert_selection(
            {**run, "preprocessing_params_name": "default"}
        )
        drop_motion_selections(recording_keys[name])
        if not (Recording & recording_keys[name]):
            Recording.populate(recording_keys[name], reserve_jobs=False)

    clean_session_groups_for_owner(MOTION_TEAM)
    SessionGroup.create_group(MOTION_TEAM, CONCAT_GROUP, members)
    concat_key = select_unmasked_concat(
        {
            "session_group_owner": MOTION_TEAM,
            "session_group_name": CONCAT_GROUP,
            "preprocessing_params_name": "default",
        }
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)

    yield {
        "t0": t0,
        "member_a": recording_keys[MEMBER_A_INTERVAL],
        "member_b": recording_keys[MEMBER_B_INTERVAL],
        "concat_key": concat_key,
    }

    drop_motion_selections(concat_key)
    for key in recording_keys.values():
        drop_motion_selections(key)
    clean_session_groups_for_owner(MOTION_TEAM)


# ---- daily-concat matching: two days of the same planted neurons --------------


@pytest.fixture(scope="module")
def planted_matching_days(dj_conn, tmp_path_factory):
    """Two ingested days of the same planted neurons, one session group each.

    Each day of ``_daily_match_fixtures.DAYS`` is written as one polymer
    session (day 1 with the planted drift), ingested, cut into its two member
    intervals and grouped as ``daily_match_day1`` / ``daily_match_day2``
    under ``DAILY_MATCH_TEAM``. Nothing is populated.

    Yields
    ------
    dict
        ``team`` and ``days``: ``{label: {"spec", "nwb_file_name", "t0",
        "session_group_name", "manual_excluded_times"}}``, where ``t0`` is the
        session's first raw timestamp and ``manual_excluded_times`` is the
        day's exclusion in ``run_v2_pipeline``'s per-member form.
    """
    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.settings import raw_dir
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
    from tests.spikesorting.v2 import _daily_match_fixtures as design
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._motion_fixtures import write_polymer_nwb

    team = design.DAILY_MATCH_TEAM

    def _drop_team_rows():
        concat_keys = (
            ConcatenatedRecordingSelection & {"session_group_owner": team}
        ).fetch("KEY", as_dict=True)
        for key in concat_keys:
            drop_motion_selections(
                {"concat_recording_id": key["concat_recording_id"]}
            )
        clean_session_groups_for_owner(team)

    _drop_team_rows()
    out_dir = tmp_path_factory.mktemp("daily_match")
    days, nwb_file_names = {}, []
    for day in design.DAYS:
        name = f"daily_match_{day.label}.nwb"
        # A raw file left by an earlier run would be reused as is; the
        # planted truth lives in this code, so always write a fresh one.
        for stale in (name, get_nwb_copy_filename(name)):
            (Path(raw_dir) / stale).unlink(missing_ok=True)
        src = write_polymer_nwb(
            out_dir / name,
            design.day_recording(day).get_traces(return_in_uV=True),
            session_start=day.session_start,
            fixture_name=f"daily_match_{day.label}",
        )
        nwb_file_name = copy_and_insert_nwb(src, dest_name=name)
        nwb_file_names.append(nwb_file_name)
        t0 = session_start_s(nwb_file_name)
        members = []
        for index, (start, stop) in enumerate(day.members_s):
            interval = f"daily match {day.label} member {index}"
            IntervalList.insert1(
                {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": interval,
                    "valid_times": np.asarray([[t0 + start, t0 + stop]]),
                    "pipeline": "daily_match_test",
                },
                skip_duplicates=True,
            )
            members.append(
                configure_v2_run_inputs(
                    nwb_file_name, team, interval_list_name=interval
                )
            )
        group = f"daily_match_{day.label}"
        SessionGroup.create_group(team, group, members)
        member, (ex_start, ex_stop) = day.exclusion
        days[day.label] = {
            "spec": day,
            "nwb_file_name": nwb_file_name,
            "t0": t0,
            "session_group_name": group,
            "manual_excluded_times": {member: [[t0 + ex_start, t0 + ex_stop]]},
        }

    yield {"team": team, "days": days}

    _drop_team_rows()
    for nwb_file_name in nwb_file_names:
        _clean_session_v2({"nwb_file_name": nwb_file_name})


#: Team owning the offset-source session group (its own owner, so no other
#: fixture's group cleanup removes it).
OFFSET_SOURCE_TEAM = "offset_source_team"


@pytest.fixture(scope="module")
def offset_source_concat(dj_conn, tmp_path_factory):
    """Unfiltered, unreferenced int16 recordings that keep a nonzero offset,
    and a masked two-member ``no_filter`` concatenation of them.

    The planted-drift session is written as int16 0.25 uV counts shifted by
    10000 counts with an NWB offset of -2500 uV (the same voltages as the
    float session). With ``reference_mode="none"`` and the ``no_filter``
    recipe, the recording stage neither filters nor references, so each
    member's artifact keeps that offset. Member A ``[16, 20) s`` masks
    ``[17, 18) s`` by hand; member B ``[23, end)`` is unmasked. Both start in
    the same power-of-two range of timestamps, so their derived sampling
    rates agree (see ``discontinuous_sources``).
    """
    import datetime as dt

    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        write_drifting_polymer_nwb,
    )

    name = "offset_source_int16"
    src = write_drifting_polymer_nwb(
        tmp_path_factory.mktemp("offset_source") / f"{name}.nwb",
        session_start=dt.datetime(2023, 8, 22, 12, tzinfo=dt.timezone.utc),
        fixture_name=name,
        seed=0,
        duration_s=DRIFT_DURATION_S,
        int16_offset_counts=10_000,
    )
    nwb_file_name = copy_and_insert_nwb(src, dest_name=f"{name}.nwb")
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, reference_mode="none"
    )
    t0 = float(
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch1("valid_times")[0][0]
    )
    t_end = t0 + DRIFT_DURATION_S - 0.01
    intervals = {
        "offset member a": [[t0 + 16.0, t0 + 20.0]],
        "offset member b": [[t0 + 23.0, t_end]],
    }
    members, recording_keys = [], []
    for interval_name, times in intervals.items():
        IntervalList.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
                "valid_times": np.asarray(times, dtype=float),
                "pipeline": "offset_source_test",
            },
            skip_duplicates=True,
        )
        run = configure_v2_run_inputs(
            nwb_file_name, OFFSET_SOURCE_TEAM, interval_list_name=interval_name
        )
        members.append(run)
        recording_key = RecordingSelection.insert_selection(
            {**run, "preprocessing_params_name": "no_filter"}
        )
        Recording.populate(recording_key, reserve_jobs=False)
        recording_keys.append(recording_key)
    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_keys[0]["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": np.array([[t0 + 17.0, t0 + 18.0]]),
        }
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)

    clean_session_groups_for_owner(OFFSET_SOURCE_TEAM)
    group = {
        "session_group_owner": OFFSET_SOURCE_TEAM,
        "session_group_name": "offset_source_concat",
    }
    SessionGroup.create_group(
        OFFSET_SOURCE_TEAM, group["session_group_name"], members
    )
    concat_key = ConcatenatedRecordingSelection.insert_selection(
        {**group, "preprocessing_params_name": "no_filter"},
        artifact_detection_ids={
            0: artifact_key["artifact_detection_id"],
            1: None,
        },
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)

    yield {
        "nwb_file_name": nwb_file_name,
        "member_a": recording_keys[0],
        "member_b": recording_keys[1],
        "artifact_key": artifact_key,
        "concat_key": concat_key,
    }

    drop_motion_selections(concat_key)
    for recording_key in recording_keys:
        drop_motion_selections(recording_key)
    clean_session_groups_for_owner(OFFSET_SOURCE_TEAM)
    _clean_session_v2({"nwb_file_name": nwb_file_name})


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
