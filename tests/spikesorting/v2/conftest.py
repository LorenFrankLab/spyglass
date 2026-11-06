"""Fixtures for the v2 spike sorting tests.

- Overrides the repo-wide autouse ``mini_insert`` with a no-op so pure-helper
  tests run without Docker; database tests request ``dj_conn``.
- Excludes generator scripts and the bootstrap helper from collection.
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
from tests.spikesorting.v2._test_tiers import (
    EXTERNAL_DATA_FIXTURES,
    SUITE_PATH,
    TIER_BUDGETS_S,
    TIER_TIMEOUTS_S,
    TIERS,
    TierClassificationError,
    classify_test,
    database_fixture_names,
    fixture_minimums,
    load_manifest,
)

# Scripts and helpers, not test modules (``test_env`` is a component name).
collect_ignore = [
    "test_env.py",
    "baseline_capture.py",
    "fixtures/generate_mearec.py",
]

# Tier and accumulated setup + call + teardown seconds, per test node id. The
# tier comes from the report's keywords (its tier marker) rather than from
# collection, because under pytest-xdist the reporting process collects nothing.
_TEST_TIER: dict[str, str] = {}
_TEST_SECONDS: dict[str, float] = {}


def pytest_configure(config):
    _TEST_TIER.clear()
    _TEST_SECONDS.clear()


def pytest_runtest_logreport(report):
    tier = next((t for t in TIERS if t in report.keywords), None)
    if tier is None:
        return
    _TEST_TIER[report.nodeid] = tier
    _TEST_SECONDS[report.nodeid] = (
        _TEST_SECONDS.get(report.nodeid, 0.0) + report.duration
    )


def _over_budget() -> list[tuple[str, str, float]]:
    """Return ``(nodeid, tier, seconds)`` for tests over their tier budget."""
    return sorted(
        (
            (nodeid, _TEST_TIER[nodeid], seconds)
            for nodeid, seconds in _TEST_SECONDS.items()
            if seconds > TIER_BUDGETS_S.get(_TEST_TIER[nodeid], float("inf"))
        ),
        key=lambda row: -row[2],
    )


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("v2_tier_budgets") and _over_budget():
        if exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("v2_tier_budgets"):
        return
    over = _over_budget()
    if not over:
        return
    terminalreporter.section("v2 tier budgets exceeded", red=True)
    for nodeid, tier, seconds in over:
        terminalreporter.line(
            f"{seconds:7.1f}s > {TIER_BUDGETS_S[tier]:g}s ({tier}) {nodeid}"
        )
    terminalreporter.line(
        "Move each test to a heavier tier (test_tiers.json) or make its "
        "setup cheaper; budgets are TIER_BUDGETS_S in _test_tiers.py."
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Assign one reviewed tier before marker selection, then check fixtures.

    The runtime-balanced CI shard called ``unit`` intentionally includes light
    database work. The ``unit`` tier instead guarantees no database fixtures
    or downloaded scientific data; ``--v2-tier unit`` also blocks DB access.
    """
    manifest = load_manifest()
    assignments = []
    failures = []
    for item in items:
        if not item.path.is_relative_to(SUITE_PATH):
            continue
        if not isinstance(item, pytest.Function):
            continue
        module = item.path.relative_to(SUITE_PATH).as_posix()
        name = item.nodeid.split("::", 1)[1].split("[", 1)[0]
        declared = None
        for node in reversed(item.listchain()):
            tiers = {mark.name for mark in node.own_markers} & set(TIERS)
            if tiers:
                if len(tiers) > 1:
                    failures.append(
                        f"{item.nodeid}: conflicting explicit tiers {sorted(tiers)}"
                    )
                declared = next(iter(tiers))
                break

        definitions = item._fixtureinfo.name2fixturedefs
        database_names = database_fixture_names(definitions)
        try:
            minimums = fixture_minimums(manifest, definitions)
            tier = classify_test(
                manifest,
                module,
                name,
                declared_tier=declared,
                fixture_tiers=minimums,
                needs_database=bool(database_names)
                or item.get_closest_marker("database") is not None,
                needs_external_data=bool(
                    set(item.fixturenames) & EXTERNAL_DATA_FIXTURES
                ),
            )
        except TierClassificationError as exc:
            failures.append(str(exc))
        else:
            assignments.append((item, tier))

    if failures:
        unique = sorted(set(failures))
        raise pytest.UsageError(
            "Invalid v2 test-tier assignments:\n" + "\n".join(unique)
        )

    # Legacy module/class markers otherwise leave DB tests matching ``-m unit``
    # even when a function-level override is db_unit. Snapshot first, normalize
    # the hierarchy once, then give each function exactly its effective tier.
    for item, _ in assignments:
        for node in item.listchain():
            if node is item or node.path.is_relative_to(SUITE_PATH):
                node.own_markers[:] = [
                    mark for mark in node.own_markers if mark.name not in TIERS
                ]
    timeouts = config.pluginmanager.hasplugin("timeout")
    for item, tier in assignments:
        item.add_marker(getattr(pytest.mark, tier))
        if timeouts and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(TIER_TIMEOUTS_S[tier]))

    requested = config.getoption("v2_tier", default=None)
    if requested is not None:
        wanted = {item for item, tier in assignments if tier == requested}
        selected = [item for item in items if item in wanted]
        deselected = [item for item in items if item not in wanted]
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope="session", autouse=True)
def mini_insert():
    """No-op override of the repo-wide sample-data ingestion fixture."""
    yield


@pytest.fixture(autouse=True)
def _guard_unit_tier_database_access(request, monkeypatch):
    """Keep isolated tests isolated when they run beside database tiers too."""
    if request.node.get_closest_marker("unit") is None:
        yield
        return

    def refuse(*args, **kwargs):
        raise AssertionError(
            "The v2 unit tier attempted DataJoint database access. "
            "Move this test to a database tier or isolate its DB boundary."
        )

    # Use the test's own ``monkeypatch``: one shared undo stack restores the
    # real ``dj.Connection.query`` even if the test also patches it.
    monkeypatch.setattr(dj.Connection, "connect", refuse)
    monkeypatch.setattr(dj.Connection, "query", refuse)
    yield


@pytest.fixture(autouse=True)
def _disable_datajoint_safemode(request):
    """Force ``dj.config['safemode']`` to boolean False for v2 tests.

    ``tests/container.py`` sets the truthy string ``"false"``, so deletes
    would prompt and fail under captured stdin. Per-test because the
    session ``dj_conn`` fixture reloads the config.
    """
    dj.config["safemode"] = False
    yield


@pytest.fixture(autouse=True)
def _isolate_si_metric_defaults():
    """Keep SpikeInterface's metric defaults from leaking between tests.

    SI merges a call's ``metric_params`` into shared class default dicts.
    Each test gets a copy, restored afterwards. No lock is held, so worker
    threads computing metrics are not blocked.

    No-op under SI < 0.101 (the legacy CI job collects one file here).
    """
    import spikeinterface as si
    from packaging.version import Version

    if Version(si.__version__) < Version("0.101"):
        yield
        return

    from spyglass.spikesorting.v2._core.si_metric_patches import (
        _si_metric_param_copies,
    )

    with _si_metric_param_copies():
        yield


# The per-PR smoke fixture. Only consumers that request ``smoke_nwb`` fetch it.
_SMOKE_FIXTURE = "mearec_polymer_smoke"


@pytest.fixture(scope="session")
def smoke_nwb():
    """Ensure the smoke NWB; a failed or corrupt download errors, not skips."""
    from tests.spikesorting.v2.fixtures._fetch import ensure_fixture

    return ensure_fixture(_SMOKE_FIXTURE, required=True)


def _eager_fetch_names():
    """Fixture stems to download at session start.

    The required fixtures, or all under ``SPYGLASS_V2_FETCH_FULL=1``. The
    smoke fixture is fetched lazily by ``smoke_nwb``.
    """
    import os

    from tests.spikesorting.v2.fixtures._fetch import FIXTURE_URLS

    if os.environ.get("SPYGLASS_V2_FETCH_FULL") == "1":
        return list(FIXTURE_URLS)
    required = os.environ.get("SPYGLASS_V2_REQUIRE_FIXTURES", "").split()
    return [n for n in required if n in FIXTURE_URLS]


def _missing_required_fixtures(required):
    """Names from ``SPYGLASS_V2_REQUIRE_FIXTURES`` that no file satisfies.

    A ``_fetch.py`` stem counts only as ``fixtures/<name>.nwb``; a raw data
    directory copy may be left from an earlier run. Other names (e.g.
    ``minirec20230622``) are looked up in the raw data directory.

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
    """Session-exit message for absent required fixtures.

    Groups stems by fix: no hosted URL in ``FIXTURE_URLS`` vs. a failed
    download.

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
    """Pre-fetch and check the fixtures this session requires.

    Existing files are hash-verified before reuse.
    """
    if session.config.getoption("v2_tier", default=None) == "unit":
        return

    import os
    import warnings

    from tests.spikesorting.v2.fixtures._fetch import (
        FixtureFetchError,
        FixtureIntegrityError,
        ensure_fixture,
    )

    for name in _eager_fetch_names():
        try:
            ensure_fixture(name, required=False)
        except FixtureIntegrityError as exc:
            # An existing invalid file would satisfy the existence checks
            # below and in consumers; it must never reach the tests.
            pytest.exit(f"[v2 fixtures] {exc}", returncode=1)
        except FixtureFetchError as exc:
            # Dependent tests skip via their ``_PATH.exists()`` guard.
            warnings.warn(f"[v2 fixtures] could not fetch {name}: {exc}")

    # A required fixture must exist, or its tests would skip and the run
    # look green. CI sets the list; unset locally.
    required = os.environ.get("SPYGLASS_V2_REQUIRE_FIXTURES", "").split()
    missing = _missing_required_fixtures(required)
    if missing:
        pytest.exit(_missing_fixtures_message(missing), returncode=1)


# Same fixture the lazy fetch above produces; named once so the two can't drift.
_DOWNSTREAM_FIXTURE_NAME = _SMOKE_FIXTURE
_DOWNSTREAM_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / f"{_DOWNSTREAM_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="package")
def populated_sorting(dj_conn, smoke_nwb):
    """Populate Recording -> ArtifactDetection -> Sorting for the smoke
    fixture, once per package.

    In conftest so sharded runs cannot leave the integrity tests looping
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

    # Unique session name: other modules' cleanup of the smoke session must
    # not cascade-delete these package-scoped rows.
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
    """A root ``CurationV2`` (no labels or merges) over the smoke sort.

    Clears the sort's curations before and after. Yields the
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
def planted_two_unit_sort(dj_conn, smoke_nwb):
    """A populated Sorting with two planted units (so a merge group exists).

    ``plant_sorter`` plants them on the smoke recording, whose own sort
    yields one unit. Tests clear curations around themselves.
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
    np.testing.assert_array_equal(
        (Sorting.Unit & sort_pk).fetch("unit_id", order_by="unit_id"),
        [0, 1],
        err_msg="The deterministic two-unit sorter must populate both units",
    )
    yield sort_pk
    _clear_curations_for(sort_pk)
    _clean_session_v2(session)


@pytest.fixture(scope="package")
def planted_three_unit_sort(dj_conn, smoke_nwb):
    """A populated Sorting with three planted units (0, 1, 2).

    Three units let a parent merge two and keep a second unit for a child to
    edit. Spikes are ~200 frames apart, so the 0.4 ms merge dedup removes
    none and merges conserve spikes. Tests clear curations around themselves.
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
    np.testing.assert_array_equal(
        (Sorting.Unit & sort_pk).fetch("unit_id", order_by="unit_id"),
        [0, 1, 2],
        err_msg="The deterministic three-unit sorter must populate every unit",
    )
    yield sort_pk
    _clear_curations_for(sort_pk)
    _clean_session_v2(session)


@pytest.fixture
def curation_evaluation_defaults(dj_conn):
    """Ensure the Lookup rows a ``CurationEvaluation`` needs exist.

    "minimal" metrics, "none" auto-curation rules, default waveform window.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    AnalyzerWaveformParameters.insert_default()


#: ``session_group_owner`` LabTeam of the chronic concat fixture and its tests.
CHRONIC_OWNER_TEAM = "chronic_concat_owner"


@pytest.fixture(scope="package")
def chronic_2_session_minirec(dj_conn, tmp_path_factory):
    """Two same-day + one next-day single-tetrode synthetic sessions.

    Identical channel positions, different planted-spike seeds. ``Recording``
    (``"default"`` preprocessing) is populated for the two same-day members
    only; the next-day session gives the same-day check a date to reject.
    Tests build and tear down their own ``SessionGroup`` /
    ``ConcatenatedRecording`` rows. ``CHRONIC_OWNER_TEAM`` groups are cleared
    on entry and teardown.

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

    # Member dicts hold only Member columns, for ``SessionGroup.create_group``.
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


#: Daily-concat intervals (s after first sample, each > 1 s). Both start in the
#: [2, 4) s float64 binade so SI's timestamp-derived sampling rates agree to
#: the 1e-9 Hz concatenation tolerance.
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
