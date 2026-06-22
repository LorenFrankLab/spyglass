"""Shared fixtures for the v2 single-session pipeline test suite.

These fixtures ingest MEArec smoke / 60s fixtures and build the
``Recording -> ArtifactDetection -> Sorting`` chain that the
single-session behavior tests exercise. They live here (rather than in
one test module) so the suite can be split across several
``test_*.py`` files in this directory while the heavy session-scoped
ingestion is still paid once.

The function-scoped ``populated_sorting`` defined here deliberately
shadows the package-scoped ``populated_sorting`` in the parent
``tests/spikesorting/v2/conftest.py``: the single-session tests run
against an isolated ``mearec_polymer_smoke`` session, while the parent
fixture serves the downstream/integrity suites against a separate
``mearec_downstream_smoke`` session. pytest's fixture-override
resolution keeps each suite bound to the right session, including the
parent's ``populated_sorting_with_curation`` (which resolves its
``populated_sorting`` dependency through the requesting test's closure).
"""

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)
_TETRODE_60S_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "mearec_tetrode_60s.nwb"
)
_NEUROPIXELS_60S_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "mearec_neuropixels_60s.nwb"
)


@pytest.fixture(scope="session")
def polymer_smoke_session(dj_conn):
    """Ingest the MEArec polymer smoke fixture and yield its session key.

    Session-scoped because NWB ingestion is the heaviest setup step
    (and intentionally not destroyed by ``_clean_session_v2``: the
    Session/Nwbfile rows live above the v2 tables and are reused across
    every test). Module scope previously implied per-module ingestion;
    promoting to session scope avoids that work without affecting test
    isolation -- the v2 tables that ARE mutated by tests use
    function-scoped fixtures below with idempotent ensure-exists setup.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}


@pytest.fixture
def recording_selection_key(polymer_smoke_session):
    """Ensure a RecordingSelection row exists for the smoke session.

    Function-scoped + ensure-exists: each test calling this fixture
    checks whether the upstream chain (SortGroupV2 + PreprocessingParameters
    default row + LabTeam) is present and rebuilds the missing pieces.
    The actual ``RecordingSelection.insert_selection`` call is
    idempotent (returns the existing PK if a matching row exists).

    Cost when state is intact: ~one DJ existence check per upstream
    table. Cost when state was wiped by a prior test: rebuild
    SortGroupV2 (~1s) and re-insert the selection (~0.1s). The
    expensive ``Recording.populate`` is deferred to the
    ``populated_recording`` fixture so callers that only need the
    selection PK aren't billed for materialization.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
        "team_name": "v2_test_team",
    }
    return RecordingSelection.insert_selection(selection_key)


@pytest.fixture(scope="session")
def polymer_60s_session(dj_conn):
    """Ingest the 60s polymer MEArec fixture for the ground-truth tests.

    Skips cleanly if the fixture has not been generated. Session-scoped
    so the ingestion (the heaviest single step in the suite) runs at
    most once even across many tests / reordered runs.
    """
    if not _POLYMER_60S_PATH.exists():
        pytest.skip(
            "60s polymer fixture not on disk -- run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py` "
            "(no --smoke) first."
        )
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    # ``ImportedSpikeSorting`` only reads ``nwbfile.units``; planted
    # ground-truth lives in the sidecar processing module now so
    # there is nothing for it to ingest from these synthetic
    # fixtures. The GT-accuracy test below reads the sidecar
    # directly via ``get_ground_truth_units_table``.
    yield {"nwb_file_name": nwb_file_name}


@pytest.fixture(scope="session")
def tetrode_60s_session(dj_conn):
    """Ingest the 60s tetrode MEArec fixture for the ground-truth tests.

    Single-tetrode probe (Frank-lab ``tetrode_12.5`` metadata): 1 shank
    × 4 contacts in a square at ±6.25 µm. Yields one sort group
    (sort_group_id=0) with 4 channels. Skips cleanly if the fixture is
    not on disk.
    """
    if not _TETRODE_60S_PATH.exists():
        pytest.skip(
            "60s tetrode fixture not on disk -- run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py` "
            "(no --smoke) first."
        )
    nwb_file_name = copy_and_insert_nwb(_TETRODE_60S_PATH)
    yield {"nwb_file_name": nwb_file_name}


@pytest.fixture
def populated_recording(recording_selection_key):
    """Ensure a Recording row exists for the smoke selection.

    Function-scoped + ensure-exists: ``Recording.populate`` is
    idempotent (no-ops if the row already exists), so the cost when
    state is intact is one DJ existence check. When a prior test
    destroyed Recording / its dependencies, this fixture rebuilds
    automatically -- no order coupling between tests.
    """
    from spyglass.spikesorting.v2.recording import Recording

    if not (Recording & recording_selection_key):
        Recording.populate(recording_selection_key, reserve_jobs=False)
    yield recording_selection_key


@pytest.fixture
def populated_sorting(populated_recording):
    """Ensure a Sorting row exists for the smoke fixture via MS5.

    Function-scoped + ensure-exists: ``Sorting.populate`` is the most
    expensive single step in the suite (~30s on the smoke fixture).
    Idempotent setup means the cost is paid once per pytest session
    in practice (DataJoint's ``populate`` is a no-op when the row
    already exists). When a prior test destroyed the Sorting row,
    this fixture rebuilds it; when it didn't, the existence check
    short-circuits.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    AnalyzerWaveformParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)
    yield sort_pk


@pytest.fixture(scope="session")
def neuropixels_60s_session(dj_conn):
    """Ingest the 60s Neuropixels MEArec fixture for GT comparison.

    Skips cleanly if the fixture has not been generated. The
    Neuropixels GT coverage is informational dense-probe
    correctness; the polymer fixture remains the shipping gate.
    """
    if not _NEUROPIXELS_60S_PATH.exists():
        pytest.skip(
            "Neuropixels-60s MEArec fixture not on disk -- run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--neuropixels --duration 60` (or equivalent) first. The "
            "dense-probe informational gate activates when the "
            "fixture is present."
        )
    nwb_file_name = copy_and_insert_nwb(_NEUROPIXELS_60S_PATH)
    # Planted GT lives in the sidecar processing module; the
    # GT-accuracy test reads it via ``get_ground_truth_units_table``.
    yield {"nwb_file_name": nwb_file_name}
