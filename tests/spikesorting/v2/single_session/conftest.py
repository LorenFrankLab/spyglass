"""Shared fixtures for the v2 single-session pipeline tests.

``populated_sorting`` here shadows the parent conftest's on purpose: these
tests use the ``mearec_polymer_smoke`` session, the parent's suites use
``mearec_downstream_smoke``.
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
def polymer_smoke_session(dj_conn, smoke_nwb):
    """Ingest the MEArec polymer smoke fixture and yield its session key.

    Session-scoped: ingestion is slow and ``_clean_session_v2`` leaves the
    Session/Nwbfile rows in place.
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

    Function-scoped and idempotent: rebuilds any missing upstream rows.
    Does not populate ``Recording`` (see ``populated_recording``).
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

    Skips if the fixture has not been generated.
    """
    if not _POLYMER_60S_PATH.exists():
        pytest.skip(
            "60s polymer fixture not on disk -- run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py` "
            "(no --smoke) first."
        )
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    # Planted GT lives in the sidecar processing module; the GT-accuracy
    # test reads it via ``get_ground_truth_units_table``.
    yield {"nwb_file_name": nwb_file_name}


@pytest.fixture(scope="session")
def tetrode_60s_session(dj_conn):
    """Ingest the 60s tetrode MEArec fixture for the ground-truth tests.

    One ``tetrode_12.5`` shank (4 contacts at ±6.25 µm), one sort group
    (``sort_group_id=0``). Skips if the fixture is not on disk.
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

    Function-scoped and idempotent, so tests do not depend on run order.
    """
    from spyglass.spikesorting.v2.recording import Recording

    if not (Recording & recording_selection_key):
        Recording.populate(recording_selection_key, reserve_jobs=False)
    yield recording_selection_key


@pytest.fixture
def populated_sorting(populated_recording):
    """Ensure a Sorting row exists for the smoke fixture via MS5.

    Function-scoped and idempotent; the ~30 s populate runs only when a
    prior test removed the row.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
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

    Skips if the fixture has not been generated. Informational; the
    polymer fixture is the gating one.
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
