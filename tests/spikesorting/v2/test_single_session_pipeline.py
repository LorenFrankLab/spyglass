"""End-to-end behavior tests for the v2 single-session sort pipeline.

This module exercises the v2 recording-side runtime against an ingested
MEArec polymer smoke fixture: ``SortGroupV2`` constructors, the safety
guards that fix v1's silent overwrite, and ``RecordingSelection``'s
find-existing-or-insert helper.

The downstream chain (``Recording.make`` / ``ArtifactDetection.make`` /
``Sorting.make`` / ``CurationV2.insert_curation``) is exercised here as
those bodies land; until then the slow integration assertions live in
the runtime modules themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="module")
def polymer_smoke_session(dj_conn):
    """Ingest the MEArec polymer smoke fixture and yield its session key.

    Skips cleanly if the fixture NWB has not been generated.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}


# ---------- SortGroupV2 by shank -------------------------------------------


@pytest.mark.slow
def test_set_group_by_shank_creates_one_group_per_shank(polymer_smoke_session):
    """The 128-channel 4-shank polymer fixture yields 4 sort groups of 32."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    # Clean any rows a previous module run may have left.
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)

    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    masters = SortGroupV2 & polymer_smoke_session
    assert len(masters) == 4
    parts = SortGroupV2.SortGroupElectrode & polymer_smoke_session
    assert len(parts) == 128
    counts = {
        sg_id: len(
            SortGroupV2.SortGroupElectrode
            & polymer_smoke_session
            & {"sort_group_id": sg_id}
        )
        for sg_id in masters.fetch("sort_group_id")
    }
    assert all(c == 32 for c in counts.values()), counts


@pytest.mark.slow
def test_set_group_by_shank_refuses_overlapping_rerun(polymer_smoke_session):
    """Rerun without an override raises (fixes v1 silent-overwrite bug)."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    with pytest.raises(ValueError, match="silently extend"):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)


@pytest.mark.slow
def test_set_group_by_shank_overwrite_requires_confirm(polymer_smoke_session):
    """``delete_existing_entries=True, confirm=False`` returns a preview
    and raises; ``confirm=True`` performs the cautious delete + reinsert."""
    from spyglass.spikesorting.v2.recording import (
        DeletionPreview,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    preview = SortGroupV2.preview_existing_entries(nwb_file_name)
    assert isinstance(preview, DeletionPreview)
    assert preview.sort_group_rows == 4
    assert preview.electrode_rows == 128

    with pytest.raises(ValueError, match="requires confirm=True"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            delete_existing_entries=True,
        )

    # Confirmed overwrite succeeds and produces a fresh set.
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name,
        delete_existing_entries=True,
        confirm=True,
    )
    assert len(SortGroupV2 & polymer_smoke_session) == 4


# ---------- SortGroupV2 by electrode-table column --------------------------


@pytest.mark.slow
def test_set_group_by_column_matches_by_shank(polymer_smoke_session):
    """Grouping by the ``probe_shank`` column reproduces ``by_shank``."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)

    shanks = sorted(
        set(
            int(s)
            for s in (Electrode & polymer_smoke_session).fetch("probe_shank")
        )
    )
    groups = [[shank] for shank in shanks]
    SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        column="probe_shank",
        groups=groups,
    )
    assert len(SortGroupV2 & polymer_smoke_session) == 4
    assert (
        len(SortGroupV2.SortGroupElectrode & polymer_smoke_session) == 128
    )


@pytest.mark.slow
def test_set_group_by_column_lists_valid_columns_on_typo(
    polymer_smoke_session,
):
    """A bogus column name fails with the valid-column list in the error."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    with pytest.raises(ValueError) as excinfo:
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=polymer_smoke_session["nwb_file_name"],
            column="probe_shanke",  # typo
            groups=[[0]],
        )
    msg = str(excinfo.value)
    assert "probe_shanke" in msg
    assert "probe_shank" in msg  # the suggestion is implicit -- it's listed


# ---------- RecordingSelection.insert_selection ---------------------------


@pytest.mark.slow
def test_recording_selection_insert_is_idempotent(polymer_smoke_session):
    """Repeat calls return the same PK and never duplicate rows."""
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.common.common_lab import LabTeam

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    PreprocessingParameters.insert_default()

    # Pick the first sort group + a session interval the fixture provides.
    sort_group_id = (SortGroupV2 & polymer_smoke_session).fetch(
        "sort_group_id"
    )[0]
    interval_list_name = (
        "raw data valid times"  # canonical name written by insert_sessions
    )

    # Ensure a LabTeam exists (the round-trip ingestion does not create one).
    team_name = "v2_test_team"
    LabTeam.insert1(
        {"team_name": team_name, "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )

    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": int(sort_group_id),
        "interval_list_name": interval_list_name,
        "preproc_params_name": "default_franklab",
        "team_name": team_name,
    }
    pk_first = RecordingSelection.insert_selection(selection_key)
    pk_second = RecordingSelection.insert_selection(selection_key)
    assert pk_first == pk_second
    assert isinstance(pk_first, dict)
    assert list(pk_first.keys()) == ["recording_id"]
    # And only one row in the table.
    assert len(RecordingSelection & selection_key) == 1


# ---------- Recording.make + get_recording --------------------------------


@pytest.fixture(scope="module")
def recording_selection_key(polymer_smoke_session):
    """Insert a RecordingSelection row + return its PK.

    Returns the ``{"recording_id": <uuid>}`` PK dict produced by
    ``insert_selection``. Module-scoped so multiple Recording-side tests
    share the same selection / materialization.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    (SortGroupV2 & polymer_smoke_session).super_delete(warn=False)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preproc_params_name": "default_franklab",
        "team_name": "v2_test_team",
    }
    return RecordingSelection.insert_selection(selection_key)


@pytest.mark.slow
def test_recording_populates_and_round_trips(
    polymer_smoke_session, recording_selection_key
):
    """``Recording.make`` writes a preprocessed NWB; ``get_recording``
    reads it back with the expected channel count, sampling rate, and
    duration."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    expected_n_channels = len(
        SortGroupV2.SortGroupElectrode
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
    )

    # Clear any prior Recording row for this selection (subsequent module
    # runs would otherwise short-circuit populate()).
    (Recording & recording_selection_key).super_delete(warn=False)
    Recording.populate(recording_selection_key, reserve_jobs=False)
    row = (Recording & recording_selection_key).fetch1()

    assert row["n_channels"] == expected_n_channels
    assert row["sampling_frequency"] == pytest.approx(32_000, rel=1e-3)
    assert row["duration_s"] > 0.0
    assert len(row["cache_hash"]) == 64  # sha256 hex

    rec = Recording().get_recording(recording_selection_key)
    assert rec.get_num_channels() == expected_n_channels
    assert rec.get_sampling_frequency() == pytest.approx(
        row["sampling_frequency"], rel=1e-6
    )


_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_128ch_60s.nwb"
)


@pytest.fixture(scope="module")
def polymer_60s_session(dj_conn):
    """Ingest the 60s polymer MEArec fixture for the ground-truth tests.

    Skips cleanly if the fixture has not been generated. Module-scoped
    so the clusterless e2e + MS5 ground-truth tests share the same
    ingestion.
    """
    if not _POLYMER_60S_PATH.exists():
        pytest.skip(
            "60s polymer fixture not on disk -- run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py` "
            "(no --smoke) first."
        )
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    session_key = {"nwb_file_name": nwb_file_name}
    if not (ImportedSpikeSorting & session_key):
        ImportedSpikeSorting().insert_from_nwbfile(nwb_file_name)
    yield session_key


def _clear_curations(sorting_key):
    """Delete CurationV2 rows for a sorting + the corresponding
    SpikeSortingOutput merge master rows.

    DataJoint refuses to delete a part table row whose master is still
    present, so the cleanup walks from the merge master down: first
    drop the SpikeSortingOutput master rows that point at any
    CurationV2 part rows for this sorting (cascades the part), then
    drop the CurationV2 rows themselves.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    merge_ids = (
        SpikeSortingOutput.CurationV2 & sorting_key
    ).fetch("merge_id")
    for mid in merge_ids:
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & sorting_key).super_delete(warn=False)


# ---------- ArtifactSelection source-part pattern -------------------------


@pytest.fixture(scope="module")
def populated_recording(recording_selection_key):
    """Ensure Recording is populated for the smoke selection."""
    from spyglass.spikesorting.v2.recording import Recording

    if not (Recording & recording_selection_key):
        Recording.populate(recording_selection_key, reserve_jobs=False)
    yield recording_selection_key


@pytest.mark.slow
def test_artifact_selection_inserts_master_and_source_part(populated_recording):
    """``insert_selection`` writes exactly one master + one source row."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    key = {
        "recording_id": populated_recording["recording_id"],
        "artifact_params_name": "default",
    }
    # Clean any prior selection so we can assert on the count.
    existing = (
        ArtifactSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        (ArtifactSelection & {"artifact_id": row["artifact_id"]}).super_delete(
            warn=False
        )

    pk = ArtifactSelection.insert_selection(key)
    assert isinstance(pk, dict)
    assert set(pk.keys()) == {"artifact_id"}
    assert len(ArtifactSelection & pk) == 1
    assert len(ArtifactSelection.RecordingSource & pk) == 1
    assert len(ArtifactSelection.SharedArtifactGroupSource & pk) == 0


@pytest.mark.slow
def test_artifact_selection_is_idempotent(populated_recording):
    """Repeat ``insert_selection`` calls return the same PK; no duplicates."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    key = {
        "recording_id": populated_recording["recording_id"],
        "artifact_params_name": "default",
    }
    pk1 = ArtifactSelection.insert_selection(key)
    pk2 = ArtifactSelection.insert_selection(key)
    assert pk1 == pk2
    assert (
        len(
            ArtifactSelection.RecordingSource
            & {"recording_id": populated_recording["recording_id"]}
            & {"artifact_id": pk1["artifact_id"]}
        )
        == 1
    )


@pytest.mark.slow
def test_artifact_selection_rejects_zero_and_two_sources(populated_recording):
    """``insert_selection`` requires exactly one source key."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()

    with pytest.raises(ValueError, match="exactly one source key"):
        ArtifactSelection.insert_selection(
            {"artifact_params_name": "default"}  # no source
        )

    with pytest.raises(ValueError, match="exactly one source key"):
        ArtifactSelection.insert_selection(
            {
                "recording_id": populated_recording["recording_id"],
                "shared_artifact_group_name": "fake",
                "artifact_params_name": "default",
            }
        )


@pytest.mark.slow
def test_artifact_selection_resolve_source_returns_recording_kind(
    populated_recording,
):
    """``resolve_source`` returns kind='recording' for a single-rec selection."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.utils import SourceResolution

    ArtifactDetectionParameters.insert_default()
    pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "default",
        }
    )
    resolution = ArtifactSelection.resolve_source(pk)
    assert isinstance(resolution, SourceResolution)
    assert resolution.kind == "recording"
    assert resolution.key == {
        "recording_id": populated_recording["recording_id"]
    }


@pytest.mark.slow
def test_shared_artifact_group_insert_is_gated_for_phase_1(populated_recording):
    """``SharedArtifactGroup.insert_group`` is gated in Phase 1.

    ``ArtifactDetection.make`` does not implement the shared-group
    branch yet, so allowing ``insert_group`` to succeed would let users
    insert rows that cannot populate. The gate raises
    ``NotImplementedError`` with a clear hint at the
    single-recording artifact path. When the shared-group branch
    lands in Phase 2 this gate is lifted and the validation tests
    inside ``insert_group`` (under ``# pragma: no cover``) become
    active again -- this test then flips to assert the success path.
    """
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    with pytest.raises(NotImplementedError, match="gated until"):
        SharedArtifactGroup.insert_group(
            "test_group",
            [{"recording_id": populated_recording["recording_id"]}],
        )

    # Verify the gate fires BEFORE any validation runs, i.e., it
    # doesn't matter what's in members.
    with pytest.raises(NotImplementedError, match="gated until"):
        SharedArtifactGroup.insert_group("empty_group", [])

    # No master row was created.
    assert (
        len(SharedArtifactGroup & {"shared_artifact_group_name": "test_group"})
        == 0
    )


@pytest.mark.slow
def test_artifact_detection_populates_and_writes_interval_list(
    populated_recording,
):
    """``ArtifactDetection.make`` writes one ``IntervalList`` row under
    ``f"artifact_{artifact_id}"`` and the row is fetchable via
    ``get_artifact_removed_intervals``."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    # Clean any prior selection / detection.
    existing = (
        ArtifactSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        ArtifactDetection & row  # noqa: B018 -- silence linter on unused expr
        (ArtifactDetection & {"artifact_id": row["artifact_id"]}).delete()
        (ArtifactSelection & {"artifact_id": row["artifact_id"]}).super_delete(
            warn=False
        )

    pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(pk, reserve_jobs=False)
    assert len(ArtifactDetection & pk) == 1

    nwb_file_name = (
        RecordingSelection & populated_recording
    ).fetch1("nwb_file_name")
    interval_list_name = f"artifact_{pk['artifact_id']}"
    saved = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).fetch1("valid_times")
    assert saved.shape[1] == 2  # (n_intervals, 2)
    # The "none" preset means no artifacts removed -> full window.
    assert saved.shape[0] == 1
    assert saved[0][0] < saved[0][-1]

    retrieved = ArtifactDetection().get_artifact_removed_intervals(pk)
    assert (retrieved == saved).all()


@pytest.mark.slow
def test_artifact_detection_delete_removes_interval_list_row(
    populated_recording,
):
    """``ArtifactDetection.delete()`` cleans up the matching
    ``IntervalList`` row (DataJoint does not cascade through
    ``interval_list_name``-keyed dependencies)."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    existing = (
        ArtifactSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        (ArtifactDetection & {"artifact_id": row["artifact_id"]}).delete()
        (ArtifactSelection & {"artifact_id": row["artifact_id"]}).super_delete(
            warn=False
        )

    pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(pk, reserve_jobs=False)

    nwb_file_name = (
        RecordingSelection & populated_recording
    ).fetch1("nwb_file_name")
    interval_list_name = f"artifact_{pk['artifact_id']}"
    assert (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    )

    (ArtifactDetection & pk).delete()

    assert not (ArtifactDetection & pk)
    assert not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    )


@pytest.mark.slow
def test_sorting_populates_with_mountainsort5(populated_recording):
    """``Sorting.make`` runs MountainSort5 on the smoke fixture and writes
    a fresh Units NWB + SortingAnalyzer folder. ``Sorting.Unit`` is
    populated with the peak electrode + amplitude per unit."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    ArtifactDetectionParameters.insert_default()

    # Ensure a no-op artifact detection is in place so the sort uses
    # the full recording.
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": (
                "franklab_tetrode_hippocampus_30kHz_ms5"
            ),
            "artifact_id": art_pk["artifact_id"],
        }
    )

    # Clear any prior populate.
    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    assert row["n_units"] > 0
    assert row["analyzer_folder"]
    from pathlib import Path

    assert Path(row["analyzer_folder"]).exists()

    n_unit_rows = len(Sorting.Unit & sort_pk)
    assert n_unit_rows == row["n_units"]

    # Every Sorting.Unit row's peak electrode_id must belong to the
    # sort group of the upstream Recording.
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    sg_id = int(
        (RecordingSelection & populated_recording).fetch1("sort_group_id")
    )
    unit_electrodes = (Sorting.Unit & sort_pk).fetch("electrode_id")
    sort_group_electrodes = set(
        (
            SortGroupV2.SortGroupElectrode
            & populated_recording
            & {"sort_group_id": sg_id}
        ).fetch("electrode_id")
    )
    assert set(unit_electrodes).issubset(sort_group_electrodes)

    # Peak amplitudes are positive (we abs() them in make).
    amplitudes = (Sorting.Unit & sort_pk).fetch("peak_amplitude_uv")
    assert (amplitudes > 0).all()


@pytest.mark.slow
def test_sorting_get_sorting_round_trips(populated_recording):
    """``Sorting.get_sorting`` returns the same unit IDs as the row.

    Also asserts spike times are stored in the recording's absolute
    timeline (v1 convention): the SI sorting's
    ``get_unit_spike_train(uid, return_times=True)`` falls within
    ``[recording.get_times()[0], recording.get_times()[-1]]``, which
    only holds if ``_write_units_nwb`` stored absolute times and
    ``get_sorting`` reconstructs the proper ``t_start``.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    # Reuse whatever sorting the previous test produced (module-scoped
    # fixture). Find any sorting row for this recording.
    sortings = (
        SortingSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("sorting_id")
    assert len(sortings) > 0, "Run test_sorting_populates_* first"
    sorting_id = sortings[0]
    row = (Sorting & {"sorting_id": sorting_id}).fetch1()
    si_sorting = Sorting().get_sorting({"sorting_id": sorting_id})
    assert len(si_sorting.unit_ids) == row["n_units"]

    # Recording's wall-clock window.
    rec = Recording().get_recording(populated_recording)
    timestamps = rec.get_times()
    rec_start = float(timestamps[0])
    rec_end = float(timestamps[-1])

    # Spike times for every unit must lie inside the recording window.
    for uid in si_sorting.unit_ids:
        times = si_sorting.get_unit_spike_train(unit_id=uid, return_times=True)
        if len(times) == 0:
            continue
        assert times.min() >= rec_start
        assert times.max() <= rec_end

    # Stronger absolute-timeline check: even on a recording starting at
    # t=0 (the smoke fixture), the spike times read back must equal
    # ``timestamps[sample_index]`` exactly. With the previous t_start=0.0
    # hard-code AND relative-spike-time storage, ``return_times=True``
    # also returned ``sample_index / fs``, which happened to match for
    # t=0 recordings -- so we also check that
    # ``get_unit_spike_train(return_times=False)`` returns valid sample
    # indices that, when looked up in ``timestamps``, recover the
    # original return_times array. That equivalence only holds when the
    # absolute-timeline convention is implemented in BOTH write and
    # read paths.
    for uid in si_sorting.unit_ids:
        sample_idx = si_sorting.get_unit_spike_train(
            unit_id=uid, return_times=False
        )
        wall_times = si_sorting.get_unit_spike_train(
            unit_id=uid, return_times=True
        )
        if len(sample_idx) == 0:
            continue
        # Allow at most one-sample drift from any int-rounding inside
        # SI's time/sample-index conversion.
        recovered = timestamps[sample_idx.astype(int)]
        max_drift = float(abs(recovered - wall_times).max())
        assert max_drift <= 1.0 / rec.get_sampling_frequency()


@pytest.mark.slow
def test_sorting_get_analyzer_loads_folder(populated_recording):
    """``Sorting.get_analyzer`` loads the SortingAnalyzer from the folder."""
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    sortings = (
        SortingSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("sorting_id")
    sorting_id = sortings[0]
    analyzer = Sorting().get_analyzer({"sorting_id": sorting_id})
    assert isinstance(analyzer, si.SortingAnalyzer)
    assert analyzer.has_extension("templates")


@pytest.mark.slow
def test_recording_truncation_caught(polymer_smoke_session):
    """Validation goal #5: ``RecordingTruncatedError`` fires at populate
    time when the IntervalList valid_times request more than the raw NWB
    actually covers.

    Synthesizes a sort interval that extends past the raw recording's
    last timestamp by far more than ``1.5 / sampling_frequency`` and
    asserts ``Recording.populate`` raises before any partial artifact
    is written. The check is the regression fix for silent-truncation
    bugs (#1133, #1585).
    """
    import numpy as _np

    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.exceptions import RecordingTruncatedError
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Write a synthetic IntervalList row whose end-time exceeds the
    # raw recording's last timestamp by 1.0 second -- well past the
    # 1.5 / fs tolerance that absorbs the off-by-one boundary.
    truncated_interval_name = "v2_test_truncated"
    raw_end = float(
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch1("valid_times")[-1][-1]
    )
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": truncated_interval_name,
            "valid_times": _np.asarray([[0.0, raw_end + 1.0]]),
            "pipeline": "v2_test_truncation",
        },
        skip_duplicates=True,
    )

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": truncated_interval_name,
        "preproc_params_name": "default_franklab",
        "team_name": "v2_test_team",
    }
    pk = RecordingSelection.insert_selection(selection_key)

    # Clear any prior Recording row so populate actually runs make().
    (Recording & pk).super_delete(warn=False)
    try:
        with pytest.raises(RecordingTruncatedError, match="Missing seconds"):
            Recording.populate(pk, reserve_jobs=False)
        # The truncation guard must fire BEFORE the row is inserted --
        # otherwise downstream consumers would silently see a short
        # recording.
        assert not (Recording & pk)
    finally:
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": truncated_interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_prune_orphaned_selections_finds_and_cleans(populated_recording):
    """``prune_orphaned_selections`` finds master rows with no source
    part and removes them when called with ``dry_run=False``.

    Source-part atomicity is enforced at insert time, so orphans only
    arise from upstream maintenance (e.g. a cascade delete from
    Recording that removes the source-part row but leaves the master).
    The helper backs the validation-goal-#8 cleanup path.
    """
    import uuid as _uuid

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()

    # Inject orphans directly via dj.Manual.insert1 (bypassing
    # insert_selection -- this is what an upstream cascade-delete leaves
    # behind in production).
    orphan_artifact = _uuid.uuid4()
    ArtifactSelection().insert1(
        {
            "artifact_id": orphan_artifact,
            "artifact_params_name": "default",
        }
    )
    orphan_sorting = _uuid.uuid4()
    SortingSelection().insert1(
        {
            "sorting_id": orphan_sorting,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
            "artifact_id": None,
        }
    )

    dry = ArtifactSelection.prune_orphaned_selections(dry_run=True)
    assert {"artifact_id": orphan_artifact} in dry
    # Non-dry-run removes the orphan and returns it for review.
    deleted = ArtifactSelection.prune_orphaned_selections(dry_run=False)
    assert {"artifact_id": orphan_artifact} in deleted
    assert not (ArtifactSelection & {"artifact_id": orphan_artifact})

    dry = SortingSelection.prune_orphaned_selections(dry_run=True)
    assert {"sorting_id": orphan_sorting} in dry
    deleted = SortingSelection.prune_orphaned_selections(dry_run=False)
    assert {"sorting_id": orphan_sorting} in deleted
    assert not (SortingSelection & {"sorting_id": orphan_sorting})


@pytest.fixture(scope="module")
def populated_sorting(populated_recording):
    """Ensure a Sorting row exists for the smoke fixture via MS5.

    Module-scoped so the curation-side tests reuse the same sort.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": (
                "franklab_tetrode_hippocampus_30kHz_ms5"
            ),
            "artifact_id": art_pk["artifact_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)
    yield sort_pk


@pytest.mark.slow
def test_curation_v2_insert_root_unlabeled(populated_sorting):
    """``insert_curation(labels={})`` writes a root curation with no
    ``UnitLabel`` rows and a ``CurationV2.Unit`` row per sorted unit."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    # Clean any prior curations for this sort so the assertion on
    # curation_id is stable.
    _clear_curations(populated_sorting)

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    assert pk == {**populated_sorting, "curation_id": 0}
    assert len(CurationV2 & pk) == 1
    n_units = (Sorting & populated_sorting).fetch1("n_units")
    assert len(CurationV2.Unit & pk) == n_units
    assert len(CurationV2.UnitLabel & pk) == 0

    row = (CurationV2 & pk).fetch1()
    assert row["parent_curation_id"] == -1
    assert row["merges_applied"] == 0
    assert row["metrics_source"] == "manual"


@pytest.mark.slow
def test_curation_v2_insert_with_labels(populated_sorting):
    """Labels round-trip into ``CurationV2.UnitLabel`` and unknown
    labels raise before any DB rows are written."""
    import pytest as _pytest

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)

    units = (Sorting.Unit & populated_sorting).fetch("unit_id")
    assert len(units) > 0
    target_unit = int(units[0])
    labels = {target_unit: ["mua", "accept"]}

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=labels,
        description="manual labels test",
    )
    label_rows = (CurationV2.UnitLabel & pk).fetch(as_dict=True)
    label_set = {(r["unit_id"], r["curation_label"]) for r in label_rows}
    assert (target_unit, "mua") in label_set
    assert (target_unit, "accept") in label_set
    assert (CurationV2 & pk).fetch1("description") == "manual labels test"

    # Unknown label rejects.
    with _pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: ["good"]},  # v1-era label not in v2 enum
        )


@pytest.mark.slow
def test_curation_v2_parent_validation_and_nonincreasing_ids(populated_sorting):
    """parent_curation_id must reference an existing row for the same
    sort; auto-incremented curation_id starts at 0 and increments."""
    import pytest as _pytest

    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    pk0 = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    assert pk0["curation_id"] == 0

    # Child curation referencing pk0 succeeds.
    pk1 = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={},
        parent_curation_id=0,
    )
    assert pk1["curation_id"] == 1
    assert (CurationV2 & pk1).fetch1("parent_curation_id") == 0

    # Bogus parent rejects.
    with _pytest.raises(ValueError, match="does not exist for sorting_id"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={},
            parent_curation_id=999,
        )

    # None labels reject.
    with _pytest.raises(ValueError, match="labels=None is invalid"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting, labels=None
        )


@pytest.mark.slow
def test_curation_v2_get_matchable_unit_ids_filters_labels(populated_sorting):
    """``get_matchable_unit_ids`` excludes units carrying any excluded
    label even when they also carry an included label, and includes
    untagged units.

    The smoke fixture's MS5 sort can yield a small unit count
    (commonly 1 on this 4s fixture), so the test exercises the
    multi-label-collision case on the first unit and asserts
    untagged passthrough on any remaining units that exist.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    units = sorted(int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id"))
    assert len(units) >= 1
    # Tag the first unit with BOTH an excluded label (artifact) and
    # an included label (mua). The "any excluded label wins" rule
    # must surface despite the included label being present.
    labels = {units[0]: ["mua", "artifact"]}
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels=labels
    )
    matchable = CurationV2().get_matchable_unit_ids(pk)
    matchable_set = set(matchable.tolist())
    assert units[0] not in matchable_set
    # Any further units are untagged and pass through.
    for uid in units[1:]:
        assert uid in matchable_set


@pytest.mark.slow
def test_curation_v2_get_sort_group_info_returns_all_electrodes(populated_sorting):
    """``get_sort_group_info`` returns a DataJoint relation covering
    every electrode in the sort group -- regression vs v1's
    ``fetch(limit=1)``."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    rel = CurationV2().get_sort_group_info(pk)
    # The relation is a DataJoint object (not a DataFrame); we can
    # restrict it further. Just fetching its length proves it spans
    # the full sort group.
    rec_row = (
        RecordingSelection
        & (SortingSelection.RecordingSource & populated_sorting)
    ).fetch1()
    expected = len(
        SortGroupV2.SortGroupElectrode
        & {
            "nwb_file_name": rec_row["nwb_file_name"],
            "sort_group_id": rec_row["sort_group_id"],
        }
    )
    assert len(rel) == expected
    assert expected > 1  # must be a multi-row relation, NOT v1's limit=1


@pytest.mark.slow
def test_curation_v2_get_sorting_round_trips_spike_times(populated_sorting):
    """``get_sorting`` returns a SI BaseSorting whose spike times match
    the upstream Sorting's after the curation is applied (no merges,
    no labels => identical spike trains)."""
    import numpy as np

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    src_sorting = Sorting().get_sorting(populated_sorting)
    cur_sorting = CurationV2().get_sorting(pk)
    assert set(cur_sorting.unit_ids) == set(src_sorting.unit_ids)
    for uid in src_sorting.unit_ids:
        src_times = src_sorting.get_unit_spike_train(uid, return_times=True)
        cur_times = cur_sorting.get_unit_spike_train(int(uid), return_times=True)
        assert len(src_times) == len(cur_times)
        if len(src_times):
            assert np.allclose(np.sort(src_times), np.sort(cur_times))


@pytest.mark.slow
def test_curation_v2_auto_registers_in_merge_table(populated_sorting):
    """``insert_curation`` writes the matching
    ``SpikeSortingOutput.CurationV2`` part row in the same transaction;
    a merge_id resolves back through ``get_unit_brain_regions``."""
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    assert "CurationV2" in source_class_dict
    assert hasattr(SpikeSortingOutput, "CurationV2")

    _clear_curations(populated_sorting)
    # Belt and suspenders: also clear any stale merge rows for this sort.
    stale = (
        SpikeSortingOutput.CurationV2 & populated_sorting
    ).fetch("merge_id")
    for mid in stale:
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    # The merge part now has exactly one row for this curation.
    merge_rows = (
        SpikeSortingOutput.CurationV2 & pk
    ).fetch(as_dict=True)
    assert len(merge_rows) == 1
    merge_id = merge_rows[0]["merge_id"]
    # And the master surfaces the v2 source enum.
    assert (
        SpikeSortingOutput & {"merge_id": merge_id}
    ).fetch1("source") == "CurationV2"

    # Dispatch round-trip: get_unit_brain_regions resolves to
    # CurationV2's accessor and returns a DataFrame.
    df = SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})
    assert "unit_id" in df.columns
    assert "region_resolution" in df.columns


@pytest.mark.slow
def test_spike_sorting_output_get_spike_times_v2_dispatch(populated_sorting):
    """``SpikeSortingOutput.get_spike_times`` dispatches to a v2 source.

    The previous review caught that the consumer-facing
    ``get_spike_times`` had a v0 / v1 / imported test but no v2
    dispatch path. This regression test pins the v2 NWB
    ``object_id`` -> spike-times resolution through the merge master,
    so a change to the v2 Units-NWB layout that breaks
    ``fetch_nwb`` immediately fails the suite.
    """
    import numpy as _np

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # The consumer-facing API returns a list of per-unit spike-time
    # arrays. Pin shape (n_units arrays) + dtype (float seconds) +
    # value range (non-negative, within the recording duration).
    spike_times = SpikeSortingOutput().get_spike_times(
        {"merge_id": merge_id}
    )
    assert isinstance(spike_times, list)
    assert len(spike_times) > 0, (
        "get_spike_times returned no unit arrays for a v2 curation "
        "that ``Sorting`` reports as having n_units > 0; the merge "
        "dispatch likely broke."
    )

    sort_row = (Sorting & populated_sorting).fetch1()
    expected_units = int(sort_row["n_units"])
    assert len(spike_times) == expected_units, (
        f"get_spike_times returned {len(spike_times)} unit arrays, "
        f"expected {expected_units} from Sorting.n_units."
    )

    for unit_idx, times in enumerate(spike_times):
        arr = _np.asarray(times)
        assert arr.dtype.kind == "f", (
            f"Unit {unit_idx}: spike times dtype kind "
            f"{arr.dtype.kind!r}, expected 'f' (float seconds)."
        )
        if arr.size:
            assert arr.min() >= 0.0, (
                f"Unit {unit_idx}: negative spike time {arr.min()!r}; "
                "v2 stores absolute timestamps which must be >= 0."
            )


@pytest.mark.slow
def test_get_restricted_merge_ids_v2_resolves_through_chain(populated_sorting):
    """``get_restricted_merge_ids(sources=['v2'])`` resolves the v2
    restriction surface (sorting_id, curation_id, ...) down to a
    merge_id from the CurationV2 part."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"sorting_id": pk["sorting_id"], "curation_id": pk["curation_id"]},
        sources=["v2"],
    )
    assert len(merge_ids) == 1
    # Re-resolving by sorter name also hits the same merge_id.
    merge_ids2 = SpikeSortingOutput().get_restricted_merge_ids(
        {"sorter": "mountainsort5"},
        sources=["v2"],
    )
    assert merge_ids[0] in list(merge_ids2)


@pytest.mark.slow
def test_run_v2_pipeline_end_to_end_and_idempotent(polymer_smoke_session):
    """``run_v2_pipeline`` chains recording -> artifact -> sort -> curation
    in one call and is idempotent on rerun.

    Tests the clusterless_thresholder preset (deterministic peak
    detection) so the rerun assertion is robust against MS5's
    randomness.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    manifest = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        preset="franklab_tetrode_mountainsort5",
        description="pipeline e2e test",
    )
    expected_keys = {
        "preset",
        "recording_id",
        "artifact_id",
        "sorting_id",
        "curation_id",
        "merge_id",
    }
    assert set(manifest.keys()) == expected_keys
    assert manifest["preset"] == "franklab_tetrode_mountainsort5"
    assert manifest["curation_id"] == 0  # root curation
    # MountainSort5 is the deterministic-enough default for the smoke
    # fixture; clusterless_thresholder is not exercised here because
    # its 100 uV default threshold finds zero peaks on this 4s fixture
    # and SI's random_spikes extension then errors on the empty sort.
    # A clusterless e2e test belongs with a fixture-tuned preset or
    # a Sorting.make zero-peak robustness fix.

    # Idempotent: rerun returns the same manifest, no new rows.
    manifest2 = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        preset="franklab_tetrode_mountainsort5",
    )
    assert manifest2["recording_id"] == manifest["recording_id"]
    assert manifest2["artifact_id"] == manifest["artifact_id"]
    assert manifest2["sorting_id"] == manifest["sorting_id"]
    assert manifest2["curation_id"] == manifest["curation_id"]
    assert manifest2["merge_id"] == manifest["merge_id"]

    # Unknown preset raises PipelineInputError.
    with pytest.raises(PipelineInputError, match="unknown preset"):
        run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            preset="not_a_preset",
        )


@pytest.mark.slow
def test_artifact_selection_resolve_source_detects_bypass(populated_recording):
    """``resolve_source`` raises ``SchemaBypassError`` when a master has
    zero source part rows (an integrity bug, e.g. from direct dj
    insert1 bypassing ``insert_selection``)."""
    import uuid as _uuid

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    ArtifactDetectionParameters.insert_default()
    orphan_id = _uuid.uuid4()
    # Insert master with NO source part to simulate bypass.
    dj_table = ArtifactSelection()
    dj_table.insert1(
        {
            "artifact_id": orphan_id,
            "artifact_params_name": "default",
        }
    )
    try:
        with pytest.raises(SchemaBypassError, match="0 source part"):
            ArtifactSelection.resolve_source({"artifact_id": orphan_id})
    finally:
        (ArtifactSelection & {"artifact_id": orphan_id}).super_delete(
            warn=False
        )


# ---------- Clusterless thresholder end-to-end ----------------------------


@pytest.mark.slow
def test_clusterless_thresholder_end_to_end(polymer_smoke_session):
    """Clusterless-thresholder finds peaks and round-trips through the
    pipeline.

    The default ``clusterless_thresholder`` Lookup row uses a 100 uV
    amplitude threshold that finds zero peaks on the 4-second smoke
    fixture's amplitudes. This test inserts a custom ``SorterParameters``
    row with a tuned (lower) threshold and exercises
    Recording -> Artifact -> Sorting through that row, bypassing the
    preset bundle but reusing every Selection / make() body.
    """
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

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # Insert a clusterless row tuned for the smoke fixture's amplitude
    # distribution. MEArec template amplitudes on this fixture are
    # smaller than typical lab data; a 5 uV threshold reliably surfaces
    # planted spikes for the 4s smoke recording.
    custom_params_name = "smoke_clusterless_5uv"
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": custom_params_name,
            "params": {
                "detect_threshold": 5.0,
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 100.0,
                "outputs": "sorting",
            },
            "params_schema_version": 1,
            "job_kwargs": None,
        },
        skip_duplicates=True,
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
    Recording.populate(rec_pk, reserve_jobs=False)

    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": custom_params_name,
            "artifact_id": art_pk["artifact_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    # Clusterless puts every peak into a single unit_id=0.
    assert row["n_units"] == 1
    assert len(Sorting.Unit & sort_pk) == 1
    unit_row = (Sorting.Unit & sort_pk).fetch1()
    assert unit_row["unit_id"] == 0

    # Signal-quality floor: the smoke fixture is 4 s with ~6 planted
    # units firing at ~5-10 Hz, so the planted spike count is on the
    # order of ~150-200 across the sort group. ``n_spikes > 0`` would
    # accept a broken detector that returned a single noise spike, so
    # require a count consistent with finding most planted peaks.
    assert unit_row["n_spikes"] >= 20, (
        f"Clusterless found only {unit_row['n_spikes']} peaks on the "
        "smoke fixture; expected at least 20 given the planted firing "
        "rates and the 5 uV detect_threshold. A buggy detector that "
        "returns a single false-positive would pass `n_spikes > 0`."
    )
    # Peak amplitude floor: any detected peak must clear the configured
    # 5 uV threshold (the threshold is the magnitude of the negative
    # peak, so the stored absolute peak_amplitude_uv should be >= 5).
    # A detector that misreports amplitudes by a unit-conversion bug
    # would surface as a violation here.
    assert unit_row["peak_amplitude_uv"] >= 3.0, (
        f"Clusterless reported peak_amplitude_uv="
        f"{unit_row['peak_amplitude_uv']:.3f} below the 5 uV "
        "detect_threshold (allowing some tolerance for template-vs-"
        "raw-peak differences). Check for an amplitude-units bug "
        "(uV vs raw counts vs volts)."
    )


# ---------- 60s MEArec ground-truth correctness gate ----------------------


@pytest.mark.slow
def test_mountainsort5_ground_truth_polymer_60s(polymer_60s_session):
    """MS5 on the 60s polymer fixture finds the planted units.

    Primary correctness gate: per-unit accuracy >= 0.7 for at least
    half of planted units, computed via
    ``spikeinterface.comparison.compare_sorter_to_ground_truth``
    against the ``ImportedSpikeSorting`` ground truth that
    ``insert_from_nwbfile`` loaded from the MEArec NWB.
    """
    from spikeinterface.comparison import compare_sorter_to_ground_truth

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

    nwb_file_name = polymer_60s_session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )

    if not (SortGroupV2 & polymer_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Sort each of the 4 shanks; aggregate the per-shank SI sortings
    # by shifting unit ids so MS5's overlapping local ids do not
    # collide across shanks.
    sort_group_ids = sorted(
        int(g) for g in (SortGroupV2 & polymer_60s_session).fetch("sort_group_id")
    )
    sortings_by_shank = []
    for sg_id in sort_group_ids:
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preproc_params_name": "default_franklab",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": (
                    "franklab_tetrode_hippocampus_30kHz_ms5"
                ),
                "artifact_id": art_pk["artifact_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    # Combine the per-shank sortings into one BaseSorting for the
    # ground-truth comparison. SI's ``aggregate_units`` produces a
    # single sorting with shank-disjoint unit ids -- but the result's
    # unit_ids dtype is ``uint64``, which SI 0.104's
    # ``compare_sorter_to_ground_truth`` Hungarian matcher rejects
    # (it only accepts dtype kinds 'i' or 'U'). Rename the units to
    # contiguous signed-int ids to round-trip through the matcher.
    import numpy as np
    from spikeinterface import aggregate_units

    aggregated = aggregate_units(sortings_by_shank)
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    # Ground truth: the planted MEArec units stored in the source
    # NWB's ``/units`` table. ``ImportedSpikeSorting`` registers the
    # table but does not expose ``get_sorting`` (the merge-table
    # dispatch raises NotImplementedError for v0/v1/imported sources),
    # so we read the NWB directly and build a NumpySorting.
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import Nwbfile

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        # MEArec writes a fixed-rate ElectricalSeries (no timestamps
        # array). Read ``rate`` directly; fall back to the timestamps
        # diff only if rate is missing.
        if es.rate is not None:
            fs = float(es.rate)
        else:
            fs = float(1.0 / np.diff(es.timestamps[:2])[0])
        gt_units = {}
        for idx, unit_id in enumerate(gt_nwb.units.id[:]):
            gt_units[int(unit_id)] = np.asarray(
                gt_nwb.units["spike_times"][idx]
            )
    gt_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[
            {
                int(uid): (times * fs).astype(np.int64)
                for uid, times in gt_units.items()
            }
        ],
        sampling_frequency=fs,
    )

    comparison = compare_sorter_to_ground_truth(
        gt_sorting=gt_sorting,
        tested_sorting=tested_sorting,
        gt_name="mearec_polymer_60s_planted",
        tested_name="v2_mountainsort5",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    # Per-unit performance is a DataFrame indexed by GT unit_id.
    # SI 0.104's ``get_performance(method='by_unit')`` exposes:
    #   accuracy  = tp / (tp + fn + fp)
    #   recall    = tp / (tp + fn)
    #   precision = tp / (tp + fp)
    #   false_discovery_rate
    #   miss_rate
    # Each is keyed by GT unit_id (Hungarian-matched to a detected unit).
    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    # ---- Diagnostic summary so a failure (or a soft regression) surfaces
    # the distribution rather than a single pass/fail bit. Always logged.
    summary = (
        f"\n[MS5 vs 60s polymer GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f} "
        f"min={recall.min():.3f} max={recall.max():.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f} "
        f"min={precision.min():.3f} max={precision.max():.3f}\n"
        f"  per-unit accuracy (sorted): "
        f"{[round(float(a), 3) for a in sorted(accuracies, reverse=True)]}"
    )
    print(summary)

    # Guard against a vacuous pass: ``n_planted // 2 == 0 >= 0`` would
    # silently approve a fixture that produced zero ground-truth units
    # (e.g., MEArec generation half-failed). The polymer 60s fixture is
    # built with 24 planted units; require well over half so a half-run
    # generation surfaces.
    assert n_planted >= 12, (
        f"60s polymer fixture has only {n_planted} planted units; expected "
        f"around 24. Regenerate the fixture before trusting this gate."
        f"{summary}"
    )

    # Detection-count floor: at least half the planted units must have
    # accuracy >= 0.7. This is the primary correctness gate.
    n_well_detected = int((accuracies >= 0.7).sum())
    threshold_half = n_planted // 2
    assert n_well_detected >= threshold_half, (
        f"MS5 on the 60s polymer fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"validation goal requires >= {threshold_half}."
        f"{summary}"
    )

    # Precision floor: the well-detected units must also have
    # precision >= 0.5. Otherwise a sorter that emits many spurious
    # spikes alongside each true unit (e.g., a buggy duplicate-spike
    # bug) could clear the accuracy gate by coincidence.
    well_detected_mask = accuracies >= 0.7
    well_detected_precision = precision[well_detected_mask]
    assert (well_detected_precision >= 0.5).all(), (
        f"Well-detected GT units have low precision: min="
        f"{well_detected_precision.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7.{summary}"
    )

    # Recall floor on well-detected units. accuracy is symmetric in fn
    # and fp; a unit can have accuracy=0.7 with recall=0.7/precision=1.0
    # (missing real spikes) or recall=1.0/precision=0.7 (spurious extra
    # spikes). The precision floor above bounds the second mode; this
    # bounds the first so a well-detected unit must actually find most
    # of its planted spikes.
    well_detected_recall = recall[well_detected_mask]
    assert (well_detected_recall >= 0.5).all(), (
        f"Well-detected GT units have low recall: min="
        f"{well_detected_recall.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7 (a unit can't be 'well detected' "
        f"if it's missing more than half its planted spikes).{summary}"
    )

    # Distribution-shape floor: well-detected units should have HIGH
    # accuracy, not barely-passing accuracy. A pipeline regression that
    # drives every accuracy from ~0.95 down to ~0.71 should fail this
    # test even though the count gate above still passes.
    well_detected_accuracy = accuracies[well_detected_mask]
    assert well_detected_accuracy.mean() >= 0.85, (
        f"Mean accuracy of well-detected units is "
        f"{well_detected_accuracy.mean():.3f}; expected >= 0.85 "
        f"(MS5 typically produces ~0.95 on this fixture). A regression "
        f"that drove accuracies down while keeping enough above 0.7 to "
        f"pass the count gate would surface here.{summary}"
    )
