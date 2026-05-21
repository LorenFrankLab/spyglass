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


# ---------- ArtifactSelection source-part pattern -------------------------


@pytest.fixture
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
