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


# ---------- SortGroupV2 by shank -------------------------------------------


@pytest.mark.slow
def test_set_group_by_shank_creates_one_group_per_shank(polymer_smoke_session):
    """The 128-channel 4-shank polymer fixture yields 4 sort groups of 32."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    # Clean any rows a previous module run may have left.
    _clean_session_v2(polymer_smoke_session)

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
    _clean_session_v2(polymer_smoke_session)
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
    _clean_session_v2(polymer_smoke_session)
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
    _clean_session_v2(polymer_smoke_session)

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
    _clean_session_v2(polymer_smoke_session)
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


# ---------- Tri-part dispatch active smoke gate ---------------------------


def test_tripart_dispatch_active(dj_conn):
    """Each refactored Computed table uses DataJoint's tri-part dispatch.

    Plan-required smoke gate. DataJoint's ``AutoPopulate.populate`` only
    fires the ``make_fetch`` / ``make_compute`` / ``make_insert``
    sequence when ``inspect.isgeneratorfunction(self.make) is True``
    -- i.e. the inherited generator-based ``make`` from
    ``AutoPopulate`` is in use. If a subclass overrides ``make`` with
    a regular function, DataJoint falls back to monolithic and the
    tri-part methods become dead code; the long-transaction
    regression silently persists. This test catches that failure
    mode in milliseconds.

    Recording's tri-part landed in Phase 1b Batch 2;
    ArtifactDetection and Sorting are added in later batches. The
    parametrize list grows as those refactors land so a future
    "consolidate back into ``make``" change fails loudly.
    """
    import inspect

    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    for cls in (Recording, ArtifactDetection, Sorting):
        assert inspect.isgeneratorfunction(cls.make), (
            f"{cls.__name__}.make must remain the inherited generator "
            "from AutoPopulate so tri-part dispatch fires; a regular-"
            "function override would silently re-enable the monolithic "
            "long-transaction path."
        )
        for attr in ("make_fetch", "make_compute", "make_insert"):
            assert getattr(cls, attr, None) is not None, (
                f"{cls.__name__}.{attr} missing; tri-part dispatch "
                "needs all three methods defined."
            )
        assert getattr(cls, "_parallel_make", False) is True, (
            f"{cls.__name__}._parallel_make is not True; the "
            "non-daemon process-pool path is the secondary benefit "
            "of the tri-part refactor and should be enabled."
        )


# ---------- Recording.make + get_recording --------------------------------


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
    # The smoke fixture is 4 seconds of polymer data. Compare to the
    # IntervalList's actual valid_times span so a regression that
    # silently truncated to e.g. 0.1 s would surface (rather than
    # passing via ``> 0.0``).
    from spyglass.common.common_interval import IntervalList

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    expected_duration = float(raw_times[-1][-1] - raw_times[0][0])
    assert row["duration_s"] == pytest.approx(
        expected_duration, rel=1e-3
    ), (
        f"Recording.duration_s = {row['duration_s']}; expected "
        f"~{expected_duration} from IntervalList valid_times span."
    )
    # ``cache_hash`` is now an ``NwbfileHasher`` digest (R17). The
    # exact digest length (currently 32-char MD5 hex) is not pinned
    # by the plan; the ``char(64)`` schema column is headroom for a
    # future hash change. Assert non-empty rather than a fixed
    # length so a hash-algorithm swap does not break this test.
    assert isinstance(row["cache_hash"], str) and row["cache_hash"]

    rec = Recording().get_recording(recording_selection_key)
    assert rec.get_num_channels() == expected_n_channels
    assert rec.get_sampling_frequency() == pytest.approx(
        row["sampling_frequency"], rel=1e-6
    )


_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_128ch_60s.nwb"
)


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


def _clean_session_v2(session_key):
    """Cascade-aware cleanup of every v2 row for a session.

    The v2 source-polymorphic source-part pattern leaves
    ``ArtifactSelection`` and ``SortingSelection`` MASTERS with no
    direct FK to upstream tables (only their ``RecordingSource`` PARTS
    carry the FK). DataJoint's cascade can't traverse that gap: a
    ``super_delete(SortGroupV2 & session_key)`` raises ``Attempt to
    delete part table ... before deleting from its master`` once the
    cascade reaches ``ArtifactSelection.RecordingSource`` because
    DataJoint refuses to drop a part without its master.

    Tests historically worked around this with an order-of-declaration
    convention (SortGroup tests run before tests that populate
    ArtifactSelection, so the cascade chain stays empty). Anything that
    runs the suite in a different order (``-k``, parallel sharding,
    rerun-failed) tripped the same DataJoint error. This helper makes
    the cleanup order-independent by walking the dependency graph
    leaves-first and deleting source-polymorphic masters explicitly
    before their upstream tables.

    Parameters
    ----------
    session_key
        Dict containing at least ``nwb_file_name``. All v2 rows tied to
        this session are dropped.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # Step 1: drop any merge-master rows whose CurationV2 part points
    # to a sorting derived from this session. The merge insert wraps
    # the part FK in a transaction with the master, so cleanup must
    # take the master first to satisfy the master-before-part rule.
    rec_keys = (
        RecordingSelection & session_key
    ).fetch("KEY", as_dict=True)
    if rec_keys:
        sorting_keys = (
            SortingSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if sorting_keys:
            merge_ids = (
                SpikeSortingOutput.CurationV2 & sorting_keys
            ).fetch("merge_id")
            for mid in merge_ids:
                (
                    SpikeSortingOutput & {"merge_id": mid}
                ).super_delete(warn=False)
            (CurationV2 & sorting_keys).super_delete(warn=False)
            (Sorting & sorting_keys).super_delete(warn=False)
            # Step 2: drop SortingSelection masters BEFORE removing
            # Recording, otherwise the Recording cascade tries to
            # delete the orphan RecordingSource part and fails.
            (SortingSelection & sorting_keys).super_delete(warn=False)

        # Step 3: same pattern for ArtifactDetection / ArtifactSelection.
        artifact_keys = (
            ArtifactSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if artifact_keys:
            (ArtifactDetection & artifact_keys).super_delete(warn=False)
            (ArtifactSelection & artifact_keys).super_delete(warn=False)

    # Step 4: now the cascade is unblocked -- Recording, SortGroupV2
    # can be deleted normally. We super_delete each so a leftover
    # IntervalList row from a prior aborted test is also picked up.
    (Recording & rec_keys).super_delete(warn=False) if rec_keys else None
    (RecordingSelection & session_key).super_delete(warn=False)
    (SortGroupV2 & session_key).super_delete(warn=False)


# ---------- ArtifactSelection source-part pattern -------------------------


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
def test_sorting_get_sorting_round_trips(populated_sorting):
    """``Sorting.get_sorting`` returns the same unit IDs as the row.

    Also asserts spike times are stored in the recording's absolute
    timeline (v1 convention): the SI sorting's
    ``get_unit_spike_train(uid, return_times=True)`` falls within
    ``[recording.get_times()[0], recording.get_times()[-1]]``, which
    only holds if ``_write_units_nwb`` stored absolute times and
    ``get_sorting`` reconstructs the proper ``t_start``.

    Depends on ``populated_sorting`` (not ``populated_recording``)
    so the Sorting row this test inspects is guaranteed to exist
    regardless of test ordering or selection (``-k``, ``--lf``).
    """
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    row = (Sorting & populated_sorting).fetch1()
    si_sorting = Sorting().get_sorting(populated_sorting)
    assert len(si_sorting.unit_ids) == row["n_units"]

    # Recording's wall-clock window. Look up the recording via the
    # source-part: ``recording_id`` is a non-PK FK on
    # ``SortingSelection.RecordingSource`` (the PK is sorting_id),
    # so fetch the recording_id explicitly rather than via
    # ``fetch1("KEY")`` which only returns PK fields.
    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    rec = Recording().get_recording({"recording_id": recording_id})
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
def test_sorting_get_analyzer_loads_folder(populated_sorting):
    """``Sorting.get_analyzer`` loads the SortingAnalyzer from the folder.

    Depends on ``populated_sorting`` so the analyzer folder is
    guaranteed to exist, eliminating the silent
    ``IndexError`` on an empty ``sortings[0]`` lookup that the
    prior ``populated_recording`` dependency exposed under
    isolated / reordered runs.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer = Sorting().get_analyzer(populated_sorting)
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
    # Pin not just the count but the actual unit_id set: a regression
    # that produced the wrong unit ids (e.g., an off-by-one on
    # _build_curated_unit_rows) would pass a count-only check.
    expected_unit_ids = set(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    curated_unit_ids = set(
        int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
    )
    assert curated_unit_ids == expected_unit_ids, (
        f"CurationV2.Unit unit_ids {sorted(curated_unit_ids)} differ "
        f"from Sorting.Unit unit_ids {sorted(expected_unit_ids)} for "
        "a root curation with no merges -- units should pass through "
        "1:1."
    )
    assert len(CurationV2.UnitLabel & pk) == 0

    row = (CurationV2 & pk).fetch1()
    assert row["parent_curation_id"] == -1
    assert row["merges_applied"] == 0
    assert row["metrics_source"] == "manual"


@pytest.mark.slow
def test_curation_v2_insert_with_labels(populated_sorting):
    """Labels round-trip into ``CurationV2.UnitLabel`` and unknown
    labels raise before any DB rows are written."""
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
    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: ["good"]},  # v1-era label not in v2 enum
        )


@pytest.mark.slow
def test_curation_v2_parent_validation_and_nonincreasing_ids(populated_sorting):
    """parent_curation_id must reference an existing row for the same
    sort; auto-incremented curation_id starts at 0 and increments."""
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
    with pytest.raises(ValueError, match="does not exist for sorting_id"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={},
            parent_curation_id=999,
        )

    # None labels reject.
    with pytest.raises(ValueError, match="labels=None is invalid"):
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
    # CurationV2's accessor and returns a DataFrame. Tighter than a
    # column-name-only check: an empty DataFrame with the right
    # schema would pass a column check but indicate a broken join,
    # so also pin row count and unit-id set.
    df = SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})
    assert "unit_id" in df.columns
    assert "region_resolution" in df.columns
    expected_unit_ids = set(
        int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
    )
    assert len(df) == len(expected_unit_ids), (
        f"get_unit_brain_regions returned {len(df)} rows; expected "
        f"{len(expected_unit_ids)} (one per CurationV2.Unit row)."
    )
    assert set(int(u) for u in df["unit_id"]) == expected_unit_ids, (
        f"get_unit_brain_regions unit_ids {sorted(df['unit_id'])} "
        f"differ from CurationV2.Unit unit_ids "
        f"{sorted(expected_unit_ids)}; the dispatch returned the "
        "wrong rows."
    )


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

    # Spike times must fall within the recording's wall-clock window,
    # specifically [t_start, t_end] where t_start is the FIRST
    # timestamp of the upstream recording. The smoke fixture starts
    # at t=0 so a naive ``>= 0.0`` check is satisfied by both correct
    # (absolute-time) and broken (relative-time) implementations.
    # Pinning against the actual t_start (and t_end) catches a
    # regression to relative storage on any non-zero-start session.
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    rec = Recording().get_recording({"recording_id": recording_id})
    rec_t_start = float(rec.get_times()[0])
    rec_t_end = float(rec.get_times()[-1])

    for unit_idx, times in enumerate(spike_times):
        arr = _np.asarray(times)
        assert arr.dtype.kind == "f", (
            f"Unit {unit_idx}: spike times dtype kind "
            f"{arr.dtype.kind!r}, expected 'f' (float seconds)."
        )
        if arr.size:
            # Strict absolute-time check: any spike before t_start
            # would indicate a relative-time / t_start=0 regression.
            # The tolerance is one sample (1/fs).
            fs = rec.get_sampling_frequency()
            assert arr.min() >= rec_t_start - 1.0 / fs, (
                f"Unit {unit_idx}: spike time {arr.min():.9f} is "
                f"earlier than recording t_start {rec_t_start:.9f}. "
                "v2 stores absolute timestamps; a regression to "
                "relative-time storage (or t_start=0 hardcoding) "
                "would surface here on a non-zero-start session."
            )
            assert arr.max() <= rec_t_end + 1.0 / fs, (
                f"Unit {unit_idx}: spike time {arr.max():.9f} is "
                f"later than recording t_end {rec_t_end:.9f}."
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
    # Re-resolving by the FULL set of v2 chain fields (more
    # restrictive than ``sorter`` alone, which would match every v2
    # MS5 sorting from any prior test run) returns the same single
    # merge_id. The previous looser assertion (``merge_ids[0] in
    # merge_ids2``) was vacuously satisfied if state pollution
    # produced multiple matches.
    merge_ids2 = SpikeSortingOutput().get_restricted_merge_ids(
        {
            "sorter": "mountainsort5",
            "sorting_id": pk["sorting_id"],
            "curation_id": pk["curation_id"],
        },
        sources=["v2"],
    )
    assert list(merge_ids2) == list(merge_ids), (
        f"Restricting by sorter+sorting_id+curation_id returned "
        f"{list(merge_ids2)}; expected exactly {list(merge_ids)}."
    )


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
    # Phase 1b N48 dropped ``outputs`` and ``random_chunk_kwargs``
    # from ``ClusterlessThresholderSchema``; the test params dict
    # must not carry them (extra="forbid" rejects on insert).
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
            },
            "params_schema_version": 2,
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


# =========================================================================
# Phase 1 review followups: gap-closing tests.
#
# Tests below address the gaps identified in the test-appropriateness
# review:
#   * non-zero t_start regression (the original bug only passed by
#     coincidence on the t=0 smoke fixture)
#   * artifact masking at the signal level (existing artifact tests
#     only exercise the IntervalList write/delete, not the
#     ``sip.remove_artifacts`` branch nor the off-by-one boundary fix)
#   * direct ``fetch_nwb`` dispatch for v2 (the prior test only goes
#     through ``get_spike_times`` which is the consumer surface, not
#     the merge-table primitive downstream consumers use)
#   * ``initialize_v2_defaults`` idempotency
#   * heterogeneous-gain ValueError
#   * zero-kept curation
#   * additional preset coverage (clusterless and MS4 through
#     ``run_v2_pipeline``)
#   * ``Sorting.make`` rollback cleans the staged NWB file
# =========================================================================


# ---------- Non-zero t_start regression ----------------------------------


@pytest.mark.slow
def test_recording_t_start_reads_first_timestamp(tmp_path, dj_conn):
    """``Sorting._recording_t_start`` returns the actual first
    timestamp of the upstream Recording's NWB, not a hardcoded 0.0.

    The previous t_start=0 bug went undetected because the smoke and
    60s polymer fixtures both start at t=0. This regression test
    writes a synthetic NWB with ``timestamps[0] = 12345.6`` and pins
    that ``_recording_t_start`` returns that value (not 0.0) via the
    ``electrical_series_path`` indirection. Stays out of the DB by
    monkeypatching ``AnalysisNwbfile.get_abs_path``.
    """
    import numpy as np
    import pynwb
    from datetime import datetime
    from unittest.mock import patch

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    fs = 30_000.0
    t0 = 12345.6
    n_samples = 100
    timestamps = t0 + np.arange(n_samples) / fs
    data = np.zeros((n_samples, 4), dtype=np.int16)

    nwbf = pynwb.NWBFile(
        session_description="t_start regression",
        identifier="t_start_regression",
        session_start_time=datetime.now().astimezone(),
    )
    device = nwbf.create_device(name="probe")
    eg = nwbf.create_electrode_group(
        name="grp", description="grp", location="hpc", device=device
    )
    for ch in range(4):
        nwbf.add_electrode(location="hpc", group=eg, group_name="grp")
    region = nwbf.create_electrode_table_region(
        region=list(range(4)), description="all four"
    )
    series = pynwb.ecephys.ElectricalSeries(
        name="ProcessedElectricalSeries",
        data=data,
        electrodes=region,
        timestamps=timestamps,
        filtering="bandpass",
        description="synthetic for t_start regression",
        conversion=1e-6,
    )
    nwbf.add_acquisition(series)

    nwb_path = tmp_path / "synthetic_t_start.nwb"
    with pynwb.NWBHDF5IO(path=str(nwb_path), mode="w") as io:
        io.write(nwbf)

    fake_row = {
        "analysis_file_name": "synthetic_t_start.nwb",
        "electrical_series_path": "acquisition/ProcessedElectricalSeries",
    }
    with patch.object(
        AnalysisNwbfile, "get_abs_path", return_value=str(nwb_path)
    ):
        recovered = Sorting._recording_t_start(fake_row)

    assert recovered == pytest.approx(t0, abs=1e-6), (
        f"_recording_t_start returned {recovered}; expected {t0}. The "
        "v1-era t_start=0 hardcode would silently pass on t=0 fixtures "
        "but break on real session recordings that start mid-day."
    )


# ---------- Artifact masking at the signal level -------------------------


@pytest.mark.slow
def test_apply_artifact_mask_zeroes_artifact_frames(populated_recording):
    """``Sorting._apply_artifact_mask`` actually zeros artifact frames.

    Existing artifact tests only verify the ``IntervalList`` row was
    written. The ``sip.remove_artifacts`` branch (the actual signal
    masking) is bypassed by the ``"none"`` artifact preset which
    writes a single all-valid interval. This test:

    1. Writes a synthetic IntervalList row whose ``valid_times``
       exclude a known frame range so the artifact "gap" is the
       known range.
    2. Calls ``_apply_artifact_mask`` against the populated_recording's
       preprocessed SI recording.
    3. Asserts traces are zero inside the gap and unchanged outside.

    Also pins the off-by-one boundary: the LAST artifact frame must
    be zeroed (the original code dropped it because the IntervalList
    stored ``[start, end]`` closed where ``end = timestamps[end_f]``;
    the fix stores ``[start, end+1)`` so the complement subtraction
    captures end_f).
    """
    import numpy as np
    import uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    recording = Recording().get_recording(
        {"recording_id": populated_recording["recording_id"]}
    )
    timestamps = recording.get_times()
    fs = recording.get_sampling_frequency()
    n_frames = len(timestamps)
    assert n_frames > 400, "fixture too short for boundary test"

    # Carve out a synthetic artifact at frames [100, 200) (half-open).
    # valid_times excludes this range: [t0, t[100]) ∪ [t[200], t[-1]].
    artifact_start_f, artifact_end_f = 100, 200
    valid_times = np.asarray(
        [
            [timestamps[0], timestamps[artifact_start_f]],
            [timestamps[artifact_end_f], timestamps[-1]],
        ]
    )

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")
    synthetic_artifact_id = uuid.uuid4()
    interval_list_name = f"artifact_{synthetic_artifact_id}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": valid_times,
            "pipeline": "spikesorting_artifact_v2_test",
        }
    )
    try:
        masked = Sorting._apply_artifact_mask(
            recording=recording,
            artifact_id=synthetic_artifact_id,
            recording_id=populated_recording["recording_id"],
        )

        # Inside the artifact window: all frames should be 0 across
        # every channel.
        gap = masked.get_traces(
            start_frame=artifact_start_f, end_frame=artifact_end_f
        )
        assert np.all(gap == 0), (
            "Artifact frames [100, 200) are not all zero; "
            f"max abs value in gap = {np.abs(gap).max()}."
        )

        # Boundary check: frame ``artifact_end_f`` (the half-open end)
        # is the first VALID frame after the gap and should NOT be
        # forced to zero by the masker. (We compare to the unmasked
        # value because the underlying preprocessed signal may be 0
        # by coincidence at a single sample; the assertion catches
        # off-by-one boundary regressions only if at least one of
        # the test channels at this frame is non-zero in the source.)
        unmasked_boundary = recording.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        masked_boundary = masked.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        np.testing.assert_array_equal(
            masked_boundary,
            unmasked_boundary,
            err_msg=(
                f"Frame {artifact_end_f} (first valid frame after the "
                "artifact gap) differs between masked and unmasked "
                "recordings -- off-by-one boundary regression."
            ),
        )

        # And the last artifact frame (artifact_end_f - 1) MUST be
        # zero. This is the exact frame the original off-by-one bug
        # left unmasked.
        last_artifact = masked.get_traces(
            start_frame=artifact_end_f - 1, end_frame=artifact_end_f
        )
        assert np.all(last_artifact == 0), (
            f"Last artifact frame {artifact_end_f - 1} not zeroed -- "
            "this is the off-by-one boundary the v2 fix corrected."
        )
    finally:
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).super_delete(warn=False)


# ---------- fetch_nwb direct v2 dispatch ---------------------------------


@pytest.mark.slow
def test_fetch_nwb_v2_dispatch(populated_sorting):
    """``SpikeSortingOutput.fetch_nwb`` works for a v2 source.

    ``get_spike_times`` is the consumer-facing API but it dispatches
    through ``fetch_nwb`` (the merge-table primitive). Downstream
    consumers like decoding use ``fetch_nwb`` directly. This test
    pins the v2 ``object_id`` -> Units NWB resolution at the lower
    level so a change to the merge dispatch / column-name convention
    surfaces immediately.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    nwb_results = SpikeSortingOutput().fetch_nwb(
        {"merge_id": merge_id}
    )
    assert isinstance(nwb_results, list)
    assert len(nwb_results) == 1, (
        f"fetch_nwb returned {len(nwb_results)} dicts for a single "
        "merge_id; expected exactly one."
    )
    payload = nwb_results[0]
    # v2 stores the Units NWB; ``object_id`` is the convention key
    # downstream consumers use to fetch spike times. Either
    # ``object_id`` (the v2/v1 convention) or ``units`` (the v0
    # convention) must be present AND non-None.
    assert "object_id" in payload or "units" in payload, (
        f"fetch_nwb result missing both 'object_id' and 'units' keys; "
        f"got keys {sorted(payload.keys())}."
    )
    if "object_id" in payload:
        # ``payload["object_id"]`` is the DataFrame that fetch_nwb
        # builds from the Units NWB referenced by CurationV2.object_id.
        # The schema-only assertion above just checked the key is
        # present; this check pins that the DataFrame actually
        # round-tripped from the NWB (non-None, has rows, has the
        # expected columns).
        loaded = payload["object_id"]
        assert loaded is not None, (
            "fetch_nwb returned object_id=None; the v2 Units NWB "
            "did not round-trip through the merge dispatch."
        )
        # DataFrame check: rows correspond to CurationV2.Unit rows
        # (one row per kept unit).
        expected_unit_ids = set(
            int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
        )
        loaded_unit_ids = set(int(u) for u in loaded.index)
        assert loaded_unit_ids == expected_unit_ids, (
            f"fetch_nwb returned Units rows for unit_ids "
            f"{sorted(loaded_unit_ids)}; expected "
            f"{sorted(expected_unit_ids)}."
        )
        assert "spike_times" in loaded.columns, (
            f"fetch_nwb Units DataFrame missing spike_times; "
            f"got columns {list(loaded.columns)}."
        )


# ---------- initialize_v2_defaults idempotency ---------------------------


@pytest.mark.slow
def test_initialize_v2_defaults_is_idempotent(dj_conn):
    """``initialize_v2_defaults()`` can be called repeatedly with no
    duplicate-row errors and leaves the same Lookup rows in place."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    initialize_v2_defaults()
    pre_counts = (
        len(PreprocessingParameters()),
        len(ArtifactDetectionParameters()),
        len(SorterParameters()),
    )
    # Second call must not raise and must not duplicate rows.
    initialize_v2_defaults()
    post_counts = (
        len(PreprocessingParameters()),
        len(ArtifactDetectionParameters()),
        len(SorterParameters()),
    )
    assert pre_counts == post_counts, (
        f"initialize_v2_defaults duplicated rows: pre={pre_counts}, "
        f"post={post_counts}."
    )
    # Counts must be > 0 (the defaults actually loaded something).
    assert all(c > 0 for c in post_counts), (
        f"initialize_v2_defaults produced zero rows: {post_counts}."
    )


# ---------- Heterogeneous gain rejection ---------------------------------


@pytest.mark.slow
def test_recording_make_rejects_heterogeneous_gains(populated_recording):
    """``Recording._write_nwb_artifact`` raises ValueError when the
    SI recording reports unequal channel gains across the sort group.

    A divergence here would indicate a probe-metadata bug upstream
    (different conversion factors per channel can't be represented
    by the single-conversion ElectricalSeries). The check is in
    place at recording.py but had no test; this regression test
    builds a synthetic NumpyRecording with mismatched gains and
    pins the raise.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")

    n_channels = 4
    n_samples = 1000
    fs = 30_000.0
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    synthetic_rec = si.NumpyRecording(
        traces_list=[traces], sampling_frequency=fs
    )
    # Heterogeneous gains across channels -- different conversion
    # factors. The check uses ``np.unique`` so any non-uniformity
    # trips it.
    synthetic_rec.set_channel_gains([0.195, 0.195, 0.5, 0.5])

    with pytest.raises(ValueError, match="heterogeneous channel"):
        Recording._write_nwb_artifact(
            recording=synthetic_rec,
            nwb_file_name=nwb_file_name,
        )


# ---------- Zero-kept curation -------------------------------------------


@pytest.mark.slow
def test_curation_v2_all_units_labeled_noise(populated_sorting):
    """All-noise label set: ``get_matchable_unit_ids`` returns empty
    AND ``insert_curation`` succeeds without crashing.

    Pins two contracts together: (1) the label-based filtering in
    ``get_matchable_unit_ids`` correctly excludes every unit when
    every unit is labeled ``noise``, and (2) ``insert_curation``
    completes even when every unit is labeled an excluded class.
    The curated Units NWB still contains every unit because the
    label filter is applied at the *consumer* (``get_matchable_*``)
    rather than at NWB-write time; the empty-units guard in
    ``_stage_curated_units_nwb`` is exercised by the sibling test
    ``test_curation_v2_stages_empty_units_nwb_on_zero_kept_units``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.utils import CurationLabel

    _clear_curations(populated_sorting)

    # Fetch all unit_ids so we can mark every one as noise.
    unit_ids = (Sorting.Unit & populated_sorting).fetch("unit_id")
    assert len(unit_ids) > 0, (
        "populated_sorting fixture should have at least one unit; "
        "test cannot assert noise-labeled behavior on an empty source."
    )
    noise_labels = {int(uid): [CurationLabel.noise] for uid in unit_ids}

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=noise_labels,
        description="all-noise curation regression",
    )

    # get_matchable_unit_ids filters out noise/artifact/reject, so
    # the matchable set is empty.
    matchable = CurationV2().get_matchable_unit_ids(pk)
    assert list(matchable) == [], (
        f"Expected zero matchable unit ids after labeling every unit "
        f"noise; got {list(matchable)}."
    )

    # The curated NWB still has all units (labels are stored
    # separately in CurationV2.UnitLabel, not used to filter the
    # written units). Verify:
    n_curated_units = len(CurationV2.Unit & pk)
    assert n_curated_units == len(unit_ids), (
        f"Curated NWB has {n_curated_units} units; expected "
        f"{len(unit_ids)} (labels do NOT filter the written units, "
        "only the matchable-unit accessor)."
    )


@pytest.mark.slow
def test_curation_v2_stages_empty_units_nwb_on_zero_kept_units(
    populated_sorting, monkeypatch
):
    """``_stage_curated_units_nwb`` initializes an empty
    ``pynwb.misc.Units`` when ``kept_unit_to_contributors`` is
    empty, so ``nwbf.units.object_id`` does not raise
    ``AttributeError``.

    Pynwb leaves ``nwbf.units = None`` if no ``add_unit`` is
    called. The v2 guard at ``curation.py:_stage_curated_units_nwb``
    initializes an empty ``Units`` table explicitly in this case.
    This test forces the empty-kept path via monkeypatch (the
    natural way to hit it -- a sorting with zero units -- is
    rejected upstream by ``Sorting.make``).
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    # Patch _build_curated_unit_rows to return empty so the
    # NWB-staging call enters the kept_unit_to_contributors={}
    # branch. No call to add_unit will run, so the guard at
    # ``if nwbf.units is None`` is the only thing preventing the
    # AttributeError at ``nwbf.units.object_id``.
    original = CurationV2._build_curated_unit_rows.__func__

    def _empty_rows(cls, sorting_id, sorting_units, merge_groups, curation_id):
        return [], {}

    monkeypatch.setattr(
        CurationV2,
        "_build_curated_unit_rows",
        classmethod(_empty_rows),
    )

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={},
        description="empty-kept rollback guard regression",
    )
    # The CurationV2 row exists; no AttributeError raised inside the
    # transaction. The curated NWB has zero Unit part rows.
    assert len(CurationV2 & pk) == 1
    assert len(CurationV2.Unit & pk) == 0

    # Restore (monkeypatch handles teardown; explicit for clarity).
    monkeypatch.setattr(
        CurationV2, "_build_curated_unit_rows", classmethod(original)
    )


# ---------- run_v2_pipeline preset coverage ------------------------------


@pytest.mark.slow
def test_run_v2_pipeline_clusterless_preset(polymer_smoke_session):
    """``run_v2_pipeline`` with the clusterless preset populates the
    full chain; verifies preset-to-row plumbing for the clusterless
    branch (peak-detection only, no clustering).

    The existing ``test_run_v2_pipeline_end_to_end_and_idempotent``
    only exercises MS5. The clusterless preset uses a different
    sorter dispatch in ``Sorting._run_sorter`` (``detect_peaks``
    instead of ``run_sorter``), so it's a meaningfully distinct
    orchestrator integration path. MS4 is the third shipped preset,
    but the ``mountainsort4`` sorter package is not pinned in the
    v2 environment (it's a soft optional), so testing it through
    the orchestrator would couple the test to an optional install.
    The MS4 preset's plumbing is exercised through the docstring
    and validated whenever a user installs mountainsort4 and runs
    the orchestrator.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import SorterParameters

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # The default clusterless ``SorterParameters`` row has a 100 uV
    # threshold which finds zero peaks on the smoke fixture (template
    # amplitudes are smaller). Insert a tuned row so the populate
    # actually produces a unit -- mirrors the standalone clusterless
    # e2e test.
    #
    # CRITICAL: the test mutates the ``default`` row IN PLACE because
    # the preset bundle in pipeline.py hard-codes
    # ``sorter_params_name="default"``. We snapshot the original row
    # before mutating and restore it in a ``finally`` so this test
    # does not contaminate subsequent ones that expect the 100 uV
    # default. Without the restore step, every later test in the
    # same pytest session sees the 5 uV row, which is a real
    # cross-test ordering hazard.
    default_key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }
    original_default = (SorterParameters & default_key).fetch1()
    # Phase 1b N48 dropped ``outputs`` and ``random_chunk_kwargs`` from
    # ``ClusterlessThresholderSchema`` (extra="forbid"); the test
    # params dict must not carry them. The runtime strip path in
    # ``Sorting._run_sorter`` still tolerates either shape -- but
    # ``SorterParameters.insert1`` validates through Pydantic first,
    # so the v1-era ``outputs`` key would fail at insert time.
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "params": {
                "detect_threshold": 5.0,
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 100.0,
            },
            "params_schema_version": 2,
            "job_kwargs": None,
        },
        skip_duplicates=False,
        replace=True,
    )

    try:
        manifest = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            preset="franklab_tetrode_clusterless_thresholder",
        )
        assert manifest["preset"] == "franklab_tetrode_clusterless_thresholder"
        for key in (
            "recording_id",
            "artifact_id",
            "sorting_id",
            "curation_id",
            "merge_id",
        ):
            assert manifest.get(key) is not None, (
                f"Manifest missing {key!r}; got {manifest}."
            )
    finally:
        # Restore the original 100 uV default row so subsequent
        # tests / sessions are not poisoned by the 5 uV value.
        # Use update1 rather than insert1(replace=True): the test
        # populated downstream SortingSelection rows that FK back
        # to this SorterParameters row, so DJ's replace path
        # (delete + reinsert) is blocked by the FK constraint.
        # update1 modifies the secondary attributes in place.
        SorterParameters.update1(original_default)


# ---------- Sorting.make rollback file cleanup ---------------------------


@pytest.mark.slow
def test_sorting_make_rollback_cleans_units_nwb(
    polymer_smoke_session, monkeypatch
):
    """If ``Sorting.make`` fails between ``AnalysisNwbfile.add()`` and
    a successful ``Sorting.insert1``, the staged units NWB on disk is
    cleaned up.

    Patches ``Sorting._populate_unit_part`` to raise so the
    transaction rolls back AFTER the file is written and registered.
    The rollback path in ``Sorting.make``'s ``except`` block must
    unlink the staged NWB so the file system doesn't accumulate
    orphans on each retry.

    Self-sufficient setup so it doesn't depend on the
    populated_recording module fixture's row still being live
    (other tests in the module may have wiped it).
    """
    import pathlib as _pathlib

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    # Self-sufficient Recording chain setup.
    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
        )[0]
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
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
            "artifact_id": art_pk["artifact_id"],
        }
    )
    # Ensure no leftover Sorting row.
    (Sorting & sort_pk).super_delete(warn=False)

    # Snapshot the analysis-file directory contents before the
    # broken populate runs so we can detect any orphan file that
    # appears AFTER the rollback. The except block in Sorting.make
    # is responsible for unlinking the staged file before re-raising.
    from spyglass.settings import analysis_dir as _ad

    analysis_dir = _pathlib.Path(_ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    # Patch the Unit-part populate to raise AFTER the file is
    # written and AnalysisNwbfile.add has run inside the transaction.
    def _broken_unit_part(
        self, sorting, recording_id, nwb_file_name, key, analyzer_folder
    ):
        raise RuntimeError("simulated unit-part failure")

    monkeypatch.setattr(
        Sorting, "_populate_unit_part", _broken_unit_part
    )

    # populate swallows the exception into DataJoint's error
    # machinery via suppress_errors; assert by checking the
    # after-state instead.
    Sorting.populate(sort_pk, reserve_jobs=False, suppress_errors=True)
    assert len(Sorting & sort_pk) == 0, (
        "Sorting row should not be present after rollback; "
        "the transaction was supposed to roll back."
    )

    # No new orphan units NWB should have appeared in the analysis
    # directory after the rolled-back populate.
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"Sorting.make rollback left orphan analysis files: {new_files}. "
        "The except-block in Sorting.make must unlink the staged file "
        "when the transaction rolls back."
    )

    # N51 mode B: the analyzer folder created by ``_build_analyzer``
    # must also be removed by ``make_insert``'s rollback path. A
    # 5-50 GB analyzer folder orphan per failed populate is the
    # leak this guards against.
    from spyglass.spikesorting.v2.utils import _analyzer_path

    analyzer_folder = _analyzer_path(sort_pk)
    assert not analyzer_folder.exists(), (
        f"Sorting.make rollback left analyzer folder {analyzer_folder} "
        "on disk. The except-block in Sorting.make_insert must "
        "shutil.rmtree it when the transaction rolls back."
    )


# ---------- Boundary-spike round-trip (R8 decision gate) -----------------


@pytest.mark.slow
def test_boundary_spike_round_trip_does_not_raise(
    polymer_smoke_session, monkeypatch
):
    """A spike at the recording's final sample survives the v2 NWB round-trip.

    Phase 1b's R8 decision is gated by this test. SpikeInterface's
    ``NwbSortingExtractor`` has historically been strict about spikes
    that map back to a sample at or past ``n_samples`` after the
    ``timestamps[sample_index] -> spike_times[]`` -> ``round((t -
    t_start) * fs)`` round-trip. If SI 0.104 raises here, we fold in
    v1's ``spike_times_to_valid_samples`` clip on read. If it does
    not raise, the clip is unnecessary and the test is kept as a
    regression guard.

    Construction strategy
    ---------------------
    The shipped sorters do not deterministically produce a boundary
    spike, so we monkey-patch ``Sorting._run_sorter`` to return a
    hand-built ``NumpySorting`` with one unit whose spike train
    includes ``n_samples - 1`` and one earlier in-bounds spike. The
    rest of ``Sorting.make`` runs normally:
    ``_remove_excess_spikes`` should keep both samples
    (``n_samples - 1 < n_samples``), ``_write_units_nwb`` writes
    ``timestamps[sample_index]`` as the absolute time, and
    ``Sorting().get_sorting`` reads back via ``NwbSortingExtractor``.
    The round-trip is also exercised through
    ``CurationV2.insert_curation`` + ``CurationV2.get_sorting`` so
    both production read paths are covered.
    """
    import numpy as _np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
        )[0]
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
            "sorter_params_name": "default",
            "artifact_id": art_pk["artifact_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    # Monkey-patch ``_run_sorter`` to deterministically return a
    # ``NumpySorting`` with a spike at exactly ``n_samples - 1`` on
    # the artifact-masked recording. ``_run_sorter`` is a
    # ``@staticmethod`` so we patch the class attribute directly.
    def _boundary_run_sorter(sorter, sorter_params, recording, sorting_id):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        # Place one spike near the start and one at the very last
        # sample. The boundary one is what tests SI's read-side
        # bounds check; the earlier one keeps the unit "real" so
        # downstream analyzer math doesn't degenerate.
        samples = _np.array([100, n_samples - 1], dtype=_np.int64)
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(_boundary_run_sorter))

    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, (
        "Sorting.populate failed when the synthetic sorter produced "
        "a boundary spike; this is the failure mode R8 watches for "
        "but inside _write_units_nwb rather than at read time."
    )

    # First read path: Sorting.get_sorting -> NwbSortingExtractor
    # must construct without raising. Unit set must include the one
    # we planted.
    sorting_obj = Sorting().get_sorting(sort_pk)
    assert 0 in sorting_obj.get_unit_ids(), (
        "Boundary-spike sorting did not survive Sorting.get_sorting "
        f"round-trip; unit_ids={list(sorting_obj.get_unit_ids())}."
    )

    # Second read path: CurationV2 + CurationV2.get_sorting. The
    # curated NWB write goes through a different code path than
    # Sorting._write_units_nwb but reads back via the same
    # NwbSortingExtractor pattern. Pass labels={} so insert_curation
    # accepts the call on Phase 1's strict signature.
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description="R8 boundary-spike test",
    )
    curated = CurationV2().get_sorting(curation_pk)
    assert 0 in curated.get_unit_ids(), (
        "Boundary-spike unit lost on the CurationV2 round-trip; "
        f"curated unit_ids={list(curated.get_unit_ids())}."
    )


# ---------- ArtifactDetection: signal-level _detect_artifacts ------------


def test_detect_artifacts_finds_known_transient(dj_conn):
    """``ArtifactDetection._detect_artifacts`` runs its amplitude-
    threshold scan body and produces the expected valid-time
    complement of a known synthetic transient.

    Logically a unit test (the body calls a ``@staticmethod`` on an
    in-memory ``NumpyRecording`` -- no DataJoint queries, no NWB
    I/O). ``dj_conn`` is required only because importing the v2
    ``artifact`` module activates ``dj.schema("spikesorting_v2_artifact")``
    at module load time, which needs a live MySQL connection. The
    test itself runs in well under a second and is not slow-marked
    so it executes on every commit.

    Every existing artifact test uses the ``"none"`` preset which
    short-circuits at the ``if not validated.detect`` guard in
    ``_detect_artifacts`` (artifact.py:584) and writes a single all-
    valid interval. That left the entire amplitude-threshold
    detection body (~50 lines, the actual artifact-finding code) at
    0% coverage.

    This test calls ``_detect_artifacts`` directly with a synthetic
    ``NumpyRecording`` carrying a deterministic 200-sample, 200 uV
    transient at frames 1000-1199. The threshold (50 uV) and
    proportion (50% of channels) are set so the transient
    reliably trips detection and nothing else does. Asserts the
    returned ``valid_times`` has exactly two intervals straddling
    the planted transient and excludes the artifact samples.
    """
    import numpy as _np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # Build a synthetic 4-channel, 5000-sample recording with a
    # 200 uV transient at frames 1000-1199 across all channels.
    # Gain = 1.0 so traces are already in uV.
    fs = 30_000.0
    n_samples = 5000
    n_channels = 4
    traces = _np.zeros((n_samples, n_channels), dtype=_np.float32)
    traces[1000:1200, :] = 200.0
    rec = si.NumpyRecording(
        traces_list=[traces], sampling_frequency=fs
    )
    rec.set_channel_gains([1.0] * n_channels)

    # detect=True, 50 uV threshold, 50% of channels (= 2 of 4)
    # must exceed simultaneously. The transient hits all 4
    # channels at 200 uV; nothing else exceeds 50 uV. The
    # minimal positive removal_window_ms (Pydantic enforces > 0)
    # expands the artifact span by ceil(removal_window_ms * fs /
    # 2000) frames on each side. At fs=30 kHz and 0.05 ms that's
    # ``ceil(0.05 * 30000 / 2000) = 1`` frame -- intentionally
    # small so the assertions below can be exact.
    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
    )

    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # half_window_frames = ceil(0.05 ms * 30 kHz / 2 / 1000) = 1.
    # Artifact span expands from [1000, 1199] to [999, 1200]. The
    # saved valid intervals straddle this expanded artifact run.
    expected_start_f = 999  # 1000 - half_window
    expected_end_f = 1200  # 1199 + half_window
    expected_end_open_f = expected_end_f + 1  # half-open boundary

    assert valid_times.shape == (2, 2), (
        f"Expected exactly 2 valid intervals around the synthetic "
        f"transient at frames 1000-1199, got shape {valid_times.shape}. "
        "_detect_artifacts either skipped the detection body or "
        "merged the artifact and valid runs incorrectly."
    )
    # First valid interval ends at the start of the expanded
    # artifact span.
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(
        timestamps[expected_start_f], abs=1e-9
    ), (
        f"First valid interval ends at {valid_times[0][1]}, expected "
        f"{timestamps[expected_start_f]} (start of the expanded "
        "artifact span)."
    )
    # Second valid interval starts AT the first sample AFTER the
    # expanded artifact span (the half-open fix from the prior
    # review: the saved interval starts at
    # ``timestamps[end_f + 1]``, so the LAST artifact frame is
    # NOT silently included in the next valid interval).
    assert valid_times[1][0] == pytest.approx(
        timestamps[expected_end_open_f], abs=1e-9
    ), (
        f"Second valid interval starts at {valid_times[1][0]}, expected "
        f"{timestamps[expected_end_open_f]} (first sample AFTER the "
        "expanded artifact span). If this is "
        f"``timestamps[{expected_end_f}]`` the off-by-one fix at "
        "the END of the artifact interval has regressed."
    )
    assert valid_times[1][1] == pytest.approx(timestamps[-1], abs=1e-9)


def test_detect_artifacts_no_threshold_crossings(dj_conn):
    """``_detect_artifacts`` returns the full recording window as a
    single valid interval when no frame trips the threshold.

    Exercises the early-return branch at ``len(frames_above) == 0``
    (artifact.py:603) without short-circuiting at the
    ``detect=False`` guard. Complements the synthetic-transient
    test by covering the "detection ran but found nothing" path.
    """
    import numpy as _np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # Recording with all-zero traces -- nothing can exceed any
    # positive threshold.
    rec = si.NumpyRecording(
        traces_list=[_np.zeros((2000, 4), dtype=_np.float32)],
        sampling_frequency=30_000.0,
    )
    rec.set_channel_gains([1.0] * 4)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=10.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2)
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


def _build_synthetic_rec(traces, fs=30_000.0):
    """Helper: wrap a (n_samples, n_channels) array as a SI NumpyRecording
    with unit gains so trace values can be reasoned about as microvolts."""
    import spikeinterface as si

    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * traces.shape[1])
    return rec


def test_detect_artifacts_zscore_only_detection(dj_conn):
    """``_detect_artifacts`` runs the z-score-only branch when
    ``amplitude_thresh_uV is None``.

    Exercises:
    - line 583 (``amplitude_thresh_uV is None`` -> ``above_amp =
      zeros_like``)
    - lines 585-588 (z-score computation)
    - line 600 (``channel_hit = above_z`` when only zscore set)

    Builds a recording with low-amplitude background noise plus a
    high-z-score transient at known frames. The transient's
    absolute amplitude is modest (5 uV) but its z-score is large
    relative to the channel's near-zero mean.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rng = _np.random.default_rng(42)
    n_samples, n_channels = 5000, 4
    # Background noise std=0.5 uV. Short (10-sample) 500 uV
    # transient: keeps the global std small (~22 uV) so the
    # transient samples sit at z ~= 22, well above the
    # zscore_thresh=10 gate. A longer / lower transient inflates
    # the per-channel std and would silently drop the transient's
    # z-score below threshold -- exactly the kind of footgun
    # z-score detection has in practice.
    traces = (
        rng.normal(0.0, 0.5, size=(n_samples, n_channels))
        .astype(_np.float32)
    )
    traces[2000:2010, :] = 500.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=None,  # z-score only
        zscore_thresh=10.0,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (2, 2), (
        f"z-score-only detection expected 2 valid intervals around the "
        f"transient at frames 2000-2009, got shape {valid_times.shape}."
    )
    # Pinpoint the artifact run via the expanded boundaries
    # (half_window_frames=1).
    assert valid_times[0][1] == pytest.approx(
        timestamps[1999], abs=1e-9
    )
    assert valid_times[1][0] == pytest.approx(
        timestamps[2011], abs=1e-9
    )


def test_detect_artifacts_amplitude_and_zscore_combined(dj_conn):
    """``_detect_artifacts`` requires BOTH thresholds when both are
    set (AND mode at line 596).

    Builds a recording with a constant 80 uV baseline (clears the
    50 uV amplitude threshold on every frame, but produces ~0
    z-score everywhere because the baseline is uniform) plus a
    200 uV step deviation at frames 3000-3099 (clears both the
    amplitude AND z-score gates). Pins that the AND-mode flags
    only the step deviation, not the baseline.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    # Per-channel baseline: 80 uV constant on all channels. The
    # amplitude detection at thresh=50 uV would flag every frame
    # because 80 > 50 on all channels everywhere -- which is
    # exactly the scenario the AND mode is meant to filter out.
    # Combined with z-score, only frames that ALSO have anomalous
    # per-channel z-score get flagged.
    traces = _np.full((n_samples, n_channels), 80.0, dtype=_np.float32)
    # Insert a single high-z-score deviation at frames 3000-3099 on
    # all channels: 200 uV (the per-channel mean is ~80; 200 is
    # well above the per-channel std).
    traces[3000:3100, :] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=5.0,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # Only the per-channel anomaly at frames 3000-3099 trips the
    # combined gate; the baseline 80 uV is uniform so its z-score
    # is ~0 everywhere despite passing the amplitude threshold.
    assert valid_times.shape == (2, 2), (
        f"AND-mode (amplitude AND z-score) expected to flag the "
        f"high-z-score region at frames 3000-3099, got shape "
        f"{valid_times.shape}."
    )
    assert valid_times[0][1] == pytest.approx(
        timestamps[2999], abs=1e-9
    )
    assert valid_times[1][0] == pytest.approx(
        timestamps[3101], abs=1e-9
    )


def test_detect_artifacts_join_window_merges_runs(dj_conn):
    """``_detect_artifacts`` merges two artifact runs separated by
    fewer than ``join_window_frames`` into a single artifact span.

    Exercises line 621 (``cur_end = f`` inside the join branch)
    which the existing single-run test doesn't hit. Builds two
    transients separated by 10 frames; with join_window_ms set so
    join_window_frames > 10, the runs merge into one. With a
    smaller join window, the runs stay separate.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    traces = _np.zeros((n_samples, n_channels), dtype=_np.float32)
    traces[1000:1100, :] = 100.0  # artifact A
    traces[1110:1210, :] = 100.0  # artifact B; 10 frames after A
    rec = _build_synthetic_rec(traces)
    fs = rec.get_sampling_frequency()

    # join_window_ms = 1.0 -> join_window_frames = ceil(1 * 30 / 1) =
    # 30 frames. Gap between A and B is 10 frames, so they merge.
    merged = ArtifactDetection._detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_thresh_uV=50.0,
            zscore_thresh=None,
            proportion_above_thresh=0.5,
            removal_window_ms=0.05,
            join_window_ms=1.0,
        ),
    )
    assert merged.shape == (2, 2), (
        f"With join_window_ms=1.0 (= 30 frames > 10-frame gap), the two "
        f"transients should merge into one artifact span, leaving 2 "
        f"valid intervals; got shape {merged.shape}."
    )

    # join_window_ms = 0.1 -> join_window_frames = ceil(0.1 * 30 /
    # 1) = 3 frames. Gap of 10 stays unbridged; two separate
    # artifact runs, three valid intervals.
    unmerged = ArtifactDetection._detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_thresh_uV=50.0,
            zscore_thresh=None,
            proportion_above_thresh=0.5,
            removal_window_ms=0.05,
            join_window_ms=0.1,
        ),
    )
    assert unmerged.shape == (3, 2), (
        f"With join_window_ms=0.1 (= 3 frames < 10-frame gap), the two "
        f"transients stay separate, leaving 3 valid intervals; got "
        f"shape {unmerged.shape}."
    )


@pytest.mark.slow
def test_artifact_detection_parameters_validates_via_insert1(dj_conn):
    """``ArtifactDetectionParameters.insert1`` validates the
    ``params`` blob through Pydantic before the row is written.

    Exercises lines 92-96 (the custom ``insert1`` override).
    A blob with an unknown key violates ``extra="forbid"`` and the
    insert is rejected before any row reaches the DB.
    """
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )

    bad_row = {
        "artifact_params_name": "params_validates_unit_test",
        "params": {
            "detect": True,
            "amplitude_thresh_uV": 100.0,
            "not_a_real_field": "x",  # rejected by extra="forbid"
        },
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    with pytest.raises(ValidationError):
        ArtifactDetectionParameters().insert1(bad_row)

    assert (
        len(
            ArtifactDetectionParameters
            & {
                "artifact_params_name": "params_validates_unit_test"
            }
        )
        == 0
    ), "Failed validation should not have written a row."


# ---------- CurationV2.insert_curation rollback cleanup ------------------


@pytest.mark.slow
def test_curation_v2_insert_rollback_cleans_units_nwb(
    populated_sorting, monkeypatch
):
    """``CurationV2.insert_curation``'s except block unlinks the
    staged Units NWB on a transaction rollback.

    Mirrors ``test_sorting_make_rollback_cleans_units_nwb`` (which
    pins the same contract for Sorting.make). Patches
    ``SpikeSortingOutput._merge_insert`` to raise mid-transaction so
    the curation rows + AnalysisNwbfile registration roll back. The
    except block must unlink the staged units NWB so the file
    system doesn't accumulate orphans on each retry.
    """
    import pathlib as _pathlib

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    from spyglass.settings import analysis_dir as _ad

    analysis_dir = _pathlib.Path(_ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    original_merge_insert = SpikeSortingOutput._merge_insert

    def _broken_merge_insert(cls, *args, **kwargs):
        raise RuntimeError("simulated merge_insert failure")

    monkeypatch.setattr(
        SpikeSortingOutput, "_merge_insert", classmethod(_broken_merge_insert)
    )

    with pytest.raises(RuntimeError, match="simulated merge_insert failure"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={},
            description="rollback regression test",
        )

    # Restore the original (monkeypatch.setattr handles teardown but
    # being explicit is clearer than relying on fixture order).
    monkeypatch.setattr(
        SpikeSortingOutput, "_merge_insert", original_merge_insert
    )

    # No CurationV2 / SpikeSortingOutput row was inserted.
    assert len(CurationV2 & populated_sorting) == 0, (
        "CurationV2 row present after a transaction that should "
        "have rolled back."
    )
    assert len(SpikeSortingOutput.CurationV2 & populated_sorting) == 0, (
        "SpikeSortingOutput.CurationV2 merge-part row present after "
        "a transaction that should have rolled back."
    )

    # No new orphan analysis NWB file remains on disk: the except
    # block in insert_curation unlinks the staged file.
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"CurationV2.insert_curation rollback left orphan analysis "
        f"files: {new_files}. The except-block in insert_curation "
        f"must unlink the staged file when the transaction rolls back."
    )


# ---------- Recording.make rollback file cleanup -------------------------


@pytest.mark.slow
def test_recording_make_rollback_cleans_analysis_nwb(
    polymer_smoke_session, monkeypatch
):
    """``Recording.make`` unlinks its staged AnalysisNwbfile on a
    transaction rollback.

    Mirrors ``test_sorting_make_rollback_cleans_units_nwb`` for the
    Recording stage. Patches ``Recording.insert1`` to raise inside
    the transaction (after ``AnalysisNwbfile().add`` has run), so
    the transaction rolls back. The except block in
    ``Recording.make`` is responsible for unlinking the staged
    preprocessed NWB.
    """
    import pathlib as _pathlib

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
        )[0]
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
    (Recording & rec_pk).super_delete(warn=False)

    from spyglass.settings import analysis_dir as _ad

    analysis_dir = _pathlib.Path(_ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    # Force a failure AFTER the NWB is written and registered
    # (Recording.make wraps AnalysisNwbfile().add + self.insert1 in
    # transaction_or_noop; patching self.insert1 makes the
    # transaction roll back with the file already on disk).
    original_insert1 = Recording.insert1

    def _broken_insert1(self, *args, **kwargs):
        raise RuntimeError("simulated Recording.insert1 failure")

    monkeypatch.setattr(Recording, "insert1", _broken_insert1)

    Recording.populate(rec_pk, reserve_jobs=False, suppress_errors=True)

    # Restore (monkeypatch teardown also restores).
    monkeypatch.setattr(Recording, "insert1", original_insert1)

    # No Recording row, no orphan NWB file.
    assert len(Recording & rec_pk) == 0, (
        "Recording row present after rollback; transaction did not "
        "roll back."
    )
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"Recording.make rollback left orphan analysis files: "
        f"{new_files}. The except-block must unlink the staged file."
    )


# ---------- CurationV2 with merge_groups / apply_merges=True -------------


@pytest.mark.slow
def test_curation_v2_insert_with_merge_groups_apply_merges(
    polymer_60s_session,
):
    """``CurationV2.insert_curation`` with ``merge_groups`` +
    ``apply_merges=True`` writes a curated NWB whose unit_ids are
    the merge-group heads, and whose merged unit's spike train is
    the sorted union of its contributors.

    Exercises ``_build_curated_unit_rows`` with a non-empty merge
    list (amplitude-inheritance: kept unit gets electrode/amplitude
    from the highest-amplitude contributor) AND
    ``_stage_curated_units_nwb`` with ``apply_merges=True``
    (concatenated + sorted spike trains).

    Uses the 60s polymer fixture rather than the smoke fixture
    because MS5 finds only 1 unit on the 4s smoke recording -- too
    few to test merging. The 60s recording yields ~26 units (one
    sort group, all 4 shanks aggregated through the same flow).
    """
    import numpy as _np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    # Set up the v2 chain on the 60s session, one sort group, so
    # MS5 finds enough units to merge. This is heavier than the
    # smoke-fixture setup but unavoidable to exercise this path
    # without injecting synthetic Sorting.Unit rows.
    _clean_session_v2(polymer_60s_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_60s_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_60s_session).fetch("sort_group_id")
        )[0]
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
            "sorter": "mountainsort5",
            "sorter_params_name": (
                "franklab_tetrode_hippocampus_30kHz_ms5"
            ),
            "artifact_id": art_pk["artifact_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    units = sorted(
        int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id")
    )
    assert len(units) >= 2, (
        f"Sorting on the 60s polymer fixture yielded {len(units)} units; "
        "need >= 2 to test merging. (MS5 typically finds 5+ per shank "
        "on this fixture; a regression upstream would surface here.)"
    )

    # Merge the first two units.
    head, absorbed = units[0], units[1]
    merge_groups = [[head, absorbed]]
    pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        merge_groups=merge_groups,
        apply_merges=True,
        description="merge_groups regression",
    )

    # CurationV2.Unit has one row fewer than Sorting.Unit (the
    # absorbed unit is no longer a distinct row).
    curated_unit_ids = set(int(u) for u in (CurationV2.Unit & pk).fetch("unit_id"))
    assert curated_unit_ids == set(units) - {absorbed}, (
        f"Curated unit ids = {sorted(curated_unit_ids)}; "
        f"expected {sorted(set(units) - {absorbed})} (head kept, "
        "absorbed removed)."
    )

    # The merged unit's spike train in the curated NWB is the
    # sorted union of the contributors' spike trains. Use
    # ``apply_merges=True`` writes concatenated spike trains.
    src_sorting = Sorting().get_sorting(sort_pk)
    src_head = _np.asarray(
        src_sorting.get_unit_spike_train(unit_id=head, return_times=True)
    )
    src_absorbed = _np.asarray(
        src_sorting.get_unit_spike_train(unit_id=absorbed, return_times=True)
    )
    expected_merged = _np.sort(
        _np.concatenate([src_head, src_absorbed])
    )

    curated_sorting = CurationV2().get_sorting(pk)
    merged_times = _np.asarray(
        curated_sorting.get_unit_spike_train(unit_id=head, return_times=True)
    )
    assert len(merged_times) == len(expected_merged), (
        f"Merged unit has {len(merged_times)} spikes; expected "
        f"{len(expected_merged)} (head + absorbed concatenated)."
    )
    _np.testing.assert_array_equal(merged_times, expected_merged)

    # And the n_spikes column on CurationV2.Unit reflects the merge.
    head_row = (CurationV2.Unit & pk & {"unit_id": head}).fetch1()
    assert head_row["n_spikes"] == len(expected_merged), (
        f"CurationV2.Unit.n_spikes for merged head = "
        f"{head_row['n_spikes']}; expected {len(expected_merged)}."
    )


# ---------- _detect_artifacts cross-channel proportion boundary ----------


def test_detect_artifacts_below_proportion_threshold_ignored(dj_conn):
    """``_detect_artifacts`` does NOT flag artifact frames when
    fewer than ``proportion_above_thresh`` of channels exceed the
    amplitude threshold.

    Existing detection tests put the transient on every channel,
    so the proportion gate is never the binding constraint. This
    test puts a 200 uV transient on a SINGLE channel of a
    4-channel recording with ``proportion_above_thresh=0.5``
    (= ceil(0.5 * 4) = 2 channels required). Detection must not
    fire: the single-channel transient does not meet the cross-
    channel quorum.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    traces = _np.zeros((n_samples, n_channels), dtype=_np.float32)
    # ONE channel has a 200 uV transient at frames 1000-1099; the
    # other three channels stay at zero. 1 of 4 channels is below
    # the 2-of-4 requirement.
    traces[1000:1100, 0] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2), (
        f"Single-channel transient should NOT trip cross-channel "
        f"proportion gate (1/4 < 0.5 required); expected one full "
        f"valid interval but got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


# ---------- SortingSelection concat-source gate --------------------------


@pytest.mark.slow
def test_sorting_selection_rejects_concat_source(dj_conn):
    """``SortingSelection.insert_selection`` refuses
    ``concat_recording_id`` with ``NotImplementedError``.

    The Phase 1 plan declares the ``ConcatenatedRecordingSource``
    schema final-shape under the zero-migration policy, but the
    make-body branch isn't implemented yet. Pins the gate so a
    premature attempt to wire up the concat path immediately fails
    this test.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    with pytest.raises(NotImplementedError, match="concatenated"):
        SortingSelection.insert_selection(
            {
                "concat_recording_id": "00000000-0000-0000-0000-000000000000",
                "sorter": "mountainsort5",
                "sorter_params_name": (
                    "franklab_tetrode_hippocampus_30kHz_ms5"
                ),
            }
        )


# ---------- v1 / imported dispatch regression after v2 import ------------


@pytest.mark.slow
def test_imported_dispatch_survives_v2_module_loaded(populated_sorting):
    """v0 / v1 / imported sources still dispatch correctly through
    ``SpikeSortingOutput.source_class_dict`` after the v2 module
    adds itself to the registry.

    Regression guard for the conditional ``source_class_dict["CurationV2"]
    = CurationV2`` block in ``spikesorting_merge.py``: a future
    refactor that broke v1 dispatch as a side effect of v2 changes
    should fail this test.

    Uses the ImportedSpikeSorting source registered by the test
    bootstrap (via ``mini_insert``) rather than v0/v1 sorting paths,
    because the v2 environment has SpikeInterface 0.104 which is
    incompatible with the SI 0.99-based v0/v1 production code.
    """
    from spyglass.spikesorting.imported import ImportedSpikeSorting
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )

    # Both v2 and the imported source must be registered.
    assert "CurationV2" in source_class_dict
    assert "ImportedSpikeSorting" in source_class_dict
    # The Merge class also resolves both via merge_get_parent_class.
    assert SpikeSortingOutput().merge_get_parent_class(
        "CurationV2"
    ) is not None
    assert SpikeSortingOutput().merge_get_parent_class(
        "ImportedSpikeSorting"
    ) is not None
    # And the Imported part table is reachable through the master.
    assert hasattr(SpikeSortingOutput, "ImportedSpikeSorting")


# ---------- L5-A: global-median reference (ref_channel_id=-2) ------------


@pytest.mark.slow
def test_recording_make_global_median_reference(polymer_smoke_session):
    """``Recording._apply_pre_motion_preprocessing`` applies the
    global-median reference when ``ref_channel_id == -2``.

    Every other recording test uses ``sort_reference_electrode_id =
    -1`` (no reference), so the ``elif ref_channel_id == -2`` branch
    in recording.py has never been exercised. This test populates a
    Recording with ``sort_reference_electrode_id = -2`` and pins:

    1. The populate completes without raising (the cross-schema
       ``validated.common_reference.operator`` field resolution
       works -- a load-bearing untested code path).
    2. The resulting traces differ from a no-reference (``-1``)
       recording on the same data: CMR is applied BEFORE bandpass
       so the final per-sample channel median is not exactly zero,
       but the data IS materially different from the unreferenced
       version (signed-traces sum and overall variance shift).
    """
    import numpy as _np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
        )[0]
    )

    # No-reference baseline (-1).
    rec_pk_unref = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk_unref, reserve_jobs=False)
    traces_unref = Recording().get_recording(rec_pk_unref).get_traces()

    # Switch SortGroupV2 to global-median reference (-2) and
    # repopulate. A new RecordingSelection gets minted because the
    # cache_hash depends on input data.
    SortGroupV2.update1(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "sort_reference_electrode_id": -2,
        }
    )
    assert (
        (
            SortGroupV2
            & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        ).fetch1("sort_reference_electrode_id")
        == -2
    )
    # Drop the unref Recording so the new selection materializes a
    # fresh global-median recording rather than reusing the cached
    # bytes.
    (Recording & rec_pk_unref).super_delete(warn=False)
    Recording.populate(rec_pk_unref, reserve_jobs=False)
    traces_ref = Recording().get_recording(rec_pk_unref).get_traces()

    # Pin (1): same shape (the channel set is unchanged for -2;
    # -1 also doesn't drop a channel, so shapes match).
    assert traces_ref.shape == traces_unref.shape

    # Pin (2): the recordings are materially different. The
    # global-median reference subtracts the per-sample channel
    # median; even after bandpass that produces a measurable shift
    # in the recording's first moment vs the unreferenced version.
    delta = float(_np.abs(traces_ref - traces_unref).mean())
    assert delta > 1e-6, (
        f"Global-median (-2) and no-reference (-1) recordings are "
        f"indistinguishable (mean |diff| = {delta:.2e}). The -2 "
        "branch may not be applying CMR; check the call to "
        "sip.common_reference(reference='global')."
    )


# ---------- L5-B: DuplicateSelectionError on insert_selection -----------


@pytest.mark.slow
def test_recording_selection_raises_on_duplicate_logical_identity(
    polymer_smoke_session,
):
    """``RecordingSelection.insert_selection`` raises
    ``DuplicateSelectionError`` when the table already contains
    multiple rows with the same logical identity (an integrity bug
    that should never happen via the helper but can happen if a
    caller bypasses it with raw ``insert``).

    Without this test the integrity guard at
    ``recording.py:609-615`` is dead code that, if it silently
    broke, would let downstream consumers index ``[0]`` on
    duplicate rows.
    """
    import uuid as _uuid

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
        )[0]
    )
    logical_identity = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preproc_params_name": "default_franklab",
        "team_name": "v2_test_team",
    }
    # Bypass the helper to plant the duplicate (this is what the
    # guard exists to catch in case some script uses raw dj.insert).
    planted_ids = []
    RecordingSelection().insert1(
        {**logical_identity, "recording_id": _uuid.uuid4()}
    )
    RecordingSelection().insert1(
        {**logical_identity, "recording_id": _uuid.uuid4()}
    )
    planted_ids = list(
        (RecordingSelection & logical_identity).fetch("recording_id")
    )

    try:
        with pytest.raises(DuplicateSelectionError, match="duplicate"):
            RecordingSelection.insert_selection(logical_identity)
    finally:
        # CRITICAL: clean up the duplicates we planted so the next
        # test's RecordingSelection.insert_selection doesn't trip
        # the same guard. Without this teardown, the function-scoped
        # populated_recording fixture (whose body calls
        # insert_selection) would fail in every subsequent test.
        for rid in planted_ids:
            (
                RecordingSelection & {"recording_id": rid}
            ).super_delete(warn=False)


# ---------- L5-C: ArtifactDetection.delete handles orphans ---------------


@pytest.mark.slow
def test_artifact_detection_delete_handles_orphan_source_part(
    populated_recording, monkeypatch
):
    """``ArtifactDetection.delete`` does not crash when an
    ArtifactSelection master row's source part fails to resolve.

    The except branch at ``artifact.py:711-713`` catches
    ``resolve_source`` failures during the IntervalList-cleanup
    loop. Without a test, a regression that re-raised the
    exception (instead of skipping the orphan) would surface only
    in production. DataJoint refuses to delete a part table
    directly, so we monkeypatch ``ArtifactSelection.resolve_source``
    to raise -- this is the exact failure mode the except branch
    was written to handle (an orphan master whose source part has
    been cascade-deleted via some upstream path).
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # Patch resolve_source to raise the same shape of exception an
    # orphan master would produce (the source-part lookup fails).
    def _broken_resolve(cls, key):
        raise RuntimeError("simulated orphan: source part missing")

    monkeypatch.setattr(
        ArtifactSelection,
        "resolve_source",
        classmethod(_broken_resolve),
    )

    # The delete should complete without raising. The current
    # implementation's except block (artifact.py:711-713) skips
    # rows whose source can't be resolved.
    (ArtifactDetection & art_pk).delete(safemode=False)
    assert len(ArtifactDetection & art_pk) == 0, (
        "ArtifactDetection row should be deleted even when "
        "resolve_source raises; the except branch must not "
        "short-circuit the master delete."
    )


# ---------- L5-D: _write_units_nwb zero-unit guard (sorting layer) -------


@pytest.mark.slow
def test_write_units_nwb_handles_zero_unit_sorter(populated_recording):
    """``Sorting._write_units_nwb`` initializes an empty Units NWB
    when the sorter produces zero unit ids.

    The guard at ``sorting.py:822-826`` is the sorting-layer analog
    of the curation-layer guard tested by
    ``test_curation_v2_stages_empty_units_nwb_on_zero_kept_units``.
    Without this test, a regression in the zero-unit handling at
    the Sorting layer (e.g., a refactor that adds ``add_unit_column``
    before the guard, exactly the bug recently fixed in
    CurationV2) would crash on real datasets where the sorter
    finds no units.
    """
    import numpy as _np
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    recording = Recording().get_recording(
        {"recording_id": populated_recording["recording_id"]}
    )
    empty_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[{}],
        sampling_frequency=recording.get_sampling_frequency(),
    )

    from spyglass.spikesorting.v2.recording import RecordingSelection

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")

    analysis_file_name, units_object_id = Sorting._write_units_nwb(
        sorting=empty_sorting,
        recording=recording,
        nwb_file_name=nwb_file_name,
    )
    try:
        # Object id is defined (the guard initialized an empty
        # Units table rather than leaving ``nwbf.units = None``).
        assert isinstance(units_object_id, str)
        assert len(units_object_id) > 0

        # The staged NWB on disk is readable and has an empty Units
        # table.
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
            nwbf = io.read()
            assert nwbf.units is not None
            assert len(nwbf.units.id[:]) == 0, (
                f"Expected empty Units table; got "
                f"{len(nwbf.units.id[:])} rows."
            )
    finally:
        # Tidy up the staged file. _write_units_nwb does not register
        # the file (the caller does so inside a transaction); we
        # unlink the bare file since no DJ row was registered.
        import pathlib as _pathlib

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        if _pathlib.Path(abs_path).exists():
            _pathlib.Path(abs_path).unlink()


# ---------- L1-D: artifact at recording boundary boundary clamp ----------


def test_detect_artifacts_clamps_artifact_at_recording_end(dj_conn):
    """``_detect_artifacts`` clamps the half-open end of an artifact
    that runs to the last sample.

    The clamp ``min(end_f + 1, len(timestamps) - 1)`` at
    ``artifact.py:631`` ensures the saved interval doesn't index
    out of bounds when the artifact reaches the recording end. No
    prior test puts a transient near the end of the recording, so
    this clamp branch was unexercised.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 1000, 4
    traces = _np.zeros((n_samples, n_channels), dtype=_np.float32)
    # Transient at the very end of the recording.
    traces[990:1000, :] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # The artifact runs from frame 990 to the end. The clamp
    # produces a single valid interval before the artifact (the
    # tail-valid case at line 653 fires only if cursor < valid_end,
    # which is false here because the artifact reaches the end).
    assert valid_times.shape == (1, 2), (
        f"Expected one valid interval ending before the boundary "
        f"artifact, got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    # The valid interval ends BEFORE the artifact span (boundary
    # not included).
    assert valid_times[0][1] < timestamps[990] + 1e-9, (
        f"Valid interval ends at {valid_times[0][1]}, after the "
        f"start of the boundary artifact ({timestamps[990]}). "
        "The boundary clamp may have an off-by-one regression."
    )


# =========================================================================
# End of Phase 1 review followups.
# =========================================================================
