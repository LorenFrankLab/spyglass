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

    The parametrize list covers Recording / ArtifactDetection /
    Sorting -- the three v2 tables that use tri-part dispatch --
    so a future "consolidate back into ``make``" change fails
    loudly.
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
    # The smoke fixture's timestamps are monotonic, so the repair never
    # fires: provenance columns stay at their no-repair defaults.
    assert int(row["timestamps_adjusted"]) == 0
    assert int(row["n_adjusted_samples"]) == 0
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
    # ``cache_hash`` is an ``NwbfileHasher`` digest. The
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
_TETRODE_60S_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_tetrode_60s.nwb"
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
        SharedArtifactGroup,
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

    # Step 4: SharedArtifactGroup tables. The master FK's Session
    # and the Member part FK's Recording, so prior shared-group
    # rows pointing at THIS session's Recording rows must be
    # cleaned before we drop Recording -- otherwise DJ refuses to
    # cascade through the part-without-master constraint. Delete
    # master + Member explicitly (master first satisfies the
    # master-before-part rule).
    shared_groups = (
        SharedArtifactGroup & session_key
    ).fetch("KEY", as_dict=True)
    if shared_groups:
        (SharedArtifactGroup & shared_groups).super_delete(warn=False)

    # Step 5: now the cascade is unblocked -- Recording, SortGroupV2
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
def test_shared_artifact_group_insert_validates_inputs(populated_recording):
    """``SharedArtifactGroup.insert_group`` validates session
    consistency, recording existence, sampling-frequency match,
    duration / n_samples match, and non-empty members.

    The time-axis check is intentionally stricter than "same
    nwb_file_name" -- ``RecordingSelection`` identity includes
    ``interval_list_name``, so same NWB does NOT imply same
    n_samples or fs. The invariants asserted here mirror what
    ``si.aggregate_channels`` requires inside ``make_compute``.
    """
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    # Empty members -> ValueError.
    with pytest.raises(ValueError, match="members list is empty"):
        SharedArtifactGroup.insert_group("empty_group", [])

    # Member missing recording_id -> ValueError.
    with pytest.raises(ValueError, match="must include 'recording_id'"):
        SharedArtifactGroup.insert_group(
            "bad_member", [{"not_recording_id": "x"}]
        )

    # Non-existent recording_id -> ValueError.
    import uuid as _uuid

    bogus_rid = _uuid.uuid4()
    with pytest.raises(ValueError, match="not in Recording"):
        SharedArtifactGroup.insert_group(
            "bogus_recording",
            [{"recording_id": bogus_rid}],
        )

    # Happy path: one real recording -> master + Member rows
    # written in one transaction. With a single member every
    # multi-member-consistency check (sessions / fs / duration)
    # trivially passes; the multi-member-divergence checks below
    # exercise the real validation paths.
    SharedArtifactGroup.insert_group(
        "v2_solo_group",
        [{"recording_id": populated_recording["recording_id"]}],
    )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )
    assert (
        len(
            SharedArtifactGroup.Member
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_durations(
    populated_recording, polymer_smoke_session
):
    """``insert_group`` rejects members whose recordings have
    different exact ``n_samples`` (different ``interval_list_name``
    on the upstream selection).

    Same NWB, same fs -- but two ``RecordingSelection`` rows
    pointing at different ``interval_list_name`` values can have
    different sample counts. ``si.aggregate_channels`` would
    crash with an opaque shape mismatch deep inside SI; the
    insert-time guard loads each member's preprocessed recording
    and asserts EXACT ``get_num_samples()`` parity so a 1-sample
    mismatch -- which the earlier duration-tolerance check let
    through -- surfaces before the user pays the populate cost.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]

    # Build a SECOND RecordingSelection on the same NWB but with a
    # shorter IntervalList, then populate Recording so we have two
    # Recording rows with the same nwb_file_name but different
    # duration_s.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    truncated = _np.asarray([[t0, t0 + 1.5]])  # 1.5s vs full ~4s
    truncated_interval = "v2_shared_group_test_truncated"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": truncated_interval,
            "valid_times": truncated,
            "pipeline": "shared_group_validation_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": truncated_interval,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    # Now attempt to group the full-window populated_recording AND
    # the truncated second_rec_pk. The strict n_samples check at
    # insert_group time MUST raise.
    with pytest.raises(ValueError, match="differing exact n_samples"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_durations",
            [
                {"recording_id": populated_recording["recording_id"]},
                {"recording_id": second_rec_pk["recording_id"]},
            ],
        )
    # No master row was created.
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_durations"}
        )
        == 0
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_dtypes(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """``insert_group`` rejects members whose preprocessed
    recordings have differing ``get_dtype()`` values.

    Constructing two real recordings of the same NWB with
    different dtypes requires preproc-param plumbing the test
    fixtures don't currently exercise; monkeypatch
    ``Recording.get_recording`` to return SI-compatible stubs that
    share ``n_samples`` but differ in ``get_dtype()``. This
    exercises the dtype branch of the strict check; the
    n_samples branch is exercised by
    ``test_shared_artifact_group_insert_rejects_mismatched_durations``
    above.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_group_test_dtype_window"
    # 2.5s window is well above the min_segment_length=1.0s filter
    # the Recording pipeline applies; we only need a SECOND
    # Recording row so the test has two distinct recording_ids to
    # group, the dtype divergence is injected via monkeypatch
    # below.
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": _np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_dtype_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": second_interval,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    rid_a = populated_recording["recording_id"]
    rid_b = second_rec_pk["recording_id"]

    class _FakeRec:
        def __init__(self, n_samples, dtype_str):
            self._n = n_samples
            self._dtype = dtype_str

        def get_num_samples(self):
            return self._n

        def get_dtype(self):
            return self._dtype

    def _fake_get_recording(self, key):
        # Both stubs share ``n_samples`` so the dtype branch -- not
        # the n_samples branch -- is the only thing that can raise.
        if str(key["recording_id"]) == str(rid_a):
            return _FakeRec(n_samples=30000, dtype_str="float32")
        return _FakeRec(n_samples=30000, dtype_str="int16")

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    with pytest.raises(ValueError, match="differing dtypes"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_dtypes",
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_dtypes"}
        )
        == 0
    )


import numpy as _np  # noqa: E402 -- used by the tests above


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
        with pytest.raises(
            RecordingTruncatedError, match="Missing: .* seconds|Missing: \\d"
        ):
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
def test_recording_truncation_multi_interval(polymer_smoke_session):
    """A disjoint multi-interval request that runs past raw coverage
    raises ``RecordingTruncatedError`` and inserts no row.

    The template ``test_recording_truncation_caught`` covers the
    single-interval over-request case; this is the disjoint analogue, and
    locks the truncation guard on the multi-interval/concat path. Two
    sort intervals with a real inter-segment gap, where the final segment
    extends 1 s past the raw recording's last valid time.

    The guard is correct on this path because the concatenated recording
    is built with ``ignore_times=True``, so its ``get_times()`` is 0-based
    and the saved-duration reference EXCLUDES the inter-segment gap (it is
    the sum of kept-frame durations, not the wall-clock envelope). The
    requested total likewise excludes gaps, so ``missing`` equals the
    over-request (here ~1 s) rather than collapsing to <= 0 -- a gap-
    inclusive envelope would instead spuriously flag every disjoint sort.
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
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    raw_start = float(raw_times[0][0])
    raw_end = float(raw_times[-1][-1])
    # First segment [raw_start, +1.5 s]; 1 s gap; second segment from
    # +2.5 s extends 1 s PAST raw_end (the over-request). After clipping
    # to raw_end the second segment is still > 1 s, so it survives the
    # min_segment_length filter and both segments take the concat path.
    assert raw_end - raw_start >= 3.5, (
        f"fixture too short for multi-interval over-request test: raw "
        f"span [{raw_start}, {raw_end}] needs >= 3.5 s"
    )
    seg1_end = raw_start + 1.5
    seg2_start = seg1_end + 1.0
    over_request_s = 1.0
    multi_interval_name = "v2_test_multi_over_request"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": multi_interval_name,
            "valid_times": _np.asarray(
                [[raw_start, seg1_end], [seg2_start, raw_end + over_request_s]]
            ),
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
        "interval_list_name": multi_interval_name,
        "preproc_params_name": "default_franklab",
        "team_name": "v2_test_team",
    }
    pk = RecordingSelection.insert_selection(selection_key)

    (Recording & pk).super_delete(warn=False)
    try:
        with pytest.raises(RecordingTruncatedError, match="Missing: \\d"):
            Recording.populate(pk, reserve_jobs=False)
        # No row may be inserted for an over-request -- otherwise a
        # downstream consumer silently reads a window that does not cover
        # what was requested.
        assert not (Recording & pk)
    finally:
        (Recording & pk).super_delete(warn=False)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": multi_interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_recording_timestamp_repair_recorded(
    polymer_smoke_session, monkeypatch
):
    """A non-monotonic source records repair provenance on the row.

    The smoke fixture's timestamps are already monotonic, so the repair
    never fires on real data. We instead force ``_repaired_timestamps``
    to report a known adjusted-sample count (its array output stays the
    real, monotonic vector so the rest of the write is unaffected). This
    exercises the provenance THREADING: ``n_changed`` must flow from
    ``_repaired_timestamps`` through ``_compute_recording_artifact`` and
    ``RecordingComputed`` into the row's ``timestamps_adjusted`` /
    ``n_adjusted_samples`` columns. The no-repair (False) case is
    asserted by ``test_recording_populates_and_round_trips``.
    """
    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
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

    raw_vt = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    interval_name = "v2_test_repair_provenance"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
            "valid_times": raw_vt,
            "pipeline": "v2_test_repair",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )

    forced_n_changed = 7
    original = Recording._repaired_timestamps

    def _fake_repaired(recording, raw_path, recording_id=None):
        # Real (monotonic) timestamps so the write succeeds; only the
        # reported adjusted-sample count is forced.
        timestamps, _ = original(
            recording, raw_path, recording_id=recording_id
        )
        return timestamps, forced_n_changed

    monkeypatch.setattr(
        Recording, "_repaired_timestamps", staticmethod(_fake_repaired)
    )

    (Recording & pk).super_delete(warn=False)
    try:
        Recording.populate(pk, reserve_jobs=False)
        row = (Recording & pk).fetch1()
        assert bool(row["timestamps_adjusted"]) is True
        assert int(row["n_adjusted_samples"]) == forced_n_changed
    finally:
        (Recording & pk).super_delete(warn=False)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_recording_single_interval_persists_repaired_timestamps(
    polymer_smoke_session, monkeypatch
):
    """A single-interval write caches the REPAIRED timestamps, not the
    frame-slice's own (uncorrected) time vector.

    Forces a detectable, consolidation-safe correction and captures what
    ``_write_nwb_artifact`` writes: the persisted vector must be the
    repaired one (``timestamps_override`` is not None on the
    single-interval path), so a ``timestamps_adjusted=True`` row cannot
    cache still-non-monotonic samples.
    """
    import numpy as _np

    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
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
    pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )

    # Return a corrected vector with a tiny ascending ramp: detectably
    # distinct from the source's own times AND with a slightly larger
    # span, so the row's duration_s (derived from the persisted vector)
    # differs from the frame-slice's own span. The ramp end (~2e-5 s) is
    # below both the sample spacing (no consolidation-boundary shift) and
    # the truncation tolerance (no spurious RecordingTruncatedError).
    orig_repaired = Recording._repaired_timestamps

    def _shifted(recording, raw_path, recording_id=None):
        timestamps, _ = orig_repaired(
            recording, raw_path, recording_id=recording_id
        )
        timestamps = _np.asarray(timestamps)
        ramp = _np.linspace(0.0, 2e-5, len(timestamps))
        return timestamps + ramp, 5

    monkeypatch.setattr(
        Recording, "_repaired_timestamps", staticmethod(_shifted)
    )

    captured = {}
    orig_write = Recording._write_nwb_artifact

    def _capture(
        recording,
        nwb_file_name,
        existing_analysis_file_name=None,
        timestamps_override=None,
    ):
        captured["override"] = timestamps_override
        captured["recording_own"] = _np.asarray(recording.get_times())
        return orig_write(
            recording,
            nwb_file_name,
            existing_analysis_file_name,
            timestamps_override,
        )

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture)
    )

    (Recording & pk).super_delete(warn=False)
    try:
        Recording.populate(pk, reserve_jobs=False)
        override = captured["override"]
        assert override is not None, (
            "single-interval write got timestamps_override=None; the "
            "repaired timestamps were not persisted"
        )
        override = _np.asarray(override)
        # The persisted vector is the repaired one, distinct from the
        # frame-slice's uncorrected get_times().
        assert not _np.array_equal(override, captured["recording_own"])
        assert override[-1] > captured["recording_own"][-1]
        # The row's duration_s tracks the PERSISTED (repaired) vector,
        # not the frame-slice's own (uncorrected, slightly shorter) span.
        # duration_s is stored single-precision, so assert relative
        # closeness rather than an absolute float64 tolerance.
        row_duration = float((Recording & pk).fetch1("duration_s"))
        override_span = float(override[-1] - override[0])
        uncorrected_span = float(
            captured["recording_own"][-1] - captured["recording_own"][0]
        )
        assert abs(row_duration - override_span) < abs(
            row_duration - uncorrected_span
        )
    finally:
        (Recording & pk).super_delete(warn=False)
        (RecordingSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_fetch_sort_group_probe_info_stable_order(
    polymer_smoke_session, monkeypatch
):
    """``_fetch_sort_group_probe_info`` returns a deterministic,
    electrode_id-ordered result.

    The tri-part populate contract hashes the ``RecordingFetched`` tuple
    twice and raises if the two fetches differ; an unordered DB fetch can
    reorder rows between calls. The fetch must therefore pin
    ``order_by="electrode_id"``. This guards that invariant directly by
    spying on DataJoint's fetch call, so dropping the kwarg fails even if
    the database happens to return primary-key/electrode order by default.
    """
    from datajoint.fetch import Fetch

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import Recording, SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    channel_ids = sorted(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch("electrode_id"),
        key=int,
    )
    assert len(channel_ids) > 1, "need a multi-channel sort group"

    fetch_order_by = []
    original_fetch_call = Fetch.__call__

    def _spy_fetch(self, *attrs, **kwargs):
        fetch_order_by.append(kwargs.get("order_by"))
        return original_fetch_call(self, *attrs, **kwargs)

    monkeypatch.setattr(Fetch, "__call__", _spy_fetch)

    first = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    second = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    assert fetch_order_by == ["electrode_id", "electrode_id"]
    assert first == second, "probe-info fetch is not deterministic"

    ref_rows = (
        Electrode * Probe
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": int(c)} for c in channel_ids]
    ).fetch(
        "probe_type",
        "electrode_group_name",
        as_dict=True,
        order_by="electrode_id",
    )
    expected = (
        tuple(r["probe_type"] for r in ref_rows),
        tuple(r["electrode_group_name"] for r in ref_rows),
    )
    assert first == expected, "probe-info fetch is not electrode_id-ordered"
    assert len(first[0]) == len(channel_ids)


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
    sort; auto-incremented curation_id starts at 0 and increments.

    For v1 surface parity ``labels=None`` is treated as no-labels
    rather than raising; this test asserts the permissive contract.
    Root-curation insertion is idempotent; the repeated-root
    assertion is exercised in ``test_root_curation_idempotent``
    below.
    """
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

    # ``labels=None`` is accepted (equivalent to ``labels={}``).
    pk_none = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=None,
        parent_curation_id=0,
    )
    assert pk_none["curation_id"] == 2


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


def test_electrode_group_sort_key_tolerates_non_numeric(dj_conn):
    """``set_group_by_shank``'s group ordering must not assume numeric
    ``electrode_group_name`` values.

    A plain ``int(name)`` raised ``ValueError`` on free-form NWB group
    names (e.g. ``"probeA"``). The sort key now sorts numeric names
    numerically (and ahead of non-numeric) and non-numeric names
    lexically, so valid datasets no longer crash while purely numeric
    names keep their natural order. (``dj_conn`` only because importing
    the module declares its schema.)
    """
    from spyglass.spikesorting.v2.recording import _electrode_group_sort_key

    # Numeric names keep NUMERIC (not lexical) order: "10" after "2".
    assert sorted(["10", "2", "1"], key=_electrode_group_sort_key) == [
        "1",
        "2",
        "10",
    ]
    # Non-numeric names do not raise and sort after the numeric ones.
    assert sorted(
        ["probeB", "2", "probeA", "1"], key=_electrode_group_sort_key
    ) == ["1", "2", "probeA", "probeB"]


def test_unit_brain_region_df_empty_keeps_full_schema(dj_conn):
    """An empty unit-region lookup returns the full column schema.

    ``pd.DataFrame([])`` would drop every column, leaving callers a frame
    with only ``region_resolution``; the builder now pins ``columns=`` so
    empty and non-empty results share a shape.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.utils import unit_brain_region_df

    empty_units = CurationV2.Unit & "unit_id = -1"  # no unit has id -1
    df = unit_brain_region_df(empty_units, "single_session")
    assert len(df) == 0
    assert {
        "unit_id",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
        "region_resolution",
    } <= set(df.columns)


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

    Uses the ``franklab_tetrode_mountainsort5`` preset (matches the
    ``preset=`` argument below). Idempotency on rerun is verified
    against MS5 by reusing the existing sorting_id rather than
    re-running the sorter, so MS5's clustering randomness does not
    affect the rerun assertion.
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
        "n_units",
    }
    assert set(manifest.keys()) == expected_keys
    assert manifest["preset"] == "franklab_tetrode_mountainsort5"
    assert manifest["curation_id"] == 0  # root curation
    assert manifest["n_units"] >= 1
    # The smoke fixture's clusterless 100 uV default IS exercised in
    # ``test_run_v2_pipeline_clusterless_default_handles_zero_units_
    # gracefully`` -- it confirms the orchestrator's zero-unit
    # short-circuit returns a partial manifest with
    # ``curation_id=None, merge_id=None`` instead of crashing inside
    # CurationV2.insert_curation. MS5 here is the populated-units
    # path.

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

    # Insert the smoke-fixture sort row from the shared constants
    # so any future param tweak lands in one place
    # (tests/spikesorting/v2/_smoke_constants.py) and the v1
    # baseline-capture + v2 pipeline + parity test all stay in
    # lockstep.
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    custom_params_name = SMOKE_CLUSTERLESS_PARAM_NAME
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": custom_params_name,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 3,
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
    # Peak amplitude sanity: must be a finite positive number.
    # NOTE: we cannot assert ``peak_amplitude_uv >= detect_threshold``
    # because ``detect_threshold`` is interpreted by SI's
    # ``detect_peaks`` in the recording's native units (raw counts,
    # since the v2 preprocessing pipeline -- bandpass +
    # common_reference -- does NOT gain-scale traces to uV before
    # detection; gain conversion happens only at NWB-write time via
    # ``ElectricalSeries.conversion``). The "detect_threshold stays
    # in microvolts" docstring inherits a v1-era assumption that
    # only holds if the recording was pre-scaled to uV. The unit-confusion
    # is pre-existing and out of scope here; document it so a future
    # maintainer doesn't reinstate the over-specified assertion. The
    # template peak (post-gain-applied via channel_gains in
    # _build_analyzer) being ~0.6 uV is consistent with a 5-count
    # detection threshold on a ~0.2 uV/count probe.
    assert unit_row["peak_amplitude_uv"] > 0, (
        f"Clusterless reported non-positive peak_amplitude_uv="
        f"{unit_row['peak_amplitude_uv']}; a detector that returns "
        "zero or negative amplitudes is broken (sanity floor)."
    )
    import math

    assert math.isfinite(unit_row["peak_amplitude_uv"])


# ---------- 60s MEArec ground-truth correctness gate ----------------------


@pytest.mark.slow
def test_mountainsort5_ground_truth_polymer_60s(polymer_60s_session):
    """MS5 on the 60s polymer fixture finds the planted units.

    Primary correctness gate: per-unit accuracy >= 0.7 for at least
    half of planted units, computed via
    ``spikeinterface.comparison.compare_sorter_to_ground_truth``
    against the sidecar ground-truth units table written by
    ``mearec_to_spyglass_nwb`` and read via
    ``get_ground_truth_units_table``.
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

    # Ground truth: the planted MEArec units stored in the sidecar
    # ``ProcessingModule("ground_truth")["units"]`` table. We don't
    # use ``ImportedSpikeSorting`` (its merge-table dispatch raises
    # NotImplementedError for v0/v1/imported sources); we read the
    # NWB directly via the sidecar helper and build a NumpySorting.
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

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
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {}
        for idx, unit_id in enumerate(gt_units_table.id[:]):
            gt_units[int(unit_id)] = np.asarray(
                gt_units_table["spike_times"][idx]
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
# Test-appropriateness review followups: gap-closing tests.
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
        # _apply_artifact_mask now takes valid_times directly (no
        # DB I/O inside Sorting.make_compute per the tri-part
        # contract). Pass the same valid_times the IntervalList row
        # carries.
        masked = Sorting._apply_artifact_mask(
            recording=recording,
            valid_times=valid_times,
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

    def _empty_rows(
        cls, sorting_id, sorting_units, merge_groups, curation_id, apply_merge
    ):
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
    # ``ClusterlessThresholderSchema`` (extra="forbid") rejects
    # ``outputs`` and ``random_chunk_kwargs``; the test params
    # dict must not carry them. The runtime strip path in
    # ``Sorting._run_sorter`` still tolerates either shape -- but
    # ``SorterParameters.insert1`` validates through Pydantic
    # first, so the v1-era ``outputs`` key would fail at insert
    # time.
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAMS,
    )

    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 3,
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


@pytest.mark.slow
def test_run_v2_pipeline_clusterless_default_handles_zero_units_gracefully(
    polymer_smoke_session,
):
    """``run_v2_pipeline`` with the SHIPPED ``clusterless_thresholder``
    default (100 uV ``detect_threshold``) on the smoke fixture
    finds zero peaks AND completes without crashing.

    On a zero-unit sort the orchestrator short-circuits curation
    because SI's ``NwbSortingExtractor`` cannot open an empty
    units NWB; the returned manifest is PARTIAL with
    ``curation_id=None, merge_id=None, n_units=0`` instead of a
    complete one. The alternative (forcing curation through)
    crashes with an opaque KeyError. The partial-manifest contract
    gives the user a clean signal that the sort ran and found
    nothing.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import Sorting

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

    # Use the SHIPPED default (100 uV) which finds zero peaks on
    # the smoke fixture. NO row mutation -- this is the production-
    # contract test.
    manifest = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        preset="franklab_tetrode_clusterless_thresholder",
    )

    # Recording / artifact / sorting IDs are populated; curation +
    # merge are explicitly None because SI's NwbSortingExtractor
    # cannot open an empty units NWB and the orchestrator short-
    # circuits curation when n_units == 0.
    for key in ("recording_id", "artifact_id", "sorting_id"):
        assert manifest.get(key) is not None, (
            f"Zero-unit manifest missing {key!r}; got {manifest}."
        )
    assert manifest["n_units"] == 0
    assert manifest["curation_id"] is None, (
        "Orchestrator should NOT run CurationV2.insert_curation on "
        "a zero-unit sort -- SI cannot open the empty units NWB "
        "and the curation step would crash. The partial-manifest "
        "contract returns ``curation_id=None`` instead."
    )
    assert manifest["merge_id"] is None

    # Sorting row exists with ``n_units == 0``.
    sort_pk = {"sorting_id": manifest["sorting_id"]}
    sort_row = (Sorting & sort_pk).fetch1()
    assert sort_row["n_units"] == 0, (
        f"Expected zero units from shipped clusterless default "
        f"on the smoke fixture, got n_units={sort_row['n_units']}. "
        "Either the smoke fixture's amplitudes changed (unlikely) "
        "or the unit-conversion semantics shifted -- audit the "
        "detect_threshold units."
    )
    assert len(Sorting.Unit & sort_pk) == 0, (
        "Sorting.Unit has rows for a zero-unit sort; the "
        "_populate_unit_part path should iterate zero ids and "
        "insert zero rows."
    )

    # ---- zero-unit loud-but-graceful loader contracts ----
    # The sort/recording/artifact are already populated, so these reuse
    # the one expensive clusterless sort above.
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        ZeroUnitAnalyzerError,
        ZeroUnitSortError,
    )

    # require_units=True turns the same zero-unit sort into a hard error.
    with pytest.raises(ZeroUnitSortError):
        run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            preset="franklab_tetrode_clusterless_thresholder",
            require_units=True,
        )

    # get_analyzer raises (no analyzer is buildable over zero units) --
    # it must not try to load the never-built folder. The stored
    # analyzer_folder column is still a real path string (guard lives in
    # get_analyzer, not the schema).
    assert (
        isinstance(sort_row["analyzer_folder"], str)
        and sort_row["analyzer_folder"]
    )
    with pytest.raises(ZeroUnitAnalyzerError):
        Sorting().get_analyzer(sort_pk)

    # get_sorting returns an EMPTY sorting (zero units is valid data),
    # not a crash on the empty units NWB.
    empty_sorting = Sorting().get_sorting(sort_pk)
    assert empty_sorting.get_num_units() == 0

    # CurationV2.get_sorting on a zero-unit curation also returns an
    # empty sorting -- it builds its OWN extractor (does not delegate),
    # so it needs the same guard.
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk, labels={}, description="zero-unit curation"
    )
    assert len(CurationV2.Unit & curation_pk) == 0
    empty_curated = CurationV2().get_sorting(curation_pk)
    assert empty_curated.get_num_units() == 0


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

    # The analyzer folder created by ``_build_analyzer`` must also
    # be removed by ``make_insert``'s rollback path. A 5-50 GB
    # analyzer folder orphan per failed populate is the leak this
    # guards against.
    from spyglass.spikesorting.v2.utils import _analyzer_path

    analyzer_folder = _analyzer_path(sort_pk)
    assert not analyzer_folder.exists(), (
        f"Sorting.make rollback left analyzer folder {analyzer_folder} "
        "on disk. The except-block in Sorting.make_insert must "
        "shutil.rmtree it when the transaction rolls back."
    )


# ---------- Boundary-spike round-trip (clip decision gate) -----------------


@pytest.mark.slow
def test_boundary_spike_round_trip_does_not_raise(
    polymer_smoke_session, monkeypatch
):
    """A spike at the recording's final sample survives the v2 NWB round-trip.

    This test gates the decision on whether to fold in v1's
    ``spike_times_to_valid_samples`` clip. SpikeInterface's
    ``NwbSortingExtractor`` has historically been strict about
    spikes that map back to a sample at or past ``n_samples`` after
    the ``timestamps[sample_index] -> spike_times[]`` -> ``round((t
    - t_start) * fs)`` round-trip. If SI 0.104 raises here, the v1
    clip is required on read; if it does not raise, the clip is
    unnecessary and the test is kept as a regression guard.

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
    def _boundary_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
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
        "a boundary spike; this is the failure mode the clip-decision "
        "gate watches for but inside _write_units_nwb rather than at "
        "read time."
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
    # accepts the call on the strict signature.
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description="boundary-spike round-trip test",
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

        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
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


def test_run_si_sorter_restores_global_job_kwargs(dj_conn, monkeypatch):
    """``_run_si_sorter`` leaves SI's global job kwargs byte-identical to
    their pre-sort state.

    ``set_global_job_kwargs`` UPDATES the global rather than replacing it,
    so a job kwarg the sort installs that is absent from the default
    global set (``chunk_size``/``total_memory``/``chunk_memory``) would
    otherwise leak into every later populate. The actual sort is stubbed
    so only the save/restore is exercised.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    si.reset_global_job_kwargs()
    before = dict(si.get_global_job_kwargs())
    assert "chunk_size" not in before  # the key we expect could leak

    monkeypatch.setattr(sis, "run_sorter", lambda **kw: "DUMMY")

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    out = Sorting._run_si_sorter(
        sorter="mountainsort5",
        sorter_params={"whiten": False},
        recording=rec,
        sorting_id="r3_test",
        job_kwargs={"n_jobs": 1, "chunk_size": 1000},
    )
    assert out == "DUMMY"

    after = dict(si.get_global_job_kwargs())
    assert after == before, (
        f"global job kwargs leaked across the sort: before={before}, "
        f"after={after}"
    )
    assert "chunk_size" not in after
    si.reset_global_job_kwargs()


def test_clusterless_detect_peaks_strips_random_seed(dj_conn, monkeypatch):
    """``_run_clusterless_thresholder`` strips ``random_seed`` from the
    job kwargs before calling ``detect_peaks``.

    ``random_seed`` is a Spyglass-side knob (already threaded into
    ``random_slices_kwargs``); it is not a valid SI job kwarg, so SI's
    ``fix_job_kwargs`` raises ``AssertionError`` if it reaches
    ``detect_peaks``. ``detect_peaks`` is stubbed to capture the job
    kwargs it receives.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["job_kwargs"] = job_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    Sorting._run_clusterless_thresholder(
        sorter_params={"detect_threshold": 5.0, "noise_levels": [1.0]},
        recording=rec,
        job_kwargs={"random_seed": 7, "n_jobs": 1},
    )

    jk = captured["job_kwargs"]
    assert jk is not None, "detect_peaks was not called"
    assert "random_seed" not in jk, (
        f"random_seed leaked into detect_peaks job kwargs: {jk}"
    )
    assert jk.get("n_jobs") == 1  # other kwargs preserved


def test_detect_artifacts_zscore_only_detection(dj_conn):
    """``_detect_artifacts`` runs the z-score-only branch when
    ``amplitude_thresh_uV is None``.

    The z-score is **across channels per frame** (matching v1's
    semantics); a frame where one channel deviates substantially
    from the others trips. Common-mode pops where every channel
    jumps together do NOT trip (per-frame mean shifts but
    per-frame std stays ~0, so z ~= 0 everywhere on that row).

    Synthetic: low-amplitude per-channel uncorrelated background +
    a 10-sample artifact on ONE channel (the others stay quiet at
    the artifact frames). Cross-channel z on that channel = high.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rng = _np.random.default_rng(42)
    # 32 channels so the cross-channel z-score on a single-channel
    # excursion can plausibly exceed ``zscore_thresh``. With N
    # channels and one channel at X while the rest are near zero,
    # the per-frame z on the spiking channel is bounded by
    # ~sqrt(N-1) (an algebraic upper limit of cross-channel
    # z-score, regardless of X). 4 channels caps at ~1.7; 32
    # channels caps at ~5.6 so a threshold of 4.0 sits inside
    # the achievable range with headroom.
    n_samples, n_channels = 5000, 32
    traces = (
        rng.normal(0.0, 0.5, size=(n_samples, n_channels))
        .astype(_np.float32)
    )
    # Single-channel artifact: channel 0 spikes to 500 uV at frames
    # 2000-2009; the other 31 channels stay quiet.
    traces[2000:2010, 0] = 500.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=None,  # z-score only
        zscore_thresh=4.0,
        proportion_above_thresh=1.0 / n_channels,  # any-1-channel triggers
        removal_window_ms=0.05,
        join_window_ms=0.0,

        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
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
    """``_detect_artifacts`` OR-combines thresholds when both are set.

    v2 matches v1's OR semantics (``np.logical_or(above_z,
    above_a)`` at ``v1/utils.py:198``). Under OR a frame is flagged
    if EITHER detector trips, so this test's synthetic 80 uV
    baseline (which clears the 50 uV amplitude threshold on every
    channel everywhere) flags the entire recording: the baseline
    frames trip amplitude alone, and the 200 uV step deviation at
    frames 3000-3099 trips both detectors. Net result:
    ``valid_times`` is empty (whole recording flagged as artifact)
    under OR; an earlier AND combine flagged only the step region.
    Keeping the synthetic setup unchanged so the test name and data
    continue to point at the AND-vs-OR semantics directly; the
    assertion is what flips.

    The cross-channel z-score is also active here: the z-score is
    computed across channels per frame (uniform baseline -> ~0
    z-score), not per channel over time. Combined with OR, the
    baseline still flags via amplitude, so the test outcome is the
    same regardless of the z-score axis change.
    """
    import numpy as _np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    # 80 uV uniform baseline clears the 50 uV amplitude threshold
    # on every channel for every frame; the 200 uV step at
    # frames 3000-3099 trips both amplitude and (in v1's
    # cross-channel form, briefly) z-score. Under OR everything
    # is artifact.
    traces = _np.full((n_samples, n_channels), 80.0, dtype=_np.float32)
    traces[3000:3100, :] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=5.0,
        proportion_above_thresh=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,

        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)

    # OR mode: baseline trips amplitude alone -> every frame is
    # flagged. ``valid_times`` is shape (0, 2). v1 parity restored
    # after an earlier silent flip to AND.
    assert valid_times.shape == (0, 2), (
        f"OR-mode (amplitude OR z-score) expected to flag the entire "
        f"recording given the 80 uV uniform baseline clears the 50 uV "
        f"amplitude threshold on every channel; got shape "
        f"{valid_times.shape}. v2 enforces v1's OR semantics; if "
        f"this fails with shape (2, 2), an AND combine has silently "
        f"regressed."
    )


def test_detect_artifacts_join_window_merges_runs(dj_conn):
    """``_detect_artifacts`` merges two artifact runs separated by
    fewer than ``join_window_frames`` into a single artifact span.

    Exercises the ``cur_end = f`` branch inside
    ``_detect_artifacts``'s join loop which the single-run test
    doesn't hit. Builds two transients separated by 10 frames;
    with join_window_ms set so join_window_frames > 10, the runs
    merge into one. With a smaller join window, the runs stay
    separate.
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

            min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
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
    #
    # ``min_length_s`` must be < the post-widening gap or the
    # sliver filter eats the middle interval. The middle sliver is
    # ``(10 frames gap) - 2*half_window_frames(=1) = 8 frames``
    # = ~0.27 ms at 30 kHz. Use 0.0001 s (0.1 ms) to keep it.
    unmerged = ArtifactDetection._detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_thresh_uV=50.0,
            zscore_thresh=None,
            proportion_above_thresh=0.5,
            removal_window_ms=0.05,
            join_window_ms=0.1,
            min_length_s=0.0001,
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


# ---------- CurationV2 with merge_groups / apply_merge=True --------------


@pytest.mark.slow
def test_curation_v2_insert_with_merge_groups_apply_merges(
    polymer_60s_session,
):
    """``CurationV2.insert_curation`` with ``merge_groups`` +
    ``apply_merge=True`` writes a curated NWB whose unit_ids are
    the merge-group heads, and whose merged unit's spike train is
    the sorted union of its contributors.

    Exercises ``_build_curated_unit_rows`` with a non-empty merge
    list (amplitude-inheritance: kept unit gets electrode/amplitude
    from the highest-amplitude contributor) AND
    ``_stage_curated_units_nwb`` with ``apply_merge=True``
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
        apply_merge=True,
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
    # ``apply_merge=True`` writes concatenated spike trains.
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
    # (apply_merge=True end-to-end half of the n_spikes invariant: DB
    # n_spikes == curated NWB spike-train length for the merged head.)
    head_row = (CurationV2.Unit & pk & {"unit_id": head}).fetch1()
    assert head_row["n_spikes"] == len(expected_merged), (
        f"CurationV2.Unit.n_spikes for merged head = "
        f"{head_row['n_spikes']}; expected {len(expected_merged)}."
    )
    assert head_row["n_spikes"] == len(merged_times)


def test_curation_n_spikes_matches_apply_merge(dj_conn):
    """``_build_curated_unit_rows`` computes ``n_spikes`` to match what
    ``_stage_curated_units_nwb`` writes for the SAME ``apply_merge``.

    A merge-group head's curated spike train is the contributor sum only
    when ``apply_merge=True`` (the staged train is the concatenation);
    with ``apply_merge=False`` the staged train is head-only, so
    ``n_spikes`` must be the head's own count -- NOT the merged sum.
    Before the fix the count was always the merged sum, so an
    ``apply_merge=False`` preview stored a count that disagreed with its
    own head-only spike train. Synthetic ``Sorting.Unit`` rows isolate
    the count logic (the ``apply_merge=True`` end-to-end half of the
    invariant is covered by
    ``test_curation_v2_insert_with_merge_groups_apply_merges``).
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    def _unit(uid, n_spikes, amp):
        return {
            "unit_id": uid,
            "n_spikes": n_spikes,
            "peak_amplitude_uv": amp,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": uid,
        }

    # head=0 (higher amplitude) absorbs 1; unit 2 passes through.
    sorting_units = [_unit(0, 100, 50.0), _unit(1, 40, 30.0), _unit(2, 7, 20.0)]
    merge_groups = [[0, 1]]

    def n_spikes_by_unit(apply_merge):
        rows, _ = CurationV2._build_curated_unit_rows(
            sorting_id="s",
            sorting_units=sorting_units,
            merge_groups=merge_groups,
            curation_id=0,
            apply_merge=apply_merge,
        )
        return {r["unit_id"]: r["n_spikes"] for r in rows}

    merged = n_spikes_by_unit(apply_merge=True)
    preview = n_spikes_by_unit(apply_merge=False)

    assert merged[0] == 140  # merged sum (matches concatenated train)
    assert preview[0] == 100  # head-only (matches head-only train)
    assert merged[2] == 7 and preview[2] == 7  # non-merged: unaffected


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

        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
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

    The ``ConcatenatedRecordingSource`` schema is declared in its
    final shape under the zero-migration policy, but the make-body
    branch is not implemented yet. Pins the gate so a premature
    attempt to wire up the concat path immediately fails this
    test.
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


@pytest.mark.slow
def test_sorting_selection_artifact_id_none_is_distinct_identity(
    populated_sorting,
):
    """``artifact_id=None`` and ``artifact_id=<uuid>`` are distinct identities.

    Regression guard: an earlier ``insert_selection`` lookup omitted
    ``artifact_id`` from the master restriction whenever it was None,
    which effectively meant "match any artifact_id." That caused a
    no-artifact ``insert_selection`` call to alias onto a pre-existing
    artifact-backed row for the same
    ``(recording_id, sorter, sorter_params_name)`` triple. Since the
    schema makes ``artifact_id`` a real nullable FK, None must be
    treated as a real, distinct identity value (``artifact_id IS NULL``)
    -- not as a wildcard.

    The ``populated_sorting`` fixture creates an artifact-backed row;
    we then request a no-artifact row with the same recording/sorter/
    params and assert the helper creates a NEW ``sorting_id`` rather
    than returning the artifact-backed one.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    artifact_backed_pk = populated_sorting
    # Derive recording_id + sorter + params from the fixture row so this
    # test does not silently drift if ``populated_sorting`` changes its
    # sorter/params choice (Finding #3 from review).
    backed_row = (SortingSelection & artifact_backed_pk).fetch1()
    rec_row = (SortingSelection.RecordingSource & artifact_backed_pk).fetch1()
    recording_id = rec_row["recording_id"]
    sorter = backed_row["sorter"]
    sorter_params_name = backed_row["sorter_params_name"]
    assert backed_row["artifact_id"] is not None, (
        "populated_sorting must yield an artifact-backed row for this "
        "test to be meaningful; got artifact_id IS NULL on the fixture."
    )

    no_artifact_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_id": None,
        }
    )
    assert no_artifact_pk["sorting_id"] != artifact_backed_pk["sorting_id"], (
        "insert_selection with artifact_id=None aliased onto the "
        "artifact-backed row; None must restrict to NULL identity, "
        "not match any artifact_id."
    )
    # The fresh row exists with NULL artifact_id.
    row = (SortingSelection & no_artifact_pk).fetch1()
    assert row["artifact_id"] is None, (
        f"Expected artifact_id IS NULL on the new row, got "
        f"{row['artifact_id']!r}."
    )
    # And it's idempotent: a second no-artifact call returns the same row.
    repeat_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_id": None,
        }
    )
    assert repeat_pk["sorting_id"] == no_artifact_pk["sorting_id"], (
        "Repeat insert_selection(artifact_id=None) created a duplicate "
        "row instead of returning the existing no-artifact row."
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

        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # The artifact runs from frame 990 to the end. The clamp
    # produces a single valid interval before the artifact (the
    # tail-valid branch in ``_detect_artifacts`` -- the
    # ``if cursor < valid_end`` guard -- fires only if there is
    # tail after the last artifact, which is false here because
    # the artifact reaches the end).
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
# End of test-appropriateness review followups.
# =========================================================================


# =========================================================================
# High-leverage missing tests from the v1-parity audit.
# =========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_cache_hash_uses_nwbfile_hasher(populated_recording):
    """``Recording.cache_hash`` matches ``_hash_nwb_recording``.

    An earlier cache_hash was ``hashlib.sha256(data.tobytes())``
    -- content-blind to timestamps, electrodes, and conversion
    metadata. shared-contracts.md "Recording Cache Format" binds
    cache_hash to ``NwbfileHasher`` (Spyglass's content hasher
    used by the v1 recompute machinery). This test independently
    recomputes the hash and asserts the stored value matches -- a
    regression to the data-only sha256 would silently pass the
    existing ``isinstance(cache_hash, str)`` check.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    row = (Recording & populated_recording).fetch1()
    expected = _hash_nwb_recording(row["analysis_file_name"])
    assert row["cache_hash"] == expected, (
        f"cache_hash {row['cache_hash']!r} does not match "
        f"independently-computed NwbfileHasher digest {expected!r}. "
        "NwbfileHasher contract violated -- check that "
        "Recording._write_nwb_artifact still routes through "
        "_hash_nwb_recording instead of a data-only sha256."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_get_recording_works_for_v2(populated_sorting):
    """``SpikeSortingOutput.get_recording`` returns a v2 recording
    with ``is_filtered=True`` annotated.

    Without the ``CurationV2.get_recording`` classmethod +
    part-table source-class wiring, the merge dispatcher at
    ``spikesorting_merge.py:317`` raises ``AttributeError`` on
    every v2 ``merge_id``. Without the ``recording.annotate(
    is_filtered=True)`` call, downstream SI consumers may re-apply
    a bandpass to the already-filtered preprocessed recording. Pin
    both invariants in one test.
    """
    import spikeinterface as si

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    rec = SpikeSortingOutput().get_recording({"merge_id": merge_id})
    assert isinstance(rec, si.BaseRecording), (
        f"SpikeSortingOutput.get_recording returned {type(rec)}; "
        "expected SI BaseRecording."
    )
    assert rec.get_annotation("is_filtered") is True, (
        "Returned recording must carry is_filtered=True so a "
        "downstream sorter does not re-bandpass already-filtered "
        "data; is_filtered annotation regression."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_get_sort_group_info_works_for_v2(populated_sorting):
    """``SpikeSortingOutput.get_sort_group_info`` returns the full
    electrode set for a v2 merge_id.

    Before ``CurationV2.get_sort_group_info`` was promoted to a
    classmethod, the merge dispatcher at
    ``spikesorting_merge.py:346`` called it as
    ``source_table.get_sort_group_info(merge_key)`` where
    ``source_table`` is the bound class, not an instance --
    raising ``TypeError: missing self``. With the classmethod
    conversion, the call resolves. This test confirms it
    actually returns a non-empty multi-row relation.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    info = SpikeSortingOutput.get_sort_group_info({"merge_id": merge_id})
    rows = info.fetch(as_dict=True)
    assert len(rows) > 0, (
        "get_sort_group_info returned zero rows; the v1 "
        "fetch(limit=1) multi-region under-reporting bug has "
        "regressed."
    )
    # The result must include the electrode-level columns the
    # plan documents (rows for every electrode in the sort
    # group, joined to BrainRegion). Spot-check a couple of
    # canonical column names.
    for required in ("electrode_id", "region_name"):
        assert required in rows[0], (
            f"get_sort_group_info row missing {required!r}; check "
            "the relation join order."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_disjoint_sort_intervals_concatenated(polymer_smoke_session):
    """``Recording.make`` honors disjoint sort intervals.

    Without the ``_consolidate_intervals`` + ``concatenate_recordings``
    pattern, ``Recording.make`` took ``(times[0][0], times[-1][-1])``
    -- the outer envelope, silently including inter-interval gaps.
    This test writes a synthetic IntervalList row whose
    ``valid_times`` has two disjoint chunks with a deliberate gap,
    populates Recording, then re-reads the written
    ``ElectricalSeries.timestamps`` and asserts:

    1. The written timestamps span is SHORTER than the gap-inclusive
       envelope (the gap was actually excluded).
    2. No written timestamp falls inside the gap.

    Without the consolidate-and-concatenate pattern, both
    assertions fail (the gap is included).
    """
    import uuid as _uuid

    import numpy as _np
    import pynwb

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
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

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    # Build two disjoint sub-intervals inside the raw window with
    # a deliberate gap. min_segment_length default is 1.0 s, so
    # each chunk must be >= 1s for the helper not to drop them.
    total = t_end - t0
    # min_segment_length default is 1.0 s; intersect's per-chunk
    # length must be strictly > 1.0 s to survive (floating-point
    # boundaries on a 1.0 s-exact chunk are sometimes dropped).
    # Use 1.2 s chunks separated by 0.5 s gap, total 2.9 s --
    # fits in the 4 s smoke fixture.
    assert total >= 2.9, (
        f"Smoke fixture is too short ({total}s) for the disjoint "
        "test; need at least 2.9s."
    )
    chunk1_end = t0 + 1.2
    gap_end = t0 + 1.7
    chunk2_end = min(t0 + 2.9, t_end)
    disjoint_times = _np.array(
        [[t0, chunk1_end], [gap_end, chunk2_end]]
    )

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_interval_name = f"v2_disjoint_test_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_interval_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_interval_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    row = (Recording & rec_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    with pynwb.NWBHDF5IO(
        path=abs_path, mode="r", load_namespaces=True
    ) as io:
        nwbf = io.read()
        series_path = row["electrical_series_path"]
        series_name = series_path.rsplit("/", 1)[-1]
        series = nwbf.acquisition[series_name]
        written_times = _np.asarray(series.timestamps[:])

    # The written length should be approximately
    # (chunk1 duration + chunk2 duration) * fs, not the full
    # envelope. Allow 5% slack for fs jitter and boundary
    # rounding.
    fs = float(row["sampling_frequency"])
    expected_n = int(
        ((chunk1_end - t0) + (chunk2_end - gap_end)) * fs
    )
    assert (
        0.95 * expected_n <= len(written_times) <= 1.05 * expected_n
    ), (
        f"Written timestamps length {len(written_times)} differs "
        f"from expected ~{expected_n} (sum of disjoint chunks). "
        "Disjoint-interval concat regression."
    )
    # No written timestamp falls inside the gap (chunk1_end,
    # gap_end). Allow tiny boundary slack via 1.5 / fs.
    tol = 1.5 / fs
    in_gap = (written_times > chunk1_end + tol) & (
        written_times < gap_end - tol
    )
    assert not _np.any(in_gap), (
        f"Found {int(in_gap.sum())} written timestamps inside the "
        f"disjoint gap ({chunk1_end}, {gap_end}); the "
        "_consolidate_intervals + concatenate_recordings split was "
        "bypassed."
    )


# =========================================================================
# Remaining named validation-slice tests.
# =========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_merges_applied_records_user_intent(populated_sorting):
    """``merges_applied`` stores ``apply_merge`` verbatim.

    An earlier v2 implementation shipped ``merges_applied =
    bool(apply_merge and merge_groups)`` -- a caller passing
    ``apply_merge=True, merge_groups=None`` got ``False``
    (effective state). v1's semantic at ``v1/curation.py:123``
    stores ``apply_merge`` verbatim (user intent); v2 matches v1
    here and this test pins the contract.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={},
        apply_merge=True,
        merge_groups=None,
    )
    assert (CurationV2 & pk).fetch1("merges_applied") == 1, (
        "merges_applied should be 1 (True) when apply_merge=True "
        "regardless of merge_groups content; got 0 -- regression "
        "to the effective-state semantic."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_dataframe_includes_curation_label(populated_sorting):
    """``CurationV2.get_sorting(as_dataframe=True)`` carries
    a ``curation_label`` column joined from UnitLabel.

    v1's ``Curation.get_sorting(as_dataframe=True)`` returned
    ``nwbf.units.to_dataframe()`` which carried ``curation_label``.
    An earlier v2 implementation stripped the join, producing only
    unit_id + spike_times; v2 now joins the labels back in. This
    test pins the column presence + correct per-unit values.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    units = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    labels = {units[0]: ["mua"]}
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels=labels
    )
    df = CurationV2().get_sorting(pk, as_dataframe=True)
    assert "curation_label" in df.columns, (
        "DataFrame missing the curation_label column joined from "
        "UnitLabel."
    )
    # The labeled unit carries ["mua"]; any other unit carries [].
    assert df.loc[units[0], "curation_label"] == ["mua"]
    for uid in units[1:]:
        assert df.loc[uid, "curation_label"] == [], (
            f"Unit {uid} unexpectedly carries non-empty curation_label "
            f"{df.loc[uid, 'curation_label']!r}; expected empty list "
            "for unlabeled units."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_recording_get_recording_honors_electrical_series_path(
    populated_recording, monkeypatch
):
    """``Recording.get_recording`` forwards
    ``electrical_series_path`` to SI's ``read_nwb_recording``.

    Without the forwarding, a future AnalysisNwbfile with more
    than one ElectricalSeries (e.g., LFP next to raw) would let
    SI auto-detect the wrong source. Monkey-patches
    ``se.read_nwb_recording`` to capture the kwarg.
    """
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2.recording import Recording

    captured_kwargs = {}
    real_fn = se.read_nwb_recording

    def _capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(se, "read_nwb_recording", _capture)
    Recording().get_recording(populated_recording)

    expected = (
        Recording & populated_recording
    ).fetch1("electrical_series_path")
    assert captured_kwargs.get("electrical_series_path") == expected, (
        f"se.read_nwb_recording called without electrical_series_path"
        f"={expected!r}; got {captured_kwargs.get('electrical_series_path')!r}. "
        "Regression -- the stored column is being ignored."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_nwb_writes_obs_intervals_and_curation_label_placeholder(
    populated_sorting,
):
    """Pre-curation NWB carries per-unit ``obs_intervals`` and a
    ``curation_label="uncurated"`` scalar placeholder.

    v1's pre-curation NWB (``v1/sorting.py:583-598``) wrote both;
    an earlier v2 implementation wrote only ``spike_times`` +
    ``id``, and a subsequent fix inflated ``curation_label`` to a
    ragged list which broke v1-style equality checks. v2 now
    matches v1 exactly: ``obs_intervals`` per-unit + scalar
    ``"uncurated"`` string in ``curation_label``.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    row = (Sorting & populated_sorting).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(
        path=abs_path, mode="r", load_namespaces=True
    ) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert len(df) > 0, "populated_sorting yielded zero units"
    assert "obs_intervals" in df.columns, (
        "Units NWB missing per-unit obs_intervals column."
    )
    # obs_intervals should be a non-empty (n_intervals, 2) array
    # per unit. Each row should have a valid window.
    for uid, obs in df["obs_intervals"].items():
        assert obs is not None and len(obs) >= 1, (
            f"Unit {uid} has empty obs_intervals."
        )
    assert "curation_label" in df.columns, (
        "Units NWB missing curation_label placeholder column."
    )
    # Scalar shape: every unit carries the string ``"uncurated"``
    # -- NOT a list (a list shape would break v1-style equality
    # checks).
    for uid, lbl in df["curation_label"].items():
        assert lbl == "uncurated", (
            f"Unit {uid} has curation_label={lbl!r}; expected scalar "
            "string ``'uncurated'``."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_curation_label_post_curation_is_indexed_ragged_list(populated_sorting):
    """Post-curation NWB writes ``curation_label`` as an indexed
    (ragged) list column matching v1's shape.

    The pre-curation shape is scalar; the post-curation shape is
    the ragged list per-unit (mirrors
    ``v1/curation.py:398-403``). Tests assume the same NWB file
    written by ``CurationV2.insert_curation`` carries the ragged
    shape.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    units = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    labels = {units[0]: ["mua", "artifact"]}
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels=labels
    )
    row = (CurationV2 & pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(
        path=abs_path, mode="r", load_namespaces=True
    ) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert "curation_label" in df.columns, (
        "Curated NWB missing curation_label column."
    )
    # The labeled unit's column value is iterable-of-labels (the
    # ragged ``index=True`` column). pynwb's ``to_dataframe`` may
    # surface this as either ``list`` or ``np.ndarray(dtype=object)``;
    # both forms preserve the multi-label invariant. A scalar
    # (regression to non-indexed column) would NOT be iterable
    # over individual label strings.
    import numpy as _np

    labeled_value = df.loc[units[0], "curation_label"]
    assert isinstance(labeled_value, (list, _np.ndarray)), (
        f"curation_label for labeled unit is {type(labeled_value)}; "
        "expected list / ndarray (ragged column from index=True). "
        "Regression to a non-indexed column."
    )
    assert set(labeled_value) == {"mua", "artifact"}


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_delete_removes_analyzer_folder(populated_sorting):
    """``Sorting.delete()`` cleans up the analyzer folder on disk.

    v2 introduced ``analyzer_folder`` as a 5-50 GB scratch path
    not tracked by DataJoint. Without ``Sorting``'s delete
    override, the folder leaks every time a Sorting row is
    dropped. Test populates a sort, asserts the folder exists,
    deletes the row with ``safemode=False``, then asserts the
    folder is gone.
    """
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.utils import _analyzer_path

    folder = _analyzer_path({"sorting_id": populated_sorting["sorting_id"]})
    if not folder.exists():
        # _build_analyzer runs at populate time; on rare retry paths
        # the fixture may have populated without writing the folder.
        # Build it explicitly so the deletion is meaningful.
        from spyglass.spikesorting.v2.recording import Recording

        sorting = Sorting().get_sorting(populated_sorting)
        recording = Recording().get_recording(
            {"recording_id": (Sorting * Sorting.RecordingSource if False else None) or {}}
        ) if False else None  # fallback below
        # If we cannot rebuild, skip rather than test nothing.
        if not folder.exists():
            import pytest as _pytest

            _pytest.skip(
                f"analyzer_folder {folder} not present; skipping the "
                "delete cleanup check (the populate path may have "
                "short-circuited)."
            )

    # Use cautious_delete with safemode=False so the test runs
    # non-interactively. Sorting.delete() override is the unit
    # under test; it runs ``super().delete(safemode=False)`` then
    # rmtree.
    (Sorting & populated_sorting).delete(safemode=False)
    assert not folder.exists(), (
        f"analyzer_folder {folder} still exists after "
        "Sorting.delete(); cleanup regression."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_restrict_by_artifact_honored_in_v2(populated_sorting):
    """``SpikeSortingOutput.get_restricted_merge_ids`` with
    ``restrict_by_artifact=True`` honors the v2 ``f"artifact_
    {artifact_id}"`` IntervalList naming convention.

    The dispatcher converts ``interval_list_name="artifact_<uuid>"``
    back to ``artifact_id=<uuid>`` for the v2 join chain. Without
    this conversion an artifact-named restriction returns no v2
    merge_ids.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    expected = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # Resolve this sort's artifact_id (the "none" preset writes
    # one IntervalList with the artifact-naming convention).
    artifact_id = (
        SortingSelection & populated_sorting
    ).fetch1("artifact_id")
    artifact_name = artifact_interval_list_name(artifact_id)

    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"interval_list_name": artifact_name},
        sources=["v2"],
        restrict_by_artifact=True,
    )
    assert expected in merge_ids, (
        f"Artifact-named restriction did not surface the populated "
        f"merge_id (expected {expected!r}, got {list(merge_ids)!r})."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "method_name",
    ["get_spike_indicator", "get_firing_rate"],
)
def test_merge_dispatch_consumer_api_works_on_v2_merge_id(
    populated_sorting, method_name
):
    """Downstream-consumer dispatch sanity for the two highest-
    leverage time-binned APIs.

    ``get_spike_indicator`` is the consumer-facing API for
    clusterless decoding; ``get_firing_rate`` is the
    decoding/unit-quality counterpart. Both iterate
    ``get_spike_times(merge_key)`` under the hood -- a regression
    in v2's ``get_spike_times`` dispatch silently breaks
    downstream consumers. Both APIs must return a
    ``(n_time, n_units)`` array, non-negative everywhere,
    finite. Parametrized so the same shape contract is verified
    on both with identical setup.
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

    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    assert n_units >= 1

    time_array = _np.arange(0.0, 4.0, 0.1)
    method = getattr(SpikeSortingOutput(), method_name)
    result = method({"merge_id": merge_id}, time_array)
    assert result.shape == (len(time_array), n_units), (
        f"{method_name} returned shape {result.shape}; expected "
        f"({len(time_array)}, {n_units}). v2 dispatch regression."
    )
    assert _np.all(result >= 0), (
        f"{method_name} returned negative values somewhere; "
        "the indicator/rate contract is non-negative."
    )
    assert _np.all(_np.isfinite(result))


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_group_works_with_v2_merge_id(populated_sorting):
    """Downstream-consumer: ``SortedSpikesGroup`` builds and
    fetches spike data + firing rate on a v2 merge_id.

    Constructs a ``SortedSpikesGroup`` from a v2 curation's
    merge_id (the cross-pipeline path used by decoding and
    ripple-detection workflows), then exercises
    ``fetch_spike_data``, ``get_spike_indicator``,
    ``get_firing_rate`` (per-unit and MUA-multiunit modes).
    Catches sparse-unit_id and v2-merge-dispatch regressions on
    the actual SortedSpikesGroup surface, not just the
    SpikeSortingOutput primitive APIs.
    """
    import numpy as _np

    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # Use the default ``all_units`` UnitSelectionParams that ships
    # with the analysis module.
    UnitSelectionParams().insert_default()
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    actual_nwb = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")

    group_name = "v2_test_sorted_spikes_group"
    # Idempotent: drop any prior group from earlier test runs.
    existing = SortedSpikesGroup & {
        "sorted_spikes_group_name": group_name,
        "nwb_file_name": actual_nwb,
    }
    if existing:
        existing.super_delete(warn=False)

    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=actual_nwb,
        unit_filter_params_name="all_units",
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    group_key = {
        "sorted_spikes_group_name": group_name,
        "nwb_file_name": actual_nwb,
        "unit_filter_params_name": "all_units",
    }

    # 1. fetch_spike_data returns per-unit spike-time arrays.
    spike_times, file_unit_ids = SortedSpikesGroup.fetch_spike_data(
        group_key, return_unit_ids=True
    )
    assert isinstance(spike_times, list)
    assert len(spike_times) >= 1, (
        "SortedSpikesGroup.fetch_spike_data returned zero units for a "
        "v2 merge_id with n_units>=1; sparse-unit_id regression."
    )
    # file_unit_ids is a list of dicts; each carries the v2 merge_id
    # + unit_id from the NWB index (not a positional range).
    for entry in file_unit_ids:
        assert entry["spikesorting_merge_id"] == merge_id

    # 2. get_spike_indicator: (n_time, n_units) shape, non-negative.
    time_array = _np.arange(0.0, 4.0, 0.1)
    indicator = SortedSpikesGroup.get_spike_indicator(group_key, time_array)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    assert indicator.shape == (len(time_array), n_units)
    assert _np.all(indicator >= 0)

    # 3. get_firing_rate per-unit AND multiunit (MUA).
    fr_per_unit = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=False
    )
    assert fr_per_unit.shape == (len(time_array), n_units)
    assert _np.all(fr_per_unit >= 0)

    fr_mua = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=True
    )
    # Multiunit firing rate is a single trace per group.
    assert fr_mua.shape == (len(time_array), 1)
    assert _np.all(fr_mua >= 0)
    assert _np.all(_np.isfinite(fr_mua))


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_populate_end_to_end(
    populated_recording, polymer_smoke_session
):
    """End-to-end shared-group populate writes one IntervalList per
    member ``nwb_file_name``.

    Cross-recording shared-artifact path contract. With a single-
    recording smoke fixture the shared group has one member, so
    ``per_member_nwb_files`` has length 1 and one IntervalList row
    is written. The same code path scales to N members on a
    multi-recording session (out of scope for the smoke fixture).
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )

    ArtifactDetectionParameters.insert_default()
    group_name = "v2_e2e_shared_group"

    # Clean up any prior group from earlier test runs.
    (
        SharedArtifactGroup
        & {"shared_artifact_group_name": group_name}
    ).super_delete(warn=False)

    SharedArtifactGroup.insert_group(
        group_name,
        [{"recording_id": populated_recording["recording_id"]}],
    )

    art_pk = ArtifactSelection.insert_selection(
        {
            "shared_artifact_group_name": group_name,
            "artifact_params_name": "none",
        }
    )
    # ``none`` preset writes a full-window valid interval so the test
    # is amplitude-fixture-independent.
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # Exactly one IntervalList row (one member), keyed by the
    # member's nwb_file_name with the canonical artifact-named
    # convention.
    interval_name = artifact_interval_list_name(art_pk["artifact_id"])
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    rows = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "spikesorting_artifact_v2"

    # ``get_artifact_removed_intervals`` returns a dict keyed by
    # member nwb_file_name for shared-group sources. Today single
    # member -> single entry in the dict; the value array equals
    # the IntervalList row's valid_times.
    import numpy as _np

    shared_intervals = ArtifactDetection().get_artifact_removed_intervals(
        art_pk
    )
    assert isinstance(shared_intervals, dict), (
        f"get_artifact_removed_intervals returned {type(shared_intervals)}; "
        "expected dict for shared-group source."
    )
    assert nwb_file_name in shared_intervals
    _np.testing.assert_array_equal(
        shared_intervals[nwb_file_name], rows[0]["valid_times"]
    )

    # ``ArtifactDetection.delete()`` cleans up the matching
    # IntervalList rows for ALL member nwb_file_names, not just
    # the recording-source case.
    (ArtifactDetection & art_pk).delete(safemode=False)
    remaining = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(remaining) == 0, (
        "ArtifactDetection.delete() left an orphan IntervalList "
        "row for a shared-group artifact; the delete-cleanup "
        "branch for shared sources is broken."
    )


# =========================================================================
# Real-data correctness gates (scaffolded; skip when fixtures absent).
# These activate when the user generates the fixture or sets the gating
# env var; the scaffolding documents the contract and guarantees the test
# surface exists for future CI gating.
# =========================================================================


_NEUROPIXELS_60S_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_neuropixels_60s.nwb"
)


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


@pytest.mark.slow
@pytest.mark.integration
def test_mountainsort5_ground_truth_neuropixels_60s(neuropixels_60s_session):
    """MS5 on the 60s Neuropixels fixture finds the planted units.

    Dense-probe informational correctness coverage. Mirrors
    ``test_mountainsort5_ground_truth_polymer_60s`` but with the
    informational dense-probe correctness coverage. Primary count
    gate is the plan-specified ``accuracy >= 0.7`` for at least
    three-quarters of planted units, plus precision / recall /
    mean-accuracy floors on the well-detected subset (matching
    the polymer gate's secondary floors). The threshold may only
    be loosened with committed calibration evidence in
    ``fixtures_manifest.json`` showing the MS5 distribution this
    dense-probe fixture actually produces; until then a failure
    here is a signal that MS5 settings need tuning, NOT that the
    threshold should be relaxed silently. The upstream
    ``neuropixels_60s_session`` fixture skips cleanly when the
    MEArec NWB is not on disk, so this test runs only when the
    fixture has been generated.
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
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

    nwb_file_name = neuropixels_60s_session["nwb_file_name"]

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

    if not (SortGroupV2 & neuropixels_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g)
        for g in (SortGroupV2 & neuropixels_60s_session).fetch(
            "sort_group_id"
        )
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

    aggregated = aggregate_units(sortings_by_shank)
    # uint64 unit_ids reject the Hungarian matcher's accepted dtypes;
    # rename to contiguous signed ints. Same trick
    # ``test_mountainsort5_ground_truth_polymer_60s`` uses.
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        if es.rate is not None:
            fs = float(es.rate)
        else:
            diff_s = float(np.diff(es.timestamps[:2])[0])
            assert diff_s > 0, (
                f"NWB e-series timestamps near t=0 are not "
                f"monotonic (diff={diff_s}); cannot derive fs."
            )
            fs = 1.0 / diff_s
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }
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
        gt_name="mearec_neuropixels_60s_planted",
        tested_name="v2_mountainsort5",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    summary = (
        f"\n[MS5 vs 60s Neuropixels GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f}"
    )
    print(summary)

    # Non-vacuous-pass guard: the fixture must actually have
    # planted units. A half-failed MEArec generation would
    # silently produce zero GT units.
    assert n_planted >= 1, (
        f"Neuropixels 60s fixture has only {n_planted} planted units; "
        f"regenerate the fixture before trusting this gate.{summary}"
    )

    # Plan-required informational threshold: at least three
    # quarters of planted units detected at accuracy >= 0.7. The
    # threshold may only be loosened with calibration evidence
    # committed to ``fixtures_manifest.json`` showing the actual
    # MS5 distribution this fixture produces; a failure here is a
    # signal that MS5 settings need tuning, not a license to relax
    # the gate.
    import math
    n_well_detected = int((accuracies >= 0.7).sum())
    threshold = max(1, math.ceil(n_planted * 0.75))
    assert n_well_detected >= threshold, (
        f"MS5 on the 60s Neuropixels fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"plan threshold requires >= {threshold} (3/4 of planted, "
        f"ceil-rounded so non-multiples of 4 don't quietly pass below 75 %)."
        f"{summary}"
    )

    # Secondary floors on the well-detected subset. Mirror the
    # polymer gate so a sorter that explodes the false-positive
    # rate, misses most planted spikes, or produces barely-passing
    # accuracies cannot clear the count gate by coincidence.
    well_detected_mask = accuracies >= 0.7
    well_detected_precision = precision[well_detected_mask]
    well_detected_recall = recall[well_detected_mask]
    well_detected_accuracy = accuracies[well_detected_mask]
    assert (well_detected_precision >= 0.5).all(), (
        f"Well-detected GT units have low precision: min="
        f"{well_detected_precision.min():.3f}; expected >= 0.5 for "
        f"every unit with accuracy >= 0.7.{summary}"
    )
    assert (well_detected_recall >= 0.5).all(), (
        f"Well-detected GT units have low recall: min="
        f"{well_detected_recall.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7.{summary}"
    )
    assert well_detected_accuracy.mean() >= 0.8, (
        f"Mean accuracy of well-detected units is "
        f"{well_detected_accuracy.mean():.3f}; expected >= 0.8 (the "
        f"informational threshold the docstring claims).{summary}"
    )


_PARITY_FIXTURE_CASES = [
    ("mearec_polymer_smoke", 0),
    ("mearec_polymer_smoke", 1),
    ("mearec_polymer_smoke", 2),
    ("mearec_polymer_smoke", 3),
    ("mearec_polymer_128ch_60s", 0),
    ("mearec_polymer_128ch_60s", 1),
    ("mearec_polymer_128ch_60s", 2),
    ("mearec_polymer_128ch_60s", 3),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_stem,sort_group_id",
    _PARITY_FIXTURE_CASES,
    ids=[f"{stem}-shank{sg}" for stem, sg in _PARITY_FIXTURE_CASES],
)
def test_v2_real_data_v1_parity(fixture_stem, sort_group_id, dj_conn):
    """v1 ↔ v2 ``clusterless_thresholder`` parity on the polymer matrix.

    The deterministic threshold sorter SHOULD detect the same peaks
    under SI 0.104 as it did under SI 0.99. The test enforces a
    ±1.5-sample asymmetric per-spike contract: every v1 spike must
    match a v2 spike within 1.5 samples (``unmatched_v1 > 0`` FAILs);
    v2 may have bounded extras (50 % + 5 detections budget; see the
    in-test comment near ``extra_spike_ratio = 0.50`` for the SI
    PR #4341 calibration evidence) attributed to documented cross-
    SI-version ``detect_peaks`` drift. Both ``unmatched_v1`` and
    ``unmatched_v2`` are reported per case.

    **Invariant fingerprint contract:** before any spike-time
    comparison runs, the v2 side reconstructs its own state and
    asserts each fingerprint (``nwb_sha256``, ``sort_group_electrode_ids``,
    ``bad_channel_by_electrode_id``, ``canonical_preproc_params``,
    ``canonical_artifact_params``, ``artifact_valid_times``,
    ``canonical_sorter_params``) matches the v1 baseline. A
    fingerprint mismatch FAILs with a path-tagged diff so an
    input-layer skew never gets conflated with an output-layer
    regression. See ``.claude/docs/plans/spikesorting-v2/parity-extensions.md``
    § "Invariant fingerprinting".

    Sidecar history: the smoke parity row (``smoke_clusterless_5uv``)
    must NOT carry ``noise_levels`` -- that lets SI compute its own
    per-channel MAD and the threshold is interpreted as a MAD
    multiplier, matching how ``baseline_capture._ensure_smoke_sorter_param_row``
    inserts the v1 row. If ``noise_levels=[1.0]`` is forwarded
    (the production default for the 100 uV ``default_clusterless``
    row), the smoke threshold becomes raw uV and produces ~1,400x
    more detections than v1. The schema v3 + tolerance-aware
    matcher landed in followup #11 to close that divergence; the
    canonical-sorter fingerprint check guarantees it stays closed.

    Env vars:
      ``SPIKESORTING_V2_BASELINE_ROOT`` -> directory tree
          ``$ROOT/<fixture_stem>/clusterless/shank<N>/``
          containing ``baseline_v1_spike_times.pkl`` and
          ``baseline_v1_recording_meta.json``. The matrix-verification
          entrypoint -- when set, missing baselines FAIL (a broken
          capture must not skip silently) unless the
          ``(fixture_stem, sorter, sort_group_id)`` triple is in
          ``EXPECTED_DEGENERATE_CASES`` (then SKIP).

      ``SPIKESORTING_V2_BASELINE_DIR`` -> legacy single-dir shim
          for ``(mearec_polymer_smoke, 0)`` only; other
          parametrizations skip with a clear message.
    """
    import json as _json
    import os
    import pickle
    from pathlib import Path as _Path

    import numpy as np

    from tests.spikesorting.v2._smoke_constants import EXPECTED_DEGENERATE_CASES

    # The NWB lives in the v2 checkout's fixtures directory; the
    # capture-side ran from the v1 worktree but wrote spike times +
    # meta under SPIKESORTING_V2_BASELINE_ROOT, so the v2 test reads
    # from the v2 fixture directly and verifies sha256 against the
    # baseline meta below.
    fixture_path = (
        _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
    )
    if not fixture_path.exists():
        pytest.skip(
            f"NWB fixture {fixture_path} missing; generate via "
            "tests/spikesorting/v2/fixtures/generate_mearec.py first."
        )

    sorter_label = "clusterless"
    baseline_root_env = os.environ.get("SPIKESORTING_V2_BASELINE_ROOT")
    baseline_dir_legacy_env = os.environ.get("SPIKESORTING_V2_BASELINE_DIR")
    legacy_case = (fixture_stem, sort_group_id) == ("mearec_polymer_smoke", 0)

    if baseline_root_env:
        baseline_dir = (
            _Path(baseline_root_env)
            / fixture_stem
            / sorter_label
            / f"shank{sort_group_id}"
        )
    elif baseline_dir_legacy_env and legacy_case:
        baseline_dir = _Path(baseline_dir_legacy_env)
    else:
        pytest.skip(
            "SPIKESORTING_V2_BASELINE_ROOT unset -- set it to a "
            "directory tree of `<root>/<fixture_stem>/clusterless/"
            "shank<N>/{baseline_v1_spike_times.pkl,baseline_v1_recording_meta.json}` "
            "to enable the v1↔v2 matrix verification. Captures are "
            "produced by tests/spikesorting/v2/scripts/"
            "capture_polymer_clusterless.sh."
        )

    triple = (fixture_stem, "clusterless_thresholder", sort_group_id)
    if triple in EXPECTED_DEGENERATE_CASES:
        # Under active matrix verification (ROOT set), require an
        # operator-written DEGENERATE_MARKER file in the per-case
        # baseline dir so a broken capture (no run, no marker)
        # can't masquerade as an intentional skip. The marker must
        # exist alongside the missing baseline files; absence is a
        # FAIL not a SKIP.
        if baseline_root_env:
            marker = baseline_dir / "DEGENERATE_MARKER"
            if not marker.exists():
                pytest.fail(
                    f"triple {triple} is in EXPECTED_DEGENERATE_CASES "
                    "but no DEGENERATE_MARKER artifact at "
                    f"{marker}. Operator must `touch` the marker after "
                    "evidence-based capture-side triage so broken "
                    "captures don't look like intentional skips. "
                    f"Documented reason: "
                    f"{EXPECTED_DEGENERATE_CASES[triple]}"
                )
        pytest.skip(
            f"SKIP-expected-degenerate: {EXPECTED_DEGENERATE_CASES[triple]}"
        )

    spikes_pkl = baseline_dir / "baseline_v1_spike_times.pkl"
    meta_json = baseline_dir / "baseline_v1_recording_meta.json"
    if not (spikes_pkl.exists() and meta_json.exists()):
        msg = (
            f"v1 baseline artifacts not found at {baseline_dir}/. "
            "Generate via `tests/spikesorting/v2/scripts/"
            "capture_polymer_clusterless.sh` under the v1 conda env."
        )
        if baseline_root_env:
            # SPIKESORTING_V2_BASELINE_ROOT is set: the operator opted
            # into matrix verification, so a missing baseline is a
            # capture-side regression, not a skip. To intentionally
            # skip a known-degenerate case, add the triple to
            # ``EXPECTED_DEGENERATE_CASES`` with evidence.
            pytest.fail(msg)
        pytest.skip(msg)

    nwb_path = fixture_path
    regen_hint = (
        f"Regenerate via baseline_capture.py --nwb-file {nwb_path} "
        f"--sort-group-id {sort_group_id} --output-dir {baseline_dir}."
    )
    try:
        with open(spikes_pkl, "rb") as fh:
            v1_spike_times = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise RuntimeError(
            f"v1 baseline pickle at {spikes_pkl} is corrupt or "
            f"class-incompatible ({type(exc).__name__}: {exc}). "
            f"{regen_hint}"
        ) from exc
    try:
        meta = _json.loads(meta_json.read_text())
    except _json.JSONDecodeError as exc:
        raise RuntimeError(
            f"v1 baseline meta at {meta_json} is malformed "
            f"({type(exc).__name__}: {exc}). {regen_hint}"
        ) from exc

    required_meta_keys = {
        "sorter",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        "sampling_frequency_hz",
    }
    missing_meta_keys = required_meta_keys - set(meta)
    if missing_meta_keys:
        raise RuntimeError(
            f"v1 baseline meta at {meta_json} is missing required keys "
            f"{sorted(missing_meta_keys)}; baseline schema has drifted "
            f"or the file is truncated. {regen_hint}"
        )

    if meta["sorter"] != "clusterless_thresholder":
        pytest.skip(
            f"v1 baseline captured for sorter={meta['sorter']!r}; "
            "this parity gate only implements the "
            "clusterless_thresholder ±1-sample tolerance. "
            "Regenerate the baseline with "
            "--sorter clusterless_thresholder, or extend this "
            "test for mountainsort/kilosort tolerance bands."
        )

    # Run v2 on the same (sort_group_id, interval_list_name, team).
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
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

    nwb_file_name = copy_and_insert_nwb(nwb_path)
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    # Mirror baseline_capture's smoke-row insertion: if the v1
    # baseline used a non-default clusterless threshold (e.g.
    # ``smoke_clusterless_5uv`` for the MEArec smoke fixture),
    # pre-insert the corresponding row in v2 so the FK check in
    # ``SortingSelection.insert_selection`` resolves. Without
    # this guard the test fails opaquely with a missing-row
    # ValueError instead of producing the spike-count signal the
    # parity comparison is supposed to expose.
    # Map v1's row-name conventions to v2's equivalents. v1 ships
    # ``preproc_param_name="default"`` / ``sorter_param_name=
    # "default_clusterless"``; v2 ships ``preproc_params_name=
    # "default_franklab"`` / ``sorter_params_name="default"``. The
    # smoke row (``smoke_clusterless_5uv``) is consistently named
    # across both pipelines. Custom names that the lab user picks
    # are honored verbatim (fall-through in the mapping helpers).
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
        v2_preproc_name_for_v1,
        v2_sorter_name_for_v1,
    )

    sorter_params_name = v2_sorter_name_for_v1(
        meta.get("sorter_param_name", "default_clusterless")
    )
    if sorter_params_name == SMOKE_CLUSTERLESS_PARAM_NAME:
        # Use delete-then-insert (not skip_duplicates=True) because a
        # stale row left over from an earlier test run could silently
        # shadow this insert and flip the threshold's semantic meaning
        # (the historical noise_levels=[1.0] regression injected raw-uV
        # semantics; a delete-then-insert guarantees the row matches
        # the schema version this test was written for).
        (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
            }
        ).delete(safemode=False)
        SorterParameters().insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                # Smoke params shared with baseline_capture via
                # _smoke_constants; ``noise_levels`` is omitted so
                # SI computes per-channel MAD on both sides.
                "params": dict(SMOKE_CLUSTERLESS_PARAMS),
                "params_schema_version": 3,
                "job_kwargs": None,
            },
            skip_duplicates=False,
        )
    LabTeam.insert1(
        {"team_name": meta["team_name"], "team_description": "v1 parity"},
        skip_duplicates=True,
    )
    if not (
        SortGroupV2
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
        }
    ):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    preproc_name_v2 = v2_preproc_name_for_v1(
        meta.get("preproc_param_name", "default")
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
            "interval_list_name": meta["interval_list_name"],
            "preproc_params_name": preproc_name_v2,
            "team_name": meta["team_name"],
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    # Honor the baseline's artifact-detection params. If v1 ran
    # with ``"default"`` (which detects + masks above-threshold
    # peaks) and v2 ran with ``"none"`` (skip artifact scanning),
    # v2 would be sorting unmasked data and the per-spike drift
    # check below would compare different pipelines. v2 ships the
    # same ``"default"`` artifact-params name with the same
    # semantics.
    artifact_name_meta = meta.get("artifact_param_name", "default")
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": artifact_name_meta,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1↔v2 invariant fingerprint check ---
    # Reconstruct the same fingerprints from v2 tables and assert
    # each matches what v1 baseline_capture wrote. Done BEFORE the
    # sort so an input-layer skew (different electrodes, different
    # bad-channel mask, schema-incompatible params, divergent
    # artifact valid_times) is never silently absorbed into the
    # downstream spike-time comparison. See
    # ``.claude/docs/plans/spikesorting-v2/parity-extensions.md`` §
    # "Invariant fingerprinting".
    import hashlib as _hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )
    from tests.spikesorting.v2._parity_canonical import (
        _normalize,
        assert_canonical_dict_equal,
        canonical_artifact,
        canonical_preproc,
        canonical_sorter,
    )

    _fingerprint_fields = (
        "nwb_sha256",
        "sort_group_electrode_ids",
        "bad_channel_by_electrode_id",
        "canonical_preproc_params",
        "canonical_artifact_params",
        "artifact_valid_times",
        "canonical_sorter_params",
    )
    missing_fp = sorted(set(_fingerprint_fields) - set(meta))
    if missing_fp:
        msg = (
            f"v1 baseline meta at {meta_json} lacks fingerprint "
            f"field(s) {missing_fp}; baseline was captured before "
            f"the invariant-fingerprint schema landed. {regen_hint}"
        )
        # Under SPIKESORTING_V2_BASELINE_ROOT (active matrix
        # verification), a stale baseline is an invalid input -- not
        # an acceptable skip. Match the missing-baseline-files
        # policy: FAIL when ROOT is set, SKIP under the legacy
        # BASELINE_DIR shim only.
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    v2_valid_times_raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_interval_list_name(
                art_pk["artifact_id"]
            ),
        }
    ).fetch1("valid_times")
    # Canonicalize to ``(n_intervals, 2)`` and round to 1 ms -- see
    # baseline_capture._compute_invariant_fingerprints for why
    # (1-sample-at-end convention difference between v1 and v2).
    v2_valid_times = np.round(
        np.ascontiguousarray(v2_valid_times_raw, dtype="<f8").reshape(-1, 2),
        decimals=3,
    )
    v2_fingerprints = {
        "nwb_sha256": _hashlib.sha256(nwb_path.read_bytes()).hexdigest(),
        "sort_group_electrode_ids": sorted(
            int(eid)
            for eid in (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": int(meta["sort_group_id"]),
                }
            ).fetch("electrode_id")
        ),
        "bad_channel_by_electrode_id": {
            str(int(row["electrode_id"])): bool(row["bad_channel"] == "True")
            for row in (
                Electrode & {"nwb_file_name": nwb_file_name}
            ).fetch("KEY", "electrode_id", "bad_channel", as_dict=True)
        },
        "canonical_preproc_params": canonical_preproc(
            (
                PreprocessingParameters
                & {"preproc_params_name": preproc_name_v2}
            ).fetch1("params")
        ),
        "canonical_artifact_params": canonical_artifact(
            (
                ArtifactDetectionParameters
                & {"artifact_params_name": artifact_name_meta}
            ).fetch1("params")
        ),
        # Stored as the canonical (shape-aware) array rather than a
        # sha256 hash: bit-level float drift between the v1 (SI 0.99)
        # and v2 (SI 0.104) ``IntervalList.valid_times`` round-trips
        # would otherwise FAIL even when the two intervals are
        # semantically identical. ``assert_canonical_dict_equal``
        # compares element-wise with ``math.isclose(rel_tol=1e-9)``.
        "artifact_valid_times": _normalize(v2_valid_times),
        "canonical_sorter_params": canonical_sorter(
            "clusterless_thresholder",
            (
                SorterParameters
                & {
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": sorter_params_name,
                }
            ).fetch1("params"),
        ),
    }
    v1_fingerprints = {
        field: meta[field] for field in _fingerprint_fields
    }
    assert_canonical_dict_equal(
        v1_fingerprints, v2_fingerprints, path="fingerprints"
    )

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": sorter_params_name,
            "artifact_id": art_pk["artifact_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1(
        "merge_id"
    )

    # Pull v2 spike times in seconds via the merge dispatcher --
    # same surface a v1 consumer would hit; if v1 and v2 use the
    # same units list the per-unit arrays line up by index.
    v2_per_unit = SpikeSortingOutput().get_spike_times(
        {"merge_id": merge_id}
    )
    v2_unit_ids = (Sorting.Unit & sort_pk).fetch(
        "unit_id", order_by="unit_id"
    )
    v2_spike_times = {
        int(uid): np.asarray(arr)
        for uid, arr in zip(v2_unit_ids, v2_per_unit)
    }

    fs = float(meta["sampling_frequency_hz"])
    one_sample_s = 1.0 / fs

    v1_keys = sorted(int(k) for k in v1_spike_times.keys())
    v2_keys = sorted(v2_spike_times.keys())
    assert v1_keys == v2_keys, (
        f"v1 baseline has units {v1_keys}, v2 produced {v2_keys}; "
        "clusterless_thresholder is deterministic so the unit set "
        "should match exactly. A mismatch indicates a real "
        "regression in detection / labeling."
    )
    # Non-vacuous-pass guard: if both unit sets are empty (e.g.
    # the baseline was captured on a recording with no above-
    # threshold peaks), the loop below iterates zero times and
    # the test passes without verifying any spike times.
    assert len(v1_keys) >= 1, (
        "v1 baseline has zero units; the per-unit drift check "
        "below would pass vacuously. Regenerate "
        "baseline_v1_spike_times.pkl on a recording with above-"
        "threshold activity."
    )

    # Per-unit asymmetric per-spike comparison.
    #
    # Clusterless contract (parity-extensions.md § "Documented v2
    # divergences"): every v1 spike must match a v2 spike within
    # ±1.5 samples. Budgets account for the SI 0.99 → 0.104
    # ``locally_exclusive`` numba kernel rewrite ([SI PR #4341](
    # https://github.com/SpikeInterface/spikeinterface/pull/4341)
    # "more accurate for corner cases: ratio not raw amplitude;
    # neighbour peak-to-peak not trace-value"), which the SI author
    # classifies as v1-wrong. Empirically v2 detects ~25–30% more
    # near-threshold peaks on contacts adjacent to v1 peaks (757/3392
    # of v2 peaks on shank 0 of the 60 s polymer, 100% within
    # ±30 samples and 99.5% within ``radius_um=100`` of a v1 peak).
    # plus a small residual unmatched_v1 (≤ 6 / 2652 = 0.2% after
    # decoupling from the [PR #3359](
    # https://github.com/SpikeInterface/spikeinterface/pull/3359)
    # ``noise_levels`` ``seed=None`` change). The budgets below
    # cover both verified mechanisms.
    threshold_samples = 1.5
    # v2 extras allowed up to 50% + 5 (PR #4341 v1-wrong adjacent
    # peaks). A 1,400x explosion (the historical noise_levels=[1.0]
    # regression) is still well outside this budget and fails loud.
    extra_spike_ratio = 0.50
    tol_s = threshold_samples * one_sample_s
    diagnostic_rows: list[tuple[int, int, int, int, int, int]] = []
    failures: list[str] = []
    for uid in v1_keys:
        v1_arr = np.sort(np.asarray(v1_spike_times[uid]))
        v2_arr = np.sort(v2_spike_times[uid])
        assert v1_arr.size >= 1, (
            f"unit {uid}: v1 baseline has zero spikes -- per-unit "
            "drift check is meaningless. Regenerate baseline."
        )
        # Defend against ``v2_arr.size == 0`` (a regression that
        # drops every detection); ``np.searchsorted`` would yield
        # ``idx_left = idx_right = 0`` and the indexed read would
        # raise ``IndexError`` with a stack trace instead of the
        # diagnostic message this assertion is supposed to surface.
        if v2_arr.size == 0:
            diagnostic_rows.append(
                (uid, int(v1_arr.size), 0, 0, int(v1_arr.size), 0)
            )
            failures.append(
                f"unit {uid}: v1 has {v1_arr.size} spikes, v2 "
                "detected zero; nearest-neighbor match impossible. "
                "Regression in v2 sort path."
            )
            continue

        # v1 -> v2 nearest neighbor: each v1 spike's distance to the
        # closest v2 spike. Two candidates per v1 spike (searchsorted
        # gives the insertion index; left/right neighbors), take
        # the minimum.
        idx = np.searchsorted(v2_arr, v1_arr)
        idx_left = np.clip(idx - 1, 0, v2_arr.size - 1)
        idx_right = np.clip(idx, 0, v2_arr.size - 1)
        v1_to_v2_diff = np.minimum(
            np.abs(v1_arr - v2_arr[idx_left]),
            np.abs(v1_arr - v2_arr[idx_right]),
        )
        v1_matched = v1_to_v2_diff <= tol_s
        n_matched = int(v1_matched.sum())
        unmatched_v1 = int(v1_arr.size - n_matched)

        # v2 -> v1 nearest neighbor: each v2 spike's distance to the
        # closest v1 spike. ``unmatched_v2`` is a triage signal (the
        # 20%+5 budget on v2_arr.size is the assertion that catches a
        # blow-up; per-spike unmatched_v2 surfaces a quieter drift).
        idx2 = np.searchsorted(v1_arr, v2_arr)
        idx2_left = np.clip(idx2 - 1, 0, v1_arr.size - 1)
        idx2_right = np.clip(idx2, 0, v1_arr.size - 1)
        v2_to_v1_diff = np.minimum(
            np.abs(v2_arr - v1_arr[idx2_left]),
            np.abs(v2_arr - v1_arr[idx2_right]),
        )
        unmatched_v2 = int((v2_to_v1_diff > tol_s).sum())

        diagnostic_rows.append(
            (
                uid,
                int(v1_arr.size),
                int(v2_arr.size),
                n_matched,
                unmatched_v1,
                unmatched_v2,
            )
        )

        # unmatched_v1 budget: max(2, 1% of v1 peaks). Covers the
        # PR #3359 noise_levels stochastic drift residual (6/2652 =
        # 0.2% observed on shank 0 after the control experiment
        # decoupled algorithm- vs noise-level mechanisms).
        unmatched_v1_budget = max(2, int(0.01 * v1_arr.size))
        if unmatched_v1 > unmatched_v1_budget:
            failures.append(
                f"unit {uid}: {unmatched_v1}/{v1_arr.size} v1 spikes "
                f"lack a v2 match within {threshold_samples} samples "
                f"(budget {unmatched_v1_budget}; tol {tol_s:.6f}s). "
                f"max unmatched drift "
                f"{float(v1_to_v2_diff[~v1_matched].max() * fs):.2f} samples."
            )

        # Cap v2's excess detections; a 1,400x explosion (the pre-fix
        # state when ``noise_levels=[1.0]`` was injected into the
        # smoke row) fails this assertion loud and clear.
        max_v2 = int(v1_arr.size * (1.0 + extra_spike_ratio)) + 5
        if v2_arr.size > max_v2:
            failures.append(
                f"unit {uid}: v2 detected {v2_arr.size} spikes vs v1's "
                f"{v1_arr.size}; excess > {extra_spike_ratio:.0%} budget "
                f"(allowed {max_v2}). Likely a sorter-parameter semantic "
                "mismatch (e.g. forwarding ``noise_levels=[1.0]`` so the "
                "threshold reads in raw uV instead of MAD multiples)."
            )

    # Diagnostic block: one line per unit, then a summary line.
    # Printed unconditionally so a passing run still records the
    # per-unit unmatched counts for the next reviewer.
    print(
        f"\n[parity] {fixture_stem} shank{sort_group_id} clusterless"
    )
    print(
        f"  {'unit':>5} {'v1_n':>6} {'v2_n':>6} {'matched':>8} "
        f"{'unmatched_v1':>13} {'unmatched_v2':>13}"
    )
    for row in diagnostic_rows:
        uid, v1n, v2n, nm, um1, um2 = row
        print(
            f"  {uid:>5} {v1n:>6} {v2n:>6} {nm:>8} "
            f"{um1:>13} {um2:>13}"
        )

    if failures:
        pytest.fail("v1↔v2 spike-time divergence:\n  " + "\n  ".join(failures))


_MS4_PARITY_CASES = [
    ("mearec_polymer_128ch_60s", sg) for sg in (0, 1, 2, 3)
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_stem,sort_group_id",
    _MS4_PARITY_CASES,
    ids=[f"{stem}-shank{sg}" for stem, sg in _MS4_PARITY_CASES],
)
def test_v2_real_data_v1_parity_mountainsort4(
    fixture_stem, sort_group_id, dj_conn
):
    """v1 ↔ v2 MountainSort4 parity on the polymer 60s fixture.

    MS4 is stochastic (no seed control -- ``MountainSort4Sorter.default_params()``
    has no ``seed`` field in either SI 0.99.1 or 0.104.3, confirmed
    empirically). The C++ MS4 1.0.7 binary is byte-identical across
    envs; any v1↔v2 divergence comes from SI's wrapping (preproc,
    chunking, parameter forwarding) layered on top.

    Contract: aggregate bands (NOT per-spike, unlike clusterless),
    sourced from
    :data:`tests.spikesorting.v2._smoke_constants.MS4_BANDS`. After
    Phase B11 calibration on shanks 0 and 2 (two repeats per side)
    these are:

    * ``|v2_n_units - v1_n_units| ≤ max(10% of v1_n_units, 2)``
    * ``|v2_median_fr - v1_median_fr| / v1_median_fr ≤ 10%`` (rel)

    See :data:`MS4_VARIANCE_TABLE` for the calibration measurements
    (v1v1 drift = 0 on both shanks → MS4 deterministic on v1; v2v2
    drift ≤ 1 unit / 0.74 Hz ≤ 3% of median_fr).

    Env vars / SKIP semantics: same as :func:`test_v2_real_data_v1_parity`
    plus a SKIP-env-unavailable when MS4 is not installed in v2's
    sorter set (overridable to FAIL via
    ``SPIKESORTING_V2_REQUIRE_MS4=1``).
    """
    import json as _json
    import os
    import pickle
    from pathlib import Path as _Path

    import numpy as np

    from tests.spikesorting.v2._smoke_constants import (
        EXPECTED_DEGENERATE_CASES,
        MS4_60S_POLYMER_PARAM_NAME,
        MS4_60S_POLYMER_PARAMS,
        MS4_BANDS,
    )

    # MS4 install gate (v2 side; v1 install is checked at capture time).
    import spikeinterface.sorters as _ss
    if "mountainsort4" not in _ss.installed_sorters():
        msg = (
            "mountainsort4 not in spikeinterface.sorters.installed_sorters(); "
            "skipping. Set SPIKESORTING_V2_REQUIRE_MS4=1 to make this a "
            "hard fail."
        )
        if os.environ.get("SPIKESORTING_V2_REQUIRE_MS4") == "1":
            pytest.fail(msg)
        pytest.skip(msg)

    fixture_path = (
        _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
    )
    if not fixture_path.exists():
        pytest.skip(
            f"NWB fixture {fixture_path} missing; generate via "
            "tests/spikesorting/v2/fixtures/generate_mearec.py first."
        )

    sorter_label = "ms4"
    baseline_root_env = os.environ.get("SPIKESORTING_V2_BASELINE_ROOT")
    if baseline_root_env:
        baseline_dir = (
            _Path(baseline_root_env)
            / fixture_stem
            / sorter_label
            / f"shank{sort_group_id}"
        )
    else:
        pytest.skip(
            "SPIKESORTING_V2_BASELINE_ROOT unset -- set it to a "
            "directory tree of `<root>/<fixture_stem>/ms4/shank<N>/"
            "{baseline_v1_spike_times.pkl,baseline_v1_recording_meta.json}` "
            "to enable the v1↔v2 MS4 matrix verification. Captures are "
            "produced by tests/spikesorting/v2/scripts/capture_polymer_ms4.sh."
        )

    triple = (fixture_stem, "mountainsort4", sort_group_id)
    if triple in EXPECTED_DEGENERATE_CASES:
        # See clusterless test for marker-artifact rationale.
        if baseline_root_env:
            marker = baseline_dir / "DEGENERATE_MARKER"
            if not marker.exists():
                pytest.fail(
                    f"triple {triple} is in EXPECTED_DEGENERATE_CASES "
                    "but no DEGENERATE_MARKER artifact at "
                    f"{marker}. Operator must `touch` the marker after "
                    "evidence-based capture-side triage. Documented "
                    f"reason: {EXPECTED_DEGENERATE_CASES[triple]}"
                )
        pytest.skip(
            f"SKIP-expected-degenerate: {EXPECTED_DEGENERATE_CASES[triple]}"
        )

    spikes_pkl = baseline_dir / "baseline_v1_spike_times.pkl"
    meta_json = baseline_dir / "baseline_v1_recording_meta.json"
    if not (spikes_pkl.exists() and meta_json.exists()):
        msg = (
            f"v1 MS4 baseline artifacts not found at {baseline_dir}/. "
            "Generate via `tests/spikesorting/v2/scripts/"
            "capture_polymer_ms4.sh` under the v1 conda env."
        )
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    with open(spikes_pkl, "rb") as fh:
        v1_spike_times = pickle.load(fh)
    meta = _json.loads(meta_json.read_text())

    if meta.get("sorter") != "mountainsort4":
        pytest.skip(
            f"v1 baseline at {baseline_dir} was captured for "
            f"sorter={meta.get('sorter')!r}; MS4 parity test refuses "
            "to apply MS4-band contract against a non-MS4 baseline."
        )

    # Set up v2 pipeline -- mirrors the clusterless test until the
    # SorterParameters insert and the comparator.
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
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

    nwb_file_name = copy_and_insert_nwb(fixture_path)
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()

    # v2-side insert of the polymer-60s MS4 row. Owned end-to-end by
    # this test pair (parity_canonical also strips schema_version, so
    # the canonical fingerprint matches v1's row even though v2's
    # Pydantic schema stamps it).
    sorter_params_name = MS4_60S_POLYMER_PARAM_NAME
    (
        SorterParameters
        & {"sorter": "mountainsort4", "sorter_params_name": sorter_params_name}
    ).delete(safemode=False)
    SorterParameters().insert1(
        {
            "sorter": "mountainsort4",
            "sorter_params_name": sorter_params_name,
            "params": dict(MS4_60S_POLYMER_PARAMS),
            "params_schema_version": 1,
            "job_kwargs": None,
        },
        skip_duplicates=False,
    )

    LabTeam.insert1(
        {"team_name": meta["team_name"], "team_description": "v1 MS4 parity"},
        skip_duplicates=True,
    )
    if not (
        SortGroupV2
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
        }
    ):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    from tests.spikesorting.v2._smoke_constants import (
        v2_preproc_name_for_v1,
    )
    preproc_name_v2 = v2_preproc_name_for_v1(
        meta.get("preproc_param_name", "default")
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
            "interval_list_name": meta["interval_list_name"],
            "preproc_params_name": preproc_name_v2,
            "team_name": meta["team_name"],
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    artifact_name_meta = meta.get("artifact_param_name", "default")
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": artifact_name_meta,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1↔v2 invariant fingerprint check (same shape as clusterless) ---
    import hashlib as _hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )
    from tests.spikesorting.v2._parity_canonical import (
        _normalize,
        assert_canonical_dict_equal,
        canonical_artifact,
        canonical_preproc,
        canonical_sorter,
    )

    _fingerprint_fields = (
        "nwb_sha256",
        "sort_group_electrode_ids",
        "bad_channel_by_electrode_id",
        "canonical_preproc_params",
        "canonical_artifact_params",
        "artifact_valid_times",
        "canonical_sorter_params",
    )
    missing_fp = sorted(set(_fingerprint_fields) - set(meta))
    if missing_fp:
        msg = (
            f"v1 MS4 baseline meta at {meta_json} lacks fingerprint "
            f"field(s) {missing_fp}; baseline was captured before the "
            "invariant-fingerprint schema landed."
        )
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    v2_valid_times_raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_interval_list_name(
                art_pk["artifact_id"]
            ),
        }
    ).fetch1("valid_times")
    v2_valid_times = np.round(
        np.ascontiguousarray(v2_valid_times_raw, dtype="<f8").reshape(-1, 2),
        decimals=3,
    )
    v2_fingerprints = {
        "nwb_sha256": _hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "sort_group_electrode_ids": sorted(
            int(eid)
            for eid in (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": int(meta["sort_group_id"]),
                }
            ).fetch("electrode_id")
        ),
        "bad_channel_by_electrode_id": {
            str(int(row["electrode_id"])): bool(row["bad_channel"] == "True")
            for row in (
                Electrode & {"nwb_file_name": nwb_file_name}
            ).fetch("KEY", "electrode_id", "bad_channel", as_dict=True)
        },
        "canonical_preproc_params": canonical_preproc(
            (
                PreprocessingParameters
                & {"preproc_params_name": preproc_name_v2}
            ).fetch1("params")
        ),
        "canonical_artifact_params": canonical_artifact(
            (
                ArtifactDetectionParameters
                & {"artifact_params_name": artifact_name_meta}
            ).fetch1("params")
        ),
        "artifact_valid_times": _normalize(v2_valid_times),
        "canonical_sorter_params": canonical_sorter(
            "mountainsort4",
            (
                SorterParameters
                & {
                    "sorter": "mountainsort4",
                    "sorter_params_name": sorter_params_name,
                }
            ).fetch1("params"),
        ),
    }
    v1_fingerprints = {field: meta[field] for field in _fingerprint_fields}
    assert_canonical_dict_equal(
        v1_fingerprints, v2_fingerprints, path="fingerprints"
    )

    # Now run the MS4 sort and compute aggregate metrics.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort4",
            "sorter_params_name": sorter_params_name,
            "artifact_id": art_pk["artifact_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")

    v2_per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    v2_spike_times = {
        i: np.asarray(arr) for i, arr in enumerate(v2_per_unit)
    }
    v1_spike_times_arrs = {
        int(uid): np.asarray(times)
        for uid, times in v1_spike_times.items()
    }

    # Use ``artifact_valid_times`` duration (the interval the SORTER
    # actually saw) as the firing-rate denominator. The pre-fix code
    # used ``approx_last_spike_s`` from the v1 meta which is the
    # last v1 spike time -- making both v1 and v2 firing rates
    # NORMALIZED BY v1's OUTPUT (not the fixed input window). The
    # fingerprint check above has already asserted v1's and v2's
    # ``artifact_valid_times`` match, so using v1's stored array is
    # equivalent to v2's reconstructed one.
    v1_avt = meta["artifact_valid_times"]
    v1_intervals = np.asarray(v1_avt["_array_data"]).reshape(
        v1_avt["_array_meta"]["shape"]
    )
    duration_s = float(np.sum(v1_intervals[:, 1] - v1_intervals[:, 0]))
    if duration_s <= 0:
        pytest.fail(
            "artifact_valid_times duration <= 0; non-comparable. "
            f"intervals={v1_intervals.tolist()}"
        )

    def _aggregate(spike_dict, dur):
        n_units = len(spike_dict)
        if n_units == 0 or dur <= 0:
            return n_units, 0.0
        rates = sorted(len(t) / dur for t in spike_dict.values())
        mid = n_units // 2
        median_fr = (
            rates[mid]
            if n_units % 2 == 1
            else (rates[mid - 1] + rates[mid]) / 2
        )
        return n_units, median_fr

    v1_n, v1_fr = _aggregate(v1_spike_times_arrs, duration_s)
    v2_n, v2_fr = _aggregate(v2_spike_times, duration_s)

    print(
        f"\n[parity-ms4] {fixture_stem} shank{sort_group_id}\n"
        f"  v1: n_units={v1_n}, median_fr={v1_fr:.3f} Hz\n"
        f"  v2: n_units={v2_n}, median_fr={v2_fr:.3f} Hz"
    )

    # MS4_BANDS = MS4_CALIBRATED post-Phase B11 (n_units ± 10% or
    # ± 2, median_fr ± 10% rel). See MS4_VARIANCE_TABLE for the
    # calibration measurements.
    n_band = max(
        v1_n * MS4_BANDS["n_units_rel_band"],
        MS4_BANDS["n_units_abs_band"],
    )
    fr_band_abs = abs(v1_fr * MS4_BANDS["median_fr_rel_band"])
    failures = []
    if abs(v2_n - v1_n) > n_band:
        failures.append(
            f"n_units divergence: |v2_n({v2_n}) - v1_n({v1_n})|={abs(v2_n - v1_n)} "
            f"> band={n_band} (rel={MS4_BANDS['n_units_rel_band']:.0%} "
            f"abs={MS4_BANDS['n_units_abs_band']})."
        )
    if v1_fr > 0 and abs(v2_fr - v1_fr) > fr_band_abs:
        failures.append(
            f"median_fr divergence: |v2_fr({v2_fr:.3f}) - v1_fr({v1_fr:.3f})|"
            f"={abs(v2_fr - v1_fr):.3f} Hz > rel-band="
            f"{MS4_BANDS['median_fr_rel_band']:.0%} ({fr_band_abs:.3f} Hz)."
        )
    if failures:
        pytest.fail("v1↔v2 MS4 aggregate-band divergence:\n  " + "\n  ".join(failures))


# MS4 GT is polymer-only. A tetrode case was tried and removed: MS4 with
# default Frank-lab tetrode params cannot resolve the 5 planted units on a
# 4-channel synthetic MEArec tetrode regardless of detect_threshold. Verified
# with pure SpikeInterface (no Spyglass code in the path) by sweeping
# detect_threshold ∈ {3, 4, 5, 6}: every threshold recovers at most 1/5 GT
# units at accuracy ≥ 0.5 (GT unit 4 is 96 % recovered but split across 4 MS4
# output units; GT units 0, 2, 3 missed entirely). This is a sorter
# limitation on low-channel-count probes, not a v2 bug — v2 byte-parity with
# v1 on the polymer MS4 path (test_v2_real_data_v1_parity_mountainsort4) is
# the canonical v2-correctness gate for MS4.
_MS4_GT_CASES = [
    ("polymer_60s", "franklab_probe_ctx_30kHz_ms4"),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "session_label,ms4_param_name",
    _MS4_GT_CASES,
    ids=[case[0] for case in _MS4_GT_CASES],
)
def test_mountainsort4_ground_truth(
    session_label, ms4_param_name, request, dj_conn
):
    """MS4 recovers planted units on the polymer MEArec fixture.

    Correctness gate independent of v1 parity. Uses SI's
    ``compare_sorter_to_ground_truth`` against the sidecar GT units
    table written by ``mearec_to_spyglass_nwb``.

    Parametrization:
      * ``polymer_60s`` -- 128-channel polymer probe (4 shanks),
        ``franklab_probe_ctx_30kHz_ms4`` params (freq 300-6000).

    Tetrode coverage was attempted and removed (see comment above
    ``_MS4_GT_CASES``): MS4 fundamentally cannot resolve a 4-channel
    synthetic tetrode regardless of detect_threshold, so the case was
    measuring a sorter limitation rather than v2 correctness.

    Threshold: at least 1/2 of planted units detected at accuracy
    >= 0.5 (looser than the MS5 polymer gate's 3/4 >= 0.7 because
    MS4 is a less-accurate sorter).
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
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

    session = request.getfixturevalue(f"{session_label}_session")
    nwb_file_name = session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    if not (LabTeam & {"team_name": "v2_test_team"}):
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
        )

    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g) for g in (SortGroupV2 & session).fetch("sort_group_id")
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
                "sorter": "mountainsort4",
                "sorter_params_name": ms4_param_name,
                "artifact_id": art_pk["artifact_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    aggregated = aggregate_units(sortings_by_shank)
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        fs = (
            float(es.rate)
            if es.rate is not None
            else 1.0 / float(np.diff(es.timestamps[:2])[0])
        )
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }
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
        gt_name=f"{session_label}_planted",
        tested_name="v2_mountainsort4",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    summary = (
        f"\n[MS4 vs {session_label} GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f}"
    )
    print(summary)

    assert n_planted >= 1, (
        f"{session_label} fixture has only {n_planted} planted units; "
        f"regenerate before trusting this gate.{summary}"
    )

    # Looser MS4 threshold (half of planted at acc >= 0.5) than the
    # MS5 polymer gate (3/4 >= 0.7) because MS4 is a less-accurate
    # sorter. Tightening this threshold without committed calibration
    # evidence is regression-prone.
    n_well_detected = int((accuracies >= 0.5).sum())
    threshold = max(1, n_planted // 2)
    assert n_well_detected >= threshold, (
        f"MS4 on {session_label} detected {n_well_detected} of "
        f"{n_planted} planted units at accuracy >= 0.5; "
        f"threshold requires >= {threshold} (1/2 of planted).{summary}"
    )


_CLUSTERLESS_GT_CASES = [
    ("polymer_60s", "smoke_clusterless_5uv"),
    ("tetrode_60s", "smoke_clusterless_5uv"),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "session_label,sorter_params_name",
    _CLUSTERLESS_GT_CASES,
    ids=[case[0] for case in _CLUSTERLESS_GT_CASES],
)
def test_clusterless_thresholder_ground_truth(
    session_label, sorter_params_name, request, dj_conn
):
    """Clusterless detector recovers planted-unit spikes (shank-aware recall).

    Correctness gate for the clusterless path. Unlike MS4/MS5, the
    clusterless detector emits unsorted peaks (not units), so SI's
    ``compare_sorter_to_ground_truth`` doesn't apply. We compute a
    shank-aware recall metric: per planted unit, compute the
    ±0.4-ms time-recall against each shank's v2 peak stream
    independently and take the max. The match constraint is "v2
    peaks from a SINGLE shank must explain the planted spikes" --
    cross-shank peaks can no longer satisfy a planted spike. Best-
    shank-per-unit (rather than nearest-shank-by-soma-position) is
    robust to cells placed between shanks or off the probe edge:
    such cells fire most strongly on whichever shank actually picks
    them up, which is an electrical question, not a Euclidean one.

    Shank-aware matching closes the false-recall window that
    time-only-across-all-shanks opens: with ~17 K peaks across
    4 shanks and 60 s, uniform-cross-channel time-only matching at
    a 0.4 ms tolerance could inflate recall by ~20 percentage points
    from unrelated peaks. Best-shank routing eliminates that
    inflation while keeping the rigorous time tolerance.

    ``delta_time = 0.4 ms`` matches SI's
    ``compare_sorter_to_ground_truth`` default and the sibling MS4/
    MS5 GT gates; the natural physical offset between a planted
    intracellular fire moment and the detected extracellular trough
    (filter group delay + propagation) is empirically ~1-3 samples
    at 32 kHz, so a sub-sample tolerance would reject valid
    detections.

    Threshold: mean recall ≥ 0.80 across planted units. Clusterless
    detect_threshold = 5σ should catch all well-isolated spikes;
    higher recall bars risk flake from near-threshold borderline
    spikes (PR #4341 algorithm change).

    Parametrization:
      * ``polymer_60s`` -- 4 shanks, planted units distributed across them
      * ``tetrode_60s`` -- 1 shank, 4 channels (trivially per-shank)
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
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
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    session = request.getfixturevalue(f"{session_label}_session")
    nwb_file_name = session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    if not (LabTeam & {"team_name": "v2_test_team"}):
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
        )

    # Pre-insert the smoke clusterless row (5 µV; doesn't ship as a v2
    # default; needs delete-then-insert for the same noise_levels-
    # semantics safety reason as the v1↔v2 parity test).
    if sorter_params_name == SMOKE_CLUSTERLESS_PARAM_NAME:
        (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
            }
        ).delete(safemode=False)
        SorterParameters().insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                "params": dict(SMOKE_CLUSTERLESS_PARAMS),
                "params_schema_version": 3,
                "job_kwargs": None,
            },
            skip_duplicates=False,
        )

    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g) for g in (SortGroupV2 & session).fetch("sort_group_id")
    )

    # Per-shank: populate clusterless sorting and keep v2 peak times
    # separate per shank so the recall matcher below can constrain
    # each planted unit's matches to a single shank's v2 peaks.
    v2_times_s_by_shank: dict[int, np.ndarray] = {}
    sampling_frequency = None
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
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                "artifact_id": art_pk["artifact_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        curation_pk = CurationV2.insert_curation(
            sorting_key=sort_pk, labels={}
        )
        merge_id = (
            SpikeSortingOutput.CurationV2 & curation_pk
        ).fetch1("merge_id")
        per_unit = SpikeSortingOutput().get_spike_times(
            {"merge_id": merge_id}
        )
        # Clusterless collapses all peaks into one synthetic unit per
        # shank. Concatenate (in case > 1 entry from get_spike_times)
        # and keep this shank's spike-times tagged by sort_group_id.
        arrays = [np.asarray(a) for a in per_unit if np.asarray(a).size]
        if arrays:
            v2_times_s_by_shank[int(sg_id)] = np.sort(
                np.concatenate(arrays).astype(np.float64)
            )
        else:
            v2_times_s_by_shank[int(sg_id)] = np.empty(0, dtype=np.float64)
        if sampling_frequency is None:
            sampling_frequency = float(
                Sorting().get_sorting(sort_pk).get_sampling_frequency()
            )

    if not v2_times_s_by_shank or all(a.size == 0 for a in v2_times_s_by_shank.values()):
        pytest.fail(f"v2 clusterless produced no peaks on {session_label}.")
    # Aggregate copy for the false-positive sanity bound (counts only).
    v2_times_s = np.sort(
        np.concatenate([a for a in v2_times_s_by_shank.values() if a.size])
    ).astype(np.float64)
    assert sampling_frequency is not None and sampling_frequency > 0

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"{nwb_file_name!r} has no sidecar GT units; regenerate."
        )
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }

    sorted_sgs = sorted(v2_times_s_by_shank.keys())

    # Shank-aware time-recall: for each planted unit, compute the
    # ±0.4-ms time-recall against EACH shank's v2 peak stream
    # separately, then take the max. The match constraint is "v2
    # peaks from a single shank must explain the planted spikes" --
    # cross-shank peaks no longer satisfy a planted spike. Compared
    # to assigning each unit to its nearest shank by soma position,
    # the max-over-shanks form is robust to cells placed between
    # shanks or off the probe edge in the shank-spanning axis: such
    # cells fire most strongly on whichever shank actually picks
    # them up (which is an electrical question, not a Euclidean-
    # nearest-shank question). Empirically on the polymer 60s
    # fixture, soma-position routing mis-routed 18/24 cells (z
    # extent [-570, +583] vs shanks at z=0/350/774/1124); empirical
    # best-shank routing gives every unit its true assignment.
    #
    # ``delta_time = 0.4 ms`` matches SI's
    # ``compare_sorter_to_ground_truth`` default (also used by the
    # sibling MS4/MS5 GT gates). At 32 kHz that is 12.8 samples; the
    # empirical (detected_peak - planted_spike) distribution on the
    # polymer 60s fixture has IQR ~1-2 samples (post-PR-#4341), so
    # 0.4 ms accepts every physically valid detection without
    # admitting unrelated noise peaks.
    tol_s = 0.4 / 1000.0

    def _recall_against(peaks_s: np.ndarray, gt_sorted: np.ndarray) -> float:
        if peaks_s.size == 0:
            return 0.0
        idx = np.searchsorted(peaks_s, gt_sorted)
        idx_left = np.clip(idx - 1, 0, peaks_s.size - 1)
        idx_right = np.clip(idx, 0, peaks_s.size - 1)
        nearest_diff = np.minimum(
            np.abs(gt_sorted - peaks_s[idx_left]),
            np.abs(gt_sorted - peaks_s[idx_right]),
        )
        return float((nearest_diff <= tol_s).mean())

    per_unit_recall: dict[int, float] = {}
    gt_unit_to_shank: dict[int, int] = {}
    for uid, gt_times in gt_units.items():
        if gt_times.size == 0:
            continue
        gt_sorted = np.sort(gt_times.astype(np.float64))
        per_shank = {
            sg: _recall_against(v2_times_s_by_shank[sg], gt_sorted)
            for sg in sorted_sgs
        }
        best_sg = max(per_shank, key=lambda sg: per_shank[sg])
        per_unit_recall[uid] = per_shank[best_sg]
        gt_unit_to_shank[uid] = best_sg

    n_planted = len(per_unit_recall)
    # Non-vacuity check FIRST: with n_planted == 0 the recall_values
    # array is empty and ``recall_values.mean()`` would raise inside
    # the summary string, masking the intended "no planted GT units"
    # error. Fail with a clear message before any aggregate stats.
    assert n_planted >= 1, (
        f"{session_label} clusterless GT: no planted units recovered "
        "from sidecar (per_unit_recall is empty -- either the fixture "
        "has zero GT units or every unit has zero spike times)."
    )
    recall_values = np.array(list(per_unit_recall.values()))
    shank_distribution = {
        sg: sum(1 for s in gt_unit_to_shank.values() if s == sg)
        for sg in sorted_sgs
    }
    summary = (
        f"\n[Clusterless vs {session_label} GT] "
        f"n_planted={n_planted}, "
        f"n_v2_peaks_total={v2_times_s.size}, "
        f"units_per_shank={shank_distribution}\n"
        f"  shank-aware time-recall (±0.4 ms, best-shank-per-unit): "
        f"mean={recall_values.mean():.3f} "
        f"median={float(np.median(recall_values)):.3f} "
        f"min={recall_values.min():.3f} max={recall_values.max():.3f}"
    )
    print(summary)

    # Threshold: mean recall >= 0.80 across planted units. Loosely
    # justifiable: well-isolated spikes should always be detected
    # by a 5σ threshold; missing some is expected for near-threshold
    # planted units (especially on the sparse-coverage tetrode).
    assert recall_values.mean() >= 0.80, (
        f"Clusterless mean time-recall {recall_values.mean():.3f} < 0.80 "
        f"on {session_label}; v2 detector missing planted spikes.{summary}"
    )

    # Per-fixture calibrated false-positive bound: clusterless emits
    # unsorted peaks (not units), so SI's precision metric doesn't
    # apply, but a regression that emits substantially more peaks than
    # the planted population must not pass this gate by virtue of high
    # recall alone. Bounds are set with ~2x headroom over empirical
    # observation so SI minor-version drift does not cause flake, but
    # tight enough to flag real false-positive regressions (the
    # previous loose 10x bound would only catch the historical 1,400x
    # ``noise_levels=[1.0]`` explosion, missing smaller regressions).
    #
    # Observed v2_peaks/planted_spike ratios (full GT gate runs):
    #   * polymer_60s  -- ~1.5x (4 shanks aggregated)
    #   * tetrode_60s  -- ~1.4x (single shank, 5 planted)
    # Bound is ``3 x observed`` so legitimate background MEArec
    # activity + adjacent-channel duplicates still pass.
    extras_ratio_max = {
        "polymer_60s": 3.0,
        "tetrode_60s": 3.0,
    }.get(session_label)
    if extras_ratio_max is None:
        # New fixture without committed calibration evidence; fall
        # back to the smoke-guard 10x bound and surface the gap.
        extras_ratio_max = 10.0
        print(
            f"WARNING: no per-fixture extras-ratio calibration for "
            f"{session_label!r}; falling back to 10x. Add a "
            f"committed observed ratio + 2-3x bound to extras_ratio_max."
        )
    total_planted_spikes = int(
        sum(t.size for t in gt_units.values() if t.size)
    )
    if total_planted_spikes > 0:
        extras_ratio = v2_times_s.size / total_planted_spikes
        assert extras_ratio <= extras_ratio_max, (
            f"Clusterless on {session_label} emitted "
            f"{v2_times_s.size} peaks vs {total_planted_spikes} "
            f"planted spikes (ratio={extras_ratio:.2f}, > "
            f"{extras_ratio_max:.1f}x calibrated bound). "
            f"Recall is met but the detector is firing well above "
            f"the planted population -- possible noise_levels / "
            f"threshold regression.{summary}"
        )


# ---------- MEDIUM error-handling fixes ----------------------------------


def test_artifact_empty_warning_has_context(dj_conn, monkeypatch):
    """The zero-artifact-frames warning names the artifact_id and the
    active thresholds, not just a bare 'zero frames' message.
    """
    import numpy as np

    from spyglass.spikesorting.v2 import artifact as artifact_mod
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    captured = []
    monkeypatch.setattr(
        artifact_mod.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg),
    )

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    validated = ArtifactDetectionParamsSchema(
        detect=True, amplitude_thresh_uV=500.0
    )
    out = artifact_mod.ArtifactDetection._detect_artifacts(
        rec, validated, context=" for artifact_id=abc-123"
    )
    assert out.shape == (1, 2)  # full window returned on zero artifacts
    assert any("abc-123" in m for m in captured), captured
    assert any("amplitude_thresh_uV" in m for m in captured)


@pytest.mark.slow
def test_set_group_by_column_surfaces_unitrode_skip(polymer_smoke_session):
    """``set_group_by_electrode_table_column`` returns a skip summary for
    a single-channel (unitrode) group instead of silently succeeding.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    eids = sorted(
        int(e)
        for e in (
            Electrode
            & {"nwb_file_name": nwb_file_name, "bad_channel": "False"}
        ).fetch("electrode_id")
    )
    assert len(eids) >= 3

    skipped = SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        column="electrode_id",
        groups=[[eids[0]], [eids[1], eids[2]]],
        omit_unitrode=True,
    )
    assert isinstance(skipped, list)
    assert any(s["reason"] == "unitrode" for s in skipped), skipped
    # The kept 2-channel group was still created.
    assert len(SortGroupV2 & polymer_smoke_session) == 1


@pytest.mark.slow
def test_artifact_delete_aborts_on_unexpected_resolve_error(
    populated_recording, monkeypatch
):
    """An UNEXPECTED resolve_source error during ``delete()`` aborts the
    delete (master not removed) rather than swallowing it and orphaning
    IntervalList rows. Only the documented SchemaBypassError is
    tolerated.
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
            "artifact_params_name": "default",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
    assert len(ArtifactDetection & art_pk) == 1

    def _boom(cls, key):
        raise RuntimeError("boom-unexpected-resolve")

    monkeypatch.setattr(
        ArtifactSelection, "resolve_source", classmethod(_boom)
    )
    with pytest.raises(RuntimeError, match="boom-unexpected-resolve"):
        (ArtifactDetection & art_pk).delete(safemode=False)
    # Aborted before super().delete(): the master row is still present.
    assert len(ArtifactDetection & art_pk) == 1


@pytest.mark.slow
def test_curation_labels_raise_on_stray_unit_ids(
    populated_sorting, monkeypatch
):
    """``insert_curation`` raises when labels reference a unit_id not in
    the sorting (the stray label would otherwise vanish silently);
    ``permissive_labels=True`` restores the warn-and-drop behavior.
    """
    from spyglass.spikesorting.v2 import curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    # Default: a stray unit_id is a hard error (no DB rows written).
    with pytest.raises(ValueError, match="999999"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={999999: ["noise"]},  # not a real unit_id
            description="stray-label test",
        )

    # Opt-in permissive: warn and drop the stray label instead.
    captured = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg),
    )
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={999999: ["noise"]},
        description="stray-label permissive test",
        permissive_labels=True,
    )
    assert any("999999" in m for m in captured), captured
    # The stray label was dropped, not persisted.
    assert 999999 not in {
        int(u) for u in (CurationV2.UnitLabel & pk).fetch("unit_id")
    }


@pytest.mark.slow
def test_curation_insert_idempotent_rejects_new_args(populated_sorting):
    """Re-inserting a root curation with non-default args raises (the
    args would be silently ignored); ``reuse_existing=True`` allows it.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})

    # A non-default description on a second root insert would be silently
    # dropped by the idempotent reuse -> raises. (Uses description rather
    # than labels so the test does not depend on the smoke sort's
    # non-deterministic unit count.)
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting, description="changed description"
        )

    # reuse_existing=True returns the existing root without raising.
    reused = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        description="changed description",
        reuse_existing=True,
    )
    assert reused["curation_id"] == pk["curation_id"]
