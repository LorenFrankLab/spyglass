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
    assert len(SortGroupV2.SortGroupElectrode & polymer_smoke_session) == 128


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
    # runs would otherwise short-circuit populate()). ``force_masters=True``
    # mirrors what ``cautious_delete`` passes to DataJoint: it deletes the
    # source-polymorphic ``ArtifactSelection`` / ``SortingSelection``
    # MASTER when the Recording cascade reaches its ``RecordingSource``
    # part (the part FKs Recording but the master does not, so a bare
    # ``super_delete`` trips DataJoint's part-before-master rule whenever
    # an earlier test attached selections to this recording_id).
    (Recording & recording_selection_key).super_delete(
        warn=False, force_masters=True
    )
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
    assert row["duration_s"] == pytest.approx(expected_duration, rel=1e-3), (
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
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
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


@pytest.mark.slow
def test_tetrode_geometry_attached(tetrode_60s_session):
    """A 4-channel ``tetrode_12.5`` sort group gets the explicit tetrode
    probe geometry, and it survives the NWB round-trip.

    Exercises the all-true branch of ``_maybe_apply_tetrode_geometry``
    (single ``tetrode_12.5`` probe, exactly 4 channels, single
    electrode group) end-to-end: a geometry-aware sorter calling
    ``get_recording`` must see the 4 contacts at the
    ``(0,0)-(0,12.5)-(12.5,0)-(12.5,12.5)`` µm square the patch
    installs. (The four negative-condition tests for the same gate are
    covered separately; this test owns the all-true happy path.)
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = tetrode_60s_session["nwb_file_name"]
    from spyglass.common.common_lab import LabTeam

    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & tetrode_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & tetrode_60s_session).fetch("sort_group_id"))[0]
    )
    # Precondition: the gate's all-true branch requires exactly 4
    # channels in this single-tetrode sort group.
    n_channels = len(
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    )
    assert n_channels == 4, (
        "tetrode fixture sort group must have 4 channels for the "
        f"geometry gate; got {n_channels}."
    )

    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preproc_params_name": "default_franklab",
        "team_name": "v2_test_team",
    }
    rec_pk = RecordingSelection.insert_selection(selection_key)
    # Clear any prior Recording row so populate re-materializes.
    (Recording & rec_pk).super_delete(warn=False, force_masters=True)
    Recording.populate(rec_pk, reserve_jobs=False)

    rec = Recording().get_recording(rec_pk)
    assert rec.get_num_channels() == 4

    # The source electrode table for this legacy-style fixture carries no
    # contact positions (all (0,0,0)); without the geometry patch all four
    # contacts would collapse onto a single point and a geometry-aware
    # sorter would be blind to the tetrode layout. Assert the patch
    # installed a 4-corner 12.5 µm square: two distinct x values 12.5 µm
    # apart, two distinct y values 12.5 µm apart, all four corners
    # present. (The patch positions the square at (0,0)-(12.5,12.5), but
    # probeinterface re-centers contacts on their centroid, so the
    # absolute origin shifts to +/-6.25; the 12.5 µm geometry -- the only
    # thing a sorter consumes -- is what we pin, not the origin.)
    locs = [
        (round(float(x), 3), round(float(y), 3))
        for x, y in rec.get_channel_locations()
    ]
    xs = sorted({x for x, _ in locs})
    ys = sorted({y for _, y in locs})
    assert len(xs) == 2 and len(ys) == 2, (
        f"tetrode contacts collapsed -- expected a 2x2 grid, got "
        f"x={xs}, y={ys} (patch did not spread the degenerate "
        "all-zero source positions)."
    )
    assert round(xs[1] - xs[0], 3) == 12.5
    assert round(ys[1] - ys[0], 3) == 12.5
    assert set(locs) == {
        (xs[0], ys[0]),
        (xs[0], ys[1]),
        (xs[1], ys[0]),
        (xs[1], ys[1]),
    }, f"tetrode contacts {sorted(locs)} do not form a 12.5 µm square."


@pytest.mark.slow
def test_write_nwb_artifact_rejects_heterogeneous_gains(
    polymer_smoke_session,
):
    """``Recording._write_nwb_artifact`` raises on heterogeneous channel
    gains instead of silently scaling every channel by ``gains[0]``.

    Exercises the gain gate behaviorally (a recording whose channels
    carry two distinct gains must abort the write), replacing the prior
    source-comment-presence check. The aborted write unlinks its stub
    analysis file, so no orphan outlives the call.
    """
    import numpy as np
    import spikeinterface.core as si_core

    from spyglass.spikesorting.v2.recording import Recording

    rng = np.random.default_rng(0)
    traces = rng.standard_normal((1000, 2)).astype("float32")
    rec = si_core.NumpyRecording([traces], sampling_frequency=30_000.0)
    # Two distinct channel gains -> the single-conversion-factor invariant
    # is violated and the write must raise.
    rec.set_channel_gains([1.0, 2.0])

    with pytest.raises(ValueError, match="heterogeneous channel gains"):
        Recording._write_nwb_artifact(
            rec, polymer_smoke_session["nwb_file_name"]
        )


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

    merge_ids = (SpikeSortingOutput.CurationV2 & sorting_key).fetch("merge_id")
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
    rec_keys = (RecordingSelection & session_key).fetch("KEY", as_dict=True)
    if rec_keys:
        sorting_keys = (
            SortingSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if sorting_keys:
            merge_ids = (SpikeSortingOutput.CurationV2 & sorting_keys).fetch(
                "merge_id"
            )
            for mid in merge_ids:
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
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
    shared_groups = (SharedArtifactGroup & session_key).fetch(
        "KEY", as_dict=True
    )
    if shared_groups:
        # force_masters=True: the cascade reaches the source-polymorphic
        # ``ArtifactSelection.SharedArtifactGroupSource`` part, whose master
        # is ``ArtifactSelection`` (not ``SharedArtifactGroup``); without it
        # DataJoint raises "delete part before master". Mirrors every other
        # super_delete in this file (Recording/RecordingSource analog).
        (SharedArtifactGroup & shared_groups).super_delete(
            warn=False, force_masters=True
        )

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
    """``insert_group`` rejects members whose recordings span
    different time windows (different ``interval_list_name`` on the
    upstream selection).

    Same NWB, same fs -- but two ``RecordingSelection`` rows
    pointing at different ``interval_list_name`` values can have
    different sample counts. ``si.aggregate_channels`` would
    crash with an opaque shape mismatch deep inside SI; the
    insert-time guard loads each member's preprocessed recording
    and requires EXACT time-axis parity. For two genuinely
    different-duration members the exact-timestamp check
    (``artifact.py`` ~460-485) trips first -- the timestamp
    vectors have different shapes -- raising before the post-loop
    ``distinct_n_samples`` check is reached; either way the
    mismatch surfaces before the user pays the populate cost. (The
    pure n_samples branch is only reachable when timestamp shapes
    match but counts differ, which the per-member ``len(timestamps)
    == n_samples`` assertion makes impossible for real recordings;
    the dtype branch is covered by the companion stub test below.)
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
    # the truncated second_rec_pk. The strict time-axis check at
    # insert_group time MUST raise -- different durations give
    # different-shape timestamp vectors, so the exact-timestamp
    # check fires before the post-loop n_samples check.
    with pytest.raises(ValueError, match="differing exact timestamps"):
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
            self._times = _np.arange(n_samples, dtype=_np.float64) / 30000.0

        def get_num_samples(self):
            return self._n

        def get_dtype(self):
            return self._dtype

        def get_num_segments(self):
            return 1

        def get_times(self):
            return self._times

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

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
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

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
    interval_list_name = f"artifact_{pk['artifact_id']}"
    assert IntervalList & {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }

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
            "sorter_params_name": ("franklab_tetrode_hippocampus_30kHz_ms5"),
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
    """``Sorting.get_sorting`` returns a v1-style frame-relative sorting.

    ``get_sorting`` returns a ``NumpySorting`` (segment frames,
    ``t_start=0``), matching v1's ``NumpySorting.from_unit_dict`` shape:

    - unit_ids match the row's ``n_units``;
    - ``return_times=False`` yields the original recording FRAME indices
      in ``[0, n_samples)`` -- recovered from the stored ABSOLUTE spike
      times via ``np.searchsorted``, so ``timestamps[frame]`` recovers the
      stored time (the ``as_dataframe=True`` ``spike_times`` column);
    - ``return_times=True`` yields frame-relative seconds (``frame / fs``),
      NOT absolute wall-clock. Absolute times live in the units NWB and
      the DataFrame path; the SI object is frame-relative like v1's.

    Depends on ``populated_sorting`` (not ``populated_recording``)
    so the Sorting row this test inspects is guaranteed to exist
    regardless of test ordering or selection (``-k``, ``--lf``).
    """
    import numpy as np

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
    n_samples = len(timestamps)
    fs = rec.get_sampling_frequency()

    # The as_dataframe path carries the stored ABSOLUTE spike times
    # (read straight from the units NWB), indexed by unit_id.
    df = Sorting().get_sorting(populated_sorting, as_dataframe=True)

    for uid in si_sorting.unit_ids:
        frames = si_sorting.get_unit_spike_train(
            unit_id=uid, return_times=False
        )
        if len(frames) == 0:
            continue
        frames = frames.astype(int)
        # Frames are valid recording sample indices.
        assert frames.min() >= 0
        assert frames.max() < n_samples
        # searchsorted round-trip: looking the frames up in the
        # recording timestamps recovers the stored ABSOLUTE spike times.
        abs_times = np.sort(np.asarray(df.loc[int(uid), "spike_times"]))
        recovered = np.sort(timestamps[frames])
        # Absolute (not relative) tolerance: spike times are seconds that
        # can be large, so the default ``rtol=1e-5`` would silently allow
        # ~ms slack on a late spike. Pin to one sample period, absolute.
        assert np.allclose(recovered, abs_times, rtol=0.0, atol=1.0 / fs)
        # return_times=True is frame-relative (t_start=0), v1 shape --
        # NOT absolute wall-clock. Pins that get_sorting returns a
        # NumpySorting, not an absolute-t_start NwbSortingExtractor.
        rel = si_sorting.get_unit_spike_train(unit_id=uid, return_times=True)
        assert np.allclose(
            np.sort(rel), np.sort(frames / fs), rtol=0.0, atol=1.0 / fs
        )


@pytest.mark.slow
def test_sorting_make_fetch_resolves_artifact_obs_intervals(populated_sorting):
    """``make_fetch`` derives obs_intervals from the ArtifactSource part.

    Regression guard for the artifact-source schema: the artifact pass
    lives on the zero-or-one ``ArtifactSource`` part, not a nullable
    ``artifact_id`` FK on the ``SortingSelection`` master. ``make_fetch`` /
    ``make_compute`` / ``_rebuild_analyzer_folder`` gate artifact masking
    on ``sel_row["artifact_id"]``; after the column was dropped, that key
    is absent on the raw ``fetch1()`` row, so ``make_fetch`` must resolve
    it via ``SortingSelection.resolve_artifact(key)`` and stash it. If it
    does not, every artifact-backed sort silently skips masking and
    writes ``obs_intervals=None`` (full-session envelope) -- a silent
    scientific-correctness regression this test exists to catch.

    ``populated_sorting`` is artifact-backed (it inserts an
    ``ArtifactSelection`` + ``ArtifactDetection`` and threads the
    ``artifact_id`` into the sorting selection), so ``resolve_artifact``
    must be non-None and ``obs_intervals`` must be populated.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # The sort is artifact-backed: exactly one ArtifactSource part row.
    artifact_id = SortingSelection.resolve_artifact(populated_sorting)
    assert artifact_id is not None, (
        "populated_sorting must be artifact-backed for this guard to be "
        "meaningful (expected one ArtifactSource part row)."
    )

    fetched = Sorting().make_fetch(populated_sorting)
    # make_fetch must have re-attached the resolved artifact_id onto
    # sel_row (the dropped master column).
    assert fetched.sel_row.get("artifact_id") == artifact_id
    # ...and therefore derived the artifact-removed observation window,
    # not the None full-session fallback.
    assert fetched.obs_intervals is not None, (
        "make_fetch returned obs_intervals=None for an artifact-backed "
        "sort; the ArtifactSource artifact_id was not resolved, so "
        "artifact masking is silently skipped."
    )


@pytest.mark.slow
def test_get_recording_currently_raises_checksum_on_missing_cache(
    populated_recording,
):
    """A18 Test 1 (verify-current-failure): ``get_recording`` raises a
    checksum ``DataJointError`` when the cache file is missing and must be
    rebuilt.

    Pins TODAY's behavior: the on-demand rebuild is NOT byte-deterministic
    versus the external-store checksum (the recompute byte-determinism work
    is deferred to main-epic Phase 2 ``RecordingArtifactRecompute*`` -- see
    overview decision 3 / the deferred C2 note). So when the canonical cache
    is absent, ``get_recording`` rebuilds the file, then re-resolves it with
    checksum validation, and DataJoint rejects the rebuilt-but-not-identical
    bytes. This test locks that failure mode: a future regression that
    silently returned the drifted rebuilt bytes (skipping the checksum) would
    flip this assertion and be caught.

    The happy-path "rebuild succeeds, hash matches" test cannot pass against
    current source and is deliberately deferred to main-epic recompute (see
    the module's "Deliberately deferred" note).
    """
    import shutil as _shutil

    from datajoint.errors import DataJointError

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording

    row = (Recording & populated_recording).fetch1()
    cache_hash = row["cache_hash"]
    assert cache_hash, "populated_recording row must carry a cache_hash"
    analysis_file_name = row["analysis_file_name"]

    # Resolve via from_schema=True (skips checksum) to find the file we are
    # about to remove; this is the same physical file get_recording's
    # existence check (default resolution) will find absent.
    abs_path = AnalysisNwbfile.get_abs_path(
        analysis_file_name, from_schema=True
    )
    assert Path(abs_path).exists()
    # Back up the valid cache: the rebuild overwrites it in place with
    # non-identical bytes, so restore the original afterward or later tests
    # reusing this shared row would themselves fail the checksum read.
    backup = str(abs_path) + ".a18t1.bak"
    _shutil.copy2(abs_path, backup)

    try:
        Path(abs_path).unlink()
        with pytest.raises(DataJointError) as excinfo:
            Recording().get_recording(populated_recording)
        msg = str(excinfo.value)
        assert "checksum" in msg.lower(), (
            "expected a checksum DataJointError from the rebuilt-file "
            f"re-read; got: {msg!r}"
        )
        assert analysis_file_name in msg, (
            "checksum error should name the analysis_file_name "
            f"{analysis_file_name!r}; got: {msg!r}"
        )
    finally:
        _shutil.move(backup, abs_path)


@pytest.mark.slow
def test_rebuild_nwb_artifact_reaches_hash_then_raises_checksum(
    populated_recording, monkeypatch
):
    """A18 Test 2 (verify-current-failure): ``_rebuild_nwb_artifact`` runs the
    rebuild pipeline far enough to call ``_hash_nwb_recording`` on the rebuilt
    artifact, then raises a checksum ``DataJointError``.

    The plan anticipated the direct rebuild "running to completion" (only
    ``get_recording``'s reader rejecting the bytes), but ground truth is
    stronger evidence: ``_write_nwb_artifact`` finishes its write and then
    calls ``_hash_nwb_recording`` to record the rebuilt ``cache_hash`` -- and
    ``_hash_nwb_recording`` reads the file through the checksum-validated
    ``get_abs_path``, which rejects the rebuilt-but-not-byte-identical file.
    So the rebuild raises a checksum error from INSIDE its own hashing step,
    not only at ``get_recording``'s re-read. (Confirmed by the call chain
    ``_rebuild_nwb_artifact`` -> ``_compute_recording_artifact`` ->
    ``_write_nwb_artifact`` -> ``_hash_nwb_recording`` -> ``get_abs_path`` ->
    DataJoint external checksum.)

    This pins TWO facts a future regression must not break silently:
    (1) the rebuild machinery is exercised up to and including the
    ``_hash_nwb_recording`` verification call (counter asserts it is reached),
    and (2) the rebuild is non-functional versus the external-store checksum
    today (it raises rather than persisting drifted bytes). When main-epic
    Phase 2 ``RecordingArtifactRecompute*`` makes the rebuild byte-deterministic
    (or reconciles the checksum), this test flips to the deferred happy path.
    The skills directive for A18 is explicit that the verify-current-failure
    tests must exercise the raising path -- this one does.
    """
    import shutil as _shutil

    from datajoint.errors import DataJointError

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.recording import Recording

    row = (Recording & populated_recording).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(
        row["analysis_file_name"], from_schema=True
    )
    # The rebuild overwrites the canonical file in place before it raises;
    # back it up so later tests reusing this shared row read the original
    # (checksum-valid) bytes.
    backup = str(abs_path) + ".a18t2.bak"
    _shutil.copy2(abs_path, backup)

    calls = {"n": 0}
    real_hash = v2_utils._hash_nwb_recording

    def _counting_hash(*args, **kwargs):
        calls["n"] += 1
        return real_hash(*args, **kwargs)

    # _write_nwb_artifact imports _hash_nwb_recording from this module at call
    # time, so patching the source attribute is observed.
    monkeypatch.setattr(v2_utils, "_hash_nwb_recording", _counting_hash)

    try:
        with pytest.raises(DataJointError) as excinfo:
            Recording()._rebuild_nwb_artifact(populated_recording)
        assert "checksum" in str(excinfo.value).lower(), (
            "expected the rebuild's hash step to raise a checksum "
            f"DataJointError; got: {str(excinfo.value)!r}"
        )
        assert calls["n"] >= 1, (
            "_rebuild_nwb_artifact raised before reaching _hash_nwb_recording; "
            "the rebuild's cache_hash verification path was not exercised, so "
            "this does not pin the intended (rebuild-reached-hashing) state."
        )
    finally:
        _shutil.move(backup, abs_path)


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
    (Recording & pk).super_delete(warn=False, force_masters=True)
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

    (Recording & pk).super_delete(warn=False, force_masters=True)
    try:
        with pytest.raises(RecordingTruncatedError, match="Missing: \\d"):
            Recording.populate(pk, reserve_jobs=False)
        # No row may be inserted for an over-request -- otherwise a
        # downstream consumer silently reads a window that does not cover
        # what was requested.
        assert not (Recording & pk)
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": multi_interval_name,
            }
        ).super_delete(warn=False)


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
        # Record only the explicitly-ordered fetches. A single
        # ``.fetch(named_attrs, order_by=...)`` re-enters ``Fetch.__call__``
        # internally in DataJoint 0.14.x (once more with ``attrs=()``,
        # carrying the same ``order_by``), so the raw call count is an
        # implementation detail. What matters is the INVARIANT: every
        # ordered probe-info fetch is keyed on ``electrode_id`` (never
        # left to unspecified DB order), which is what keeps the tri-part
        # DeepHash stable. Asserting an exact call count would over-fit
        # to DataJoint's internal recursion.
        ob = kwargs.get("order_by")
        if ob is not None:
            fetch_order_by.append(ob)
        return original_fetch_call(self, *attrs, **kwargs)

    monkeypatch.setattr(Fetch, "__call__", _spy_fetch)

    first = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    second = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    assert fetch_order_by, "probe-info fetch issued no explicitly-ordered fetch"
    assert all(ob == "electrode_id" for ob in fetch_order_by), (
        "probe-info fetch issued a non-electrode_id ordered fetch; the "
        f"order_by values seen were {fetch_order_by}. Every ordered fetch "
        "must be electrode_id-keyed for the tri-part DeepHash to stay stable."
    )
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
    # The SortingSelection master has no artifact_id FK (artifact state
    # lives on the ArtifactSource part); a master row with no source part
    # is exactly the orphan this exercises.
    SortingSelection().insert1(
        {
            "sorting_id": orphan_sorting,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
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
            "sorter_params_name": ("franklab_tetrode_hippocampus_30kHz_ms5"),
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

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
def test_metrics_source_invalid_preserves_cause(populated_sorting):
    """A bogus ``metrics_source`` re-raise preserves the enum cause.

    ``insert_curation`` coerces ``metrics_source`` through the
    ``MetricsSource`` enum and re-raises a friendlier ``ValueError`` on a
    typo. The re-raise uses ``from exc`` so the underlying enum
    ``ValueError`` is preserved as ``__cause__`` -- without it the
    original "'bogus' is not a valid MetricsSource" context is lost from
    the traceback. (Reaching the enum check requires a real Sorting row;
    the earlier "not in Sorting" guard fires first otherwise.)
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    with pytest.raises(ValueError) as excinfo:
        CurationV2.insert_curation(
            sorting_key=populated_sorting, metrics_source="bogus"
        )
    # The friendly message names the offending value...
    assert "metrics_source" in str(excinfo.value)
    # ...and the underlying enum ValueError is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert excinfo.value.__cause__ is not excinfo.value


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

    pk0 = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    units = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
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
def test_curation_v2_get_sort_group_info_returns_all_electrodes(
    populated_sorting,
):
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    src_sorting = Sorting().get_sorting(populated_sorting)
    cur_sorting = CurationV2().get_sorting(pk)
    assert set(cur_sorting.unit_ids) == set(src_sorting.unit_ids)
    for uid in src_sorting.unit_ids:
        src_times = src_sorting.get_unit_spike_train(uid, return_times=True)
        cur_times = cur_sorting.get_unit_spike_train(
            int(uid), return_times=True
        )
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
    stale = (SpikeSortingOutput.CurationV2 & populated_sorting).fetch(
        "merge_id"
    )
    for mid in stale:
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    # The merge part now has exactly one row for this curation.
    merge_rows = (SpikeSortingOutput.CurationV2 & pk).fetch(as_dict=True)
    assert len(merge_rows) == 1
    merge_id = merge_rows[0]["merge_id"]
    # And the master surfaces the v2 source enum.
    assert (SpikeSortingOutput & {"merge_id": merge_id}).fetch1(
        "source"
    ) == "CurationV2"

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


def test_handle_existing_rejects_duplicate_sort_group_ids(dj_conn):
    """Explicit ``sort_group_ids`` with intra-list duplicates raises up
    front instead of failing late on a DataJoint duplicate-key error
    when the master rows are inserted. Auto-allocated ranges
    (``explicit_sort_group_ids=False``) are always unique, so the check
    is gated on the explicit path. (Found by generalizing from the
    merge-curation ``len(list)`` vs ``len(set)`` bug -- the same pattern
    lived here.)
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    # Duplicates surface on a fresh session (no existing rows): the
    # check runs BEFORE the ``len(existing) == 0`` early return so it
    # fires either way.
    with pytest.raises(ValueError, match="duplicate"):
        SortGroupV2._handle_existing(
            nwb_file_name="nonexistent.nwb",
            new_sort_group_ids=[3, 3, 4],
            explicit_sort_group_ids=True,
            delete_existing_entries=False,
            confirm=False,
        )
    # The auto-allocated path is exempt (range() is unique by
    # construction); a same-shape list with no duplicate must NOT raise.
    SortGroupV2._handle_existing(
        nwb_file_name="nonexistent.nwb",
        new_sort_group_ids=[3, 4, 5],
        explicit_sort_group_ids=True,
        delete_existing_entries=False,
        confirm=False,
    )


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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # The consumer-facing API returns a list of per-unit spike-time
    # arrays. Pin shape (n_units arrays) + dtype (float seconds) +
    # value range (non-negative, within the recording duration).
    spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    # gracefully`` -- it confirms a zero-unit sort still yields an empty
    # (but real) curation + merge row. MS5 here is the populated-units
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
@pytest.mark.integration
def test_run_v2_pipeline_idempotent_row_counts(polymer_smoke_session):
    """A second ``run_v2_pipeline`` leaves exactly ONE row per stage.

    ``test_run_v2_pipeline_end_to_end_and_idempotent`` checks that the
    rerun returns the same PKs, but PK-equality alone does not prove the
    second run inserted no duplicate rows -- a Selection whose insert
    helper failed to dedup would return the same PK yet leave two rows.
    This asserts ``len(... & pk) == 1`` on every stage table after the
    second run, proving the rerun is a true no-op.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    kwargs = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        preset="franklab_tetrode_mountainsort5",
    )
    manifest = run_v2_pipeline(**kwargs)
    # Second run must be a pure no-op (MS5 reuses the existing sorting_id
    # rather than re-clustering, so this does not depend on sorter
    # determinism).
    manifest2 = run_v2_pipeline(**kwargs)
    assert manifest2["merge_id"] == manifest["merge_id"]

    # Count by LOGICAL identity, not by the generated UUID PK. The PKs
    # (recording_id, artifact_id, sorting_id, merge_id) are fresh
    # ``uuid.uuid4()`` values, so ``& {pk: <uuid>}`` is always 1 by
    # construction and would NOT catch a second logical selection inserted
    # with a different UUID, nor a second root curation for the sorting.
    # Each count below uses the same logical key ``insert_selection``
    # dedups on (and the root-curation identity the orchestrator reuses).
    rec_row = (
        RecordingSelection & {"recording_id": manifest["recording_id"]}
    ).fetch1()
    rec_logical = {k: v for k, v in rec_row.items() if k != "recording_id"}

    art_row = (
        ArtifactSelection * ArtifactSelection.RecordingSource
        & {"artifact_id": manifest["artifact_id"]}
    ).fetch1()

    sort_row = (
        SortingSelection
        * SortingSelection.RecordingSource
        * SortingSelection.ArtifactSource
        & {"sorting_id": manifest["sorting_id"]}
    ).fetch1()

    stage_counts = {
        "RecordingSelection": len(RecordingSelection & rec_logical),
        "ArtifactSelection": len(
            ArtifactSelection * ArtifactSelection.RecordingSource
            & {
                "recording_id": art_row["recording_id"],
                "artifact_params_name": art_row["artifact_params_name"],
            }
        ),
        "SortingSelection": len(
            SortingSelection
            * SortingSelection.RecordingSource
            * SortingSelection.ArtifactSource
            & {
                "sorter": sort_row["sorter"],
                "sorter_params_name": sort_row["sorter_params_name"],
                "recording_id": sort_row["recording_id"],
                "artifact_id": sort_row["artifact_id"],
            }
        ),
        # A re-run must NOT mint a second ROOT curation for the sorting.
        "CurationV2_root": len(
            CurationV2
            & {
                "sorting_id": manifest["sorting_id"],
                "parent_curation_id": -1,
            }
        ),
        # ...nor a second merge entry for that curation.
        "SpikeSortingOutput.CurationV2": len(
            SpikeSortingOutput.CurationV2
            & {
                "sorting_id": manifest["sorting_id"],
                "curation_id": manifest["curation_id"],
            }
        ),
    }
    assert all(c == 1 for c in stage_counts.values()), (
        "run_v2_pipeline rerun left duplicate LOGICAL rows (expected "
        f"exactly 1 per stage): {stage_counts}"
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
            "params_schema_version": 4,
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
        "rates and the detect_threshold of 5 (a MAD multiplier, not "
        "µV -- see the note below). A buggy detector that returns a "
        "single false-positive would pass `n_spikes > 0`."
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
        int(g)
        for g in (SortGroupV2 & polymer_60s_session).fetch("sort_group_id")
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


# ---------- Persisted-timestamp readback regression ----------------------


@pytest.mark.slow
def test_recording_timestamps_reads_persisted_vector(tmp_path, dj_conn):
    """``Sorting._recording_timestamps`` returns the upstream Recording's
    actual persisted timestamp vector (first value non-zero), not a
    hardcoded/affine grid from 0.0.

    ``get_sorting`` maps absolute spike times back to frames via
    ``np.searchsorted`` against this vector, so a wrong t_start (the
    old v1-era 0.0 hardcode) or an affine reconstruction would
    mis-map every frame. The smoke and 60s polymer fixtures both start
    at t=0, which hid the bug; this writes a synthetic NWB with
    ``timestamps[0] = 12345.6`` and pins that the full vector comes
    back via the ``electrical_series_path`` indirection. Stays out of
    the DB by monkeypatching ``AnalysisNwbfile.get_abs_path``.
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
        recovered = Sorting._recording_timestamps(fake_row)

    assert len(recovered) == n_samples, (
        f"_recording_timestamps returned {len(recovered)} samples; "
        f"expected the full {n_samples}-sample vector."
    )
    assert recovered[0] == pytest.approx(t0, abs=1e-6), (
        f"_recording_timestamps[0] returned {recovered[0]}; expected "
        f"{t0}. The v1-era t_start=0 hardcode would silently pass on "
        "t=0 fixtures but break on real session recordings that start "
        "mid-day."
    )
    assert np.allclose(recovered, timestamps), (
        "_recording_timestamps must return the persisted timestamp "
        "vector verbatim (searchsorted readback depends on it)."
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    nwb_results = SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
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
    assert all(
        c > 0 for c in post_counts
    ), f"initialize_v2_defaults produced zero rows: {post_counts}."


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
            "params_schema_version": 4,
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
            assert (
                manifest.get(key) is not None
            ), f"Manifest missing {key!r}; got {manifest}."
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

    A zero-unit sort yields an EMPTY (but real) curation + merge row --
    matching v1's empty Units table -- so the manifest is complete
    (``curation_id`` + ``merge_id`` present, ``n_units=0``) and downstream
    code can treat it like any other ``SpikeSortingOutput`` row instead of
    special-casing a ``None`` merge_id.
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

    # A zero-unit sort still produces a COMPLETE, merge-keyable manifest:
    # an empty curation + merge row (v1 parity), not a partial None.
    for key in ("recording_id", "artifact_id", "sorting_id"):
        assert (
            manifest.get(key) is not None
        ), f"Zero-unit manifest missing {key!r}; got {manifest}."
    assert manifest["n_units"] == 0
    assert manifest["curation_id"] is not None, (
        f"Zero-unit sort should still write an empty curation row; "
        f"got {manifest}."
    )
    assert manifest["merge_id"] is not None, (
        "Zero-unit sort should still write an empty merge row so the "
        f"result is merge-keyable; got {manifest}."
    )

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

    # The empty curation that run_v2_pipeline created is real and
    # merge-keyable, with zero Unit rows; CurationV2.get_sorting on it
    # returns an empty sorting (it builds its OWN extractor, so it needs
    # the same zero-unit guard).
    curation_pk = {
        "sorting_id": manifest["sorting_id"],
        "curation_id": manifest["curation_id"],
    }
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
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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

    monkeypatch.setattr(Sorting, "_populate_unit_part", _broken_unit_part)

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

    Regression guard for the final-sample boundary. ``_write_units_nwb``
    stores ``timestamps[sample_index]`` as the absolute spike time;
    ``get_sorting`` reads it back by mapping absolute time -> frame with
    ``np.searchsorted`` (``_spike_times_to_frames``), which clips any
    spike that floating-point-rounds to ``n_samples`` (one past the end)
    -- the v1 ``spike_times_to_valid_samples`` clip, now folded in. This
    test pins that a spike at ``n_samples - 1`` survives both read paths
    without raising.

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
    ``Sorting().get_sorting`` reads back via the searchsorted map.
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
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_boundary_run_sorter)
    )

    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, (
        "Sorting.populate failed when the synthetic sorter produced "
        "a boundary spike; this is the boundary failure mode, inside "
        "_write_units_nwb rather than at read time."
    )

    # First read path: Sorting.get_sorting -> searchsorted frame map
    # must build without raising. Unit set must include the one we
    # planted.
    sorting_obj = Sorting().get_sorting(sort_pk)
    assert 0 in sorting_obj.get_unit_ids(), (
        "Boundary-spike sorting did not survive Sorting.get_sorting "
        f"round-trip; unit_ids={list(sorting_obj.get_unit_ids())}."
    )

    # Second read path: CurationV2 + CurationV2.get_sorting. The
    # curated NWB write goes through a different code path than
    # Sorting._write_units_nwb but reads back via the same
    # searchsorted frame map. Pass labels={} so insert_curation
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


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_recovers_frames_across_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """``get_sorting`` recovers original frames across a wall-clock gap.

    Builds a Recording over two DISJOINT sort intervals -- so the
    persisted timeline is gap-preserving (non-uniform) -- and
    monkeypatches ``_run_sorter`` to plant a spike near the start
    (chunk 1) and one near the end (chunk 2, after the gap). Both
    ``Sorting.get_sorting`` and ``CurationV2.get_sorting`` must read the
    planted FRAME indices back exactly.

    Regression for the v1-parity readback bug: the old
    ``NwbSortingExtractor`` readback inverts the stored absolute spike
    times affinely (``round((t - t_start) * fs)``), which shifts every
    frame after the gap by ``gap * fs`` (here landing past the
    gap-excluded sample count). v1 -- and now v2 -- map back with
    ``np.searchsorted`` against the actual recording timestamps, which
    recovers the original frame. The spike-TIME surfaces round-trip
    correctly either way; only the FRAME indices expose the bug, so this
    test asserts on frames.
    """
    import uuid as _uuid

    import numpy as _np

    from spyglass.common.common_interval import IntervalList
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
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
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
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    # Two 1.2 s chunks separated by a 0.5 s wall-clock gap.
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_readback_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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

    planted: dict = {}

    def _planted_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = _np.array([100, n_samples - 100], dtype=_np.int64)
        rec_times = _np.asarray(recording.get_times())
        planted["samples"] = samples
        # The affine inverse of the post-gap frame's absolute time; if
        # this still equals the original frame the gap is too small to
        # expose the bug, so the assertion below would be vacuous.
        planted["affine_post_gap"] = int(
            round(
                (rec_times[samples[1]] - rec_times[0])
                * recording.get_sampling_frequency()
            )
        )
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint Sorting.populate failed"
    assert planted["affine_post_gap"] != int(
        planted["samples"][1]
    ), "test setup: gap too small to expose the affine shift"

    si_sorting = Sorting().get_sorting(sort_pk)
    frames = _np.asarray(si_sorting.get_unit_spike_train(unit_id=0))
    _np.testing.assert_array_equal(
        _np.sort(frames),
        _np.sort(planted["samples"]),
        err_msg=(
            "Sorting.get_sorting did not recover the original frames "
            "across the wall-clock gap (affine readback shifts post-gap "
            f"frames). got={frames.tolist()}, "
            f"planted={planted['samples'].tolist()}"
        ),
    )

    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description="disjoint frame-readback test",
    )
    cur_sorting = CurationV2().get_sorting(curation_pk)
    cur_frames = _np.asarray(cur_sorting.get_unit_spike_train(unit_id=0))
    _np.testing.assert_array_equal(
        _np.sort(cur_frames),
        _np.sort(planted["samples"]),
        err_msg="CurationV2.get_sorting lost frames across the gap",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_obs_intervals_no_artifact_respects_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """No-artifact obs_intervals split at the gap on a disjoint recording.

    When a sort has NO artifact pass (``artifact_id`` is None),
    ``_write_units_nwb`` falls back to the recording's recorded window(s)
    for each unit's ``obs_intervals``. On a DISJOINT recording that must
    be one interval per recorded chunk, NOT a single envelope spanning the
    wall-clock gap (which would inflate the observation duration /
    firing-rate window). The artifact-backed path was already gap-split;
    this pins the no-ArtifactSource fallback.
    """
    import uuid as _uuid

    import numpy as _np
    import pynwb

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
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
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    gap_mid = 0.5 * (chunk1_end + gap_end)
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_obs_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    # SortingSelection with NO artifact_id -> obs_intervals=None fallback.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _planted_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = _np.array([100, n_samples - 100], dtype=_np.int64)
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint no-artifact Sorting.populate failed"

    analysis_file_name = (Sorting & sort_pk).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        obs = _np.asarray(io.read().units["obs_intervals"][0])

    assert obs.shape == (2, 2), (
        "no-artifact obs_intervals over a disjoint recording must be one "
        f"interval per recorded chunk; got {obs.tolist()}"
    )
    for start, end in obs:
        assert not (start < gap_mid < end), (
            f"obs_interval [{start}, {end}] spans the inter-chunk gap "
            f"(gap_mid={gap_mid}); the envelope fallback was used."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_get_merged_sorting_keeps_cross_gap_pair(
    polymer_smoke_session, monkeypatch
):
    """Lazy get_merged_sorting must not drop a cross-gap boundary pair.

    On a DISJOINT recording the units NWB frames are contiguous across the
    excluded wall-clock gap, so two merged contributors firing at chunk 1's
    last frame and chunk 2's first frame are frame-ADJACENT but ~0.5 s apart
    in real time. A frame-space 0.4 ms dedup (SI's MergeUnitsSorting) would
    wrongly drop one; the abs-time dedup keeps both (and matches the
    apply_merge=True stored train). Regression for the lazy-merge fix.
    """
    import uuid as _uuid

    import numpy as _np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
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
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_merge_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    planted = {}

    def _two_unit_gap_boundary_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        ts = _np.asarray(recording.get_times())
        fs_local = recording.get_sampling_frequency()
        k = int(
            _np.flatnonzero(_np.diff(ts) > 1.5 / fs_local)[0]
        )  # chunk1 last
        # unit 0: a chunk-1 spike + chunk-1's LAST frame; unit 1: chunk-2's
        # FIRST frame + a later chunk-2 spike. Frames k and k+1 are adjacent
        # but separated by the wall-clock gap.
        planted["k"] = k
        u0 = _np.array([100, k], dtype=_np.int64)
        u1 = _np.array([k + 1, k + 1 + 100], dtype=_np.int64)
        return si.NumpySorting.from_unit_dict(
            [{0: u0, 1: u1}], sampling_frequency=fs_local
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_two_unit_gap_boundary_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk

    # apply_merge=False preview curation merging the two units, then lazy.
    pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        merge_groups=[[0, 1]],
        apply_merge=False,
        parent_curation_id=-1,
        description="disjoint cross-gap merge test",
    )
    merged = CurationV2().get_merged_sorting(pk)
    assert merged.get_num_units() == 1, "the two units should merge into one"
    merged_frames = set(
        int(f)
        for uid in merged.get_unit_ids()
        for f in merged.get_unit_spike_train(unit_id=uid)
    )
    k = planted["k"]
    # Both gap-boundary frames survive (frame-space dedup would drop one).
    assert k in merged_frames and (k + 1) in merged_frames, (
        f"cross-gap boundary pair (frames {k}, {k + 1}) was not preserved; "
        f"merged frames={sorted(merged_frames)}"
    )
    assert len(merged_frames) == 4, (
        f"all 4 planted spikes should survive (no cross-gap dedup); "
        f"got {sorted(merged_frames)}"
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
    ``_detect_artifacts`` (artifact.py:846) and writes a single all-
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
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
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
    (artifact.py:926) without short-circuiting at the
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
    assert (
        "random_seed" not in jk
    ), f"random_seed leaked into detect_peaks job kwargs: {jk}"
    assert jk.get("n_jobs") == 1  # other kwargs preserved


@pytest.mark.slow
def test_clusterless_detect_peaks_strips_threshold_unit(dj_conn, monkeypatch):
    """``_run_clusterless_thresholder`` strips ``threshold_unit`` before
    calling ``detect_peaks``.

    ``threshold_unit`` is a Spyglass-side knob (it selects how
    ``noise_levels`` is derived); it is NOT a ``detect_peaks`` method
    kwarg, so leaving it in the params dict would reach SI and raise an
    unexpected-keyword error at sort time. ``detect_peaks`` is stubbed to
    capture the method kwargs it receives; assert ``threshold_unit`` is
    absent while the real detector knobs survive.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["method_kwargs"] = method_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    Sorting._run_clusterless_thresholder(
        sorter_params={
            "detect_threshold": 5.0,
            "threshold_unit": "uv",
            "noise_levels": [1.0],
        },
        recording=rec,
        job_kwargs={"n_jobs": 1},
    )

    mk = captured["method_kwargs"]
    assert mk is not None, "detect_peaks was not called"
    assert (
        "threshold_unit" not in mk
    ), f"threshold_unit leaked into detect_peaks method kwargs: {mk}"
    # The real detector knobs survive the strip.
    assert mk.get("detect_threshold") == 5.0


def test_build_analyzer_strips_random_seed(dj_conn, monkeypatch, tmp_path):
    """``_build_analyzer`` strips ``random_seed`` before
    ``SortingAnalyzer.compute``.

    ``random_seed`` is a Spyglass-side knob (consumed by the sorter and
    the whitening pin); SI's ``compute`` rejects it with
    ``AssertionError: please remove {'random_seed'}``. The analyzer
    factory + ``compute`` are stubbed to capture the kwargs without a
    real sort.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.setattr(
        v2_utils, "_analyzer_path", lambda key: tmp_path / "analyzer"
    )
    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        si, "create_sorting_analyzer", lambda **k: _FakeAnalyzer()
    )

    class _FakeSorting:
        def get_num_units(self):
            return 2

    Sorting._build_analyzer(
        _FakeSorting(),
        None,
        {"sorting_id": "test-sorting-id"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"random_seed": 7, "n_jobs": 1},
    )

    jk = captured["kwargs"]
    assert (
        "random_seed" not in jk
    ), f"random_seed leaked into analyzer.compute kwargs: {jk}"
    assert jk.get("n_jobs") == 1  # other job kwargs preserved


def test_build_analyzer_compute_args(dj_conn, monkeypatch, tmp_path):
    """``_build_analyzer`` requests the right extensions + analyzer kwargs.

    ``test_build_analyzer_strips_random_seed`` captures only the
    ``compute`` job-kwargs; the analyzer factory kwargs and the extension
    set it computes are unasserted. A regression dropping ``sparse=True``
    (dense templates -> wrong peak-channel attribution) or
    ``return_in_uV=True`` (peak amplitudes in raw counts, not µV) would
    pass that test. This pins both the ``create_sorting_analyzer`` kwargs
    and the exact ``analyzer.compute`` extension list + per-extension
    params (the persisted ``peak_amplitude_uv`` / peak channel depend on
    every one of them).
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.setattr(
        v2_utils, "_analyzer_path", lambda key: tmp_path / "analyzer"
    )
    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["compute_args"] = args
            captured["compute_kwargs"] = kwargs

    def _fake_create(**kwargs):
        captured["create_kwargs"] = kwargs
        return _FakeAnalyzer()

    monkeypatch.setattr(si, "create_sorting_analyzer", _fake_create)

    class _FakeSorting:
        def get_num_units(self):
            return 2

    Sorting._build_analyzer(
        _FakeSorting(),
        None,
        {"sorting_id": "test-sorting-id"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"random_seed": 3, "n_jobs": 1},
    )

    # ``create_sorting_analyzer`` kwargs: sparse + µV-return are
    # correctness-critical for the persisted peak amplitude / channel.
    ck = captured["create_kwargs"]
    assert ck.get("sparse") is True, f"expected sparse=True, got {ck}"
    assert (
        ck.get("return_in_uV") is True
    ), f"expected return_in_uV=True, got {ck}"
    assert ck.get("format") == "binary_folder"

    # ``compute`` extension set (positional first arg).
    ca = captured["compute_args"]
    assert ca, "analyzer.compute was called with no positional extension list"
    assert set(ca[0]) == {
        "random_spikes",
        "noise_levels",
        "templates",
        "waveforms",
    }, f"unexpected extension set: {ca[0]}"

    # Per-extension params: the seeded random-spikes subsample (honoring
    # the per-row random_seed) and the waveform window.
    ext_params = captured["compute_kwargs"]["extension_params"]
    assert (
        ext_params["random_spikes"]["seed"] == 3
    ), "random_spikes seed must honor the job_kwargs random_seed override"
    assert ext_params["random_spikes"]["max_spikes_per_unit"] == 500
    assert ext_params["waveforms"] == {"ms_before": 1.0, "ms_after": 2.0}
    # The Spyglass-only random_seed knob is still stripped from the
    # forwarded job kwargs (SI.compute would reject it).
    assert "random_seed" not in captured["compute_kwargs"]


@pytest.mark.slow
def test_analyzer_rebuild_is_seeded_reproducible(
    dj_conn, monkeypatch, tmp_path
):
    """``_build_analyzer`` seeds ``random_spikes`` so rebuilds are stable.

    The ``random_spikes`` extension uniformly subsamples each unit's
    spikes down to ``max_spikes_per_unit=500`` before computing
    templates; the SI 0.104 default is ``seed=None`` (verified against
    ``ComputeRandomSpikes._set_params``), so without a pinned seed two
    builds of the same sort pick different subsets and the persisted
    peak amplitude / peak channel drift. ``_build_analyzer`` now passes
    ``seed=0``.

    CRITICAL: the seed only changes anything for units with MORE than
    500 spikes -- at or below 500 every spike is selected and the build
    is deterministic regardless of the seed. The MEArec smoke fixture is
    4 s (~tens of spikes/unit), so it would pass this test even with the
    seed reverted (false confidence). This test therefore uses a
    synthetic 40 s, 20 Hz recording whose every unit fires >500 spikes,
    and asserts subsampling actually fired. It drives the real
    ``Sorting._build_analyzer`` (not SI directly) so reverting the seed
    line makes it fail.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import (
        generate_ground_truth_recording,
        template_tools,
    )

    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.sorting import Sorting

    recording, sorting = generate_ground_truth_recording(
        durations=[40.0],
        num_channels=8,
        num_units=3,
        generate_sorting_kwargs={
            "firing_rates": 20,
            "refractory_period_ms": 4.0,
        },
        seed=0,
    )
    totals = {
        int(u): len(sorting.get_unit_spike_train(unit_id=u))
        for u in sorting.unit_ids
    }
    assert min(totals.values()) > 500, (
        "fixture must exceed max_spikes_per_unit=500 so random_spikes "
        f"actually subsamples; got per-unit totals {totals}"
    )

    def _build_and_read(folder):
        # ``_build_analyzer`` imports ``_analyzer_path`` from utils at
        # call time, so patching the utils symbol redirects the output
        # folder for this build.
        monkeypatch.setattr(v2_utils, "_analyzer_path", lambda key: folder)
        Sorting._build_analyzer(
            sorting,
            recording,
            {"sorting_id": "repro-test"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={"n_jobs": 1},
        )
        analyzer = si.load_sorting_analyzer(folder)
        selected = np.asarray(
            analyzer.get_extension("random_spikes").get_data()
        )
        peak_channels = template_tools.get_template_extremum_channel(
            analyzer, outputs="id"
        )
        peak_amplitudes = template_tools.get_template_extremum_amplitude(
            analyzer
        )
        return selected, peak_channels, peak_amplitudes

    sel1, chans1, amps1 = _build_and_read(tmp_path / "build_a")
    sel2, chans2, amps2 = _build_and_read(tmp_path / "build_b")

    # Subsampling actually fired: 500 selected per unit, strictly fewer
    # than the total available -- otherwise the seed is a no-op and the
    # test proves nothing.
    assert len(sel1) == 500 * len(totals)
    assert len(sel1) < sum(totals.values())

    # The two seeded builds are bit-identical in the quantities the
    # pipeline persists (peak channel + peak amplitude per unit) and in
    # the underlying random_spikes selection itself.
    np.testing.assert_array_equal(
        sel1, sel2, err_msg="random_spikes selection differs across builds"
    )
    assert chans1 == chans2
    assert amps1.keys() == amps2.keys()
    for unit_id in amps1:
        assert amps1[unit_id] == amps2[unit_id], (
            f"peak amplitude for unit {unit_id} is not reproducible across "
            "seeded analyzer rebuilds"
        )


@pytest.mark.slow
def test_analyzer_random_seed_override_is_honored(
    dj_conn, monkeypatch, tmp_path
):
    """``_build_analyzer`` honors the per-row ``random_seed`` override.

    ``random_seed`` in ``job_kwargs`` is the established per-row knob that
    the whitening / clusterless-noise pins read
    (``(job_kwargs or {}).get("random_seed", 0)``). The analyzer's
    ``random_spikes`` subsample must read the SAME knob, not a hardcoded
    0 -- otherwise a user changing the seed gets a different sort but the
    same analyzer subsample. Asserts: two builds with ``random_seed=7``
    agree with each other, and differ from the default (seed 0) build --
    proving the override flows through to the extension.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import generate_ground_truth_recording

    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.sorting import Sorting

    recording, sorting = generate_ground_truth_recording(
        durations=[40.0],
        num_channels=8,
        num_units=3,
        generate_sorting_kwargs={
            "firing_rates": 20,
            "refractory_period_ms": 4.0,
        },
        seed=0,
    )
    totals = sum(
        len(sorting.get_unit_spike_train(unit_id=u)) for u in sorting.unit_ids
    )

    def _selection_for_seed(folder, random_seed):
        monkeypatch.setattr(v2_utils, "_analyzer_path", lambda key: folder)
        Sorting._build_analyzer(
            sorting,
            recording,
            {"sorting_id": "seed-override-test"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={"n_jobs": 1, "random_seed": random_seed},
        )
        analyzer = si.load_sorting_analyzer(folder)
        return np.asarray(analyzer.get_extension("random_spikes").get_data())

    seed7_a = _selection_for_seed(tmp_path / "seed7_a", 7)
    seed7_b = _selection_for_seed(tmp_path / "seed7_b", 7)
    seed0 = _selection_for_seed(tmp_path / "seed0", 0)

    # Subsampling fired (otherwise the seed is a no-op and the override
    # can't be observed).
    assert len(seed7_a) < totals
    # The override is deterministic for a fixed seed...
    np.testing.assert_array_equal(
        seed7_a,
        seed7_b,
        err_msg="random_seed=7 is not reproducible across builds",
    )
    # ...and a different seed selects a different subset, proving the
    # override is read (not the hardcoded 0).
    assert not np.array_equal(seed7_a, seed0), (
        "random_seed override did not change the analyzer subsample -- the "
        "extension seed is ignoring job_kwargs['random_seed']."
    )


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
    traces = rng.normal(0.0, 0.5, size=(n_samples, n_channels)).astype(
        _np.float32
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
    assert valid_times[0][1] == pytest.approx(timestamps[1999], abs=1e-9)
    assert valid_times[1][0] == pytest.approx(timestamps[2011], abs=1e-9)


def test_detect_artifacts_amplitude_and_zscore_combined(dj_conn):
    """``_detect_artifacts`` OR-combines thresholds when both are set.

    v2 matches v1's OR semantics (``np.logical_or(above_z,
    above_a)`` at ``spikesorting/utils.py:198``). Under OR a frame is flagged
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
            & {"artifact_params_name": "params_validates_unit_test"}
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
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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
    (Recording & rec_pk).super_delete(warn=False, force_masters=True)

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
    ``apply_merge=True`` writes a curated NWB whose merged-unit id is a
    fresh ``max(source unit_ids) + 1`` (v1 parity, ``v1/curation.py:361``)
    and whose merged spike train is the sorted union of its contributors.
    Surviving source units are written first; merged ids are appended.

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
        sorted((SortGroupV2 & polymer_60s_session).fetch("sort_group_id"))[0]
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
            "sorter_params_name": ("franklab_tetrode_hippocampus_30kHz_ms5"),
            "artifact_id": art_pk["artifact_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    units = sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert len(units) >= 2, (
        f"Sorting on the 60s polymer fixture yielded {len(units)} units; "
        "need >= 2 to test merging. (MS5 typically finds 5+ per shank "
        "on this fixture; a regression upstream would surface here.)"
    )

    # Merge the first two units.
    head, absorbed = units[0], units[1]
    merge_groups = [[head, absorbed]]
    # v1 parity (``v1/curation.py:361``): merged-unit id is
    # ``max(source unit_ids) + 1``; compute it ahead so the next insert
    # can label it (``v1/curation.py:391``: labels applied AFTER merge
    # against final unit_ids).
    merged_id = max(units) + 1

    # v1 parity (``v1/curation.py:391``): labels on the FRESH merged id
    # (max+1) attach to the merged unit; labels on absorbed contributors
    # are silently dropped (v1 writes labels by iterating final
    # unit_ids and never visits absorbed ones -- a warning is logged
    # for informativeness but the insert succeeds).
    pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={merged_id: ["mua"], absorbed: ["noise"]},
        merge_groups=merge_groups,
        apply_merge=True,
        description="merge_groups regression",
    )
    label_pairs = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & pk).fetch(as_dict=True)
    }
    # Label on the fresh merged id survives.
    assert (merged_id, "mua") in label_pairs, label_pairs
    # Label on the absorbed contributor is silently dropped (v1 parity).
    assert absorbed not in {uid for uid, _ in label_pairs}, label_pairs

    curated_unit_ids = set(
        int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
    )
    expected_curated = (set(units) - {head, absorbed}) | {merged_id}
    assert curated_unit_ids == expected_curated, (
        f"Curated unit ids = {sorted(curated_unit_ids)}; "
        f"expected {sorted(expected_curated)} (merged_id={merged_id} "
        "replaces both head and absorbed)."
    )

    # The merged unit's spike train in the curated NWB is the
    # contributors' union with SI's 0.4 ms membership-aware cross-unit
    # duplicate removal: a sub-0.4 ms pair from DIFFERENT contributors is
    # one physical spike double-detected (a neuron's refractory period
    # forbids genuine sub-0.4 ms firing), so it is dropped. v1's lazy
    # get_merged_sorting did this; v2 applies it to the apply_merge=True
    # stored train too (via _dedup_merged_spike_times).
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    src_sorting = Sorting().get_sorting(sort_pk)
    src_head = _np.asarray(
        src_sorting.get_unit_spike_train(unit_id=head, return_times=True)
    )
    src_absorbed = _np.asarray(
        src_sorting.get_unit_spike_train(unit_id=absorbed, return_times=True)
    )
    raw_concat = _np.sort(_np.concatenate([src_head, src_absorbed]))
    expected_merged = _dedup_merged_spike_times(
        [src_head, src_absorbed], 0.4e-3
    )
    # The 60s fixture's two merged units share at least one cross-unit
    # double-detection, so dedup is genuinely exercised end-to-end.
    assert len(expected_merged) < len(raw_concat), (
        "fixture precondition: merged contributors should share a "
        "sub-0.4 ms cross-unit double-detection so dedup fires "
        f"(raw={len(raw_concat)}, deduped={len(expected_merged)})"
    )

    curated_sorting = CurationV2().get_sorting(pk)
    merged_times = _np.asarray(
        curated_sorting.get_unit_spike_train(
            unit_id=merged_id, return_times=True
        )
    )
    assert len(merged_times) == len(expected_merged), (
        f"Merged unit ({merged_id}) has {len(merged_times)} spikes; "
        f"expected {len(expected_merged)} (0.4 ms-deduped union)."
    )
    _np.testing.assert_array_equal(merged_times, expected_merged)

    # n_spikes on CurationV2.Unit for the merged id matches its train
    # (end-to-end half of the n_spikes invariant under v1 parity).
    merged_row = (CurationV2.Unit & pk & {"unit_id": merged_id}).fetch1()
    assert merged_row["n_spikes"] == len(expected_merged), (
        f"CurationV2.Unit.n_spikes for merged_id={merged_id} = "
        f"{merged_row['n_spikes']}; expected {len(expected_merged)}."
    )
    assert merged_row["n_spikes"] == len(merged_times)

    # v1-parity unit ORDER: surviving source units first (in original
    # source order), then the appended merged id last. Matches v1's
    # pop-then-append pattern AND SI's lazy MergeUnitsSorting (originals
    # retained + merged appended), so unit-array consumers comparing
    # applied vs preview/lazy paths see the same per-unit order.
    expected_order = [u for u in units if u not in (head, absorbed)] + [
        merged_id
    ]
    actual_order = [int(u) for u in curated_sorting.get_unit_ids()]
    assert actual_order == expected_order, (
        f"NWB unit order = {actual_order}; expected {expected_order} "
        "(surviving source units first, merged_id appended last)."
    )

    # --- v1-parity PREVIEW half (apply_merge=False), reusing the sort.
    # A preview curation keeps EVERY original unit (the contributor is
    # NOT dropped) + the contributor's label, and records the proposed
    # merge in MergeGroup; get_merged_sorting applies it lazily.
    preview_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={absorbed: ["mua"]},
        merge_groups=merge_groups,
        apply_merge=False,
        parent_curation_id=pk["curation_id"],
        description="merge_groups preview (apply_merge=False)",
    )
    preview_unit_ids = set(
        int(u) for u in (CurationV2.Unit & preview_pk).fetch("unit_id")
    )
    assert preview_unit_ids == set(units), (
        f"apply_merge=False preview unit ids = {sorted(preview_unit_ids)}; "
        f"expected ALL {sorted(units)} (contributor preserved, v1 parity)."
    )
    # The contributor's label survives (it would vanish if the unit were
    # dropped from the preview).
    preview_labels = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & preview_pk).fetch(as_dict=True)
    }
    assert (absorbed, "mua") in preview_labels, preview_labels
    # The preview sorting still has the contributor as its own unit.
    preview_sorting = CurationV2().get_sorting(preview_pk)
    assert absorbed in set(int(u) for u in preview_sorting.get_unit_ids())
    # Symmetric MergeGroup provenance: every CurationV2.Unit row has at
    # least one MergeGroup row keyed by its OWN unit_id, so
    # ``Unit * MergeGroup`` on ``unit_id`` does not silently drop the
    # absorbed contributors. The proposed merge stays: (head,head),
    # (head,absorbed); the absorbed contributor also carries
    # (absorbed,absorbed) as a 1-element self-entry.
    mg_unit_ids = {
        int(r["unit_id"])
        for r in (CurationV2.MergeGroup & preview_pk).fetch(as_dict=True)
    }
    assert preview_unit_ids <= mg_unit_ids, (
        "Every CurationV2.Unit row should have >=1 MergeGroup row keyed "
        f"by its own unit_id; missing {sorted(preview_unit_ids - mg_unit_ids)}."
    )
    # get_merged_sorting applies the proposed merge lazily AND, like v1's
    # lazy path (SI default delta_time_ms=0.4), removes cross-unit
    # double-detections. So two units collapse to one and the total spike
    # count drops by exactly the number of duplicates the apply_merge=True
    # path removed for the same [head, absorbed] group (n_duplicates).
    merged_lazy = CurationV2().get_merged_sorting(preview_pk)
    assert merged_lazy.get_num_units() == len(units) - 1
    preview_total = sum(
        len(preview_sorting.get_unit_spike_train(unit_id=u))
        for u in preview_sorting.get_unit_ids()
    )
    merged_total = sum(
        len(merged_lazy.get_unit_spike_train(unit_id=u))
        for u in merged_lazy.get_unit_ids()
    )
    n_duplicates = len(raw_concat) - len(expected_merged)
    assert merged_total == preview_total - n_duplicates, (
        merged_total,
        preview_total,
        n_duplicates,
    )
    # Frame-level equality (not just counts): the lazy preview's merged
    # train must equal the apply_merge=True stored train sample-for-sample
    # -- both ids are ``max(units) + 1`` (= merged_id). A count-only check
    # would pass even if the two paths binned spikes to different frames.
    lazy_merged_frames = _np.sort(
        _np.asarray(merged_lazy.get_unit_spike_train(unit_id=merged_id))
    )
    applied_merged_frames = _np.sort(
        _np.asarray(curated_sorting.get_unit_spike_train(unit_id=merged_id))
    )
    _np.testing.assert_array_equal(lazy_merged_frames, applied_merged_frames)


@pytest.mark.slow
@pytest.mark.integration
def test_lazy_vs_applied_merge_frames_equal(polymer_smoke_session, monkeypatch):
    """Lazy ``get_merged_sorting`` preview == ``apply_merge=True`` stored
    train, frame-for-frame, on a CONTIGUOUS and a DISJOINT 2-unit sort.

    A planted 2-unit sorter gives deterministic, known spike frames so the
    merged train is computed by hand. Two contributor spikes 1 frame apart
    WITHIN a chunk (cross-unit) are a coincident double-detection -> the
    0.4 ms dedup drops one; two spikes 1 frame apart ACROSS the wall-clock
    gap are seconds apart in real time -> NOT deduped (the gap-correctness
    the absolute-time dedup guarantees). The disjoint applied-merge path is
    otherwise untested. Both the lazy and the stored paths must produce the
    same frames, and that frame set must equal the hand-computed merge.
    """
    import uuid as _uuid

    import numpy as _np
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
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
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # The planted sorter reads its target frames from this closure so the
    # same monkeypatch serves both the contiguous and the disjoint sort.
    state: dict = {}

    def _planted_two_unit_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        u0, u1 = state["u0"], state["u1"]
        samples = _np.concatenate([u0, u1]).astype(_np.int64)
        labels = _np.concatenate(
            [_np.zeros(len(u0)), _np.ones(len(u1))]
        ).astype(_np.int32)
        order = _np.argsort(samples)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_two_unit_sorter)
    )

    def _build_sort(interval_name, valid_times):
        IntervalList.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
                "valid_times": valid_times,
                "pipeline": "v2_lazy_applied_merge_test",
            },
            skip_duplicates=True,
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": interval_name,
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
        return rec_pk, sort_pk

    def _assert_lazy_equals_applied(sort_pk, expected_frames):
        root_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
        preview_pk = CurationV2.insert_curation(
            sorting_key=sort_pk,
            merge_groups=[[0, 1]],
            apply_merge=False,
            parent_curation_id=root_pk["curation_id"],
            description="lazy preview",
        )
        applied_pk = CurationV2.insert_curation(
            sorting_key=sort_pk,
            merge_groups=[[0, 1]],
            apply_merge=True,
            parent_curation_id=root_pk["curation_id"],
            description="applied merge",
        )
        merged_id = 2  # max(0, 1) + 1
        lazy_frames = _np.sort(
            _np.asarray(
                CurationV2()
                .get_merged_sorting(preview_pk)
                .get_unit_spike_train(unit_id=merged_id)
            )
        )
        applied_frames = _np.sort(
            _np.asarray(
                CurationV2()
                .get_sorting(applied_pk)
                .get_unit_spike_train(unit_id=merged_id)
            )
        )
        _np.testing.assert_array_equal(
            lazy_frames,
            applied_frames,
            err_msg="lazy preview merged frames != apply_merge=True stored",
        )
        _np.testing.assert_array_equal(
            lazy_frames,
            _np.asarray(expected_frames, dtype=lazy_frames.dtype),
            err_msg="merged frames do not match the hand-computed merge",
        )

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint half"

    # --- Contiguous half: one interval, uniform timeline. Plant u0/u1
    # with a within-chunk cross-unit coincidence (100 & 101) that dedups,
    # plus distinct spikes; merged = [100, 500, 600, 900].
    state["u0"] = _np.array([100, 500, 900])
    state["u1"] = _np.array([101, 600])
    contig_name = f"v2_lvam_contig_{_uuid.uuid4().hex[:8]}"
    _, contig_sort_pk = _build_sort(
        contig_name, _np.array([[t0, min(t0 + 2.9, t_end)]])
    )
    Sorting.populate(contig_sort_pk, reserve_jobs=False)
    _assert_lazy_equals_applied(contig_sort_pk, [100, 500, 600, 900])

    # --- Disjoint half: two chunks, 0.5 s gap. Recover the chunk-1/chunk-2
    # boundary frame ``k`` from the populated timeline, then plant a
    # within-chunk coincidence (100 & 101 -> dedup) AND a frame-adjacent
    # but cross-gap pair (k & k+1 -> NOT deduped). merged = [100, k, k+1].
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_name = f"v2_lvam_disjoint_{_uuid.uuid4().hex[:8]}"
    disjoint_rec_pk, disjoint_sort_pk = _build_sort(
        disjoint_name,
        _np.array([[t0, chunk1_end], [gap_end, chunk2_end]]),
    )
    rec = Recording().get_recording(disjoint_rec_pk)
    times = _np.asarray(rec.get_times())
    fs = rec.get_sampling_frequency()
    gaps = _np.flatnonzero(_np.diff(times) > 1.5 / fs)
    assert len(gaps) == 1, f"expected one gap, found {len(gaps)}"
    k = int(gaps[0])  # chunk-1's last frame
    assert k >= 200, "chunk 1 too short to plant frame 100/101"
    state["u0"] = _np.array([100, k])
    state["u1"] = _np.array([101, k + 1])
    Sorting.populate(disjoint_sort_pk, reserve_jobs=False)
    _assert_lazy_equals_applied(disjoint_sort_pk, [100, k, k + 1])


def test_curation_n_spikes_matches_apply_merge(dj_conn):
    """``_build_curated_unit_rows`` writes the v1-parity unit set +
    ``n_spikes`` for each ``apply_merge`` mode.

    ``apply_merge=True`` collapses a merge group to a fresh
    ``max(source unit_ids) + 1`` id (v1 parity, ``v1/curation.py:361``)
    carrying the summed contributor train; BOTH the head and the
    absorbed contributors are dropped. ``apply_merge=False`` is a
    v1-style preview (``v1/curation.py:359``): every original unit passes
    through 1:1 with its own ``n_spikes`` and the proposed merge lives in
    MergeGroup for lazy application, so the contributor is preserved --
    not silently dropped. Synthetic ``Sorting.Unit`` rows isolate the row
    logic (the ``apply_merge=True`` end-to-end half is covered by
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

    # apply_merge=True (v1 parity, ``v1/curation.py:361``): the merge
    # group [0, 1] collapses to a fresh id ``max(0,1,2) + 1 = 3``
    # carrying the summed train (140); BOTH the head (0) AND the absorbed
    # (1) are gone. Unit 2 (non-merged) passes through.
    assert merged == {3: 140, 2: 7}, merged
    # apply_merge=False (v1 preview parity): EVERY original unit is
    # present with its OWN count -- both the head (0) and the contributor
    # (1) are preserved as standalone units; the proposed merge lives in
    # MergeGroup.
    assert preview == {0: 100, 1: 40, 2: 7}, preview


def test_curation_two_merge_groups_assign_ids_in_user_input_order(dj_conn):
    """``_build_curated_unit_rows`` assigns fresh merged ids in
    USER-PROVIDED group order (v1 parity, ``v1/curation.py:359``
    iterates ``for merge_group in merge_groups:``). For user input
    ``[[2, 3], [0, 1]]`` the FIRST group ``[2, 3]`` gets the first new
    id ``max(source unit_ids) + 1``, and ``[0, 1]`` gets the next one --
    matching v1. (NOTE: the lazy merge path -- ``get_merged_sorting`` on
    an apply_merge=False preview -- iterates by kept-uid ascending and
    can diverge here; v1 parity for the applied path is the explicit
    goal.)
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

    sorting_units = [
        _unit(0, 10, 50.0),
        _unit(1, 20, 30.0),
        _unit(2, 30, 40.0),
        _unit(3, 40, 25.0),
        _unit(4, 50, 20.0),
    ]
    # User input: bigger-min group FIRST. v1 (and v2 now) iterate in
    # this order; the FIRST group ``[2, 3]`` gets the first new id.
    merge_groups = [[2, 3], [0, 1]]

    rows, kept = CurationV2._build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        curation_id=0,
        apply_merge=True,
    )
    n_spikes_by_unit = {r["unit_id"]: r["n_spikes"] for r in rows}
    # max(0..4) + 1 = 5 -> the FIRST input group [2, 3] gets new_id 5
    # carrying summed n_spikes 70; the second group [0, 1] gets new_id 6
    # with n_spikes 30. Non-merged unit 4 passes through with own count.
    assert n_spikes_by_unit == {4: 50, 5: 70, 6: 30}, n_spikes_by_unit

    # kept_to_contributors keys reflect v1 order: surviving source units
    # FIRST (in source order), then merged ids appended ascending.
    assert list(kept.keys()) == [4, 5, 6], list(kept.keys())

    # apply_merge=False with the same input: heads are min(group) for
    # each multi-group, so kept_to_contributors keys (after sort + sym
    # aug) are deterministic regardless of input order.
    rows_preview, kept_preview = CurationV2._build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        curation_id=0,
        apply_merge=False,
    )
    preview_unit_ids = {r["unit_id"] for r in rows_preview}
    # Every original unit present (v1 preview parity).
    assert preview_unit_ids == {0, 1, 2, 3, 4}, preview_unit_ids
    # Heads are min(group) -- 0 for [0, 1], 2 for [2, 3] -- so a join on
    # MergeGroup.unit_id retrieves the proposed merges from the smallest
    # contributor regardless of how the user listed the group.
    assert 0 in kept_preview and kept_preview[0] == [0, 1]
    assert 2 in kept_preview and kept_preview[2] == [2, 3]


def test_curation_rejects_invalid_merge_groups(dj_conn):
    """All merge-group validation runs BEFORE any early return -- a
    zero-unit sort with non-empty merge_groups, intra-group duplicates,
    and references to nonexistent unit_ids all raise rather than
    silently no-op or fall through to staging that would double-count
    contributors. Covers the empty/singleton shape contract too.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    def _unit(uid):
        return {
            "unit_id": uid,
            "n_spikes": 10,
            "peak_amplitude_uv": 50.0,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": uid,
        }

    nonempty = [_unit(0), _unit(1)]
    cases = [
        # Shape (>= 2 members), regardless of apply_merge.
        (nonempty, [[0]], "fewer than 2"),
        (nonempty, [[]], "fewer than 2"),
        (nonempty, [[0, 1], []], "fewer than 2"),
        # Intra-group duplicates: list-length check alone would miss
        # ``[0, 0]`` (would silently double-count contributor 0).
        (nonempty, [[0, 0]], "duplicate members"),
        (nonempty, [[0, 0, 1]], "duplicate members"),
        # Zero-unit sort with non-empty merge_groups: the validation
        # runs before the empty-by-id early return, so the id check
        # fires.
        ([], [[0, 1]], "not in Sorting.Unit"),
        # Empty/singleton groups on a zero-unit sort still raise their
        # shape error before the id check is reached.
        ([], [[]], "fewer than 2"),
        ([], [[0]], "fewer than 2"),
    ]
    for apply_merge in (True, False):
        for sorting_units, merge_groups, match in cases:
            with pytest.raises(ValueError, match=match):
                CurationV2._build_curated_unit_rows(
                    sorting_id="s",
                    sorting_units=sorting_units,
                    merge_groups=merge_groups,
                    curation_id=0,
                    apply_merge=apply_merge,
                )


def test_si_merge_units_drops_same_sample_unless_delta_is_none(dj_conn):
    """SI 0.104's ``MergeUnitsSorting`` collapses exact same-sample
    spikes from different merged units UNLESS ``delta_time_ms=None``.

    ``delta_time_ms=0`` is NOT a "no duplicate check" knob -- SI still
    drops contributors that fired on the same sample frame. Only
    ``delta_time_ms=None`` disables the check entirely, which is the v1
    ``np.concatenate`` semantic ``get_merged_sorting`` needs. Probe-test
    that locks the SI contract our code depends on; if SI changes its
    handling of ``delta_time_ms=0`` (or ``None``), this fails loudly.
    """
    import numpy as _np
    import spikeinterface as si
    import spikeinterface.curation as sc

    ns = si.NumpySorting.from_unit_dict(
        # Two contributors share sample 10 -- the exact-coincident case.
        {0: _np.array([10]), 1: _np.array([10])},
        sampling_frequency=30000.0,
    )
    new_id_none = list(
        sc.MergeUnitsSorting(
            ns, units_to_merge=[[0, 1]], delta_time_ms=None
        ).get_unit_ids()
    )[0]
    n_none = len(
        sc.MergeUnitsSorting(
            ns, units_to_merge=[[0, 1]], delta_time_ms=None
        ).get_unit_spike_train(unit_id=new_id_none)
    )
    new_id_zero = list(
        sc.MergeUnitsSorting(
            ns, units_to_merge=[[0, 1]], delta_time_ms=0
        ).get_unit_ids()
    )[0]
    n_zero = len(
        sc.MergeUnitsSorting(
            ns, units_to_merge=[[0, 1]], delta_time_ms=0
        ).get_unit_spike_train(unit_id=new_id_zero)
    )
    assert n_none == 2, (
        "delta_time_ms=None must preserve both same-sample contributor "
        f"spikes (got {n_none}); SI behavior has changed."
    )
    assert n_zero == 1, (
        "delta_time_ms=0 should still drop the same-sample duplicate "
        f"(got {n_zero}); SI behavior has changed, re-audit "
        "get_merged_sorting."
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
    """No-artifact and artifact-backed selections are distinct identities.

    Regression guard: an earlier ``insert_selection`` lookup omitted
    ``artifact_id`` from the master restriction whenever it was None,
    which effectively meant "match any artifact_id." That caused a
    no-artifact ``insert_selection`` call to alias onto a pre-existing
    artifact-backed row for the same
    ``(recording_id, sorter, sorter_params_name)`` triple. Under the
    ``ArtifactSource`` part-table design, "no artifact pass" is "no
    ``ArtifactSource`` row" and must stay a distinct identity from an
    artifact-backed selection -- not a wildcard that aliases onto it.

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
    assert SortingSelection.resolve_artifact(artifact_backed_pk) is not None, (
        "populated_sorting must yield an artifact-backed row (an "
        "ArtifactSource part row) for this test to be meaningful."
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
        "artifact-backed row; no-artifact must be a distinct identity, "
        "not match any artifact_id."
    )
    # The fresh row has no ArtifactSource part row.
    assert SortingSelection.resolve_artifact(no_artifact_pk) is None
    assert len(SortingSelection.ArtifactSource & no_artifact_pk) == 0
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


@pytest.mark.slow
def test_sorting_selection_artifact_source_part_shape(populated_recording):
    """Artifact state lives on the ArtifactSource part, not a master FK.

    A no-artifact selection has zero ArtifactSource rows; an
    artifact-backed selection has exactly one whose artifact_id
    ``resolve_artifact`` returns. The ArtifactSource part does NOT leak
    into ``resolve_source`` -- an artifact-backed sort still resolves to
    exactly one recording source.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
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

    sorter_kwargs = {
        "recording_id": populated_recording["recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
    }

    # No-artifact selection: zero ArtifactSource rows; resolve_artifact None.
    no_art_pk = SortingSelection.insert_selection(dict(sorter_kwargs))
    assert len(SortingSelection.ArtifactSource & no_art_pk) == 0
    assert SortingSelection.resolve_artifact(no_art_pk) is None

    # Artifact-backed selection: exactly one ArtifactSource row.
    art_sort_pk = SortingSelection.insert_selection(
        {**sorter_kwargs, "artifact_id": art_pk["artifact_id"]}
    )
    assert len(SortingSelection.ArtifactSource & art_sort_pk) == 1
    assert (
        SortingSelection.resolve_artifact(art_sort_pk) == art_pk["artifact_id"]
    )

    # ArtifactSource did NOT leak into the recording-source resolution.
    resolved = SortingSelection.resolve_source(art_sort_pk)
    assert resolved.kind == "recording"
    assert resolved.key == {"recording_id": populated_recording["recording_id"]}


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
    assert SpikeSortingOutput().merge_get_parent_class("CurationV2") is not None
    assert (
        SpikeSortingOutput().merge_get_parent_class("ImportedSpikeSorting")
        is not None
    )
    # And the Imported part table is reachable through the master.
    assert hasattr(SpikeSortingOutput, "ImportedSpikeSorting")


# ---------- L5-A: global-median reference (reference_mode) ------------


@pytest.mark.slow
def test_recording_make_global_median_reference(polymer_smoke_session):
    """``Recording._apply_pre_motion_preprocessing`` applies the
    global-median reference when ``reference_mode == "global_median"``.

    Every other recording test uses ``reference_mode = "none"``, so the
    ``elif reference_mode == "global_median"`` branch in recording.py has
    never been exercised. This test populates a Recording with
    ``reference_mode = "global_median"`` and pins:

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
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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

    # Switch SortGroupV2 to global-median reference and repopulate. A
    # new RecordingSelection gets minted because the cache_hash depends
    # on input data.
    SortGroupV2.update1(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "reference_mode": "global_median",
        }
    )
    assert (
        SortGroupV2
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    ).fetch1("reference_mode") == "global_median"
    # Drop the unref Recording so the new selection materializes a
    # fresh global-median recording rather than reusing the cached
    # bytes.
    (Recording & rec_pk_unref).super_delete(warn=False, force_masters=True)
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


@pytest.mark.slow
def test_recording_no_filter_preset_skips_bandpass(polymer_smoke_session):
    """The ``no_filter`` preset materializes an UNfiltered recording.

    T5 made ``bandpass_filter`` optional: the ``no_filter`` preset ships
    ``bandpass_filter=None`` and ``_apply_pre_motion_preprocessing`` must
    SKIP the bandpass step (not silently pass a wide-band that still
    filters). This populates two recordings on the same data -- one with
    the bandpassed ``default_franklab`` preset, one with ``no_filter`` --
    and pins: (1) the no_filter populate completes (the ``None`` guard
    does not crash), and (2) its traces differ materially from the
    bandpassed version, which only holds if the filter was actually
    skipped. If the ``if bandpass_filter is not None`` guard were missing
    or inverted, no_filter would still bandpass and the two recordings
    would be (nearly) identical.
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
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    common = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "team_name": "v2_test_team",
    }
    # Bandpassed reference (default_franklab ships a real 300-6000 Hz band).
    bp_pk = RecordingSelection.insert_selection(
        {**common, "preproc_params_name": "default_franklab"}
    )
    Recording.populate(bp_pk, reserve_jobs=False)
    traces_bp = Recording().get_recording(bp_pk).get_traces()

    # no_filter ships bandpass_filter=None -> the filter step is skipped.
    # Distinct preproc_params_name => distinct cache_hash => distinct row,
    # so no delete/repopulate dance is needed.
    nf_pk = RecordingSelection.insert_selection(
        {**common, "preproc_params_name": "no_filter"}
    )
    Recording.populate(nf_pk, reserve_jobs=False)
    traces_nf = Recording().get_recording(nf_pk).get_traces()

    # (1) shapes match (no_filter still references the same channels).
    assert traces_nf.shape == traces_bp.shape
    # (2) skipping the bandpass leaves low-frequency / DC content the
    # 300 Hz high-pass would have removed, so the unfiltered traces are
    # materially different from the bandpassed ones.
    delta = float(_np.abs(traces_nf - traces_bp).mean())
    assert delta > 1e-6, (
        "no_filter and default_franklab recordings are indistinguishable "
        f"(mean |diff| = {delta:.2e}); the bandpass was NOT skipped for "
        "no_filter. Check the `if bandpass_filter is not None` guard in "
        "_apply_pre_motion_preprocessing."
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
    ``recording.py:727-730`` is dead code that, if it silently
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
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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
            (RecordingSelection & {"recording_id": rid}).super_delete(
                warn=False
            )


# ---------- L5-C: ArtifactDetection.delete handles orphans ---------------


@pytest.mark.slow
def test_artifact_detection_delete_handles_orphan_source_part(
    populated_recording, monkeypatch
):
    """``ArtifactDetection.delete`` skips an orphan whose source part can no
    longer be resolved, and still deletes the master.

    An orphan master (its source-part row cascade-deleted via some upstream
    path) makes ``resolve_source`` raise ``SchemaBypassError`` -- zero source
    rows. ``ArtifactDetection.delete``'s IntervalList-cleanup loop catches
    exactly that (``except SchemaBypassError``, narrowed by Phase 1 E3), logs,
    skips that row's paired IntervalList cleanup, and lets the master delete
    proceed. We patch ``resolve_source`` to raise the real orphan exception so
    a regression that re-raised it (instead of skipping) would fail here.

    Companion ``test_artifact_detection_delete_aborts_on_unexpected_resolve_error``
    pins the other side of E3's narrowed except: a NON-orphan (unexpected)
    error must propagate and abort the delete rather than be swallowed.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # An orphan source part is exactly what resolve_source reports as
    # SchemaBypassError (zero/multiple source rows); simulate that shape.
    def _orphan_resolve(cls, key):
        raise SchemaBypassError("simulated orphan: zero source part rows")

    monkeypatch.setattr(
        ArtifactSelection,
        "resolve_source",
        classmethod(_orphan_resolve),
    )

    # The delete completes without raising; the orphan row's cleanup is
    # skipped but the master delete proceeds.
    (ArtifactDetection & art_pk).delete(safemode=False)
    assert len(ArtifactDetection & art_pk) == 0, (
        "ArtifactDetection row should be deleted even when resolve_source "
        "reports an orphan; the SchemaBypassError branch must not "
        "short-circuit the master delete."
    )


@pytest.mark.slow
def test_artifact_detection_delete_aborts_on_unexpected_resolve_error(
    populated_recording, monkeypatch
):
    """``ArtifactDetection.delete`` aborts (does NOT swallow) an UNEXPECTED
    ``resolve_source`` error, leaving the master in place.

    Phase 1 E3 narrowed the cleanup-loop except to ``SchemaBypassError`` (the
    orphan shape). Any other exception is a genuine bug: swallowing it would
    silently delete the master and orphan its IntervalList rows. The narrowed
    except must let it propagate BEFORE ``super().delete()`` runs, so the
    master survives and the operator sees the failure. ``SchemaBypassError``
    subclasses ``RuntimeError``, so a bare ``RuntimeError`` is genuinely
    outside the caught type.
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

    def _unexpected_resolve(cls, key):
        raise RuntimeError("unexpected resolve_source bug")

    monkeypatch.setattr(
        ArtifactSelection,
        "resolve_source",
        classmethod(_unexpected_resolve),
    )

    with pytest.raises(RuntimeError, match="unexpected resolve_source bug"):
        (ArtifactDetection & art_pk).delete(safemode=False)
    # Abort happened before the master delete -- the row must still exist.
    assert len(ArtifactDetection & art_pk) == 1, (
        "an unexpected resolve_source error must abort the delete before the "
        "master is removed, not silently orphan it"
    )


# ---------- L5-D: _write_units_nwb zero-unit guard (sorting layer) -------


@pytest.mark.slow
def test_write_units_nwb_handles_zero_unit_sorter(populated_recording):
    """``Sorting._write_units_nwb`` initializes an empty Units NWB
    when the sorter produces zero unit ids.

    The guard at ``sorting.py:1690-1698`` is the sorting-layer analog
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
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
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
    ``artifact.py:973`` ensures the saved interval doesn't index
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

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

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
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
    expected_n = int(((chunk1_end - t0) + (chunk2_end - gap_end)) * fs)
    assert 0.95 * expected_n <= len(written_times) <= 1.05 * expected_n, (
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


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_valid_times_respect_disjoint_gap(polymer_smoke_session):
    """End-to-end: artifact-removed valid_times never span a gap.

    Builds a disjoint Recording (two chunks separated by a wall-clock
    gap), runs ``ArtifactDetection`` with the ``none`` preset
    (detect=False), and asserts the persisted artifact ``IntervalList``
    valid_times split at the gap rather than returning one envelope
    spanning it. Subtracting/returning over a single
    ``[timestamps[0], timestamps[-1]]`` envelope (the old behavior) would
    reintroduce the gap -- inflating obs_intervals duration and letting
    sub-min_length slivers survive. The artifact-detected (detect=True)
    per-chunk subtraction is pinned by the synthetic ``_detect_artifacts``
    test in ``test_disjoint_artifact.py``.
    """
    import uuid as _uuid

    import numpy as _np

    from spyglass.common.common_interval import IntervalList
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

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
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
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    gap_mid = 0.5 * (chunk1_end + gap_end)
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_artifact_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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

    valid_times = ArtifactDetection().get_artifact_removed_intervals(art_pk)

    assert len(valid_times) >= 2, (
        "disjoint recording must yield >= 2 artifact valid intervals "
        f"(one per chunk); got {valid_times.tolist()}"
    )
    for start, end in valid_times:
        assert not (start < gap_mid < end), (
            f"artifact valid interval [{start}, {end}] spans the "
            f"inter-chunk gap (gap_mid={gap_mid})."
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
        "DataFrame missing the curation_label column joined from " "UnitLabel."
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

    expected = (Recording & populated_recording).fetch1(
        "electrical_series_path"
    )
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
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert len(df) > 0, "populated_sorting yielded zero units"
    assert (
        "obs_intervals" in df.columns
    ), "Units NWB missing per-unit obs_intervals column."
    # obs_intervals should be a non-empty (n_intervals, 2) array
    # per unit. Each row should have a valid window.
    for uid, obs in df["obs_intervals"].items():
        assert (
            obs is not None and len(obs) >= 1
        ), f"Unit {uid} has empty obs_intervals."
    assert (
        "curation_label" in df.columns
    ), "Units NWB missing curation_label placeholder column."
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
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert (
        "curation_label" in df.columns
    ), "Curated NWB missing curation_label column."
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
    # ``_build_analyzer`` runs at populate time, so the folder is a
    # precondition of this test. Treat absence as a FAILURE rather than a
    # vacuous skip: if it is missing, the populate path is broken and the
    # delete-cleanup assertion below would pass without exercising the
    # rmtree it is meant to verify.
    assert folder.exists(), (
        f"precondition: analyzer_folder {folder} should exist after the "
        "populated_sorting fixture; the populate path did not write it."
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    expected = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # Resolve this sort's artifact_id (the "none" preset writes
    # one IntervalList with the artifact-naming convention). The artifact
    # pass lives on the ArtifactSource part, not the master, so read it
    # through the resolver rather than a master column.
    artifact_id = SortingSelection.resolve_artifact(populated_sorting)
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
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
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # Use the default ``all_units`` UnitSelectionParams that ships
    # with the analysis module.
    UnitSelectionParams().insert_default()
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    actual_nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

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
        SharedArtifactGroup & {"shared_artifact_group_name": group_name}
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
    Path(__file__).resolve().parent / "fixtures" / "mearec_neuropixels_60s.nwb"
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
        for g in (SortGroupV2 & neuropixels_60s_session).fetch("sort_group_id")
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

    **NOT a per-PR CI gate.** This is a manual / nightly baseline-verification
    matrix: all parametrized cases SKIP unless ``SPIKESORTING_V2_BASELINE_ROOT``
    (or the legacy single-case ``SPIKESORTING_V2_BASELINE_DIR``) points at
    operator-captured v1 baselines. The ``pytest-v2`` workflow does not set
    either, so reviewers should not count these cases as per-PR coverage --
    they activate only under the capture workflow described in "Env vars" below.

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
    fixture_path = _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
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
                "params_schema_version": 4,
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
            for row in (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
                "KEY", "electrode_id", "bad_channel", as_dict=True
            )
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
    v1_fingerprints = {field: meta[field] for field in _fingerprint_fields}
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
    curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")

    # Pull v2 spike times in seconds via the merge dispatcher --
    # same surface a v1 consumer would hit; if v1 and v2 use the
    # same units list the per-unit arrays line up by index.
    v2_per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    v2_unit_ids = (Sorting.Unit & sort_pk).fetch("unit_id", order_by="unit_id")
    v2_spike_times = {
        int(uid): np.asarray(arr) for uid, arr in zip(v2_unit_ids, v2_per_unit)
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
        # 50%+5 budget on v2_arr.size is the assertion that catches a
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
    print(f"\n[parity] {fixture_stem} shank{sort_group_id} clusterless")
    print(
        f"  {'unit':>5} {'v1_n':>6} {'v2_n':>6} {'matched':>8} "
        f"{'unmatched_v1':>13} {'unmatched_v2':>13}"
    )
    for row in diagnostic_rows:
        uid, v1n, v2n, nm, um1, um2 = row
        print(f"  {uid:>5} {v1n:>6} {v2n:>6} {nm:>8} " f"{um1:>13} {um2:>13}")

    if failures:
        pytest.fail("v1↔v2 spike-time divergence:\n  " + "\n  ".join(failures))


_MS4_PARITY_CASES = [("mearec_polymer_128ch_60s", sg) for sg in (0, 1, 2, 3)]


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

    fixture_path = _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
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
            for row in (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
                "KEY", "electrode_id", "bad_channel", as_dict=True
            )
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
    v2_spike_times = {i: np.asarray(arr) for i, arr in enumerate(v2_per_unit)}
    v1_spike_times_arrs = {
        int(uid): np.asarray(times) for uid, times in v1_spike_times.items()
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
        pytest.fail(
            "v1↔v2 MS4 aggregate-band divergence:\n  " + "\n  ".join(failures)
        )


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
    >= 0.5 (looser than the MS5 polymer gate's 1/2 >= 0.7 because
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
    # MS5 polymer gate (1/2 >= 0.7) because MS4 is a less-accurate
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
    detect_threshold = 5 (a MAD multiplier, not 5σ or 5 µV -- with
    ``noise_levels`` omitted SI scales the threshold by per-channel
    MAD) should catch all well-isolated spikes; higher recall bars
    risk flake from near-threshold borderline spikes (PR #4341
    algorithm change).

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

    # Pre-insert the smoke clusterless row (detect_threshold=5, a MAD
    # multiplier -- not µV; doesn't ship as a v2 default; needs
    # delete-then-insert for the same noise_levels-semantics safety
    # reason as the v1↔v2 parity test).
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
                "params_schema_version": 4,
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
        curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
        merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1(
            "merge_id"
        )
        per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
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

    if not v2_times_s_by_shank or all(
        a.size == 0 for a in v2_times_s_by_shank.values()
    ):
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
        assert (
            gt_units_table is not None
        ), f"{nwb_file_name!r} has no sidecar GT units; regenerate."
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
    # by a detect_threshold of 5 (a MAD multiplier, not 5σ); missing
    # some is expected for near-threshold planted units (especially on
    # the sparse-coverage tetrode).
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
    total_planted_spikes = int(sum(t.size for t in gt_units.values() if t.size))
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
            Electrode & {"nwb_file_name": nwb_file_name, "bad_channel": "False"}
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

    monkeypatch.setattr(ArtifactSelection, "resolve_source", classmethod(_boom))
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


# ---------- SortGroupV2 reference-mode split --------------------------------


@pytest.mark.slow
def test_sort_group_reference_mode_enforced(polymer_smoke_session):
    """SortGroupV2.insert1 enforces the reference_mode / electrode invariants.

    ``reference_mode`` is validated against the ``ReferenceMode`` Literal
    at insert time (a typo raises), ``"specific"`` requires a
    ``reference_electrode_id``, and the non-specific modes reject one. A
    valid ``"specific"`` row stores both fields; a default (mode omitted)
    row stores ``"none"`` with a NULL electrode.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    base = {"nwb_file_name": nwb_file_name, "sort_group_id": 900}

    # Typo'd mode rejected by the Literal guard (varchar, not enum).
    with pytest.raises(ValueError, match="reference_mode"):
        SortGroupV2.insert1({**base, "reference_mode": "globalmedian"})

    # specific requires an electrode id.
    with pytest.raises(ValueError, match="requires a non-null"):
        SortGroupV2.insert1({**base, "reference_mode": "specific"})

    # non-specific must not carry an electrode id.
    with pytest.raises(ValueError, match="must leave"):
        SortGroupV2.insert1(
            {
                **base,
                "reference_mode": "global_median",
                "reference_electrode_id": 3,
            }
        )

    # A valid specific row inserts and stores both fields.
    SortGroupV2.insert1(
        {**base, "reference_mode": "specific", "reference_electrode_id": 0}
    )
    stored = (SortGroupV2 & base).fetch1(
        "reference_mode", "reference_electrode_id"
    )
    assert stored == ("specific", 0)

    # A default (mode omitted) row stores "none" with NULL electrode.
    default_base = {"nwb_file_name": nwb_file_name, "sort_group_id": 901}
    SortGroupV2.insert1(default_base)
    mode, eid = (SortGroupV2 & default_base).fetch1(
        "reference_mode", "reference_electrode_id"
    )
    assert mode == "none"
    assert eid is None


@pytest.mark.slow
def test_recording_specific_reference_drops_ref_channel(polymer_smoke_session):
    """specific mode references then drops the reference electrode.

    The reference electrode is included in the channel slice so
    ``common_reference`` can subtract it, then removed -- the cached
    recording must NOT carry it, while the sort group's own electrodes
    remain. Uses a cross-group reference electrode so the dropped channel
    is unambiguously distinct from the sorted set.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )

    # Group by shank with NO reference so we can pick a reference electrode
    # from a DIFFERENT group than the one we sort.
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sg_ids = sorted(
        int(s)
        for s in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert len(sg_ids) >= 2, "need >=2 shanks to isolate a cross-group ref"
    target_sg, other_sg = sg_ids[0], sg_ids[1]
    target_elecs = [
        int(e)
        for e in (
            SortGroupV2.SortGroupElectrode
            & {**polymer_smoke_session, "sort_group_id": target_sg}
        ).fetch("electrode_id")
    ]
    ref_id = int(
        (
            SortGroupV2.SortGroupElectrode
            & {**polymer_smoke_session, "sort_group_id": other_sg}
        ).fetch("electrode_id")[0]
    )
    assert ref_id not in target_elecs

    # Re-group with a specific cross-group reference electrode.
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name,
        reference_mode="specific",
        reference_electrode_id=ref_id,
        sort_group_ids=sg_ids,
        delete_existing_entries=True,
        confirm=True,
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": target_sg,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    rec = Recording().get_recording(rec_pk)
    cached_ids = {int(c) for c in rec.get_channel_ids()}
    assert ref_id not in cached_ids, (
        "specific-mode reference electrode leaked into the cached "
        "recording; it must be dropped after referencing."
    )
    assert (
        set(target_elecs) <= cached_ids
    ), "sort-group electrodes missing from the cached recording."


@pytest.mark.slow
def test_curation_label_validation_all_paths(populated_sorting):
    """Curation labels are validated on EVERY insert path.

    Both ``CurationV2.insert_curation(labels=...)`` and a direct
    ``CurationV2.UnitLabel.insert1`` reject a typo'd label, and
    ``allow_custom_labels=True`` accepts a label outside the canonical set
    through both paths.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])

    # Path 1: insert_curation rejects a typo'd label.
    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: ["noies"]},
        )

    # Path 1 with the escape hatch: a custom label is accepted.
    pk_custom = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={target_unit: ["my_custom_tag"]},
        allow_custom_labels=True,
    )
    custom_rows = {
        r["curation_label"]
        for r in (CurationV2.UnitLabel & pk_custom).fetch(as_dict=True)
    }
    assert "my_custom_tag" in custom_rows

    # Path 2: a direct UnitLabel.insert1 rejects a typo'd label (the
    # part-table override, not only insert_curation).
    direct_row = {
        **pk_custom,
        "unit_id": target_unit,
        "curation_label": "noies",
    }
    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.UnitLabel.insert1(direct_row)

    # Path 2 with the escape hatch: the direct insert accepts a custom
    # label when allow_custom_labels=True.
    CurationV2.UnitLabel.insert1(
        {**pk_custom, "unit_id": target_unit, "curation_label": "another_tag"},
        allow_custom_labels=True,
    )
    assert (
        CurationV2.UnitLabel
        & pk_custom
        & {"unit_id": target_unit, "curation_label": "another_tag"}
    )


@pytest.mark.slow
def test_unit_label_insert_rejects_ordered_row(populated_sorting):
    """A direct ordered-sequence (tuple) UnitLabel insert is validated.

    DataJoint accepts positional/ordered rows, but the canonical-label
    guard reads ``row["curation_label"]`` -- a membership test that is
    False for a tuple, so an ordered row of
    ``(sorting_id, curation_id, unit_id, "noies")`` could slip a bogus
    label past validation, breaking the "validate on every insert path"
    guarantee. The override must reject non-mapping rows (or validate
    them) rather than silently passing them through.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={target_unit: ["mua"]}
    )

    ordered = (
        pk["sorting_id"],
        pk["curation_id"],
        target_unit,
        "noies",  # not in CurationLabel
    )
    with pytest.raises(
        ValueError, match="requires mapping|not in CurationLabel"
    ):
        CurationV2.UnitLabel.insert1(ordered)
    with pytest.raises(
        ValueError, match="requires mapping|not in CurationLabel"
    ):
        CurationV2.UnitLabel.insert([ordered])
    # allow_custom_labels=True must NOT let an ordered row bypass the
    # mapping check (the type guard precedes the custom-label opt-out).
    with pytest.raises(ValueError, match="requires mapping"):
        CurationV2.UnitLabel.insert1(ordered, allow_custom_labels=True)


@pytest.mark.slow
def test_insert_curation_rejects_scalar_string_label(populated_sorting):
    """A scalar string label value is rejected, not split per-character.

    ``labels={unit_id: "custom_tag"}`` (a bare string instead of a list
    of labels) must raise -- NOT be coerced to
    ``["c","u","s","t","o","m",...]`` and written as per-character labels.
    The bug bites hardest with ``allow_custom_labels=True``, which skips
    the per-value canonical check, so the test pins both the default and
    the escape-hatch paths.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])

    with pytest.raises(ValueError, match="must be a list of labels"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: "mua"},  # scalar string, not ["mua"]
        )
    with pytest.raises(ValueError, match="must be a list of labels"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: "custom_tag"},
            allow_custom_labels=True,
        )


# ===========================================================================
# A27: recording.py untested branches.
#
# reference_mode validation (post-T2: the integer sentinel was replaced by a
# ``reference_mode`` varchar validated against the ReferenceMode Literal),
# the all-shanks-filtered guard, the omit_ref no-op under the default mode,
# additive inserts, length-mismatch, and the empty-match guard on the
# electrode-table-column constructor. The RecordingSelection duplicate guard
# (test_recording_selection_raises_on_duplicate_logical_identity, len==2) and
# _electrode_group_sort_key (test_electrode_group_sort_key_tolerates_non_numeric)
# are already covered, so they are not re-added here.
# ===========================================================================


@pytest.mark.usefixtures("dj_conn")
def test_sort_group_rejects_invalid_reference_mode():
    """A27/T2: ``SortGroupV2.insert1`` rejects an unknown ``reference_mode``.

    The integer ``sort_reference_electrode_id`` sentinel was replaced (T2) by
    a ``reference_mode`` varchar validated against the ``ReferenceMode``
    Literal at the insert boundary. An invalid mode (here ``"banana"``)
    raises before any DB write -- the varchar's typo guard standing in for a
    MySQL enum. The message names the valid modes.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    with pytest.raises(ValueError, match="ReferenceMode"):
        SortGroupV2().insert1(
            {
                "nwb_file_name": "a27_irrelevant_.nwb",
                "sort_group_id": 0,
                "reference_mode": "banana",
            }
        )


@pytest.mark.usefixtures("dj_conn")
def test_sort_group_reference_electrode_id_consistency():
    """A27/T2: ``reference_electrode_id`` is non-null iff mode=='specific'.

    The split's second invariant: a 'specific' reference needs a channel to
    subtract, and a non-specific mode must not carry a stray channel id the
    runtime would silently ignore. Both directions raise at insert.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    base = {"nwb_file_name": "a27_irrelevant_.nwb", "sort_group_id": 0}
    # 'specific' without a channel id.
    with pytest.raises(ValueError, match="requires a non-null"):
        SortGroupV2().insert1(
            {
                **base,
                "reference_mode": "specific",
                "reference_electrode_id": None,
            }
        )
    # non-specific WITH a stray channel id.
    with pytest.raises(ValueError, match="must leave"):
        SortGroupV2().insert1(
            {
                **base,
                "reference_mode": "global_median",
                "reference_electrode_id": 3,
            }
        )


@pytest.mark.slow
def test_set_group_by_shank_all_shanks_filtered_raises(polymer_smoke_session):
    """A27: when every shank is filtered out, ``set_group_by_shank`` raises.

    The smoke fixture's 4 shanks all live in one electrode group ("0"), so
    ``omit_ref_electrode_group=True`` with ``reference_mode='specific'`` and a
    reference electrode in that group skips ALL shanks (the skip is keyed on
    the electrode group, and the whole probe is one group). No sort group
    survives -> ValueError naming "no sort groups".
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="no sort groups produced"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            omit_ref_electrode_group=True,
            reference_mode="specific",
            reference_electrode_id=0,
        )


@pytest.mark.slow
def test_set_group_by_shank_omit_ref_noop_under_default_mode(
    polymer_smoke_session,
):
    """A27/T2: ``omit_ref_electrode_group=True`` is a no-op under the default
    ``reference_mode='none'``.

    Post-T2 the omit-reference-group skip is gated on
    ``reference_mode == 'specific'`` (a 'none'/'global_median' reference names
    no electrode to exclude). So with the default mode the flag drops nothing:
    all 4 shanks become sort groups and the returned skip list has no
    reference-group entries. (Pre-T2 this branch keyed off the integer
    sentinel; the assertion now tracks the live reference_mode source.)
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    skipped = SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name,
        omit_ref_electrode_group=True,  # no-op: mode defaults to "none"
    )

    assert len(SortGroupV2 & polymer_smoke_session) == 4, (
        "omit_ref_electrode_group must not drop groups under reference_mode="
        "'none'"
    )
    assert not any(
        s.get("reason") == "reference_electrode_group" for s in skipped
    ), "no group should be skipped for the reference-electrode reason"


@pytest.mark.slow
def test_set_group_by_shank_additive_insert_with_explicit_ids(
    polymer_smoke_session,
):
    """A27: explicit non-overlapping ``sort_group_ids`` opt into an additive
    insert that coexists with prior rows.

    The default rerun refuses to silently extend (covered elsewhere); passing
    explicit non-overlapping ids is the opt-in. After grouping once (ids
    0-3), a second call with ids 10-13 leaves all eight groups present.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    first_ids = set(
        int(i)
        for i in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert first_ids == {0, 1, 2, 3}

    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, sort_group_ids=[10, 11, 12, 13]
    )
    all_ids = set(
        int(i)
        for i in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert all_ids == {
        0,
        1,
        2,
        3,
        10,
        11,
        12,
        13,
    }, "additive insert with explicit ids must coexist with prior rows"


@pytest.mark.slow
def test_set_group_by_shank_length_mismatch_raises(polymer_smoke_session):
    """A27: ``sort_group_ids`` whose length differs from the derived group
    count raises.

    The fixture derives 4 groups from shank metadata; passing only two ids
    is a mismatch the helper rejects before any insert.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="Lengths must match"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, sort_group_ids=[0, 1]
        )


@pytest.mark.slow
def test_set_group_by_electrode_table_column_empty_match_raises(
    polymer_smoke_session,
):
    """A27: a ``groups`` sublist matching zero electrodes raises.

    ``set_group_by_electrode_table_column`` with a value that no electrode
    carries produces an empty group, which is a user error (typo'd id) the
    helper rejects with a message naming the offending sort group.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="matched no electrodes"):
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb_file_name,
            column="electrode_id",
            groups=[[10_000_000]],  # no electrode has this id
        )


@pytest.mark.slow
def test_recording_multi_interval_saved_times_use_concat_path(
    polymer_smoke_session,
):
    """A27: a multi-interval (disjoint) selection derives ``saved_times`` from
    the gap-EXCLUDED concat path, not the gap-spanning override.

    For ``n_selected_intervals > 1`` the recording is built with
    ``concatenate_recordings(ignore_times=True)``, so ``saved_times`` is the
    0-based concat times whose span excludes the inter-segment gap. The row's
    ``duration_s`` must therefore equal the SUM of the kept-interval durations
    (gap-excluded), clearly shorter than the wall-clock envelope
    (last_end - first_start). This is the persisted half of the C1 guard
    rationale.
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
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
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
    total = raw_end - raw_start
    # Each 0.3*total segment must clear min_segment_length (1.0 s), so the
    # fixture needs >= ~3.5 s of raw coverage.
    assert total >= 3.5, f"fixture too short ({total}s) for multi-interval test"

    # Two disjoint segments, both well within raw coverage, separated by a
    # real gap of ~0.3*total.
    seg1 = [raw_start, raw_start + 0.3 * total]
    seg2 = [raw_start + 0.6 * total, raw_start + 0.9 * total]
    gap_excluded = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0])
    envelope = seg2[1] - seg1[0]

    interval_name = "v2_a27_multi_valid"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
            "valid_times": _np.asarray([seg1, seg2]),
            "pipeline": "v2_a27_multi",
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
    try:
        Recording.populate(pk, reserve_jobs=False)
        duration_s = float((Recording & pk).fetch1("duration_s"))
        # Concat path: duration ~= sum of kept durations (gap-excluded).
        assert abs(duration_s - gap_excluded) < 0.1 * total, (
            f"duration_s={duration_s} not close to the gap-excluded sum "
            f"{gap_excluded} (concat path); total={total}"
        )
        # And clearly NOT the gap-spanning wall-clock envelope.
        assert duration_s < envelope - 0.1 * total, (
            f"duration_s={duration_s} looks like the gap-SPANNING envelope "
            f"{envelope}; the override path was used instead of concat"
        )
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_recording_fresh_write_cleanup_unlinks_staged_file(
    polymer_smoke_session, monkeypatch
):
    """A27: a failure AFTER the fresh ``_write_nwb_artifact`` unlinks the
    staged file (no half-written artifact outlives a failed compute).

    On the fresh-write path (``existing_analysis_file_name is None``), if a
    later metadata step raises after the file is on disk, the staged file is
    unlinked before the exception propagates. We wrap ``_write_nwb_artifact``
    to record the staged name, then make the post-write
    ``_get_recording_timestamps`` raise; afterwards the staged file must be
    gone and no Recording row may exist.
    """
    from spyglass.common import IntervalList  # noqa: F401
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import recording as rec_mod
    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
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

    captured = {}
    real_write = Recording._write_nwb_artifact

    def _capture_write(*args, **kwargs):
        result = real_write(*args, **kwargs)
        captured["analysis_file_name"] = result[0]
        return result

    real_ts = utils_mod._get_recording_timestamps

    def _raise_after_write(*args, **kwargs):
        # Earlier calls (restrict / repair) succeed; only the post-write
        # saved-times derivation raises so we exercise the fresh-write
        # cleanup branch rather than failing before the file is staged.
        if "analysis_file_name" in captured:
            raise RuntimeError("staged-file cleanup probe: post-write boom")
        return real_ts(*args, **kwargs)

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture_write)
    )
    monkeypatch.setattr(
        utils_mod, "_get_recording_timestamps", _raise_after_write
    )
    # recording.py imports the symbol at call time from utils, so patching
    # the utils module is sufficient; guard against a module-level rebind.
    monkeypatch.setattr(
        rec_mod, "_get_recording_timestamps", _raise_after_write, raising=False
    )

    try:
        with pytest.raises(Exception, match="post-write boom"):
            Recording.populate(pk, reserve_jobs=False)
        assert (
            "analysis_file_name" in captured
        ), "precondition: _write_nwb_artifact must have run and staged a file"
        staged = captured["analysis_file_name"]
        abs_path = AnalysisNwbfile.get_abs_path(staged)
        assert not Path(
            abs_path
        ).exists(), (
            f"fresh-write cleanup did not unlink the staged file {staged!r}"
        )
        assert not (Recording & pk), "no Recording row may be inserted"
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_recording_rebuild_path_keeps_existing_file_on_failure(
    populated_sorting, monkeypatch
):
    """A27 (complement): a post-write failure on the REBUILD path
    (``existing_analysis_file_name`` set) does NOT unlink the file.

    The fresh-write path unlinks a staged file on failure; the rebuild path
    must not, because there the file IS the canonical cache (a mid-write
    failure surfaces via the caller's hash-mismatch check, not by destroying
    the only copy). We drive ``_compute_recording_artifact`` directly with an
    ``existing_analysis_file_name`` and the same post-write
    ``_get_recording_timestamps`` failure, then assert the file survives. This
    pins the false branch of the ``existing_analysis_file_name is None`` guard.

    Uses a freshly created AnalysisNwbfile as the stand-in cache so the shared
    fixture's real recording file is never touched.
    """
    from spyglass.common.common_nwbfile import (
        AnalysisNwbfile,
        Nwbfile,
    )
    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    fetched = Recording().make_fetch({"recording_id": recording_id})
    nwb_file_name = fetched.sel["nwb_file_name"]

    # A real on-disk file standing in for the canonical cache.
    existing = AnalysisNwbfile().create(nwb_file_name)
    existing_abs = AnalysisNwbfile.get_abs_path(existing)
    assert Path(existing_abs).exists(), "precondition: cache file created"

    captured = {}
    real_write = Recording._write_nwb_artifact
    real_ts = utils_mod._get_recording_timestamps

    def _capture_write(*args, **kwargs):
        result = real_write(*args, **kwargs)
        captured["written"] = True
        return result

    def _raise_after_write(*args, **kwargs):
        if "written" in captured:
            raise RuntimeError("rebuild post-write boom")
        return real_ts(*args, **kwargs)

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture_write)
    )
    monkeypatch.setattr(
        utils_mod, "_get_recording_timestamps", _raise_after_write
    )

    raw_path = Nwbfile().get_abs_path(nwb_file_name)
    try:
        with pytest.raises(Exception, match="rebuild post-write boom"):
            Recording()._compute_recording_artifact(
                raw_path=raw_path,
                nwb_file_name=nwb_file_name,
                interval_list_name=fetched.sel["interval_list_name"],
                channel_ids=fetched.channel_ids,
                reference_mode=fetched.reference_mode,
                reference_electrode_id=fetched.reference_electrode_id,
                sort_valid_times=fetched.sort_valid_times,
                raw_valid_times=fetched.raw_valid_times,
                preproc_validated=fetched.preproc_validated,
                probe_types=fetched.probe_types,
                electrode_group_names=fetched.electrode_group_names,
                existing_analysis_file_name=existing,  # REBUILD path
            )
        assert (
            "written" in captured
        ), "precondition: _write_nwb_artifact must have run before the failure"
        # The rebuild path must NOT unlink the canonical cache file.
        assert Path(existing_abs).exists(), (
            "rebuild path unlinked the existing cache file -- it must only "
            "unlink on the fresh-write (existing is None) path"
        )
    finally:
        Path(existing_abs).unlink(missing_ok=True)
        # The create() call may have registered an AnalysisNwbfile row.
        if AnalysisNwbfile & {"analysis_file_name": existing}:
            (AnalysisNwbfile & {"analysis_file_name": existing}).delete(
                safemode=False
            )


# ===========================================================================
# A28: curation.py untested branches.
#
# Invalid metrics_source, the POST-E5 idempotent-root guard (now LIVE -- E5
# merged), across-group merge overlap, the concat-source ambiguity guard on
# the CurationV2 accessor, the missing-sorting_id guard, get_merged_sorting
# early returns, the curation_label column-add gate, the empty-part guards,
# and the singleton merge-group gate. The include_labels filter (V5) and the
# scalar-string label guard are already covered, so they are not re-added.
# ===========================================================================


def test_insert_curation_rejects_invalid_metrics_source(populated_sorting):
    """A28: an unknown ``metrics_source`` raises naming the valid options.

    ``metrics_source`` is coerced through the ``MetricsSource`` enum so a typo
    fails at the Python boundary (with the valid set) rather than at the
    DataJoint enum-mismatch layer.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    with pytest.raises(ValueError, match="MetricsSource"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            metrics_source="not_a_real_source",
        )


def test_insert_curation_idempotent_root_rejects_nondefault_args(
    populated_sorting_with_curation, populated_sorting
):
    """A28/E5 (LIVE): a second root insert with non-default args raises.

    Post-E5, returning the existing root while silently dropping the caller's
    new labels/merge_groups/description is a footgun, so it raises unless
    ``reuse_existing=True``. The fixture already inserted a root; a second
    call passing a description must raise, and ``reuse_existing=True`` must
    return the existing key instead.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    existing = populated_sorting_with_curation
    # Non-default args + no reuse -> raises (E5 footgun guard).
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            description="a second description that would be ignored",
        )
    # reuse_existing=True returns the existing root key unchanged.
    reused = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        description="ignored but explicitly tolerated",
        reuse_existing=True,
    )
    assert reused["curation_id"] == existing["curation_id"]


def test_insert_curation_rejects_merge_group_overlap(populated_sorting):
    """A28: a unit appearing in two merge groups raises.

    A unit can belong to at most one merge group; an overlap is a user error
    the validator rejects (the overlap check runs against ``Sorting.Unit``
    BEFORE any NWB staging). The MEArec smoke sort yields a single unit, so a
    second ``Sorting.Unit`` row is planted (electrode FK copied from the real
    unit) purely to satisfy the membership check -- the raise fires during
    validation, before any spike train is read, and the planted row is removed
    in the finally.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    existing = [
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    ]
    real_unit = (Sorting.Unit & populated_sorting).fetch(as_dict=True)[0]
    u0 = int(real_unit["unit_id"])
    # ``max(existing) + 1`` guarantees a non-colliding planted id even if the
    # sorter ever returns contiguous unit ids -- ``u0 + 1`` would duplicate an
    # existing Sorting.Unit row and fail the insert before the overlap
    # validator (the actual unit under test) ever runs.
    u1 = max(existing) + 1
    planted = {**real_unit, "unit_id": u1}
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.Unit.insert1(planted, allow_direct_insert=True)
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        # u0 (and u1) appear in both groups -> overlap.
        with pytest.raises(ValueError, match="overlap"):
            CurationV2.insert_curation(
                sorting_key=populated_sorting,
                merge_groups=[[u0, u1], [u1, u0]],
                apply_merge=True,
            )
    finally:
        (Sorting.Unit & {**populated_sorting, "unit_id": u1}).delete_quick()


def test_insert_curation_rejects_singleton_merge_group(populated_sorting):
    """A28: a singleton merge group is rejected upstream (layered defense).

    A single-unit "merge group" is a likely typo (v1 silently renamed it);
    v2 raises at the >=2-members gate before ``next_merged_id`` is reached, so
    the singleton never produces a spurious fresh id.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    uid = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])
    with pytest.raises(ValueError, match="at least 2"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            merge_groups=[[uid]],
            apply_merge=True,
        )


def test_insert_curation_rejects_missing_sorting_id():
    """A28: a ``sorting_id`` not in ``Sorting`` raises a clear ValueError.

    Translates what would be a raw FK IntegrityError into a "populate Sorting
    first" message.
    """
    import uuid

    from spyglass.spikesorting.v2.curation import CurationV2

    with pytest.raises(ValueError, match="not in Sorting"):
        CurationV2.insert_curation(sorting_key={"sorting_id": uuid.uuid4()})


@pytest.mark.usefixtures("dj_conn")
def test_curation_get_unit_brain_regions_concat_raises():
    """A28: the CurationV2 accessor raises ``ConcatBrainRegionAmbiguousError``
    for a concat-backed sorting (mirror of the Sorting accessor).

    The guard reads the sorting's source before touching ``CurationV2.Unit``,
    so a concat ``SortingSelection`` (planted via the FK-checks-off bypass) is
    enough to exercise the raise.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ConcatBrainRegionAmbiguousError):
            CurationV2().get_unit_brain_regions(
                {"sorting_id": sid, "curation_id": 0},
                allow_anchor_member=False,
            )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_curation_get_unit_brain_regions_concat_anchor_member_df(
    populated_sorting,
):
    """A28: ``CurationV2.get_unit_brain_regions(..., allow_anchor_member=True)``
    returns the ``unit_brain_region_df`` DataFrame labeled ``anchor_member``.

    The opt-in path returns the DataFrame (NOT a ``SourceResolution``), same
    shape as the ``Sorting`` accessor. Non-vacuous: a ``CurationV2.Unit`` row
    is planted with an Electrode FK copied from the populated fixture so the
    BrainRegion join yields a real row carrying
    ``region_resolution == 'anchor_member'``.
    """
    import datetime as dt
    import uuid

    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import SourceResolution

    template_unit = (Sorting.Unit & populated_sorting).fetch(as_dict=True)[0]
    # CurationV2.Unit carries the same Electrode FK + amplitude/spike columns
    # as Sorting.Unit; drop the Sorting PK and re-key onto the planted curation.
    unit_fields = {k: v for k, v in template_unit.items() if k != "sorting_id"}

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a28_concat_fake.nwb",
                "object_id": "a28-concat-object-id",
                "analyzer_folder": "/nonexistent/a28_concat",
                "n_units": 1,
                "time_of_sort": dt.datetime(2020, 1, 1),
            },
            allow_direct_insert=True,
        )
        CurationV2.insert1(
            {
                "sorting_id": sid,
                "curation_id": 0,
                "analysis_file_name": "a28_concat_curation_fake.nwb",
                "object_id": "a28-concat-curation-object-id",
                "description": "a28 concat anchor probe",
            },
            allow_direct_insert=True,
        )
        CurationV2.Unit.insert1(
            {**unit_fields, "sorting_id": sid, "curation_id": 0, "unit_id": 0},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        result = CurationV2().get_unit_brain_regions(
            {"sorting_id": sid, "curation_id": 0}, allow_anchor_member=True
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, SourceResolution)
        assert "region_resolution" in result.columns
        assert len(result) == 1, "anchor-member df should carry the one unit"
        assert (result["region_resolution"] == "anchor_member").all()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (CurationV2.Unit & {"sorting_id": sid}).delete_quick()
            (CurationV2 & {"sorting_id": sid}).delete_quick()
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_get_merged_sorting_returns_base_when_no_merges(
    populated_sorting_with_curation,
):
    """A28: ``get_merged_sorting`` returns the base sorting unchanged when no
    merge group has more than one contributor.

    The fixture's root curation has no merges, so the lazy-merge path
    short-circuits and the returned sorting carries exactly the base
    unit_ids.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    base = CurationV2().get_sorting(key)
    merged = CurationV2().get_merged_sorting(key)
    assert list(merged.unit_ids) == list(base.unit_ids)


def test_get_merged_sorting_returns_base_when_merges_applied(
    populated_sorting_with_curation,
):
    """A28: ``get_merged_sorting`` returns the base verbatim when
    ``merges_applied`` is set (the base is already merged at insert).

    The MEArec smoke sort has a single unit, so a real 2-unit merge cannot be
    built; the short-circuit is keyed solely on the ``merges_applied`` flag,
    so we flip it on the root curation and assert the accessor returns the
    base sorting without attempting any lazy re-merge over absorbed
    contributors. (Function-scoped fixture; the flag flip dies with it.)
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    CurationV2.update1({**key, "merges_applied": 1})

    base = CurationV2().get_sorting(key)
    merged = CurationV2().get_merged_sorting(key)
    # merges_applied short-circuit: identical unit set, no re-merge attempt.
    assert list(merged.unit_ids) == list(base.unit_ids)


def test_insert_curation_empty_labels_skip_unit_label_insert(
    populated_sorting_with_curation,
):
    """A28: with empty labels the ``UnitLabel`` insert is skipped, while the
    ``MergeGroup`` self-entries are still written.

    The ``if unit_label_rows:`` / ``if merge_group_rows:`` guards skip a
    zero-row batch insert. With ``labels={}`` ``unit_label_rows`` is empty so
    ``UnitLabel`` stays empty; but ``merge_group_rows`` is NOT empty even
    without merges -- every ``CurationV2.Unit`` row gets a 1-element
    self-entry so ``Unit * MergeGroup`` preserves all units. This pins both
    guard branches: one skipped, one taken.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    n_units = len(CurationV2.Unit & key)
    assert n_units >= 1, "fixture curation must have >= 1 unit"
    # unit_label_rows empty -> guard skipped the UnitLabel insert.
    assert len(CurationV2.UnitLabel & key) == 0
    # merge_group_rows non-empty -> one self-entry per unit (guard taken).
    assert len(CurationV2.MergeGroup & key) == n_units


def test_curation_label_column_added_only_with_nonempty_labels(
    populated_sorting,
):
    """A28: the units NWB carries a ``curation_label`` column iff at least one
    unit has a non-empty label.

    Adding the column for an all-empty curation would trip pynwb's empty-list
    dtype inference, so it is gated on ``any(labels)``. We inspect the staged
    units NWB column set directly for both cases.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    def _units_colnames(curation_key):
        analysis_file_name = (CurationV2 & curation_key).fetch1(
            "analysis_file_name"
        )
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            return list(nwbf.units.colnames) if nwbf.units is not None else []

    # All-empty labels: no curation_label column.
    _clear_curations(populated_sorting)
    empty_key = CurationV2.insert_curation(sorting_key=populated_sorting)
    assert "curation_label" not in _units_colnames(empty_key)

    # At least one non-empty label: column present.
    _clear_curations(populated_sorting)
    uid = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])
    labeled_key = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={uid: ["mua"]}
    )
    try:
        assert "curation_label" in _units_colnames(labeled_key)
    finally:
        _clear_curations(populated_sorting)


# ===========================================================================
# A29: pipeline.py untested branches.
#
# list_presets enumeration, run_v2_pipeline idempotency (existing-root
# short-circuit), and preset -> manifest wiring. The franklab MS4 preset is
# exercised end-to-end where MS4 is runnable (it ships in installed_sorters()
# but its runtime is unavailable in the SI 0.104 image, so that test
# self-skips); a runnable MS5 substitute pins the wiring unconditionally.
# ===========================================================================


def _prepare_pipeline_session(session):
    """initialize defaults + team + a single sort group for the session.

    Returns ``(nwb_file_name, sort_group_id, team_name)`` ready for
    ``run_v2_pipeline``.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = session["nwb_file_name"]
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
    )
    return nwb_file_name, sort_group_id, "v2_test_team"


@pytest.mark.usefixtures("dj_conn")
def test_list_presets_enumerates_all_presets():
    """A29: ``list_presets()`` returns exactly the names in ``_PRESETS``.

    Behavioral (not a signature check): the helper must enumerate every
    registered preset so a notebook user can discover them. A preset added to
    ``_PRESETS`` but missing from ``list_presets()`` (or vice versa) fails
    here.
    """
    from spyglass.spikesorting.v2.pipeline import _PRESETS, list_presets

    presets = list_presets()
    assert set(presets) == set(
        _PRESETS
    ), "list_presets() must enumerate exactly the registered _PRESETS keys"
    # The three shipped presets are present (guards an accidental rename).
    for name in (
        "franklab_tetrode_mountainsort4",
        "franklab_tetrode_mountainsort5",
        "franklab_tetrode_clusterless_thresholder",
    ):
        assert name in presets


@pytest.mark.slow
def test_run_v2_pipeline_idempotent_existing_root(polymer_smoke_session):
    """A29: re-running ``run_v2_pipeline`` returns the same curation via the
    existing-root short-circuit, and a child curation does not divert it.

    The orchestrator looks for a root (``parent_curation_id=-1``) curation and
    returns it rather than staging a duplicate. The clusterless preset is the
    cheapest runnable path; require_units=False so a quiet-shank zero-unit
    result still yields a (stable) curation row.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    common = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        preset="franklab_tetrode_clusterless_thresholder",
    )
    try:
        first = run_v2_pipeline(**common)
        second = run_v2_pipeline(**common)
        assert (
            second["curation_id"] == first["curation_id"]
        ), "second run must return the existing root curation, not a duplicate"
        assert second["merge_id"] == first["merge_id"]

        # A child curation (parent != -1) must NOT divert the short-circuit.
        CurationV2.insert_curation(
            sorting_key={"sorting_id": first["sorting_id"]},
            parent_curation_id=first["curation_id"],
            description="child curation",
        )
        third = run_v2_pipeline(**common)
        assert (
            third["curation_id"] == first["curation_id"]
        ), "a non-root child must not divert the root short-circuit"
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_run_v2_pipeline_preset_wiring_to_manifest(polymer_smoke_session):
    """A29: the chosen preset's sorter is the one recorded in the manifest's
    SortingSelection.

    Runs the MS5 preset (a runnable stand-in for the MS4 preset, whose runtime
    is unavailable here) and asserts the manifest's ``sorting_id`` resolves to
    a SortingSelection whose ``sorter`` / ``sorter_params_name`` match the
    preset bundle -- the preset -> Lookup-row -> manifest wiring.
    """
    from spyglass.spikesorting.v2.pipeline import _PRESETS, run_v2_pipeline
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    preset_name = "franklab_tetrode_mountainsort5"
    try:
        manifest = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name=team_name,
            preset=preset_name,
        )
        assert manifest["preset"] == preset_name
        sel = (
            SortingSelection & {"sorting_id": manifest["sorting_id"]}
        ).fetch1()
        bundle = _PRESETS[preset_name]
        assert sel["sorter"] == bundle.sorter
        assert sel["sorter_params_name"] == bundle.sorter_params_name
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_run_v2_pipeline_mountainsort4_preset(polymer_smoke_session):
    """A29: the franklab MS4 preset runs end-to-end where MS4 is runnable.

    ``mountainsort4`` appears in ``installed_sorters()`` but its runtime is not
    available in the SI 0.104 test image, so this self-skips on the sorter
    failure. Where MS4 is runnable it asserts the manifest carries the MS4
    sorter wiring.
    """
    from spikeinterface.sorters.utils import SpikeSortingError

    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    try:
        try:
            manifest = run_v2_pipeline(
                nwb_file_name=nwb_file_name,
                sort_group_id=sort_group_id,
                interval_list_name="raw data valid times",
                team_name=team_name,
                preset="franklab_tetrode_mountainsort4",
            )
        except SpikeSortingError as exc:
            # Narrow to the sorter-RUNTIME failure only: mountainsort4 ships in
            # installed_sorters() but its runtime is unavailable in some
            # images. Preset-lookup / manifest / DB / wiring regressions raise
            # other exception types and must FAIL, not skip.
            pytest.skip(f"mountainsort4 runtime not available: {exc!r}")
        sel = (
            SortingSelection & {"sorting_id": manifest["sorting_id"]}
        ).fetch1()
        assert sel["sorter"] == "mountainsort4"
        assert (
            sel["sorter_params_name"]
            == "franklab_tetrode_hippocampus_30kHz_ms4"
        )
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
@pytest.mark.integration
def test_disjoint_multi_gap_readback_and_artifact(
    polymer_smoke_session, monkeypatch
):
    """Spike readback + artifact valid_times are gap-correct across MULTIPLE
    gaps (3 chunks, 2 gaps), not just the first.

    Every other disjoint DB test is single-gap (two chunks). This builds a
    three-chunk recording and asserts: ``_base_intervals_from_timestamps``
    (exercised via the populated timeline) yields one valid interval per
    chunk -- none spanning either gap -- and ``get_sorting`` recovers a
    spike planted in EACH chunk exactly (the per-chunk ``searchsorted``
    readback must stay correct past the second gap, not only the first).
    Closes review "not-checked" item #6.
    """
    import uuid as _uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
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
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    # Three 1.05 s chunks separated by two 0.25 s gaps (3.65 s total).
    if (t_end - t0) < 3.7:
        pytest.skip(
            f"smoke fixture window {t_end - t0:.2f}s too short for a "
            "3-chunk (2-gap) disjoint test (need >= 3.7s)."
        )
    c1 = (t0, t0 + 1.05)
    c2 = (t0 + 1.30, t0 + 2.35)
    c3 = (t0 + 2.60, t0 + 3.65)
    disjoint_times = _np.array([list(c1), list(c2), list(c3)])

    disjoint_name = f"v2_multigap_{_uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_multi_gap_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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

    # Artifact valid_times: detect=False ("none") returns the recorded
    # window(s) split per chunk -> exactly 3 intervals, none spanning a gap.
    interval_name = f"artifact_{art_pk['artifact_id']}"
    saved = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch1("valid_times")
    assert saved.shape == (3, 2), (
        f"expected one valid interval per chunk (3, 2); got {saved.shape}. "
        "_base_intervals_from_timestamps mis-split a multi-gap timeline."
    )
    gap1_mid = 0.5 * (c1[1] + c2[0])
    gap2_mid = 0.5 * (c2[1] + c3[0])
    for start, end in saved:
        assert not (start < gap1_mid < end), "valid interval spans gap 1"
        assert not (start < gap2_mid < end), "valid interval spans gap 2"

    # Plant one spike in EACH chunk, recover the boundary frames from the
    # populated timeline, then assert get_sorting reads all three back
    # exactly (past BOTH gaps).
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_id": art_pk["artifact_id"],
        }
    )
    rec = Recording().get_recording(rec_pk)
    times = _np.asarray(rec.get_times())
    fs = rec.get_sampling_frequency()
    gaps = _np.flatnonzero(_np.diff(times) > 1.5 / fs)
    assert len(gaps) == 2, f"expected two gaps, found {len(gaps)}"
    k1, k2 = int(gaps[0]), int(gaps[1])  # last frame of chunks 1, 2
    planted_frames = _np.sort(
        _np.array([50, k1 + 10, k2 + 10], dtype=_np.int64)
    )
    # Sanity: one frame in each chunk.
    assert 50 < k1 and k1 + 10 < k2 and k2 + 10 < times.size

    def _planted_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        labels = _np.zeros(planted_frames.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[planted_frames],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    si_sorting = Sorting().get_sorting(sort_pk)
    frames = _np.sort(_np.asarray(si_sorting.get_unit_spike_train(unit_id=0)))
    _np.testing.assert_array_equal(
        frames,
        planted_frames,
        err_msg=(
            "get_sorting did not recover all planted frames across two "
            f"gaps. got={frames.tolist()}, planted={planted_frames.tolist()}"
        ),
    )


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_multi_member_union(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """Two-member ``SharedArtifactGroup``: the union scan sees an artifact
    on ONE member's channels, and every member shares the written times.

    Every existing shared-group populate test has a single member, so the
    ``si.aggregate_channels`` union-scan branch of ``make_compute`` is
    untested. Here two time-aligned members are grouped; a supra-threshold
    transient is planted on member A's channels only, member B is clean.
    With ``proportion_above_thresh=0.4`` over the 8-channel union, A's 4
    channels alone exceed the required count, so the union scan removes the
    artifact window -- a per-member scan of the CLEAN member B would not.
    The single session writes one shared ``IntervalList`` row that every
    member resolves to, so all members see identical artifact-removed times.

    The two member recordings are loaded as synthetic SI recordings via a
    monkeypatched ``Recording.get_recording`` (planting a real artifact in
    fixture data is not otherwise possible); the DB rows / FK chain are
    real.
    """
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.utils import artifact_interval_list_name

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    fs = 30000.0
    n_samples = 90000  # 3 s
    n_ch = 4
    art_lo, art_hi = 45000, 45050  # transient at 1.5 s

    def _synth_recording(with_artifact):
        traces = _np.zeros((n_samples, n_ch), dtype=_np.int16)
        if with_artifact:
            traces[art_lo:art_hi, :] = 5000
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_channel_gains([1.0] * n_ch)  # µV == counts
        rec.set_channel_offsets([0.0] * n_ch)
        return rec

    # Second Recording row in the SAME session (a distinct recording_id to
    # group). Its loaded SI object is replaced below; only the DB row /
    # FK matters here.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_union_window"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": _np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_union_test",
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

    rid_a = populated_recording["recording_id"]  # member A: artifact
    rid_b = second_rec_pk["recording_id"]  # member B: clean

    def _fake_get_recording(self, key):
        return _synth_recording(
            with_artifact=str(key["recording_id"]) == str(rid_a)
        )

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    # Custom param: detect with proportion 0.4 so 4 of 8 union channels
    # (member A's) suffice to flag a frame; min_length 0.1 s so both
    # post-split halves survive.
    params_name = "phase4_union_test"
    ArtifactDetectionParameters.insert1(
        {
            "artifact_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_thresh_uV=1000.0,
                zscore_thresh=None,
                proportion_above_thresh=0.4,
                removal_window_ms=1.0,
                join_window_ms=1.0,
                min_length_s=0.1,
            ).model_dump(),
            "params_schema_version": 2,
        },
        skip_duplicates=True,
    )

    group_name = "v2_multi_member_union_group"

    def _clear_shared_group():
        # Master-first cleanup: drop any ArtifactDetection +
        # ArtifactSelection whose source part references this group BEFORE
        # the group itself. An interrupted prior run can leave a
        # SharedArtifactGroupSource part whose master must go first;
        # deleting the group ahead of it would orphan/block on that part.
        stale_art_ids = (
            ArtifactSelection.SharedArtifactGroupSource
            & {"shared_artifact_group_name": group_name}
        ).fetch("artifact_id")
        for aid in stale_art_ids:
            (ArtifactDetection & {"artifact_id": aid}).delete(safemode=False)
            (ArtifactSelection & {"artifact_id": aid}).super_delete(warn=False)
        (
            SharedArtifactGroup & {"shared_artifact_group_name": group_name}
        ).super_delete(warn=False)

    _clear_shared_group()
    try:
        SharedArtifactGroup.insert_group(
            group_name,
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
        assert (
            len(
                SharedArtifactGroup.Member
                & {"shared_artifact_group_name": group_name}
            )
            == 2
        )

        art_pk = ArtifactSelection.insert_selection(
            {
                "shared_artifact_group_name": group_name,
                "artifact_params_name": params_name,
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)

        interval_name = artifact_interval_list_name(art_pk["artifact_id"])
        rows = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).fetch(as_dict=True)
        # Single session -> one shared row covering every member.
        assert (
            len(rows) == 1
        ), f"expected one shared IntervalList row; got {len(rows)}"
        valid_times = rows[0]["valid_times"]

        # The union scan removed member A's artifact: the single base chunk
        # is split into exactly two valid intervals around the 1.5 s
        # transient, and no valid interval contains it. A clean-only
        # (member B) scan would leave one full-window interval instead.
        art_mid = 0.5 * (art_lo + art_hi) / fs  # ~1.5 s
        assert valid_times.shape == (2, 2), (
            f"union scan should split into 2 intervals at the artifact; got "
            f"valid_times={valid_times.tolist()} (shape {valid_times.shape}). "
            "A per-member scan of the clean member would leave a single "
            "full-window interval -- the union channels were not combined."
        )
        for start, end in valid_times:
            assert not (start < art_mid < end), (
                f"a valid interval [{start}, {end}] spans the artifact at "
                f"{art_mid}s; the union scan did not remove it."
            )
        # Coverage survives on both sides of the artifact.
        assert any(s <= 0.5 <= e for s, e in valid_times), "pre-artifact lost"
        assert any(s <= 2.4 <= e for s, e in valid_times), "post-artifact lost"

        # Every member resolves to the SAME artifact-removed times (single
        # shared row; the per-member dict has one session entry equal to it).
        shared = ArtifactDetection().get_artifact_removed_intervals(art_pk)
        assert isinstance(shared, dict) and nwb_file_name in shared
        _np.testing.assert_array_equal(shared[nwb_file_name], valid_times)
    finally:
        _clear_shared_group()
