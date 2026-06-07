"""Read-side correct-key selection when MORE THAN ONE merge_id exists.

The rest of the v2 suite proves insert-side integrity (duplicate detection,
distinct-identity minting) thoroughly, but every *consumer* accessor on
``SpikeSortingOutput`` -- ``get_recording``, ``get_sort_group_info``,
``get_spike_times``, ``get_spike_indicator``, ``get_firing_rate``,
``get_unit_brain_regions``, and ``get_restricted_merge_ids`` -- was only
ever exercised with a SINGLE merge_id in the merge table. With one row a
broken restriction is indistinguishable from a correct one: a dispatch that
forgot its ``& merge_get_part(key)`` filter would return "the only row" and
pass anyway.

These tests populate TWO genuinely distinguishable v2 merge_ids and assert
each accessor returns the data for the *requested* merge_id and not the
other. The two sorts are built on two different sort groups (different
electrode sets) and given deterministic, disjoint planted unit sets via a
monkeypatched ``Sorting._run_sorter`` (the same plant pattern as
``test_boundary_spike_round_trip_does_not_raise``), so:

* electrode-keyed accessors (``get_recording``, ``get_sort_group_info``)
  discriminate by electrode membership (shank 0 vs shank 1), and
* unit-keyed accessors (``get_spike_times`` / ``get_spike_indicator`` /
  ``get_firing_rate`` / ``get_unit_brain_regions``) discriminate by unit
  count (2 vs 3) and disjoint spike-time range.

Each test first establishes a *selectivity baseline* -- an unrestricted or
cross query that sees BOTH merge_ids -- before asserting the restricted
query isolates one. That proves the filter discriminates rather than
trivially matching one-of-one (the pattern from
``test_merge_id_artifact_resolution.py``).

Self-contained: ingests the smoke fixture under a UNIQUE session name so
its cleanup never cascades into the package-scoped ``populated_sorting``
rows other modules depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
_SESSION_NWB = "mearec_multientry.nwb"

# Deterministic planted spike frames (well within the ~3 s / ~90k-sample
# smoke recording). Group-A frames are all < 6000; group-B frames are all
# >= 40000, so the absolute spike-time ranges are disjoint and a wrong-key
# read is caught by both unit count AND time range.
_A_UNIT_FRAMES = {
    0: [500, 1500, 2500, 3500, 4500, 5500],
    1: [800, 1800, 2800, 3800, 4800, 5800],
}
_B_UNIT_FRAMES = {
    0: [40000, 41000, 42000, 43000, 44000, 45000],
    1: [40300, 41300, 42300, 43300, 44300, 45300],
    2: [40600, 41600, 42600, 43600, 44600, 45600],
}


def _make_plant(unit_frames: dict[int, list[int]]):
    """Build a ``_run_sorter`` replacement that returns a fixed NumpySorting.

    ``_run_sorter`` is a staticmethod dispatched from ``Sorting.make_compute``
    for every sorter (clusterless and SI alike), so patching it plants a
    deterministic unit set independent of the real sorter and of the smoke
    fixture's stochastic unit yield.
    """

    def _plant(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        samples = np.concatenate(
            [np.asarray(f, dtype=np.int64) for f in unit_frames.values()]
        )
        labels = np.concatenate(
            [
                np.full(len(f), uid, dtype=np.int32)
                for uid, f in unit_frames.items()
            ]
        )
        order = np.argsort(samples)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    return _plant


def _build_sort_on_group(sort_group_id, plant, nwb_file_name):
    """Populate Recording -> ArtifactDetection(none) -> Sorting(planted) ->
    root CurationV2 on one sort group; return (sort_pk, merge_id, rec_pk)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
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
    (Sorting & sort_pk).super_delete(warn=False)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(plant))
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if not (Sorting & sort_pk):
        pytest.skip("planted Sorting.populate produced no row")

    cur_pk = CurationV2.insert_curation(sorting_key=sort_pk)
    merge_id = (SpikeSortingOutput.CurationV2 & cur_pk).fetch1("merge_id")
    return sort_pk, merge_id, rec_pk


@pytest.fixture(scope="module")
def two_v2_merge_ids(dj_conn):
    """Two distinguishable v2 merge_ids on one session.

    Sort A: sort group 0, 2 planted units (early spikes).
    Sort B: sort group 1, 3 planted units (later, disjoint spikes).

    Yields a context dict with both merge_ids, recording ids, planted unit
    counts, and the per-group electrode sets. Cleans the unique session on
    setup and teardown (the persistent test DB carries rows across runs).
    """
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH, dest_name=_SESSION_NWB)
    session_key = {"nwb_file_name": nwb_file_name}

    _clean_session_v2(session_key)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 multi-entry"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    sg_ids = sorted(
        int(s) for s in (SortGroupV2 & session_key).fetch("sort_group_id")
    )
    assert len(sg_ids) >= 2, (
        "multi-entry dispatch test needs >=2 sort groups; smoke fixture "
        f"yielded {sg_ids}"
    )
    sg_a, sg_b = sg_ids[0], sg_ids[1]

    def _elec_set(sg_id):
        return {
            int(e)
            for e in (
                SortGroupV2.SortGroupElectrode
                & session_key
                & {"sort_group_id": sg_id}
            ).fetch("electrode_id")
        }

    elec_a, elec_b = _elec_set(sg_a), _elec_set(sg_b)
    assert elec_a and elec_b and elec_a.isdisjoint(elec_b), (
        "sort groups must have disjoint, non-empty electrode sets; "
        f"A={sorted(elec_a)} B={sorted(elec_b)}"
    )

    sort_a, mid_a, rec_a = _build_sort_on_group(
        sg_a, _make_plant(_A_UNIT_FRAMES), nwb_file_name
    )
    sort_b, mid_b, rec_b = _build_sort_on_group(
        sg_b, _make_plant(_B_UNIT_FRAMES), nwb_file_name
    )
    assert mid_a != mid_b, "the two sorts must mint distinct merge_ids"

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")

    yield {
        "merge_id_a": mid_a,
        "merge_id_b": mid_b,
        "recording_id_a": rec_a["recording_id"],
        "recording_id_b": rec_b["recording_id"],
        "n_units_a": len(_A_UNIT_FRAMES),
        "n_units_b": len(_B_UNIT_FRAMES),
        "elec_a": elec_a,
        "elec_b": elec_b,
        "t_start": float(raw_times[0][0]),
        "t_end": float(raw_times[-1][-1]),
    }

    _clean_session_v2(session_key)


# --------------------------------------------------------------------------
# Unit-keyed accessors: discriminate by unit count + disjoint time range.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_get_spike_times_selects_only_target_merge_id(two_v2_merge_ids):
    """``get_spike_times({"merge_id": A})`` returns A's planted trains only.

    With two merge_ids present, a dispatch that dropped its
    ``merge_get_part`` restriction would return both sorts' units. Asserts
    the unit count (2 vs 3) AND that the returned spike-time range matches
    the requested merge_id's disjoint planted window.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids
    sso = SpikeSortingOutput()

    st_a = sso.get_spike_times({"merge_id": ctx["merge_id_a"]})
    st_b = sso.get_spike_times({"merge_id": ctx["merge_id_b"]})

    assert len(st_a) == ctx["n_units_a"], (
        f"merge_id A returned {len(st_a)} unit trains; expected "
        f"{ctx['n_units_a']} (a wrong-key read would surface B's count)"
    )
    assert len(st_b) == ctx["n_units_b"], (
        f"merge_id B returned {len(st_b)} unit trains; expected "
        f"{ctx['n_units_b']}"
    )

    a_all = np.concatenate([np.asarray(t) for t in st_a])
    b_all = np.concatenate([np.asarray(t) for t in st_b])
    assert a_all.size and b_all.size
    # Planted frames are disjoint (A < 6000, B >= 40000) -> times disjoint.
    assert a_all.max() < b_all.min(), (
        "A's spike times must precede all of B's (disjoint planted windows); "
        f"max(A)={a_all.max():.4f} min(B)={b_all.min():.4f} -- overlap means "
        "the wrong merge_id's trains were returned"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_get_spike_indicator_and_firing_rate_match_target_units(
    two_v2_merge_ids,
):
    """``get_spike_indicator`` / ``get_firing_rate`` width tracks the
    requested merge_id's unit count (2 for A, 3 for B)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids
    time = np.linspace(ctx["t_start"], ctx["t_end"], 200)

    ind_a = SpikeSortingOutput.get_spike_indicator(
        {"merge_id": ctx["merge_id_a"]}, time
    )
    ind_b = SpikeSortingOutput.get_spike_indicator(
        {"merge_id": ctx["merge_id_b"]}, time
    )
    assert ind_a.shape == (len(time), ctx["n_units_a"]), ind_a.shape
    assert ind_b.shape == (len(time), ctx["n_units_b"]), ind_b.shape

    fr_a = SpikeSortingOutput.get_firing_rate(
        {"merge_id": ctx["merge_id_a"]}, time
    )
    fr_b = SpikeSortingOutput.get_firing_rate(
        {"merge_id": ctx["merge_id_b"]}, time
    )
    assert fr_a.shape[-1] == ctx["n_units_a"], fr_a.shape
    assert fr_b.shape[-1] == ctx["n_units_b"], fr_b.shape


# --------------------------------------------------------------------------
# Electrode-keyed accessors: discriminate by electrode membership.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_get_sort_group_info_returns_target_electrodes(two_v2_merge_ids):
    """``get_sort_group_info({"merge_id": A})`` returns sort group A's
    electrodes, disjoint from B's (catches a recording/electrode mis-join)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids

    info_a = SpikeSortingOutput.get_sort_group_info(
        {"merge_id": ctx["merge_id_a"]}
    )
    info_b = SpikeSortingOutput.get_sort_group_info(
        {"merge_id": ctx["merge_id_b"]}
    )
    got_a = {int(e) for e in info_a.fetch("electrode_id")}
    got_b = {int(e) for e in info_b.fetch("electrode_id")}

    assert got_a == ctx["elec_a"], (
        "merge_id A's sort-group info returned the wrong electrode set"
    )
    assert got_b == ctx["elec_b"]
    assert got_a.isdisjoint(got_b)


@pytest.mark.slow
@pytest.mark.integration
def test_get_recording_returns_target_recording(two_v2_merge_ids):
    """``get_recording({"merge_id": A})`` loads A's recording: 32 channels
    whose ids are disjoint from B's (a dropped restriction would return one
    recording for both keys)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids

    rec_a = SpikeSortingOutput.get_recording({"merge_id": ctx["merge_id_a"]})
    rec_b = SpikeSortingOutput.get_recording({"merge_id": ctx["merge_id_b"]})

    ch_a = {int(c) for c in rec_a.get_channel_ids()}
    ch_b = {int(c) for c in rec_b.get_channel_ids()}
    assert len(ch_a) == len(ctx["elec_a"]) == 32
    assert len(ch_b) == len(ctx["elec_b"]) == 32
    assert ch_a.isdisjoint(ch_b), (
        "the two merge_ids resolved to overlapping recordings; the merge "
        "restriction did not isolate by merge_id"
    )


# --------------------------------------------------------------------------
# Per-unit brain regions through the MERGE dispatcher (not CurationV2 direct).
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_get_unit_brain_regions_selects_target_units(two_v2_merge_ids):
    """``SpikeSortingOutput.get_unit_brain_regions({"merge_id": A})`` returns
    A's units (2) keyed to group-A electrodes, not B's (3)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids

    df_a = SpikeSortingOutput.get_unit_brain_regions(
        {"merge_id": ctx["merge_id_a"]}
    )
    df_b = SpikeSortingOutput.get_unit_brain_regions(
        {"merge_id": ctx["merge_id_b"]}
    )
    assert df_a["unit_id"].nunique() == ctx["n_units_a"], (
        f"merge_id A reported {df_a['unit_id'].nunique()} units; expected "
        f"{ctx['n_units_a']}"
    )
    assert df_b["unit_id"].nunique() == ctx["n_units_b"]

    elec_in_a = {int(e) for e in df_a["electrode_id"]}
    assert elec_in_a <= ctx["elec_a"], (
        "A's units are attributed to electrodes outside sort group A"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_get_unit_brain_regions_rejects_ambiguous_key(two_v2_merge_ids):
    """An OR restriction matching BOTH merge_ids hits the ``len(part_rows)
    != 1`` guard and raises -- the guard only does real work when >=2
    merge_ids exist (with one row it passes vacuously)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids
    ambiguous = [
        {"merge_id": ctx["merge_id_a"]},
        {"merge_id": ctx["merge_id_b"]},
    ]
    with pytest.raises(ValueError, match="exactly one"):
        SpikeSortingOutput.get_unit_brain_regions(ambiguous)


# --------------------------------------------------------------------------
# Public merge-id resolution across two recordings.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_get_restricted_merge_ids_discriminates_by_recording(two_v2_merge_ids):
    """``get_restricted_merge_ids`` returns only the merge_id whose recording
    matches the key. Selectivity baseline: each recording resolves to exactly
    its own merge_id, and the two are distinct."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_v2_merge_ids
    sso = SpikeSortingOutput()

    got_a = sso.get_restricted_merge_ids(
        {"recording_id": ctx["recording_id_a"]}, sources=["v2"]
    )
    got_b = sso.get_restricted_merge_ids(
        {"recording_id": ctx["recording_id_b"]}, sources=["v2"]
    )
    assert list(got_a) == [ctx["merge_id_a"]], got_a
    assert list(got_b) == [ctx["merge_id_b"]], got_b
