"""CurationV2 merge-group tests: canonical ids and lazy vs applied merges."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2
from tests.spikesorting.v2._sorter_stub import plant_sorter
from tests.spikesorting.v2.single_session._helpers import (
    _clear_curations,
    fixture_nwb_path,
)


@pytest.mark.slow
def test_curation_v2_insert_with_merge_groups_apply_merges(
    polymer_60s_session,
):
    """Applied merges get a fresh ``max(source unit_ids) + 1`` id (v1
    parity) whose train is the deduplicated union of its contributors;
    surviving units come first, merged ids last.

    Uses the 60s fixture: MS5 finds only 1 unit on the smoke recording.
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    units = sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert len(units) >= 2, (
        f"Sorting on the 60s polymer fixture yielded {len(units)} units; "
        "need >= 2 to test merging. (MS5 typically finds 5+ per shank "
        "on this fixture; a regression upstream would surface here.)"
    )

    head, absorbed = units[0], units[1]
    merge_groups = [[head, absorbed]]
    # Computed ahead so the insert can label it; labels apply to final ids.
    merged_id = max(units) + 1

    # Both curations branch off a plain root so the raw ids are valid
    # merge inputs; a child of the merged curation cannot see absorbed ids.
    root_pk = CurationV2.insert_curation(sorting_key=sort_pk)

    # v1 parity: labels on absorbed contributors are dropped with a warning.
    pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={merged_id: ["mua"], absorbed: ["noise"]},
        merge_groups=merge_groups,
        apply_merge=True,
        parent_curation_id=root_pk["curation_id"],
        description="merge_groups regression",
    )
    label_pairs = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & pk).fetch(as_dict=True)
    }
    assert (merged_id, "mua") in label_pairs, label_pairs
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

    # Stored train = union minus cross-contributor pairs < 0.4 ms (one
    # spike detected twice); SI's MergeUnitsSorting dedup is the reference.
    from spikeinterface.curation.mergeunitssorting import (
        get_non_duplicated_events,
    )

    src_sorting = Sorting().get_sorting(sort_pk)
    src_head = np.asarray(
        src_sorting.get_unit_spike_train(unit_id=head, return_times=True)
    )
    src_absorbed = np.asarray(
        src_sorting.get_unit_spike_train(unit_id=absorbed, return_times=True)
    )
    raw_concat = np.sort(np.concatenate([src_head, src_absorbed]))
    expected_merged = get_non_duplicated_events(
        [src_head.astype(float), src_absorbed.astype(float)], 0.4e-3
    )
    # The fixture's merged pair has at least one double-detection.
    assert len(expected_merged) < len(raw_concat), (
        "fixture precondition: merged contributors should share a "
        "sub-0.4 ms cross-unit double-detection so dedup fires "
        f"(raw={len(raw_concat)}, deduped={len(expected_merged)})"
    )

    curated_sorting = CurationV2().get_sorting(pk)
    merged_times = np.asarray(
        curated_sorting.get_unit_spike_train(
            unit_id=merged_id, return_times=True
        )
    )
    assert len(merged_times) == len(expected_merged), (
        f"Merged unit ({merged_id}) has {len(merged_times)} spikes; "
        f"expected {len(expected_merged)} (0.4 ms-deduped union)."
    )
    np.testing.assert_array_equal(merged_times, expected_merged)

    merged_row = (CurationV2.Unit & pk & {"unit_id": merged_id}).fetch1()
    assert merged_row["n_spikes"] == len(expected_merged), (
        f"CurationV2.Unit.n_spikes for merged_id={merged_id} = "
        f"{merged_row['n_spikes']}; expected {len(expected_merged)}."
    )
    assert merged_row["n_spikes"] == len(merged_times)

    # Survivors in source order, merged id last (matches v1 and SI's lazy
    # MergeUnitsSorting).
    expected_order = [u for u in units if u not in (head, absorbed)] + [
        merged_id
    ]
    actual_order = [int(u) for u in curated_sorting.get_unit_ids()]
    assert actual_order == expected_order, (
        f"NWB unit order = {actual_order}; expected {expected_order} "
        "(surviving source units first, merged_id appended last)."
    )

    # Preview (apply_merge=False): keeps every unit and label; the merge
    # lives in MergeGroup and get_merged_sorting applies it lazily.
    preview_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={absorbed: ["mua"]},
        merge_groups=merge_groups,
        apply_merge=False,
        parent_curation_id=root_pk["curation_id"],
        description="merge_groups preview (apply_merge=False)",
    )
    preview_unit_ids = set(
        int(u) for u in (CurationV2.Unit & preview_pk).fetch("unit_id")
    )
    assert preview_unit_ids == set(units), (
        f"apply_merge=False preview unit ids = {sorted(preview_unit_ids)}; "
        f"expected ALL {sorted(units)} (contributor preserved, v1 parity)."
    )
    preview_labels = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & preview_pk).fetch(as_dict=True)
    }
    assert (absorbed, "mua") in preview_labels, preview_labels
    preview_sorting = CurationV2().get_sorting(preview_pk)
    assert absorbed in set(int(u) for u in preview_sorting.get_unit_ids())
    # Every Unit row has a MergeGroup self-entry, so ``Unit * MergeGroup``
    # keeps absorbed contributors.
    mg_unit_ids = {
        int(r["unit_id"])
        for r in (CurationV2.MergeGroup & preview_pk).fetch(as_dict=True)
    }
    assert preview_unit_ids <= mg_unit_ids, (
        "Every CurationV2.Unit row should have >=1 MergeGroup row keyed "
        f"by its own unit_id; missing {sorted(preview_unit_ids - mg_unit_ids)}."
    )
    # Lazy merge dedups like the applied path (SI delta_time_ms=0.4).
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
    # Frame-level equality; counts alone miss frame-binning differences.
    lazy_merged_frames = np.sort(
        np.asarray(merged_lazy.get_unit_spike_train(unit_id=merged_id))
    )
    applied_merged_frames = np.sort(
        np.asarray(curated_sorting.get_unit_spike_train(unit_id=merged_id))
    )
    np.testing.assert_array_equal(lazy_merged_frames, applied_merged_frames)


@pytest.mark.slow
@pytest.mark.integration
def test_lazy_vs_applied_merge_frames_equal(polymer_smoke_session, monkeypatch):
    """Lazy and applied merges give the same hand-computed frames on a
    contiguous and a disjoint sort.

    Frame-adjacent spikes within a chunk are deduped; frame-adjacent spikes
    across a wall-clock gap are not.
    """
    import uuid

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        u0, u1 = state["u0"], state["u1"]
        samples = np.concatenate([u0, u1]).astype(np.int64)
        labels = np.concatenate([np.zeros(len(u0)), np.ones(len(u1))]).astype(
            np.int32
        )
        order = np.argsort(samples)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _planted_two_unit_sorter)

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
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "clusterless_thresholder",
                "sorter_params_name": "default",
                "artifact_detection_id": art_pk["artifact_detection_id"],
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
        lazy_frames = np.sort(
            np.asarray(
                CurationV2()
                .get_merged_sorting(preview_pk)
                .get_unit_spike_train(unit_id=merged_id)
            )
        )
        applied_frames = np.sort(
            np.asarray(
                CurationV2()
                .get_sorting(applied_pk)
                .get_unit_spike_train(unit_id=merged_id)
            )
        )
        np.testing.assert_array_equal(
            lazy_frames,
            applied_frames,
            err_msg="lazy preview merged frames != apply_merge=True stored",
        )
        np.testing.assert_array_equal(
            lazy_frames,
            np.asarray(expected_frames, dtype=lazy_frames.dtype),
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
    state["u0"] = np.array([100, 500, 900])
    state["u1"] = np.array([101, 600])
    contig_name = f"v2_lvam_contig_{uuid.uuid4().hex[:8]}"
    _, contig_sort_pk = _build_sort(
        contig_name, np.array([[t0, min(t0 + 2.9, t_end)]])
    )
    Sorting.populate(contig_sort_pk, reserve_jobs=False)
    _assert_lazy_equals_applied(contig_sort_pk, [100, 500, 600, 900])

    # --- Disjoint half: two chunks, 0.5 s gap. Recover the chunk-1/chunk-2
    # boundary frame ``k`` from the populated timeline, then plant a
    # within-chunk coincidence (100 & 101 -> dedup) AND a frame-adjacent
    # but cross-gap pair (k & k+1 -> NOT deduped). merged = [100, k, k+1].
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_name = f"v2_lvam_disjoint_{uuid.uuid4().hex[:8]}"
    disjoint_rec_pk, disjoint_sort_pk = _build_sort(
        disjoint_name,
        np.array([[t0, chunk1_end], [gap_end, chunk2_end]]),
    )
    rec = Recording().get_recording(disjoint_rec_pk)
    times = np.asarray(rec.get_times())
    fs = rec.get_sampling_frequency()
    gaps = np.flatnonzero(np.diff(times) > 1.5 / fs)
    assert len(gaps) == 1, f"expected one gap, found {len(gaps)}"
    k = int(gaps[0])  # chunk-1's last frame
    assert k >= 200, "chunk 1 too short to plant frame 100/101"
    state["u0"] = np.array([100, k])
    state["u1"] = np.array([101, k + 1])
    Sorting.populate(disjoint_sort_pk, reserve_jobs=False)
    _assert_lazy_equals_applied(disjoint_sort_pk, [100, k, k + 1])


def test_curation_n_spikes_matches_apply_merge(dj_conn):
    """``build_curated_unit_rows`` writes the v1-parity unit set and
    ``n_spikes`` for each ``apply_merge`` mode.

    ``apply_merge=True`` replaces the group with a fresh ``max + 1`` id
    carrying the summed train; ``apply_merge=False`` keeps every unit.
    """
    from spyglass.spikesorting.v2._curation.transforms import (
        build_curated_unit_rows,
    )

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
        rows, _ = build_curated_unit_rows(
            sorting_id="s",
            sorting_units=sorting_units,
            merge_groups=merge_groups,
            curation_id=0,
            apply_merge=apply_merge,
        )
        return {r["unit_id"]: r["n_spikes"] for r in rows}

    merged = n_spikes_by_unit(apply_merge=True)
    preview = n_spikes_by_unit(apply_merge=False)

    # [0, 1] -> fresh id 3 with 140 spikes; unit 2 passes through.
    assert merged == {3: 140, 2: 7}, merged
    assert preview == {0: 100, 1: 40, 2: 7}, preview


@pytest.mark.slow
@pytest.mark.integration
def test_merge_group_contributor_fk_rejects_unknown_unit(populated_sorting):
    """A direct ``CurationV2.MergeGroup`` insert with a contributor that is
    not a ``Sorting.Unit`` of this sort raises ``IntegrityError``.
    """
    from datajoint.errors import IntegrityError

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})

    # A real kept-unit row to hang the bad contributor off of.
    kept = (CurationV2.Unit & pk).fetch("KEY", as_dict=True)[0]
    real_unit_ids = set(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    bogus_contributor = max(real_unit_ids) + 1000
    assert bogus_contributor not in real_unit_ids

    with pytest.raises(IntegrityError):
        CurationV2.MergeGroup.insert1(
            {**kept, "contributor_unit_id": bogus_contributor}
        )


def test_curation_two_merge_groups_assign_ids_in_canonical_min_order(dj_conn):
    """``build_curated_unit_rows`` numbers fresh merged ids by ascending
    min contributor, independent of input group order, matching the lazy
    ``get_merged_sorting`` path. Unlike v1, which uses input order.
    """
    from spyglass.spikesorting.v2._curation.transforms import (
        build_curated_unit_rows,
    )

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
    # Bigger-min group listed first.
    merge_groups = [[2, 3], [0, 1]]

    rows, kept = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        curation_id=0,
        apply_merge=True,
    )
    n_spikes_by_unit = {r["unit_id"]: r["n_spikes"] for r in rows}
    # [0, 1] -> 5, [2, 3] -> 6; unit 4 passes through.
    assert n_spikes_by_unit == {4: 50, 5: 30, 6: 70}, n_spikes_by_unit

    assert list(kept.keys()) == [4, 5, 6], list(kept.keys())
    assert kept[5] == [0, 1] and kept[6] == [2, 3], kept

    # Preview heads are min(group) regardless of input order.
    rows_preview, kept_preview = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        curation_id=0,
        apply_merge=False,
    )
    preview_unit_ids = {r["unit_id"] for r in rows_preview}
    assert preview_unit_ids == {0, 1, 2, 3, 4}, preview_unit_ids
    assert 0 in kept_preview and kept_preview[0] == [0, 1]
    assert 2 in kept_preview and kept_preview[2] == [2, 3]


@pytest.mark.usefixtures("dj_conn")
def test_merged_unit_inherits_max_amplitude_contributor_electrode():
    """A merged unit inherits the electrode and amplitude of its
    highest-amplitude contributor, which sets its brain region.
    """
    from spyglass.spikesorting.v2._curation.transforms import (
        build_curated_unit_rows,
    )

    def _unit(uid, n_spikes, amp, electrode_id):
        return {
            "unit_id": uid,
            "n_spikes": n_spikes,
            "peak_amplitude_uv": amp,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": electrode_id,
        }

    # Head (unit 0) has the lower amplitude.
    sorting_units = [
        _unit(0, 100, 30.0, 10),
        _unit(1, 50, 80.0, 20),
    ]
    rows, _ = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=[[0, 1]],
        curation_id=0,
        apply_merge=True,
    )
    assert len(rows) == 1, "expected a single merged unit"
    merged = rows[0]
    assert merged["unit_id"] == 2  # fresh max(source ids)+1
    assert merged["electrode_id"] == 20, "merged unit took the wrong electrode"
    assert merged["peak_amplitude_uv"] == 80.0
    assert merged["n_spikes"] == 150  # summed train

    # Reversed amplitudes: inheritance follows amplitude, not position.
    sorting_units_rev = [
        _unit(0, 100, 90.0, 10),
        _unit(1, 50, 25.0, 20),
    ]
    rows_rev, _ = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units_rev,
        merge_groups=[[0, 1]],
        curation_id=0,
        apply_merge=True,
    )
    assert rows_rev[0]["electrode_id"] == 10
    assert rows_rev[0]["peak_amplitude_uv"] == 90.0


@pytest.mark.slow
@pytest.mark.integration
def test_applied_and_lazy_merge_ids_match_for_out_of_order_groups(
    polymer_smoke_session, monkeypatch
):
    """With two merge groups listed out of min order, applied and lazy
    merges assign the same fresh id to the same frames.
    """
    import uuid

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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

    # Well-separated frames, so no dedup.
    planted = {
        0: np.array([100, 700]),
        1: np.array([300, 900]),
        2: np.array([200, 800]),
        3: np.array([400, 1000]),
    }

    def _planted_four_unit_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.concatenate([planted[u] for u in (0, 1, 2, 3)]).astype(
            np.int64
        )
        labels = np.concatenate(
            [np.full(len(planted[u]), u) for u in (0, 1, 2, 3)]
        ).astype(np.int32)
        order = np.argsort(samples)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _planted_four_unit_sorter)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])

    interval_name = f"v2_oo_merge_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
            "valid_times": np.array([[t0, min(t0 + 2.9, t_end)]]),
            "pipeline": "v2_oo_merge_test",
        },
        skip_duplicates=True,
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert int((Sorting & sort_pk).fetch1("n_units")) == 4

    root_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    oo_groups = [[2, 3], [0, 1]]
    applied_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        merge_groups=oo_groups,
        apply_merge=True,
        parent_curation_id=root_pk["curation_id"],
        description="applied",
    )
    preview_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        merge_groups=oo_groups,
        apply_merge=False,
        parent_curation_id=root_pk["curation_id"],
        description="lazy preview",
    )

    applied = CurationV2().get_sorting(applied_pk)
    lazy = CurationV2().get_merged_sorting(preview_pk)

    def _frames(sorting, uid):
        return np.sort(np.asarray(sorting.get_unit_spike_train(unit_id=uid)))

    # [0, 1] -> 4, [2, 3] -> 5 on both paths.
    expected = {
        4: np.sort(np.concatenate([planted[0], planted[1]])),
        5: np.sort(np.concatenate([planted[2], planted[3]])),
    }
    for fresh_id, exp in expected.items():
        np.testing.assert_array_equal(
            _frames(applied, fresh_id),
            _frames(lazy, fresh_id),
            err_msg=f"applied vs lazy frames differ for merged id {fresh_id}",
        )
        np.testing.assert_array_equal(
            _frames(applied, fresh_id),
            exp.astype(_frames(applied, fresh_id).dtype),
            err_msg=f"merged id {fresh_id} frames != hand-computed union",
        )


@pytest.mark.slow
@pytest.mark.integration
def test_v2_sorting_nwb_excludes_parent_units(
    dj_conn, tmp_path, monkeypatch, smoke_nwb
):
    """The Sorting analysis NWB holds only the v2 units, never the raw
    NWB's ``/units``. MEArec fixtures have empty ``/units``, so one is planted.
    """

    import numpy as np
    import pynwb
    import spikeinterface as si

    from spyglass.common import AnalysisNwbfile
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
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    fixture = fixture_nwb_path("mearec_polymer_smoke")
    if not fixture.exists():
        pytest.skip("smoke fixture not found")

    parent_ids = [100, 101, 102]
    planted = tmp_path / "planted_units_src.nwb"
    with pynwb.NWBHDF5IO(str(fixture), mode="r") as read_io:
        nwbf = read_io.read()
        nwbf.add_unit(spike_times=[0.01, 0.02, 0.03], id=parent_ids[0])
        nwbf.add_unit(spike_times=[0.04, 0.05], id=parent_ids[1])
        nwbf.add_unit(spike_times=[0.06], id=parent_ids[2])
        with pynwb.NWBHDF5IO(str(planted), mode="w") as write_io:
            write_io.export(src_io=read_io, nwbfile=nwbf)

    nwb_file_name = copy_and_insert_nwb(planted, dest_name="mearec_1437.nwb")
    session = {"nwb_file_name": nwb_file_name}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 parent-units strip",
        },
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
    )

    # v2 ids {0, 1} are disjoint from the parent ids.
    def _planted_two_unit_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.array([200, 400, 600, 800], dtype=np.int64)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            [samples], [labels], recording.get_sampling_frequency()
        )

    plant_sorter(monkeypatch, _planted_two_unit_sorter)

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    v2_unit_ids = {int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id")}
    assert v2_unit_ids == {0, 1}, f"planted sorter expected, got {v2_unit_ids}"

    analysis_file = (Sorting & sort_pk).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file)
    with pynwb.NWBHDF5IO(abs_path, mode="r") as io:
        anwb = io.read()
        nwb_unit_ids = (
            set() if anwb.units is None else {int(u) for u in anwb.units.id[:]}
        )

    assert nwb_unit_ids == v2_unit_ids
    assert not (
        set(parent_ids) & nwb_unit_ids
    ), "parent /units leaked into the v2 Sorting analysis NWB"

    _clean_session_v2(session)


def test_curation_merge_ids_assigned_in_canonical_min_order(dj_conn):
    """Applied-path merged ids follow ascending min contributor,
    independent of input group order, matching the lazy preview.
    """
    from spyglass.spikesorting.v2._curation.transforms import (
        build_curated_unit_rows,
    )

    def _unit(uid, n_spikes, amp):
        return {
            "unit_id": uid,
            "n_spikes": n_spikes,
            "peak_amplitude_uv": amp,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": uid,
        }

    sorting_units = [_unit(i, 10 * (i + 1), 50.0 - i) for i in range(5)]
    _, kept = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=[[3, 4], [0, 1]],
        curation_id=0,
        apply_merge=True,
    )
    # [0, 1] -> 5, [3, 4] -> 6.
    assert kept[5] == [0, 1], kept
    assert kept[6] == [3, 4], kept


@pytest.mark.parametrize("apply_merge", [True, False])
@pytest.mark.parametrize(
    "unit_ids, merge_groups, match",
    [
        # Shape (>= 2 members), regardless of apply_merge.
        pytest.param([0, 1], [[0]], "fewer than 2", id="singleton-group"),
        pytest.param([0, 1], [[]], "fewer than 2", id="empty-group"),
        pytest.param(
            [0, 1], [[0, 1], []], "fewer than 2", id="trailing-empty-group"
        ),
        # Intra-group duplicates would double-count contributor 0.
        pytest.param(
            [0, 1], [[0, 0]], "duplicate members", id="duplicate-pair"
        ),
        pytest.param(
            [0, 1], [[0, 0, 1]], "duplicate members", id="duplicate-in-triple"
        ),
        # Zero-unit sort: validation runs before the empty early return.
        pytest.param(
            [], [[0, 1]], "not in Sorting.Unit", id="zero-unit-unknown-ids"
        ),
        # Shape errors fire before the id check.
        pytest.param([], [[]], "fewer than 2", id="zero-unit-empty-group"),
        pytest.param([], [[0]], "fewer than 2", id="zero-unit-singleton-group"),
    ],
)
def test_curation_rejects_invalid_merge_groups(
    unit_ids, merge_groups, match, apply_merge, dj_conn
):
    """Invalid merge groups raise before any early return, including on a
    zero-unit sort.
    """
    from spyglass.spikesorting.v2._curation.transforms import (
        build_curated_unit_rows,
    )

    def _unit(uid):
        return {
            "unit_id": uid,
            "n_spikes": 10,
            "peak_amplitude_uv": 50.0,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": uid,
        }

    sorting_units = [_unit(uid) for uid in unit_ids]
    with pytest.raises(ValueError, match=match):
        build_curated_unit_rows(
            sorting_id="s",
            sorting_units=sorting_units,
            merge_groups=merge_groups,
            curation_id=0,
            apply_merge=apply_merge,
        )


def test_si_merge_units_drops_same_sample_unless_delta_is_none(dj_conn):
    """SI 0.104's ``MergeUnitsSorting`` drops same-sample spikes from
    different units even at ``delta_time_ms=0``; only ``None`` disables
    the duplicate check.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.curation as sc

    ns = si.NumpySorting.from_unit_dict(
        # Two contributors share sample 10.
        {0: np.array([10]), 1: np.array([10])},
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
        f"(got {n_zero}); SI behavior has changed, re-check "
        "get_merged_sorting."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_merges_applied_records_user_intent(populated_sorting):
    """``merges_applied`` stores ``apply_merge`` verbatim, even with no
    merge groups (v1 parity).
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


def test_insert_curation_rejects_merge_group_overlap(populated_sorting):
    """A unit appearing in two merge groups raises.

    The smoke sort has one unit, so a second ``Sorting.Unit`` row is
    planted; validation raises before any spike train is read.
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
    # ``u0 + 1`` could collide with an existing unit id.
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
    """A singleton merge group is rejected upstream (layered defense).

    It raises before ``next_merged_id``, so no spurious fresh id is made.
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


def test_get_merged_sorting_returns_base_when_no_merges(
    populated_sorting_with_curation,
):
    """``get_merged_sorting`` returns the base sorting unchanged when no
    merge group has more than one contributor.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    base = CurationV2().get_sorting(key)
    merged = CurationV2().get_merged_sorting(key)
    assert list(merged.unit_ids) == list(base.unit_ids)


def test_get_merged_sorting_returns_base_when_merges_applied(
    populated_sorting_with_curation,
):
    """``get_merged_sorting`` returns the base verbatim when
    ``merges_applied`` is set (the base is already merged at insert).

    The smoke sort has one unit, so the flag is flipped on the root
    curation of a function-scoped fixture.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    # Root has no dependents, so bypassing the FactoryOnlyMaster guard is safe.
    CurationV2.update1({**key, "merges_applied": 1}, allow_master_mutation=True)

    base = CurationV2().get_sorting(key)
    merged = CurationV2().get_merged_sorting(key)
    assert list(merged.unit_ids) == list(base.unit_ids)
