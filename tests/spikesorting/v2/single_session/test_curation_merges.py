"""CurationV2 merge-group tests (canonical IDs, lazy-vs-applied merges) for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2
from tests.spikesorting.v2.single_session._helpers import _clear_curations


@pytest.mark.slow
def test_curation_v2_insert_with_merge_groups_apply_merges(
    polymer_60s_session,
):
    """``CurationV2.insert_curation`` with ``merge_groups`` +
    ``apply_merge=True`` writes a curated NWB whose merged-unit id is a
    fresh ``max(source unit_ids) + 1`` (v1 parity, ``v1/curation.py:361``)
    and whose merged spike train is the sorted union of its contributors.
    Surviving source units are written first; merged ids are appended.

    Exercises ``build_curated_unit_rows`` with a non-empty merge
    list (amplitude-inheritance: kept unit gets electrode/amplitude
    from the highest-amplitude contributor) AND
    ``write_curated_units_nwb`` with ``apply_merge=True``
    (concatenated + sorted spike trains).

    Uses the 60s polymer fixture rather than the smoke fixture
    because MS5 finds only 1 unit on the 4s smoke recording -- too
    few to test merging. The 60s recording yields ~26 units (one
    sort group, all 4 shanks aggregated through the same flow).
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
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
    src_head = np.asarray(
        src_sorting.get_unit_spike_train(unit_id=head, return_times=True)
    )
    src_absorbed = np.asarray(
        src_sorting.get_unit_spike_train(unit_id=absorbed, return_times=True)
    )
    raw_concat = np.sort(np.concatenate([src_head, src_absorbed]))
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

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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
        samples = np.concatenate([u0, u1]).astype(np.int64)
        labels = np.concatenate(
            [np.zeros(len(u0)), np.ones(len(u1))]
        ).astype(np.int32)
        order = np.argsort(samples)
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
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
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
    contig_name = f"v2_lvam_contig_{_uuid.uuid4().hex[:8]}"
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
    disjoint_name = f"v2_lvam_disjoint_{_uuid.uuid4().hex[:8]}"
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
    """``build_curated_unit_rows`` writes the v1-parity unit set +
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
    from spyglass.spikesorting.v2._curation_transforms import (
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


@pytest.mark.slow
@pytest.mark.integration
def test_merge_group_contributor_fk_rejects_unknown_unit(populated_sorting):
    """A direct ``CurationV2.MergeGroup`` insert with a contributor that is
    not a real ``Sorting.Unit`` row raises ``IntegrityError``.

    ``contributor_unit_id`` is a FK to ``Sorting.Unit``:
    ``insert_curation`` validates contributors in Python, but the schema
    must also reject a bogus contributor on a direct part-table insert that
    bypasses the helper -- otherwise merge provenance can be silently
    corrupted. The shared ``sorting_id`` means the FK also enforces that the
    contributor belongs to THIS sort.
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
    """``build_curated_unit_rows`` assigns fresh merged ids in ascending
    MIN-CONTRIBUTOR order, INDEPENDENT of user-input group order (C3). For
    user input ``[[2, 3], [0, 1]]`` the smallest-min group ``[0, 1]`` gets
    the first new id ``max(source unit_ids) + 1``, and ``[2, 3]`` gets the
    next one. This matches the lazy ``get_merged_sorting`` preview path
    (which numbers merges by ascending kept-uid), so apply_merge=True and an
    apply_merge=False preview assign the SAME id to the SAME content group.

    v2 deliberately departs here from v1's user-iteration-order labels
    (``v1/curation.py:359``): the merged-unit id is an arbitrary fresh
    label, so only which group gets ``max+1`` changes for reordered input
    -- spike content and unit count are identical -- and matching the
    applied and lazy paths is the more important contract.
    """
    from spyglass.spikesorting.v2._curation_transforms import (
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
    # User input: bigger-min group FIRST. Canonical assignment IGNORES
    # this order and numbers by ascending min(group), so ``[0, 1]`` (min 0)
    # gets the first new id even though the user listed ``[2, 3]`` first.
    merge_groups = [[2, 3], [0, 1]]

    rows, kept = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        curation_id=0,
        apply_merge=True,
    )
    n_spikes_by_unit = {r["unit_id"]: r["n_spikes"] for r in rows}
    # max(0..4) + 1 = 5 -> smallest-min group [0, 1] gets new_id 5 carrying
    # summed n_spikes 30; group [2, 3] gets new_id 6 with n_spikes 70.
    # Non-merged unit 4 passes through with own count.
    assert n_spikes_by_unit == {4: 50, 5: 30, 6: 70}, n_spikes_by_unit

    # kept_to_contributors keys: surviving source units FIRST (in source
    # order), then merged ids appended in ascending-min order.
    assert list(kept.keys()) == [4, 5, 6], list(kept.keys())
    assert kept[5] == [0, 1] and kept[6] == [2, 3], kept

    # apply_merge=False with the same input: heads are min(group) for
    # each multi-group, so kept_to_contributors keys (after sort + sym
    # aug) are deterministic regardless of input order.
    rows_preview, kept_preview = build_curated_unit_rows(
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


@pytest.mark.usefixtures("dj_conn")
def test_merged_unit_inherits_max_amplitude_contributor_electrode():
    """A merged unit inherits the electrode + amplitude of its
    HIGHEST-amplitude contributor (not the head/min). Brain region is
    reached through the unit's Electrode FK, so this electrode inheritance
    IS the region attribution -- a ``max -> min`` regression would silently
    mis-attribute every merged unit's brain region while n_spikes / id
    assertions still pass (audit test-hardening #8). Deterministic via
    ``build_curated_unit_rows`` (no populate).
    """
    from spyglass.spikesorting.v2._curation_transforms import (
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

    # Unit 0: LOWER amplitude on electrode 10 (and it is the min/head of the
    # group). Unit 1: HIGHER amplitude on electrode 20. The merge must
    # inherit unit 1's electrode + amplitude, NOT the head's.
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

    # Reverse the amplitudes (now the HEAD/min unit dominates) -> the merged
    # unit must follow the amplitude, landing on the head's electrode. This
    # pins that inheritance tracks amplitude, not group position.
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
    """For TWO merge groups listed OUT of min-contributor order, the
    apply_merge=True stored sorting and the apply_merge=False lazy
    ``get_merged_sorting`` assign the SAME fresh id to the SAME content
    group, frame-for-frame. Existing tests cover the single-group case and
    the ``build_curated_unit_rows`` id assignment; this pins the >=2-group
    lazy ``get_merge_groups`` order_by path (audit test-hardening #16).
    """
    import uuid as _uuid

    import numpy as np
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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

    # Four planted units with well-separated frames (no coincidences -> no
    # cross-unit dedup), so each merged train is just the sorted union.
    planted = {
        0: np.array([100, 700]),
        1: np.array([300, 900]),
        2: np.array([200, 800]),
        3: np.array([400, 1000]),
    }

    def _planted_four_unit_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
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

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_four_unit_sorter)
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

    interval_name = f"v2_oo_merge_{_uuid.uuid4().hex[:8]}"
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
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
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
    # Groups listed OUT of min order ([2,3] before [0,1]).
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

    # Canonical-min id assignment, regardless of input order: [0,1] (min 0)
    # -> max(0..3)+1 = 4, [2,3] (min 2) -> 5. Same fresh id -> same content
    # group on BOTH the applied and lazy paths.
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
def test_v2_sorting_nwb_excludes_parent_units(dj_conn, tmp_path, monkeypatch):
    """The v2 Sorting analysis NWB contains ONLY the v2 sorted units, never
    the raw NWB's parent ``/units`` table (#1437). Plant a non-empty
    ``/units`` on a copy of the fixture, run Recording->Artifact->Sorting with
    a planted 2-unit sorter, then assert the analysis NWB's unit ids equal
    exactly ``Sorting.Unit`` and contain NONE of the planted parent ids
    (audit test-hardening #11). All MEArec fixtures keep the parent ``/units``
    empty, so this is the only exercise of the strip invariant.
    """
    from pathlib import Path

    import numpy as np
    import pynwb
    import spikeinterface as si

    from spyglass.common import AnalysisNwbfile
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "mearec_polymer_smoke.nwb"
    )
    if not fixture.exists():
        pytest.skip("smoke fixture not found")

    # Plant a non-empty /units table (parent ids 100/101/102) on a copy via
    # the pynwb modify-and-export pattern.
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
        {"team_name": "v2_test_team", "team_description": "v2 #1437"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
    )

    # Planted 2-unit sorter -> v2 unit ids {0, 1}, disjoint from the parent
    # ids {100, 101, 102} so any leak is unambiguous.
    def _planted_two_unit_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        samples = np.array([200, 400, 600, 800], dtype=np.int64)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            [samples], [labels], recording.get_sampling_frequency()
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_two_unit_sorter)
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
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
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

    # The analysis NWB units are EXACTLY the v2 sort, and none of the planted
    # parent ids leaked through ``AnalysisNwbfile().create``.
    assert nwb_unit_ids == v2_unit_ids
    assert not (
        set(parent_ids) & nwb_unit_ids
    ), "parent /units leaked into the v2 Sorting analysis NWB (#1437)"

    _clean_session_v2(session)


def test_curation_merge_ids_assigned_in_canonical_min_order(dj_conn):
    """Applied-path fresh merged ids are numbered in ascending
    min-contributor order, INDEPENDENT of user-input group order, so they
    match the lazy ``get_merged_sorting`` preview path (which numbers merges
    in ``get_merge_groups`` / ascending-kept-uid order). This is what makes
    ``apply_merge=True`` and an ``apply_merge=False`` preview assign the SAME
    fresh id to the SAME content group -- the preview==apply contract (C3).

    The merged-unit integer id is an arbitrary fresh ``max(unit_ids)+1``
    label; canonicalizing its assignment order (rather than following v1's
    user-iteration order) changes only which group gets ``max+1`` for
    reordered input, never spike content or unit count.
    """
    from spyglass.spikesorting.v2._curation_transforms import (
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
    # User lists the bigger-min group FIRST; canonical assignment ignores
    # this order and numbers by ascending min(group).
    _, kept = build_curated_unit_rows(
        sorting_id="s",
        sorting_units=sorting_units,
        merge_groups=[[3, 4], [0, 1]],
        curation_id=0,
        apply_merge=True,
    )
    # max(0..4) + 1 = 5. Smallest-min group [0, 1] (min 0) -> 5; next
    # group [3, 4] (min 3) -> 6 -- regardless of the [[3, 4], [0, 1]] order.
    assert kept[5] == [0, 1], kept
    assert kept[6] == [3, 4], kept


def test_curation_rejects_invalid_merge_groups(dj_conn):
    """All merge-group validation runs BEFORE any early return -- a
    zero-unit sort with non-empty merge_groups, intra-group duplicates,
    and references to nonexistent unit_ids all raise rather than
    silently no-op or fall through to staging that would double-count
    contributors. Covers the empty/singleton shape contract too.
    """
    from spyglass.spikesorting.v2._curation_transforms import (
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
                build_curated_unit_rows(
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
    import numpy as np
    import spikeinterface as si
    import spikeinterface.curation as sc

    ns = si.NumpySorting.from_unit_dict(
        # Two contributors share sample 10 -- the exact-coincident case.
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
        f"(got {n_zero}); SI behavior has changed, re-audit "
        "get_merged_sorting."
    )


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
