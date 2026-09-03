"""Per-session outputs derived from a curated concatenated sorting."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def concat_member_curation(chronic_2_session_minirec):
    """Build a deterministic applied-merge curation spanning two members."""
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAMS,
    )

    sub = chronic_2_session_minirec
    owner = sub["owner"]
    group_key = {
        "session_group_owner": owner,
        "session_group_name": "sg_concat_member_curation",
    }
    clean_session_groups_for_owner(owner)
    SessionGroup.create_group(
        owner, group_key["session_group_name"], sub["same_day_members"]
    )
    concat_key = ConcatenatedRecordingSelection.insert_selection(
        {
            **group_key,
            "preprocessing_params_name": sub["preprocessing_params_name"],
            "motion_correction_params_name": "none",
        }
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)
    first_end = int(
        (
            ConcatenatedRecording.MemberBoundary
            & concat_key
            & {"member_index": 0}
        ).fetch1("end_sample")
    )

    params_name = "concat_member_curation_two_unit"
    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
    ):
        del sorter, sorter_params, sorting_id, job_kwargs, execution_params
        frames = np.asarray(
            [100, 200, first_end + 100, first_end + 200], dtype=np.int64
        )
        labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[frames],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    sorting_key = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_key["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
        }
    )
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        Sorting.populate(sorting_key, reserve_jobs=False)
    finally:
        patch.undo()

    root_curation_key = CurationV2.insert_curation(
        sorting_key=sorting_key,
        labels={0: ["accept"], 1: ["mua"]},
        description="concat member labeled root",
    )
    curation_key = CurationV2.create_merged_curation(
        sorting_key=sorting_key,
        merge_groups=[[0, 1]],
        parent_curation_id=root_curation_key["curation_id"],
        description="concat member applied-merge test",
    )
    ConcatMemberCuration.populate(curation_key, reserve_jobs=False)
    rows = (ConcatMemberCuration & curation_key).fetch(
        as_dict=True, order_by="member_index"
    )
    yield {
        "concat_key": concat_key,
        "sorting_key": sorting_key,
        "root_curation_key": root_curation_key,
        "curation_key": curation_key,
        "rows": rows,
        "members": sub["same_day_members"],
    }

    clean_session_groups_for_owner(owner)
    (SorterParameters & {"sorter_params_name": params_name}).super_delete(
        warn=False, safemode=False
    )


@pytest.mark.slow
def test_member_rows_preserve_units_spikes_labels_and_wall_clock(
    concat_member_curation,
):
    """Every member retains the curated namespace on its own timestamps."""
    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
        recording_timestamps,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    ctx = concat_member_curation
    curation_key = ctx["curation_key"]
    rows = ctx["rows"]
    assert [int(row["member_index"]) for row in rows] == [0, 1]
    assert not any(int(row["n_units"]) != 1 for row in rows)

    curated = CurationV2.get_sorting(curation_key)
    expected = ConcatenatedRecording().split_sorting_by_session(
        curated, ctx["concat_key"]
    )
    parent_labels = CurationV2._labels_by_unit(curation_key)
    assert len(parent_labels) == 1
    expected_unit_ids = {int(unit_id) for unit_id in curated.unit_ids}

    snapshots = (
        ConcatenatedRecordingSelection.MemberSnapshot & ctx["concat_key"]
    ).fetch(as_dict=True, order_by="member_index")
    concat_counts = {
        int(unit_id): len(curated.get_unit_spike_train(unit_id=unit_id))
        for unit_id in curated.unit_ids
    }
    member_counts = {unit_id: 0 for unit_id in expected_unit_ids}

    for row, snapshot in zip(rows, snapshots):
        member_key = {
            **curation_key,
            "member_index": int(row["member_index"]),
        }
        member_sorting = ConcatMemberCuration.get_sorting(member_key)
        assert {int(unit_id) for unit_id in member_sorting.unit_ids} == (
            expected_unit_ids
        )

        split_key = (
            snapshot["nwb_file_name"],
            int(snapshot["sort_group_id"]),
            snapshot["interval_list_name"],
            snapshot["team_name"],
        )
        expected_member = expected[split_key]
        for unit_id in expected_unit_ids:
            actual_frames = member_sorting.get_unit_spike_train(unit_id=unit_id)
            np.testing.assert_array_equal(
                actual_frames,
                expected_member.get_unit_spike_train(unit_id=unit_id),
            )
            member_counts[unit_id] += len(actual_frames)

        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        abs_times, sample_indices, _obs = (
            read_units_abs_times_and_sample_indices(abs_path)
        )
        timestamps = recording_timestamps(
            (Recording & {"recording_id": snapshot["recording_id"]}).fetch1()
        )
        for unit_id in expected_unit_ids:
            frames = sample_indices[unit_id]
            np.testing.assert_array_equal(
                abs_times[unit_id], timestamps[frames]
            )
            if len(abs_times[unit_id]):
                assert float(abs_times[unit_id].min()) >= float(timestamps[0])
                assert float(abs_times[unit_id].max()) <= float(timestamps[-1])

        units = (ConcatMemberCuration & member_key).fetch_nwb()[0]["object_id"]
        stored_labels = {
            int(unit_id): list(unit_row["curation_label"])
            for unit_id, unit_row in units.iterrows()
        }
        assert stored_labels == parent_labels

    assert member_counts == concat_counts


@pytest.mark.slow
def test_member_rows_register_and_dispatch_through_merge_table(
    concat_member_curation,
):
    """Each member is a normal session-scoped SpikeSortingOutput source."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    ctx = concat_member_curation
    curation_key = ctx["curation_key"]
    assert not (SpikeSortingOutput.CurationV2 & curation_key)
    merge_rows = (SpikeSortingOutput.ConcatMemberCuration & curation_key).fetch(
        as_dict=True
    )
    assert len(merge_rows) == len(ctx["rows"]) == 2

    expected_nwbs = {member["nwb_file_name"] for member in ctx["members"]}
    seen_nwbs = set()
    for merge_row in merge_rows:
        merge_key = {"merge_id": merge_row["merge_id"]}
        source_row = (
            ConcatMemberCuration
            & {
                "sorting_id": merge_row["sorting_id"],
                "curation_id": merge_row["curation_id"],
                "member_index": merge_row["member_index"],
            }
        ).fetch1()
        seen_nwbs.add(source_row["nwb_file_name"])

        recording = SpikeSortingOutput.get_recording(merge_key)
        sorting = SpikeSortingOutput.get_sorting(merge_key)
        assert recording.get_num_samples() > 0
        assert len(sorting.unit_ids) == int(source_row["n_units"])
        assert set(
            SpikeSortingOutput.get_sort_group_info(merge_key).fetch(
                "nwb_file_name"
            )
        ) == {source_row["nwb_file_name"]}
        SpikeSortingOutput.assert_decoding_merge_ids_ok([merge_row["merge_id"]])

    assert seen_nwbs == expected_nwbs
