"""Per-session outputs derived from a curated concatenated sorting."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._concat_helpers import select_unmasked_concat


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
    concat_key = select_unmasked_concat(
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


@pytest.mark.slow
def test_decoding_duplicate_guard_is_scoped_to_one_member(
    concat_member_curation,
):
    """Different members coexist; two curations of one member do not."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    ctx = concat_member_curation
    root_member_key = {**ctx["root_curation_key"], "member_index": 0}
    ConcatMemberCuration.populate(root_member_key, reserve_jobs=False)
    root_member_id = (
        SpikeSortingOutput.ConcatMemberCuration & root_member_key
    ).fetch1("merge_id")

    child_rows = sorted(ctx["rows"], key=lambda row: int(row["member_index"]))
    child_ids = [
        (
            SpikeSortingOutput.ConcatMemberCuration
            & {
                "sorting_id": row["sorting_id"],
                "curation_id": row["curation_id"],
                "member_index": int(row["member_index"]),
            }
        ).fetch1("merge_id")
        for row in child_rows
    ]

    SpikeSortingOutput.assert_decoding_merge_ids_ok(child_ids)
    with pytest.raises(ValueError, match="same sorting/member"):
        SpikeSortingOutput.assert_decoding_merge_ids_ok(
            [root_member_id, child_ids[0]]
        )


@pytest.mark.slow
def test_session_consumers_read_member_row(concat_member_curation):
    """Sorted-spike groups and unit annotations consume the member NWB."""
    import numpy as np

    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    ctx = concat_member_curation
    row = ctx["rows"][0]
    member_key = {
        "sorting_id": row["sorting_id"],
        "curation_id": row["curation_id"],
        "member_index": int(row["member_index"]),
    }
    merge_id = (SpikeSortingOutput.ConcatMemberCuration & member_key).fetch1(
        "merge_id"
    )
    direct_units = (ConcatMemberCuration & member_key).fetch_nwb()[0][
        "object_id"
    ]
    unit_id = int(direct_units.index[0])
    expected_spikes = direct_units.loc[unit_id, "spike_times"]

    UnitSelectionParams.insert_default()
    group_name = "concat_member_curation_consumer"
    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=row["nwb_file_name"],
        unit_filter_params_name="all_units",
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    group_key = {
        "nwb_file_name": row["nwb_file_name"],
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": group_name,
    }
    spikes, ids = SortedSpikesGroup.fetch_spike_data(
        group_key, return_unit_ids=True
    )
    assert ids == [{"spikesorting_merge_id": merge_id, "unit_id": unit_id}]
    np.testing.assert_array_equal(spikes[0], expected_spikes)

    annotation_key = {
        "spikesorting_merge_id": merge_id,
        "unit_id": unit_id,
        "annotation": "cell_type",
        "label": "test_cell",
    }
    UnitAnnotation().add_annotation(annotation_key)
    annotation_unit_key = {
        key: annotation_key[key] for key in ("spikesorting_merge_id", "unit_id")
    }
    annotated = (UnitAnnotation & annotation_unit_key).fetch_unit_spikes()
    np.testing.assert_array_equal(annotated[0], expected_spikes)

    # The generic merge FK cannot encode the group's Session. The explicit
    # member-session guard rejects a valid member output placed in the wrong
    # session rather than silently mixing incompatible wall-clock axes.
    with pytest.raises(ValueError, match="different session"):
        SortedSpikesGroup().create_group(
            group_name="concat_member_wrong_session",
            nwb_file_name=ctx["rows"][1]["nwb_file_name"],
            unit_filter_params_name="all_units",
            keys=[{"spikesorting_merge_id": merge_id}],
        )


@pytest.mark.slow
def test_waveform_features_group_rejects_other_sessions_member(
    concat_member_curation,
):
    """A clusterless group keyed by session A must refuse member B's output.

    ``UnitWaveformFeaturesGroup`` is keyed by ``Session`` while its
    ``UnitFeatures`` part only foreign-keys a merge id, the same structural gap
    ``SortedSpikesGroup`` closes with ``assert_merge_ids_match_session``. The
    duplicate-curation / unapplied-merge guard alone lets a valid member output
    into the wrong session's decode with that member's wall-clock timestamps.
    """
    from spyglass.decoding.v1.clusterless import UnitWaveformFeaturesGroup
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = concat_member_curation
    row = ctx["rows"][0]
    merge_id = (
        SpikeSortingOutput.ConcatMemberCuration
        & {
            "sorting_id": row["sorting_id"],
            "curation_id": row["curation_id"],
            "member_index": int(row["member_index"]),
        }
    ).fetch1("merge_id")

    with pytest.raises(ValueError, match="different session"):
        UnitWaveformFeaturesGroup().create_group(
            nwb_file_name=ctx["rows"][1]["nwb_file_name"],
            group_name="concat_member_wrong_session",
            keys=[
                {
                    "spikesorting_merge_id": merge_id,
                    "features_param_name": "concat_member_amplitude",
                }
            ],
        )


@pytest.mark.slow
def test_waveform_features_use_member_recording(concat_member_curation):
    """The SI 0.104 waveform path extracts against the member recording."""
    import numpy as np

    from spyglass.decoding.v1.waveform_features import (
        UnitWaveformFeatures,
        UnitWaveformFeaturesSelection,
        WaveformFeaturesParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    row = concat_member_curation["rows"][0]
    member_key = {
        "sorting_id": row["sorting_id"],
        "curation_id": row["curation_id"],
        "member_index": int(row["member_index"]),
    }
    merge_id = (SpikeSortingOutput.ConcatMemberCuration & member_key).fetch1(
        "merge_id"
    )
    params_name = "concat_member_amplitude"
    WaveformFeaturesParams.insert1(
        {
            "features_param_name": params_name,
            "params": {
                "waveform_extraction_params": {
                    "ms_before": 0.2,
                    "ms_after": 0.2,
                    "max_spikes_per_unit": None,
                    "n_jobs": 1,
                    "chunk_duration": "1s",
                },
                "waveform_features_params": {
                    "amplitude": {
                        "peak_sign": "neg",
                        "estimate_peak_time": False,
                    }
                },
            },
        },
        skip_duplicates=True,
    )
    selection = {
        "spikesorting_merge_id": merge_id,
        "features_param_name": params_name,
    }
    UnitWaveformFeaturesSelection.insert1(selection, skip_duplicates=True)
    UnitWaveformFeatures.populate(selection, reserve_jobs=False)
    assert UnitWaveformFeatures & selection

    spike_times, features = (UnitWaveformFeatures & selection).fetch_data()
    direct = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    assert len(spike_times) == len(features) == len(direct) == 1
    np.testing.assert_array_equal(spike_times[0], direct[0])
    assert features[0].shape[0] == len(direct[0])


@pytest.mark.slow
def test_preview_member_output_is_rejected_for_decoding(
    concat_member_curation,
):
    """A member row inherits its parent curation's unapplied-merge guard."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    ctx = concat_member_curation
    preview = CurationV2.propose_merge_curation(
        sorting_key=ctx["sorting_key"],
        merge_groups=[[0, 1]],
        parent_curation_id=ctx["root_curation_key"]["curation_id"],
        description="concat member preview guard test",
    )
    member_key = {**preview, "member_index": 0}
    ConcatMemberCuration.populate(member_key, reserve_jobs=False)
    merge_id = (SpikeSortingOutput.ConcatMemberCuration & member_key).fetch1(
        "merge_id"
    )
    with pytest.raises(ValueError, match="NOT applied"):
        SpikeSortingOutput.assert_decoding_merge_ids_ok([merge_id])


@pytest.mark.slow
def test_direct_member_delete_reclaims_file_and_repopulates(
    concat_member_curation,
):
    """A member row owns a reclaimable file and remains regenerable."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    curation_key = concat_member_curation["curation_key"]
    row = (ConcatMemberCuration & curation_key & {"member_index": 0}).fetch1()
    member_key = {name: row[name] for name in ConcatMemberCuration.primary_key}
    analysis_key = {"analysis_file_name": row["analysis_file_name"]}
    analysis_path = Path(
        AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    )
    merge_key = {
        "merge_id": (
            SpikeSortingOutput.ConcatMemberCuration & member_key
        ).fetch1("merge_id")
    }

    (ConcatMemberCuration & member_key).delete(
        force_permission=True, safemode=False
    )
    assert not (ConcatMemberCuration & member_key)
    assert not (SpikeSortingOutput & merge_key)
    assert not (AnalysisNwbfile & analysis_key)
    assert not analysis_path.exists()

    ConcatMemberCuration.populate(curation_key, reserve_jobs=False)
    assert ConcatMemberCuration & member_key
    assert SpikeSortingOutput.ConcatMemberCuration & member_key


@pytest.mark.slow
def test_delete_cascades_member_rows_and_reclaims_files(
    concat_member_curation, monkeypatch
):
    """Deleting a concat curation removes member outputs and owned files."""
    from pathlib import Path

    import spyglass.spikesorting.v2.concat_member_curation as member_mod
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    ctx = concat_member_curation
    curation_key = ctx["curation_key"]
    rows = (ConcatMemberCuration & curation_key).fetch(as_dict=True)
    analysis_names = [str(row["analysis_file_name"]) for row in rows]
    paths = [
        Path(AnalysisNwbfile.get_abs_path(name)) for name in analysis_names
    ]
    merge_ids = list(
        (SpikeSortingOutput.ConcatMemberCuration & curation_key).fetch(
            "merge_id"
        )
    )
    assert rows and all(path.exists() for path in paths)

    seen: list[str] = []
    monkeypatch.setattr(
        member_mod.logger,
        "info",
        lambda message, *args, **kwargs: seen.append(str(message)),
    )
    (CurationV2 & curation_key).delete(force_permission=True, dry_run=True)
    preview = "\n".join(seen)
    assert all(name in preview for name in analysis_names)
    assert CurationV2 & curation_key
    assert ConcatMemberCuration & curation_key

    monkeypatch.setattr(
        "datajoint.table.user_choice", lambda *args, **kwargs: "no"
    )
    (CurationV2 & curation_key).delete(force_permission=True, safemode=True)
    assert CurationV2 & curation_key
    assert ConcatMemberCuration & curation_key
    assert all(path.exists() for path in paths)

    (CurationV2 & curation_key).delete(force_permission=True, safemode=False)
    assert not (CurationV2 & curation_key)
    assert not (ConcatMemberCuration & curation_key)
    assert not (SpikeSortingOutput & [{"merge_id": mid} for mid in merge_ids])
    assert not (
        AnalysisNwbfile
        & [{"analysis_file_name": name} for name in analysis_names]
    )
    assert not any(path.exists() for path in paths)
