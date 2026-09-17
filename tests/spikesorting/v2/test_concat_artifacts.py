"""Artifact masking on member and concatenated frame coordinates."""

import numpy as np
import pytest


def test_mask_members_preserves_disjoint_times_and_exact_boundaries():
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._concat_recording import (
        mask_member_recordings,
        observation_intervals,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    first = NumpyRecording(np.ones((1000, 2), dtype="float32"), 1000)
    first.set_times(
        np.r_[10 + np.arange(500) / 1000, 20 + np.arange(500) / 1000]
    )
    second = NumpyRecording(np.ones((1000, 2), dtype="float32"), 1000)
    second.set_times(100 + np.arange(1000) / 1000)
    valid = [
        np.array([[10, 10.499], [20, 20.45]]),
        np.array([[100.02, 100.999]]),
    ]
    masked, ranges = mask_member_recordings([first, second], valid)
    assert ranges == [(950, 1000), (1000, 1020)]
    for original, result, intervals in zip([first, second], masked, valid):
        np.testing.assert_array_equal(result.get_times(), original.get_times())
        np.testing.assert_array_equal(
            result.get_traces(),
            apply_artifact_mask(original, intervals).get_traces(),
        )
    np.testing.assert_array_equal(masked[0].get_traces()[499], [1, 1])
    np.testing.assert_array_equal(masked[0].get_traces()[950:], 0)
    np.testing.assert_array_equal(masked[1].get_traces()[:20], 0)
    np.testing.assert_allclose(
        observation_intervals(2000, 1000, ranges), [[0, 0.95], [1.02, 2]]
    )


def test_unmasked_members_keep_their_samples():
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._concat_recording import (
        mask_member_recordings,
    )

    recording = NumpyRecording(np.ones((100, 2), dtype="float32"), 1000)
    recordings, ranges = mask_member_recordings([recording], [None])
    assert recordings == [recording]
    assert ranges == []


@pytest.mark.slow
@pytest.mark.parametrize("motion", ["none", "rigid_fast"])
def test_detected_artifacts_survive_concat_rebuild_and_member_export(
    chronic_2_session_minirec,
    monkeypatch,
    motion,
    curation_evaluation_defaults,
):
    from pathlib import Path

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _concat_recording as concat_services
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        artifact_frame_ranges,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        MotionCorrectionParameters,
        SessionGroup,
    )

    motion_name = "artifact_test_rigid" if motion != "none" else "none"
    if motion != "none":
        MotionCorrectionParameters.insert1(
            {
                "motion_correction_params_name": motion_name,
                "params": {
                    "preset": "rigid_fast",
                    "preset_kwargs": {
                        "interpolate_motion_kwargs": {
                            "border_mode": "force_extrapolate"
                        }
                    },
                },
            },
            skip_duplicates=True,
        )
    fixture = chronic_2_session_minirec
    name = f"artifact_concat_{motion}"
    SessionGroup.create_group(
        fixture["owner"], name, fixture["same_day_members"]
    )
    group = {
        "session_group_owner": fixture["owner"],
        "session_group_name": name,
    }
    request = {
        **group,
        "preprocessing_params_name": fixture["preprocessing_params_name"],
        "motion_correction_params_name": (motion_name),
    }
    recordings = [
        Recording().get_recording(key) for key in fixture["recording_pks"]
    ]
    detection_ids, valid_times, ranges = {}, [], []
    for i, (recording, rec_key) in enumerate(
        zip(recordings, fixture["recording_pks"])
    ):
        # Detect rare, real high-amplitude periods in each fixed-seed recording.
        threshold = float(
            np.quantile(np.abs(recording.get_traces(return_in_uV=True)), 0.999)
        )
        recipe = f"concat_artifacts_{i}"
        ArtifactDetectionParameters.insert1(
            {
                "artifact_detection_params_name": recipe,
                "params": {
                    "amplitude_threshold_uv": threshold,
                    "proportion_above_threshold": 0.25,
                    "min_length_s": 0.001,
                },
            },
            skip_duplicates=True,
            allow_duplicate_params=True,
        )
        art_key = RecordingArtifactSelection.insert_selection(
            {
                **rec_key,
                "artifact_detection_params_name": recipe,
                "manual_excluded_times": [
                    [
                        float(recording.sample_index_to_time(200)),
                        float(recording.sample_index_to_time(210)),
                    ]
                ],
            }
        )
        RecordingArtifactDetection.populate(art_key, reserve_jobs=False)
        detection_ids[i] = art_key["artifact_detection_id"]
        valid = RecordingArtifactDetection().get_artifact_removed_intervals(
            art_key
        )
        valid_times.append(valid)
        ranges.append(artifact_frame_ranges(recording, valid))
        assert ranges[-1], "Fixture must actually exercise nonempty masks."
        assert any(start <= 200 and end >= 210 for start, end in ranges[-1])

    observed = []
    original = concat_services.build_concatenated_recording

    def inspect_motion_input(member_recordings, **kwargs):
        for member, excluded in zip(member_recordings, ranges):
            for start, end in excluded:
                np.testing.assert_array_equal(
                    member.get_traces(start_frame=start, end_frame=end), 0
                )
        observed.append(True)
        return original(member_recordings, **kwargs)

    monkeypatch.setattr(
        concat_services, "build_concatenated_recording", inspect_motion_input
    )
    key = ConcatenatedRecordingSelection.insert_selection(
        request, artifact_detection_ids=detection_ids
    )
    assert (
        ConcatenatedRecordingSelection.insert_selection(
            request, artifact_detection_ids=detection_ids
        )
        == key
    )
    different = ConcatenatedRecordingSelection.insert_selection(
        request, artifact_detection_ids={**detection_ids, 1: None}
    )
    assert different != key
    with pytest.raises(ValueError, match="must be populated for recording"):
        ConcatenatedRecordingSelection.insert_selection(
            request,
            artifact_detection_ids={0: detection_ids[1], 1: detection_ids[0]},
        )
    with pytest.raises(ValueError, match="referenced"):
        (
            RecordingArtifactDetection
            & {"artifact_detection_id": detection_ids[0]}
        ).delete(safemode=False)

    ConcatenatedRecording.populate(key, reserve_jobs=False)
    row = (ConcatenatedRecording & key).fetch1()
    combined = ConcatenatedRecording().get_recording(key)
    offset = 0
    for i, (recording, excluded, valid) in enumerate(
        zip(recordings, ranges, valid_times)
    ):
        for start, end in excluded:
            np.testing.assert_array_equal(
                combined.get_traces(
                    start_frame=offset + start, end_frame=offset + end
                ),
                0,
            )
        persisted = (
            ConcatenatedRecording.MemberBoundary & key & {"member_index": i}
        ).fetch1("member_valid_times")
        np.testing.assert_array_equal(persisted, valid)
        offset += recording.get_num_samples()
    assert len(observed) == 1
    assert combined.get_num_samples() == offset
    assert np.diff(row["obs_intervals"], axis=1).sum() < row["total_duration_s"]

    if motion == "none":
        before = combined.get_traces()
        Path(AnalysisNwbfile.get_abs_path(row["analysis_file_name"])).unlink()
        after = ConcatenatedRecording().get_recording(key)
        np.testing.assert_array_equal(after.get_traces(), before)
        assert len(observed) == 2

    # Inspect the actual sorter input, then plant two deterministic units so
    # the test covers interval propagation even if a detector finds no units.
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2._pipeline_reporting import (
        _observed_duration_s,
    )
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._smoke_constants import SMOKE_CLUSTERLESS_PARAMS

    def sort_masked(sorter, sorter_params, recording, sorting_id, **kwargs):
        cursor = 0
        samples, labels = [], []
        for member, excluded in zip(recordings, ranges):
            for start, end in excluded:
                np.testing.assert_array_equal(
                    recording.get_traces(
                        start_frame=cursor + start, end_frame=cursor + end
                    ),
                    0,
                )
            samples.extend([cursor + 100, cursor + 200])
            labels.extend([0, 1])
            cursor += member.get_num_samples()
        return NumpySorting.from_samples_and_labels(
            [np.asarray(samples)],
            [np.asarray(labels)],
            recording.get_sampling_frequency(),
        )

    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "artifact_concat_test",
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    sorting_key = SortingSelection.insert_selection(
        {
            **key,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "artifact_concat_test",
        }
    )
    monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(sort_masked))
    Sorting.populate(sorting_key, reserve_jobs=False)
    # Both analyzer rebuild routes must see the exact materialized mask, not
    # reload an unmasked member or apply member wall-clock intervals to concat.
    from spyglass.spikesorting.v2._sorting_analyzer import (
        reconstruct_recording_and_sorting,
        reconstruct_recording_for_sorting_from_resolved,
    )

    rebuilt, canonical_sorting = reconstruct_recording_and_sorting(
        Sorting(), sorting_key
    )
    resolved = reconstruct_recording_for_sorting_from_resolved(
        recording_row=row, source_kind="concatenated_recording"
    )
    for recording in (rebuilt, resolved):
        np.testing.assert_array_equal(
            recording.get_traces(), combined.get_traces()
        )
    assert set(canonical_sorting.unit_ids) == {0, 1}
    assert _observed_duration_s(sorting_key["sorting_id"]) == pytest.approx(
        sum(
            (
                member.get_num_samples()
                - sum(end - start for start, end in excluded)
            )
            / member.sampling_frequency
            for member, excluded in zip(recordings, ranges)
        )
    )
    root = CurationV2.insert_curation(
        sorting_key=sorting_key, labels={0: ["accept"], 1: ["accept"]}
    )
    merged = CurationV2.create_merged_curation(
        sorting_key=sorting_key,
        parent_curation_id=root["curation_id"],
        merge_groups=[[0, 1]],
    )
    for table, table_key in [
        (Sorting, sorting_key),
        (CurationV2, root),
        (CurationV2, merged),
    ]:
        path = AnalysisNwbfile.get_abs_path(
            (table & table_key).fetch1("analysis_file_name")
        )
        _, _, intervals = read_units_abs_times_and_sample_indices(path)
        for valid in intervals.values():
            np.testing.assert_array_equal(valid, row["obs_intervals"])
    # Artifact restrictions include source-owned member detections; a masked
    # concat cannot leak into a request for no artifact pass.
    assert not (
        CurationV2.resolve_restriction({"artifact_detection_id": None}) & merged
    )
    assert (
        CurationV2.resolve_restriction(
            {"artifact_detection_id": detection_ids[0]}
        )
        & merged
    )
    ConcatMemberCuration.populate(merged, reserve_jobs=False)
    for member in (ConcatMemberCuration & merged).fetch(as_dict=True):
        path = AnalysisNwbfile.get_abs_path(member["analysis_file_name"])
        times, frames, intervals = read_units_abs_times_and_sample_indices(path)
        i = member["member_index"]
        for uid, valid in intervals.items():
            np.testing.assert_array_equal(valid, valid_times[i])
            np.testing.assert_allclose(
                times[uid], recordings[i].get_times()[frames[uid]]
            )

    # The analysis snapshot uses the same global unit decision on each
    # member's real session timeline, through the downstream group accessor.
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )

    selection = select_units_for_analysis(merged)
    try:
        from spyglass.spikesorting.v2.curation_api import CurationRef

        evaluation = CurationRef.from_key(merged).evaluate(
            metric_params_name="minimal", auto_curation_rules_name="none"
        )
        metrics = evaluation.metrics
        # The planted sorter deliberately returned events in the masked interval.
        # Observed metrics count only available events and use sample exposure;
        # the original SI firing-rate column keeps its own full-timeline meaning.
        observed_count = sum(
            not any(start <= frame < stop for start, stop in excluded)
            for excluded in ranges
            for frame in (100, 200)
        )
        duration = _observed_duration_s(sorting_key["sorting_id"])
        assert metrics.loc[2, "observed_duration_s"] == pytest.approx(duration)
        assert metrics.loc[2, "observed_firing_rate_hz"] == pytest.approx(
            observed_count / duration
        )
        assert metrics.loc[2, "firing_rate"] == pytest.approx(
            4 * combined.sampling_frequency / combined.get_num_samples()
        )
        assert selection.included_unit_ids == (2,)
        assert len(selection.groups) == len(recordings)
        for selected in selection.groups:
            _, identities = SortedSpikesGroup.fetch_spike_data(
                dict(selected.group_key), return_unit_ids=True
            )
            assert identities == [
                {"spikesorting_merge_id": selected.merge_id, "unit_id": 2}
            ]
            observed = selected.observation
            assert observed.duration_s > 0 and not observed.unknown_sources
            i = selected.member_index
            member = recordings[i]
            expected = (
                member.get_num_samples()
                - sum(end - start for start, end in ranges[i])
            ) / member.sampling_frequency
            assert observed.duration_s == pytest.approx(expected)
            assert not observed.contains([member.get_times()[200]])[0]
    finally:
        for selected in selection.groups:
            (SortedSpikesGroup & dict(selected.group_key)).super_delete(
                warn=False
            )

    # The nullable detection FK must cascade through the concat master, not
    # just remove one snapshot member and leave a truncated selection behind.
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    member_keys = (ConcatMemberCuration & merged).fetch("KEY")
    (
        RecordingArtifactDetection & {"artifact_detection_id": detection_ids[0]}
    ).cascade_delete(safemode=False)
    assert not (ConcatenatedRecordingSelection & key)
    assert not (ConcatenatedRecordingSelection & different)
    assert not (Sorting & sorting_key)
    assert not (CurationV2 & sorting_key)
    assert not (ConcatMemberCuration & merged)
    assert not (SpikeSortingOutput.ConcatMemberCuration & member_keys)


@pytest.mark.slow
def test_member_artifact_failure_retry_and_reuse(
    chronic_2_session_minirec, monkeypatch
):
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2 import _pipeline_presets as presets
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
    )
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import SorterParameters, Sorting
    from tests.spikesorting.v2._smoke_constants import SMOKE_CLUSTERLESS_PARAMS

    fixture = chronic_2_session_minirec
    name = "artifact_retry_test"
    SessionGroup.create_group(
        fixture["owner"], name, fixture["same_day_members"]
    )
    first = Recording().get_recording(fixture["recording_pks"][0])
    threshold = float(
        np.quantile(np.abs(first.get_traces(return_in_uV=True)), 0.999)
    )
    ArtifactDetectionParameters.insert1(
        {
            "artifact_detection_params_name": name,
            "params": {
                "amplitude_threshold_uv": threshold,
                "proportion_above_threshold": 0.25,
                "min_length_s": 0.001,
            },
        },
        allow_duplicate_params=True,
    )
    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": name,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
        },
        allow_duplicate_params=True,
    )
    base = presets._PIPELINE_PRESETS[
        "franklab_probe_hippocampus_30khz_ms5_2026_06"
    ]
    monkeypatch.setitem(
        presets._PIPELINE_PRESETS,
        name,
        base.model_copy(
            update={
                "preprocessing_params_name": fixture[
                    "preprocessing_params_name"
                ],
                "artifact_detection_params_name": name,
                "motion_correction_params_name": "none",
                "sorter": "clusterless_thresholder",
                "sorter_params_name": name,
            }
        ),
    )
    original_populate = RecordingArtifactDetection.populate
    calls = []

    def interrupt_second_member(key, **kwargs):
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("Interrupted member detection")
        return original_populate(key, **kwargs)

    monkeypatch.setattr(
        RecordingArtifactDetection,
        "populate",
        staticmethod(interrupt_second_member),
    )
    monkeypatch.setattr(
        Sorting,
        "_run_sorter",
        staticmethod(
            lambda sorter, sorter_params, recording, sorting_id, **kwargs: NumpySorting.from_unit_dict(
                {0: np.array([100, 1000, 10000])},
                recording.get_sampling_frequency(),
            )
        ),
    )
    request = {
        "concat_session_group_owner": fixture["owner"],
        "concat_session_group_name": name,
        "pipeline_preset": name,
    }
    with pytest.raises(PipelineStageError) as error:
        run_v2_pipeline(**request)
    assert error.value.stage == "member_artifact_detection"
    completed = error.value.partial_run_summary["member_artifacts"]
    assert len(completed) == 1 and completed[0]["masked_duration_s"] > 0
    retry = run_v2_pipeline(**request)
    assert [member["status"] for member in retry["member_artifacts"]] == [
        "reused",
        "computed",
    ]
    assert retry["artifact_masked_duration_s"] == pytest.approx(
        sum(member["masked_duration_s"] for member in retry["member_artifacts"])
    )
    again = run_v2_pipeline(**request)
    assert again["sorting_id"] == retry["sorting_id"]
    assert again["concat_recording_id"] == retry["concat_recording_id"]
    assert again["member_artifact_detection_status"] == "reused"
    assert again["sorting_status"] == "reused"

    # A detection that leaves no usable time must identify the failing member,
    # even when its recording and detection were already populated.
    artifact_id = again["member_artifacts"][1]["artifact_detection_id"]
    recording_id = again["member_recording_ids"][1]
    original_intervals = (
        RecordingArtifactDetection.get_artifact_removed_intervals
    )

    def empty_second_member(self, key, as_dict=False):
        if key["artifact_detection_id"] == artifact_id:
            return np.empty((0, 2))
        return original_intervals(self, key, as_dict=as_dict)

    monkeypatch.setattr(
        RecordingArtifactDetection,
        "get_artifact_removed_intervals",
        empty_second_member,
    )
    with pytest.raises(PipelineStageError) as error:
        run_v2_pipeline(**request)
    assert error.value.stage == "member_artifact_detection"
    assert str(artifact_id) in str(error.value)
    assert str(recording_id) in str(error.value)
    assert len(error.value.partial_run_summary["member_artifacts"]) == 1
