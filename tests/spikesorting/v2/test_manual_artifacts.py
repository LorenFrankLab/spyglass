"""Manual exclusions compose with automatic masks without changing time axes."""

import numpy as np
import pytest

from tests.spikesorting.v2._sorter_stub import plant_sorter


@pytest.mark.parametrize("disjoint", [False, True])
def test_manual_exclusions_mask_exact_samples_and_keep_detected_artifacts(
    disjoint,
):
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._recording.concat import (
        mask_member_recordings,
    )
    from spyglass.spikesorting.v2._artifacts.manual import (
        apply_manual_exclusions,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    recording = NumpyRecording(np.ones((1000, 2), dtype="float32"), 1000)
    times = 10 + np.arange(1000) / 1000
    if disjoint:
        times[500:] += 5
    recording.set_times(times)
    # Existing automatic artifact [100, 110), preserving the recording gap.
    valid = np.array([[times[0], times[100]], [times[110], times[-1]]])
    if disjoint:
        valid = np.array(
            [
                [times[0], times[100]],
                [times[110], times[499]],
                [times[500], times[-1]],
            ]
        )
    excluded = [
        [times[200], times[210]],
        [times[499], times[499] + 0.0005],
        [times[-1], times[-1] + 0.001],
    ]
    updated = apply_manual_exclusions(valid, excluded, 0)
    expected = np.ones((1000, 2), dtype="float32")
    expected[100:110] = 0
    expected[200:210] = 0
    expected[499] = 0
    expected[-1] = 0
    np.testing.assert_array_equal(
        apply_artifact_mask(recording, updated).get_traces(), expected
    )
    members, _ = mask_member_recordings([recording], [updated])
    np.testing.assert_array_equal(members[0].get_traces(), expected)
    np.testing.assert_array_equal(members[0].get_times(), times)


def test_manual_exclusion_keeps_short_interval_it_does_not_touch():
    """``min_length_s`` drops slivers an exclusion cuts, not an untouched
    valid interval shorter than it."""
    from spyglass.spikesorting.v2._artifacts.manual import (
        apply_manual_exclusions,
    )

    valid = np.array([[0.0, 10.0], [20.0, 20.5], [30.0, 40.0]])
    updated = apply_manual_exclusions(valid, [[35.0, 39.5]], 1.0)
    np.testing.assert_array_equal(
        updated, [[0.0, 10.0], [20.0, 20.5], [30.0, 35.0]]
    )


def test_manual_exclusion_normalization_and_identity():
    import uuid

    from spyglass.spikesorting.v2._artifacts.manual import (
        normalize_manual_exclusions,
        resolve_manual_exclusions,
    )
    from spyglass.spikesorting.v2._core.selection_identity import (
        artifact_detection_identity_payload,
    )

    intervals = [[3, 4], [1, 2], [2, 3]]
    assert normalize_manual_exclusions(intervals) == [[1, 4]]
    assert resolve_manual_exclusions({0: []}, concat=True) == {}
    with pytest.raises(ValueError):
        resolve_manual_exclusions({1.5: [[1, 2]]}, concat=True)
    with pytest.raises(TypeError, match="member_index mapping is for"):
        resolve_manual_exclusions({0: [[1, 2]]}, concat=False)
    with pytest.raises(TypeError, match="must map member_index"):
        resolve_manual_exclusions([[1, 2]], concat=True)
    with pytest.raises(ValueError):
        normalize_manual_exclusions([[2, 1]])
    kwargs = {
        "recording_id": uuid.uuid4(),
        "artifact_detection_params_name": "none",
    }
    a = artifact_detection_identity_payload(**kwargs)
    b = artifact_detection_identity_payload(
        **kwargs, manual_excluded_times=intervals
    )
    assert a != b
    assert b == artifact_detection_identity_payload(
        **kwargs, manual_excluded_times=[[1, 4]]
    )


def test_manual_exclusions_outside_sorted_time_are_reported():
    """An exclusion that would remove no sample is named with the session's
    sorted time range, so a seconds-from-start mistake is visible."""
    from spyglass.spikesorting.v2._artifacts.manual import (
        ineffective_manual_exclusions,
        manual_exclusion_problem,
    )

    t0 = 1.65e9
    valid = [[t0, t0 + 100.0], [t0 + 200.0, t0 + 300.0]]
    exclusions = [
        [10.0, 20.0],  # seconds from start
        [t0 + 50.0, t0 + 60.0],  # inside valid time
        [t0 + 120.0, t0 + 150.0],  # entirely inside the gap
        [t0 + 300.0, t0 + 301.0],  # removes the last sample
        [t0 + 160.0, t0 + 210.0],  # starts in the gap, overlaps valid time
        [t0 - 1.0, t0],  # half-open: ends where valid time starts
    ]
    assert ineffective_manual_exclusions(exclusions, valid) == [
        [10.0, 20.0],
        [t0 - 1.0, t0],
        [t0 + 120.0, t0 + 150.0],
    ]
    assert ineffective_manual_exclusions([[t0 + 50, t0 + 60]], valid) == []
    message = manual_exclusion_problem(
        exclusions,
        valid,
        nwb_file_name="x_.nwb",
        interval_list_name="raw data valid times",
    )
    assert "[10.0, 20.0)" in message
    assert "1650000000.000 to 1650000300.000 s" in message
    assert "not seconds from the recording start" in message
    assert "x_.nwb" in message
    assert (
        manual_exclusion_problem(
            [[t0 + 50, t0 + 60]],
            valid,
            nwb_file_name="x_.nwb",
            interval_list_name="raw data valid times",
        )
        is None
    )


def test_manual_exclusions_enable_the_stage_without_mutating_the_preset():
    from spyglass.spikesorting.v2._artifacts.manual import (
        artifact_recipe_with_manual_exclusions,
    )
    from spyglass.spikesorting.v2._orchestration.presets import _PipelinePreset

    preset = _PipelinePreset(
        preprocessing_params_name="p",
        sorter="s",
        sorter_params_name="s",
        metric_params_name="m",
        auto_curation_rules_name="r",
    )
    enabled = artifact_recipe_with_manual_exclusions(preset, [[10, 11]])
    assert enabled.artifact_detection_params_name == "none"
    assert preset.artifact_detection_params_name is None


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("concat", [False, True], ids=["single", "concat"])
@pytest.mark.parametrize(
    "automatic", [False, True], ids=["manual-only", "combined"]
)
def test_pipeline_runner_applies_manual_exclusions(
    chronic_2_session_minirec, monkeypatch, request, concat, automatic
):
    """Public runner inputs reach the exact samples passed to the sorter."""
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2._orchestration import presets
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        run_v2_concat_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.recording import Recording
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
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    fixture = chronic_2_session_minirec
    name = f"manual_runner_{concat}_{automatic}"

    def cleanup_case():
        # Custom recipes must not become content twins in later lookup tests
        # or collide on a repeat run. Remove only this case's dependent rows,
        # leaving the module-scoped chronic Recording inputs available.
        sorter_key = {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": name,
        }
        for key in (SortingSelection & sorter_key).fetch("KEY", as_dict=True):
            clear_curations_for(key)
            (Sorting & key).super_delete(warn=False)
            (SortingSelection & key).super_delete(warn=False)
        if concat:
            group_key = {
                "session_group_owner": fixture["owner"],
                "session_group_name": name,
            }
            for key in (ConcatenatedRecordingSelection & group_key).fetch(
                "KEY", as_dict=True
            ):
                (ConcatenatedRecording & key).super_delete(warn=False)
                (ConcatenatedRecordingSelection & key).super_delete(warn=False)
            (SessionGroup & group_key).super_delete(warn=False)
        params_key = {"artifact_detection_params_name": name}
        for key in (RecordingArtifactSelection & params_key).fetch(
            "KEY", as_dict=True
        ):
            if RecordingArtifactDetection & key:
                (RecordingArtifactDetection & key).delete(safemode=False)
            (RecordingArtifactSelection & key).super_delete(warn=False)
        (ArtifactDetectionParameters & params_key).super_delete(warn=False)
        (SorterParameters & sorter_key).super_delete(warn=False)

    request.addfinalizer(cleanup_case)
    recording_keys = fixture["recording_pks"][: 2 if concat else 1]
    recordings = [Recording().get_recording(key) for key in recording_keys]
    originals = [recording.get_traces() for recording in recordings]
    exclusions, frame_ranges = {}, []
    for index, recording in enumerate(recordings):
        start = int((2 + 0.5 * index) * recording.sampling_frequency)
        stop = start + int(0.01 * recording.sampling_frequency)
        frame_ranges.append((start, stop))
        exclusions[index] = [
            [
                float(recording.sample_index_to_time(start)),
                float(recording.sample_index_to_time(stop)),
            ]
        ]
        assert np.any(originals[index][start:stop] != 0)

    if automatic:
        ArtifactDetectionParameters.insert1(
            {
                "artifact_detection_params_name": name,
                "params": {
                    "amplitude_threshold_uv": float(
                        np.quantile(
                            np.abs(recordings[0].get_traces(return_in_uV=True)),
                            0.999,
                        )
                    ),
                    "proportion_above_threshold": 0.25,
                    "min_length_s": 0.001,
                },
            },
            allow_duplicate_params=True,
        )
    # Distinct sorter rows keep the parametrized cases independent even when
    # their single-session recording and manual intervals are identical.
    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": name,
            "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        },
        allow_duplicate_params=True,
    )
    # The fixture's recordings use the generic "default" preprocessing row.
    preset = presets._PIPELINE_PRESETS[
        "franklab_clusterless_2026_11"
    ].model_copy(
        update={
            "preprocessing_params_name": fixture["preprocessing_params_name"],
            "artifact_detection_params_name": name if automatic else None,
            "sorter_params_name": name,
        }
    )
    monkeypatch.setitem(presets._PIPELINE_PRESETS, name, preset)
    received = []

    def sort(sorter, sorter_params, recording, sorting_id, **kwargs):
        received.append(recording.get_traces())
        return NumpySorting.from_unit_dict(
            {0: np.array([100, 1000, 10000])}, recording.sampling_frequency
        )

    plant_sorter(monkeypatch, sort)
    if concat:
        SessionGroup.create_group(
            fixture["owner"], name, fixture["same_day_members"]
        )
        inputs = {
            "session_group_owner": fixture["owner"],
            "session_group_name": name,
            "manual_excluded_times": exclusions,
        }
    else:
        inputs = {
            **fixture["same_day_members"][0],
            "team_name": fixture["owner"],
            "manual_excluded_times": exclusions[0],
        }
    run = run_v2_concat_pipeline if concat else run_v2_pipeline
    result = run(**inputs, pipeline_preset=name)
    assert len(received) == 1
    expected = np.concatenate(originals)
    manual_frames = np.zeros(len(expected), dtype=bool)
    offset = 0
    for original, (start, stop) in zip(originals, frame_ranges, strict=True):
        manual_frames[offset + start : offset + stop] = True
        offset += len(original)
    np.testing.assert_array_equal(received[0][manual_frames], 0)
    if automatic:
        assert np.any(
            ~manual_frames
            & np.any(expected != 0, axis=1)
            & np.all(received[0] == 0, axis=1)
        ), "Automatic artifacts must remain masked alongside manual exclusions."
    else:
        expected[manual_frames] = 0
        np.testing.assert_array_equal(received[0], expected)

    artifact_ids = (
        [
            member["artifact_detection_id"]
            for member in result["member_artifacts"]
        ]
        if concat
        else [result["artifact_detection_id"]]
    )
    for index, artifact_id in enumerate(artifact_ids):
        selection = (
            RecordingArtifactSelection & {"artifact_detection_id": artifact_id}
        ).fetch1()
        assert selection["artifact_detection_params_name"] == (
            name if automatic else "none"
        )
        np.testing.assert_array_equal(
            selection["manual_excluded_times"], exclusions[index]
        )
    # Identical requests reuse the sort, including the manual-mask identity.
    again = run(**inputs, pipeline_preset=name)
    assert again["sorting_id"] == result["sorting_id"]
    assert again["sorting_status"] == "reused"
    assert len(received) == 1


@pytest.mark.integration
def test_runners_reject_exclusions_outside_sorted_time(
    chronic_2_session_minirec,
):
    """Seconds counted from the recording start overlap no session time; the
    runners refuse them before any artifact row is written, naming the
    interval and the session range."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection
    from spyglass.spikesorting.v2.exceptions import (
        IneffectiveManualExclusionError,
        PreflightError,
    )
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_concat_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup

    initialize_v2_defaults()
    fixture = chronic_2_session_minirec
    member = fixture["same_day_members"][0]
    t0 = float(
        (
            IntervalList
            & {
                "nwb_file_name": member["nwb_file_name"],
                "interval_list_name": "raw data valid times",
            }
        ).fetch1("valid_times")[0][0]
    )
    before_start = [[t0 - 10.0, t0 - 9.0]]
    inputs = {
        **member,
        "team_name": fixture["owner"],
        "pipeline_preset": "franklab_clusterless_2026_11",
    }
    n_selections = len(RecordingArtifactSelection())

    report = preflight_v2_pipeline(**inputs, manual_excluded_times=before_start)
    (check,) = [
        c
        for c in report.checks
        if c.name == "manual_exclusions_overlap_sorted_time"
    ]
    assert check.ok is False
    assert f"[{t0 - 10.0!r}, {t0 - 9.0!r})" in check.fix
    assert f"{t0:.3f} to " in check.fix
    assert member["nwb_file_name"] in check.fix

    with pytest.raises(PreflightError, match="not seconds from the recording"):
        run_v2_pipeline(**inputs, manual_excluded_times=before_start)
    with pytest.raises(
        IneffectiveManualExclusionError, match="insert_selection"
    ):
        run_v2_pipeline(
            **inputs, manual_excluded_times=before_start, preflight=False
        )
    assert len(RecordingArtifactSelection()) == n_selections

    group = {
        "session_group_owner": fixture["owner"],
        "session_group_name": "manual_exclusion_outside_time",
    }
    SessionGroup.create_group(
        fixture["owner"],
        group["session_group_name"],
        fixture["same_day_members"],
    )
    try:
        with pytest.raises(PreflightError, match="concat member 1"):
            run_v2_concat_pipeline(
                **group,
                pipeline_preset="franklab_clusterless_2026_11",
                manual_excluded_times={1: before_start},
            )
        with pytest.raises(PreflightError, match=r"member_index \[5\]"):
            run_v2_concat_pipeline(
                **group,
                pipeline_preset="franklab_clusterless_2026_11",
                manual_excluded_times={5: [[t0 + 1.0, t0 + 2.0]]},
            )
    finally:
        (SessionGroup & group).super_delete(warn=False)
    assert len(RecordingArtifactSelection()) == n_selections
