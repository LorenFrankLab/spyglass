"""Manual exclusions compose with automatic masks without changing time axes."""

import numpy as np
import pytest


@pytest.mark.parametrize("disjoint", [False, True])
def test_manual_exclusions_mask_exact_samples_and_keep_detected_artifacts(
    disjoint,
):
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._concat_recording import (
        mask_member_recordings,
    )
    from spyglass.spikesorting.v2._manual_artifacts import (
        apply_manual_exclusions,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
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


def test_manual_exclusion_normalization_and_identity():
    import uuid

    from spyglass.spikesorting.v2._manual_artifacts import (
        normalize_manual_exclusions,
        resolve_manual_exclusions,
    )
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
    )

    intervals = [[3, 4], [1, 2], [2, 3]]
    assert normalize_manual_exclusions(intervals) == [[1, 4]]
    assert resolve_manual_exclusions({0: []}, concat=True) == {}
    with pytest.raises(ValueError):
        resolve_manual_exclusions({1.5: [[1, 2]]}, concat=True)
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


def test_manual_exclusions_enable_the_stage_without_mutating_the_preset():
    from spyglass.spikesorting.v2._manual_artifacts import (
        artifact_recipe_with_manual_exclusions,
    )
    from spyglass.spikesorting.v2._pipeline_presets import _PipelinePreset

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
    chronic_2_session_minirec, monkeypatch, concat, automatic
):
    """Public runner inputs reach the exact samples passed to the sorter."""
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2 import _pipeline_presets as presets
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import SorterParameters, Sorting

    fixture = chronic_2_session_minirec
    name = f"manual_runner_{concat}_{automatic}"
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
    preset = presets._PIPELINE_PRESETS[
        "franklab_clusterless_2026_06"
    ].model_copy(
        update={
            "artifact_detection_params_name": name if automatic else None,
            "sorter_params_name": name,
            "motion_correction_params_name": "none" if concat else None,
        }
    )
    monkeypatch.setitem(presets._PIPELINE_PRESETS, name, preset)
    received = []

    def sort(sorter, sorter_params, recording, sorting_id, **kwargs):
        received.append(recording.get_traces())
        return NumpySorting.from_unit_dict(
            {0: np.array([100, 1000, 10000])}, recording.sampling_frequency
        )

    monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(sort))
    if concat:
        SessionGroup.create_group(
            fixture["owner"], name, fixture["same_day_members"]
        )
        inputs = {
            "concat_session_group_owner": fixture["owner"],
            "concat_session_group_name": name,
            "manual_excluded_times": exclusions,
        }
    else:
        inputs = {
            **fixture["same_day_members"][0],
            "team_name": fixture["owner"],
            "manual_excluded_times": exclusions[0],
        }
    result = run_v2_pipeline(**inputs, pipeline_preset=name)
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
    again = run_v2_pipeline(**inputs, pipeline_preset=name)
    assert again["sorting_id"] == result["sorting_id"]
    assert again["sorting_status"] == "reused"
    assert len(received) == 1
