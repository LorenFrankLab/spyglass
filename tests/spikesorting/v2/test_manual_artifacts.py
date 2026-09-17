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
