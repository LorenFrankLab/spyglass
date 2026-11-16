"""Clusterless detections keep their frame and its recorded timestamp.

Planted deltas sit on both sides of an acquisition gap, so a rate-based
``frame / fs`` clock would put every post-gap spike seconds early.
"""

from __future__ import annotations

import numpy as np

_FS = 30_000.0
_N = 60_000
_GAP_FRAME = 30_000
_GAP_S = 5.0
# (frame, channel); includes the last pre-gap and the first post-gap frame.
_PLANTED = [
    (1_000, 0),
    (9_000, 2),
    (29_999, 1),
    (_GAP_FRAME, 3),
    (31_000, 0),
    (45_500, 2),
    (59_000, 1),
]


def _gapped_recording():
    import spikeinterface as si

    traces = np.zeros((_N, 4), dtype=np.float32)
    for frame, channel in _PLANTED:
        traces[frame, channel] = -300.0
    recording = si.NumpyRecording(traces, sampling_frequency=_FS)
    recording.set_dummy_probe_from_locations(
        np.array([[0, 0], [0, 200], [0, 400], [0, 600]])
    )
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    timestamps = 1_000.0 + np.arange(_N) / _FS
    timestamps[_GAP_FRAME:] += _GAP_S
    recording.set_times(timestamps, with_warning=False)
    return recording, timestamps


def test_clusterless_frames_and_stored_times_follow_the_recorded_clock():
    """Each planted delta is detected within one sample, and the stored
    spike time is the recording's timestamp at the detected frame."""
    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )
    from spyglass.spikesorting.v2._sorting.dispatch import (
        resolve_sort_config,
        run_clusterless_thresholder,
    )
    from spyglass.spikesorting.v2._storage.units_nwb import (
        _sample_indices_to_times_by_unit,
    )

    recording, timestamps = _gapped_recording()
    config = resolve_sort_config(
        "clusterless_thresholder",
        ClusterlessThresholderSchema().model_dump(),
        job_kwargs={"n_jobs": 1, "progress_bar": False, "random_seed": 0},
    )
    sorting = run_clusterless_thresholder(
        config.si_sorter_params, recording, config.job_kwargs
    )
    (unit_id,) = sorting.unit_ids
    frames = np.sort(sorting.get_unit_spike_train(unit_id))
    planted = np.array([frame for frame, _ in _PLANTED])
    assert len(frames) == len(planted), frames
    assert np.max(np.abs(frames - planted)) <= 1, frames - planted

    times = _sample_indices_to_times_by_unit(recording, {0: frames})[0]
    np.testing.assert_array_equal(times, timestamps[frames])
    # The gap survives: the first post-gap spike is 5 s after the last pre-gap.
    assert times[3] - times[2] > _GAP_S
