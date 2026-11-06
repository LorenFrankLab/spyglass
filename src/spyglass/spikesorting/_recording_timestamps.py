"""Whole-recording timestamps shared by the v0, v1, and v2 pipelines.

This module depends only on NumPy: it declares no ``dj.schema`` and imports no
schema module, so database-free code can import it.
"""

import numpy as np


def _get_recording_timestamps(recording):
    """Return the timestamps of every segment of ``recording``, concatenated.

    ``recording.get_times()`` returns only one segment's times, so a
    multi-segment recording is stitched into a single ``(total_frames,)``
    float64 array. A single-segment recording returns ``get_times()``.

    Parameters
    ----------
    recording : si.BaseRecording

    Returns
    -------
    np.ndarray
        ``(total_frames,)`` timestamps in seconds.
    """
    num_segments = recording.get_num_segments()

    if num_segments <= 1:
        return recording.get_times()

    frames_per_segment = [0] + [
        recording.get_num_frames(segment_index=i) for i in range(num_segments)
    ]

    cumsum_frames = np.cumsum(frames_per_segment)
    total_frames = np.sum(frames_per_segment)

    timestamps = np.zeros((total_frames,))
    for i in range(num_segments):
        start_index = cumsum_frames[i]
        end_index = cumsum_frames[i + 1]
        timestamps[start_index:end_index] = recording.get_times(segment_index=i)

    return timestamps
