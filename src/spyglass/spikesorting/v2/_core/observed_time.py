"""Observed sample spans and exposure metrics.

Intervals here are half-open seconds. Convert the NWB/artifact endpoint
convention through the sorting mask before using it for duration or binning.
Population availability, which the merge table and unit groups also read,
lives in ``spyglass.spikesorting._observed_time``.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from spyglass.spikesorting._intervals import _normalize
from spyglass.spikesorting._numerical import (
    finite_intervals,
    finite_scalar,
    finite_vector,
)
from spyglass.spikesorting._observed_time import contains_times

OBSERVATION_VERSION = 2


def observed_intervals(recording, valid_times):
    """Convert stored valid times into half-open, gap-preserving exposure."""
    from spyglass.spikesorting.v2._core.signal_math import (
        _segment_times_at,
        base_intervals_and_gaps,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        artifact_frame_ranges,
        complement_frame_ranges,
    )

    valid_times = finite_intervals(
        valid_times, name="observed_intervals valid_times"
    )
    if len(valid_times) == 0:
        return np.empty((0, 2))
    excluded = artifact_frame_ranges(recording, valid_times)
    n = recording.get_num_samples()
    fs = finite_scalar(
        recording.sampling_frequency,
        name="observed_intervals sampling_frequency",
        positive=True,
    )
    boundaries = np.r_[0, base_intervals_and_gaps(recording).gap_after + 1, n]
    kept = []
    for start, stop in complement_frame_ranges(excluded, n):
        cuts = np.r_[
            start,
            boundaries[(boundaries > start) & (boundaries < stop)],
            stop,
        ].astype(np.int64)
        for first, end in pairwise(cuts):
            t0 = float(_segment_times_at(recording, np.array([first]))[0])
            t1 = (
                float(_segment_times_at(recording, np.array([end - 1]))[0])
                + 1.0 / fs
            )
            kept.append((t0, t1))
    result = finite_intervals(
        kept, name="observed_intervals computed intervals"
    )
    if len(result) > 1:
        bad = np.flatnonzero(result[1:, 0] < result[:-1, 1])
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                "observed_intervals: computed intervals are not sorted and "
                f"disjoint; interval {result[i].tolist()} overlaps the next "
                f"interval {result[i + 1].tolist()}."
            )
    return result


def observed_metrics(
    spike_times, intervals, *, bin_duration_s=60.0, origin=0.0
):
    """Rate and exposure-weighted presence on the original time axis.

    Presence means at least one observed spike per fixed-width bin. Partly
    observed bins contribute only their observed duration; excluded bins
    contribute nothing. No observed exposure yields unavailable metrics.
    ``intervals`` are first sorted, merged and stripped of zero-length rows,
    so duplicated or overlapping time counts once. Malformed windows or event
    vectors raise rather than becoming missing exposure or discarded spikes.
    """
    spike_times = finite_vector(
        spike_times, name="observed_metrics spike_times"
    )
    intervals = finite_intervals(intervals, name="observed_metrics intervals")
    origin = finite_scalar(origin, name="observed_metrics origin")
    bin_duration_s = finite_scalar(
        bin_duration_s, name="observed_metrics bin_duration_s", positive=True
    )
    intervals = _normalize(intervals)
    duration = float(np.diff(intervals, axis=1).sum())
    spikes = spike_times[contains_times(intervals, spike_times)]
    exposure = {}
    for start, stop in intervals:
        first = int(np.floor((start - origin) / bin_duration_s))
        last = int(np.ceil((stop - origin) / bin_duration_s))
        for index in range(first, last):
            bin_start = origin + index * bin_duration_s
            seconds = min(stop, bin_start + bin_duration_s) - max(
                start, bin_start
            )
            if seconds > 0:
                exposure[index] = exposure.get(index, 0.0) + seconds
    occupied = set(
        np.floor((spikes - origin) / bin_duration_s).astype(np.int64)
    )
    return {
        "observed_duration_s": duration,
        "observed_firing_rate_hz": (
            len(spikes) / duration if duration else np.nan
        ),
        "observed_presence_ratio": (
            sum(
                seconds
                for index, seconds in exposure.items()
                if index in occupied
            )
            / duration
            if duration
            else np.nan
        ),
    }
