"""Observed sample spans, exposure metrics, and population availability.

Intervals here are half-open seconds. Convert the NWB/artifact endpoint
convention through the sorting mask before using it for duration or binning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from spyglass.spikesorting.v2._core.numerical import (
    finite_intervals,
    finite_scalar,
    finite_vector,
)
from spyglass.spikesorting.v2._core.signal_math import (
    _normalize,
    intersect_intervals,
)

OBSERVATION_VERSION = 2


def compact_observation_intervals(intervals_by_unit):
    """Store each distinct interval mask once, with stable per-unit references."""
    cached = {}
    masks, unit_mask_ids = [], {}
    for unit, intervals in sorted(
        intervals_by_unit.items(), key=lambda item: int(item[0])
    ):
        intervals = finite_intervals(
            intervals,
            name=f"compact_observation_intervals unit {unit} intervals",
        )
        key = intervals.tobytes()
        if key not in cached:
            cached[key] = len(masks)
            masks.append(intervals.tolist())
        unit_mask_ids[str(unit)] = cached[key]
    return {"masks": masks, "unit_mask_ids": unit_mask_ids}


def normalize_observation_provenance(provenance):
    """Read the former per-unit representation without changing its content."""
    if "observed_intervals_by_unit" not in provenance:
        return provenance
    normalized = dict(provenance)
    by_unit = normalized.pop("observed_intervals_by_unit")
    normalized["observation_intervals"] = (
        None if by_unit is None else compact_observation_intervals(by_unit)
    )
    return normalized


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


def contains_times(intervals, times):
    """Membership without a mask at the recording's acquisition rate."""
    intervals = finite_intervals(intervals, name="contains_times intervals")
    # Membership queries retain their scalar/array shape; the event-vector
    # requirement belongs to observed_metrics rather than this predicate.
    times = np.asarray(times)
    shape = times.shape
    times = finite_vector(
        times.reshape(-1), name="contains_times times"
    ).reshape(shape)
    if not len(intervals):
        return np.zeros(times.shape, dtype=bool)
    indices = np.searchsorted(intervals[:, 0], times, side="right") - 1
    return (indices >= 0) & (times < intervals[np.maximum(indices, 0), 1])


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


@dataclass(frozen=True)
class ObservationAvailability:
    """Common availability of selected units; None means unknown coverage.

    Unknown sources do not invent restrictions. Known restrictions still
    apply in a mixed population, and unknown sources remain explicit. Known
    windows must be finite, forward, sorted, and disjoint. Stored arrays are
    owned read-only snapshots, independent of the caller's array.
    """

    intervals: np.ndarray | None
    unknown_sources: tuple[str, ...] = ()

    def __post_init__(self):
        if self.intervals is None:
            return
        arr = finite_intervals(
            self.intervals,
            name="ObservationAvailability intervals",
            readonly=True,
        )
        if len(arr) > 1:
            bad = np.flatnonzero(arr[1:, 0] < arr[:-1, 1])
            if bad.size:
                i = int(bad[0])
                raise ValueError(
                    "ObservationAvailability: intervals must be sorted by "
                    f"start and disjoint; interval {arr[i].tolist()} "
                    f"overlaps the next interval {arr[i + 1].tolist()}."
                )
        object.__setattr__(self, "intervals", arr)

    @property
    def duration_s(self):
        return (
            None
            if self.intervals is None
            else float(np.diff(self.intervals, axis=1).sum())
        )

    def contains(self, times):
        if self.intervals is not None:
            return contains_times(self.intervals, times)
        times = np.asarray(times)
        checked = finite_vector(
            times.reshape(-1), name="ObservationAvailability contains times"
        )
        return np.ones(checked.shape, dtype=bool).reshape(times.shape)

    def restrict(self, intervals):
        intervals = finite_intervals(
            intervals, name="ObservationAvailability restrict intervals"
        )
        if self.intervals is None:
            return intervals
        return intersect_intervals(intervals, self.intervals)

    def valid_bins(self, time):
        """Only bins wholly inside an observed span are available."""
        time = finite_vector(
            time, name="ObservationAvailability valid_bins time"
        )
        if self.intervals is None:
            return np.ones(time.shape, dtype=bool)
        if not len(self.intervals) or not len(time):
            return np.zeros(time.shape, dtype=bool)
        # Match SortedSpikesGroup's digitize(time[1:-1]) convention. Its final
        # output slot has zero width; represent its timestamp's availability.
        ends = np.r_[time[1:], time[-1]]
        indices = np.searchsorted(self.intervals[:, 0], time, side="right") - 1
        return self.contains(time) & (
            ends <= self.intervals[np.maximum(indices, 0), 1]
        )


def population_availability(snapshots):
    """Combine selected-unit observation snapshots; ignore empty members."""
    common = None
    unknown = []
    for source, unit_ids, provenance in snapshots:
        if not len(unit_ids):
            continue
        observation = normalize_observation_provenance(provenance).get(
            "observation_intervals"
        )
        if observation is None:
            unknown.append(str(source))
            continue
        indices = dict.fromkeys(
            observation["unit_mask_ids"][str(u)] for u in unit_ids
        )
        for index in indices:
            mask = observation["masks"][index]
            intervals = finite_intervals(
                mask, name=f"population_availability source {source} intervals"
            )
            common = (
                intervals
                if common is None
                else intersect_intervals(common, intervals)
            )
    return ObservationAvailability(common, tuple(unknown))
