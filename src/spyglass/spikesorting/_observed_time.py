"""Observed-time availability shared by the merge table and unit groups.

Intervals here are half-open seconds. A unit's observation snapshot stores
each distinct interval mask once, with per-unit references into it; a
population's availability is the intersection of its selected units' masks.

This module depends only on NumPy: it declares no ``dj.schema`` and imports no
schema module, so database-free code can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spyglass.spikesorting._intervals import intersect_intervals
from spyglass.spikesorting._numerical import finite_intervals, finite_vector


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
