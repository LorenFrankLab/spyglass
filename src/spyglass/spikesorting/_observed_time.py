"""Observed-time availability shared by the merge table and unit groups.

- Intervals are half-open, in seconds.
- A snapshot stores each distinct interval mask once, referenced per unit.
- Population availability is the intersection of its units' masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spyglass.utils.interval import Interval
from spyglass.utils.numerical import finite_intervals, finite_vector


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
    """Convert ``observed_intervals_by_unit`` provenance to the compact form."""
    if "observed_intervals_by_unit" not in provenance:
        return provenance
    normalized = dict(provenance)
    by_unit = normalized.pop("observed_intervals_by_unit")
    normalized["observation_intervals"] = (
        None if by_unit is None else compact_observation_intervals(by_unit)
    )
    return normalized


def contains_times(intervals, times):
    """Return whether each time falls inside a row of ``intervals``.

    ``intervals`` are half-open, sorted by start, and do not overlap.
    """
    intervals = finite_intervals(intervals, name="contains_times intervals")
    # Keep the input shape of ``times``.
    times = np.asarray(times)
    shape = times.shape
    times = finite_vector(
        times.reshape(-1), name="contains_times times"
    ).reshape(shape)
    if not len(intervals):
        return np.zeros(times.shape, dtype=bool)
    indices = np.searchsorted(intervals[:, 0], times, side="right") - 1
    return (indices >= 0) & (times < intervals[np.maximum(indices, 0), 1])


def intersect_interval_sets(interval_sets):
    """Intersect ``(n, 2)`` sorted, non-overlapping interval arrays.

    Returns the ``(m, 2)`` spans in every set, with touching spans merged:
    the observed time of a merged unit or of a unit population. ``None``
    entries are skipped; no sets gives ``(0, 2)``.
    """
    sets = [
        np.asarray(s, dtype=float).reshape(-1, 2)
        for s in interval_sets
        if s is not None
    ]
    if not sets:
        return np.empty((0, 2), dtype=float)
    acc = sets[0]
    for other in sets[1:]:
        acc = (
            Interval(acc, warn=False)
            .intersect(other, merge_touching=True)
            .times.reshape(-1, 2)
        )
        if acc.size == 0:
            break
    return acc


@dataclass(frozen=True)
class ObservationAvailability:
    """Common availability of selected units; None means unknown coverage.

    Unknown sources add no restriction and are listed in ``unknown_sources``.
    Intervals must be finite, forward, sorted, and disjoint; they are stored
    as a read-only copy.
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
        return intersect_interval_sets([intervals, self.intervals])

    def valid_bins(self, time):
        """Only bins wholly inside an observed span are available."""
        time = finite_vector(
            time, name="ObservationAvailability valid_bins time"
        )
        if self.intervals is None:
            return np.ones(time.shape, dtype=bool)
        if not len(self.intervals) or not len(time):
            return np.zeros(time.shape, dtype=bool)
        # Matches SortedSpikesGroup's digitize(time[1:-1]); last bin is empty.
        ends = np.r_[time[1:], time[-1]]
        indices = np.searchsorted(self.intervals[:, 0], time, side="right") - 1
        return self.contains(time) & (
            ends <= self.intervals[np.maximum(indices, 0), 1]
        )


def population_availability(snapshots):
    """Combine selected-unit observation snapshots; ignore empty members."""
    masks = []
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
        masks.extend(
            finite_intervals(
                observation["masks"][index],
                name=f"population_availability source {source} intervals",
            )
            for index in indices
        )
    common = intersect_interval_sets(masks) if masks else None
    return ObservationAvailability(common, tuple(unknown))
