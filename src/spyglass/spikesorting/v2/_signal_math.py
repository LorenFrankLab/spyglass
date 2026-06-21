"""Pure signal / frame / interval math for the spike sorting tables.

Timestamp <-> frame conversion, interval consolidation, and the
absolute-time merge dedup used by the recording, artifact, sorting, and
curation stages. Split out of ``utils.py`` for cohesion; these functions
are pure (NumPy + ``logger`` only, no DataJoint / SpikeInterface table
access) and are re-exported from ``utils`` for backward compatibility.
"""

from __future__ import annotations


def assert_freq_max_below_nyquist(
    freq_max: float, sampling_frequency: float, *, context: str = ""
) -> None:
    """Raise if a bandpass ``freq_max`` is at or above the Nyquist (fs/2).

    SciPy's filter design requires ``freq_max < fs/2``; a value at or above it
    fails opaquely deep inside the design call, far from the misconfiguration.
    Checking here -- where ``fs`` is known but the Pydantic schema is not --
    gives an actionable message. ``context`` is prepended to name the call site
    (e.g. ``"apply_pre_motion_preprocessing: "``).
    """
    nyquist = sampling_frequency / 2.0
    if freq_max >= nyquist:
        raise ValueError(
            f"{context}bandpass freq_max={freq_max} Hz is at or above the "
            f"Nyquist frequency ({nyquist} Hz = sampling_rate/2 = "
            f"{sampling_frequency}/2); SciPy's filter design requires "
            "freq_max < Nyquist. Lower freq_max below Nyquist."
        )


def assert_positive_sampling_frequency(
    sampling_frequency, *, context: str = ""
) -> float:
    """Return ``sampling_frequency`` as a float, or raise if not finite and > 0.

    Frame <-> time conversion divides by the sampling frequency (or by the
    sample period ``1/fs``), so a zero, negative, NaN, or infinite value
    silently yields an infinite / NaN sample period and mis-maps every frame
    instead of failing. ``context`` is prepended to name the call site.
    """
    import numpy as np

    fs = float(sampling_frequency)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(
            f"{context}sampling frequency must be a finite positive number; "
            f"got {sampling_frequency!r}."
        )
    return fs


def assert_monotonic_timestamps(timestamps, *, context: str = "") -> None:
    """Raise if ``timestamps`` is empty or steps backward.

    ``searchsorted``-based frame mapping (``_consolidate_intervals`` /
    ``_spike_times_to_frames``) and the first/last-sample reads assume a
    non-empty, monotonically non-decreasing wall-clock vector; an out-of-order
    or empty vector silently mis-slices the recording rather than failing.
    Equal consecutive timestamps are allowed (``searchsorted`` handles
    duplicates); only a strictly backward step is rejected. ``context`` is
    prepended to name the call site.
    """
    import numpy as np

    ts = np.asarray(timestamps)
    if ts.size == 0:
        raise ValueError(f"{context}timestamp vector is empty.")
    if not np.all(np.isfinite(ts)):
        # ``np.diff`` across a NaN is NaN and ``NaN < 0`` is False, so the
        # backward-step check below silently passes a NaN-containing (and
        # NaN-masked out-of-order) vector -- yet ``searchsorted`` sorts NaN as
        # +inf and mis-brackets every spike after it. Reject non-finite first.
        raise ValueError(
            f"{context}timestamps contain non-finite values (NaN/inf); "
            "searchsorted-based frame mapping would silently mis-slice the "
            "recording (NaN sorts as +inf)."
        )
    if ts.size > 1 and bool(np.any(np.diff(ts) < 0)):
        raise ValueError(
            f"{context}timestamps must be monotonically non-decreasing "
            "(searchsorted-based frame mapping would otherwise silently "
            "mis-slice the recording); found a backward step."
        )


def _get_recording_timestamps(
    recording,
    override=None,
):
    """Return the absolute-time vector for a SpikeInterface recording.

    Two responsibilities:

    1. **Multi-segment NWB support.** SpikeInterface's
       ``recording.get_times()`` only returns the active-segment
       times; a multi-segment NWB (epoch-stitched recordings) would
       silently report just segment 0. This helper concatenates the
       per-segment timestamps into one ``(total_frames,)`` array so
       downstream code sees the whole-session timeline.

    2. **Caller-supplied persisted-timestamps override.** The make
       path persists the source's actual wall-clock timestamps (which
       SI's ``frame_slice`` / ``concatenate_recordings`` drop);
       ``_restrict_recording`` derives that vector and threads it back
       here so the saved start/end and the NWB write use the persisted
       times. ``override`` is that hook: when not ``None``, return it
       verbatim. Helpers called outside the make path (e.g.
       ``get_recording``) leave ``override=None`` and get the
       segment-aware ``get_times()`` concatenation.

    Parameters
    ----------
    recording : si.BaseRecording
        Source recording whose timestamps are needed.
    override : array-like, optional
        Pre-computed persisted timestamps to return verbatim. Callers
        that have the persisted timestamp vector pass it here so this
        helper does not re-derive it from the recording.

    Returns
    -------
    array-like
        ``(total_frames,)`` wall-clock seconds. Lazy timestamp overrides are
        returned as-is; otherwise the return is a ``numpy.ndarray``.
    """
    import numpy as np

    if override is not None:
        if getattr(override, "_spyglass_lazy_timestamps", False):
            return override
        return np.asarray(override)

    num_segments = recording.get_num_segments()
    if num_segments <= 1:
        return recording.get_times()

    frames_per_segment = [0] + [
        recording.get_num_frames(segment_index=i) for i in range(num_segments)
    ]
    cumsum_frames = np.cumsum(frames_per_segment)
    total_frames = int(cumsum_frames[-1])

    timestamps = np.zeros((total_frames,), dtype=np.float64)
    for i in range(num_segments):
        start_index = int(cumsum_frames[i])
        end_index = int(cumsum_frames[i + 1])
        timestamps[start_index:end_index] = recording.get_times(segment_index=i)
    return timestamps


def _consolidate_intervals(intervals, timestamps):
    """Convert ``(start_time, stop_time)`` second intervals to frame indices.

    Sorts and merges overlapping/adjacent intervals, then maps each onto a
    half-open ``[start_frame, end_frame_exclusive)`` pair. The end uses
    ``searchsorted(side="right")`` -- the count of timestamps ``<= stop_time``
    -- which is exactly the exclusive end ``frame_slice`` expects, so the
    final sample of each interval is retained.

    Parameters
    ----------
    intervals : array-like
        Iterable of ``(start_seconds, stop_seconds)`` tuples. The
        helper sorts and consolidates overlapping/adjacent
        intervals before returning.
    timestamps : numpy.ndarray
        Monotonically non-decreasing wall-clock timestamps for the recording
        (validated by ``assert_monotonic_timestamps`` before the searchsorted
        frame mapping).

    Returns
    -------
    numpy.ndarray
        ``(n_consolidated, 2)`` array of ``(start_frame,
        end_frame_exclusive)`` integer pairs suitable for
        ``recording.frame_slice(start_frame=..., end_frame=...)``.
    """
    import numpy as np

    intervals = np.asarray(intervals)
    if intervals.ndim == 1:
        intervals = intervals.reshape(-1, 2)
    if intervals.shape[1] != 2:
        raise ValueError("Input array must have shape (N_Intervals, 2).")

    # Sort defensively; stable ordering by start.
    if not np.all(intervals[:-1] <= intervals[1:]):
        intervals = intervals[np.argsort(intervals[:, 0])]

    assert_monotonic_timestamps(
        timestamps, context="_consolidate_intervals: "
    )
    start_indices = np.searchsorted(timestamps, intervals[:, 0], side="left")
    # Exclusive end: ``side="right"`` returns the count of timestamps <= value,
    # which is exactly the half-open end ``frame_slice`` expects.
    stop_indices = np.searchsorted(timestamps, intervals[:, 1], side="right")

    consolidated = []
    start, stop = int(start_indices[0]), int(stop_indices[0])
    for next_start, next_stop in zip(start_indices, stop_indices):
        next_start = int(next_start)
        next_stop = int(next_stop)
        # Overlap / adjacency in exclusive-end form: next_start <= stop
        # (== means strictly adjacent).
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            consolidated.append((start, stop))
            start, stop = next_start, next_stop

    consolidated.append((start, stop))
    return np.asarray(consolidated, dtype=np.int64)


def _spike_times_to_frames(recording_times, spike_times, n_samples, unit_id):
    """Map absolute spike times (seconds) to recording frame indices.

    Spike times are persisted in the recording's ABSOLUTE wall-clock
    timeline (``timestamps[frame]``). For disjoint sort intervals that
    timeline is non-uniform (it carries the wall-clock gaps that
    ``concatenate_recordings(ignore_times=True)`` drops), so the inverse
    map must compare against the actual timestamps -- NOT an affine
    ``round((t - t_start) * fs)``, which would shift every frame after a
    gap by the accumulated gap and can push frames past ``n_samples``.
    Bracket each spike with ``np.searchsorted`` and return the nearest
    timestamp so tiny serialization/round-trip noise (e.g.
    ``timestamps[i] + 1e-12``) does not shift the frame by +1.

    A spike more than two typical sample periods from its nearest timestamp
    raises: that is an alignment/units error (or a spike in a disjoint
    wall-clock gap), not floating-point roundoff. Near-boundary spikes within
    that tolerance are snapped to the nearest valid sample so the returned
    frame count always matches ``spike_times`` -- dropping them desyncs
    downstream per-spike features.

    Parameters
    ----------
    recording_times : np.ndarray, shape (n_samples,)
        Recording timestamps in seconds, monotonically increasing.
    spike_times : np.ndarray, shape (n_spikes,)
        Spike times for a single unit in seconds.
    n_samples : int
        Total number of samples in the recording.
    unit_id : int or str
        Identifier of the unit, used only in the warning message.

    Returns
    -------
    np.ndarray, shape (n_spikes,)
        Frame indices in ``[0, n_samples)``. A spike whose absolute time
        floating-point-rounds slightly away from the stored timestamp is
        snapped to the nearest sample (count preserved, so downstream
        per-spike features stay aligned). A spike genuinely outside the
        recording or inside a disjoint gap raises.

    Raises
    ------
    ValueError
        If a spike time is more than a couple of sample periods from the
        nearest timestamp -- an upstream alignment/units error, not FP
        rounding.
    """
    import numpy as np

    recording_times = np.asarray(recording_times, dtype=float)
    spike_times = np.atleast_1d(np.asarray(spike_times, dtype=float))
    n_samples = int(n_samples)
    if spike_times.size == 0:
        return np.asarray([], dtype=np.int64)
    if recording_times.size != n_samples:
        raise ValueError(
            "_spike_times_to_frames: recording_times length "
            f"({recording_times.size}) does not match n_samples ({n_samples})."
        )
    if n_samples == 0:
        raise ValueError(
            f"Unit {unit_id} has spike times but the recording has zero samples."
        )

    assert_monotonic_timestamps(
        recording_times, context=f"_spike_times_to_frames (unit {unit_id}): "
    )
    insert_indices = np.searchsorted(recording_times, spike_times, side="left")
    right_indices = np.clip(insert_indices, 0, n_samples - 1)
    left_indices = np.clip(insert_indices - 1, 0, n_samples - 1)

    left_dist = np.abs(spike_times - recording_times[left_indices])
    right_dist = np.abs(recording_times[right_indices] - spike_times)
    use_left = left_dist < right_dist
    spike_frames = np.where(use_left, left_indices, right_indices).astype(
        np.int64, copy=False
    )
    nearest_dist = np.where(use_left, left_dist, right_dist)

    if recording_times.size >= 2:
        diffs = np.diff(recording_times)
        positive_diffs = diffs[diffs > 0]
        sample_period = (
            float(np.median(positive_diffs)) if positive_diffs.size else 0.0
        )
    else:
        sample_period = 0.0
    tol = 2.0 * sample_period  # a couple of sample periods covers FP roundoff

    far_mask = nearest_dist > tol
    if bool(far_mask.any()):
        worst = int(np.argmax(nearest_dist))
        nearest_t = float(recording_times[spike_frames[worst]])
        raise ValueError(
            f"Unit {unit_id} has spike time(s) up to "
            f"{float(nearest_dist[worst]):.6g}s from the nearest recording "
            f"sample (spike_time={float(spike_times[worst]):.6g}s, "
            f"nearest_timestamp={nearest_t:.6g}s) -- more than {tol:.6g}s "
            "from any sample, so this is an alignment/units error, not "
            "floating-point rounding. Inspect the upstream spike times."
        )

    outside_mask = (spike_times < recording_times[0]) | (
        spike_times > recording_times[-1]
    )
    n_outside = int(outside_mask.sum())
    if n_outside > 0:
        from spyglass.utils import logger

        logger.warning(
            f"Unit {unit_id} has {n_outside} spike(s) just outside the "
            "recording timestamp envelope but within floating-point tolerance. "
            "Snapping to the nearest sample to keep the frame count aligned "
            "with the persisted spike_times."
        )
    return spike_frames


# Coincidence window (ms) for cross-unit duplicate-spike removal when
# merging units. Matches SpikeInterface's ``MergeUnitsSorting`` default.
# A neuron's refractory period (~1-2 ms) means a genuine spike train never
# has a sub-0.4 ms pair,
# so this only removes double-detections of one physical event shared
# across merged contributors. Lives here next to
# ``_dedup_merged_spike_times`` -- the algorithm it parameterizes -- so the
# ``curation`` schema module and ``_units_nwb`` import it from this pure
# layer rather than the lower-level ``_units_nwb`` reaching back into the
# schema module.
_MERGE_DEDUP_DELTA_MS = 0.4


def _dedup_merged_spike_times(times_list, delta_s):
    """Membership-aware duplicate-spike removal for a merged unit.

    Faithful time-space port of SpikeInterface's
    ``MergeUnitsSorting`` / ``get_non_duplicated_events``: concatenate the
    contributor spike trains, sort, and drop a spike ONLY when it is
    within ``delta_s`` seconds of the previous spike AND came from a
    DIFFERENT contributor -- i.e. a cross-unit double-detection of one
    physical event. A neuron's refractory period (~1-2 ms) guarantees a
    single unit never fires twice within the ~0.4 ms window, so a
    within-unit close pair is a genuine event and is never touched
    (``diff(membership) == 0`` keeps it). The first spike is always kept.

    Applied on BOTH the lazy preview path and the ``apply_merge=True`` staged
    path so the previewed and stored merged trains agree and neither retains
    the double-detection artifacts.

    Parameters
    ----------
    times_list : list[array-like]
        One spike-time array (seconds) per contributor unit.
    delta_s : float
        Coincidence window in seconds (e.g. ``0.4e-3``).

    Returns
    -------
    np.ndarray
        Sorted, deduplicated merged spike times (seconds).
    """
    import numpy as np

    arrays = [np.asarray(t, dtype=float) for t in times_list]
    concat = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    if concat.size == 0:
        return concat
    order = concat.argsort(kind="mergesort")
    times_sorted = concat[order]
    membership = np.concatenate(
        [np.full(arr.shape, i) for i, arr in enumerate(arrays)]
    )[order]
    # Keep a spike iff it is far enough from the previous one OR shares
    # the previous one's contributor (mirrors SI's
    # ``(diff(times) > delta) | (diff(membership) == 0)``); always keep
    # the first.
    keep = np.nonzero(
        (np.diff(times_sorted) > delta_s) | (np.diff(membership) == 0)
    )[0]
    keep = np.concatenate([[0], keep + 1])
    return times_sorted[keep]


def _base_intervals_from_timestamps(timestamps, fs):
    """Split a (possibly gap-preserving) timestamp vector into recorded chunks.

    A disjoint v2 recording persists the concatenated per-sort-interval
    timestamp slices (``Recording._restrict_recording``'s
    ``timestamps_override``), so consecutive samples WITHIN a chunk differ
    by ~1 sample period while an inter-chunk wall-clock gap shows up as a
    diff > 1.5 sample periods (a missing sample). Returns one
    ``[start, end]`` (inclusive first/last sample times, seconds) per
    chunk so an artifact complement built per chunk never spans a gap. A
    contiguous recording yields a single ``[t0, t_end]`` interval, so this is
    a no-op
    for the common single-interval case.

    The ``1.5 / fs`` threshold is robust against sub-sample timestamp
    jitter (within-chunk diffs are ~1 sample period) and matches the
    ±1.5-sample tolerance used in the spike-time readback; any genuine
    disjoint gap is orders of magnitude larger.

    Parameters
    ----------
    timestamps : array-like, shape (n_samples,)
        Recording timestamps in seconds, monotonically increasing.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    list[list[float]]
        ``[[start, end], ...]`` inclusive per-chunk bounds; ``[]`` for an
        empty input.
    """
    import numpy as np

    ts = np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return []
    assert_monotonic_timestamps(
        ts, context="_base_intervals_from_timestamps: "
    )
    fs = assert_positive_sampling_frequency(
        fs, context="_base_intervals_from_timestamps: "
    )
    sample_period = 1.0 / float(fs)
    # Index i marks a gap when ts[i+1] - ts[i] exceeds 1.5 sample periods
    # (i.e. at least one sample of wall-clock time is missing).
    gap_after = np.flatnonzero(np.diff(ts) > 1.5 * sample_period)
    starts = [0, *(int(i) + 1 for i in gap_after)]
    ends = [*(int(i) for i in gap_after), ts.size - 1]
    return [[float(ts[s]), float(ts[e])] for s, e in zip(starts, ends)]
