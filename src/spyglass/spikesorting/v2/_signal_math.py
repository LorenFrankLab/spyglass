"""Pure signal / frame / interval math for the spike sorting tables.

Timestamp <-> frame conversion, interval consolidation, and the
absolute-time merge dedup used by the recording, artifact, sorting, and
curation stages. Split out of ``utils.py`` for cohesion; these functions
are pure (NumPy + ``logger`` only, no DataJoint / SpikeInterface table
access) and are re-exported from ``utils`` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, NamedTuple


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


_MAX_ARTIFACT_FRAME_FRACTION = 0.5


def assert_artifact_frame_fraction(
    n_artifact_frames, n_samples, *, context: str = ""
) -> None:
    """Raise if the per-frame artifact set exceeds a sane fraction of the
    recording.

    Both the detection scan (``scan_artifact_frames``: frames flagged above
    threshold) and the sort-time mask (``apply_artifact_mask``: the complement
    of the kept ``valid_times``) materialize one int64 frame index per artifact
    sample. Under a misconfigured (too-loose) threshold, or a ``valid_times``
    that keeps almost nothing, that array is O(n_samples) -- hundreds of MB to
    GB on a long, many-channel recording -- and the subsequent per-frame pass is
    correspondingly slow. Past ``_MAX_ARTIFACT_FRAME_FRACTION`` (half the
    recording) this is a misconfiguration, not artifact removal: fail fast with
    the realized fraction rather than allocating the array. ``context`` names the
    call site so the message points at the right knob.

    ``n_samples <= 0`` forms no fraction and is a no-op -- the callers guard the
    empty-recording / zero-artifact cases separately.
    """
    if n_samples <= 0:
        return
    if n_artifact_frames > _MAX_ARTIFACT_FRAME_FRACTION * n_samples:
        from spyglass.spikesorting.v2.exceptions import (
            ArtifactFractionExceededError,
        )

        raise ArtifactFractionExceededError(
            f"{context}{n_artifact_frames} of {n_samples} samples "
            f"({100.0 * n_artifact_frames / n_samples:.1f}%) are artifact "
            f"frames, exceeding the "
            f"{100.0 * _MAX_ARTIFACT_FRAME_FRACTION:.0f}% guard. Materializing "
            "an index per artifact sample would allocate O(n_samples) memory "
            "and run a correspondingly slow per-frame pass; masking this much "
            "would also leave too little signal to sort. This indicates a "
            "misconfigured detector (raise amplitude_threshold_uv / "
            "zscore_threshold, lower proportion_above_threshold, or reduce "
            "removal_window_ms) or a valid_times override that keeps almost "
            "nothing."
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


def intersect_interval_sets(interval_sets):
    """Intersect a list of ``(n, 2)`` sorted, non-overlapping interval arrays.

    Returns the ``(m, 2)`` float array of time spans contained in EVERY input
    set -- the conservative observation window for a merged unit whose
    contributors have differing ``obs_intervals``. Each input is a
    ``(n, 2)`` ``[start, end]`` array assumed sorted by start and disjoint (the
    sort-time writer's ``obs_intervals`` shape). An empty input list returns an
    empty ``(0, 2)`` array; a single set returns itself. Identical inputs (the
    common single-sort case, where every unit shares one window) return that
    shared window unchanged.
    """
    import numpy as np

    sets = [
        np.asarray(s, dtype=float).reshape(-1, 2)
        for s in interval_sets
        if s is not None
    ]
    if not sets:
        return np.empty((0, 2), dtype=float)
    acc = sets[0]
    for other in sets[1:]:
        merged = []
        i = j = 0
        while i < len(acc) and j < len(other):
            lo = max(acc[i, 0], other[j, 0])
            hi = min(acc[i, 1], other[j, 1])
            if lo < hi:
                merged.append((lo, hi))
            # Advance whichever interval ends first (standard two-pointer
            # interval intersection over two sorted, disjoint sets).
            if acc[i, 1] < other[j, 1]:
                i += 1
            else:
                j += 1
        acc = np.asarray(merged, dtype=float).reshape(-1, 2)
        if acc.size == 0:
            break
    return acc


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

    assert_monotonic_timestamps(timestamps, context="_consolidate_intervals: ")
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
    jitter (within-chunk diffs are ~1 sample period); any genuine
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
    assert_monotonic_timestamps(ts, context="_base_intervals_from_timestamps: ")
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


# --------------------------------------------------------------------------- #
# Chunked / affine timestamp helpers (no full-vector materialization).
#
# A persisted v2 recording loads its timestamps as a LAZY vector
# (``read_nwb_recording(load_time_vector=True)`` stores the h5py-backed pynwb
# ``timestamps`` object; a saved recording mmaps it). SpikeInterface's
# ``recording.get_times()`` calls ``np.asarray(time_vector)`` -- it materializes
# the whole ``n_samples``-length float64 vector (~824 MB for 1 h @ 30 kHz, 8
# bytes/sample) AND caches it back on the segment. ``sample_index_to_time``
# instead indexes ``time_vector[frames]`` (a lazy h5py/mmap slice read) or, for
# a rate-based recording, computes ``frames / fs + t_start`` -- so these helpers
# map frames<->time and find gaps with peak memory bounded by the query/chunk
# size, not by ``n_samples``. ``sample_index_to_time(i)`` is bit-identical to
# ``get_times()[i]`` in both timestamp modes (pinned by the equivalence tests in
# ``tests/spikesorting/v2/test_signal_math.py``), so the outputs match the
# full-vector path exactly.
#
# MAINTAINER NOTE (SpikeInterface version): the pinned spikeinterface==0.104.3
# (see pyproject.toml -- pinned for sorter param schemas, unrelated to
# timestamps) exposes NO frame-bounded ``get_times``: calling
# ``recording.get_times(start_frame=..., end_frame=...)`` raises TypeError, and
# the bounds-less ``get_times()`` materializes (and caches) the whole vector.
# So these helpers map frames<->time exclusively through
# ``recording.sample_index_to_time(frames)``, which slices the lazy
# h5py/mmap timestamp vector (or computes affine times) without materializing.
# SpikeInterface > 0.104.3 refactored time handling into ``core/time_series.py``
# and ADDED a bounded ``get_times(start_frame, end_frame)`` that lazily slices.
# If/when the spikeinterface pin is raised past 0.104.3, ``base_intervals_and
# _gaps`` / ``timestamp_fingerprint`` chunk loops MAY switch their per-chunk
# ``_segment_times_at(recording, np.arange(start, end))`` to the (then-cheaper)
# ``recording.get_times(segment_index=0, start_frame=start, end_frame=end)``.
# This is an optional simplification, NOT a fix: ``sample_index_to_time`` is
# unchanged in the newer source and stays correct + lazy either way, so there is
# no urgency. The equivalence + bounded-memory tests guard both paths.
# --------------------------------------------------------------------------- #


def _recording_has_explicit_time_vector(recording, *, segment_index=0) -> bool:
    """Return whether ``recording`` carries explicit per-frame timestamps.

    Uses the public ``get_time_info`` predicate (``time_vector`` is not None),
    which returns the lazy ``time_vector`` attribute without materializing it,
    so this stays O(1) and never triggers a ``get_times()`` allocation. An
    extractor lacking ``get_time_info`` is treated as explicit so callers use
    the exact per-frame path rather than an affine shortcut.
    """
    try:
        return (
            recording.get_time_info(segment_index=segment_index).get(
                "time_vector"
            )
            is not None
        )
    except AttributeError:
        return True


def _segment_times_at(recording, frames, *, segment_index=0):
    """Absolute times (s) for arbitrary ``frames`` without materializing.

    Maps each frame index to its wall-clock time via
    ``recording.sample_index_to_time`` -- a lazy h5py/mmap slice read for an
    explicit ``time_vector`` or the affine ``frames / fs + t_start`` for a
    rate-based recording -- so the full 8-byte-per-sample vector is never built
    or cached. Fancy-indexing an h5py ``time_vector`` requires strictly
    increasing indices, but ``frames`` may be unsorted with duplicates (binary
    -search midpoints), so the read is done on the sorted unique indices and
    scattered back to the caller's order and shape.

    Parameters
    ----------
    recording : si.BaseRecording
    frames : array-like of int
        Frame indices in ``[0, n_samples)``.
    segment_index : int, optional

    Returns
    -------
    numpy.ndarray of float64
        Wall-clock seconds, same shape as ``frames``.
    """
    import numpy as np

    frames = np.asarray(frames, dtype=np.int64)
    flat = frames.reshape(-1)
    if flat.size == 0:
        return np.empty(frames.shape, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    uniq, inverse = np.unique(flat[order], return_inverse=True)
    vals = np.asarray(
        recording.sample_index_to_time(uniq, segment_index=segment_index),
        dtype=np.float64,
    )
    out = np.empty(flat.shape, dtype=np.float64)
    out[order] = vals[inverse]
    return out.reshape(frames.shape)


def frames_for_times(recording, times_s, *, segment_index=0):
    """Frame index per query time, == ``searchsorted(get_times(), t, "left")``.

    Returns, for each ``times_s[k]``, the smallest frame ``i`` in
    ``[0, n_samples]`` with ``get_times()[i] >= times_s[k]`` -- exactly
    ``numpy.searchsorted(recording.get_times(), times_s, side="left")``, the
    half-open frame mapping the artifact-mask complement walk needs. Computed by
    a vectorized binary search over ``sample_index_to_time`` so peak memory is
    bounded by the query count, not ``n_samples``. Because
    ``sample_index_to_time(i)`` is bit-identical to ``get_times()[i]`` for both
    rate-based and explicit recordings, the result is identical to the
    full-vector searchsorted.

    Assumes monotonically non-decreasing timestamps (the persisted-recording
    invariant); binary search over a non-monotonic vector mis-maps silently,
    exactly as ``searchsorted`` would.

    Parameters
    ----------
    recording : si.BaseRecording
    times_s : array-like
        Query times in seconds.
    segment_index : int, optional

    Returns
    -------
    numpy.ndarray of int64
        Frame indices in ``[0, n_samples]``, shape ``np.shape(np.atleast_1d(
        times_s))``.
    """
    import numpy as np

    times = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
    # Reject non-finite query times. searchsorted sorts NaN as +inf, so
    # a NaN/Inf query silently maps to frame n (or 0) instead of failing. Note
    # this guards ONLY finiteness: out-of-range *finite* times intentionally
    # clamp to [0, n] (the searchsorted contract the artifact-mask complement
    # walk depends on, pinned by test_frames_for_times_matches_full_vector_*).
    if not np.all(np.isfinite(times)):
        raise ValueError(
            "frames_for_times: query times contain non-finite (NaN/Inf) "
            f"values: {np.asarray(times_s).tolist()!r}. searchsorted-based "
            "frame mapping would silently mis-map them (NaN sorts as +inf)."
        )
    n = int(recording.get_num_samples(segment_index=segment_index))
    lo = np.zeros(times.shape, dtype=np.int64)
    hi = np.full(times.shape, n, dtype=np.int64)
    if n == 0:
        return lo
    # ~ceil(log2(n)) iterations. ``mid`` is clipped to a valid frame for the
    # read; converged (lo == hi) entries are not updated.
    while True:
        active = lo < hi
        if not bool(active.any()):
            break
        mid = (lo + hi) // 2
        vals = _segment_times_at(
            recording,
            np.minimum(mid, n - 1),
            segment_index=segment_index,
        )
        go_right = active & (vals < times)
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)
    return lo


class BaseIntervalsAndGaps(NamedTuple):
    """Return of :func:`base_intervals_and_gaps`.

    ``base_intervals`` are ``[start, end]`` second pairs (one per recorded
    chunk); ``gap_after`` are int64 FRAME indices of inter-chunk wall-clock
    discontinuities. Naming the two fields makes their differing units (seconds
    vs frame indices) explicit at the call site.
    """

    base_intervals: list
    gap_after: Any  # numpy.ndarray[int64] (np is lazy-imported in this module)


def base_intervals_and_gaps(recording, fs=None, *, segment_index=0):
    """Recorded chunks (seconds) and inter-chunk gap frame indices, chunked.

    Streams the timeline in ~1 s chunks via ``sample_index_to_time`` (bounded
    peak memory, no ``get_times()`` materialization) to derive the same gap
    structure the full-vector path computes from ``get_times()``. Generalizes
    the chunked scan in ``_units_nwb._base_intervals_from_recording`` to also
    emit the gap frame indices. Returns:

    * ``base_intervals`` -- one ``[start, end]`` (inclusive first/last sample
      times, seconds) per recorded chunk; identical to
      ``_base_intervals_from_timestamps(recording.get_times(), fs)``. A
      contiguous recording yields a single ``[t0, t_end]``.
    * ``gap_after`` -- int64 frame indices ``i`` where
      ``get_times()[i + 1] - get_times()[i] > 1.5 / fs`` (a wall-clock
      discontinuity from disjoint sort intervals); identical to
      ``np.flatnonzero(np.diff(recording.get_times()) > 1.5 / fs)``. Empty for a
      contiguous or rate-based recording.

    Monotonicity is validated per chunk and across chunk boundaries (the
    ``searchsorted``-based consumers assume it), preserving the full-vector
    path's ``assert_monotonic_timestamps`` guard at bounded memory.

    Parameters
    ----------
    recording : si.BaseRecording
    fs : float, optional
        Sampling frequency; read from the recording when ``None``.
    segment_index : int, optional

    Returns
    -------
    BaseIntervalsAndGaps
        Named ``(base_intervals, gap_after)``; unpacks like a plain tuple.
    """
    import numpy as np

    if fs is None:
        fs = recording.get_sampling_frequency()
    fs = assert_positive_sampling_frequency(
        fs, context="base_intervals_and_gaps: "
    )
    n_samples = int(recording.get_num_samples(segment_index=segment_index))
    if n_samples == 0:
        return BaseIntervalsAndGaps([], np.empty(0, dtype=np.int64))

    if not _recording_has_explicit_time_vector(
        recording, segment_index=segment_index
    ):
        # Rate-based recording: uniform timestamps, no wall-clock gaps. Map the
        # two endpoints affinely instead of scanning n_samples frames.
        endpoints = _segment_times_at(
            recording,
            np.array([0, n_samples - 1], dtype=np.int64),
            segment_index=segment_index,
        )
        return BaseIntervalsAndGaps(
            [[float(endpoints[0]), float(endpoints[1])]],
            np.empty(0, dtype=np.int64),
        )

    sample_period = 1.0 / float(fs)
    chunk_size = max(1, int(round(float(fs))))
    intervals: list[list[float]] = []
    gap_after: list[int] = []
    current_start = None
    prev_time = None
    for start_frame in range(0, n_samples, chunk_size):
        end_frame = min(n_samples, start_frame + chunk_size)
        times = _segment_times_at(
            recording,
            np.arange(start_frame, end_frame, dtype=np.int64),
            segment_index=segment_index,
        )
        if times.size == 0:
            continue
        assert_monotonic_timestamps(times, context="base_intervals_and_gaps: ")
        if current_start is None:
            current_start = float(times[0])
        else:
            if float(times[0]) < prev_time:
                raise ValueError(
                    "base_intervals_and_gaps: timestamps step backward across "
                    "a chunk boundary (searchsorted-based frame mapping would "
                    "silently mis-slice the recording)."
                )
            if float(times[0]) - prev_time > 1.5 * sample_period:
                # Wall-clock gap at the chunk boundary: between global frame
                # (start_frame - 1) and start_frame.
                intervals.append([float(current_start), float(prev_time)])
                gap_after.append(start_frame - 1)
                current_start = float(times[0])
        local_gaps = np.flatnonzero(np.diff(times) > 1.5 * sample_period)
        for gap_idx in local_gaps:
            intervals.append([float(current_start), float(times[gap_idx])])
            gap_after.append(start_frame + int(gap_idx))
            current_start = float(times[gap_idx + 1])
        prev_time = float(times[-1])
    if current_start is not None:
        intervals.append([float(current_start), float(prev_time)])
    return BaseIntervalsAndGaps(
        intervals, np.asarray(gap_after, dtype=np.int64)
    )


def timestamp_fingerprint(recording, *, segment_index=0):
    """SHA-256 over the recording's timestamp vector, computed chunked.

    A content fingerprint for the ``(n_samples,)`` float64 timestamp vector that
    two recordings share iff their timestamps are byte-for-byte equal -- the
    bounded-memory replacement for ``np.array_equal`` over two full vectors when
    checking that shared-artifact-group members are time-aligned. Reads the
    vector in ~1 s slices via ``sample_index_to_time`` (never the full
    ``get_times()`` materialization) and folds each slice's float64 bytes into
    the digest, prefixed by ``n_samples`` so different-length vectors cannot
    collide.

    The digest uses native byte order (``np.int64``/``float64`` ``tobytes``) and
    is only ever compared in-process against another fingerprint computed the
    same way (the shared-artifact-group member check); it is never persisted or
    compared across hosts, so endianness is not a concern. Do not store it for a
    later cross-host comparison without normalizing byte order.

    Parameters
    ----------
    recording : si.BaseRecording
    segment_index : int, optional

    Returns
    -------
    bytes
        The 32-byte SHA-256 digest of the timestamp vector.
    """
    import hashlib

    import numpy as np

    n_samples = int(recording.get_num_samples(segment_index=segment_index))
    hasher = hashlib.sha256()
    hasher.update(np.int64(n_samples).tobytes())
    if n_samples == 0:
        return hasher.digest()
    chunk_size = max(1, int(round(float(recording.get_sampling_frequency()))))
    for start_frame in range(0, n_samples, chunk_size):
        end_frame = min(n_samples, start_frame + chunk_size)
        times = _segment_times_at(
            recording,
            np.arange(start_frame, end_frame, dtype=np.int64),
            segment_index=segment_index,
        )
        hasher.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
    return hasher.digest()
