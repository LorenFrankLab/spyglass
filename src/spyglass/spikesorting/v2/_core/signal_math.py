"""Pure signal / frame / interval math for the spike sorting tables.

Timestamp/frame conversion and interval helpers shared by the v2 stages. No
DataJoint access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple


def assert_freq_max_below_nyquist(
    freq_max: float, sampling_frequency: float, *, context: str = ""
) -> None:
    """Raise if a bandpass ``freq_max`` is at or above the Nyquist (fs/2).

    SciPy otherwise fails opaquely inside filter design. ``context`` prefixes
    the message.
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

    A bad value would silently mis-map every frame. ``context`` prefixes the
    message.
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
    """Raise if ``timestamps`` is empty, non-finite, or steps backward.

    ``searchsorted`` frame mapping would otherwise silently mis-slice. Equal
    consecutive timestamps are allowed. ``context`` prefixes the message.
    """
    import numpy as np

    ts = np.asarray(timestamps)
    if ts.size == 0:
        raise ValueError(f"{context}timestamp vector is empty.")
    if not np.all(np.isfinite(ts)):
        # The backward-step check below would pass NaN (NaN < 0 is False).
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
    """Raise if artifact frames exceed ``_MAX_ARTIFACT_FRAME_FRACTION``.

    Past half the recording this is a misconfigured detector, and the per-frame
    index array would cost O(n_samples) memory. No-op when ``n_samples <= 0``.
    ``context`` prefixes the message.
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

    Concatenates all segments (``get_times()`` returns only one). Returns
    ``override`` verbatim when given, since SI's ``frame_slice`` and
    ``concatenate_recordings`` drop the source's wall-clock timestamps.

    Parameters
    ----------
    recording : si.BaseRecording
        Source recording whose timestamps are needed.
    override : array-like, optional
        Persisted timestamps to return verbatim.

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

    from spyglass.spikesorting._recording_timestamps import (
        _get_recording_timestamps as _concatenated_segment_times,
    )

    return _concatenated_segment_times(recording)


def subtract_intervals(base, removed, *, inclusive_stop=False) -> list[tuple]:
    """Remove ``removed`` from each ``base`` interval separately.

    Each base interval is cut on its own, so no output spans the space
    between two base intervals. ``removed`` must be sorted by start; its rows
    may overlap one another, and a row that does not overlap a base interval
    (including a zero-length row) leaves it unchanged.

    Parameters
    ----------
    base : iterable of (start, stop)
        Intervals to cut, in seconds.
    removed : sequence of (start, stop)
        Half-open ``[start, stop)`` intervals to remove, sorted by start.
    inclusive_stop : bool, optional
        ``False`` (default): a base interval's ``stop`` is an exclusive
        bound. A removal starting at ``stop`` does not touch the interval,
        and only pieces of positive length are kept.
        ``True``: ``stop`` is the last sample's time. A removal starting at
        ``stop`` ends the prior piece at ``np.nextafter(stop, -np.inf)`` so it
        is not read as a kept endpoint, and a zero-length final piece
        ``(stop, stop)`` is kept.

    Returns
    -------
    list[tuple[float, float]]
        Kept ``(start, stop)`` pieces in base order.
    """
    import numpy as np

    removed = list(removed)
    # Running max of removal stops lets searchsorted skip finished rows.
    reach = np.maximum.accumulate(
        np.asarray(removed, dtype=float).reshape(-1, 2)[:, 1]
    )
    kept = []
    for start, stop in base:
        start, stop = float(start), float(stop)
        cursor = start
        first = int(np.searchsorted(reach, start, side="right"))
        for cut_start, cut_stop in removed[first:]:
            if cut_start > stop or (cut_start == stop and not inclusive_stop):
                break
            if cut_stop <= cursor or cut_stop <= cut_start:
                continue  # already removed, or zero length
            if cursor < cut_start:
                end = (
                    np.nextafter(cut_start, -np.inf)
                    if cut_start == stop
                    else cut_start
                )
                kept.append((cursor, end))
            cursor = max(cursor, cut_stop)
        if cursor < stop or (inclusive_stop and cursor == stop):
            kept.append((cursor, stop))
    return kept


def intersect_interval_sets(interval_sets):
    """Intersect a list of ``(n, 2)`` sorted, non-overlapping interval arrays.

    Returns the ``(m, 2)`` spans in every set: the observation window of a
    merged unit. ``None`` entries are skipped; no sets gives ``(0, 2)``.
    """
    import numpy as np

    from spyglass.spikesorting._intervals import intersect_intervals

    sets = [
        np.asarray(s, dtype=float).reshape(-1, 2)
        for s in interval_sets
        if s is not None
    ]
    if not sets:
        return np.empty((0, 2), dtype=float)
    acc = sets[0]
    for other in sets[1:]:
        acc = intersect_intervals(acc, other)
        if acc.size == 0:
            break
    return acc


def frames_with_window_in_one_span(frames, spans, *, n_before, n_after):
    """Say which frames have their whole waveform window inside one span.

    A frame ``s`` reads ``[s - n_before, s + n_after)``, so a kept waveform
    never crosses a join, gap, artifact exclusion, or recording end.

    Parameters
    ----------
    frames : array-like of int, shape (n_frames,)
        Spike or peak frames of a single-segment recording, in any order.
    spans : sequence of (int, int)
        Sorted, non-overlapping half-open frame spans ``[start, end)`` with
        ``start < end``; adjacent spans are allowed.
    n_before, n_after : int
        Waveform window samples before and from the frame.

    Returns
    -------
    numpy.ndarray of bool, shape (n_frames,)
        ``True`` where the frame and its whole window lie in one span.

    Raises
    ------
    ValueError
        ``spans`` is not sorted, non-overlapping and non-empty per span, or
        ``n_before`` / ``n_after`` is negative.
    """
    import numpy as np

    frames = np.asarray(frames, dtype=np.int64)
    if n_before < 0 or n_after < 0:
        raise ValueError(
            f"waveform window ({n_before}, {n_after}) must not be negative."
        )
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    starts, ends = spans[:, 0], spans[:, 1]
    if np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError(
            "spans must be sorted, non-overlapping [start, end) frame ranges "
            f"with start < end; got {spans.tolist()}."
        )
    span = np.searchsorted(starts, frames, side="right") - 1
    found = span >= 0
    kept = np.zeros(frames.shape, dtype=bool)
    kept[found] = (frames[found] - n_before >= starts[span[found]]) & (
        frames[found] + n_after <= ends[span[found]]
    )
    return kept


# Duplicate-spike window (ms) when merging units; SpikeInterface's
# ``MergeUnitsSorting`` default. Well under a refractory period.
_MERGE_DEDUP_DELTA_MS = 0.4


# --------------------------------------------------------------------------- #
# Bounded-memory timestamp helpers. SI's ``get_times()`` materializes and
# caches the full float64 vector (~824 MB per hour at 30 kHz); these read
# only the frames they need and match ``get_times()`` exactly.


def _segment_times_at(recording, frames, *, segment_index=0):
    """Absolute times (s) for arbitrary ``frames`` without materializing.

    h5py fancy indexing needs strictly increasing indices, so this reads the
    sorted unique frames and scatters back to the caller's order.

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
    uniq, inverse = np.unique(flat, return_inverse=True)
    vals = np.asarray(
        recording.sample_index_to_time(uniq, segment_index=segment_index),
        dtype=np.float64,
    )
    return vals[inverse].reshape(frames.shape)


def _segment_times_slice(recording, start, stop, *, segment_index=0):
    """Read contiguous timestamps without HDF5 point indexing or full loads."""
    import numpy as np

    info = recording.get_time_info(segment_index=segment_index)
    if info["time_vector"] is not None:
        return np.asarray(info["time_vector"][start:stop], dtype=np.float64)
    return np.asarray(
        recording.sample_index_to_time(
            np.arange(start, stop, dtype=np.int64), segment_index=segment_index
        ),
        dtype=np.float64,
    )


def frames_for_times(recording, times_s, *, side="left"):
    """Bounded-memory equivalent of ``searchsorted(get_times(), t, side)``.

    A vectorized binary search, so memory scales with the query count.
    Out-of-range finite times clamp to ``[0, n_samples]``. Assumes
    non-decreasing timestamps.

    Parameters
    ----------
    recording : si.BaseRecording
        Read on segment 0.
    times_s : array-like
        Query times in seconds.
    side : {"left", "right"}, optional
        Use "right" for the exclusive frame bound of an inclusive end time.

    Returns
    -------
    numpy.ndarray of int64
        Frame indices in ``[0, n_samples]``, shape ``np.shape(np.atleast_1d(
        times_s))``.
    """
    import numpy as np

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    times = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
    # NaN would silently map to frame n, as searchsorted sorts it as +inf.
    if not np.all(np.isfinite(times)):
        raise ValueError(
            "frames_for_times: query times contain non-finite (NaN/Inf) "
            f"values: {np.asarray(times_s).tolist()!r}. searchsorted-based "
            "frame mapping would silently mis-map them (NaN sorts as +inf)."
        )
    n = int(recording.get_num_samples(segment_index=0))
    lo = np.zeros(times.shape, dtype=np.int64)
    hi = np.full(times.shape, n, dtype=np.int64)
    if n == 0:
        return lo
    while True:
        active = lo < hi
        if not bool(active.any()):
            break
        mid = (lo + hi) // 2
        vals = _segment_times_at(
            recording,
            np.minimum(mid, n - 1),
        )
        go_right = active & (
            (vals < times) if side == "left" else (vals <= times)
        )
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)
    return lo


class BaseIntervalsAndGaps(NamedTuple):
    """Return of :func:`base_intervals_and_gaps`.

    ``base_intervals`` are ``[start, end]`` seconds; ``gap_after`` are int64
    frame indices.
    """

    base_intervals: list
    gap_after: Any  # numpy.ndarray[int64] (np is lazy-imported in this module)


def base_intervals_and_gaps(recording, fs=None):
    """Recorded chunks (seconds) and inter-chunk gap frame indices, chunked.

    Streams the timeline in ~1 s slices and validates monotonicity. A gap is a
    step greater than ``1.5 / fs``.

    Parameters
    ----------
    recording : si.BaseRecording
        Read on segment 0.
    fs : float, optional
        Sampling frequency; read from the recording when ``None``.

    Returns
    -------
    BaseIntervalsAndGaps
        - ``base_intervals``: inclusive ``[start, end]`` seconds per
          contiguous chunk.
        - ``gap_after``: frames ``i`` where a gap follows ``i``; equals
          ``np.flatnonzero(np.diff(get_times()) > 1.5 / fs)``.
    """
    import numpy as np

    if fs is None:
        fs = recording.get_sampling_frequency()
    fs = assert_positive_sampling_frequency(
        fs, context="base_intervals_and_gaps: "
    )
    n_samples = int(recording.get_num_samples(segment_index=0))
    if n_samples == 0:
        return BaseIntervalsAndGaps([], np.empty(0, dtype=np.int64))

    if not recording.has_time_vector(segment_index=0):
        # Rate-based recordings have no gaps.
        endpoints = _segment_times_at(
            recording,
            np.array([0, n_samples - 1], dtype=np.int64),
        )
        return BaseIntervalsAndGaps(
            [[float(endpoints[0]), float(endpoints[1])]],
            np.empty(0, dtype=np.int64),
        )

    sample_period = 1.0 / float(fs)
    chunk_size = max(1, round(float(fs)))
    intervals: list[list[float]] = []
    gap_after: list[int] = []
    current_start = None
    prev_time = None
    for start_frame in range(0, n_samples, chunk_size):
        end_frame = min(n_samples, start_frame + chunk_size)
        times = _segment_times_slice(
            recording,
            start_frame,
            end_frame,
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
                # Gap at the chunk boundary.
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


def timestamp_fingerprint(recording):
    """SHA-256 over the recording's timestamp vector, computed chunked.

    Equal iff the timestamps are byte-identical. Uses native byte order, so do
    not persist it or compare across hosts.

    Parameters
    ----------
    recording : si.BaseRecording
        Read on segment 0.

    Returns
    -------
    bytes
        The 32-byte SHA-256 digest of the timestamp vector.
    """
    import hashlib

    import numpy as np

    n_samples = int(recording.get_num_samples(segment_index=0))
    hasher = hashlib.sha256()
    hasher.update(np.int64(n_samples).tobytes())
    if n_samples == 0:
        return hasher.digest()
    chunk_size = max(1, round(float(recording.get_sampling_frequency())))
    for start_frame in range(0, n_samples, chunk_size):
        end_frame = min(n_samples, start_frame + chunk_size)
        times = _segment_times_slice(
            recording,
            start_frame,
            end_frame,
        )
        hasher.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
    return hasher.digest()


def resolve_peak_sign(params) -> str:
    """Map a validated sorter params blob to a SpikeInterface ``peak_sign``.

    - ``peak_sign``: clusterless thresholder, Tridesclous2.
    - ``detect_sign`` (-1/0/1): MountainSort 4/5.
    - ``detection.method_kwargs.peak_sign``: SpykingCircus2.
    - ``invert_sign``: Kilosort4.
    - otherwise ``"neg"`` (SI's default).

    Parameters
    ----------
    params : Mapping or None
        The validated ``SorterParameters.params`` blob. A non-mapping
        returns ``"neg"``.

    Returns
    -------
    str
        One of ``"neg"`` / ``"pos"`` / ``"both"``.
    """
    if not isinstance(params, Mapping):
        return "neg"
    if params.get("peak_sign") is not None:
        return str(params["peak_sign"])
    if params.get("detect_sign") is not None:
        return {-1: "neg", 0: "both", 1: "pos"}.get(
            int(params["detect_sign"]), "neg"
        )
    detection = params.get("detection")
    if isinstance(detection, Mapping):
        method_kwargs = detection.get("method_kwargs")
        if (
            isinstance(method_kwargs, Mapping)
            and method_kwargs.get("peak_sign") is not None
        ):
            return str(method_kwargs["peak_sign"])
    if params.get("invert_sign"):
        return "pos"
    return "neg"


def _assert_noise_levels_length(
    noise_levels: list[float] | None, n_channels: int
) -> None:
    """Reject an explicit ``noise_levels`` of an unusable length.

    Length must be 1 (broadcast) or ``n_channels``; otherwise SI's
    ``detect_peaks`` silently truncates or raises an opaque ``IndexError``.

    Parameters
    ----------
    noise_levels : list[float] | None
        The resolved noise levels. ``None`` is valid (SI estimates MAD).
    n_channels : int
        Number of channels on the recording the detector runs over.

    Raises
    ------
    ValueError
        If ``noise_levels`` is a string, a scalar, or of the wrong length.
    """
    if noise_levels is None:
        return
    if isinstance(noise_levels, (str, bytes)):
        raise ValueError(
            "clusterless noise_levels must be a numeric sequence with length "
            "1 (broadcast) or n_channels; got a string/bytes value."
        )
    try:
        length = len(noise_levels)
    except TypeError as exc:
        raise ValueError(
            "clusterless noise_levels must be a numeric sequence with length "
            "1 (broadcast) or n_channels; got a scalar value."
        ) from exc
    if length not in (1, n_channels):
        raise ValueError(
            "clusterless noise_levels must have length 1 (broadcast) or "
            f"n_channels={n_channels}; got {length}."
        )
