"""Artifact-mask service behind ``Sorting``."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting._intervals import (
    intersect_intervals,
    merge_sorted_intervals,
)
from spyglass.spikesorting.v2._core.signal_math import (
    _segment_times_at,
    assert_artifact_frame_fraction,
    assert_positive_sampling_frequency,
    base_intervals_and_gaps,
    frames_for_times,
)
from spyglass.spikesorting.v2._storage.spikeinterface import (
    use_pickle_recording_serialization,
)
from spyglass.spikesorting.v2.exceptions import EmptyArtifactValidTimesError
from spyglass.utils import logger


def _validate_artifact_intervals(
    recording, valid_times, artifact_detection_id, recording_id
):
    """Validate the interval contract and bounded recording envelope."""
    n_segments = recording.get_num_segments()
    if n_segments != 1:
        raise ValueError(
            "apply_artifact_mask: expected a single-segment recording; got "
            f"{n_segments} segments. The v2 sort recording is a mono-segment "
            "concatenated timeline; a multi-segment recording signals an "
            "upstream construction error."
        )

    valid_times = np.asarray(valid_times, dtype=float)
    if valid_times.size == 0:
        raise EmptyArtifactValidTimesError(
            "Artifact-removed valid_times is empty for "
            f"artifact_detection_id={artifact_detection_id!r}, "
            f"recording_id={recording_id!r}: the artifact-detection pass kept "
            "zero seconds of the recording. Masking would zero the "
            "entire recording and the sort would run over all-zeros. "
            "Re-run RecordingArtifactDetection with looser thresholds or "
            "override the artifact-detection selection."
        )
    if valid_times.ndim != 2 or valid_times.shape[1] != 2:
        raise ValueError(
            "apply_artifact_mask: valid_times must be an (n, 2) array "
            f"of (start, end) seconds; got shape {valid_times.shape}."
        )
    # NaN passes every comparison check below.
    if not np.all(np.isfinite(valid_times)):
        raise ValueError(
            "apply_artifact_mask: valid_times contains non-finite (NaN/Inf) "
            f"values: {valid_times.tolist()!r}. The artifact-removed intervals "
            "must be finite seconds; a non-finite bound signals an "
            "alignment/units error upstream."
        )
    starts = valid_times[:, 0]
    ends = valid_times[:, 1]
    if np.any(ends < starts):
        raise ValueError(
            "apply_artifact_mask: valid_times has an interval whose "
            "end precedes its start; each interval must be "
            "(start <= end)."
        )
    if valid_times.shape[0] > 1 and (
        np.any(np.diff(starts) < 0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(
            "apply_artifact_mask: valid_times must be sorted by start "
            "time and non-overlapping (the complement walker assumes "
            f"monotonic, disjoint input); got {valid_times.tolist()!r}. "
            "Sort and merge the intervals before passing them."
        )

    # Never materialize the full timestamp vector (~824 MB for 1 h at 30 kHz).
    n_samples = int(recording.get_num_samples(segment_index=0))
    if n_samples == 0:
        raise ValueError(
            "apply_artifact_mask: recording has zero samples; there is "
            "nothing to mask."
        )
    t_first, t_last = (
        float(t)
        for t in _segment_times_at(
            recording, np.array([0, n_samples - 1], dtype=np.int64)
        )
    )
    # Endpoint-only check; Recording.make guarantees interior monotonicity.
    if not (np.isfinite(t_first) and np.isfinite(t_last) and t_last >= t_first):
        raise ValueError(
            "apply_artifact_mask: recording endpoints are non-finite or step "
            f"backward (t_first={t_first}, t_last={t_last}); the persisted "
            "recording's monotonic timestamp invariant is violated and the "
            "searchsorted frame mapping would silently mis-mask."
        )
    # The walk would silently clip a units error (ms vs s). Allow one sample
    # of slop for the concat exclusive end, plus one float step.
    envelope_tol = 1.0 / assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="apply_artifact_mask: "
    )
    envelope_start = np.nextafter(t_first - envelope_tol, -np.inf)
    envelope_stop = np.nextafter(t_last + envelope_tol, np.inf)
    if starts.min() < envelope_start or ends.max() > envelope_stop:
        raise ValueError(
            "apply_artifact_mask: valid_times "
            f"{valid_times.tolist()!r} fall outside the recording envelope "
            f"[{t_first}, {t_last}] seconds (tol={envelope_tol:g}s); an "
            "interval before the first or past the last sample signals an "
            "alignment/units error (e.g. milliseconds vs seconds)."
        )
    return valid_times, n_samples, t_first, t_last


def artifact_frame_ranges(
    recording, valid_times, *, artifact_detection_id=None, recording_id=None
):
    """Map excluded periods to half-open frame ranges with bounded memory.

    Parameters
    ----------
    recording : si.BaseRecording
        The recording whose actual timestamps define frame coordinates.
        This mapping function does not modify it.
    valid_times : numpy.ndarray
        Artifact-removed ``(n, 2)`` array of (start, end) seconds,
        sorted by start and non-overlapping.
    artifact_detection_id : optional
        Keyword-only. Used only in the empty-``valid_times`` error
        message. Default ``None``.
    recording_id : optional
        Keyword-only. Used only in the empty-``valid_times`` error
        message. Default ``None``.

    Returns
    -------
    list[tuple[int, int]]
        Excluded half-open frame ranges; empty when every frame is valid.

    Raises
    ------
    EmptyArtifactValidTimesError
        If ``valid_times`` is empty, which would zero the whole recording.
    ValueError
        If ``valid_times`` is not an ``(n, 2)`` array, has an
        interval with ``end < start``, or is not sorted-by-start and
        non-overlapping; the walker never sorts or merges.
    """
    valid_times, n_samples, t_first, t_last = _validate_artifact_intervals(
        recording, valid_times, artifact_detection_id, recording_id
    )
    # None marks the exclusive recording end.
    ends = valid_times[:, 1]
    gap_time_pairs: list[tuple[float, float | None]] = []
    cursor = t_first
    for vs, ve in valid_times:
        if vs > cursor:
            gap_time_pairs.append((float(cursor), float(vs)))
        cursor = max(cursor, ve)
    if cursor < t_last:
        gap_time_pairs.append((float(cursor), None))

    frame_ranges: list[tuple[int, int]] = []
    if gap_time_pairs:
        start_frames = frames_for_times(
            recording, [s for s, _ in gap_time_pairs]
        )
        # t_last is a placeholder for None; the loop uses n_samples instead.
        end_frames = frames_for_times(
            recording, [t_last if e is None else e for _, e in gap_time_pairs]
        )
        for (_, e_time), start, end in zip(
            gap_time_pairs, start_frames, end_frames
        ):
            start = int(start)
            end = n_samples if e_time is None else int(end)
            if end > start:
                frame_ranges.append((start, end))

    # A width-1 range followed by a clock gap is the last valid sample of a
    # chunk, not an artifact.
    sample_period = 1.0 / assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="apply_artifact_mask: "
    )

    def _is_interchunk_boundary_range(start, end):
        if not (end - start == 1 and end < n_samples):
            return False
        ts_start, ts_end = _segment_times_at(
            recording, np.array([start, end], dtype=np.int64)
        )
        return (ts_end - ts_start) > 1.5 * sample_period and np.any(
            ends == ts_start
        )

    frame_ranges = [
        (s, e)
        for (s, e) in frame_ranges
        if not _is_interchunk_boundary_range(s, e)
    ]

    if not frame_ranges:
        return []

    # Data sanity, not memory: refuse to sort a sliver.
    assert_artifact_frame_fraction(
        sum(end - start for start, end in frame_ranges),
        n_samples,
        context="apply_artifact_mask: ",
    )

    return frame_ranges


def apply_artifact_mask(
    recording, valid_times, *, artifact_detection_id=None, recording_id=None
):
    """Mask excluded periods lazily, preserving frames and timestamps."""
    ranges = artifact_frame_ranges(
        recording,
        valid_times,
        artifact_detection_id=artifact_detection_id,
        recording_id=recording_id,
    )
    return silence_frame_ranges(recording, ranges)


def has_nonzero_offset(recording) -> bool:
    """Say whether a recording keeps a nonzero channel calibration offset.

    A stored zero reads as 0 uV only when every offset is 0. Filtering and
    referencing zero the offset.

    Parameters
    ----------
    recording : si.BaseRecording

    Returns
    -------
    bool
    """
    offsets = recording.get_channel_offsets()
    return offsets is not None and bool(np.any(offsets != 0))


def recording_with_zero_offset(recording):
    """Present a recording so that a stored zero reads as 0 uV.

    Returned unchanged unless it has a nonzero offset, in which case it is
    converted to float32 µV with ``scale_to_uV``.

    Parameters
    ----------
    recording : si.BaseRecording
        The recording to be masked.

    Returns
    -------
    si.BaseRecording
        ``recording`` itself, or its float32 microvolt view.
    """
    if not has_nonzero_offset(recording):
        return recording
    import spikeinterface.preprocessing as sip

    return sip.scale_to_uV(recording)


def silence_frame_ranges(recording, frame_ranges):
    """Silence validated half-open frame ranges to 0 uV.

    Zeroed lazily. A recording with a nonzero offset is first converted
    (:func:`recording_with_zero_offset`); others keep their dtype and
    calibration.

    Parameters
    ----------
    recording : si.BaseRecording
        A single-segment recording.
    frame_ranges : sequence of (int, int)
        Validated, sorted, half-open ``[start, end)`` frame ranges.

    Returns
    -------
    si.BaseRecording
        The silenced recording, or ``recording`` when there are no ranges.
    """
    import spikeinterface.preprocessing as sip

    if not len(frame_ranges):
        return recording
    recording = recording_with_zero_offset(recording)

    # Lazy and O(n_ranges) memory; remove_artifacts also mis-masks a boundary
    # frame of a padded window in SI 0.104.
    masked = sip.silence_periods(
        recording,
        list_periods=[frame_ranges],
        mode="zeros",
    )

    # SilencedPeriodsRecording claims JSON support but cannot reload from it.
    use_pickle_recording_serialization(masked)
    return masked


def complement_frame_ranges(
    excluded_ranges: list[tuple[int, int]], n_samples: int
) -> list[tuple[int, int]]:
    """Half-open valid-frame ranges left after removing ``excluded_ranges``.

    ``excluded_ranges`` may be unsorted, overlapping, or adjacent; they are
    sorted and merged before the complement in ``[0, n_samples)`` is taken.

    Parameters
    ----------
    excluded_ranges : list[tuple[int, int]]
        Half-open ``(start, end)`` frame ranges to remove. Need not be
        sorted, non-overlapping, or merged.
    n_samples : int
        Total number of frames; the complement is bounded to ``[0,
        n_samples)``.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, disjoint, half-open valid frame ranges. ``[(0, n_samples)]``
        when ``excluded_ranges`` is empty.
    """
    n_samples = int(n_samples)
    merged = merge_sorted_intervals(
        sorted((int(a), int(b)) for a, b in excluded_ranges)
    )

    valid: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            valid.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < n_samples:
        valid.append((cursor, n_samples))
    return valid


def boundary_spans_from_timestamps(recording) -> list[tuple[int, int]]:
    """Half-open spans that never cross a wall-clock gap in ``recording``'s
    persisted timestamps.

    A gap is a timestamp step greater than ``1.5 / fs``. A recording without
    an explicit time vector yields one span.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment recording whose persisted timestamps define frame
        coordinates. Not modified.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, disjoint, half-open frame spans covering ``[0,
        recording.get_num_samples())``.

    Raises
    ------
    ValueError
        If ``recording`` has more than one segment; the v2 sort/statistics
        pipeline is single-segment only.
    """
    n_segments = recording.get_num_segments()
    if n_segments != 1:
        raise ValueError(
            "boundary_spans_from_timestamps: expected a single-segment "
            f"recording; got {n_segments} segments."
        )
    n = recording.get_num_samples()
    cuts = sorted(
        {
            0,
            n,
            *(int(g) + 1 for g in base_intervals_and_gaps(recording).gap_after),
        }
    )
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


class Continuity(NamedTuple):
    """Continuity spans and the real timestamps that bound each one.

    Attributes
    ----------
    spans : list[tuple[int, int]]
        Half-open frame spans of uninterrupted acquisition.
    start_s : list[float]
        Each span's first timestamp on its own acquisition clock (s).
    end_s : list[float]
        Each span's last timestamp on that clock (s): the time of frame
        ``b - 1``, not ``start + (b - a) / fs``, which can drift.
    """

    spans: list[tuple[int, int]]
    start_s: list[float]
    end_s: list[float]


def continuity_from_timestamps(recording) -> Continuity:
    """Continuity spans of ``recording`` with their first and last timestamps.

    Boundaries are read with scalar lookups, not the full time vector.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment recording carrying its persisted timestamps.

    Returns
    -------
    Continuity
    """
    spans = boundary_spans_from_timestamps(recording)
    return Continuity(
        spans=spans,
        start_s=[float(recording.sample_index_to_time(a)) for a, _ in spans],
        end_s=[float(recording.sample_index_to_time(b - 1)) for _, b in spans],
    )


def statistics_spans(
    n_samples: int,
    excluded_ranges: list[tuple[int, int]],
    boundary_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Artifact-free frame spans that each lie inside a single boundary span.

    Adjacent output spans are never merged; the edge between them is a join.

    Parameters
    ----------
    n_samples : int
        Total number of frames.
    excluded_ranges : list[tuple[int, int]]
        Half-open artifact-masked frame ranges (need not be sorted/merged).
    boundary_spans : list[tuple[int, int]]
        Disjoint half-open frame spans that never cross a join (e.g. from
        ``boundary_spans_from_timestamps`` / ``_recording.concat.
        concat_continuity``); adjacent spans may touch.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, half-open statistics spans.

    Raises
    ------
    ValueError
        If no artifact-free frame lies inside any boundary span.
    """
    n_samples = int(n_samples)
    valid = complement_frame_ranges(excluded_ranges, n_samples)
    valid_samples = sum(b - a for a, b in valid)
    out = [
        (int(a), int(b))
        for a, b in intersect_intervals(valid, boundary_spans, merge=False)
    ]
    if not out:
        raise ValueError(
            "statistics_spans: no artifact-free samples inside any "
            "acquisition span."
        )
    masked_fraction = (
        (n_samples - valid_samples) / n_samples if n_samples else 0.0
    )
    logger.info(
        "statistics_spans: masked_fraction=%.4f across %d statistics span(s) "
        "(n_samples=%d).",
        masked_fraction,
        len(out),
        n_samples,
    )
    return out


def corrected_statistics_spans(
    recording,
    row: dict,
    *,
    source_n_samples: int,
    concat_statistics_spans,
    obs_intervals,
    artifact_detection_id,
    recording_id,
) -> list[tuple[int, int]]:
    """Return a corrected recording's statistics spans after checking them.

    The row's copied spans must equal those derived from the corrected
    traces and the sort's mask (single recording) or the concatenation's
    stored spans, and the frame count must match the source's.

    Parameters
    ----------
    recording : si.BaseRecording
        The loaded corrected recording.
    row : dict
        Its ``MotionCorrectedRecording`` row (``motion_corrected_recording_id``,
        ``n_samples``, ``statistics_spans`` ``(n, 2)``).
    source_n_samples : int
        The source's frame count.
    concat_statistics_spans : numpy.ndarray or None
        A concat source's stored ``(n, 2)`` spans; ``None`` for a single
        recording.
    obs_intervals : numpy.ndarray or None
        The sort's artifact-removed valid times, ``(n_intervals, 2)`` in
        seconds, or ``None`` without an artifact detection.
    artifact_detection_id : uuid.UUID or None
        The sort's pinned detection (single recording only).
    recording_id : uuid.UUID or None
        The single-recording source, for error messages.

    Returns
    -------
    list[tuple[int, int]]
        The half-open frame spans.

    Raises
    ------
    ValueError
        If the frame count or the spans disagree with the source's.
    """
    from spyglass.spikesorting.v2._motion.estimation import normalize_spans

    n_samples = int(recording.get_num_samples())
    if not n_samples == int(row["n_samples"]) == int(source_n_samples):
        raise ValueError(
            "Sorting: motion-corrected recording "
            f"{row['motion_corrected_recording_id']} has {n_samples} "
            f"frames (row: {int(row['n_samples'])}); its source has "
            f"{int(source_n_samples)}."
        )
    spans = normalize_spans(row["statistics_spans"])
    if concat_statistics_spans is None:
        excluded = []
        if artifact_detection_id is not None:
            excluded = artifact_frame_ranges(
                recording,
                obs_intervals,
                artifact_detection_id=artifact_detection_id,
                recording_id=recording_id,
            )
        expected = normalize_spans(
            statistics_spans(
                n_samples,
                excluded,
                boundary_spans_from_timestamps(recording),
            )
        )
    else:
        expected = normalize_spans(concat_statistics_spans)
    if spans != expected:
        raise ValueError(
            "Sorting: motion-corrected recording "
            f"{row['motion_corrected_recording_id']} carries statistics "
            f"spans {spans}, but its source and the sort's mask give "
            f"{expected}."
        )
    return spans


def spans_cover_recording(spans, n_samples: int) -> bool:
    """True when ``spans`` is ``None`` or exactly ``[(0, n_samples)]``.

    In this case span estimators defer to SpikeInterface unchanged.
    """
    return spans is None or list(spans) == [(0, int(n_samples))]


def sample_span_data(
    recording,
    spans: list[tuple[int, int]],
    *,
    target_samples: int,
    max_piece: int,
    seed,
    return_in_uV: bool,
):
    """Randomly sample traces from ``recording`` without crossing a span edge.

    When ``target_samples`` covers every valid frame, each span is read
    once. Otherwise each span's quota (largest-remainder share of its
    length) is filled with random pieces of at most ``max_piece`` frames,
    returned in start-frame order.

    Parameters
    ----------
    recording : si.BaseRecording
    spans : list[tuple[int, int]]
        Half-open frame spans to sample from (e.g. ``statistics_spans``).
    target_samples : int
        Keyword-only. Requested row budget.
    max_piece : int
        Keyword-only. Maximum contiguous frames read per random draw.
    seed : int
        Keyword-only. Seeds ``numpy.random.default_rng``.
    return_in_uV : bool
        Keyword-only. Forwarded to ``recording.get_traces``.

    Returns
    -------
    numpy.ndarray
        ``(min(target_samples, total_valid), n_channels)`` traces.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array([b - a for a, b in spans], dtype=np.int64)
    total_valid = int(lengths.sum())
    effective_target = min(int(target_samples), total_valid)
    if effective_target == total_valid:
        data = np.concatenate(
            [
                recording.get_traces(
                    start_frame=a, end_frame=b, return_in_uV=return_in_uV
                )
                for a, b in spans
            ],
            axis=0,
        )
        return data
    # Quotas sum to effective_target and never exceed a span's length.
    raw = effective_target * lengths / total_valid
    quota = np.minimum(np.floor(raw).astype(np.int64), lengths)
    for i in np.argsort(-(raw - np.floor(raw))):
        if quota.sum() >= effective_target:
            break
        if quota[i] < lengths[i]:
            quota[i] += 1
    pieces: list[tuple[int, np.ndarray]] = []
    for (a, b), q in zip(spans, quota):
        remaining = int(q)
        while remaining > 0:
            piece = min(max_piece, remaining, b - a)
            s = int(rng.integers(a, b - piece + 1))
            pieces.append(
                (
                    s,
                    recording.get_traces(
                        start_frame=s,
                        end_frame=s + piece,
                        return_in_uV=return_in_uV,
                    ),
                )
            )
            remaining -= piece
    pieces.sort(key=lambda item: item[0])
    data = np.concatenate([piece for _, piece in pieces], axis=0)
    return data


def sample_span_snippet_starts(
    spans: list[tuple[int, int]],
    *,
    nsamples: int,
    n_snippets: int,
    seed,
):
    """Draw fixed-length snippet start frames from ``spans``.

    Uniform over every admissible start ``s`` (``a <= s`` and
    ``s + nsamples <= b``) across all spans.

    Parameters
    ----------
    spans : list[tuple[int, int]]
        Half-open frame spans to draw snippet starts from.
    nsamples : int
        Keyword-only. Snippet length in frames.
    n_snippets : int
        Keyword-only. Number of starts to draw.
    seed : int
        Keyword-only. Seeds ``numpy.random.default_rng``.

    Returns
    -------
    numpy.ndarray
        ``(n_snippets,)`` int64 sorted start frames.

    Raises
    ------
    ValueError
        If no span admits a snippet of length ``nsamples``.
    """
    nsamples = int(nsamples)
    counts = np.array(
        [max(0, (b - a) - nsamples + 1) for a, b in spans], dtype=np.int64
    )
    total = int(counts.sum())
    if total == 0:
        raise ValueError(
            "sample_span_snippet_starts: no span admits a snippet of length "
            f"{nsamples}; every span is shorter than nsamples."
        )
    rng = np.random.default_rng(seed)
    n_snippets = int(n_snippets)
    span_choices = rng.choice(len(spans), size=n_snippets, p=counts / total)
    starts = np.empty(n_snippets, dtype=np.int64)
    for k, i in enumerate(span_choices):
        a, b = spans[i]
        starts[k] = rng.integers(a, b - nsamples + 1)
    return np.sort(starts)


def sorting_statistics_spans(
    recording,
    traces,
    *,
    source_kind,
    recording_id,
    artifact_detection_id,
    obs_intervals,
    concat_statistics_spans,
    source_n_samples,
):
    """Mask a sort's input and resolve its statistics spans.

    Statistics spans are the artifact-free, join-free frame ranges that
    noise and whitening estimates sample from.

    - Motion-corrected: persisted masked; spans checked against its row.
    - Single recording: masked here; boundaries from its timestamps.
    - Concatenation: masks and spans were stored upstream.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The sort input, loaded from the effective traces.
    traces : EffectiveTraces
        The sort's effective traces (``kind``, ``row``,
        ``apply_artifact_mask``).
    source_kind : str
        ``"recording"`` or ``"concatenated_recording"``.
    recording_id : str
        The sort's anchor ``recording_id``.
    artifact_detection_id : str or None
        The sort's artifact detection, if any.
    obs_intervals : numpy.ndarray or None
        Artifact-removed valid times, shape ``(n_intervals, 2)`` in seconds.
    concat_statistics_spans : numpy.ndarray or None
        A concat source's stored spans; ``None`` for a single recording.
    source_n_samples : int or None
        The source's frame count a corrected recording must keep.

    Returns
    -------
    tuple
        ``(recording, statistics_spans)``: the recording to sort (silenced
        over the artifact frames of a masked single recording, otherwise
        unchanged) and the spans as ``(start_frame, end_frame)`` pairs.
    """
    if traces.kind == "motion_corrected_recording":
        spans = corrected_statistics_spans(
            recording,
            traces.row,
            source_n_samples=source_n_samples,
            concat_statistics_spans=concat_statistics_spans,
            obs_intervals=obs_intervals,
            artifact_detection_id=artifact_detection_id,
            recording_id=(recording_id if source_kind == "recording" else None),
        )
    elif source_kind == "recording":
        boundary_spans = boundary_spans_from_timestamps(recording)
        excluded_ranges = []
        if traces.apply_artifact_mask:
            excluded_ranges = artifact_frame_ranges(
                recording,
                obs_intervals,
                artifact_detection_id=artifact_detection_id,
                recording_id=recording_id,
            )
            recording = silence_frame_ranges(recording, excluded_ranges)
        spans = statistics_spans(
            recording.get_num_samples(), excluded_ranges, boundary_spans
        )
    else:
        from spyglass.spikesorting.v2._motion.estimation import normalize_spans

        spans = normalize_spans(concat_statistics_spans)
    return recording, spans
