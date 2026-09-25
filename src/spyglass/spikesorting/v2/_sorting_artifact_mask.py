"""Artifact-mask service behind ``Sorting``.

``apply_artifact_mask`` zeros the complement of the artifact-removed
``valid_times`` on the recording before sorting, so the sorter never sees
artifact frames. ``make_fetch`` already fetched ``valid_times`` (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), so this operates purely on the SpikeInterface recording.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The mask needs
none of that at import, so ``Sorting`` stays a thin orchestrator. Same "thin
DataJoint shell over pure/IO services" direction as ``_artifact_compute`` /
``_selection_identity`` / ``_analyzer_cache`` / ``_curation_transforms`` /
``_units_nwb`` / ``_sorting_dispatch`` / ``_sorting_units`` /
``_sorting_analyzer``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: numpy / SpikeInterface / the typed
``EmptyArtifactValidTimesError`` are imported lazily inside the function, which
touches no DB at call time.
"""

from __future__ import annotations


def artifact_frame_ranges(
    recording, valid_times, *, artifact_detection_id=None, recording_id=None
):
    """Map excluded periods to half-open frame ranges with bounded memory.

    ``valid_times`` is the artifact-removed (start, end) seconds
    array from the upstream ``IntervalList``; ``make_fetch``
    already fetched it as ``obs_intervals`` so ``make_compute``
    passes it through here instead of re-issuing the DB lookup
    (the tri-part contract forbids DB I/O inside compute).

    ``artifact_detection_id`` / ``recording_id`` are used only to make the
    empty-``valid_times`` error message actionable.

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
        If ``valid_times`` is empty -- masking would zero the whole
        recording, so the sort must fail loudly instead of running
        over all-zeros.
    ValueError
        If ``valid_times`` is not an ``(n, 2)`` array, has an
        interval with ``end < start``, or is not sorted-by-start and
        non-overlapping. The complement walker assumes monotonic,
        disjoint input; an unsorted/overlapping list would silently
        under-mask. (The fetched ``obs_intervals`` are monotonic in
        practice; this guards a hand-built curation override. Strict
        input is intentional -- silent sort/merge is deferred.)
    """
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _segment_times_at,
        assert_artifact_frame_fraction,
        assert_positive_sampling_frequency,
        frames_for_times,
    )
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )

    # Single-segment precondition. The complement walk and the single-segment
    # ``list_periods=[frame_ranges]`` call below both assume ``segment_index=0``
    # only; the v2 sort recording is always a mono-segment concatenated timeline
    # (``concatenate_recordings``). A multi-segment recording signals an upstream
    # construction error -- fail with a clear message instead of the cryptic
    # ``IndexError`` SpikeInterface's ``silence_periods`` raises when
    # ``list_periods`` is shorter than the segment count.
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
            "Re-run ArtifactDetection with looser thresholds or "
            "override the artifact-detection selection."
        )
    if valid_times.ndim != 2 or valid_times.shape[1] != 2:
        raise ValueError(
            "_apply_artifact_mask: valid_times must be an (n, 2) array "
            f"of (start, end) seconds; got shape {valid_times.shape}."
        )
    # Reject NaN/Inf before any comparison. NaN slips silently through
    # the < / sorted checks below (every NaN comparison is False), so a
    # non-finite interval would otherwise under-mask instead of failing loudly.
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
            "_apply_artifact_mask: valid_times has an interval whose "
            "end precedes its start; each interval must be "
            "(start <= end)."
        )
    if valid_times.shape[0] > 1 and (
        np.any(np.diff(starts) < 0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(
            "_apply_artifact_mask: valid_times must be sorted by start "
            "time and non-overlapping (the complement walker assumes "
            f"monotonic, disjoint input); got {valid_times.tolist()!r}. "
            "Sort and merge the intervals before passing them."
        )

    # Map the artifact-removed valid intervals to the complement frame ranges
    # WITHOUT materializing the recording's full timestamp vector.
    # ``recording.get_times()`` builds (and caches) a concrete float64 array of
    # every sample (~824 MB for 1 h @ 30 kHz, 8 bytes/sample); instead we read
    # only the two recording endpoints and binary-search the (few) interval
    # boundaries via ``frames_for_times`` / ``_segment_times_at``, both of which
    # index the h5py-/mmap-backed timestamps lazily (peak memory bounded by the
    # boundary count, not n_samples). The persisted recording's timestamps are
    # monotonically non-decreasing by construction (Recording.make) -- the
    # invariant the searchsorted-equivalent ``frames_for_times`` mapping assumes
    # (the prior full-vector ``assert_monotonic_timestamps`` guard would force
    # the materialization this avoids).
    n_samples = int(recording.get_num_samples(segment_index=0))
    if n_samples == 0:
        raise ValueError(
            "apply_artifact_mask: recording has zero samples; there is "
            "nothing to mask (the prior get_times() path raised on the empty "
            "timestamp vector)."
        )
    t_first, t_last = (
        float(t)
        for t in _segment_times_at(
            recording, np.array([0, n_samples - 1], dtype=np.int64)
        )
    )
    # Bounded (two-endpoint) monotonicity tripwire replacing the removed
    # full-vector ``assert_monotonic_timestamps``: catch gross corruption
    # (empty/reversed/NaN-bracketed vector) loudly instead of silently
    # mis-masking, without materializing. NOTE: this checks only the endpoints
    # -- an INTERIOR backward step (which Recording.make's ordering invariant
    # rules out) is no longer detected here, unlike the pre-refactor whole-
    # vector scan; the chunked ``detect_artifacts`` path still validates
    # monotonicity per chunk.
    if not (np.isfinite(t_first) and np.isfinite(t_last) and t_last >= t_first):
        raise ValueError(
            "apply_artifact_mask: recording endpoints are non-finite or step "
            f"backward (t_first={t_first}, t_last={t_last}); the persisted "
            "recording's monotonic timestamp invariant is violated and the "
            "searchsorted frame mapping would silently mis-mask."
        )
    # Reject intervals outside the recording envelope before the walk.
    # The complement walk silently clips to [t_first, t_last], so an interval
    # starting before the first sample or ending past the last (a units/
    # alignment error -- e.g. ms vs s) would be quietly ignored rather than
    # flagged. Allow the exclusive end used by concatenated recordings and
    # one sample-period of endpoint slop. The equivalent expressions n / fs
    # and (n - 1) / fs + 1 / fs can differ by one floating-point step.
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
    # Walk the valid intervals left-to-right in seconds, collecting the
    # complement (artifact gaps) as ``(start_time, end_time)`` pairs;
    # ``end_time is None`` marks the open tail that extends to the exclusive
    # end of the recording (frame n_samples, matching the old
    # ``end = len(timestamps)``). The boundary times are then batch-mapped to
    # frames in one binary search each.
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
        # Map every end in one search: the open tail's ``None`` end maps to
        # frame ``n_samples`` (the exclusive recording end, which
        # ``frames_for_times`` cannot produce from a query time), so pass
        # ``t_last`` as a harmless placeholder that the loop overrides.
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

    # Drop pure inter-chunk-gap ranges. For a DISJOINT recording the
    # gap-respecting valid_times leave a single boundary frame between
    # two chunks; the complement walk emits it as a width-1 range whose
    # successor is a wall-clock discontinuity. That frame is the last
    # real sample of the preceding chunk (valid) -- masking it would
    # zero a good sample per gap. A genuine 1-frame artifact instead
    # has ~1-sample spacing to its neighbor, so it is kept. A manual cut uses
    # a predecessor float to distinguish an excluded final sample from the
    # chunk's inclusive endpoint. Read only the two
    # boundary frames per width-1 candidate instead of indexing a full vector.
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

    # Data-sanity guard: a valid_times that keeps almost nothing makes the
    # artifact complement span most of the recording. The interval-native
    # silence_periods below keeps peak memory O(n_ranges) regardless of how many
    # samples are masked, so this is NO LONGER a memory guard -- it is a loud
    # "you are masking more than the bound; the sort would run on a sliver"
    # data-sanity check, mainly for a hand-built valid_times override (the normal
    # pipeline's detect_artifacts guard fires first; see the strict-input note).
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


def silence_frame_ranges(recording, frame_ranges):
    """Silence validated half-open frame ranges without expanding sample indices."""
    import spikeinterface.preprocessing as sip

    if not len(frame_ranges):
        return recording

    # Mask the artifact RANGES with the interval-native ``silence_periods``
    # rather than expanding them to one trigger per sample. ``list_periods``
    # takes ``(start, end_frame)`` tuples per segment with a HALF-OPEN ``end``
    # (frames ``[start, end)`` are zeroed -- matching ``frame_ranges`` and the
    # prior ``np.arange(s, e)`` expansion, verified), zeros them lazily in a
    # ``SilencedPeriodsRecording``, and never materializes an
    # O(n_artifact_frames) index array. This also sidesteps the SI 0.104
    # ``remove_artifacts`` boundary bug the per-frame path worked around
    # (single-sample triggers left the first frame of a contiguous run unmasked
    # under ``ms_before/ms_after=0``); ``silence_periods`` zeros the slice
    # directly, so no run edge is dropped.
    masked = sip.silence_periods(
        recording,
        list_periods=[frame_ranges],
        mode="zeros",
    )

    # Force the pickle serialization path for the masked recording.
    # ``SilencedPeriodsRecording`` stores its artifact intervals in
    # ``_kwargs["periods"]`` as a *structured* numpy array, which cannot survive
    # a JSON round-trip (JSON has no structured-array type: the reload receives a
    # plain nested list and ``__init__`` raises "periods must be a np.array with
    # dtype ..."). SpikeInterface nonetheless reports this recording as
    # JSON-serializable, so ``run_sorter`` dumps it to
    # ``spikeinterface_recording.json`` and dies on reload -- but only when
    # artifact detection actually flags intervals (so this object is built at
    # all). Marking it non-JSON-serializable makes ``run_sorter`` fall back to
    # pickle, which round-trips correctly. (No public setter exists; the private
    # ``_serializability`` flag is the supported mechanism. The proper fix is
    # upstream in SpikeInterface.)
    masked._serializability["json"] = False
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
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in excluded_ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

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

    A gap is a step between consecutive persisted timestamps greater than
    ``1.5 / fs`` (``base_intervals_and_gaps``, ``_signal_math.py``); a
    recording with no explicit time vector (uniform sample-rate timestamps)
    has no gaps and yields a single span covering the whole recording.

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
    from spyglass.spikesorting.v2._signal_math import base_intervals_and_gaps

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


def concat_boundary_spans(
    member_recordings, member_starts: list[int]
) -> list[tuple[int, int]]:
    """Union of each member's boundary spans, offset into concat-frame
    coordinates.

    Parameters
    ----------
    member_recordings : list[si.BaseRecording]
        Per-member recordings, ordered by ``member_index``.
    member_starts : list[int]
        Each member's cumulative start frame in the concatenated recording,
        same order/length as ``member_recordings``.

    Returns
    -------
    list[tuple[int, int]]
        Half-open concat-frame spans; a join between two members is never
        inside a single span.
    """
    out: list[tuple[int, int]] = []
    for rec, start in zip(member_recordings, member_starts, strict=True):
        out.extend(
            (start + a, start + b)
            for a, b in boundary_spans_from_timestamps(rec)
        )
    return out


def statistics_spans(
    n_samples: int,
    excluded_ranges: list[tuple[int, int]],
    boundary_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Artifact-free frame spans that each lie inside a single boundary span.

    The intersection of the artifact-free complement of ``excluded_ranges``
    with ``boundary_spans``. Two output spans that are adjacent in frame
    coordinates are never merged: adjacency across a boundary-span edge is
    exactly the join information the spans exist to preserve.

    Parameters
    ----------
    n_samples : int
        Total number of frames.
    excluded_ranges : list[tuple[int, int]]
        Half-open artifact-masked frame ranges (need not be sorted/merged).
    boundary_spans : list[tuple[int, int]]
        Half-open frame spans that never cross a join (e.g. from
        ``boundary_spans_from_timestamps`` / ``concat_boundary_spans``).

    Returns
    -------
    list[tuple[int, int]]
        Sorted, half-open statistics spans.

    Raises
    ------
    ValueError
        If no artifact-free frame lies inside any boundary span.
    """
    from spyglass.utils import logger

    n_samples = int(n_samples)
    valid = complement_frame_ranges(excluded_ranges, n_samples)
    valid_samples = sum(b - a for a, b in valid)
    out: list[tuple[int, int]] = []
    for a, b in valid:
        for c, d in boundary_spans:
            lo, hi = max(a, c), min(b, d)
            if lo < hi:
                out.append((lo, hi))
    if not out:
        raise ValueError(
            "statistics_spans: no artifact-free samples inside any "
            "acquisition span."
        )
    out = sorted(out)
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


def spans_cover_recording(spans, n_samples: int) -> bool:
    """True when ``spans`` is ``None`` or exactly ``[(0, n_samples)]``.

    The span estimators delegate to SpikeInterface's own unchanged path in
    exactly this case, so an unmasked, unjoined recording gets
    SpikeInterface's estimates bit for bit.
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

    When ``target_samples`` covers every valid frame, every span is read
    once, in span order, with no randomness. Otherwise each span's exact
    quota of rows is apportioned by largest-remainder rounding of its
    length share, then filled with randomly-placed contiguous pieces (each
    at most ``max_piece`` frames and never longer than its own span); the
    pieces are concatenated in sorted start-frame order (not draw order).

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
    import numpy as np

    rng = np.random.default_rng(seed)
    lengths = np.array([b - a for a, b in spans], dtype=np.int64)
    total_valid = int(lengths.sum())
    effective_target = min(int(target_samples), total_valid)
    if effective_target == total_valid:
        # The budget covers every valid sample: read each span once, no
        # overlapping draws.
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
    # Largest-remainder apportionment: quotas sum to effective_target exactly
    # and never exceed a span's length.
    raw = effective_target * lengths / total_valid
    quota = np.minimum(np.floor(raw).astype(np.int64), lengths)
    for i in np.argsort(-(raw - np.floor(raw))):
        if quota.sum() >= effective_target:
            break
        if quota[i] < lengths[i]:
            quota[i] += 1
    # (start_frame, piece) pairs, so the final concatenation can be ordered
    # by start frame across all spans rather than by draw order.
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

    Starts are drawn uniformly from the set of admissible positions ``s``
    with ``a <= s`` and ``s + nsamples <= b`` across all spans, i.e. each
    admissible position is equally likely regardless of which span holds
    it (implemented as span selection weighted by each span's admissible-
    position count, then a uniform draw within the chosen span). A span
    shorter than ``nsamples`` admits no positions and is never chosen.

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
    import numpy as np

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
