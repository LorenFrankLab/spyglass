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


def apply_artifact_mask(
    recording, valid_times, *, artifact_detection_id=None, recording_id=None
):
    """Zero out the complement of ``valid_times`` on the recording.

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
        The recording to mask; the complement of ``valid_times`` is
        zeroed in place of the kept intervals.
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
    si.BaseRecording
        The artifact-masked recording, or the input recording
        unchanged when no artifact frames need zeroing.

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
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._signal_math import (
        _segment_times_at,
        assert_artifact_frame_fraction,
        assert_positive_sampling_frequency,
        frames_for_times,
    )
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
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
    if not (
        np.isfinite(t_first)
        and np.isfinite(t_last)
        and t_last >= t_first
    ):
        raise ValueError(
            "apply_artifact_mask: recording endpoints are non-finite or step "
            f"backward (t_first={t_first}, t_last={t_last}); the persisted "
            "recording's monotonic timestamp invariant is violated and the "
            "searchsorted frame mapping would silently mis-mask."
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
    # has ~1-sample spacing to its neighbor, so it is kept. (A 1-sample
    # artifact landing exactly on a chunk's final sample is the lone
    # uncovered edge; negligible at the chunk boundary.) Read only the two
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
        return (ts_end - ts_start) > 1.5 * sample_period

    frame_ranges = [
        (s, e)
        for (s, e) in frame_ranges
        if not _is_interchunk_boundary_range(s, e)
    ]

    if not frame_ranges:
        return recording

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
    return sip.silence_periods(
        recording,
        list_periods=[frame_ranges],
        mode="zeros",
    )
