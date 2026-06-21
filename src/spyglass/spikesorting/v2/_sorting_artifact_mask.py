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
    import numpy as _np
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._signal_math import (
        assert_monotonic_timestamps,
        assert_positive_sampling_frequency,
    )
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )

    valid_times = _np.asarray(valid_times, dtype=float)
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
    if _np.any(ends < starts):
        raise ValueError(
            "_apply_artifact_mask: valid_times has an interval whose "
            "end precedes its start; each interval must be "
            "(start <= end)."
        )
    if valid_times.shape[0] > 1 and (
        _np.any(_np.diff(starts) < 0) or _np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(
            "_apply_artifact_mask: valid_times must be sorted by start "
            "time and non-overlapping (the complement walker assumes "
            f"monotonic, disjoint input); got {valid_times.tolist()!r}. "
            "Sort and merge the intervals before passing them."
        )

    timestamps = recording.get_times()
    assert_monotonic_timestamps(timestamps, context="apply_artifact_mask: ")
    # Walk the valid intervals left-to-right, collecting the
    # complement (artifact gaps) as a list of (start_frame,
    # end_frame) pairs. Each pair is materialized via
    # ``np.arange`` and concatenated -- ~100x faster than the
    # equivalent ``list.extend(range(start, end))`` for
    # multi-million-sample recordings (Python-int boxing for
    # every frame is the bottleneck of the loop form).
    frame_ranges: list[tuple[int, int]] = []
    cursor = timestamps[0]
    for vs, ve in valid_times:
        if vs > cursor:
            start = int(_np.searchsorted(timestamps, cursor))
            end = int(_np.searchsorted(timestamps, vs))
            if end > start:
                frame_ranges.append((start, end))
        cursor = max(cursor, ve)
    if cursor < timestamps[-1]:
        start = int(_np.searchsorted(timestamps, cursor))
        end = len(timestamps)
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
    # uncovered edge; negligible at the chunk boundary.)
    sample_period = 1.0 / assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="apply_artifact_mask: "
    )
    frame_ranges = [
        (s, e)
        for (s, e) in frame_ranges
        if not (
            e - s == 1
            and e < len(timestamps)
            and (timestamps[e] - timestamps[s]) > 1.5 * sample_period
        )
    ]

    if not frame_ranges:
        return recording

    artifact_frames = _np.concatenate(
        [_np.arange(s, e, dtype=_np.int64) for s, e in frame_ranges]
    )
    # SI's ``list_triggers`` is documented as a list-of-arrays, one
    # per event channel. We pass a single numpy array wrapping ALL
    # artifact frames so the mask zeros every artifact sample
    # exactly once, independent of how SI interprets a bare Python
    # list of scalars in future minor versions.
    #
    # ``ms_before=None, ms_after=None`` is the documented "single
    # sample" mode and uses the ``pad is None`` code path in SI
    # 0.104's RemoveArtifactsRecordingSegment.get_traces (each
    # trigger zeros exactly its own frame). ``ms_before=0.0,
    # ms_after=0.0`` looks equivalent but triggers a boundary
    # bug in SI 0.104 where the first artifact frame in any
    # contiguous run is left unmasked (the ``trig - pad[0] <= 0``
    # branch assigns to an empty slice).
    return sip.remove_artifacts(
        recording=recording,
        list_triggers=[artifact_frames],
        ms_before=None,
        ms_after=None,
        mode="zeros",
    )
