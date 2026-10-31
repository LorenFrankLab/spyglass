"""Database-free artifact detection and interval-row construction.

These functions are the interval core behind the artifact-detection tables:
building the artifact-removed ``valid_times`` from a preprocessed
recording (``scan_artifact_frames`` -> ``detect_artifacts``), building the
``IntervalList`` row contents (``build_artifact_interval_rows``), building
the ownership part rows (``build_artifact_interval_part_rows``). Reading
stored rows and ownership cleanup live in ``_artifact_readers``. These functions are
pure (non-DB) compute over SpikeInterface objects, aside from
``detect_artifacts``'s diagnostic ``logger`` calls. The per-chunk kernels
live in ``_artifact_compute``; ``scan_artifact_frames`` drives them through
SpikeInterface's ``ChunkRecordingExecutor``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all numpy / SpikeInterface / spyglass dependencies
are imported lazily inside the functions. Construction never queries or
mutates a database.
"""

from __future__ import annotations


def scan_artifact_frames(recording, validated, job_kwargs=None):
    """Flag contiguous artifact-frame RUNS via a chunked ``ChunkRecordingExecutor``.

    Scans chunk by chunk (``chunk_duration='1s'`` when ``job_kwargs`` has no
    chunk-size key) and concatenates the per-chunk flagged-frame RUN ranges
    into one ascending ``(n_runs, 2)`` array. Peak memory is about
    ``4 × chunk_frames × n_channels × 4 bytes`` per chunk instead of the full
    recording (≈110 GB for 1 h, 64 channels, 30 kHz), and returning runs
    rather than per-frame indices bounds the result by the number of
    artifact events.

    Only SI ``job_keys`` from ``job_kwargs`` reach the executor. ``n_jobs=1``
    (the default) runs serially and passes the live recording to the worker;
    a process pool receives the ``to_dict()`` blob instead.

    Parameters
    ----------
    recording : si.BaseRecording
        Preprocessed recording to scan for artifact frames.
    validated : ArtifactDetectionParamsSchema
        Validated detection parameters supplying the amplitude /
        z-score thresholds and the proportion-above-threshold.
    job_kwargs : dict, optional
        Merged SI job-kwargs blob. Default ``None`` runs serially with
        a 1 s chunk.

    Returns
    -------
    np.ndarray
        Contiguous flagged-frame RUNS as ascending ``(start_inclusive,
        end_inclusive)`` frame pairs, shape ``(n_runs, 2)`` (empty
        ``(0, 2)`` when nothing is flagged). ``detect_artifacts`` joins these
        across chunk seams and splits them at wall-clock gaps.

    Raises
    ------
    ArtifactFractionExceededError
        If the total flagged sample count exceeds the
        ``_MAX_ARTIFACT_FRAME_FRACTION`` guard (a misconfigured detector).
    """
    import numpy as np
    from spikeinterface.core.job_tools import (
        ChunkRecordingExecutor,
        ensure_n_jobs,
        job_keys,
    )

    from spyglass.spikesorting.v2._artifacts.compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )
    from spyglass.spikesorting.v2._core.signal_math import (
        assert_artifact_frame_fraction,
    )

    resolved = dict(job_kwargs or {})
    exec_kwargs = {k: resolved[k] for k in job_keys if k in resolved}
    # With no chunk-size key ChunkRecordingExecutor loads the whole recording
    # as one chunk, so default to 1 s for callers (e.g. direct
    # ``detect_artifacts`` calls) that did not merge SI's global job kwargs.
    _CHUNK_KEYS = (
        "chunk_size",
        "chunk_duration",
        "chunk_memory",
        "total_memory",
    )
    if not any(k in exec_kwargs for k in _CHUNK_KEYS):
        exec_kwargs["chunk_duration"] = "1s"
    n_jobs = ensure_n_jobs(recording, n_jobs=exec_kwargs.get("n_jobs", 1))
    rec_arg = recording if n_jobs == 1 else recording.to_dict()
    init_args = (
        rec_arg,
        validated.zscore_threshold,
        validated.amplitude_threshold_uv,
        validated.proportion_above_threshold,
    )
    executor = ChunkRecordingExecutor(
        recording=recording,
        func=_compute_artifact_chunk,
        init_func=_init_artifact_worker,
        init_args=init_args,
        handle_returns=True,
        job_name="detect_artifact_frames",
        **exec_kwargs,
    )
    per_chunk = executor.run()
    if not per_chunk:
        return np.empty((0, 2), dtype=np.int64)
    runs = np.concatenate(per_chunk)  # (total_runs, 2) ascending (start, end)
    # Flagging more than the bound is a misconfigured detector: fail here
    # rather than at the mask stage.
    total_flagged = int(np.sum(runs[:, 1] - runs[:, 0] + 1)) if runs.size else 0
    assert_artifact_frame_fraction(
        total_flagged,
        recording.get_num_samples(segment_index=0),
        context="Artifact-detection scan: ",
    )
    return runs


def _split_runs_at_gaps(runs, gap_after):
    """Split each ``(start, end_inclusive)`` run at every wall-clock gap inside it.

    A fixed-size scan chunk can straddle a wall-clock discontinuity, so a single
    contiguous flagged-frame run may span a gap. A gap must be a split point for
    the join, so splitting runs here guarantees no run fed to the join spans a
    gap, and the joined spans -- and therefore the final valid_times -- equal
    those of a per-frame join that splits at every gap. Bounded by
    ``n_runs + n_gaps`` (both small), never by the flagged-frame count.

    Parameters
    ----------
    runs : np.ndarray
        Ascending, disjoint ``(start, end_inclusive)`` frame-range pairs,
        shape ``(n_runs, 2)``.
    gap_after : np.ndarray
        Frame indices that are the last sample before a wall-clock gap.

    Returns
    -------
    np.ndarray
        The runs with any gap-spanning run split, shape ``(n_runs', 2)``.
    """
    import numpy as np

    if gap_after.size == 0:
        return runs
    out = []
    for start, end in runs:
        start = int(start)
        end = int(end)
        # A gap at frame g (g+1 is across the discontinuity) splits the run only
        # when it falls strictly inside it (g < end); g == end means the gap is
        # at the run's trailing edge, so the whole run is on one side already.
        gaps_in = gap_after[(gap_after >= start) & (gap_after < end)]
        if gaps_in.size == 0:
            out.append((start, end))
            continue
        prev = start
        for g in gaps_in:
            out.append((prev, int(g)))
            prev = int(g) + 1
        out.append((prev, end))
    return np.array(out, dtype=np.int64)


def detect_artifacts(recording, validated, context="", job_kwargs=None):
    """Run amplitude / z-score artifact scan on a SI recording.

    When ``detect`` is False the full recording window is returned
    untouched.

    The threshold scan runs chunk by chunk via ``scan_artifact_frames``
    (SpikeInterface's ``ChunkRecordingExecutor``) so peak memory is bounded
    by the chunk size rather than the full recording -- see that function's
    docstring for the per-chunk memory formula and the default chunk size.

    Parameters
    ----------
    recording : si.BaseRecording
        Preprocessed recording to scan.
    validated : ArtifactDetectionParamsSchema
        Validated detection parameters (thresholds, removal / join
        windows, ``min_length_s``, and the ``detect`` flag).
    context : str, optional
        Caller-supplied string (e.g.
        ``" for artifact_detection_id=... recording_id=..."``)
        appended to the zero-frames warning so an operator can
        identify which selection scanned empty. Default ``""``.
    job_kwargs : dict, optional
        Merged per-row / config / global SI job-kwargs blob,
        resolved by the caller via ``_resolved_job_kwargs`` and
        forwarded to the executor. Default ``None``.

    Returns
    -------
    np.ndarray
        Artifact-removed valid times in seconds, shape
        ``(n_intervals, 2)``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import (
        _segment_times_at,
        assert_positive_sampling_frequency,
        base_intervals_and_gaps,
        subtract_intervals,
    )
    from spyglass.utils import logger

    fs = assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="detect_artifacts: "
    )
    # Recorded chunks (split at wall-clock gaps) and the frame index of the last
    # sample before each gap (diff > 1.5 sample periods), read chunk by chunk
    # rather than via ``get_times()`` (~824 MB for 1 h at 30 kHz). Every
    # returned interval stays inside one chunk: one spanning a gap would
    # inflate obs_intervals and let sub-min_length slivers borrow gap time.
    base_intervals, gap_after = base_intervals_and_gaps(recording, fs)
    if not validated.detect:
        logger.info(
            "Artifact detection: detect=False; returning the recorded "
            "window(s) as valid intervals."
        )
        return np.asarray(base_intervals)

    logger.info(
        "Artifact detection: scanning with "
        f"amplitude_threshold_uv={validated.amplitude_threshold_uv}, "
        f"zscore_threshold={validated.zscore_threshold}, "
        f"proportion_above_threshold={validated.proportion_above_threshold}, "
        f"removal_window_ms={validated.removal_window_ms}, "
        f"join_window_ms={validated.join_window_ms}, "
        f"min_length_s={validated.min_length_s}."
    )

    # Degenerate-configuration guards. The z-score detector standardizes
    # ACROSS channels within a frame, so it is amplitude-sensitive only with
    # >= 3 channels: on 1 channel it is identically zero, and on 2 channels it
    # is a constant +/-1 for any two distinct values (independent of
    # amplitude). Either way a z-score-only config flags every frame or none,
    # never by amplitude.
    n_channels = recording.get_num_channels()
    if validated.zscore_threshold is not None and n_channels < 3:
        if validated.amplitude_threshold_uv is None:
            from spyglass.spikesorting.v2.exceptions import (
                InsufficientZScoreChannelsError,
            )

            raise InsufficientZScoreChannelsError(
                "Artifact detection: zscore_threshold is the only detector on a "
                f"{n_channels}-channel recording{context}, but the cross-channel "
                "z-score is amplitude-sensitive only with >= 3 channels (on 1 "
                "channel it is identically zero; on 2 it is a constant +/-1 for "
                "any two distinct values), so it would not detect artifacts by "
                "amplitude. Use amplitude_threshold_uv for 1-2 channel sort "
                "groups."
            )
        logger.warning(
            "Artifact detection: zscore_threshold is inert on a "
            f"{n_channels}-channel recording{context} (the cross-channel "
            "z-score is amplitude-insensitive with < 3 channels); only "
            "amplitude_threshold_uv will fire."
        )
    # proportion_above_threshold rounds UP (ceil) to a channel count, so on a
    # small group a sub-1.0 proportion can silently require ALL channels (e.g.
    # 0.7 on a stereotrode -> ceil(1.4)=2 of 2 = 100%). Warn so the realized
    # requirement is visible; the detection math itself is unchanged.
    n_required = int(np.ceil(validated.proportion_above_threshold * n_channels))
    if validated.proportion_above_threshold < 1.0 and n_required >= n_channels:
        logger.warning(
            "Artifact detection: proportion_above_threshold="
            f"{validated.proportion_above_threshold} on a {n_channels}-channel "
            f"group rounds up (ceil) to requiring ALL {n_channels} channels"
            f"{context}; the effective threshold is stricter than the nominal "
            "fraction. Lower proportion_above_threshold or accept the "
            "all-channel requirement."
        )

    # The per-frame math (in ``_artifact_compute``) OR-combines the amplitude
    # and across-channel z-score detectors (an AND would make the
    # dual-threshold mode less sensitive than either single-threshold mode);
    # the z-score uses each frame's own channels, so chunk boundaries do not
    # change the flagged set.
    runs = scan_artifact_frames(recording, validated, job_kwargs)
    if len(runs) == 0:
        # Distinguishes attempted-and-empty from the detect=False skip.
        logger.warning(
            "Artifact detection: scan found zero artifact frames"
            f"{context} (amplitude_threshold_uv="
            f"{validated.amplitude_threshold_uv}, zscore_threshold="
            f"{validated.zscore_threshold}, proportion_above_threshold="
            f"{validated.proportion_above_threshold}); returning the "
            "recorded window(s) as valid intervals."
        )
        return np.asarray(base_intervals)

    half_window_frames = int(
        np.ceil(validated.removal_window_ms * 1e-3 * fs / 2)
    )
    join_window_frames = int(np.ceil(validated.join_window_ms * 1e-3 * fs))

    # Frame indices are contiguous across a wall-clock gap, so the join and the
    # removal-window expansion must both be gap-aware: frame-adjacent artifacts
    # in different chunks are seconds apart, and joining or dilating across
    # the gap would over-mask the neighboring chunk.
    n = recording.get_num_samples(segment_index=0)

    # Split runs at gaps first (a scan chunk can straddle one), then join runs
    # within join_window; the result equals a per-frame join's.
    runs = _split_runs_at_gaps(runs, gap_after)
    spans = []
    cur_start = int(runs[0, 0])
    cur_end = int(runs[0, 1])
    for r_start, r_end in runs[1:]:
        r_start = int(r_start)
        r_end = int(r_end)
        crosses_gap = bool(
            np.any((gap_after >= cur_end) & (gap_after < r_start))
        )
        if r_start - cur_end <= join_window_frames and not crosses_gap:
            cur_end = max(cur_end, r_end)
        else:
            spans.append((cur_start, cur_end))
            cur_start = r_start
            cur_end = r_end
    spans.append((cur_start, cur_end))

    # Cap the removal-window expansion at the chunk boundary on each side.
    clipped_spans = []
    for start_f, end_f in spans:
        left = gap_after[gap_after < start_f]
        chunk_start = int(left[-1]) + 1 if left.size else 0
        right = gap_after[gap_after >= end_f]
        chunk_end = int(right[0]) if right.size else n - 1
        start_f = max(chunk_start, start_f - half_window_frames)
        end_f = min(chunk_end, end_f + half_window_frames)
        # Half-open [start, end): ``end_f`` is the inclusive last artifact
        # sample, so ``end_f + 1`` keeps it out of the complement (the saved
        # valid_times). At a chunk's last sample, ``end_f + 1`` is the next
        # chunk's first timestamp; the per-chunk subtraction below clips to
        # exactly that element (its ``base_start``), so valid_times never
        # cross the gap.
        clipped_spans.append((start_f, min(end_f + 1, n - 1)))

    # One batched read for every span boundary (a noisy recording can have
    # hundreds of spans; per-span reads are many tiny h5py reads).
    boundary_frames = np.array(
        [frame for span in clipped_spans for frame in span], dtype=np.int64
    )
    boundary_times = _segment_times_at(recording, boundary_frames)
    artifact_intervals = [
        [float(boundary_times[2 * i]), float(boundary_times[2 * i + 1])]
        for i in range(len(clipped_spans))
    ]

    # Subtract per base chunk, not from one ``[t0, t_end]`` envelope, so no
    # kept interval spans a gap ``Recording.make`` excluded.
    # ``artifact_intervals`` is start-sorted.
    kept = subtract_intervals(base_intervals, artifact_intervals)

    # Drop slivers shorter than ``min_length_s``: a noisy recording otherwise
    # leaves millisecond intervals the mask iterates one by one, and SI
    # sorters may crash on them.
    if kept:
        kept = [
            [start, end]
            for start, end in kept
            if (end - start) >= validated.min_length_s
        ]
    if not kept:
        # Surface the cause now rather than as an
        # EmptyArtifactValidTimesError at sort time.
        logger.warning(
            "Artifact detection: after removing artifacts and dropping "
            f"intervals shorter than min_length_s={validated.min_length_s}, "
            f"NO valid time remains{context}. The sorter will reject this "
            "recording (EmptyArtifactValidTimesError). Loosen the thresholds "
            "(amplitude_threshold_uv / zscore_threshold / "
            "proportion_above_threshold), reduce removal_window_ms, or lower "
            "min_length_s."
        )
        return np.empty((0, 2))
    return np.asarray(kept)


def build_artifact_interval_rows(
    key, valid_times, nwb_file_name, per_member_nwb_files=()
):
    """Build the artifact-removed ``IntervalList`` row contents.

    Returns ONE row dict per distinct member ``nwb_file_name`` so each
    affected session sees the artifact times. For the single-recording
    path this is one row keyed by the master's ``nwb_file_name``. Every
    row carries the same ``valid_times`` (the detection ran once over the
    -- possibly unioned -- channels) and the
    ``spikesorting_artifact_detection_v2``
    pipeline tag. The interval name is centralized in
    ``artifact_detection_interval_list_name``.

    Parameters
    ----------
    key : dict
        Restriction carrying ``artifact_detection_id``, used to build
        the interval list name.
    valid_times : np.ndarray
        Artifact-removed valid times, shape ``(n_intervals, 2)``.
    nwb_file_name : str
        Parent session used as the fallback target when
        ``per_member_nwb_files`` is empty.
    per_member_nwb_files : tuple, optional
        Distinct member ``nwb_file_name`` s to build one row each.
        Default ``()`` falls back to ``(nwb_file_name,)``.

    Returns
    -------
    list[dict]
        One ``IntervalList`` row dict per target ``nwb_file_name``.
    """
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    interval_list_name = artifact_detection_interval_list_name(
        key["artifact_detection_id"]
    )
    targets = per_member_nwb_files or (nwb_file_name,)
    return [
        {
            "nwb_file_name": member_nwb,
            "interval_list_name": interval_list_name,
            "valid_times": valid_times,
            "pipeline": "spikesorting_artifact_detection_v2",
        }
        for member_nwb in targets
    ]


def build_artifact_interval_part_rows(key, interval_rows):
    """Build the ``*ArtifactDetection.RemovedInterval`` ownership rows.

    Parameters
    ----------
    key : dict
        Restriction carrying ``artifact_detection_id``.
    interval_rows : iterable of dict
        Rows produced by :func:`build_artifact_interval_rows`.

    Returns
    -------
    list[dict]
        One part row per generated ``IntervalList`` row. Each row carries
        only the ``artifact_detection_id`` (detection PK) plus the ``IntervalList`` PK, so
        it is safe to pass to the part table without relying on
        ``ignore_extra_fields``.
    """
    return [
        {
            "artifact_detection_id": key["artifact_detection_id"],
            "nwb_file_name": row["nwb_file_name"],
            "interval_list_name": row["interval_list_name"],
        }
        for row in interval_rows
    ]
