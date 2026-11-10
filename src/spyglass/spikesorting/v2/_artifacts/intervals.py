"""Artifact detection and interval-row construction.

Builds artifact-removed ``valid_times`` and the ``IntervalList`` / ownership
rows. Readers live in ``_artifacts.readers``; per-chunk kernels in
``_artifacts.compute``.
"""

from __future__ import annotations

import numpy as np

from spyglass.spikesorting.v2._artifacts.compute import (
    _compute_artifact_chunk,
    _init_artifact_worker,
)
from spyglass.spikesorting.v2._artifacts.naming import (
    artifact_detection_interval_list_name,
)
from spyglass.spikesorting.v2._core.signal_math import (
    _segment_times_at,
    assert_artifact_frame_fraction,
    assert_positive_sampling_frequency,
    base_intervals_and_gaps,
    subtract_intervals,
)
from spyglass.spikesorting.v2.exceptions import InsufficientZScoreChannelsError
from spyglass.utils import logger


def scan_artifact_frames(recording, validated, job_kwargs=None):
    """Flag contiguous artifact-frame RUNS via a chunked ``ChunkRecordingExecutor``.

    Chunk size defaults to 1 s when ``job_kwargs`` sets none, so peak memory
    scales with the chunk, not the recording. Only SI ``job_keys`` reach the
    executor.

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
    from spikeinterface.core.job_tools import (
        ChunkRecordingExecutor,
        ensure_n_jobs,
        job_keys,
    )

    resolved = dict(job_kwargs or {})
    exec_kwargs = {k: resolved[k] for k in job_keys if k in resolved}
    # Without a chunk-size key the executor loads the whole recording at once.
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
    total_flagged = int(np.sum(runs[:, 1] - runs[:, 0] + 1)) if runs.size else 0
    assert_artifact_frame_fraction(
        total_flagged,
        recording.get_num_samples(segment_index=0),
        context="Artifact-detection scan: ",
    )
    return runs


def _split_runs_at_gaps(runs, gap_after):
    """Split each ``(start, end_inclusive)`` run at every wall-clock gap inside it.

    A scan chunk can straddle a gap; splitting here keeps the join from
    bridging one.

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
    if gap_after.size == 0:
        return runs
    out = []
    for start, end in runs:
        start = int(start)
        end = int(end)
        # g == end is a gap at the trailing edge, which needs no split.
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

    Parameters
    ----------
    recording : si.BaseRecording
        Preprocessed recording to scan.
    validated : ArtifactDetectionParamsSchema
        Validated detection parameters (thresholds, removal / join
        windows, ``min_length_s``, and the ``detect`` flag).
    context : str, optional
        Suffix for warnings identifying the selection. Default ``""``.
    job_kwargs : dict, optional
        Resolved SI job kwargs (``_resolved_job_kwargs``). Default ``None``.

    Returns
    -------
    np.ndarray
        Artifact-removed valid times in seconds, shape
        ``(n_intervals, 2)``.
    """
    fs = assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="detect_artifacts: "
    )
    # Every returned interval stays inside one gap-free chunk; spanning a gap
    # would let sub-min_length slivers borrow gap time.
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

    # The cross-channel z-score ignores amplitude with fewer than 3 channels.
    n_channels = recording.get_num_channels()
    if validated.zscore_threshold is not None and n_channels < 3:
        if validated.amplitude_threshold_uv is None:
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
    # ceil() can turn a sub-1.0 proportion into all channels on small groups.
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

    # Frames are contiguous across wall-clock gaps, so the join and the
    # removal-window expansion must not cross a gap.
    n = recording.get_num_samples(segment_index=0)

    runs = _split_runs_at_gaps(runs, gap_after)
    run_starts = runs[:, 0]
    run_ends = runs[:, 1]
    # A new span starts past join_window or across a gap.
    prev_ends = run_ends[:-1]
    next_starts = run_starts[1:]
    breaks = next_starts - prev_ends > join_window_frames
    if gap_after.size:
        first_gap = np.searchsorted(gap_after, prev_ends, side="left")
        breaks |= (first_gap < gap_after.size) & (
            gap_after[np.minimum(first_gap, gap_after.size - 1)] < next_starts
        )
    break_idx = np.flatnonzero(breaks)
    span_starts = run_starts[np.concatenate(([0], break_idx + 1))]
    span_ends = run_ends[np.concatenate((break_idx, [len(runs) - 1]))]

    # Cap the removal-window expansion at the span's chunk boundaries.
    if gap_after.size:
        n_left = np.searchsorted(gap_after, span_starts, side="left")
        chunk_start = np.where(
            n_left > 0, gap_after[np.maximum(n_left - 1, 0)] + 1, 0
        )
        right = np.searchsorted(gap_after, span_ends, side="left")
        chunk_end = np.where(
            right < gap_after.size,
            gap_after[np.minimum(right, gap_after.size - 1)],
            n - 1,
        )
    else:
        chunk_start = np.zeros_like(span_starts)
        chunk_end = np.full_like(span_ends, n - 1)
    clipped_starts = np.maximum(chunk_start, span_starts - half_window_frames)
    clipped_ends = np.minimum(chunk_end, span_ends + half_window_frames)
    # Half-open end excludes the last artifact sample from valid_times. At a
    # chunk end it lands on the next chunk's start, which subtraction clips.
    clipped_ends = np.minimum(clipped_ends + 1, n - 1)

    # One batched read; per-span reads are many tiny h5py reads.
    boundary_frames = np.column_stack((clipped_starts, clipped_ends)).ravel()
    boundary_times = _segment_times_at(recording, boundary_frames)
    artifact_intervals = [
        [float(boundary_times[2 * i]), float(boundary_times[2 * i + 1])]
        for i in range(len(span_starts))
    ]

    # Per base chunk, so no kept interval spans a gap.
    kept = subtract_intervals(base_intervals, artifact_intervals)

    # Drop short slivers (SI sorters may crash on them). Untouched chunks stay;
    # preprocessing already chose which short chunks to keep.
    untouched = {(float(start), float(end)) for start, end in base_intervals}
    kept = [
        [start, end]
        for start, end in kept
        if (end - start) >= validated.min_length_s or (start, end) in untouched
    ]
    if not kept:
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

    One row per distinct member ``nwb_file_name``, all with the same
    ``valid_times``.

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
        One part row per ``IntervalList`` row, holding only the detection and
        ``IntervalList`` primary keys.
    """
    return [
        {
            "artifact_detection_id": key["artifact_detection_id"],
            "nwb_file_name": row["nwb_file_name"],
            "interval_list_name": row["interval_list_name"],
        }
        for row in interval_rows
    ]
