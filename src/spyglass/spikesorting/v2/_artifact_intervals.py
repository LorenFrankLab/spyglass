"""Artifact-removed interval construction + IntervalList persistence.

These functions are the interval core behind the artifact-detection tables:
building the artifact-removed ``valid_times`` from a preprocessed
recording (``scan_artifact_frames`` -> ``detect_artifacts``), building the
``IntervalList`` row contents (``build_artifact_interval_rows``), building
the ownership part rows (``build_artifact_interval_part_rows``), reading
those rows back (``read_artifact_removed_intervals``, and for one recording
``read_recording_artifact_valid_times``), and the delete-time
IntervalList cleanup policy (``collect_artifact_interval_rows_to_remove``
+ ``remove_artifact_interval_rows``). The construction functions are
pure (non-DB) compute over SpikeInterface objects, aside from
``detect_artifacts``'s diagnostic ``logger`` calls. The per-chunk kernels
live in ``_artifact_compute``; ``scan_artifact_frames`` drives them through
SpikeInterface's ``ChunkRecordingExecutor``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all numpy / SpikeInterface / spyglass dependencies
are imported lazily inside the functions. The persistence functions touch
the DB at call time, lazy-importing ``common.IntervalList`` and the
``artifact`` result tables (cycle-free, since ``artifact`` is fully imported
by then).
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

    from spyglass.spikesorting.v2._artifact_compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )
    from spyglass.spikesorting.v2._signal_math import (
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

    from spyglass.spikesorting.v2._signal_math import (
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
    from spyglass.spikesorting.v2.utils import (
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


def read_artifact_removed_intervals(key, as_dict=False):
    """Return the artifact-removed ``valid_times`` for ``key``.

    Single-recording source: one ``IntervalList`` row keyed by
    the recording's parent ``nwb_file_name`` -- returned as a
    plain ``(n_intervals, 2)`` ndarray (or, with ``as_dict=True``,
    a one-key ``{nwb_file_name: ndarray}`` dict).

    Shared-artifact-group source: ``make_insert`` writes one
    ``IntervalList`` row per distinct member ``nwb_file_name``
    (today single-session, so length 1). Returns a dict
    keyed by ``nwb_file_name`` mapping to the per-member
    ``valid_times`` array. All values are equal across keys
    (the detection ran ONCE over the unioned channels and
    ``make_insert`` wrote the same array per member), so a
    caller that wants a single array can ``next(iter(d.values()))``.

    Parameters
    ----------
    key : dict
        Restriction selecting a single artifact-detection row (either
        ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``);
        must include ``artifact_detection_id``.
    as_dict : bool, optional
        If ``False`` (default), the return type depends on the
        source: a plain ``(n_intervals, 2)`` ndarray for a
        single-recording source, a ``{nwb_file_name: ndarray}`` dict
        for a shared-artifact-group source. If ``True``, BOTH sources
        return the dict shape (a single-recording result is wrapped
        as a one-key dict), so source-agnostic callers can avoid
        branching on the return type.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        For a single-recording source with ``as_dict=False``, the
        ``(n_intervals, 2)`` artifact-removed ``valid_times`` array.
        For a shared-artifact-group source, or any source with
        ``as_dict=True``, a dict mapping each member ``nwb_file_name``
        to its ``(n_intervals, 2)`` array.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        SharedGroupArtifactDetection,
    )

    if "artifact_detection_id" not in key:
        raise ValueError(
            "get_artifact_removed_intervals: key must include "
            "'artifact_detection_id'."
        )
    # The id is unique across the two result tables, so it is in exactly one.
    if RecordingArtifactDetection & key:
        return RecordingArtifactDetection().get_artifact_removed_intervals(
            key, as_dict=as_dict
        )
    if SharedGroupArtifactDetection & key:
        return SharedGroupArtifactDetection().get_artifact_removed_intervals(
            key, as_dict=as_dict
        )
    raise ValueError(
        "get_artifact_removed_intervals: artifact_detection_id "
        f"{key['artifact_detection_id']!r} is not in RecordingArtifactDetection "
        "or SharedGroupArtifactDetection. Populate the artifact detection "
        "before reading its removed intervals."
    )


def read_recording_artifact_valid_times(
    artifact_detection_id, nwb_file_name: str, *, caller: str
):
    """Return one recording's artifact-removed ``valid_times``.

    Reads through :func:`read_artifact_removed_intervals`, which validates
    that the detection's ``RemovedInterval`` part rows own the
    ``IntervalList``, rather than fetching the ``IntervalList`` by its
    reconstructed name: that direct fetch would accept a partially deleted
    detection or a hand-inserted same-name ``IntervalList``. The per-nwb dict
    form covers single-recording and shared-group detections alike.

    Parameters
    ----------
    artifact_detection_id : uuid.UUID
        The per-source detection id.
    nwb_file_name : str
        The recording's parent NWB file.
    caller : str
        Prefix of the error message.

    Returns
    -------
    np.ndarray
        The ``(n_intervals, 2)`` artifact-removed valid times in seconds.

    Raises
    ------
    ValueError
        If the detection holds no intervals for ``nwb_file_name``.
    """
    intervals_by_nwb = read_artifact_removed_intervals(
        {"artifact_detection_id": artifact_detection_id}, as_dict=True
    )
    if nwb_file_name not in intervals_by_nwb:
        raise ValueError(
            f"{caller}: artifact-removed intervals for "
            f"nwb_file_name={nwb_file_name!r} not found among "
            f"{sorted(intervals_by_nwb)} for artifact_detection_id="
            f"{artifact_detection_id!r}; the artifact-detection row may be "
            "partially deleted."
        )
    return intervals_by_nwb[nwb_file_name]


def read_owned_artifact_intervals(detection_cls, key):
    """Return the artifact-removed ``valid_times`` a detection row owns.

    Source-agnostic reader for the split ``*ArtifactDetection`` result
    tables: reads the detection row's OWN ``RemovedInterval`` part rows and
    the ``IntervalList`` rows they own, keyed by ``nwb_file_name``. Because
    the source kind is structural (a ``RecordingArtifactDetection`` owns
    exactly one row; a ``SharedGroupArtifactDetection`` owns one per distinct
    member ``nwb_file_name``), this reader stays uniform -- the per-table
    ``get_artifact_removed_intervals`` shapes the return (a bare array for a
    single-recording source, the dict for a shared-group source).

    Reads only OWNED part rows, so the returned set is exactly what
    ``make_insert`` wrote -- a missing part-row IntervalList row surfaces
    loudly through ``fetch1`` (a partially-deleted detection) rather than
    being silently dropped.

    Parameters
    ----------
    detection_cls : dj.Computed
        The split result table whose ownership part rows to read
        (``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``).
    key : dict
        Restriction selecting a single detection row; must include
        ``artifact_detection_id``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{nwb_file_name: (n_intervals, 2) valid_times}`` for every owned
        ``RemovedInterval`` row.
    """
    from spyglass.common import IntervalList

    if "artifact_detection_id" not in key:
        raise ValueError(
            f"{detection_cls.__name__}.get_artifact_removed_intervals: key "
            "must include 'artifact_detection_id'."
        )
    part_rows = (detection_cls.RemovedInterval & key).fetch(
        "nwb_file_name", "interval_list_name", as_dict=True
    )
    if not part_rows:
        raise ValueError(
            f"{detection_cls.__name__}.get_artifact_removed_intervals: "
            f"{key!r} has no RemovedInterval part rows. Detection "
            "rows must own their generated IntervalList rows through the part "
            "table; re-populate this artifact detection."
        )
    result = {}
    for part_row in part_rows:
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": part_row["nwb_file_name"],
                "interval_list_name": part_row["interval_list_name"],
            }
        ).fetch1("valid_times")
        result[part_row["nwb_file_name"]] = valid_times
    return result


def collect_artifact_interval_rows_to_remove(rows, detection_cls):
    """Resolve the artifact ``IntervalList`` rows paired with master rows.

    Layer-2 delete-cleanup helper: fetches the owned ``RemovedInterval`` part
    rows BEFORE the master delete, while they still exist, and returns the
    matching ``{nwb_file_name, interval_list_name}`` restrictions. The caller
    removes those ``IntervalList`` rows after the master delete succeeds.

    Parameters
    ----------
    rows : list of dict
        Detection master row dicts, fetched before the master delete while
        the ownership part rows still exist.
    detection_cls : dj.Computed
        The split result table owning the part rows
        (``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``).

    Returns
    -------
    list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions to remove
        after the master delete commits.
    """
    interval_rows_to_remove = []
    part_table = detection_cls.RemovedInterval
    for row in rows:
        part_rows = (part_table & row).fetch(
            "nwb_file_name", "interval_list_name", as_dict=True
        )
        if not part_rows:
            raise ValueError(
                f"{detection_cls.__name__}.delete: "
                f"{row!r} has no RemovedInterval part rows. "
                "Refusing to guess interval ownership from naming; repair "
                "or re-populate the artifact detection before deleting it."
            )
        interval_rows_to_remove.extend(part_rows)
    return interval_rows_to_remove


def remove_artifact_interval_rows(restrictions):
    """Delete the artifact-removed ``IntervalList`` rows for ``restrictions``.

    Companion to ``collect_artifact_interval_rows_to_remove``: removes the
    matching IntervalList rows AFTER the artifact-detection master delete
    committed. Skips any restriction that matches nothing.

    Parameters
    ----------
    restrictions : list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions from
        ``collect_artifact_interval_rows_to_remove``.
    """
    from spyglass.common import IntervalList

    # SpyglassMixin ``.delete`` re-checks team permission (the caller already
    # passed it on the detection rows for the same nwb_file_name);
    # safemode=False only skips the re-prompt. ``super_delete`` would skip
    # the check and could delete other users' rows in a shared session.
    for restriction in restrictions:
        rows = IntervalList & restriction
        if len(rows) == 0:
            continue
        rows.delete(safemode=False)
