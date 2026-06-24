"""Artifact-removed interval construction + IntervalList persistence.

These functions are the interval core behind ``ArtifactDetection``:
building the artifact-removed ``valid_times`` from a preprocessed
recording (``scan_artifact_frames`` -> ``detect_artifacts``), building the
``IntervalList`` row contents (``build_artifact_interval_rows``), building
the ownership part rows (``build_artifact_interval_part_rows``), reading
those rows back (``read_artifact_removed_intervals``), and the delete-time
IntervalList cleanup policy (``collect_artifact_interval_rows_to_remove``
+ ``remove_artifact_interval_rows``). The construction functions are
pure (non-DB) compute over SpikeInterface objects -- aside from
``detect_artifacts``'s diagnostic ``logger`` calls; the persistence
functions touch the DB at CALL time via lazy imports so the table class
stays a thin orchestrator (fetch -> compute -> write/delete).

Why this lives in its own module rather than in ``artifact.py``:
``artifact.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The interval
construction needs none of that at import, so ``ArtifactDetection``
becomes a thin orchestrator. Same "thin DataJoint shell over pure/IO
services" direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_dispatch`` / ``_recording_restriction`` / ``_recording_geometry`` /
``_recording_preprocessing`` / ``_recording_nwb``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all numpy / SpikeInterface / spyglass dependencies
are imported lazily inside the functions. The persistence functions
(``read_artifact_removed_intervals``, ``collect_artifact_interval_rows_to_remove``,
``remove_artifact_interval_rows``) DO touch the DB at call time -- they
lazy-import ``common.IntervalList``, the v2 ``recording.RecordingSelection``,
plus ``ArtifactDetection`` / ``ArtifactDetectionSelection`` /
``SharedArtifactGroup`` back from ``artifact`` (a backward import that is
cycle-free because ``artifact`` is fully imported by call time). The pure-compute kernels live in
``_artifact_compute``; ``scan_artifact_frames`` drives them through
SpikeInterface's ``ChunkRecordingExecutor``.
"""

from __future__ import annotations


def scan_artifact_frames(recording, validated, job_kwargs=None):
    """Flag artifact frame indices via a chunked ``ChunkRecordingExecutor``.

    Scans the recording chunk by chunk (defaulting to
    ``chunk_duration='1s'`` when ``job_kwargs`` carries no chunk-size key,
    so the scan is bounded
    for every caller, not only the production path that merges SI's global
    job kwargs) and concatenates the per-chunk flagged frame
    indices into one ascending array. Peak working set is bounded by the
    chunk size -- roughly ``4 × chunk_frames × n_channels × 4 bytes`` (raw
    int slice + float32 µV copy + abs + z-score intermediate) -- rather than
    by the full recording, which a full-``get_traces`` load would
    materialize at ``~4 × n_samples × n_channels × 4 bytes`` (≈27 GB for a
    1-hour 64-channel 30 kHz recording).

    ``job_kwargs`` is the merged SI job-kwargs blob; only recognized
    ``job_keys`` (``n_jobs``, ``chunk_duration``, ``pool_engine``, ...) are
    forwarded to the executor. ``n_jobs=1`` (the default) runs
    serially in-process and passes the live recording to each worker; a
    multi-process pool receives the ``to_dict()`` blob instead.

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
        Ascending flagged frame indices, shape ``(n_flagged,)``.
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

    resolved = dict(job_kwargs or {})
    exec_kwargs = {k: resolved[k] for k in job_keys if k in resolved}
    # Chunk BY DEFAULT. ChunkRecordingExecutor with no chunk-size key
    # processes the whole recording in a single chunk -- i.e. the exact
    # full-traces memory profile this restoration removed. A caller that
    # reaches here without a chunk key (any direct ``detect_artifacts``
    # call, not just the production ``make_compute`` path that merges SI's
    # global ``chunk_duration='1s'``) must still be bounded, so default the
    # chunk to 1 s of data. Callers can override via job_kwargs (and a
    # single-pass-in-memory mode is still reachable with an explicit
    # ``chunk_size`` spanning the recording).
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
        return np.empty(0, dtype=np.int64)
    return np.concatenate(per_chunk)


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
        assert_positive_sampling_frequency,
    )
    from spyglass.spikesorting.v2.utils import (
        _base_intervals_from_timestamps,
    )
    from spyglass.utils import logger

    timestamps = recording.get_times()
    fs = assert_positive_sampling_frequency(
        recording.get_sampling_frequency(), context="detect_artifacts: "
    )
    # Recorded chunks, split at wall-clock discontinuities. For a
    # contiguous recording this is a single ``[t0, t_end]``; for
    # disjoint sort intervals it is one interval per chunk, so the
    # detect=False / zero-artifact returns AND the artifact complement
    # below never span an inter-chunk gap (which would inflate
    # obs_intervals duration and let sub-min_length slivers survive by
    # borrowing gap time).
    base_intervals = _base_intervals_from_timestamps(timestamps, fs)
    if not validated.detect:
        logger.info(
            "ArtifactDetection: detect=False; returning the recorded "
            "window(s) as valid intervals."
        )
        return np.asarray(base_intervals)

    # Log the threshold configuration so a population report can
    # audit which detection mode actually fired per row.
    logger.info(
        "ArtifactDetection: scanning with "
        f"amplitude_threshold_uv={validated.amplitude_threshold_uv}, "
        f"zscore_threshold={validated.zscore_threshold}, "
        f"proportion_above_threshold={validated.proportion_above_threshold}, "
        f"removal_window_ms={validated.removal_window_ms}, "
        f"join_window_ms={validated.join_window_ms}, "
        f"min_length_s={validated.min_length_s}."
    )

    # Degenerate-configuration guards. The z-score detector is
    # cross-channel WITHIN a frame, so on a single-channel group it is
    # identically zero -- a z-score-only config would silently flag nothing.
    n_channels = recording.get_num_channels()
    if validated.zscore_threshold is not None and n_channels < 2:
        if validated.amplitude_threshold_uv is None:
            from spyglass.spikesorting.v2.exceptions import (
                SingleChannelZScoreError,
            )

            raise SingleChannelZScoreError(
                "ArtifactDetection: zscore_threshold is the only detector on a "
                f"{n_channels}-channel recording{context}, but the z-score is "
                "computed ACROSS channels within each frame -- with <2 "
                "channels it is identically zero, so NO artifacts would be "
                "detected. Use amplitude_threshold_uv for single-channel sort "
                "groups."
            )
        logger.warning(
            "ArtifactDetection: zscore_threshold is inert on a "
            f"{n_channels}-channel recording{context} (the cross-channel "
            "z-score is identically zero on one channel); only "
            "amplitude_threshold_uv will fire."
        )
    # proportion_above_threshold rounds UP (ceil) to a channel count, so on a
    # small group a sub-1.0 proportion can silently require ALL channels (e.g.
    # 0.7 on a stereotrode -> ceil(1.4)=2 of 2 = 100%). Warn so the realized
    # requirement is visible; the detection math itself is unchanged.
    n_required = int(
        np.ceil(validated.proportion_above_threshold * n_channels)
    )
    if validated.proportion_above_threshold < 1.0 and n_required >= n_channels:
        logger.warning(
            "ArtifactDetection: proportion_above_threshold="
            f"{validated.proportion_above_threshold} on a {n_channels}-channel "
            f"group rounds up (ceil) to requiring ALL {n_channels} channels"
            f"{context}; the effective threshold is stricter than the nominal "
            "fraction. Lower proportion_above_threshold or accept the "
            "all-channel requirement."
        )

    # Chunked scan via ChunkRecordingExecutor (see
    # ``scan_artifact_frames``). The per-frame detection math --
    # µV traces (gain+offset applied by SpikeInterface), amplitude
    # threshold, and the
    # across-channel (``axis=1``) z-score with its ``+1e-12`` std
    # epsilon -- lives in the module-level ``_compute_artifact_chunk``
    # worker. The z-score is computed on each frame's own columns, so
    # chunk boundaries (which split the time axis) leave the flagged
    # frame set unchanged; this is the equivalence the regression test
    # pins. The worker OR-combines the two detectors (an AND would make
    # the dual-threshold mode strictly less sensitive than either
    # single-threshold mode).
    frames_above = scan_artifact_frames(recording, validated, job_kwargs)
    if len(frames_above) == 0:
        # Warn so a downstream consumer noticing "all valid times" can
        # see whether detection was attempted-and-empty vs skipped.
        logger.warning(
            "ArtifactDetection: scan found zero artifact frames"
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

    # Inter-chunk wall-clock gaps: frame index of the last sample
    # before each gap (diff > 1.5 sample periods). Frame indices are
    # contiguous across a gap, so BOTH the join below and the
    # removal-window expansion further down must be gap-aware --
    # otherwise an artifact near a chunk edge bridges/spills into the
    # neighboring chunk across the gap.
    n = len(timestamps)
    gap_after = np.flatnonzero(np.diff(timestamps) > 1.5 * (1.0 / fs))

    # Build artifact intervals in frame indices, then convert to
    # seconds and subtract per base chunk. Join nearby artifact frames
    # into spans, but NEVER across a timestamp gap: two artifacts in
    # different chunks are frame-adjacent yet seconds apart in
    # wall-clock, so joining them would create a span straddling the
    # gap that over-masks the earlier chunk's tail. The gap-aware
    # guard below keeps the join from bridging a large disjoint gap.
    spans = []
    cur_start = frames_above[0]
    cur_end = frames_above[0]
    for f in frames_above[1:]:
        crosses_gap = bool(np.any((gap_after >= cur_end) & (gap_after < f)))
        if f - cur_end <= join_window_frames and not crosses_gap:
            cur_end = f
        else:
            spans.append((cur_start, cur_end))
            cur_start = f
            cur_end = f
    spans.append((cur_start, cur_end))

    # Cap the removal-window expansion at the CHUNK boundary on each
    # side, for the same reason: a frame-space window (``end_f +
    # half_window_frames``) on an artifact near a chunk edge would
    # reach into the neighboring chunk's first samples across the gap
    # and -- after per-chunk clipping -- remove them. Capping the
    # window at the chunk frame bounds keeps it from crossing a gap
    # orders of magnitude larger than the window.
    artifact_intervals = []
    for start_f, end_f in spans:
        left = gap_after[gap_after < start_f]
        chunk_start = int(left[-1]) + 1 if left.size else 0
        right = gap_after[gap_after >= end_f]
        chunk_end = int(right[0]) if right.size else n - 1
        start_f = max(chunk_start, start_f - half_window_frames)
        end_f = min(chunk_end, end_f + half_window_frames)
        # ``end_f`` is the INCLUSIVE last artifact sample, but the
        # interval is stored as a half-open ``[start, end)`` where
        # ``end`` is the first non-artifact sample. Otherwise the
        # complement (saved as valid_times) would silently include
        # ``timestamps[end_f]`` -- an artifact sample -- in the
        # next valid interval, and ``Sorting._apply_artifact_mask``
        # would fail to mask that sample before the sort.
        #
        # When ``end_f == chunk_end`` (artifact reaches the chunk's last
        # sample), ``end_f + 1`` indexes the NEXT chunk's first
        # timestamp, so this intermediate interval can span the
        # inter-chunk gap. That is harmless:
        # ``artifact_intervals`` is a local never exposed to consumers,
        # and the per-chunk subtraction below clips each interval to its
        # own base chunk. The half-open ``end`` is exactly the same
        # array element the next chunk uses as its ``base_start``, so the
        # clip is an identity (not a float coincidence) and the saved
        # valid_times never cross the gap.
        end_time = timestamps[min(end_f + 1, n - 1)]
        artifact_intervals.append([timestamps[start_f], end_time])

    # Subtract artifact intervals from each recorded base chunk
    # SEPARATELY so a kept valid interval never spans an inter-chunk
    # wall-clock gap. Subtracting from one envelope
    # ``[timestamps[0], timestamps[-1]]`` would reintroduce the gaps
    # ``Recording.make`` excluded -- inflating obs_intervals duration
    # and letting a sub-min_length sliver survive by borrowing gap
    # time. ``artifact_intervals`` is start-sorted; clip each to the
    # current chunk and walk the complement within it.
    kept = []
    artifact_index = 0
    n_artifact_intervals = len(artifact_intervals)
    for base_start, base_end in base_intervals:
        while (
            artifact_index < n_artifact_intervals
            and artifact_intervals[artifact_index][1] <= base_start
        ):
            artifact_index += 1
        cursor = base_start
        scan_index = artifact_index
        while (
            scan_index < n_artifact_intervals
            and artifact_intervals[scan_index][0] < base_end
        ):
            art_start, art_end = artifact_intervals[scan_index]
            clipped_start = max(art_start, base_start)
            clipped_end = min(art_end, base_end)
            if clipped_end <= clipped_start:
                scan_index += 1
                continue  # artifact does not overlap this chunk
            if clipped_start > cursor:
                kept.append([cursor, clipped_start])
            cursor = max(cursor, clipped_end)
            scan_index += 1
        if cursor < base_end:
            kept.append([cursor, base_end])

    # Drop valid-interval slivers shorter than ``min_length_s``
    # (default 1.0 s) before returning. Without this, a noisy
    # recording with frequent artifacts leaves millisecond-scale
    # slivers between artifact intervals that downstream
    # ``Sorting._apply_artifact_mask`` iterates one by one; SI
    # sorters may also crash on micro-intervals.
    if kept:
        kept = [
            [start, end]
            for start, end in kept
            if (end - start) >= validated.min_length_s
        ]
    if not kept:
        # Artifacts WERE found (we are past the zero-frames early return), but
        # removal-window dilation + the min_length_s sliver filter dropped
        # every valid interval. Warn here (mirroring the zero-frames warning)
        # so the operator sees the cause at detection time rather than three
        # stages later as an EmptyArtifactValidTimesError at sort time.
        logger.warning(
            "ArtifactDetection: after removing artifacts and dropping "
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
    """Build ``ArtifactDetection.ArtifactRemovedInterval`` ownership rows.

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
        only the ``ArtifactDetection`` PK plus the ``IntervalList`` PK, so
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
        Restriction selecting a single ``ArtifactDetection`` row;
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
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    # Validate the key BEFORE resolving the source so a missing
    # ``artifact_detection_id`` surfaces this clear ValueError rather than a
    # source-resolution / SchemaBypassError from ``resolve_source``.
    if "artifact_detection_id" not in key:
        raise ValueError(
            "ArtifactDetection.get_artifact_removed_intervals: key "
            "must include 'artifact_detection_id'."
        )
    source = ArtifactDetectionSelection.resolve_source(key)
    part_rows = (
        ArtifactDetection.ArtifactRemovedInterval & key
    ).fetch("nwb_file_name", "interval_list_name", as_dict=True)
    if not part_rows:
        raise ValueError(
            "ArtifactDetection.get_artifact_removed_intervals: "
            f"{key!r} has no ArtifactRemovedInterval part rows. "
            "ArtifactDetection rows must own their generated IntervalList "
            "rows through the part table; re-populate this artifact "
            "detection."
        )

    if source.kind == "recording":
        if len(part_rows) != 1:
            raise ValueError(
                "ArtifactDetection.get_artifact_removed_intervals: "
                f"recording-backed {key!r} has {len(part_rows)} "
                "ArtifactRemovedInterval part rows; expected exactly one."
            )
        nwb_file_name = part_rows[0]["nwb_file_name"]
        interval_list_name = part_rows[0]["interval_list_name"]
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # as_dict wraps the single array as a one-key dict so callers
        # can treat both source kinds uniformly (see Parameters).
        return {nwb_file_name: valid_times} if as_dict else valid_times

    # shared_artifact_group source: collect the member
    # nwb_file_names through the Member -> RecordingSelection
    # join, fetch each member's IntervalList row, return a
    # name->valid_times dict.
    members = (
        SharedArtifactGroup.Member * RecordingSelection
        & {
            "shared_artifact_group_name": source.key[
                "shared_artifact_group_name"
            ]
        }
    ).fetch("nwb_file_name", as_dict=True)
    member_nwb_files = list(dict.fromkeys(m["nwb_file_name"] for m in members))
    rows = (IntervalList & part_rows).fetch(
        "nwb_file_name", "valid_times", as_dict=True
    )
    result = {row["nwb_file_name"]: row["valid_times"] for row in rows}
    # Match the single-recording ``fetch1`` loudness: a member whose
    # artifact-removed IntervalList row is absent would otherwise be
    # SILENTLY dropped from the dict, and a source-agnostic caller
    # (especially one relying on ``as_dict=True``) would treat the
    # short dict as complete. A missing row means a partially-deleted
    # ArtifactDetection, so raise naming the member(s) instead.
    missing = [nwb for nwb in member_nwb_files if nwb not in result]
    if missing:
        raise ValueError(
            "ArtifactDetection.get_artifact_removed_intervals: shared "
            f"artifact group {source.key['shared_artifact_group_name']!r}"
            f" is missing artifact-removed IntervalList row(s) for "
            f"member(s) {missing} "
            f"(part rows={part_rows!r}). The "
            "ArtifactDetection row may be partially deleted; "
            "re-populate it."
        )
    return result


def collect_artifact_interval_rows_to_remove(rows):
    """Resolve the artifact ``IntervalList`` rows paired with master rows.

    Layer-2 delete-cleanup helper: fetches the owned
    ``ArtifactRemovedInterval`` part rows BEFORE the master delete, while
    they still exist, and returns the matching
    ``{nwb_file_name, interval_list_name}`` restrictions. The caller removes
    those ``IntervalList`` rows after the master delete succeeds.

    Parameters
    ----------
    rows : list of dict
        ``ArtifactDetection`` master row dicts, fetched before the
        master delete while the ownership part rows still exist.

    Returns
    -------
    list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions to remove
        after the master delete commits.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    interval_rows_to_remove = []
    for row in rows:
        part_rows = (
            ArtifactDetection.ArtifactRemovedInterval & row
        ).fetch("nwb_file_name", "interval_list_name", as_dict=True)
        if not part_rows:
            raise ValueError(
                "ArtifactDetection.delete: "
                f"{row!r} has no ArtifactRemovedInterval part rows. "
                "Refusing to guess interval ownership from naming; repair "
                "or re-populate the artifact detection before deleting it."
            )
        interval_rows_to_remove.extend(part_rows)
    return interval_rows_to_remove


def remove_artifact_interval_rows(restrictions):
    """Delete the artifact-removed ``IntervalList`` rows for ``restrictions``.

    Companion to ``collect_artifact_interval_rows_to_remove``: removes the
    matching IntervalList rows AFTER the ``ArtifactDetection`` master delete
    committed. Skips any restriction that matches nothing.

    Parameters
    ----------
    restrictions : list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions from
        ``collect_artifact_interval_rows_to_remove``.
    """
    from spyglass.common import IntervalList

    # Clean up the matching IntervalList rows through cautious_delete
    # (the user already passed the same team-permission check on the
    # parent ArtifactDetection rows above keyed by the same
    # nwb_file_name, so the IntervalList check is a re-verification,
    # not a bypass). super_delete here would silently delete other
    # users' rows under shared lab sessions.
    for restriction in restrictions:
        rows = IntervalList & restriction
        if len(rows) == 0:
            continue
        rows.delete(safemode=False)
