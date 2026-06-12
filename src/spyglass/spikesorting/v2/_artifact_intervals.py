"""Artifact-removed interval construction + IntervalList persistence.

These functions are the interval core behind ``ArtifactDetection``:
building the artifact-removed ``valid_times`` from a preprocessed
recording (``scan_artifact_frames`` -> ``detect_artifacts``), building the
``IntervalList`` row contents (``build_artifact_interval_rows``), reading
those rows back (``read_artifact_removed_intervals``), and the delete-time
IntervalList cleanup policy (``collect_artifact_interval_rows_to_remove``
+ ``remove_artifact_interval_rows``). The two construction functions are
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
``_sorting_compute`` / ``_recording_materialization``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all numpy / SpikeInterface / spyglass dependencies
are imported lazily inside the functions. The persistence functions
(``read_artifact_removed_intervals``, ``collect_artifact_interval_rows_to_remove``,
``remove_artifact_interval_rows``) DO touch the DB at call time -- they
lazy-import ``common.IntervalList``, the v2 ``recording`` tables, and
``ArtifactSelection`` / ``SharedArtifactGroup`` back from ``artifact``
(a backward import that is cycle-free because ``artifact`` is fully
imported by call time). The pure-compute kernels live in
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
    by the full recording, which the predecessor full-``get_traces`` load
    materialized at ``~4 × n_samples × n_channels × 4 bytes`` (≈27 GB for a
    1-hour 64-channel 30 kHz recording). Restores v1's chunked path
    (``v1/artifact.py:_get_artifact_times`` +
    ``spikesorting/utils.py:_compute_artifact_chunk``) self-contained in v2.

    ``job_kwargs`` is the merged SI job-kwargs blob; only recognized
    ``job_keys`` (``n_jobs``, ``chunk_duration``, ``pool_engine``, ...) are
    forwarded to the executor. ``n_jobs=1`` (the default, matching v1) runs
    serially in-process and passes the live recording to each worker; a
    multi-process pool receives the ``to_dict()`` blob instead.

    Returns the ascending ndarray of flagged frame indices.
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
        validated.zscore_thresh,
        validated.amplitude_thresh_uV,
        validated.proportion_above_thresh,
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

    Returns an ``ndarray`` of shape ``(n_intervals, 2)`` containing
    the artifact-removed valid times in seconds. When ``detect`` is
    False the full recording window is returned untouched.

    The threshold scan runs chunk by chunk via ``scan_artifact_frames``
    (SpikeInterface's ``ChunkRecordingExecutor``) so peak memory is bounded
    by the chunk size rather than the full recording -- see that function's
    docstring for the per-chunk memory formula and the default chunk size.

    ``job_kwargs`` is the merged per-row / config / global SI job-kwargs
    blob; the caller resolves it via ``_resolved_job_kwargs`` and passes it
    through to the executor. ``context`` is an optional caller-supplied
    string (e.g. ``" for artifact_id=... recording_id=..."``) appended to
    the zero-frames warning so an operator can identify which selection
    scanned empty.
    """
    import numpy as _np

    from spyglass.spikesorting.v2.utils import (
        _base_intervals_from_timestamps,
    )
    from spyglass.utils import logger

    timestamps = recording.get_times()
    fs = recording.get_sampling_frequency()
    # Recorded chunks, split at wall-clock discontinuities. For a
    # contiguous recording this is a single ``[t0, t_end]``; for
    # disjoint sort intervals it is one interval per chunk, so the
    # detect=False / zero-artifact returns AND the artifact complement
    # below never span an inter-chunk gap (which would inflate
    # obs_intervals duration and let sub-min_length slivers survive by
    # borrowing gap time). v1 subtracts artifacts from the explicit
    # ``sort_interval_valid_times`` (``v1/artifact.py:327``).
    base_intervals = _base_intervals_from_timestamps(timestamps, fs)
    if not validated.detect:
        logger.info(
            "ArtifactDetection: detect=False; returning the recorded "
            "window(s) as valid intervals."
        )
        return _np.asarray(base_intervals)

    # Log the threshold configuration so a population report can
    # audit which detection mode actually fired per row. Matches
    # v1's logger.info at v1/artifact.py:258-261.
    logger.info(
        "ArtifactDetection: scanning with "
        f"amplitude_thresh_uV={validated.amplitude_thresh_uV}, "
        f"zscore_thresh={validated.zscore_thresh}, "
        f"proportion_above_thresh={validated.proportion_above_thresh}, "
        f"removal_window_ms={validated.removal_window_ms}, "
        f"join_window_ms={validated.join_window_ms}, "
        f"min_length_s={validated.min_length_s}."
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
    # pins. v1 OR-combined the two detectors at
    # ``spikesorting/utils.py:198``; the worker preserves that (an AND
    # would make the dual-threshold mode strictly less sensitive than
    # either single-threshold mode).
    frames_above = scan_artifact_frames(recording, validated, job_kwargs)
    if len(frames_above) == 0:
        # Matches v1's warning at v1/artifact.py:318 so a
        # downstream consumer noticing "all valid times" can see
        # whether detection was attempted-and-empty vs skipped.
        logger.warning(
            "ArtifactDetection: scan found zero artifact frames"
            f"{context} (amplitude_thresh_uV="
            f"{validated.amplitude_thresh_uV}, zscore_thresh="
            f"{validated.zscore_thresh}, proportion_above_thresh="
            f"{validated.proportion_above_thresh}); returning the "
            "recorded window(s) as valid intervals."
        )
        return _np.asarray(base_intervals)

    half_window_frames = int(
        _np.ceil(validated.removal_window_ms * 1e-3 * fs / 2)
    )
    join_window_frames = int(_np.ceil(validated.join_window_ms * 1e-3 * fs))

    # Inter-chunk wall-clock gaps: frame index of the last sample
    # before each gap (diff > 1.5 sample periods). Frame indices are
    # contiguous across a gap, so BOTH the join below and the
    # removal-window expansion further down must be gap-aware --
    # otherwise an artifact near a chunk edge bridges/spills into the
    # neighboring chunk across the gap.
    n = len(timestamps)
    gap_after = _np.flatnonzero(_np.diff(timestamps) > 1.5 * (1.0 / fs))

    # Build artifact intervals in frame indices, then convert to
    # seconds and subtract per base chunk. Join nearby artifact frames
    # into spans, but NEVER across a timestamp gap: two artifacts in
    # different chunks are frame-adjacent yet seconds apart in
    # wall-clock, so joining them would create a span straddling the
    # gap that over-masks the earlier chunk's tail. v1's effective
    # join is in timestamp space (after time-based window expansion),
    # so a large disjoint gap is never bridged.
    spans = []
    cur_start = frames_above[0]
    cur_end = frames_above[0]
    for f in frames_above[1:]:
        crosses_gap = bool(_np.any((gap_after >= cur_end) & (gap_after < f)))
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
    # and -- after per-chunk clipping -- remove them. v1's window is
    # time-based (``timestamps[end] + half_win`` seconds), which
    # cannot cross a gap orders of magnitude larger than the window;
    # capping at the chunk frame bounds reproduces that.
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
        # inter-chunk gap. That is harmless (audit finding #4):
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
    for base_start, base_end in base_intervals:
        cursor = base_start
        for art_start, art_end in artifact_intervals:
            clipped_start = max(art_start, base_start)
            clipped_end = min(art_end, base_end)
            if clipped_end <= clipped_start:
                continue  # artifact does not overlap this chunk
            if clipped_start > cursor:
                kept.append([cursor, clipped_start])
            cursor = max(cursor, clipped_end)
        if cursor < base_end:
            kept.append([cursor, base_end])

    # Drop valid-interval slivers shorter than ``min_length_s``
    # (default 1.0 s) before returning. Without this, a noisy
    # recording with frequent artifacts leaves millisecond-scale
    # slivers between artifact intervals that downstream
    # ``Sorting._apply_artifact_mask`` iterates one by one; SI
    # sorters may also crash on micro-intervals. Matches v1's
    # hardcoded ``min_length=1`` at
    # ``src/spyglass/spikesorting/v1/artifact.py:327-328``.
    if kept:
        kept = [
            [start, end]
            for start, end in kept
            if (end - start) >= validated.min_length_s
        ]
    return _np.asarray(kept) if kept else _np.empty((0, 2))


def build_artifact_interval_rows(
    key, valid_times, nwb_file_name, per_member_nwb_files=()
):
    """Build the artifact-removed ``IntervalList`` row contents.

    Returns ONE row dict per distinct member ``nwb_file_name`` so each
    affected session sees the artifact times. For the single-recording
    path this is one row keyed by the master's ``nwb_file_name``. Every
    row carries the same ``valid_times`` (the detection ran once over the
    -- possibly unioned -- channels) and the ``spikesorting_artifact_v2``
    pipeline tag. The interval name is centralized in
    ``artifact_interval_list_name``.
    """
    from spyglass.spikesorting.v2.utils import artifact_interval_list_name

    interval_list_name = artifact_interval_list_name(key["artifact_id"])
    # Backwards compat: callers that constructed
    # ``ArtifactComputed`` without ``per_member_nwb_files``
    # (older test stubs) fall back to the single
    # ``nwb_file_name``.
    targets = per_member_nwb_files or (nwb_file_name,)
    return [
        {
            "nwb_file_name": member_nwb,
            "interval_list_name": interval_list_name,
            "valid_times": valid_times,
            "pipeline": "spikesorting_artifact_v2",
        }
        for member_nwb in targets
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
        must include ``artifact_id``.
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
    from spyglass.spikesorting.v2.artifact import (
        ArtifactSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )

    source = ArtifactSelection.resolve_source(key)
    if "artifact_id" not in key:
        raise ValueError(
            "ArtifactDetection.get_artifact_removed_intervals: key "
            "must include 'artifact_id'."
        )
    interval_list_name = artifact_interval_list_name(key["artifact_id"])

    if source.kind == "recording":
        nwb_file_name = (
            RecordingSelection & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name")
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
    rows = (
        IntervalList
        & [
            {
                "nwb_file_name": nwb,
                "interval_list_name": interval_list_name,
            }
            for nwb in member_nwb_files
        ]
    ).fetch("nwb_file_name", "valid_times", as_dict=True)
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
            f"(interval_list_name={interval_list_name!r}). The "
            "ArtifactDetection row may be partially deleted; "
            "re-populate it."
        )
    return result


def collect_artifact_interval_rows_to_remove(rows):
    """Resolve the artifact ``IntervalList`` rows paired with master rows.

    Layer-2 delete-cleanup helper: DataJoint does not cascade through
    ``interval_list_name``-keyed dependencies, so the artifact-removed
    IntervalList rows must be cleaned up explicitly when an
    ``ArtifactDetection`` master is deleted. Given the master row dicts
    (fetched BEFORE the master delete, while the source-part join still
    resolves), returns the list of ``{nwb_file_name, interval_list_name}``
    restrictions the caller should remove after the master delete commits.

    A genuinely broken source (zero/multiple source part rows -- the only
    thing ``resolve_source`` raises) cannot be resolved to its IntervalList
    target; that row is logged and skipped (its IntervalList rows are left
    in place) rather than aborting the whole cleanup. Any OTHER exception
    propagates.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_interval_list_name,
    )
    from spyglass.utils import logger

    interval_rows_to_remove = []
    for row in rows:
        try:
            source = ArtifactSelection.resolve_source(row)
        except SchemaBypassError as resolve_exc:
            # A genuinely broken source (zero/multiple source part
            # rows -- the only thing ``resolve_source`` raises) cannot
            # be resolved to its IntervalList target. Log, skip ITS
            # paired cleanup, and let the master delete proceed. Any
            # OTHER exception is an unexpected bug: DO NOT swallow it
            # (a broad ``except`` would silently orphan IntervalList
            # rows). Let it propagate to abort the delete BEFORE the
            # master is removed, so the operator sees the failure.
            logger.error(
                "ArtifactDetection.delete: resolve_source failed for "
                f"{ {k: row[k] for k in ('artifact_id',)} }; leaving "
                f"IntervalList rows in place. Underlying error: "
                f"{resolve_exc!r}"
            )
            continue
        interval_list_name = artifact_interval_list_name(row["artifact_id"])
        if source.kind == "recording":
            nwb_file_name = (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("nwb_file_name")
            interval_rows_to_remove.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": interval_list_name,
                }
            )
        elif source.kind == "shared_artifact_group":
            # ``make_insert`` wrote one IntervalList row per
            # distinct member ``nwb_file_name``; collect them
            # all for cleanup. The Member -> RecordingSelection
            # join surfaces the per-member nwb_file_name set.
            members = (
                SharedArtifactGroup.Member * RecordingSelection
                & {
                    "shared_artifact_group_name": source.key[
                        "shared_artifact_group_name"
                    ]
                }
            ).fetch("nwb_file_name", as_dict=True)
            seen = set()
            for m in members:
                nwb = m["nwb_file_name"]
                if nwb in seen:
                    continue
                seen.add(nwb)
                interval_rows_to_remove.append(
                    {
                        "nwb_file_name": nwb,
                        "interval_list_name": interval_list_name,
                    }
                )
    return interval_rows_to_remove


def remove_artifact_interval_rows(restrictions):
    """Delete the artifact-removed ``IntervalList`` rows for ``restrictions``.

    Companion to ``collect_artifact_interval_rows_to_remove``: removes the
    matching IntervalList rows AFTER the ``ArtifactDetection`` master delete
    committed. Skips any restriction that matches nothing.
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
