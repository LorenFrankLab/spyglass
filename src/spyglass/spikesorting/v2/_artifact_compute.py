"""DB-free worker kernels for the chunked artifact scan.

These two functions are the pure-compute core of ``ArtifactDetection`` --
``_init_artifact_worker`` (per-worker initializer) and
``_compute_artifact_chunk`` (per-chunk detector). They depend ONLY on
``numpy`` and ``spikeinterface``; they touch no DataJoint schema, table, or
``spyglass.common`` import.

Why this lives in its own module rather than in ``artifact.py``:
``ArtifactDetection._scan_artifact_frames`` runs these via SpikeInterface's
``ChunkRecordingExecutor``, which on a multi-process pool (``n_jobs>1``) spawns
worker processes. On macOS the start method is ``spawn``, so each worker is a
fresh interpreter that re-imports the module DEFINING the worker function.
``artifact.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the ``from spyglass.common import ...`` side-effect import,
both of which open a DB connection AT IMPORT. A worker only needs these two
DB-free functions, so defining them here keeps the spawn re-import connection-free:
``n_jobs>1`` artifact detection then works even when a worker (or a DB-isolated
HPC compute node) cannot reach the database, and avoids one redundant DB
connection per worker. ``artifact.py`` re-exports both names, so existing
``from ...v2.artifact import _compute_artifact_chunk`` call sites keep working.

These kernels are kept as a self-contained, DB-free copy so the
spawn-workers need no DB connection -- importing an equivalent helper from a
schema module would pull DataJoint back in at spawn-import time.
"""

from __future__ import annotations

import numpy as np


def _init_artifact_worker(
    recording,
    zscore_threshold,
    amplitude_threshold_uv,
    proportion_above_threshold,
):
    """Per-worker initializer for the chunked artifact scan.

    On a multi-process pool the ``recording`` arrives as a ``to_dict()`` blob
    and is re-hydrated with ``si.load`` (the SI 0.104+ name for the former
    ``load_extractor``); on the single-process / thread path the live
    recording object is passed straight through. ``n_required`` is constant
    across chunks, so it is
    resolved once here and cached in the worker context rather than
    recomputed per chunk.

    Parameters
    ----------
    recording : si.BaseRecording or dict
        Live recording (single-process path) or a ``to_dict()`` blob
        (multi-process path) re-hydrated with ``si.load``.
    zscore_threshold : float or None
        Across-channel z-score threshold, or ``None`` to disable.
    amplitude_threshold_uv : float or None
        Absolute amplitude threshold in µV, or ``None`` to disable.
    proportion_above_threshold : float
        Fraction of channels that must be flagged for a frame to count
        as an artifact; converted to the per-chunk ``n_required`` count.

    Returns
    -------
    dict
        Worker context with keys ``recording``, ``zscore_threshold``,
        ``amplitude_threshold_uv``, and ``n_required``.
    """
    import spikeinterface as si

    recording = si.load(recording) if isinstance(recording, dict) else recording
    n_channels = len(recording.get_channel_ids())
    return {
        "recording": recording,
        "zscore_threshold": zscore_threshold,
        "amplitude_threshold_uv": amplitude_threshold_uv,
        "n_required": int(np.ceil(proportion_above_threshold * n_channels)),
    }


def _compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Flag artifact frame indices within a ``[start_frame, end_frame)`` chunk.

    Reproduces the EXACT per-frame detection math of the former full-in-memory
    scan, applied to a single chunk: read traces already in µV, then
    OR-combine the amplitude and across-channel z-score detectors.
    The across-channel (``axis=1``) z-score uses only the chunk row's own
    columns, so it is identical regardless of where the chunk boundaries fall --
    this is what makes the chunked output frame-identical to the in-memory one.
    Peak working set per chunk ≈ ``4 × (end_frame - start_frame) × n_channels ×
    4 bytes`` (µV float32 slice + abs + z-score intermediate),
    independent of the full recording length.

    Parameters
    ----------
    segment_index : int
        SpikeInterface segment index to read traces from.
    start_frame : int
        Inclusive start frame of the chunk.
    end_frame : int
        Exclusive end frame of the chunk.
    worker_ctx : dict
        Worker context built by :func:`_init_artifact_worker`.

    Returns
    -------
    np.ndarray
        Contiguous flagged-frame RUNS as ascending ``(start_inclusive,
        end_inclusive)`` GLOBAL frame pairs, shape ``(n_runs, 2)``. Returning
        runs rather than one index per flagged frame bounds both this return and
        the executor's collected result to the number of artifact EVENTS, not
        the number of artifact SAMPLES -- a fully-flagged chunk is one run, not
        ``chunk_len`` ids. ``detect_artifacts`` joins runs across chunk seams
        and splits them at wall-clock gaps, so the chunk boundaries (which can
        fall mid-run) do not change the result.
    """
    recording = worker_ctx["recording"]
    zscore_threshold = worker_ctx["zscore_threshold"]
    amplitude_threshold_uv = worker_ctx["amplitude_threshold_uv"]
    n_required = worker_ctx["n_required"]

    # ``return_in_uV=True`` lets SpikeInterface apply the recording's stored
    # per-channel gain AND offset, returning microvolts directly. This is the
    # threshold's unit, and it avoids re-implementing the count->µV conversion
    # (a gain-only scaling would silently ignore any non-zero channel offset).
    traces_uv = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float32)
    absolute = np.abs(traces_uv)

    if amplitude_threshold_uv is not None:
        above_amp = absolute > amplitude_threshold_uv
    else:
        above_amp = np.zeros_like(absolute, dtype=bool)
    if zscore_threshold is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        above_z = zscores > zscore_threshold
    else:
        above_z = np.zeros_like(absolute, dtype=bool)

    if amplitude_threshold_uv is not None and zscore_threshold is not None:
        channel_hit = above_amp | above_z
    elif amplitude_threshold_uv is not None:
        channel_hit = above_amp
    else:
        channel_hit = above_z

    # Collapse the per-frame flagged mask to contiguous run edges via a padded
    # boolean diff: rising edge (+1) = inclusive run start, falling edge (-1) =
    # one past the inclusive run end. This keeps the worker's return O(n_runs)
    # rather than O(n_flagged) for the whole chunk.
    flagged = channel_hit.sum(axis=1) >= n_required
    edges = np.diff(
        np.concatenate(([False], flagged, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return np.stack(
        [starts + start_frame, ends + start_frame], axis=1
    ).astype(np.int64)
