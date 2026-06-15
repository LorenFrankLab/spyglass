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

This is the v2-local, DB-free counterpart of v1's
``spikesorting/utils.py:_init_artifact_worker`` / ``_compute_artifact_chunk``
(v2 keeps its own copy rather than importing the v1-era helper, which itself
pulls DataJoint).
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

    Mirrors v1's ``spikesorting/utils.py:_init_artifact_worker`` (this is a
    self-contained v2 copy -- v2 must not import the v1 helper). On a
    multi-process pool the ``recording`` arrives as a ``to_dict()`` blob and is
    re-hydrated with ``si.load`` (SI 0.104 renamed v1's ``load_extractor``);
    on the single-process / thread path the live recording object is passed
    straight through. ``n_required`` is constant across chunks, so it is
    resolved once here and cached in the worker context rather than
    recomputed per chunk.
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
    Returns the GLOBAL (``+ start_frame``) ascending frame indices flagged in
    this chunk.

    Peak working set per chunk ≈ ``4 × (end_frame - start_frame) × n_channels ×
    4 bytes`` (µV float32 slice + abs + z-score intermediate),
    independent of the full recording length.
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

    local = (channel_hit.sum(axis=1) >= n_required).nonzero()[0]
    return local + start_frame
