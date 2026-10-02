"""DB-free worker kernels for the chunked artifact scan.

These two functions are the pure-compute core of artifact detection --
``_init_artifact_worker`` (per-worker initializer) and
``_compute_artifact_chunk`` (per-chunk detector). They depend ONLY on
``numpy`` and ``spikeinterface``; they touch no DataJoint schema, table, or
``spyglass.common`` import.

Why they are not in ``artifact.py``: ``scan_artifact_frames`` runs them via
SpikeInterface's ``ChunkRecordingExecutor``, whose ``n_jobs>1`` pool spawns
workers (``spawn`` on macOS) that re-import the module DEFINING the worker
function. Importing a schema module such as ``artifact.py`` opens a DB
connection, so keeping the kernels here lets ``n_jobs>1`` detection run on
workers (or DB-isolated HPC nodes) that cannot reach the database. Do not
import helpers from a schema module here. ``artifact.py`` re-exports both
names.
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

    # Fail loudly on a non-finite chunk instead of silently reporting "no
    # artifacts": a NaN (or Inf) sample compares False against every
    # threshold below (``np.abs(nan) > x`` is False, and a NaN channel
    # poisons that channel's mean/std for the z-score too), so a corrupted
    # chunk -- one bad sample, or a whole all-NaN chunk -- would otherwise
    # flag nothing and vanish into an empty interval list. ``get_traces``
    # already read every sample above, so this scan is nearly free.
    non_finite = ~np.isfinite(traces_uv)
    if non_finite.any():
        channel_ids = recording.get_channel_ids()
        counts = non_finite.sum(axis=0)
        per_channel = ", ".join(
            f"channel {channel_ids[i]}: {int(counts[i])}"
            for i in range(len(counts))
            if counts[i] > 0
        )
        raise ValueError(
            "_compute_artifact_chunk: non-finite (NaN/Inf) samples in "
            f"segment {segment_index} frames [{start_frame}, {end_frame}): "
            f"{per_channel} non-finite sample(s) each."
        )

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
    edges = np.diff(np.concatenate(([False], flagged, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return np.stack([starts + start_frame, ends + start_frame], axis=1).astype(
        np.int64
    )
