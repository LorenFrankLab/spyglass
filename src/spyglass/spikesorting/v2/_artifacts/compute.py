"""Worker kernels for the chunked artifact scan.

Spawned ``n_jobs>1`` workers re-import the module defining the worker
function, and importing a schema module opens a DB connection. Keep this
module free of DataJoint and ``spyglass.common`` imports.
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

    Resolves ``n_required`` once per worker.

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

    OR-combines the amplitude and across-channel z-score detectors on µV
    traces. The z-score uses only each frame's own channels, so chunk
    boundaries do not change the result.

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
        Contiguous flagged-frame runs as ascending ``(start_inclusive,
        end_inclusive)`` global frame pairs, shape ``(n_runs, 2)``. Runs keep
        the result sized by events, not samples; ``detect_artifacts`` joins
        runs across chunk seams.
    """
    recording = worker_ctx["recording"]
    zscore_threshold = worker_ctx["zscore_threshold"]
    amplitude_threshold_uv = worker_ctx["amplitude_threshold_uv"]
    n_required = worker_ctx["n_required"]

    # SI applies gain and offset; the scaled read is already float32.
    traces_uv = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float32, copy=False)

    # NaN compares False against every threshold, so it would flag nothing.
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

    channel_hit = np.zeros(traces_uv.shape, dtype=bool)
    if amplitude_threshold_uv is not None:
        channel_hit |= np.abs(traces_uv) > amplitude_threshold_uv
    if zscore_threshold is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        channel_hit |= zscores > zscore_threshold

    # Padded diff: +1 is a run start, -1 is one past a run end.
    flagged = channel_hit.sum(axis=1) >= n_required
    edges = np.diff(np.concatenate(([False], flagged, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return np.stack([starts + start_frame, ends + start_frame], axis=1).astype(
        np.int64
    )
