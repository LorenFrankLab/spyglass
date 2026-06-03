"""Hermetic absolute-outcome test for gain-aware µV artifact detection.

``artifact._compute_artifact_chunk`` scales raw counts to microvolts with
the recording's stored per-channel gains (``traces_uv = traces * gains``)
before comparing against ``amplitude_thresh_uV``. Every other direct
artifact test uses ``gain = 1.0``, where counts and µV are numerically
identical, so the conversion itself is never exercised. This test plants a
peak at a KNOWN non-unity gain and asserts the absolute detect / no-detect
outcome at thresholds that straddle the converted value -- a result that
only holds if the gain is actually applied (treating counts as µV would
flag the peak at the high threshold).

The detection math is pure / synthetic (a NumPy recording, no populate),
but importing ``artifact`` declares its ``@schema`` tables, which needs a
live connection to the local test database. The module therefore requests
``dj_conn`` (matching the existing ``_detect_artifacts`` tests in
``test_disjoint_artifact.py``); it does not run a sort, so it is not
``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("dj_conn")


def _planted_recording(gain=0.195, n_ch=4, n_frames=200):
    """SI recording: zeros + a 3000-count peak (f50) and 2000-count (f100).

    At ``gain = 0.195`` µV/count the peaks are 585 µV and 390 µV. Returns
    the recording with that gain stored on every channel.
    """
    import spikeinterface as si

    traces = np.zeros((n_frames, n_ch), dtype=np.float32)
    traces[50, :] = 3000.0  # -> 585 µV at gain 0.195
    traces[100, :] = 2000.0  # -> 390 µV at gain 0.195
    rec = si.NumpyRecording([traces], sampling_frequency=30000.0)
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    return rec, n_frames


def _flagged_frames(rec, n_frames, amplitude_thresh_uV):
    """Run the amplitude-only chunk detector over the whole recording.

    ``proportion_above_thresh = 1.0`` requires every channel to exceed the
    threshold; ``zscore_thresh = None`` disables the z-score detector so the
    outcome is governed purely by the µV amplitude comparison.
    """
    from spyglass.spikesorting.v2.artifact import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )

    ctx = _init_artifact_worker(
        rec,
        zscore_thresh=None,
        amplitude_thresh_uV=amplitude_thresh_uV,
        proportion_above_thresh=1.0,
    )
    return _compute_artifact_chunk(
        segment_index=0,
        start_frame=0,
        end_frame=n_frames,
        worker_ctx=ctx,
    )


def test_gain_conversion_detects_peak_above_uv_threshold():
    """Threshold 500 µV: only the 585 µV peak (frame 50) is flagged.

    The 390 µV bump (frame 100) is below threshold. Exact frame set.
    """
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_thresh_uV=500.0)
    np.testing.assert_array_equal(flagged, np.array([50]))


def test_gain_conversion_no_detect_when_uv_below_threshold():
    """Threshold 700 µV: NOTHING is flagged -- proves the gain is applied.

    The peak is 3000 counts -> 585 µV < 700, so no frame is flagged. If the
    conversion were skipped (counts compared directly), 3000 > 700 would
    wrongly flag frame 50. The empty result is the gain-correctness signal.
    """
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_thresh_uV=700.0)
    assert flagged.tolist() == [], (
        "frame 50 (3000 counts = 585 µV) must not exceed a 700 µV threshold; "
        f"got {flagged.tolist()} -- gain conversion was not applied."
    )


def test_gain_conversion_low_threshold_flags_both_peaks():
    """Threshold 300 µV: both the 585 µV and 390 µV peaks are flagged."""
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_thresh_uV=300.0)
    np.testing.assert_array_equal(flagged, np.array([50, 100]))
