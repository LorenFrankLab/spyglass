"""Regression tests for v1-parity spike-time -> frame readback.

v2 originally read sorted units back with
``NwbSortingExtractor(file_path, fs, t_start)``, which inverts the
stored absolute spike times *affinely*
(``frame = round((t - t_start) * fs)``). That is correct only when the
recording's timeline is a uniform ``t_start + i/fs`` grid. For DISJOINT
sort intervals, v2 persists gap-preserving (non-uniform) timestamps, so
affine inversion shifts every frame after a gap by the accumulated
wall-clock gap -- and can push frames past the gap-excluded sample
count. v1 maps back with ``np.searchsorted`` against the actual
timestamps, recovering the original frame exactly.

The hermetic test here pins the readback helper's math; the slow
integration test (``test_get_sorting_recovers_frames_across_gap``)
exercises the full ``Sorting.get_sorting`` / ``CurationV2.get_sorting``
read paths on a real disjoint recording with a monkeypatched sorter.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_spike_times_to_frames_recovers_frames_on_gap_timeline():
    """searchsorted readback recovers original frames where affine fails.

    Build a non-uniform (gap-preserving) timeline: 100 contiguous
    samples, then a 5-second wall-clock gap, then 100 more contiguous
    samples. A spike planted at frame 150 (in the second chunk) is
    stored as its absolute time ``timestamps[150]``. Affine inversion
    ``round((t - t0) * fs)`` would return ``150 + gap*fs`` (shifted off
    the end); ``searchsorted`` recovers exactly 150.
    """
    from spyglass.spikesorting.v2.utils import _spike_times_to_frames

    fs = 30000.0
    dt = 1.0 / fs
    chunk = np.arange(100) * dt
    t0 = 10.0
    gap = 5.0
    timestamps = np.concatenate([t0 + chunk, t0 + chunk[-1] + gap + dt + chunk])
    n_samples = timestamps.size  # 200, gap-excluded
    planted_frames = np.array([10, 150], dtype=np.int64)
    stored_spike_times = timestamps[planted_frames]

    # Affine inversion (the buggy path) shifts the post-gap frame.
    affine = np.round((stored_spike_times - timestamps[0]) * fs).astype(
        np.int64
    )
    assert affine[1] != 150, "test setup: gap must be large enough to shift"
    assert affine[1] >= n_samples, (
        "test setup: affine frame should land out of bounds past the "
        "gap-excluded sample count"
    )

    frames = _spike_times_to_frames(
        timestamps, stored_spike_times, n_samples, unit_id=0
    )
    np.testing.assert_array_equal(frames, planted_frames)


def test_spike_times_to_frames_clamps_out_of_bounds():
    """A spike rounding to ``n_samples`` (one past the end) is CLAMPED to the
    last sample, not dropped.

    Floating-point round-trip at the final sample can land searchsorted at
    ``n_samples``. Dropping it (the prior behavior, and v1's) desyncs the v2
    analyzer's per-spike features from the persisted ``spike_times`` and trips
    ``UnitWaveformFeatures``'s ``n_feat == n_spikes`` guard. Clamping to the
    last valid sample keeps the count aligned (the spike is at ~the final
    sample anyway).
    """
    from spyglass.spikesorting.v2.utils import _spike_times_to_frames

    fs = 30000.0
    timestamps = np.arange(100) / fs + 5.0
    n_samples = timestamps.size
    # A time just past the last timestamp -> searchsorted returns 100.
    spike_times = np.array([timestamps[50], timestamps[-1] + 1.0 / fs])
    frames = _spike_times_to_frames(
        timestamps, spike_times, n_samples, unit_id=7
    )
    assert frames.tolist() == [50, 99], (
        "out-of-bounds spike (frame == n_samples) must be clamped to "
        f"n_samples-1 (count preserved), not dropped; got {frames.tolist()}"
    )
