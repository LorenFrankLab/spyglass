"""Hermetic tests for membership-aware merge duplicate-spike removal.

When merging contributor units, a single physical spike double-detected
across two units appears twice in the naive concatenation. A neuron's
refractory period (~1-2 ms) guarantees a single unit never fires twice
within the ~0.4 ms window, so any sub-0.4 ms pair from DIFFERENT
contributors is a double-detection artifact (removed), while a close pair
from the SAME contributor is a genuine event (kept). ``utils.
_dedup_merged_spike_times`` mirrors SpikeInterface's
``get_non_duplicated_events`` (the dedup v1's lazy ``get_merged_sorting``
applied via SI's default ``delta_time_ms=0.4``) so v2's stored
(apply_merge=True) and previewed (get_merged_sorting) merged trains agree
and neither keeps the artifacts.
"""

from __future__ import annotations

import numpy as np


def test_dedup_drops_cross_unit_coincident_spike():
    """A spike within delta from a DIFFERENT contributor is dropped."""
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    delta_s = 0.4e-3
    unit_a = np.array([0.010, 0.020, 0.030])
    # 0.020 + 0.0001 s (0.1 ms < 0.4 ms) is a cross-unit double-detection.
    unit_b = np.array([0.0201, 0.050])
    out = _dedup_merged_spike_times([unit_a, unit_b], delta_s)
    # The 0.0201 duplicate is removed; everything else survives, sorted.
    np.testing.assert_allclose(out, [0.010, 0.020, 0.030, 0.050])


def test_dedup_keeps_within_unit_close_pair():
    """A close pair from the SAME contributor is kept (membership-aware)."""
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    delta_s = 0.4e-3
    # Two spikes 0.1 ms apart but BOTH from unit_a -> not a cross-unit
    # duplicate; a naive time-only dedup would wrongly drop one.
    unit_a = np.array([0.010, 0.0101])
    unit_b = np.array([0.050])
    out = _dedup_merged_spike_times([unit_a, unit_b], delta_s)
    np.testing.assert_allclose(out, [0.010, 0.0101, 0.050])


def test_dedup_no_duplicates_is_concatenate_sort():
    """With no near-coincident cross-unit pairs, it is just concat+sort."""
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    out = _dedup_merged_spike_times(
        [np.array([0.030, 0.010]), np.array([0.020, 0.040])], 0.4e-3
    )
    np.testing.assert_allclose(out, [0.010, 0.020, 0.030, 0.040])


def test_dedup_empty():
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    out = _dedup_merged_spike_times([np.array([]), np.array([])], 0.4e-3)
    assert out.size == 0


def test_dedup_matches_spikeinterface_get_non_duplicated_events():
    """Faithful port: same result as SI on a frame-quantized input.

    Build two contributor trains in FRAME space, run SI's
    ``get_non_duplicated_events`` (frame delta) and our time-space helper
    (seconds delta) at the same fs, and assert the kept frame sets match.
    """
    from spikeinterface.curation.mergeunitssorting import (
        get_non_duplicated_events,
    )

    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    fs = 30000.0
    rm_dup_delta = int(0.4e-3 * fs)  # SI's frame delta (= 12)
    rng = np.random.default_rng(0)
    fa = np.sort(rng.integers(0, 3000, size=40)).astype(np.int64)
    fb = np.sort(rng.integers(0, 3000, size=40)).astype(np.int64)
    si_frames = get_non_duplicated_events([fa, fb], rm_dup_delta)

    # Time-space (seconds) on the same events; delta matched to the frame
    # quantization so the boundary agrees.
    ta, tb = fa / fs, fb / fs
    ours = _dedup_merged_spike_times([ta, tb], rm_dup_delta / fs)
    np.testing.assert_allclose(np.sort(ours), np.sort(si_frames / fs))
