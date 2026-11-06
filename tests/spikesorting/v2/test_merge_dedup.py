"""Hermetic tests for membership-aware merge duplicate-spike removal.

A sub-0.4 ms pair from different contributors is one spike detected twice
and is removed; a close pair from one contributor is kept. The stored path
must agree with SpikeInterface's ``get_non_duplicated_events``, which backs
the lazy ``get_merged_sorting``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dedup(times_list, delta_s):
    """Run the production dedup with frames derived from the times."""
    from spyglass.spikesorting.v2._storage.units_nwb import (
        _dedup_merged_spike_times_and_frames,
    )

    frames_list = [
        np.round(np.asarray(t, dtype=float) * 1e5).astype(np.int64)
        for t in times_list
    ]
    times, frames = _dedup_merged_spike_times_and_frames(
        times_list, frames_list, delta_s
    )
    # Each kept frame stays aligned with its kept time.
    np.testing.assert_array_equal(
        frames, np.round(times * 1e5).astype(np.int64)
    )
    return times


def test_dedup_drops_cross_unit_coincident_spike():
    """A spike within delta from a DIFFERENT contributor is dropped."""
    delta_s = 0.4e-3
    unit_a = np.array([0.010, 0.020, 0.030])
    # 0.020 + 0.0001 s (0.1 ms < 0.4 ms) is a cross-unit double-detection.
    unit_b = np.array([0.0201, 0.050])
    out = _dedup([unit_a, unit_b], delta_s)
    np.testing.assert_allclose(out, [0.010, 0.020, 0.030, 0.050])


def test_dedup_keeps_within_unit_close_pair():
    """A close pair from the SAME contributor is kept (membership-aware)."""
    delta_s = 0.4e-3
    # Both from unit_a, so not a cross-unit duplicate.
    unit_a = np.array([0.010, 0.0101])
    unit_b = np.array([0.050])
    out = _dedup([unit_a, unit_b], delta_s)
    np.testing.assert_allclose(out, [0.010, 0.0101, 0.050])


def test_dedup_no_duplicates_is_concatenate_sort():
    """With no near-coincident cross-unit pairs, it is just concat+sort."""
    out = _dedup([np.array([0.030, 0.010]), np.array([0.020, 0.040])], 0.4e-3)
    np.testing.assert_allclose(out, [0.010, 0.020, 0.030, 0.040])


def test_dedup_empty():
    out = _dedup([np.array([]), np.array([])], 0.4e-3)
    assert out.size == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dedup_matches_spikeinterface(seed):
    """The stored-path dedup equals SpikeInterface's on random trains."""
    from spikeinterface.curation.mergeunitssorting import (
        get_non_duplicated_events,
    )

    rng = np.random.default_rng(seed)
    trains = [np.sort(rng.uniform(0.0, 1.0, size=200)) for _ in range(3)]
    np.testing.assert_array_equal(
        _dedup(trains, 0.4e-3), get_non_duplicated_events(trains, 0.4e-3)
    )
