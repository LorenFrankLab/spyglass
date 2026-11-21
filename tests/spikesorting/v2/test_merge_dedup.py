"""Hermetic tests for membership-aware merge duplicate-spike removal.

A spike within 0.4 ms after the last KEPT spike of a different contributor
is one spike detected twice and is removed; a close pair from one contributor
is kept. The stored and lazy merge paths share this rule.
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


def _last_kept_reference(times_list, delta_s):
    """Naive loop: drop a spike only against the last kept spike."""
    kept: list[tuple[float, int]] = []
    events = sorted(
        (float(t), unit) for unit, train in enumerate(times_list) for t in train
    )
    for t, unit in events:
        if kept and unit != kept[-1][1] and t - kept[-1][0] <= delta_s:
            continue
        kept.append((t, unit))
    return np.array([t for t, _ in kept])


def test_dedup_compares_to_the_last_kept_spike():
    """A dropped duplicate does not shadow the next genuine spike."""
    delta_s = 0.4e-3
    unit_a = np.array([1.0, 1.0006])
    unit_b = np.array([1.0003])
    out = _dedup([unit_a, unit_b], delta_s)
    np.testing.assert_allclose(out, [1.0, 1.0006])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n_units", [2, 3])
def test_dedup_matches_last_kept_reference(seed, n_units):
    """Random dense trains equal the naive last-kept loop."""
    rng = np.random.default_rng(seed)
    trains = [np.sort(rng.uniform(0.0, 0.5, size=400)) for _ in range(n_units)]
    np.testing.assert_array_equal(
        _dedup(trains, 0.4e-3), _last_kept_reference(trains, 0.4e-3)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dedup_matches_si_keep_first_iterative_for_refractory_units(seed):
    """Without within-unit close pairs the rule is SI's keep_first_iterative."""
    from spikeinterface.curation.curation_tools import find_duplicated_spikes

    rng = np.random.default_rng(seed)
    fs = 1e5
    censor = 40  # samples; delta sits between grid points to avoid ties
    delta_s = 40.5 / fs
    trains = []
    for _ in range(3):
        isi = rng.integers(censor + 1, 400, size=300)
        trains.append(np.cumsum(isi) / fs)
    frames = np.sort(
        np.concatenate([np.round(t * fs).astype(np.int64) for t in trains])
    )
    duplicates = find_duplicated_spikes(
        frames, censor, method="keep_first_iterative"
    )
    expected = np.delete(frames, duplicates) / fs
    np.testing.assert_allclose(_dedup(trains, delta_s), expected)
