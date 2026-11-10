"""Spike-indicator firing rates: Hz scaling and zero-unit handling.

* The rate is in spikes/s: its time integral equals the spike count,
  independent of bin width.
* Zero-unit curations (``require_units=False``) return a ``(n_time, 0)``
  rate instead of raising.

DB-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting._observed_time import ObservationAvailability
from spyglass.utils.spikesorting import (
    firing_rate_from_spike_indicator,
    firing_rate_over_observed_runs,
    observed_spike_indicator,
)

pytestmark = pytest.mark.unit


def test_zero_units_returns_empty_columns():
    """0 units -> (n_time, 0), not a crash."""
    time = np.linspace(0.0, 1.0, 200)
    out = firing_rate_from_spike_indicator(np.zeros((200, 0)), time=time)
    assert out.shape == (200, 0)


def test_zero_units_multiunit_returns_zero_rate():
    """multiunit over 0 units is a single all-zero rate column."""
    time = np.linspace(0.0, 1.0, 200)
    out = firing_rate_from_spike_indicator(
        np.zeros((200, 0)), time=time, multiunit=True
    )
    assert out.shape == (200, 1)
    assert np.all(out == 0.0)


@pytest.mark.parametrize(
    "intervals", [None, [[0.0, 0.4], [0.6, 1.0]]], ids=["unknown", "gapped"]
)
def test_zero_units_observed_indicator_and_rate(intervals):
    """The merge-table / group path keeps the empty shape over observed runs:
    0 units -> a (n_time, 0) indicator and rate, and a multiunit rate that is
    zero inside the observed runs and NaN outside them."""
    time = np.linspace(0.0, 1.0, 200)
    observation = ObservationAvailability(intervals)

    indicator, valid = observed_spike_indicator([], time, observation)
    assert indicator.shape == (200, 0)
    np.testing.assert_array_equal(valid, observation.valid_bins(time))

    rate = firing_rate_over_observed_runs(indicator, time, valid)
    assert rate.shape == (200, 0)

    mua = firing_rate_over_observed_runs(indicator, time, valid, multiunit=True)
    assert mua.shape == (200, 1)
    assert np.all(mua[valid, 0] == 0.0)
    assert np.all(np.isnan(mua[~valid, 0]))
    if intervals is not None:
        assert (~valid).any()


def test_nonzero_units_unaffected():
    """The non-empty path is unchanged: one finite rate column per unit."""
    time = np.linspace(0.0, 1.0, 500)
    ind = np.zeros((500, 2))
    ind[100, 0] = 1
    ind[300, 1] = 1
    out = firing_rate_from_spike_indicator(ind, time=time)
    assert out.shape == (500, 2)
    assert np.all(np.isfinite(out))
    assert (out > 0).any()


def test_firing_rate_integral_equals_spike_count():
    """The time-integral of the rate equals the number of spikes, in Hz.

    A normalized kernel conserves mass when its support stays in the array.
    An isolated spike peaks at ``1/(sigma*sqrt(2*pi))`` Hz.
    """
    fs = 1000.0
    n = 1000
    time = np.arange(n) / fs  # uniform grid -> 1/median(diff(time)) == fs
    dt = 1.0 / fs
    sigma = 0.015
    # Inside [120, 880] (8 sigma margin) and >12 sigma apart.
    spike_idx = [300, 500, 700]
    ind = np.zeros((n, 1))
    ind[spike_idx, 0] = 1.0

    rate = firing_rate_from_spike_indicator(
        ind, time=time, smoothing_sigma=sigma
    )

    assert rate.shape == (n, 1)
    integral = rate[:, 0].sum() * dt
    assert integral == pytest.approx(len(spike_idx), rel=1e-6)
    # ~26.6 Hz isolated peak.
    expected_peak_hz = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    assert rate.max() == pytest.approx(expected_peak_hz, rel=1e-2)
    assert rate.max() < fs


def test_firing_rate_per_unit_and_multiunit_linearity():
    """Per-unit columns integrate to each unit's spike count, and the
    multiunit rate equals the summed per-unit columns.

    The Gaussian filter is linear, so summing before or after smoothing
    agree.
    """
    fs = 1000.0
    n = 1000
    time = np.arange(n) / fs
    dt = 1.0 / fs
    ind = np.zeros((n, 2))
    ind[[250, 450, 650], 0] = 1.0  # unit 0: 3 spikes
    ind[[400, 600], 1] = 1.0  # unit 1: 2 spikes

    per_unit = firing_rate_from_spike_indicator(ind, time=time)
    assert per_unit.shape == (n, 2)
    assert per_unit[:, 0].sum() * dt == pytest.approx(3, rel=1e-6)
    assert per_unit[:, 1].sum() * dt == pytest.approx(2, rel=1e-6)

    mua = firing_rate_from_spike_indicator(ind, time=time, multiunit=True)
    assert mua.shape == (n, 1)
    np.testing.assert_allclose(
        mua[:, 0], per_unit.sum(axis=1), rtol=1e-9, atol=1e-12
    )
    assert mua[:, 0].sum() * dt == pytest.approx(5, rel=1e-6)


@pytest.mark.parametrize("fs", [1000.0, 2000.0])
def test_firing_rate_scale_tracks_time_grid(fs):
    """The Hz scale comes from the time grid, so the same spike pattern over
    one second integrates to the same spike count regardless of bin width.
    """
    n_spikes = 4
    n = int(fs)  # one second of data
    time = np.arange(n) / fs
    ind = np.zeros((n, 1))
    ind[np.linspace(n * 0.3, n * 0.7, n_spikes).astype(int), 0] = 1.0

    rate = firing_rate_from_spike_indicator(ind, time=time)

    assert rate[:, 0].sum() / fs == pytest.approx(n_spikes, rel=1e-6)
