"""``firing_rate_from_spike_indicator``: Hz scaling and zero-unit handling.

The function converts a spike-indicator matrix (counts per time bin) into a
firing rate in **spikes per second**: it multiplies the per-bin counts by the
sampling frequency (recovered as ``1/median(diff(time))``) and convolves with a
normalized Gaussian. Two properties matter scientifically and are pinned here:

* **Hz scaling.** The time-integral of the rate equals the spike count, and the
  scale is derived from the time grid -- a rate accidentally returned in
  counts/bin (missing the ``* sampling_frequency``) would be off by a factor of
  ``fs`` and would track the bin width instead of staying invariant to it.
* **Zero-unit handling.** v2 supports zero-unit curations
  (``require_units=False``); they flow into ``SpikeSortingOutput.get_firing_rate``
  / ``SortedSpikesGroup.get_firing_rate`` through this function, which did
  ``np.stack([], axis=1)`` -> ``ValueError`` on an empty (0-column) indicator. A
  zero-unit group should return an empty-but-shaped rate, not crash.

Hermetic -- pure function, no DB.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

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

    The per-bin count is scaled by ``fs`` (spikes/second) then convolved with a
    normalized Gaussian (scipy ``gaussian_filter1d``, ``mode="constant"``). A
    normalized kernel conserves total mass for spikes whose +/-8 sigma support
    stays inside the array, so ``sum(rate) * dt == spike_count`` to machine
    precision. The isolated-spike peak is ``1/(sigma*sqrt(2*pi))`` Hz
    (``fs`` cancels), a second, magnitude-level check on the Hz scaling.
    """
    fs = 1000.0
    n = 1000
    time = np.arange(n) / fs  # uniform grid -> 1/median(diff(time)) == fs
    dt = 1.0 / fs
    sigma = 0.015
    # Spikes well inside [8*sigma*fs, n - 8*sigma*fs] == [120, 880], so the
    # Gaussian support never spills off either end and mass is conserved. They
    # are >12 sigma apart, so each peak is an isolated single-spike peak.
    spike_idx = [300, 500, 700]
    ind = np.zeros((n, 1))
    ind[spike_idx, 0] = 1.0

    rate = firing_rate_from_spike_indicator(
        ind, time=time, smoothing_sigma=sigma
    )

    assert rate.shape == (n, 1)
    integral = rate[:, 0].sum() * dt
    assert integral == pytest.approx(len(spike_idx), rel=1e-6)
    # Magnitude is a true rate in Hz (~26.6 Hz isolated peak), not counts/bin.
    expected_peak_hz = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    assert rate.max() == pytest.approx(expected_peak_hz, rel=1e-2)
    assert rate.max() < fs


def test_firing_rate_per_unit_and_multiunit_linearity():
    """Per-unit columns integrate to each unit's spike count, and the
    multiunit rate equals the summed per-unit columns.

    ``multiunit=True`` sums the indicators across units *then* smooths; the
    per-unit path smooths each column. Because the Gaussian filter is linear,
    summing-before equals summing-after to machine precision -- a regression
    that summed counts on the wrong axis, or double-counted, would break this.
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

    A counts-per-bin bug (missing the ``* sampling_frequency``) would make the
    integral scale with ``dt`` and differ between the two sampling rates.
    """
    n_spikes = 4
    n = int(fs)  # one second of data
    time = np.arange(n) / fs
    ind = np.zeros((n, 1))
    ind[np.linspace(n * 0.3, n * 0.7, n_spikes).astype(int), 0] = 1.0

    rate = firing_rate_from_spike_indicator(ind, time=time)

    assert rate[:, 0].sum() / fs == pytest.approx(n_spikes, rel=1e-6)
