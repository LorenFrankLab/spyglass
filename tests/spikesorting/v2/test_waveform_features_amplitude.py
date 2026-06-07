"""Numerical correctness of ``_get_peak_amplitude`` (clusterless features).

``spyglass.utils.waveforms._get_peak_amplitude`` extracts the per-spike,
per-channel amplitude that clusterless decoding consumes. The existing
clusterless feature tests check shape / 1:1 alignment / sign of the
extracted feature, but never a specific value, and they only ever run the
``estimate_peak_time=False`` (center-sample) path -- the
``estimate_peak_time=True`` branch (argmin/argmax peak finding + mode across
spikes), which holds the only nontrivial numerical logic, is uncovered.

These are hermetic unit tests: a tiny stub supplies a known waveform array
as the *input*, and the assertions pin the *output* of the function under
test. Expected values are derived by independent array indexing (a known,
hand-placed peak), never by calling ``_get_peak_amplitude`` itself, so the
checks are not tautological. No database or SpikeInterface objects needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.utils.waveforms import _get_peak_amplitude

_N_SPIKES, _N_TIME, _N_CHANNELS = 3, 5, 4
_CENTER = _N_TIME // 2  # 2


class _StubWaveformExtractor:
    """Minimal stand-in exposing only ``get_waveforms(unit_idx)``.

    ``_get_peak_amplitude`` calls exactly that one method; the stub returns
    a fixed ``(n_spikes, n_time, n_channels)`` array so the test controls
    the input and asserts the function's computation on it. This is a test
    double for the input, not a mock whose behavior is under test.
    """

    def __init__(self, waveforms: np.ndarray):
        self._waveforms = waveforms

    def get_waveforms(self, unit_idx):  # noqa: D401 - stub
        return self._waveforms


def test_center_sample_returned_when_not_estimating_peak_time():
    """``estimate_peak_time=False`` returns the center time sample
    (``n_time // 2``) for every spike, shape ``(n_spikes, n_channels)``."""
    waveforms = np.arange(
        _N_SPIKES * _N_TIME * _N_CHANNELS, dtype=float
    ).reshape(_N_SPIKES, _N_TIME, _N_CHANNELS)

    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="neg",
        estimate_peak_time=False,
    )

    np.testing.assert_array_equal(out, waveforms[:, _CENTER])
    assert out.shape == (_N_SPIKES, _N_CHANNELS)
    # Not an edge sample -- guards an off-by-n_time-index regression.
    assert not np.array_equal(out, waveforms[:, 0])


def test_negative_peak_time_is_found_off_center():
    """``estimate_peak_time=True, peak_sign='neg'`` locates the most-negative
    sample's time index (here t=1, not the center t=2)."""
    waveforms = np.zeros((_N_SPIKES, _N_TIME, _N_CHANNELS))
    waveforms[:, 1, 2] = -100.0  # negative peak at t=1 on channel 2

    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="neg",
        estimate_peak_time=True,
    )

    np.testing.assert_array_equal(out, waveforms[:, 1])
    assert out[0, 2] == -100.0
    # Distinct from the center slice (all zeros) -> proves the True branch
    # found the real peak time rather than defaulting to the center.
    assert not np.array_equal(out, waveforms[:, _CENTER])


def test_positive_peak_time_is_found_off_center():
    """``peak_sign='pos'`` locates the most-positive sample's time index."""
    waveforms = np.zeros((_N_SPIKES, _N_TIME, _N_CHANNELS))
    waveforms[:, 3, 1] = 200.0  # positive peak at t=3

    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="pos",
        estimate_peak_time=True,
    )

    np.testing.assert_array_equal(out, waveforms[:, 3])


def test_both_peak_sign_uses_absolute_magnitude():
    """``peak_sign='both'`` locates the largest-|amplitude| sample, so a large
    negative deflection wins over smaller positive ones."""
    waveforms = np.zeros((_N_SPIKES, _N_TIME, _N_CHANNELS))
    waveforms[:, 0, 0] = 50.0  # small positive elsewhere
    waveforms[:, 4, 0] = -500.0  # largest magnitude at t=4

    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="both",
        estimate_peak_time=True,
    )

    np.testing.assert_array_equal(out, waveforms[:, 4])


def test_peak_time_uses_mode_across_spikes():
    """A single shared peak-time index (the mode across spikes) is applied to
    every spike, so a spike whose own peak is elsewhere is sampled at the
    shared index.

    Pins the documented behavior: ``_get_peak_amplitude`` collapses per-spike
    peak indices to their mode, then slices all spikes at that one index.
    """
    waveforms = np.zeros((_N_SPIKES, _N_TIME, _N_CHANNELS))
    # spikes 0,1 peak at t=1; spike 2 peaks at t=3 -> mode is t=1.
    waveforms[0, 1, 0] = -100.0
    waveforms[1, 1, 0] = -100.0
    waveforms[2, 3, 0] = -100.0

    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="neg",
        estimate_peak_time=True,
    )

    np.testing.assert_array_equal(out, waveforms[:, 1])
    # Spike 2's true peak (t=3) is missed because the shared mode index (t=1)
    # is used for all spikes -> its sampled value is 0.
    assert out[2, 0] == 0.0


@pytest.mark.parametrize("estimate_peak_time", [False, True])
def test_one_row_per_spike(estimate_peak_time):
    """Output has exactly one amplitude row per input spike (1:1 alignment),
    on both code paths."""
    waveforms = np.random.default_rng(0).normal(
        size=(_N_SPIKES, _N_TIME, _N_CHANNELS)
    )
    out = _get_peak_amplitude(
        _StubWaveformExtractor(waveforms),
        unit_idx=0,
        peak_sign="neg",
        estimate_peak_time=estimate_peak_time,
    )
    assert out.shape == (_N_SPIKES, _N_CHANNELS)
