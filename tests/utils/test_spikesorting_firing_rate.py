"""Zero-unit handling in ``firing_rate_from_spike_indicator``."""

import numpy as np
import pytest

from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

pytestmark = pytest.mark.unit


def test_zero_units_returns_empty_columns():
    """Zero units give a (n_time, 0) rate, not an error."""
    time = np.linspace(0.0, 1.0, 200)
    out = firing_rate_from_spike_indicator(np.zeros((200, 0)), time=time)
    assert out.shape == (200, 0)


def test_zero_units_multiunit_returns_zero_rate():
    """Multiunit over zero units is one all-zero column."""
    time = np.linspace(0.0, 1.0, 200)
    out = firing_rate_from_spike_indicator(
        np.zeros((200, 0)), time=time, multiunit=True
    )
    assert out.shape == (200, 1)
    assert np.all(out == 0.0)


def test_nonzero_units_unaffected():
    """One finite rate column per unit."""
    time = np.linspace(0.0, 1.0, 500)
    indicator = np.zeros((500, 2))
    indicator[100, 0] = 1
    indicator[300, 1] = 1
    out = firing_rate_from_spike_indicator(indicator, time=time)
    assert out.shape == (500, 2)
    assert np.all(np.isfinite(out))
    assert (out > 0).any()
