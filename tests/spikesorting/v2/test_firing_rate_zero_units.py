"""``firing_rate_from_spike_indicator`` handles a zero-unit group.

v2 supports zero-unit curations (``require_units=False``); they flow into
``SpikeSortingOutput.get_firing_rate`` / ``SortedSpikesGroup.get_firing_rate``
through the shared ``firing_rate_from_spike_indicator``, which did
``np.stack([], axis=1)`` -> ``ValueError`` on an empty (0-column) indicator.
A zero-unit group should return an empty-but-shaped rate, not crash.

Hermetic -- pure function, no DB.
"""

from __future__ import annotations

import numpy as np

from spyglass.utils.spikesorting import firing_rate_from_spike_indicator


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
