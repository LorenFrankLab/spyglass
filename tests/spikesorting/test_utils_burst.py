"""Database-free checks for burst-pair helpers."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def mini_insert():
    """These tests need no sample-data ingestion."""
    yield


def _peak_amps(n_channels, n_spikes=10, seed=0):
    rng = np.random.default_rng(seed)
    return {unit: rng.normal(size=(n_spikes, n_channels)) for unit in (1, 2, 3)}


@pytest.mark.parametrize(
    "pairs, n_channels",
    [
        ([(1, 2)], 4),  # single pair
        ([(1, 2), (2, 3)], 6),  # more channels than columns
        ([(1, 2)], 2),  # fewer channels than columns
    ],
)
def test_plot_burst_pair_peaks(pairs, n_channels):
    import matplotlib.pyplot as plt

    from spyglass.spikesorting.utils_burst import plot_burst_pair_peaks

    fig = plot_burst_pair_peaks(pairs, _peak_amps(n_channels))
    try:
        axes = np.asarray(fig.axes).reshape(len(pairs), 4)
        n_plotted = min(n_channels, 4)
        for row in axes:
            titles = [ax.get_title() for ax in row]
            assert titles[:n_plotted] == [
                f"channel {i + 1}" for i in range(n_plotted)
            ]
            assert all(not t for t in titles[n_plotted:])
    finally:
        plt.close(fig)


def test_isi_violation_threshold_is_milliseconds():
    from spyglass.spikesorting.utils_burst import calculate_isi_violation

    # Merged train ISIs: 1 ms, 2 ms, 10 ms over 4 spikes.
    peak1 = np.array([0.0, 0.003])
    peak2 = np.array([0.001, 0.013])

    assert calculate_isi_violation(peak1, peak2) == 1 / 4
    assert calculate_isi_violation(peak1, peak2, isi_threshold_ms=2.5) == 2 / 4


def test_isi_violation_deprecated_seconds_name_warns():
    from spyglass.spikesorting.utils_burst import calculate_isi_violation

    peak1 = np.array([0.0, 0.003])
    peak2 = np.array([0.001, 0.013])

    with pytest.warns(DeprecationWarning, match="isi_threshold_ms"):
        result = calculate_isi_violation(peak1, peak2, isi_threshold_s=2.5)
    assert result == 2 / 4
