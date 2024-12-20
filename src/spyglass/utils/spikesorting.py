import numpy as np
from ripple_detection import get_multiunit_population_firing_rate


def firing_rate_from_spike_indicator(
    spike_indicator: np.ndarray,
    time: np.array,
    multiunit: bool = False,
    smoothing_sigma: float = 0.015,
):
    """Calculate firing rate from spike indicator."""
    if spike_indicator.ndim == 1:
        spike_indicator = spike_indicator[:, np.newaxis]

    sampling_frequency = 1 / np.median(np.diff(time))

    if multiunit:
        spike_indicator = spike_indicator.sum(axis=1, keepdims=True)
    return np.stack(
        [
            get_multiunit_population_firing_rate(
                indicator[:, np.newaxis],
                sampling_frequency,
                smoothing_sigma,
            )
            for indicator in spike_indicator.T
        ],
        axis=1,
    )
