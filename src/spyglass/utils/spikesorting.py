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
    if spike_indicator.shape[1] == 0:
        # Zero-unit group (e.g. a v2 require_units=False curation, or all
        # units label-filtered out): there is nothing to stack, so return an
        # empty-but-shaped rate instead of letting np.stack([]) raise.
        return np.zeros((spike_indicator.shape[0], 0))
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


def contiguous_observed_runs(
    time: np.ndarray,
    mask: np.ndarray,
    sampling_frequency: float,
) -> list[np.ndarray]:
    """Split observed samples into maximal contiguous runs.

    A run ends where the next sample is unobserved or where the timestamps
    jump by more than 1.5 sample periods, so no run spans time the units were
    not observed over. Smoothing or event detection applied per run therefore
    never carries information across a gap.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Bin times, ascending.
    mask : np.ndarray, shape (n_time,)
        True for observed samples.
    sampling_frequency : float
        Samples per second, used for the gap tolerance (1.5 / fs).

    Returns
    -------
    list of np.ndarray
        One integer index array per run, in time order. Empty when no sample
        is observed.
    """
    time = np.asarray(time)
    mask = np.asarray(mask, dtype=bool)

    if time.size == 0 or not mask.any():
        return []

    dt = 1.0 / sampling_frequency
    is_run_start = np.r_[
        True,
        (~mask[:-1]) | (~mask[1:]) | (np.diff(time) > 1.5 * dt),
    ]
    run_ids = np.cumsum(is_run_start)

    return [
        np.flatnonzero(mask & (run_ids == run_id))
        for run_id in np.unique(run_ids[mask])
    ]
