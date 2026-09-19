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

    observed = np.flatnonzero(mask)
    if observed.size == 0:
        return []

    dt = 1.0 / sampling_frequency
    breaks = (np.diff(observed) > 1) | (  # an unobserved sample between
        np.diff(time[observed]) > 1.5 * dt  # or a jump in the clock
    )

    return np.split(observed, np.flatnonzero(breaks) + 1)


def firing_rate_over_runs(
    spike_indicator: np.ndarray,
    time: np.ndarray,
    runs: list[np.ndarray],
    multiunit: bool = False,
    smoothing_sigma: float = 0.015,
) -> np.ndarray:
    """Smooth spike counts within each contiguous observed run.

    Every sample outside the given runs holds ``np.nan``: it is time the
    units were not observed over, or time the clock skipped, so no rate is
    defined there and none is carried across it. Pass the runs from
    ``contiguous_observed_runs`` so the smoothing splits wherever that
    splitter says a run ends.

    Parameters
    ----------
    spike_indicator : np.ndarray, shape (n_time,) or (n_time, n_units)
        Per-bin spike counts. Samples outside ``runs`` are never read, so
        they may hold ``np.nan``.
    time : np.ndarray, shape (n_time,)
        Bin times, ascending. Only the median sample period is used.
    runs : list of np.ndarray
        One integer index array per contiguous run, in time order.
    multiunit : bool, optional
        If True, sum the units into one population rate, by default False.
    smoothing_sigma : float, optional
        Standard deviation of the Gaussian smoother in seconds, by default
        0.015.

    Returns
    -------
    np.ndarray, shape (n_time, n_units) or (n_time, 1) when ``multiunit``
        Firing rate in spikes/second, ``np.nan`` outside ``runs``.
    """
    spike_indicator = np.asarray(spike_indicator, dtype=float)
    if spike_indicator.ndim == 1:
        spike_indicator = spike_indicator[:, np.newaxis]

    counts = (
        spike_indicator.sum(axis=1, keepdims=True)
        if multiunit
        else spike_indicator
    )
    sampling_frequency = 1 / np.median(np.diff(time))

    firing_rate = np.full(counts.shape, np.nan)
    for run in runs:
        for unit in range(counts.shape[1]):
            firing_rate[run, unit] = get_multiunit_population_firing_rate(
                counts[run, unit, np.newaxis],
                sampling_frequency,
                smoothing_sigma,
            )
    return firing_rate
