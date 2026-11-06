"""Multiunit event detection over observed time only.

Runs `ripple_detection.multiunit_HSE_detector`'s steps per contiguous
observed run with one shared normalization, because the detector smears NaN
(unobserved) bins across the smoothed rate.
"""

import warnings

import numpy as np
import pandas as pd
from ripple_detection.core import (
    _validate_normalization_params,
    exclude_close_events,
    exclude_movement,
    get_multiunit_population_firing_rate,
    normalize_signal,
    threshold_by_zscore,
)
from ripple_detection.detectors import _get_event_stats

from spyglass.utils.spikesorting import contiguous_observed_runs

# Same text as ripple_detection's multiunit_HSE_detector warning
SPEED_NORMALIZATION_DEPRECATION = (
    "The 'use_speed_threshold_for_zscore' parameter is deprecated. "
    "Use 'normalization_mask=speed < speed_threshold' instead."
)


def normalize_observed_rate(
    time: np.ndarray,
    multiunit: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: float,
    mask: np.ndarray,
    speed_threshold: float = 4.0,
    smoothing_sigma: float = 0.015,
    use_speed_threshold_for_zscore: bool = False,
    normalization_method: str = "zscore",
    normalization_mask: np.ndarray = None,
    normalization_time_range: tuple = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Smooth the population rate per observed run, normalize it once.

    Normalization uses all observed samples, including runs too short to
    hold an event.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Bin times in seconds.
    multiunit : np.ndarray, shape (n_time, n_units)
        Spike indicator or counts. Unobserved bins are ignored, so they may
        hold NaN.
    speed : np.ndarray, shape (n_time,)
        Speed in cm/s.
    sampling_frequency : float
        Samples per second.
    mask : np.ndarray, shape (n_time,)
        True for observed samples.
    speed_threshold, smoothing_sigma, use_speed_threshold_for_zscore, \
normalization_method, normalization_mask, normalization_time_range
        As in `ripple_detection.multiunit_HSE_detector`, including its
        selector precedence and validation.

    Returns
    -------
    normalized : np.ndarray, shape (n_time,)
        Normalized firing rate, NaN outside the observed runs.
    runs : list of np.ndarray
        The observed runs, as index arrays in time order.

    Raises
    ------
    ValueError
        If an observed bin holds a non-finite firing rate, or if the
        normalization selectors are invalid.
    """
    time = np.asarray(time)
    multiunit = np.asarray(multiunit)
    speed = np.asarray(speed)
    mask = np.asarray(mask, dtype=bool)

    _validate_normalization_params(
        normalization_method,
        normalization_mask,
        normalization_time_range,
        time,
    )

    runs = contiguous_observed_runs(time, mask, sampling_frequency)
    normalized = np.full(time.shape, np.nan)
    if not runs:
        return normalized, runs

    observed = np.concatenate(runs)
    rate = np.full(time.shape, np.nan)
    for run in runs:
        rate[run] = get_multiunit_population_firing_rate(
            multiunit[run], sampling_frequency, smoothing_sigma
        )
    if not np.isfinite(rate[observed]).all():
        raise ValueError(
            "An observed bin produced a non-finite firing rate. The spike "
            "indicator holds NaN inside a run the units were observed over, "
            "so the observed mask and the indicator disagree."
        )

    if use_speed_threshold_for_zscore:
        warnings.warn(
            SPEED_NORMALIZATION_DEPRECATION,
            DeprecationWarning,
            stacklevel=2,
        )
        if normalization_mask is None and normalization_time_range is None:
            normalization_mask = speed < speed_threshold

    if normalization_mask is not None:
        normalization_mask = np.asarray(normalization_mask)
        if normalization_mask.shape[0] == time.shape[0]:
            normalization_mask = normalization_mask[observed]
        # Other lengths pass through so normalize_signal raises

    normalized[observed] = normalize_signal(
        rate[observed],
        time=time[observed],
        method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )

    return normalized, runs


def detect_multiunit_events_in_observed_runs(
    time: np.ndarray,
    multiunit: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: float,
    mask: np.ndarray,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.015,
    close_event_threshold: float = 0.0,
    use_speed_threshold_for_zscore: bool = False,
    normalization_method: str = "zscore",
    normalization_mask: np.ndarray = None,
    normalization_time_range: tuple = None,
) -> pd.DataFrame:
    """Detect high synchrony events within each contiguous observed run.

    - Matches `ripple_detection.multiunit_HSE_detector` when every sample is
      observed and timestamps never jump over 1.5 sample periods.
    - Events never span a gap or timestamp jump; `close_event_threshold`
      merges only within a run.
    - Runs shorter than `minimum_duration` still feed the normalization.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Bin times in seconds.
    multiunit : np.ndarray, shape (n_time, n_units)
        Spike indicator or counts. Unobserved bins may hold NaN.
    speed : np.ndarray, shape (n_time,)
        Speed in cm/s.
    sampling_frequency : float
        Samples per second.
    mask : np.ndarray, shape (n_time,)
        True for observed samples.
    speed_threshold, minimum_duration, zscore_threshold, smoothing_sigma, \
close_event_threshold, use_speed_threshold_for_zscore, \
normalization_method, normalization_mask, normalization_time_range
        As in `ripple_detection.multiunit_HSE_detector`.

    Returns
    -------
    pd.DataFrame
        The detector's event statistics, indexed by a one-based
        chronological `event_number`. Empty with the detector's own columns
        when no run holds an event.
    """
    time = np.asarray(time)
    speed = np.asarray(speed)

    normalized, runs = normalize_observed_rate(
        time,
        multiunit,
        speed,
        sampling_frequency,
        mask,
        speed_threshold=speed_threshold,
        smoothing_sigma=smoothing_sigma,
        use_speed_threshold_for_zscore=use_speed_threshold_for_zscore,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )
    observed = np.concatenate(runs) if runs else np.array([], dtype=int)
    minimum_samples = int(np.ceil(minimum_duration * sampling_frequency))

    event_stats = []
    for run in runs:
        if run.size < minimum_samples:
            continue
        events = threshold_by_zscore(
            normalized[run], time[run], minimum_duration, zscore_threshold
        )
        events = exclude_movement(
            events,
            speed[run],
            time[run],
            speed_threshold=speed_threshold,
        )
        events = exclude_close_events(events, close_event_threshold)
        stats = _get_event_stats(events, time[run], normalized[run], speed[run])
        if len(stats):
            event_stats.append(stats)

    if not event_stats:
        return _get_event_stats(
            [], time[observed], normalized[observed], speed[observed]
        )

    # Per-run tables each restart event_number at 1
    detected = pd.concat(event_stats, ignore_index=True).sort_values(
        "start_time", kind="stable"
    )
    detected.index = pd.RangeIndex(1, len(detected) + 1, name="event_number")

    return detected
