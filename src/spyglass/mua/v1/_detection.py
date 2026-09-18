"""Multiunit event detection over observed time only.

`ripple_detection.multiunit_HSE_detector` assumes every sample it is given
is contiguous evidence. A group whose units were not observed for the whole
time axis carries NaN in those bins, and handing that straight to the
detector corrupts the result twice over: `gaussian_smooth` (truncate=8)
spreads each NaN bin over +/- 8 sigma of the smoothed rate, and the
z-score's `nan_policy="omit"` then renormalizes over whatever survived. In
a reproduction, a 380 ms unobserved interval erased both bursts beside it
and two background fluctuations were reported as events in their place.

The functions here run the detector's own steps over each contiguous
observed run instead, sharing one normalization across all observed
samples, so unobserved time removes no real event and invents none. They
hold no DataJoint tables, so the logic is unit testable without a database.
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

# The detector's own text, re-emitted so the deprecated parameter behaves
# identically here (ripple_detection.detectors.multiunit_HSE_detector).
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

    The Gaussian kernel is applied to each contiguous observed run on its
    own, so smoothed values never mix spikes from either side of a gap. The
    normalization statistics then come from all observed samples at once,
    including runs too short to hold an event: they carry baseline
    information even when no event can be extracted from them.

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
        As in `ripple_detection.multiunit_HSE_detector`. Selector precedence
        and validation stay the detector's own: a user mask or time range
        wins over the deprecated speed threshold, and supplying both
        selectors raises.

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
        # A mask of any other length is passed through unsliced so that
        # normalize_signal raises its own length error.

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

    Reproduces `ripple_detection.multiunit_HSE_detector` exactly when every
    sample is observed. Otherwise events are extracted from each observed
    run separately, so no event can span time the units were not observed
    over. Runs shorter than `minimum_duration` cannot hold an event and are
    skipped at extraction, after they have contributed to the shared
    normalization. `close_event_threshold` merges events within a run and
    never across a gap: two events on either side of unobserved time stay
    two events however close their timestamps are.

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

    # Each run's table restarts its one-based event_number, so two runs
    # would carry duplicate indices into the NWB table. Renumber
    # chronologically after concatenation.
    detected = pd.concat(event_stats, ignore_index=True).sort_values(
        "start_time", kind="stable"
    )
    detected.index = pd.RangeIndex(1, len(detected) + 1, name="event_number")

    return detected
