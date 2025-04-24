import numpy as np
from scipy.ndimage import find_objects, label
from scipy.stats import median_abs_deviation


def mad_artifact_detector(
    recording: None,
    mad_thresh: float = 6.0,
    proportion_above_thresh: float = 0.1,
    removal_window_ms: float = 10.0,
    sampling_frequency: float = 1000.0,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect LFP artifacts using the median absolute deviation method.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    mad_thresh : float, optional
        Threshold on the median absolute deviation scaled LFPs, defaults to 6.0
    proportion_above_thresh : float, optional
        Proportion of electrodes that need to be above the threshold, defaults
        to 1.0
    removal_window_ms : float, optional
        Width of the window in milliseconds to mask out per artifact
        (window/2 removed on each side of threshold crossing), defaults to 1 ms
    sampling_frequency : float, optional
        Sampling frequency of the recording extractor, defaults to 1000.0

    Returns
    -------
    artifact_removed_valid_times : np.ndarray
        Intervals of valid times where artifacts were not detected,
        unit: seconds
    artifact_intervals : np.ndarray
        Intervals in which artifacts are detected (including removal windows),
        unit: seconds
    """

    timestamps = np.asarray(recording.timestamps)
    lfps = np.asarray(recording.data)

    mad = median_abs_deviation(lfps, axis=0, nan_policy="omit", scale="normal")
    mad = np.where(np.isclose(mad, 0.0) | ~np.isfinite(mad), 1.0, mad)
    is_artifact = _is_above_proportion_thresh(
        _mad_scale_lfps(lfps, mad), mad_thresh, proportion_above_thresh
    )

    MILLISECONDS_PER_SECOND = 1000.0
    half_removal_window_s = (removal_window_ms / MILLISECONDS_PER_SECOND) * 0.5
    half_removal_window_idx = int(half_removal_window_s * sampling_frequency)
    is_artifact = _extend_array_by_window(is_artifact, half_removal_window_idx)

    artifact_intervals_s = np.array(
        _get_time_intervals_from_bool_array(is_artifact, timestamps)
    )

    valid_times = np.array(
        _get_time_intervals_from_bool_array(~is_artifact, timestamps)
    )

    return valid_times, artifact_intervals_s


def _mad_scale_lfps(lfps: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Scale LFPs by median absolute deviation.

    Parameters
    ----------
    lfps : np.ndarray, shape (n_samples, n_electrodes)
        LFPs to scale
    mad : np.ndarray, shape (n_electrodes,)
        Median absolute deviation of LFPs

    Returns
    -------
    mad_scaled_lfps : np.ndarray, shape (n_samples, n_electrodes)

    """
    return np.abs(lfps - np.nanmedian(lfps, axis=0)) / mad


def _is_above_proportion_thresh(
    mad_scaled_lfps: np.ndarray,
    mad_thresh: np.ndarray,
    proportion_above_thresh: float = 1.0,
) -> np.ndarray:
    """Return array of bools for samples > thresh on proportion of electodes.

    Parameters
    ----------
    mad_scaled_lfps : np.ndarray, shape (n_samples, n_electrodes)
        LFPs scaled by median absolute deviation
    mad_thresh : float
        Threshold on the median absolute deviation scaled LFPs
    proportion_above_thresh : float
        Proportion of electrodes that need to be above the threshold

    Returns
    -------
    is_above_thresh : np.ndarray, shape (n_samples,)
        Whether each sample is above the threshold on the proportion of
        electrodes
    """
    n_electrodes = mad_scaled_lfps.shape[1]
    thresholded_count = np.sum(mad_scaled_lfps > mad_thresh, axis=1)
    return thresholded_count > (proportion_above_thresh * n_electrodes)


def _get_time_intervals_from_bool_array(
    bool_array: np.ndarray, timestamps: np.ndarray
) -> list[list[float]]:
    """Return time intervals from a boolean array.

    Parameters
    ----------
    bool_array : np.ndarray, shape (n_samples,)
        Boolean array
    timestamps : np.ndarray, shape (n_samples,)
        Timestamps corresponding to the boolean array

    Returns
    -------
    time_intervals : list[list[float]]
        Time intervals corresponding to the boolean array
    """
    labels, n_labels = label(bool_array)

    try:
        return [
            timestamps[labels == label_id][[0, -1]]
            for label_id in range(1, n_labels + 1)
        ]
    except IndexError:
        return []


def _extend_array_by_window(
    bool_array: np.ndarray, window_size: int = 0
) -> np.ndarray:
    """Extend a boolean array by a window size on each side.

    Parameters
    ----------
    bool_array : np.ndarray, shape (n_samples,)
        Boolean array to extend
    window_size : int, optional
        Window size to extend by on each side, defaults to 0

    Returns
    -------
    new_bool_array : np.ndarray, shape (n_samples,)
        Boolean array extended by the window size on each side
    """
    labels, _ = label(bool_array)

    n_samples = len(bool_array)

    new_bool_array = np.zeros_like(bool_array)

    for loc in find_objects(labels):
        start_ind, stop_ind = loc[0].start, loc[0].stop

        # extend the interval by window_size on each side
        start_ind -= window_size
        stop_ind += window_size

        # make sure the interval is within the array
        start_ind = 0 if start_ind < 0 else start_ind
        stop_ind = n_samples if stop_ind >= n_samples else stop_ind

        new_bool_array[start_ind:stop_ind] = True

    return new_bool_array
