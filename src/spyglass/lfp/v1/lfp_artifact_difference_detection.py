import warnings
from typing import Union

import numpy as np
import scipy.signal

from spyglass.common.common_interval import Interval
from spyglass.utils import logger
from spyglass.utils.nwb_helper_fn import get_valid_intervals


def difference_artifact_detector(
    recording: None,
    timestamps: None,
    amplitude_thresh_1st: Union[float, None] = None,
    amplitude_thresh_2nd: Union[float, None] = None,
    proportion_above_thresh_1st: float = 1.0,
    proportion_above_thresh_2nd: float = 1.0,
    removal_window_ms: float = 1.0,
    local_window_ms: float = 1.0,
    sampling_frequency: float = 1000.0,
    referencing: bool = False,
):
    """Detects times during which artifacts do and do not occur.

    Artifacts are defined as periods where the absolute value of the change in
    LFP exceeds amplitude change thresholds on the proportion of channels
    specified, with the period extended by the removal_window_ms/2 on each side.
    amplitude change threshold values of None are ignored.

    Parameters
    ----------
    recording : lfp eseries zscore_thresh : float, optional
        Stdev threshold for exclusion, should be >=0, defaults to None
    amplitude_thresh_1st : float, optional
        Amplitude (ad units) threshold for exclusion, should be >=0, defaults to
        None
    amplitude_thresh_2nd : float, optional
        Amplitude (ad units) threshold for exclusion, should be >=0, defaults to
        None
    proportion_above_thresh_1st : float, optional, should be>0 and <=1
        Proportion of electrodes that need to have threshold crossings, defaults
        to 1
    proportion_above_thresh_2nd : float, optional, should be>0 and <=1
        Proportion of electrodes that need to have threshold crossings, defaults
        to 1
    removal_window_ms : float, optional
        Width of the window in milliseconds to mask out per artifact (window/2
        removed on each side of threshold crossing), defaults to 1 ms
    referencing : bool, optional
        Whether or not the data passed to this function is referenced, defaults
        to False

    Returns
    -------
    artifact_removed_valid_times : np.ndarray
        Intervals of valid times where artifacts were not detected, unit:
        seconds
    artifact_intervals : np.ndarray
        Intervals in which artifacts are detected (including removal windows),
        unit: seconds
    """

    # NOTE: 7-17-23 updated to remove recording.data, since it will converted to
    # numpy array before referencing check for referencing flag

    if referencing:
        logger.info("referencing activated. may be set to -1")

    # valid_timestamps = recording.timestamps
    valid_timestamps = timestamps

    local_window = int(local_window_ms / 2)

    # if both thresholds are None, we skip artifact detection
    if amplitude_thresh_1st is None:
        recording_interval = np.asarray(
            [valid_timestamps[0], valid_timestamps[-1]]
        )
        artifact_times_empty = np.asarray([])
        logger.info("Amplitude threshold is None, skipping artifact detection")
        return recording_interval, artifact_times_empty

    # verify threshold parameters
    (
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    ) = _check_artifact_thresholds(
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    )

    # want to detect frames without parallel processing
    # compute the number of electrodes that have to be above threshold
    nelect_above_1st = np.ceil(proportion_above_thresh_1st * recording.shape[1])
    nelect_above_2nd = np.ceil(proportion_above_thresh_2nd * recording.shape[1])
    logger.info("num tets 1", nelect_above_1st, "num tets 2", nelect_above_2nd)
    logger.info("data shape", recording.shape)

    # find the artifact occurrences using one or both thresholds, across
    # channels

    if amplitude_thresh_1st is not None:
        # first find times with large amp change: sum diff over several timebins

        diff_array = np.diff(recording, axis=0)
        window = np.ones((3, 1)) if referencing else np.ones((15, 1))

        # sum differences over bins using convolution for speed
        width = int((window.size - 1) / 2)
        diff_array = np.pad(
            diff_array,
            pad_width=((width, width), (0, 0)),
            mode="constant",
        )
        diff_array_5 = scipy.signal.convolve(diff_array, window, mode="valid")

        artifact_times_all_5 = np.sum(
            (np.abs(diff_array_5) > amplitude_thresh_1st), axis=1
        )

        above_thresh_1st = np.where(artifact_times_all_5 >= nelect_above_1st)[0]

        # second, find artifacts with large baseline change
        logger.info("thresh", amplitude_thresh_2nd, "window", local_window)

        big_artifacts = np.zeros(
            (recording.shape[1], above_thresh_1st.shape[0])
        )
        for art_count in np.arange(above_thresh_1st.shape[0]):
            if above_thresh_1st[art_count] <= local_window:
                local_min = local_max = above_thresh_1st[art_count]
            else:
                local_max = np.max(
                    recording[
                        above_thresh_1st[art_count]
                        - local_window : above_thresh_1st[art_count]
                        + local_window,
                        :,
                    ],
                    axis=0,
                )
                local_min = np.min(
                    recording[
                        above_thresh_1st[art_count]
                        - local_window : above_thresh_1st[art_count]
                        + local_window,
                        :,
                    ],
                    axis=0,
                )
            big_artifacts[:, art_count] = (
                np.abs(local_max - local_min) > amplitude_thresh_2nd
            )

        # sum columns in big artficat, then compare to nelect_above_2nd
        above_thresh = above_thresh_1st[
            np.sum(big_artifacts, axis=0) >= nelect_above_2nd
        ]

    artifact_frames = above_thresh.copy()
    logger.info("detected ", artifact_frames.shape[0], " artifacts")

    if len(artifact_frames) == 0:
        recording_interval = np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        )
        artifact_times_empty = np.asarray([])
        logger.info("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # generate intervals for artifact frame indices
    artifact_intervals_s = Interval(
        artifact_frames, from_inds=True
    ).add_removal_window(
        removal_window_ms=removal_window_ms, timestamps=valid_timestamps
    )

    valid_intervals = get_valid_intervals(
        valid_timestamps, sampling_frequency, 1.5, 0.000001
    )

    # these are artifact times - need to subtract these from valid timestamps
    artifact_valid_times = Interval(valid_intervals).intersect(
        artifact_intervals_s
    )

    starts = np.searchsorted(valid_timestamps, artifact_valid_times[:, 0])
    ends = np.searchsorted(valid_timestamps, artifact_valid_times[:, 1])
    list_triggers = [np.arange(start, end) for start, end in zip(starts, ends)]

    new_array = np.array(np.concatenate(list_triggers))

    new_timestamps = np.delete(valid_timestamps, new_array)

    artifact_removed_valid_times = get_valid_intervals(
        new_timestamps, sampling_frequency, 1.5, 0.000001
    )

    return artifact_removed_valid_times, artifact_intervals_s


def _check_artifact_thresholds(
    amplitude_thresh_1st,
    amplitude_thresh_2nd,
    proportion_above_thresh_1st,
    proportion_above_thresh_2nd,
):
    """Alerts user to likely unintended parameters. Not exhaustive verification.

    Parameters
    ----------
        amplitude_thresh_1st: float
        amplitude_thresh_2nd: float
        proportion_above_thresh_1st: float
        proportion_above_thresh_2nd: float

    Return
    ------
        amplitude_thresh_1st: float
        amplitude_thresh_2nd: float
        proportion_above_thresh_1st: float
        proportion_above_thresh_2nd: float

    Raise
    ------
    ValueError: if signal thresholds are negative
    """
    signal_thresholds = [  # amplitude or zscore thresh should be not negative
        t for t in [amplitude_thresh_1st, amplitude_thresh_1st] if t is not None
    ]

    for t in signal_thresholds:
        if t < 0:
            raise ValueError(
                f"Amplitude thresholds must be >= 0, or None. Received {t}"
            )

    bound_warn = (
        "Warning: proportion_above_thresh must be a proportion >0 and <=1.\n"
        + "Replacing {} with {}"
    )

    def clamp(n, min_n=0.01, max_n=1):  # replace n outside bounds with bound
        if n < 0:
            warnings.warn(bound_warn.format(n, min_n))
        elif n < min_n:  # handle case where n is btwn 0 and low bound
            return n
        elif n > max_n:
            warnings.warn(bound_warn.format(n, max_n))
        return max(min(n, max_n), min_n)

    proportion_above_thresh_1st = clamp(proportion_above_thresh_1st)
    proportion_above_thresh_2nd = clamp(proportion_above_thresh_2nd)

    return (
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    )
