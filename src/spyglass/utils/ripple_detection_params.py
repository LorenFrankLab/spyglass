"""Apply stored ripple_detection parameter dicts with ripple_detection 2.0.

`RippleParameters` and `MuaEventsParameters` store keyword arguments for
ripple_detection's detectors. ripple_detection 2.0 removed two keywords that
1.x accepted, so a row stored for 1.x would fail at populate. This module
translates them into the 2.0 keyword with the same meaning, so existing rows
keep working without editing the database.
"""

import numpy as np

from spyglass.utils.logging import logger


def detection_kwargs_from_params(
    params: dict, time: np.ndarray, speed: np.ndarray
) -> dict:
    """Return detector keyword arguments for a stored parameter dict.

    Keys that ripple_detection 2.0 still accepts pass through unchanged. The
    two keys 2.0 removed become a ``normalization_mask`` with the 1.x meaning:

    - ``normalization_time_range=(start, end)`` becomes
      ``normalization_mask=(time >= start) & (time <= end)``.
    - ``use_speed_threshold_for_zscore=True`` (multiunit HSE detector) becomes
      ``normalization_mask=speed < speed_threshold``, the strict comparison
      1.x used, with the dict's ``speed_threshold`` or the detector default of
      4.0 cm/s. As in 1.x, a time range takes precedence over the flag.
      ``False`` is dropped.

    Parameters
    ----------
    params : dict
        Stored keyword arguments for the detector.
    time : np.ndarray, shape (n_time,)
        Sample times in seconds, as passed to the detector.
    speed : np.ndarray, shape (n_time,)
        Speed in cm/s, as passed to the detector.

    Returns
    -------
    kwargs : dict
        Keyword arguments accepted by ripple_detection 2.0. ``params`` is not
        modified.

    Raises
    ------
    ValueError
        If ``params`` holds both ``normalization_mask`` and
        ``normalization_time_range``, which 1.x also rejected.
    """
    kwargs = dict(params)
    time_range = kwargs.pop("normalization_time_range", None)
    use_speed = kwargs.pop("use_speed_threshold_for_zscore", False)

    if time_range is not None:
        if kwargs.get("normalization_mask") is not None:
            raise ValueError(
                "Cannot specify both 'normalization_mask' and "
                "'normalization_time_range'."
            )
        start, end = time_range
        time = np.asarray(time)
        kwargs["normalization_mask"] = (time >= start) & (time <= end)
        logger.warning(
            "Stored parameter normalization_time_range=%s was removed in "
            "ripple_detection 2.0; applying it as normalization_mask="
            "(time >= %s) & (time <= %s)",
            time_range,
            start,
            end,
        )
    elif use_speed and kwargs.get("normalization_mask") is None:
        speed_threshold = kwargs.get("speed_threshold", 4.0)
        kwargs["normalization_mask"] = np.asarray(speed) < speed_threshold
        logger.warning(
            "Stored parameter use_speed_threshold_for_zscore=True was removed "
            "in ripple_detection 2.0; applying it as normalization_mask="
            "speed < %s",
            speed_threshold,
        )

    return kwargs
