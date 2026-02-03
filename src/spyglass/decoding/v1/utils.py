import inspect
import logging
from typing import List

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.ndimage import label


def create_interval_labels(
    is_missing: npt.NDArray[np.bool_],
) -> npt.NDArray[np.intp]:
    """Create interval labels from a missing data mask.

    Uses scipy.ndimage.label to identify contiguous regions of valid data
    (where is_missing=False) and assigns sequential integer labels.

    Parameters
    ----------
    is_missing : npt.NDArray[np.bool_], shape (n_time,)
        Boolean mask where True indicates time points outside intervals

    Returns
    -------
    interval_labels : npt.NDArray[np.intp], shape (n_time,)
        Integer labels where:
        - -1 indicates time points outside any interval
        - 0, 1, 2, ... indicate the 1st, 2nd, 3rd interval index
    """
    # label() returns 1-indexed labels (0=outside, 1=first region, 2=second, ...)
    # We want: -1=outside, 0=first interval, 1=second interval, ...
    raw_labels, _ = label(~is_missing)
    return raw_labels - 1


def concatenate_interval_results(
    interval_results: List[xr.Dataset],
) -> xr.Dataset:
    """Concatenate results from multiple intervals along time dimension.

    All datasets must have compatible structure (same variables, compatible
    coordinates except time). Time coordinates will be concatenated and an
    interval_labels coordinate will be added to track which interval each
    time point belongs to.

    Parameters
    ----------
    interval_results : List[xr.Dataset], length n_intervals
        Results from each decoding interval. Each dataset must have a 'time'
        dimension and coordinate. Empty datasets should be filtered out before
        calling this function.

    Returns
    -------
    xr.Dataset
        Concatenated results with interval_labels coordinate.
        The interval_labels coordinate contains integer values where each
        value indicates which interval the corresponding time point belongs to.

    Raises
    ------
    ValueError
        If interval_results is empty or contains empty datasets

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> ds1 = xr.Dataset({"x": ("time", [1, 2, 3])}, coords={"time": [0.0, 0.1, 0.2]})
    >>> ds2 = xr.Dataset({"x": ("time", [4, 5])}, coords={"time": [1.0, 1.1]})
    >>> result = concatenate_interval_results([ds1, ds2])
    >>> result.interval_labels.values
    array([0, 0, 0, 1, 1])
    >>> result.time.values
    array([0. , 0.1, 0.2, 1. , 1.1])
    """
    if not interval_results:
        raise ValueError("All decoding intervals are empty")

    # Validate each result has time points
    for i, result in enumerate(interval_results):
        if len(result.time) == 0:
            raise ValueError(
                f"Interval {i} has empty time dimension - "
                f"should not be included in interval_results list"
            )

    # Pre-allocate with known size for efficiency
    total_length = sum(len(result.time) for result in interval_results)
    interval_labels = np.empty(total_length, dtype=np.intp)

    offset = 0
    for interval_idx, result in enumerate(interval_results):
        n_times = len(result.time)
        interval_labels[offset : offset + n_times] = interval_idx
        offset += n_times

    concatenated = xr.concat(interval_results, dim="time")
    return concatenated.assign_coords(interval_labels=("time", interval_labels))


def _get_interval_range(key: dict) -> tuple[float, float]:
    """Return maximum range of model times in encoding/decoding intervals.

    Note: This function accesses the database to fetch interval times.

    Parameters
    ----------
    key : dict
        The decoding selection key

    Returns
    -------
    tuple[float, float]
        The minimum and maximum times for the model
    """
    # Lazy import to avoid database connection at module load time
    from spyglass.common.common_interval import IntervalList

    encoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["encoding_interval"],
        }
    ).fetch1("valid_times")

    decoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["decoding_interval"],
        }
    ).fetch1("valid_times")

    return (
        float(
            min(
                np.asarray(encoding_interval).min(),
                np.asarray(decoding_interval).min(),
            )
        ),
        float(
            max(
                np.asarray(encoding_interval).max(),
                np.asarray(decoding_interval).max(),
            )
        ),
    )


def get_valid_kwargs(
    classifier,
    decoding_kwargs: dict,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Get valid fit and predict kwargs, warning about any ignored kwargs.

    Inspects the classifier's fit and predict method signatures to determine
    which kwargs are valid. Logs a warning if any provided kwargs are not
    valid for either method.

    Parameters
    ----------
    classifier : object
        Classifier instance with fit and predict methods
    decoding_kwargs : dict
        User-provided kwargs for fit/predict
    logger : logging.Logger
        Logger for warnings

    Returns
    -------
    fit_kwargs : dict
        Kwargs valid for classifier.fit
    predict_kwargs : dict
        Kwargs valid for classifier.predict
    """
    fit_sig = inspect.signature(classifier.fit)
    valid_fit_kwargs: set[str] = set(fit_sig.parameters.keys())
    predict_sig = inspect.signature(classifier.predict)
    valid_predict_kwargs: set[str] = set(predict_sig.parameters.keys())

    # Warn about kwargs that are not valid for either fit or predict
    if decoding_kwargs:
        all_valid_kwargs = valid_fit_kwargs | valid_predict_kwargs
        ignored_kwargs = set(decoding_kwargs.keys()) - all_valid_kwargs
        if ignored_kwargs:
            logger.warning(
                f"The following decoding_kwargs are not valid for "
                f"classifier.fit or classifier.predict and will be ignored: "
                f"{sorted(ignored_kwargs)}. "
                f"Valid fit kwargs: {sorted(valid_fit_kwargs)}. "
                f"Valid predict kwargs: {sorted(valid_predict_kwargs)}."
            )

    fit_kwargs = {
        k: value
        for k, value in decoding_kwargs.items()
        if k in valid_fit_kwargs
    }
    predict_kwargs = {
        k: value
        for k, value in decoding_kwargs.items()
        if k in valid_predict_kwargs
    }

    return fit_kwargs, predict_kwargs
