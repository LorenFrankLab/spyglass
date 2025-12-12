from typing import List, Tuple

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

    Parameters
    ----------
    interval_results : List[xr.Dataset], length n_intervals
        Results from each decoding interval

    Returns
    -------
    xr.Dataset
        Concatenated results with interval_labels coordinate

    Raises
    ------
    ValueError
        If interval_results is empty
    """
    if not interval_results:
        raise ValueError("All decoding intervals are empty")

    interval_labels = []
    for interval_idx, result in enumerate(interval_results):
        interval_labels.extend([interval_idx] * len(result.time))

    concatenated = xr.concat(interval_results, dim="time")
    return concatenated.assign_coords(interval_labels=("time", interval_labels))


def _get_interval_range(key: dict) -> Tuple[float, float]:
    """Return maximum range of model times in encoding/decoding intervals

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
        min(
            np.asarray(encoding_interval).min(),
            np.asarray(decoding_interval).min(),
        ),
        max(
            np.asarray(encoding_interval).max(),
            np.asarray(decoding_interval).max(),
        ),
    )
