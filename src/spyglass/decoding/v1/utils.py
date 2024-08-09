import numpy as np

from spyglass.common.common_interval import IntervalList


def _get_interval_range(key):
    """Return maximum range of model times in encoding/decoding intervals

    Parameters
    ----------
    key : dict
        The decoding selection key

    Returns
    -------
    Tuple[float, float]
        The minimum and maximum times for the model
    """
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
