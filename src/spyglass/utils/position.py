import numpy as np


def convert_to_pixels(data, frame_size=None, cm_to_pixels=1.0):
    """Converts from cm to pixels and flips the y-axis.
    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    cm_to_pixels : float

    Returns
    -------
    converted_data : ndarray, shape (n_time, 2)
    """
    return data / cm_to_pixels


def fill_nan(variable, video_time, variable_time):
    """Fill in missing values in variable with nans at video_time points."""
    video_ind = np.digitize(variable_time, video_time[1:])

    n_video_time = len(video_time)
    try:
        n_variable_dims = variable.shape[1]
        filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
    except IndexError:
        filled_variable = np.full((n_video_time,), np.nan)
    filled_variable[video_ind] = variable

    return filled_variable
