import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


def get_centroid(position1, position2):
    """Finds the midpoint of two positions.

    Parameters
    ----------
    position1, position2 : np.ndarray, shape (n_time, 2)

    Returns
    -------
    centroid_position : np.ndarray, shape (n_time, 2)

    """
    return (position1 + position2) / 2


def get_distance(position1, position2):
    """Finds the distance between two positions

    Parameters
    ----------
    position1, position2 : np.ndarray, shape (n_time, 2)

    Returns
    -------
    distance : np.ndarray, shape (n_time,)

    """
    return np.linalg.norm(position2 - position1, axis=1)


def get_angle(back_LED, front_LED):
    """Returns the angle between the front and back LEDs in radians

    Parameters
    ----------
    back_LED, front_LED : np.ndarray, shape (n_time, 2)

    Returns
    -------
    head_orientation : np.ndarray, shape (n_time,)

    """
    return np.arctan2(
        front_LED[:, 1] - back_LED[:, 1], front_LED[:, 0] - back_LED[:, 0]
    )


def interpolate_nan(position, time=None):
    """Interpolate to fill nan values.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, 2)
    time : None or np.ndarray, shape (n_time,)

    Returns
    -----
    position : np.ndarray, shape (n_time, 2)

    """

    n_time, n_position_dims = position.shape

    if time is None:
        time = np.arange(n_time)

    not_nan = np.all(np.isfinite(position), axis=1)

    interpolated_position = []

    for position_ind in range(n_position_dims):
        f = interpolate.interp1d(
            time[not_nan],
            position[not_nan, position_ind],
            bounds_error=False,
            kind="linear",
        )
        interpolated_position.append(
            np.where(
                np.isfinite(position[:, position_ind]),
                position[:, position_ind],
                f(time),
            )
        )
    return np.stack(interpolated_position, axis=1)


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """
    return gaussian_filter1d(
        data,
        sigma * sampling_frequency,
        truncate=truncate,
        axis=axis,
        mode="constant",
    )


def get_velocity(position, time=None, sigma=15, sampling_frequency=1):
    if time is None:
        time = np.arange(position.shape[0])

    return gaussian_smooth(
        np.gradient(position, time, axis=0),
        sigma,
        sampling_frequency,
        axis=0,
        truncate=8,
    )


def get_speed(position, time=None, sigma=15, sampling_frequency=1):
    velocity = get_velocity(
        position, time=time, sigma=sigma, sampling_frequency=sampling_frequency
    )
    return np.sqrt(np.sum(velocity**2, axis=1))
