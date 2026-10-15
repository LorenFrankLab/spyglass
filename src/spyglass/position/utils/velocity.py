"""Shared velocity computation for V1 and V2 position pipelines."""

import numpy as np


def compute_velocity(
    position: np.ndarray,
    timestamps: np.ndarray,
    smooth_std_dev: float = None,
    sampling_rate: float = None,
) -> tuple:
    """Compute 2D velocity and scalar speed from position timeseries.

    Uses ``np.gradient`` (central differences) on both x and y simultaneously,
    optionally applies a Gaussian smooth to the 2D velocity vector, then
    returns the vector and its magnitude.  Smoothing in 2D before taking the
    magnitude is mathematically correct for turning trajectories; applying
    Gaussian smooth to the scalar magnitude after the fact attenuates turns
    differently and produces a systematic speed bias.

    Parameters
    ----------
    position : np.ndarray
        Shape (n_frames, 2).  Column 0 is x, column 1 is y (cm).
    timestamps : np.ndarray
        Shape (n_frames,).  Time in seconds.
    smooth_std_dev : float, optional
        Gaussian smoothing kernel standard deviation in **seconds**.
        ``None`` (default) skips smoothing.
    sampling_rate : float, optional
        Frames per second.  Derived from ``timestamps`` when not supplied.

    Returns
    -------
    velocity_2d : np.ndarray
        Shape (n_frames, 2).  x- and y-velocity in cm/s.
    speed : np.ndarray
        Shape (n_frames,).  Euclidean speed in cm/s.
    """
    velocity_2d = np.gradient(position, timestamps, axis=0)

    if smooth_std_dev:
        from position_tools.core import gaussian_smooth

        if sampling_rate is None:
            sampling_rate = float(1.0 / np.median(np.diff(timestamps)))
        velocity_2d = gaussian_smooth(
            velocity_2d, smooth_std_dev, sampling_rate, axis=0, truncate=8
        )

    speed = np.sqrt(np.sum(velocity_2d**2, axis=1))
    return velocity_2d, speed
