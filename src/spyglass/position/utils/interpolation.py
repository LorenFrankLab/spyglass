"""Interpolation and smoothing utilities for pose estimation.

This module provides tool-agnostic methods for interpolating and smoothing
pose tracking data. Supports various smoothing algorithms and constraints.

Functions
---------
interp_position
    Interpolate position (x, y) over NaN spans with constraints
smooth_moving_avg
    Moving average smoothing using bottleneck
smooth_savgol
    Savitzky-Golay filter smoothing
get_smoothing_function
    Retrieve smoothing function by name
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spyglass.utils.logging import logger

try:
    import bottleneck as bn
except ImportError:
    bn = None

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None


def interp_position(
    pos_df: pd.DataFrame,
    spans_to_interp: List[Tuple[int, int]],
    coord_cols: Tuple[str, str] = ("x", "y"),
    max_pts_to_interp: Optional[int] = None,
    max_cm_to_interp: Optional[float] = None,
) -> pd.DataFrame:
    """Linearly interpolate position coordinates over NaN spans.

    Interpolates x and y coordinates between valid bounding points,
    subject to constraints on span length and distance traveled.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with time index
        Columns should include coord_cols (e.g., 'x', 'y')
    spans_to_interp : List[Tuple[int, int]]
        List of (start_idx, stop_idx) tuples indicating NaN spans
        Indices are positional indices (not timestamps)
    coord_cols : Tuple[str, str], optional
        Column names for x and y coordinates, by default ("x", "y")
    max_pts_to_interp : int, optional
        Maximum number of points in a span to interpolate
        Spans longer than this will remain NaN
        Default: None (no limit)
    max_cm_to_interp : float, optional
        Maximum distance (in cm or pixels) between bounding points
        Spans with larger distances will remain NaN
        Default: None (no limit)

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated position values

    Examples
    --------
    >>> # Interpolate over small NaN spans
    >>> pos_df_interp = interp_position(
    ...     pos_df,
    ...     spans_to_interp=[(10, 12), (20, 22)],
    ...     max_pts_to_interp=5,
    ...     max_cm_to_interp=20.0
    ... )

    Notes
    -----
    - Spans at the beginning or end of data are left as NaN
    - Interpolation is linear between bounding points
    - Constraints prevent unrealistic interpolation across large gaps
    """
    idx = pd.IndexSlice
    x_col, y_col = coord_cols

    if max_pts_to_interp is None:
        max_pts_to_interp = float("inf")
    if max_cm_to_interp is None:
        max_cm_to_interp = float("inf")

    no_x_msg = "Index {ind} has no {coord}point with which to interpolate"
    no_interp_msg = "Index {start} to {stop} not interpolated"

    def _get_new_coord(
        coord_vals: List[float],
        span_start: int,
        span_stop: int,
        start_time: float,
        stop_time: float,
    ) -> np.ndarray:
        """Interpolate coordinate values over span."""
        return np.interp(
            x=pos_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[coord_vals[0], coord_vals[-1]],
        )

    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        idx_span = idx[span_start:span_stop]

        # Can't interpolate if span extends to end
        if (span_stop + 1) >= len(pos_df):
            pos_df.loc[idx_span, idx[[x_col, y_col]]] = np.nan
            logger.info(no_x_msg.format(ind=ind, coord="end"))
            continue

        # Can't interpolate if span starts at beginning
        if span_start < 1:
            pos_df.loc[idx_span, idx[[x_col, y_col]]] = np.nan
            logger.info(no_x_msg.format(ind=ind, coord="start"))
            continue

        # Get bounding coordinate values
        x = [
            pos_df[x_col].iloc[span_start - 1],
            pos_df[x_col].iloc[span_stop + 1],
        ]
        y = [
            pos_df[y_col].iloc[span_start - 1],
            pos_df[y_col].iloc[span_stop + 1],
        ]

        # Check constraints
        span_len = int(span_stop - span_start + 1)
        distance = np.linalg.norm(
            np.array([x[0], y[0]]) - np.array([x[1], y[1]])
        )

        if span_len > max_pts_to_interp or distance > max_cm_to_interp:
            pos_df.loc[idx_span, idx[[x_col, y_col]]] = np.nan
            logger.info(no_interp_msg.format(start=span_start, stop=span_stop))
            continue

        # Use timestamps BEFORE and AFTER the span for interpolation
        time_before = pos_df.index[span_start - 1]
        time_after = pos_df.index[span_stop + 1]

        # Interpolate
        x_new = _get_new_coord(
            x, span_start, span_stop, time_before, time_after
        )
        y_new = _get_new_coord(
            y, span_start, span_stop, time_before, time_after
        )

        # Get timestamps within the span
        span_times = pos_df.index[span_start : span_stop + 1]
        pos_df.loc[idx[span_times[0] : span_times[-1]], idx[x_col]] = x_new
        pos_df.loc[idx[span_times[0] : span_times[-1]], idx[y_col]] = y_new

    return pos_df


def smooth_moving_avg(
    pos_df: pd.DataFrame,
    smoothing_duration: float,
    sampling_rate: float,
    coord_cols: Tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Smooth position coordinates using moving average.

    Uses bottleneck's move_mean for efficient computation.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with time index
    smoothing_duration : float
        Duration of smoothing window in seconds
        Window size = smoothing_duration * sampling_rate
    sampling_rate : float
        Sampling rate in Hz
    coord_cols : Tuple[str, str], optional
        Column names for x and y coordinates, by default ("x", "y")

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed position values

    Raises
    ------
    ImportError
        If bottleneck is not installed

    Examples
    --------
    >>> # Smooth with 50ms window at 30 Hz
    >>> pos_df_smooth = smooth_moving_avg(
    ...     pos_df,
    ...     smoothing_duration=0.05,
    ...     sampling_rate=30.0
    ... )

    Notes
    -----
    - Window size is rounded to nearest integer
    - Uses min_count=1 to handle NaN values gracefully
    - Modifies DataFrame in place and returns it
    """
    if bn is None:
        raise ImportError(
            "bottleneck is required for moving average smoothing. "
            "Install with: pip install bottleneck"
        )

    idx = pd.IndexSlice
    x_col, y_col = coord_cols
    moving_avg_window = int(np.round(smoothing_duration * sampling_rate))

    # Extract x, y arrays
    xy_arr = pos_df.loc[:, idx[[x_col, y_col]]].values

    # Apply moving average
    smoothed_xy_arr = bn.move_mean(
        xy_arr, window=moving_avg_window, axis=0, min_count=1
    )

    # Update DataFrame
    pos_df.loc[:, idx[x_col]], pos_df.loc[:, idx[y_col]] = [
        *zip(*smoothed_xy_arr.tolist())
    ]

    return pos_df


def smooth_savgol(
    pos_df: pd.DataFrame,
    window_length: int,
    polyorder: int = 3,
    coord_cols: Tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Smooth position coordinates using Savitzky-Golay filter.

    The Savitzky-Golay filter fits successive sub-sets of adjacent data
    points with a low-degree polynomial by linear least squares.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with time index
    window_length : int
        Length of the filter window (must be odd and > polyorder)
    polyorder : int, optional
        Order of polynomial to fit, by default 3
    coord_cols : Tuple[str, str], optional
        Column names for x and y coordinates, by default ("x", "y")

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed position values

    Raises
    ------
    ImportError
        If scipy is not installed
    ValueError
        If window_length is even or <= polyorder

    Examples
    --------
    >>> # Smooth with Savitzky-Golay filter
    >>> pos_df_smooth = smooth_savgol(
    ...     pos_df,
    ...     window_length=11,
    ...     polyorder=3
    ... )

    Notes
    -----
    - Window length must be odd and greater than polyorder
    - Better preserves features than moving average
    - Handles NaN values by propagating them
    """
    if savgol_filter is None:
        raise ImportError(
            "scipy is required for Savitzky-Golay smoothing. "
            "Install with: pip install scipy"
        )

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")

    idx = pd.IndexSlice
    x_col, y_col = coord_cols

    # Apply Savitzky-Golay filter to each coordinate
    # Note: savgol_filter can handle NaN by using mode='interp'
    x_smooth = savgol_filter(
        pos_df[x_col].values,
        window_length,
        polyorder,
        mode="interp",
    )
    y_smooth = savgol_filter(
        pos_df[y_col].values,
        window_length,
        polyorder,
        mode="interp",
    )

    pos_df.loc[:, idx[x_col]] = x_smooth
    pos_df.loc[:, idx[y_col]] = y_smooth

    return pos_df


def smooth_gaussian(
    pos_df: pd.DataFrame,
    std_dev: float,
    sampling_rate: float,
    coord_cols: Tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Smooth position coordinates using Gaussian filter.

    Uses position_tools.core.gaussian_smooth for efficient computation.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with time index
    std_dev : float
        Standard deviation of Gaussian kernel in seconds
    sampling_rate : float
        Sampling rate in Hz
    coord_cols : Tuple[str, str], optional
        Column names for x and y coordinates, by default ("x", "y")

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed position values

    Raises
    ------
    ImportError
        If position_tools is not installed

    Examples
    --------
    >>> # Smooth with 1ms Gaussian kernel at 30 Hz
    >>> pos_df_smooth = smooth_gaussian(
    ...     pos_df,
    ...     std_dev=0.001,
    ...     sampling_rate=30.0
    ... )

    Notes
    -----
    - Gaussian smoothing preserves temporal relationships better than moving average
    - Uses truncate=8 for efficiency
    """
    try:
        from position_tools.core import gaussian_smooth
    except ImportError:
        raise ImportError(
            "position_tools is required for Gaussian smoothing. "
            "Install with: pip install position-tools"
        )

    idx = pd.IndexSlice
    x_col, y_col = coord_cols

    # Apply Gaussian smoothing to each coordinate
    x_smooth = gaussian_smooth(
        pos_df[x_col].values,
        std_dev,
        sampling_rate,
        axis=0,
        truncate=8,
    )
    y_smooth = gaussian_smooth(
        pos_df[y_col].values,
        std_dev,
        sampling_rate,
        axis=0,
        truncate=8,
    )

    pos_df.loc[:, idx[x_col]] = x_smooth
    pos_df.loc[:, idx[y_col]] = y_smooth

    return pos_df


# Dictionary mapping smoothing method names to functions
SMOOTHING_METHODS: Dict[str, Callable] = {
    "moving_avg": smooth_moving_avg,
    "savgol": smooth_savgol,
    "gaussian": smooth_gaussian,
}


def get_smoothing_function(method: str) -> Callable:
    """Get smoothing function by name.

    Parameters
    ----------
    method : str
        Name of smoothing method
        Options: "moving_avg", "savgol", "gaussian"

    Returns
    -------
    Callable
        Smoothing function

    Raises
    ------
    ValueError
        If method is not recognized

    Examples
    --------
    >>> smooth_func = get_smoothing_function("moving_avg")
    >>> pos_df_smooth = smooth_func(pos_df, smoothing_duration=0.05, sampling_rate=30.0)
    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(
            f"Unknown smoothing method: {method}. "
            f"Available methods: {list(SMOOTHING_METHODS.keys())}"
        )
    return SMOOTHING_METHODS[method]
