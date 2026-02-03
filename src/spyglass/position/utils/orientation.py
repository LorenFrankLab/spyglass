"""Orientation calculation utilities for pose estimation.

This module provides tool-agnostic methods for calculating animal orientation
from tracked bodyparts. Supports multiple strategies depending on marker
configuration.

Functions
---------
two_pt_orientation
    Calculate orientation from vector between two points
bisector_orientation
    Calculate orientation from perpendicular bisector of two markers
no_orientation
    Return NaN array (no orientation calculated)
interp_orientation
    Interpolate orientation over NaN spans
smooth_orientation
    Complete orientation processing pipeline with unwrap-smooth-wrap
"""

from itertools import groupby
from operator import itemgetter
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from position_tools.core import gaussian_smooth
except ImportError:
    gaussian_smooth = None


def get_span_start_stop(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and stop indices of consecutive index spans.

    Parameters
    ----------
    indices : np.ndarray
        Array of indices to group into consecutive spans

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, stop) tuples for each consecutive span

    Examples
    --------
    >>> indices = np.array([0, 1, 2, 5, 6, 10])
    >>> get_span_start_stop(indices)
    [(0, 2), (5, 6), (10, 10)]
    """
    span_inds = []
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        span_inds.append((group[0], group[-1]))
    return span_inds


def two_pt_orientation(
    pos_df: pd.DataFrame,
    point1: str,
    point2: str,
    **kwargs,
) -> np.ndarray:
    """Calculate orientation from vector between two bodyparts.

    Orientation is defined as the angle from point2 to point1, measured
    counter-clockwise from the positive x-axis. Result is in radians [-π, π].

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
        where coord is 'x' or 'y'
    point1 : str
        Name of first bodypart (head/front marker)
    point2 : str
        Name of second bodypart (tail/back marker)
    **kwargs
        Additional keyword arguments (ignored, for compatibility)

    Returns
    -------
    np.ndarray
        Orientation in radians, shape (n_frames,)
        NaN where either point is NaN

    Examples
    --------
    >>> # pos_df has columns: ('green_led', 'x'), ('green_led', 'y'),
    >>> #                      ('red_led', 'x'), ('red_led', 'y')
    >>> orientation = two_pt_orientation(
    ...     pos_df, point1='green_led', point2='red_led'
    ... )
    >>> orientation[0]  # Angle in radians
    1.5707963267948966  # π/2 (pointing up)

    Notes
    -----
    - Orientation is arctan2(y1-y2, x1-x2)
    - Result is in range [-π, π]
    - Returns NaN where either point has NaN coordinates
    """
    orientation = np.arctan2(
        pos_df[point1]["y"] - pos_df[point2]["y"],
        pos_df[point1]["x"] - pos_df[point2]["x"],
    )
    return orientation.values


def no_orientation(
    pos_df: pd.DataFrame,
    fill_with: float = np.nan,
    **kwargs,
) -> np.ndarray:
    """Return array of constant values (typically NaN) for orientation.

    Used when orientation calculation is not needed or not possible.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data (only used for length)
    fill_with : float, optional
        Value to fill array with, by default np.nan
    **kwargs
        Additional keyword arguments (ignored, for compatibility)

    Returns
    -------
    np.ndarray
        Array of shape (n_frames,) filled with fill_with

    Examples
    --------
    >>> orientation = no_orientation(pos_df)  # All NaN
    >>> orientation = no_orientation(pos_df, fill_with=0.0)  # All zeros
    """
    n_frames = len(pos_df)
    orientation = np.full(
        shape=(n_frames,), fill_value=fill_with, dtype=np.float32
    )
    return orientation


def bisector_orientation(
    pos_df: pd.DataFrame,
    led1: str,
    led2: str,
    led3: str,
    **kwargs,
) -> np.ndarray:
    """Calculate orientation from perpendicular bisector of two markers.

    This method uses two markers (led1, led2) assumed to be perpendicular to
    the orientation direction, plus a third marker (led3) to determine
    forward/backward direction.

    The orientation is the perpendicular bisector of the line connecting
    led1 and led2, with direction determined by led3.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
    led1 : str
        Name of first perpendicular marker
    led2 : str
        Name of second perpendicular marker (opposite side)
    led3 : str
        Name of forward/backward marker (determines direction)
    **kwargs
        Additional keyword arguments (ignored, for compatibility)

    Returns
    -------
    np.ndarray
        Orientation in radians, shape (n_frames,)
        Range: [-π, π]

    Raises
    ------
    ValueError
        If all three markers are collinear (cannot determine direction)

    Examples
    --------
    >>> # Three LEDs: two on sides, one on head
    >>> orientation = bisector_orientation(
    ...     pos_df, led1='left_led', led2='right_led', led3='head_led'
    ... )

    Notes
    -----
    Algorithm:
    1. Compute vector from led2 to led1: (x_vec, y_vec)
    2. Perpendicular vector: (-y_vec, x_vec) / |vector|
    3. Use led3 position to determine sign of orientation
    4. Special handling when y_vec ≈ 0 (horizontal alignment)
    """
    orient = np.full(len(pos_df), np.nan)  # Initialize with NaNs

    # Vector from led2 to led1
    x_vec = pos_df[led1]["x"] - pos_df[led2]["x"]
    y_vec = pos_df[led1]["y"] - pos_df[led2]["y"]
    y_eq0 = np.isclose(y_vec, 0)

    # Special case: when y_vec is zero, led1 and led2 are horizontally aligned
    # Compare to led3 to determine if pointing up or down
    orient[y_eq0 & pos_df[led3]["y"].gt(pos_df[led1]["y"])] = np.pi / 2
    orient[y_eq0 & pos_df[led3]["y"].lt(pos_df[led1]["y"])] = -np.pi / 2

    # Error case: all three markers collinear
    y_1, y_2, y_3 = pos_df[led1]["y"], pos_df[led2]["y"], pos_df[led3]["y"]
    if np.any(y_eq0 & np.isclose(y_1, y_2) & np.isclose(y_2, y_3)):
        raise ValueError(
            "Cannot determine orientation from bisector: "
            "all three markers are collinear. "
            f"Markers: {led1}, {led2}, {led3}"
        )

    # General case: compute perpendicular bisector
    # Perpendicular to (x_vec, y_vec) is (-y_vec, x_vec)
    length = np.sqrt(x_vec**2 + y_vec**2)
    norm_x = (-y_vec / length)[~y_eq0]
    norm_y = (x_vec / length)[~y_eq0]
    orient[~y_eq0] = np.arctan2(norm_y, norm_x)

    return orient


def interp_orientation(
    df: pd.DataFrame,
    spans_to_interp: List[Tuple[int, int]],
    **kwargs,
) -> pd.DataFrame:
    """Linearly interpolate orientation over NaN spans.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'orientation' column and time index
    spans_to_interp : List[Tuple[int, int]]
        List of (start_idx, stop_idx) tuples indicating NaN spans
    **kwargs
        Additional keyword arguments (ignored, for compatibility)

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated orientation values

    Notes
    -----
    - Spans at the beginning or end of data are left as NaN
    - Interpolation is linear between bounding points
    - Input orientation should be unwrapped before calling
    """
    idx = pd.IndexSlice
    no_x_msg = "Index {ind} has no {x}point with which to interpolate"
    df_orient = df["orientation"]

    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        idx_span = idx[span_start:span_stop]

        # Can't interpolate if span extends to end
        if (span_stop + 1) >= len(df):
            df.loc[idx_span, idx["orientation"]] = np.nan
            print(no_x_msg.format(ind=ind, x="stop"))
            continue

        # Can't interpolate if span starts at beginning
        if span_start < 1:
            df.loc[idx_span, idx["orientation"]] = np.nan
            print(no_x_msg.format(ind=ind, x="start"))
            continue

        # Get bounding orientation values and timestamps
        orient = [df_orient.iloc[span_start - 1], df_orient.iloc[span_stop + 1]]

        # Use timestamps BEFORE and AFTER the span for interpolation
        time_before = df.index[span_start - 1]
        time_after = df.index[span_stop + 1]

        # Interpolate at the span timestamps
        span_times = df.index[span_start : span_stop + 1]
        orientnew = np.interp(
            x=span_times,
            xp=[time_before, time_after],
            fp=[orient[0], orient[-1]],
        )
        df.loc[idx[span_times[0] : span_times[-1]], idx["orientation"]] = (
            orientnew
        )

    return df


def smooth_orientation(
    orientation: np.ndarray,
    timestamps: np.ndarray,
    std_dev: float,
    interpolate: bool = True,
) -> np.ndarray:
    """Complete orientation processing pipeline with unwrap-smooth-wrap.

    This function performs the standard orientation processing workflow:
    1. Unwrap orientation (handle -π/π discontinuities)
    2. Interpolate over NaN spans (if requested)
    3. Gaussian smooth
    4. Wrap back to [-π, π]

    Parameters
    ----------
    orientation : np.ndarray
        Raw orientation in radians, shape (n_frames,)
        May contain NaN values
    timestamps : np.ndarray
        Timestamps for each frame, shape (n_frames,)
    std_dev : float
        Standard deviation for Gaussian smoothing (in seconds)
    interpolate : bool, optional
        Whether to interpolate over NaN spans before smoothing,
        by default True

    Returns
    -------
    np.ndarray
        Smoothed orientation in radians [-π, π], shape (n_frames,)

    Raises
    ------
    ImportError
        If position_tools is not installed (required for gaussian_smooth)

    Examples
    --------
    >>> orientation_smooth = smooth_orientation(
    ...     orientation=raw_orientation,
    ...     timestamps=timestamps,
    ...     std_dev=0.001,  # 1 ms smoothing
    ...     interpolate=True
    ... )

    Notes
    -----
    - Unwrapping prevents smoothing artifacts at -π/π boundary
    - Gaussian smoothing uses position_tools.core.gaussian_smooth
    - Final wrapping ensures result is in standard range [-π, π]
    """
    if gaussian_smooth is None:
        raise ImportError(
            "position_tools is required for orientation smoothing. "
            "Install with: pip install position-tools"
        )

    # Identify NaN locations
    is_nan = np.isnan(orientation)

    # 1. Unwrap orientation (handle -π/π discontinuities)
    unwrap_orientation = orientation.copy()
    # Only unwrap non-NaN values
    unwrap_orientation[~is_nan] = np.unwrap(orientation[~is_nan])

    # 2. Interpolate over NaN spans (if requested)
    if interpolate and np.any(is_nan):
        # Find NaN spans
        nan_indices = np.where(is_nan)[0]
        nan_spans = get_span_start_stop(nan_indices)

        # Create DataFrame for interpolation
        unwrap_df = pd.DataFrame(
            unwrap_orientation,
            columns=["orientation"],
            index=timestamps,
        )

        # Interpolate
        unwrap_df = interp_orientation(unwrap_df, nan_spans)
        unwrap_orientation = unwrap_df["orientation"].to_numpy()

    # 3. Gaussian smooth
    sampling_rate = 1 / np.median(np.diff(timestamps))
    smooth_orient = gaussian_smooth(
        unwrap_orientation,
        std_dev,
        sampling_rate,
        axis=0,
        truncate=8,
    )

    # 4. Wrap back to [-π, π]
    orientation_final = np.angle(np.exp(1j * smooth_orient))

    return orientation_final
