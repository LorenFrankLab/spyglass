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
span_length
    Length of an inclusive (start, stop) span
get_good_spans
    Merge nearby good spans for V1-compatible span detection
mask_short_valid_islands
    Mark short valid islands surrounded by NaN as missing
"""

from typing import Dict, List, Optional, Tuple, Callable

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


def _interpolate_spans_generic(
    df: pd.DataFrame,
    spans_to_interp: List[Tuple[int, int]],
    get_boundary_values: Callable,
    apply_interpolated_values: Callable,
    validate_span: Optional[Callable] = None,
    get_error_context: Optional[Callable] = None,
) -> pd.DataFrame:
    """Generic span interpolation utility for position and orientation data.

    This function centralizes the common interpolation logic shared between
    orientation and position interpolation functions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time index
    spans_to_interp : List[Tuple[int, int]]
        List of (start_idx, stop_idx) tuples indicating NaN spans
    get_boundary_values : Callable
        Function that extracts boundary values for interpolation
        Signature: (df, span_start, span_stop) -> values_dict
    apply_interpolated_values : Callable
        Function that applies interpolated values back to dataframe
        Signature: (df, span_start, span_stop, interpolated_values, **kwargs) -> None
    validate_span : Optional[Callable]
        Optional function to validate if span should be interpolated
        Signature: (df, span_start, span_stop, boundary_values) -> (bool, str)
        Returns (is_valid, error_msg)
    get_error_context : Optional[Callable]
        Optional function to provide context for error messages
        Signature: (ind, error_type) -> str

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values
    """
    idx = pd.IndexSlice
    default_error_msg = (
        "Index {ind} has no {error_type}point with which to interpolate"
    )

    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        span_times = df.index[span_start : span_stop + 1]

        # Can't interpolate if span extends to end
        if (span_stop + 1) >= len(df):
            error_msg = (
                get_error_context(ind, "end")
                if get_error_context
                else default_error_msg.format(ind=ind, error_type="end")
            )
            logger.info_msg(error_msg)
            continue

        # Can't interpolate if span starts at beginning
        if span_start < 1:
            error_msg = (
                get_error_context(ind, "start")
                if get_error_context
                else default_error_msg.format(ind=ind, error_type="start")
            )
            logger.info_msg(error_msg)
            continue

        # Get boundary values for interpolation
        boundary_values = get_boundary_values(df, span_start, span_stop)

        # Optional validation (e.g., distance/length constraints)
        if validate_span:
            is_valid, validation_error = validate_span(
                df, span_start, span_stop, boundary_values
            )
            if not is_valid:
                logger.info_msg(validation_error)
                continue

        # Use timestamps BEFORE and AFTER the span for interpolation
        time_before = df.index[span_start - 1]
        time_after = df.index[span_stop + 1]
        span_times = df.index[span_start : span_stop + 1]

        # Perform interpolation and apply values
        apply_interpolated_values(
            df,
            span_start,
            span_stop,
            boundary_values,
            span_times,
            time_before,
            time_after,
            idx,
        )

    return df


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
    x_col, y_col = coord_cols

    if max_pts_to_interp is None:
        max_pts_to_interp = float("inf")
    if max_cm_to_interp is None:
        max_cm_to_interp = float("inf")

    def get_boundary_values(df, span_start, span_stop):
        """Extract x,y boundary values for position interpolation."""
        return {
            "x": [
                df[x_col].iloc[span_start - 1],
                df[x_col].iloc[span_stop + 1],
            ],
            "y": [
                df[y_col].iloc[span_start - 1],
                df[y_col].iloc[span_stop + 1],
            ],
        }

    def validate_span(df, span_start, span_stop, boundary_values):
        """Validate span constraints for position interpolation."""
        span_len = int(span_stop - span_start + 1)
        distance = np.linalg.norm(
            np.array([boundary_values["x"][0], boundary_values["y"][0]])
            - np.array([boundary_values["x"][1], boundary_values["y"][1]])
        )

        if span_len > max_pts_to_interp or distance > max_cm_to_interp:
            return False, f"Index {span_start} to {span_stop} not interpolated"
        return True, ""

    def apply_interpolated_values(
        df,
        span_start,
        span_stop,
        boundary_values,
        span_times,
        time_before,
        time_after,
        idx,
    ):
        """Apply interpolated position values to dataframe."""
        # Interpolate x and y coordinates
        x_new = np.interp(
            x=span_times,
            xp=[time_before, time_after],
            fp=[boundary_values["x"][0], boundary_values["x"][1]],
        )
        y_new = np.interp(
            x=span_times,
            xp=[time_before, time_after],
            fp=[boundary_values["y"][0], boundary_values["y"][1]],
        )

        # Apply interpolated values
        df.loc[idx[span_times[0] : span_times[-1]], idx[x_col]] = x_new
        df.loc[idx[span_times[0] : span_times[-1]], idx[y_col]] = y_new

        # Set NaN values for failed spans
        span_times = df.index[span_start : span_stop + 1]
        if len(x_new) == 0:  # Validation failed
            df.loc[span_times, [x_col, y_col]] = np.nan

    def get_error_context(ind, error_type):
        """Generate error messages for position interpolation."""
        return f"Index {ind} has no {error_type}point with which to interpolate"

    return _interpolate_spans_generic(
        pos_df,
        spans_to_interp,
        get_boundary_values,
        apply_interpolated_values,
        validate_span,
        get_error_context,
    )


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
    moving_avg_window = max(
        1, int(np.round(smoothing_duration * sampling_rate))
    )

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


def span_length(span: Tuple[int, int]) -> int:
    """Return the number of frames spanned by an inclusive (start, stop) tuple.

    Parameters
    ----------
    span : Tuple[int, int]
        Inclusive ``(start, stop)`` positional-index pair.

    Returns
    -------
    int
        ``stop - start``.  Zero for a single-frame span.
    """
    return span[-1] - span[0]


def get_good_spans(
    bad_inds_mask: np.ndarray,
    inds_to_span: int = 50,
) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
    """Find and optionally merge spans of valid (non-bad) indices.

    Two good spans separated by ≤ ``inds_to_span`` bad frames are merged into
    a single span in ``modified_spans``.  This matches V1's behaviour in
    ``position_dlc_position``, where the merged span determines the region
    used for bidirectional jump detection.

    Parameters
    ----------
    bad_inds_mask : np.ndarray
        Boolean mask, length n.  ``True`` = bad/missing index.
    inds_to_span : int, optional
        Maximum gap (frames) between two good spans before they are
        merged.  Default 50.

    Returns
    -------
    good : list of (int, int) or None
        Unmodified spans of consecutive valid indices.
        ``None`` when no valid indices exist.
    modified_spans : list of (int, int)
        Good spans after merging neighbours separated by ≤ ``inds_to_span``
        bad frames.

    Notes
    -----
    Moved from V1 ``position_dlc_position`` so V2 can share the implementation.
    """
    from spyglass.position.utils.orientation import get_span_start_stop

    good = get_span_start_stop(np.arange(len(bad_inds_mask))[~bad_inds_mask])

    if len(good) < 1:
        return None, good
    if len(good) == 1:
        return good, good

    modified_spans: List[Tuple[int, int]] = []
    for (start1, stop1), (start2, stop2) in zip(good[:-1], good[1:]):
        check_existing = [
            entry
            for entry in modified_spans
            if start1 in range(entry[0] - inds_to_span, entry[1] + inds_to_span)
        ]
        if check_existing:
            modify_ind = modified_spans.index(check_existing[0])
            if (start2 - stop1) <= inds_to_span:
                modified_spans[modify_ind] = (check_existing[0][0], stop2)
            else:
                modified_spans[modify_ind] = (check_existing[0][0], stop1)
                modified_spans.append((start2, stop2))
            continue
        if (start2 - stop1) <= inds_to_span:
            modified_spans.append((start1, stop2))
        else:
            modified_spans.append((start1, stop1))
            modified_spans.append((start2, stop2))
    return good, modified_spans


def mask_short_valid_islands(
    is_nan: np.ndarray,
    min_island_len: int,
) -> np.ndarray:
    """Mark short valid islands surrounded by NaN as missing.

    A *valid island* is a consecutive run of non-NaN frames flanked by NaN on
    both sides.  Very short islands make poor interpolation anchors — linear
    interpolation across a long bad stretch that happens to contain a 1–2 frame
    valid island can produce unrealistic trajectories.  Masking them first lets
    ``interp_position`` treat the whole region as a single uninterpolatable gap.

    Parameters
    ----------
    is_nan : np.ndarray
        Boolean array of length n.  ``True`` = NaN / missing data.
    min_island_len : int
        Minimum island length (frames, inclusive) to retain as valid.
        Islands strictly shorter than this that are surrounded by NaN on
        both sides are set to ``True`` (masked).  Pass 0 or negative to
        disable.

    Returns
    -------
    np.ndarray
        Copy of ``is_nan`` with short surrounded islands also set to ``True``.
    """
    from spyglass.position.utils.orientation import get_span_start_stop

    if min_island_len <= 0:
        return is_nan.copy()
    is_nan = is_nan.copy()
    valid_indices = np.where(~is_nan)[0]
    if len(valid_indices) == 0:
        return is_nan
    n = len(is_nan)
    for start, stop in get_span_start_stop(valid_indices):
        if (stop - start + 1) >= min_island_len:
            continue
        surrounded = (start > 0 and is_nan[start - 1]) and (
            stop < n - 1 and is_nan[stop + 1]
        )
        if surrounded:
            is_nan[start : stop + 1] = True
    return is_nan


# V1 Compatibility alias
_key_to_smooth_func_dict = {
    "moving_avg": smooth_moving_avg,
}
