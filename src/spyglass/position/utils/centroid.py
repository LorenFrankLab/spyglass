"""Centroid calculation utilities for pose estimation.

This module provides tool-agnostic methods for calculating centroids from
tracked bodyparts. Supports 1-point, 2-point, and 4-point configurations.

Functions
---------
calculate_centroid
    Calculate centroid from 1, 2, or 4 bodyparts with validation
get_1pt_centroid
    Passthrough single bodypart as centroid
get_2pt_centroid
    Average of two bodyparts with fallback logic
get_4pt_centroid
    Complex averaging for 4-LED configuration (green + 3 red LEDs)
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from position_tools import get_distance
except ImportError:
    get_distance = None


def calculate_centroid(
    pos_df: pd.DataFrame,
    points: Dict[str, str],
    max_LED_separation: Optional[float] = None,
) -> np.ndarray:
    """Calculate centroid from 1, 2, or 4 bodyparts.

    This is the main entry point for centroid calculation. It dispatches
    to specialized functions based on the number of points provided.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
        where coord is 'x' or 'y'
    points : Dict[str, str]
        Dictionary mapping point roles to bodypart names.
        - 1-point: {"point1": "bodypart_name"}
        - 2-point: {"point1": "name1", "point2": "name2"}
        - 4-point: {"greenLED": "green", "redLED_C": "center",
                    "redLED_L": "left", "redLED_R": "right"}
    max_LED_separation : float, optional
        Maximum allowed distance between LEDs (in pixels).
        Required for 2-point and 4-point configurations.
        Points farther apart will be treated as invalid.

    Returns
    -------
    np.ndarray
        Centroid positions, shape (n_frames, 2)
        Contains NaN where centroid cannot be calculated

    Raises
    ------
    ValueError
        If number of points is not 1, 2, or 4
        If max_LED_separation not provided for multi-point configs
    ImportError
        If position_tools not installed (required for distance checks)

    Examples
    --------
    >>> # 1-point centroid (simple passthrough)
    >>> centroid = calculate_centroid(
    ...     pos_df,
    ...     points={"point1": "nose"}
    ... )

    >>> # 2-point centroid (average of two markers)
    >>> centroid = calculate_centroid(
    ...     pos_df,
    ...     points={"point1": "green_led", "point2": "red_led"},
    ...     max_LED_separation=15.0
    ... )

    >>> # 4-point centroid (complex LED configuration)
    >>> centroid = calculate_centroid(
    ...     pos_df,
    ...     points={
    ...         "greenLED": "green",
    ...         "redLED_C": "red_center",
    ...         "redLED_L": "red_left",
    ...         "redLED_R": "red_right"
    ...     },
    ...     max_LED_separation=15.0
    ... )

    Notes
    -----
    The 4-point configuration uses a priority-based averaging strategy:
    1. If green and center are valid → average them
    2. If green and left/right are valid → average left, right, green
    3. If only left/right are valid → average left and right
    4. If only center is valid → use center
    5. Otherwise → NaN
    """
    # Validate inputs
    n_points = len(points)
    if n_points not in [1, 2, 4]:
        raise ValueError(
            f"Invalid number of points: {n_points}. " "Must be 1, 2, or 4."
        )

    if max_LED_separation is None and n_points != 1:
        raise ValueError(
            "max_LED_separation must be provided for "
            f"{n_points}-point centroid calculation"
        )

    if n_points > 1 and get_distance is None:
        raise ImportError(
            "position_tools is required for multi-point centroid calculation. "
            "Install with: pip install position-tools"
        )

    # Dispatch to appropriate function
    if n_points == 1:
        return get_1pt_centroid(pos_df, points)
    elif n_points == 2:
        return get_2pt_centroid(pos_df, points, max_LED_separation)
    else:  # n_points == 4
        return get_4pt_centroid(pos_df, points, max_LED_separation)


def get_1pt_centroid(
    pos_df: pd.DataFrame,
    points: Dict[str, str],
) -> np.ndarray:
    """Calculate 1-point centroid (simple passthrough).

    The centroid is simply the position of the single bodypart.
    Where the bodypart is NaN, the centroid is also NaN.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
    points : Dict[str, str]
        Dictionary with single entry: {"point1": "bodypart_name"}

    Returns
    -------
    np.ndarray
        Centroid positions, shape (n_frames, 2)

    Examples
    --------
    >>> centroid = get_1pt_centroid(
    ...     pos_df,
    ...     points={"point1": "nose"}
    ... )
    """
    point_name = points.get("point1")
    if point_name is None:
        raise ValueError("points dict must contain 'point1' key")

    # Extract x, y coordinates
    idx = pd.IndexSlice
    coords = pos_df.loc[:, idx[point_name, ["x", "y"]]].to_numpy()

    return coords


def get_2pt_centroid(
    pos_df: pd.DataFrame,
    points: Dict[str, str],
    max_LED_separation: float,
) -> np.ndarray:
    """Calculate 2-point centroid with fallback logic.

    Priority:
    1. If both points valid and close enough → average them
    2. If only point1 valid → use point1
    3. If only point2 valid → use point2
    4. Otherwise → NaN

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
    points : Dict[str, str]
        Dictionary: {"point1": "name1", "point2": "name2"}
    max_LED_separation : float
        Maximum allowed distance between points (pixels)

    Returns
    -------
    np.ndarray
        Centroid positions, shape (n_frames, 2)

    Examples
    --------
    >>> centroid = get_2pt_centroid(
    ...     pos_df,
    ...     points={"point1": "green_led", "point2": "red_led"},
    ...     max_LED_separation=15.0
    ... )
    """
    point1 = points.get("point1")
    point2 = points.get("point2")

    if point1 is None or point2 is None:
        raise ValueError("points dict must contain 'point1' and 'point2' keys")

    point_names = [point1, point2]
    idx = pd.IndexSlice

    # Extract coordinates for each point
    coords = {
        p: pos_df.loc[:, idx[p, ["x", "y"]]].to_numpy() for p in point_names
    }

    # Identify NaN frames for each point
    nans = {p: np.isnan(coord).any(axis=1) for p, coord in coords.items()}

    # Initialize centroid with NaN
    centroid = np.full((len(pos_df), 2), np.nan)

    # Case 1: Both points valid and close enough
    both_valid = ~nans[point1] & ~nans[point2]
    if np.any(both_valid):
        # Check distance constraint
        too_far = (
            get_distance(coords[point1], coords[point2]) >= max_LED_separation
        )
        valid_and_close = both_valid & ~too_far

        if np.any(valid_and_close):
            coord_arrays = np.array(
                [
                    coords[point1][valid_and_close],
                    coords[point2][valid_and_close],
                ]
            )
            centroid[valid_and_close] = np.nanmean(coord_arrays, axis=0)

    # Case 2: Only point1 valid
    only_point1 = ~nans[point1] & nans[point2]
    centroid[only_point1] = coords[point1][only_point1]

    # Case 3: Only point2 valid
    only_point2 = nans[point1] & ~nans[point2]
    centroid[only_point2] = coords[point2][only_point2]

    return centroid


def get_4pt_centroid(
    pos_df: pd.DataFrame,
    points: Dict[str, str],
    max_LED_separation: float,
) -> np.ndarray:
    """Calculate 4-point centroid for complex LED configuration.

    This implements the Frank Lab's 4-LED configuration logic:
    - greenLED: Front marker
    - redLED_C: Center back marker
    - redLED_L: Left back marker
    - redLED_R: Right back marker

    Priority (in order):
    1. If green and center valid → average them
    2. If green and left/right valid → average midpoint(left, right) and green
    3. If only left/right valid → average left and right
    4. If green and one side valid → average them
    5. If only center valid → use center
    6. Otherwise → NaN

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position data with MultiIndex columns: (bodypart, coord)
    points : Dict[str, str]
        Dictionary with keys: "greenLED", "redLED_C", "redLED_L", "redLED_R"
    max_LED_separation : float
        Maximum allowed distance between LEDs (pixels)

    Returns
    -------
    np.ndarray
        Centroid positions, shape (n_frames, 2)

    Examples
    --------
    >>> centroid = get_4pt_centroid(
    ...     pos_df,
    ...     points={
    ...         "greenLED": "green",
    ...         "redLED_C": "red_center",
    ...         "redLED_L": "red_left",
    ...         "redLED_R": "red_right"
    ...     },
    ...     max_LED_separation=15.0
    ... )

    Notes
    -----
    This function enforces distance constraints between all LED pairs
    to prevent using corrupted tracking data.
    """
    green = points.get("greenLED")
    red_C = points.get("redLED_C")
    red_L = points.get("redLED_L")
    red_R = points.get("redLED_R")

    if None in [green, red_C, red_L, red_R]:
        raise ValueError(
            "points dict must contain keys: "
            "'greenLED', 'redLED_C', 'redLED_L', 'redLED_R'"
        )

    point_names = [green, red_C, red_L, red_R]
    idx = pd.IndexSlice

    # Extract coordinates for each point
    coords = {
        p: pos_df.loc[:, idx[p, ["x", "y"]]].to_numpy() for p in point_names
    }

    # Identify NaN frames for each point
    nans = {p: np.isnan(coord).any(axis=1) for p, coord in coords.items()}

    # Initialize centroid with NaN
    centroid = np.full((len(pos_df), 2), np.nan)

    # Helper function to check distance constraint
    def _check_distance_pairs(
        point_list: List[str], mask: np.ndarray
    ) -> np.ndarray:
        """Check all pairs in point_list satisfy distance constraint."""
        valid_mask = mask.copy()
        for p1, p2 in combinations(point_list, 2):
            too_far = get_distance(coords[p1], coords[p2]) >= max_LED_separation
            valid_mask = valid_mask & ~too_far
        return valid_mask

    # Helper function to calculate centroid for specific points and mask
    def _set_centroid(
        point_list: List[str],
        mask: np.ndarray,
        use_midpoint: Optional[Tuple[str, str]] = None,
    ):
        """Set centroid values for frames matching mask."""
        if not np.any(mask):
            return

        # Check distance constraints
        valid_mask = _check_distance_pairs(point_list, mask)
        if not np.any(valid_mask):
            return

        # Calculate centroid
        if use_midpoint is not None:
            # Special case: average of midpoint and another point
            p1, p2 = use_midpoint
            midpoint = (coords[p1][valid_mask] + coords[p2][valid_mask]) / 2
            other_point = [p for p in point_list if p not in use_midpoint][0]
            coord_arrays = np.array([midpoint, coords[other_point][valid_mask]])
        else:
            coord_arrays = np.array([coords[p][valid_mask] for p in point_list])

        centroid[valid_mask] = np.nanmean(coord_arrays, axis=0)

    # Priority 1: Green and center valid
    _set_centroid(
        [green, red_C],
        ~nans[green] & ~nans[red_C],
    )

    # Priority 2: Green and left/right valid (use midpoint of left/right)
    _set_centroid(
        [green, red_L, red_R],
        ~nans[green] & nans[red_C] & ~nans[red_L] & ~nans[red_R],
        use_midpoint=(red_L, red_R),
    )

    # Priority 3: Only left/right valid
    _set_centroid(
        [red_L, red_R],
        nans[green] & nans[red_C] & ~nans[red_L] & ~nans[red_R],
    )

    # Priority 4: Green and one side valid
    for side, other in [(red_L, red_R), (red_R, red_L)]:
        _set_centroid(
            [green, side],
            ~nans[green] & nans[red_C] & ~nans[side] & nans[other],
        )

    # Priority 5: Only center valid
    only_center = nans[green] & ~nans[red_C]
    centroid[only_center] = coords[red_C][only_center]

    return centroid
