"""Shared utilities for position estimation across V1 and V2.

This module contains tool-agnostic algorithms for pose processing:
- Orientation calculation (from 2-3 bodyparts)
- Centroid calculation (from 1-4 bodyparts)
- Interpolation and smoothing

These utilities are used by both V1 (DLC-specific) and V2 (multi-tool) pipelines.
"""

from .centroid import (
    calculate_centroid,
    get_1pt_centroid,
    get_2pt_centroid,
    get_4pt_centroid,
)
from .general import get_most_recent_file, get_param_names, sanitize_filename
from .interpolation import (
    SMOOTHING_METHODS,
    get_smoothing_function,
    interp_position,
    smooth_gaussian,
    smooth_moving_avg,
    smooth_savgol,
)
from .orientation import (
    bisector_orientation,
    interp_orientation,
    no_orientation,
    smooth_orientation,
    two_pt_orientation,
)

__all__ = [
    # Centroid
    "calculate_centroid",
    "get_1pt_centroid",
    "get_2pt_centroid",
    "get_4pt_centroid",
    # Interpolation & Smoothing
    "interp_position",
    "smooth_moving_avg",
    "smooth_savgol",
    "smooth_gaussian",
    "get_smoothing_function",
    "SMOOTHING_METHODS",
    # Orientation
    "two_pt_orientation",
    "bisector_orientation",
    "no_orientation",
    "interp_orientation",
    "smooth_orientation",
    # General utilities
    "get_most_recent_file",
    "get_param_names",
    "sanitize_filename",
]
