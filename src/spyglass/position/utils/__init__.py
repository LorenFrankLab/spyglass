"""Shared utilities for position estimation across V1 and V2.

This module contains tool-agnostic algorithms for pose processing:
- Orientation calculation (from 2-3 bodyparts)
- Centroid calculation (from 1-4 bodyparts)
- Interpolation and smoothing
- Video generation with pose overlay

These utilities are used by both V1 (DLC-specific) and V2 (multi-tool) pipelines.
"""

from .centroid import (
    calculate_centroid,
    get_1pt_centroid,
    get_2pt_centroid,
    get_4pt_centroid,
)
from .dlc_io import (
    DLCProjectReader,
    do_pose_estimation,
    get_dlc_bodyparts,
    get_dlc_model_eval,
    get_dlc_scorer,
    get_most_recent_file,
    parse_dlc_h5_output,
    read_yaml,
    reformat_dlc_data,
    save_yaml,
    suppress_print_from_package,
    test_mode_suppress,
    validate_dlc_file,
)
from .general import get_param_names, sanitize_filename
from .interpolation import (
    SMOOTHING_METHODS,
    get_smoothing_function,
    interp_position,
    smooth_gaussian,
    smooth_moving_avg,
    smooth_savgol,
)
from .make_video import VideoMaker, make_video
from .orientation import (
    bisector_orientation,
    get_span_start_stop,
    interp_orientation,
    no_orientation,
    smooth_orientation,
    two_pt_orientation,
)
from .tool_strategies import (
    DLCStrategy,
    NDXPoseStrategy,
    PoseToolStrategy,
    SLEAPStrategy,
    ToolStrategyFactory,
)
from .validation import (
    validate_centroid_params,
    validate_interpolation_params,
    validate_option,
    validate_orientation_params,
    validate_required_keys,
    validate_smoothing_params,
)

__all__ = [
    # Centroid
    "calculate_centroid",
    "get_1pt_centroid",
    "get_2pt_centroid",
    "get_4pt_centroid",
    # DLC I/O
    "DLCProjectReader",
    "do_pose_estimation",
    "get_dlc_bodyparts",
    "get_dlc_model_eval",
    "get_dlc_scorer",
    "get_most_recent_file",
    "parse_dlc_h5_output",
    "read_yaml",
    "reformat_dlc_data",
    "save_yaml",
    "suppress_print_from_package",
    "test_mode_suppress",
    "validate_dlc_file",
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
    "get_span_start_stop",
    # Tool Strategies
    "PoseToolStrategy",
    "DLCStrategy",
    "SLEAPStrategy",
    "NDXPoseStrategy",
    "ToolStrategyFactory",
    # Validation
    "validate_option",
    "validate_required_keys",
    "validate_smoothing_params",
    "validate_orientation_params",
    "validate_centroid_params",
    "validate_interpolation_params",
    # General utilities
    "get_most_recent_file",
    "get_param_names",
    "sanitize_filename",
    # Video generation
    "VideoMaker",
    "make_video",
]
