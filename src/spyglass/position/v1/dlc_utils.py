# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
import grp
import logging
import os
import pwd
import subprocess
from collections.abc import Sequence
from itertools import combinations, groupby
from operator import itemgetter
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
from position_tools import get_distance

from spyglass.common.common_usage import ActivityLog

# Import validation functions from utils module
from spyglass.position.utils.validation import validate_list, validate_option
from spyglass.settings import pose_output_dir, test_mode
from spyglass.utils.logging import logger, stream_handler

# validate_smooth_params functionality moved to utils.validation module


def _set_permissions(directory, mode, username: str, groupname: str = None):
    """
    Use to recursively set ownership and permissions for
    directories/files that result from the DLC pipeline

    Parameters
    ----------
    directory : str or PosixPath
        path to target directory
    mode : stat read-out
        list of permissions to set using stat package
        (e.g. mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    username : str
        username of user to set as owner and apply permissions to
    groupname : str
        existing Linux groupname to set as group owner
        and apply permissions to

    Returns
    -------
    None
    """
    ActivityLog().deprecate_log(  # pragma: no cover
        "dlc_utils: _set_permissions"
    )

    directory = Path(directory)  # pragma: no cover
    assert directory.exists(), f"Target directory: {directory} does not exist"
    uid = pwd.getpwnam(username).pw_uid  # pragma: no cover
    if groupname:  # pragma: no cover
        gid = grp.getgrnam(groupname).gr_gid
    else:  # pragma: no cover
        gid = None
    for dirpath, _, filenames in os.walk(directory):  # pragma: no cover
        os.chown(dirpath, uid, gid)
        os.chmod(dirpath, mode)
        for filename in filenames:
            os.chown(os.path.join(dirpath, filename), uid, gid)
            os.chmod(os.path.join(dirpath, filename), mode)


def file_log(this_log, console=False):
    """Decorator to add a file handler to a logger.

    Parameters
    ----------
    this_log : logging.Logger
        Logger to add file handler to.
    console : bool, optional
        If True, logged info will also be printed to console. Default False.

    Example
    -------
    @file_log(this_log, console=True)
    def func(self, *args, **kwargs):
        pass
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not (log_path := getattr(self, "log_path", None)):
                log_path = f"temp_{self.__class__.__name__}.log"
            file_handler = logging.FileHandler(log_path, mode="a")
            file_fmt = logging.Formatter(
                "[%(asctime)s][%(levelname)s] Spyglass "
                + "%(filename)s:%(lineno)d: %(message)s",
                datefmt="%y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            this_log.addHandler(file_handler)
            if not console:
                this_log.removeHandler(logger.handlers[0])
            try:
                return func(self, *args, **kwargs)
            finally:
                if not console:
                    this_log.addHandler(stream_handler)
                this_log.removeHandler(file_handler)
                file_handler.close()

        return wrapper

    return decorator


def get_dlc_root_data_dir():  # pragma: no cover
    """Returns list of potential root directories for DLC data"""
    ActivityLog().deprecate_log("dlc_utils: get_dlc_root_data_dir")
    if "custom" in dj.config:
        if "dlc_root_data_dir" in dj.config["custom"]:
            dlc_root_dirs = dj.config.get("custom", {}).get("dlc_root_data_dir")
    if not dlc_root_dirs:
        return [
            "/nimbus/deeplabcut/projects/",
            "/nimbus/deeplabcut/output/",
            "/cumulus/deeplabcut/",
        ]
    elif not isinstance(dlc_root_dirs, Sequence):
        return list(dlc_root_dirs)
    else:
        return dlc_root_dirs


def get_dlc_processed_data_dir() -> str:  # pragma: no cover
    """Returns session_dir relative to custom 'dlc_output_dir' root"""
    ActivityLog().deprecate_log("dlc_utils: get_dlc_processed_data_dir")
    if "custom" in dj.config:
        if "dlc_output_dir" in dj.config["custom"]:
            dlc_output_dir = dj.config.get("custom", {}).get("dlc_output_dir")
    if dlc_output_dir:
        return Path(dlc_output_dir)
    else:
        return Path("/nimbus/deeplabcut/output/")


def find_full_path(root_directories, relative_path):  # pragma: no cover
    """
    from Datajoint Elements - unused
    Given a relative path, search and return the full-path
     from provided potential root directories (in the given order)
        :param root_directories: potential root directories
        :param relative_path: the relative path to find the valid root directory
        :return: full-path (Path object)
    """
    ActivityLog().deprecate_log("dlc_utils: find_full_path")
    relative_path = _to_Path(relative_path)

    if relative_path.exists():
        return relative_path

    # Turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, Path)):
        root_directories = [_to_Path(root_directories)]

    for root_dir in root_directories:
        if (_to_Path(root_dir) / relative_path).exists():
            return _to_Path(root_dir) / relative_path

    raise FileNotFoundError(
        f"No valid full-path found (from {root_directories})"
        f" for {relative_path}"
    )


def find_root_directory(root_directories, full_path):  # pragma: no cover
    """
    From datajoint elements - unused
    Given multiple potential root directories and a full-path,
    search and return one directory that is the parent of the given path
        :param root_directories: potential root directories
        :param full_path: the full path to search the root directory
        :return: root_directory (Path object)
    """
    ActivityLog().deprecate_log("dlc_utils: find_full_path")
    full_path = _to_Path(full_path)

    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} does not exist!")

    # Turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, Path)):
        root_directories = [_to_Path(root_directories)]

    try:
        return next(
            _to_Path(root_dir)
            for root_dir in root_directories
            if _to_Path(root_dir) in set(full_path.parents)
        )

    except StopIteration as exc:
        raise FileNotFoundError(
            f"No valid root directory found (from {root_directories})"
            f" for {full_path}"
        ) from exc


def _to_Path(path):
    """
    Convert the input "path" into a Path object
    Handles one odd Windows/Linux incompatibility of the "\\"
    """
    return Path(str(path).replace("\\", "/"))


def get_gpu_memory():  # pragma: no cover
    """Queries the gpu cluster and returns the memory use for each core.
    This is used to evaluate which GPU cores are available to run jobs on
    (i.e. pose estimation, DLC model training)

    Returns
    -------
    memory_use_values : dict
        dictionary with core number as key and memory in use as value.

    Raises
    ------
    RuntimeError
        if subproccess command errors.
    """

    def output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    query_cmd = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(
            subprocess.check_output(query_cmd.split(), stderr=subprocess.STDOUT)
        )[1:]
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Get GPU memory errored: Code {err.returncode}, {err.output}"
        ) from err
    memory_use_values = {
        i: int(x.split()[0]) for i, x in enumerate(memory_use_info)
    }
    return memory_use_values


def get_span_start_stop(indices):
    """Get start and stop indices of spans of consecutive indices"""
    span_inds = []
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        span_inds.append((group[0], group[-1]))
    return span_inds


# Video/path helpers moved to utils.general — re-export for backward compat
from spyglass.position.utils.general import (
    _check_packets,  # noqa: F401
    _convert_mp4,
    find_mp4,
    get_video_info,
    infer_output_dir,
)

# smooth_moving_avg moved to utils.interpolation module
# Use compatibility wrapper below for V1 API
from spyglass.position.utils.interpolation import smooth_moving_avg

# Import function mapping for backward compatibility
_key_to_smooth_func_dict = {
    "moving_avg": smooth_moving_avg,
}


# Centroid class moved to utils.centroid module for reusability
# Import for V1 backward compatibility
from spyglass.position.utils.centroid import Centroid

# orientation functions moved to utils.orientation module
# Import them for V1 compatibility
from spyglass.position.utils.interpolation import interp_position
from spyglass.position.utils.orientation import (
    bisector_orientation,
    interp_orientation,
    no_orientation,
    two_pt_orientation,
)

# Add new functions for orientation calculation here


# === V1 Compatibility Layer ===
# These functions maintain V1's legacy API while using shared utils internally


def interp_pos(dlc_df: pd.DataFrame, spans_to_interp, **kwargs) -> pd.DataFrame:
    """V1-compatible wrapper for interp_position function.

    This function provides backward compatibility for V1 code that expects
    the old interp_pos signature and behavior.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        Position dataframe with x,y columns
    spans_to_interp : list of tuples
        List of (start, stop) index pairs for interpolation spans
    **kwargs
        Additional interpolation parameters

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated positions

    Notes
    -----
    Deprecated. Use ``spyglass.position.utils.interpolation.interp_position``
    directly. This alias will be removed in a future release.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interp_pos",
        alt="spyglass.position.utils.interpolation.interp_position",
    )
    # Use the newer function with default x,y columns
    return interp_position(
        pos_df=dlc_df,
        spans_to_interp=spans_to_interp,
        coord_cols=("x", "y"),
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["max_pts_to_interp", "max_cm_to_interp"]
        },
    )


def smooth_moving_avg_v1(
    interp_df: pd.DataFrame,
    smoothing_duration: float,
    sampling_rate: int,
    **kwargs,
) -> pd.DataFrame:
    """V1-compatible moving average smoothing function.

    This function provides backward compatibility for V1 code that expects
    specific MultiIndex column handling.

    Parameters
    ----------
    interp_df : pd.DataFrame
        Position dataframe with MultiIndex columns for x,y coordinates
    smoothing_duration : float
        Duration of smoothing window in seconds
    sampling_rate : int
        Sampling rate in Hz
    **kwargs
        Additional arguments (for compatibility)

    Returns
    -------
    pd.DataFrame
        Smoothed position dataframe

    Examples
    --------
    >>> smoothed_df = smooth_moving_avg_v1(pos_df, 0.5, 30)  # 0.5s window at 30Hz

    Notes
    -----
    Deprecated. Use ``spyglass.position.utils.interpolation.smooth_moving_avg``
    directly. This alias will be removed in a future release.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "smooth_moving_avg_v1",
        alt="spyglass.position.utils.interpolation.smooth_moving_avg",
    )
    try:
        import bottleneck as bn
    except ImportError:
        raise ImportError(
            "bottleneck is required for moving average smoothing. "
            "Install with: pip install bottleneck"
        )

    idx = pd.IndexSlice
    moving_avg_window = int(np.round(smoothing_duration * sampling_rate))

    xy_arr = interp_df.loc[:, idx[("x", "y")]].values
    smoothed_xy_arr = bn.move_mean(
        xy_arr, window=moving_avg_window, axis=0, min_count=1
    )
    interp_df.loc[:, idx["x"]], interp_df.loc[:, idx["y"]] = [
        *zip(*smoothed_xy_arr.tolist())
    ]
    return interp_df


def two_pt_head_orientation(pos_df: pd.DataFrame, **params):
    """V1-compatible two-point orientation calculation.

    This function provides backward compatibility for V1 code that uses
    params.pop() to extract bodypart names.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position dataframe with MultiIndex columns
    **params
        V1-style parameters with bodypart1 and bodypart2 keys

    Returns
    -------
    np.ndarray
        Orientation in radians
    """

    BP1 = params.pop("bodypart1", None)
    BP2 = params.pop("bodypart2", None)

    if BP1 is None or BP2 is None:
        raise ValueError("bodypart1 and bodypart2 must be specified")

    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "two_pt_head_orientation",
        alt="spyglass.position.utils.orientation.two_pt_orientation",
    )
    # Use the modern function with proper parameter mapping
    return two_pt_orientation(pos_df, point1=BP1, point2=BP2)


def red_led_bisector_orientation(pos_df: pd.DataFrame, **params):
    """V1-compatible bisector orientation calculation.

    Determines orientation based on 2 equally-spaced identifiers
    (red/green LEDs). Identifiers are assumed to be perpendicular to the
    orientation direction. A third object is needed to determine forward/backward.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Position dataframe with MultiIndex columns
    **params
        V1-style parameters with led1, led2, and led3 keys

    Returns
    -------
    np.ndarray
        Orientation in radians
    """
    LED1 = params.pop("led1", None)
    LED2 = params.pop("led2", None)
    LED3 = params.pop("led3", None)

    if any(x is None for x in [LED1, LED2, LED3]):
        raise ValueError("led1, led2, and led3 must be specified")

    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "red_led_bisector_orientation",
        alt="spyglass.position.utils.orientation.bisector_orientation",
    )
    # Use the modern bisector function with proper parameter mapping
    return bisector_orientation(pos_df, led1=LED1, led2=LED2, led3=LED3)


# V1 compatibility constants
_key_to_smooth_func_dict = {
    "moving_avg": smooth_moving_avg_v1,
}


# Orientation function lookup dictionary
_key_to_func_dict = {
    "none": no_orientation,
    "red_green_orientation": two_pt_head_orientation,
    "red_led_bisector": red_led_bisector_orientation,
}
