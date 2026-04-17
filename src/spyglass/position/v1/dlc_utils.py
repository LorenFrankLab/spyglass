# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
import grp
import logging
import os
import pwd
import subprocess
import sys
from collections.abc import Sequence
from itertools import combinations, groupby
from operator import itemgetter
from pathlib import Path, PosixPath
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from position_tools import get_distance

from spyglass.common.common_behav import VideoFile
from spyglass.common.common_usage import ActivityLog

# Import validation functions from utils module
from spyglass.position.utils.validation import validate_list, validate_option
from spyglass.settings import dlc_output_dir, dlc_video_dir, raw_dir, test_mode
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


def file_log(logger, console=False):
    """Decorator to add a file handler to a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger to add file handler to.
    console : bool, optional
        If True, logged info will also be printed to console. Default False.

    Example
    -------
    @file_log(logger, console=True)
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
            logger.addHandler(file_handler)
            if not console:
                logger.removeHandler(logger.handlers[0])
            try:
                return func(self, *args, **kwargs)
            finally:
                if not console:
                    logger.addHandler(stream_handler)
                logger.removeHandler(file_handler)
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


def infer_output_dir(key, makedir=True):
    """Return the expected pose_estimation_output_dir.

    Parameters
    ----------
    key: DataJoint key specifying a pairing of VideoFile and Model.
    """

    file_name = key.get("nwb_file_name")
    dlc_model_name = key.get("dlc_model_name")
    epoch = key.get("epoch")

    if not all([file_name, dlc_model_name, epoch]):
        raise ValueError(
            "Key must contain 'nwb_file_name', 'dlc_model_name', and 'epoch'"
        )

    nwb_file_name = key["nwb_file_name"].split("_.")[0]
    output_dir = Path(dlc_output_dir) / Path(
        f"{nwb_file_name}/{nwb_file_name}_{key['epoch']:02}"
        f"_model_" + key["dlc_model_name"].replace(" ", "-")
    )
    if makedir:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_Path(path):
    """
    Convert the input "path" into a Path object
    Handles one odd Windows/Linux incompatibility of the "\\"
    """
    return Path(str(path).replace("\\", "/"))


def get_video_info(key):
    """Returns video path for a given key.

    Given nwb_file_name and interval_list_name returns specified
    video file filename, path, meters_per_pixel, and timestamps.

    Parameters
    ----------
    key : dict
        Dictionary containing nwb_file_name and interval_list_name as keys

    Returns
    -------
    video_filepath : str
        path to the video file, including video filename
    video_filename : str
        filename of the video
    meters_per_pixel : float
        meters per pixel conversion factor
    timestamps : np.array
        timestamps of the video
    """
    import pynwb

    vf_key = {k: val for k, val in key.items() if k in VideoFile.heading}
    video_query = VideoFile & vf_key

    if not video_query:
        VideoFile().make(vf_key, verbose=False)

    if len(video_query) != 1:
        logger.warning(f"Found {len(video_query)} videos for {vf_key}")
        return None, None, None, None

    video_info = video_query.fetch1()
    nwb_path = f"{raw_dir}/{video_info['nwb_file_name']}"

    with pynwb.NWBHDF5IO(path=nwb_path, mode="r") as in_out:
        nwb_file = in_out.read()
        nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
        try:
            video_filepath = VideoFile.get_abs_path(vf_key)
        except FileNotFoundError as e:
            logger.warning(f"Video file not found, skipping: {e}")
            return None, None, None, None
        video_dir = os.path.dirname(video_filepath) + "/"
        video_filename = video_filepath.split(video_dir)[-1]
        meters_per_pixel = nwb_video.device.meters_per_pixel
        timestamps = np.asarray(nwb_video.timestamps)

    return video_dir, video_filename, meters_per_pixel, timestamps


def find_mp4(
    video_path: Union[str, PosixPath],
    output_path: Union[str, PosixPath] = dlc_video_dir,
    video_filename: str = None,
    video_filetype: str = "h264",
):
    """Check for video file and convert to .mp4 if necessary.

    Parameters
    ----------
    video_path : str or PosixPath object
        path to directory of the existing video file without filename
    output_path : str or PosixPath object
        path to directory where converted video will be saved
    video_filename : str, Optional
        filename of the video to convert, if not provided, video_filetype must
        be and all video files of video_filetype in the directory will be
        converted
    video_filetype : str or List, Default 'h264', Optional
        If video_filename is not provided,
        all videos of this filetype will be converted to .mp4

    Returns
    -------
    PosixPath object
        path to converted video file
    """

    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    video_files = (
        [Path(video_path) / video_filename]
        if video_filename
        else [p for p in Path(video_path).glob(f"*.{video_filetype}")]
    )

    if len(video_files) != 1:
        raise FileNotFoundError(
            f"Found {len(video_files)} video files: {video_files}"
        )
    video_filepath = video_files[0]

    if video_filepath.exists() and video_filepath.suffix == ".mp4":
        return video_filepath

    video_file = (
        video_filepath.as_posix()
        .rsplit(video_filepath.parent.as_posix(), maxsplit=1)[-1]
        .split("/")[-1]
    )
    return _convert_mp4(
        video_file, video_path, output_path, videotype="mp4", count_frames=True
    )


def _convert_mp4(
    filename: str,
    video_path: str,
    dest_path: str,
    videotype: str = "mp4",
    count_frames=False,
    return_output=True,
):
    """converts video to mp4 using passthrough for frames
    Parameters
    ----------
    filename: str
        filename of video including suffix (e.g. .h264)
    video_path: str
        path of input video excluding filename
    dest_path: str
        path of output video excluding filename, but no suffix (e.g. no '.mp4')
    videotype: str
        str of filetype to convert to (currently only accepts 'mp4')
    count_frames: bool
        if True counts the number of frames in both videos instead of packets
    return_output: bool
        if True returns the destination filename
    """
    if videotype not in ["mp4"]:
        raise NotImplementedError(f"videotype {videotype} not implemented")

    orig_filename = filename
    video_path = Path(video_path) / filename

    dest_filename = os.path.splitext(filename)[0]
    if ".1" in dest_filename:
        dest_filename = os.path.splitext(dest_filename)[0]
    dest_path = Path(f"{dest_path}/{dest_filename}.{videotype}")
    if dest_path.exists():
        logger.info(f"{dest_path} already exists, skipping conversion")
        return dest_path

    try:
        sys.stdout.flush()
        convert_process = subprocess.Popen(
            [
                "ffmpeg",
                "-vsync",
                "passthrough",
                "-i",
                f"{video_path.as_posix()}",
                "-codec",
                "copy",
                f"{dest_path.as_posix()}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as err:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            f"Video convert errored: Code {err.returncode}, {err.output}"
        ) from err
    out, _ = convert_process.communicate()
    logger.info(f"Finished converting {filename}")

    # check packets match orig file
    logger.info(f"Checking packets match orig file: {dest_filename}")
    orig_packets = _check_packets(video_path, count_frames=count_frames)
    dest_packets = _check_packets(dest_path, count_frames=count_frames)
    if orig_packets != dest_packets:  # pragma: no cover
        logger.warning(  # pragma: no cover
            f"Conversion error: {orig_filename} -> {dest_filename}"
        )

    if return_output:
        return dest_path


def _check_packets(file, count_frames=False):
    checked = "frames" if count_frames else "packets"
    try:
        check_process = subprocess.Popen(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                f"-count_{checked}",
                "-show_entries",
                f"stream=nb_read_{checked}",
                "-of",
                "csv=p=0",
                file.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as err:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            f"Check packets error: Code {err.returncode}, {err.output}"
        ) from err
    out, _ = check_process.communicate()
    decoded_out = out.decode("utf-8").split("\n")[0]
    if decoded_out.isnumeric():
        return int(decoded_out)
    raise ValueError(f"Check packets error: {out}")


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
    This is a compatibility wrapper around the newer interp_position function.
    New code should use interp_position directly for better performance and
    more flexible column specification.
    """
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
    """
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
