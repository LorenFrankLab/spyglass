# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
import grp
import logging
import os
import pwd
import subprocess
import sys
from collections.abc import Sequence
from functools import reduce
from itertools import combinations, groupby
from operator import itemgetter
from pathlib import Path, PosixPath
from typing import Iterable, Union

import datajoint as dj
import numpy as np
import pandas as pd
from position_tools import get_distance

from spyglass.common.common_behav import VideoFile
from spyglass.common.common_usage import ActivityLog
from spyglass.settings import dlc_output_dir, dlc_video_dir, raw_dir
from spyglass.utils.logging import logger, stream_handler


def validate_option(
    option=None,
    options: list = None,
    name="option",
    types: tuple = None,
    val_range: tuple = None,
    permit_none=False,
):
    """Validate that option is in a list options or a list of types.

    Parameters
    ----------
    option : str, optional
        If none, runs no checks.
    options : lis, optional
        If provided, option must be in options.
    name : st, optional
        If provided, name of option to use in error message.
    types : tuple, optional
        If provided, option must be an instance of one of the types in types.
    val_range : tuple, optional
        If provided, option must be in range (min, max)
    permit_none : bool, optional
        If True, permit option to be None. Default False.

    Raises
    ------
    ValueError
        If option is not in options.
    """
    if option is None and not permit_none:
        raise ValueError(f"{name} cannot be None")

    if options and option not in options:
        raise KeyError(
            f"Unknown {name}: {option} " f"Available options: {options}"
        )

    if types is not None and not isinstance(types, Iterable):
        types = (types,)

    if types is not None and not isinstance(option, types):
        raise TypeError(f"{name} is {type(option)}. Available types {types}")

    if val_range and not (val_range[0] <= option <= val_range[1]):
        raise ValueError(f"{name} must be in range {val_range}")


def validate_list(
    required_items: list,
    option_list: list = None,
    name="List",
    condition="",
    permit_none=False,
):
    """Validate that option_list contains all items in required_items.

    Parameters
    ---------
    required_items : list
    option_list : list, optional
        If provided, option_list must contain all items in required_items.
    name : str, optional
        If provided, name of option_list to use in error message.
    condition : str, optional
        If provided, condition in error message as 'when using X'.
    permit_none : bool, optional
        If True, permit option_list to be None. Default False.
    """
    if option_list is None:
        if permit_none:
            return
        else:
            raise ValueError(f"{name} cannot be None")
    if condition:
        condition = f" when using {condition}"
    if any(x not in required_items for x in option_list):
        raise KeyError(
            f"{name} must contain all items in {required_items}{condition}."
        )


def validate_smooth_params(params):
    """If params['smooth'], validate method is in list and duration type"""
    if not params.get("smooth"):
        return
    smoothing_params = params.get("smoothing_params")
    validate_option(option=smoothing_params, name="smoothing_params")
    validate_option(
        option=smoothing_params.get("smooth_method"),
        name="smooth_method",
        options=_key_to_smooth_func_dict,
    )
    validate_option(
        option=smoothing_params.get("smoothing_duration"),
        name="smoothing_duration",
        types=(int, float),
    )


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
        video_filepath = VideoFile.get_abs_path(vf_key)
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


def interp_pos(dlc_df, spans_to_interp, **kwargs):
    """Interpolate x and y positions in DLC dataframe"""
    idx = pd.IndexSlice

    no_x_msg = "Index {ind} has no {coord}point with which to interpolate"
    no_interp_msg = "Index {start} to {stop} not interpolated"
    max_pts_to_interp = kwargs.get("max_pts_to_interp", float("inf"))
    max_cm_to_interp = kwargs.get("max_cm_to_interp", float("inf"))

    def _get_new_dim(dim, span_start, span_stop, start_time, stop_time):
        return np.interp(
            x=dlc_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[dim[0], dim[-1]],
        )

    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        idx_span = idx[span_start:span_stop]

        if (span_stop + 1) >= len(dlc_df):
            dlc_df.loc[idx_span, idx[["x", "y"]]] = np.nan
            logger.info(no_x_msg.format(ind=ind, coord="end"))
            continue
        if span_start < 1:
            dlc_df.loc[idx_span, idx[["x", "y"]]] = np.nan
            logger.info(no_x_msg.format(ind=ind, coord="start"))
            continue

        x = [dlc_df["x"].iloc[span_start - 1], dlc_df["x"].iloc[span_stop + 1]]
        y = [dlc_df["y"].iloc[span_start - 1], dlc_df["y"].iloc[span_stop + 1]]

        span_len = int(span_stop - span_start + 1)
        start_time = dlc_df.index[span_start]
        stop_time = dlc_df.index[span_stop]
        change = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))

        if span_len > max_pts_to_interp or change > max_cm_to_interp:
            dlc_df.loc[idx_span, idx[["x", "y"]]] = np.nan
            logger.info(no_interp_msg.format(start=span_start, stop=span_stop))
            if change > max_cm_to_interp:
                continue

        xnew = _get_new_dim(x, span_start, span_stop, start_time, stop_time)
        ynew = _get_new_dim(y, span_start, span_stop, start_time, stop_time)

        dlc_df.loc[idx[start_time:stop_time], idx["x"]] = xnew
        dlc_df.loc[idx[start_time:stop_time], idx["y"]] = ynew

    return dlc_df


def interp_orientation(df, spans_to_interp, **kwargs):
    """Interpolate orientation in DLC dataframe"""
    idx = pd.IndexSlice
    no_x_msg = "Index {ind} has no {x}point with which to interpolate"
    df_orient = df["orientation"]

    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        idx_span = idx[span_start:span_stop]
        if (span_stop + 1) >= len(df):
            df.loc[idx_span, idx["orientation"]] = np.nan
            logger.info(no_x_msg.format(ind=ind, x="stop"))
            continue
        if span_start < 1:
            df.loc[idx_span, idx["orientation"]] = np.nan
            logger.info(no_x_msg.format(ind=ind, x="start"))
            continue

        orient = [df_orient.iloc[span_start - 1], df_orient.iloc[span_stop + 1]]

        start_time = df.index[span_start]
        stop_time = df.index[span_stop]
        orientnew = np.interp(
            x=df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[orient[0], orient[-1]],
        )
        df.loc[idx[start_time:stop_time], idx["orientation"]] = orientnew
    return df


def smooth_moving_avg(
    interp_df, smoothing_duration: float, sampling_rate: int, **kwargs
):
    """Smooths x and y positions in DLC dataframe"""
    import bottleneck as bn

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


_key_to_smooth_func_dict = {
    "moving_avg": smooth_moving_avg,
}


def two_pt_head_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on vector between two points"""
    BP1 = params.pop("bodypart1", None)
    BP2 = params.pop("bodypart2", None)
    orientation = np.arctan2(
        (pos_df[BP1]["y"] - pos_df[BP2]["y"]),
        (pos_df[BP1]["x"] - pos_df[BP2]["x"]),
    )
    return orientation


def no_orientation(pos_df: pd.DataFrame, **params):
    """Returns an array of NaNs for orientation"""
    fill_value = params.pop("fill_with", np.nan)
    n_frames = len(pos_df)
    orientation = np.full(
        shape=(n_frames), fill_value=fill_value, dtype=np.float16
    )
    return orientation


def red_led_bisector_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on 2 equally-spaced identifiers

    Identifiers are assumed to be perpendicular to the orientation direction.
    A third object is needed to determine forward/backward
    """  # timeit reported 3500x improvement for vectorized implementation
    LED1 = params.pop("led1", None)
    LED2 = params.pop("led2", None)
    LED3 = params.pop("led3", None)

    orient = np.full(len(pos_df), np.nan)  # Initialize with NaNs
    x_vec = pos_df[LED1]["x"] - pos_df[LED2]["x"]
    y_vec = pos_df[LED1]["y"] - pos_df[LED2]["y"]
    y_eq0 = np.isclose(y_vec, 0)

    # when y_vec is zero, 1&2 are equal. Compare to 3, determine if up or down
    orient[y_eq0 & pos_df[LED3]["y"].gt(pos_df[LED1]["y"])] = np.pi / 2
    orient[y_eq0 & pos_df[LED3]["y"].lt(pos_df[LED1]["y"])] = -np.pi / 2

    # Handling error case where y_vec is zero and all Ys are the same
    y_1, y_2, y_3 = pos_df[LED1]["y"], pos_df[LED2]["y"], pos_df[LED3]["y"]
    if np.any(y_eq0 & np.isclose(y_1, y_2) & np.isclose(y_2, y_3)):
        raise Exception(  # pragma: no cover
            "Cannot determine head direction from bisector"
        )

    # General case where y_vec is not zero. Use arctan2 to determine orientation
    length = np.sqrt(x_vec**2 + y_vec**2)
    norm_x = (-y_vec / length)[~y_eq0]
    norm_y = (x_vec / length)[~y_eq0]
    orient[~y_eq0] = np.arctan2(norm_y, norm_x)

    return orient


# Add new functions for orientation calculation here

_key_to_func_dict = {
    "none": no_orientation,
    "red_green_orientation": two_pt_head_orientation,
    "red_led_bisector": red_led_bisector_orientation,
}


class Centroid:
    def __init__(self, pos_df, points, max_LED_separation=None):
        if max_LED_separation is None and len(points) != 1:
            raise ValueError("max_LED_separation must be provided")
        if len(points) not in [1, 2, 4]:
            raise ValueError("Invalid number of points")

        self.pos_df = pos_df
        self.max_LED_separation = max_LED_separation
        self.points_dict = points
        self.point_names = list(points.values())
        self.idx = pd.IndexSlice
        self.centroid = np.zeros(shape=(len(pos_df), 2))
        self.coords = {
            p: pos_df.loc[:, self.idx[p, ("x", "y")]].to_numpy()
            for p in self.point_names
        }
        self.nans = {
            p: np.isnan(coord).any(axis=1) for p, coord in self.coords.items()
        }

        if len(points) == 1:
            self.get_1pt_centroid()
            return
        if len(points) in [2, 4]:  # 4 also requires 2
            self.get_2pt_centroid()
        if len(points) == 4:
            self.get_4pt_centroid()

    def calc_centroid(
        self,
        mask: tuple,
        points: list = None,
        replace: bool = False,
        midpoint: bool = False,
        logical_or: bool = False,
    ):
        """Calculate the centroid of the points in the mask

        Parameters
        ----------
        mask : Union[tuple, list]
            Tuple of masks to apply to the points. Default is np.logical_and
            over a tuple. If a list is passed, then np.logical_or is used.
            List cannoot be used with logical_or=True
        points : list, optional
            List of points to calculate the centroid of. For replace, not needed
        replace : bool, optional
            Special case for replacing mask with nans, by default False
        logical_or : bool, optional
            Whether to use logical_and or logical_or to combine mask tuple.
        """
        if isinstance(mask, list):  # pragma: no cover
            mask = [reduce(np.logical_and, m) for m in mask]

        # Check that combinations of points close enough
        if points is not None and len(points) > 1:
            for pair in combinations(points, 2):
                mask = (*mask, ~self.too_sep(pair[0], pair[1]))

        func = np.logical_or if logical_or else np.logical_and
        mask = reduce(func, mask)

        if not np.any(mask):  # pragma: no cover
            return
        if replace:
            self.centroid[mask] = np.nan
            return
        if len(points) == 3:
            self.coords["midpoint"] = (
                self.coords[points[0]] + self.coords[points[1]]
            ) / 2
            points = ["midpoint", points[2]]
        coord_arrays = np.array([self.coords[point][mask] for point in points])
        self.centroid[mask] = np.nanmean(coord_arrays, axis=0)

    def too_sep(self, point1, point2):
        """Check if points are too far apart"""
        return (
            get_distance(self.coords[point1], self.coords[point2])
            >= self.max_LED_separation
        )

    def get_1pt_centroid(self):
        """Passthrough. If point is NaN, then centroid is NaN."""
        PT1 = self.points_dict.get("point1", None)
        mask = ~self.nans[PT1]  # For good points, centroid is the point
        self.centroid[mask] = self.coords[PT1][mask]
        self.centroid[~mask] = np.nan  # For bad points, centroid is NaN

    def get_2pt_centroid(self):
        """Calculate centroid for two points"""
        self.calc_centroid(  # Good points
            points=self.point_names,
            mask=(~self.nans[p] for p in self.point_names),
        )
        self.calc_centroid(mask=self.nans.values(), replace=True)  # All bad
        for point in self.point_names:  # only one point
            self.calc_centroid(
                points=[point],
                mask=(
                    ~self.nans[point],
                    *[self.nans[p] for p in self.point_names if p != point],
                ),
            )

    def get_4pt_centroid(self):
        """Calculate centroid for four points.

        If green and center are good, then centroid is average.
        If green and left/right are good, then centroid is average.
        If only left/right are good, then centroid is the average of left/right.
        If only the center is good, then centroid is the center.
        """
        green = self.points_dict.get("greenLED", None)
        red_C = self.points_dict.get("redLED_C", None)
        red_L = self.points_dict.get("redLED_L", None)
        red_R = self.points_dict.get("redLED_R", None)

        self.calc_centroid(  # Good green and center
            points=[green, red_C],
            mask=(~self.nans[green], ~self.nans[red_C]),
        )

        self.calc_centroid(  # green, left/right - average left/right
            points=[red_L, red_R, green],
            mask=(
                ~self.nans[green],
                self.nans[red_C],
                ~self.nans[red_L],
                ~self.nans[red_R],
            ),
        )

        self.calc_centroid(  # only left/right
            points=[red_L, red_R],
            mask=(
                self.nans[green],
                self.nans[red_C],
                ~self.nans[red_L],
                ~self.nans[red_R],
            ),
        )

        for side, other in [red_L, red_R], [red_R, red_L]:
            self.calc_centroid(  # green and one side are good, others are NaN
                points=[side, green],
                mask=(
                    ~self.nans[green],
                    self.nans[red_C],
                    ~self.nans[side],
                    self.nans[other],
                ),
            )

        self.calc_centroid(  # green is NaN, red center is good
            points=[red_C],
            mask=(self.nans[green], ~self.nans[red_C]),
        )
