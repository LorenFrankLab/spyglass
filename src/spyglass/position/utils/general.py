"""General utilities for position processing.

This module contains general-purpose helper functions used across
the position pipeline. These are re-exported from the parent utils.py
module for backwards compatibility.
"""

import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path, PosixPath
from typing import Union

import numpy as np
import pandas as pd

from spyglass.common.common_behav import VideoFile
from spyglass.settings import pose_output_dir, pose_video_dir, raw_dir
from spyglass.utils.logging import logger, stream_handler


def flatten_multiindex(pose_df: "pd.DataFrame") -> "pd.DataFrame":
    """Flatten a 3-level MultiIndex DataFrame to 2 levels by dropping scorer.

    Converts columns from (scorer, bodypart, coord) to (bodypart, coord).
    DataFrames that are already 2-level or non-MultiIndex are returned as-is.

    Parameters
    ----------
    pose_df : pd.DataFrame
        DataFrame whose columns may be a 3-level MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with at most 2 column levels.
    """
    if not isinstance(pose_df.columns, pd.MultiIndex):
        return pose_df
    if pose_df.columns.nlevels == 3:
        pose_df = pose_df.copy()
        pose_df.columns = pose_df.columns.droplevel(0)
    return pose_df


def get_param_names(func):
    """Get parameter names for a function signature.

    Parameters
    ----------
    func : callable
        Function to inspect

    Returns
    -------
    list
        List of parameter names

    Examples
    --------
    >>> def my_func(a, b, c=3):
    ...     pass
    >>> get_param_names(my_func)
    ['a', 'b', 'c']
    """
    if not callable(func):
        return []  # If called when func is not imported
    return list(inspect.signature(func).parameters)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove special characters.

    Parameters
    ----------
    filename : str
        Original filename

    Returns
    -------
    str
        Sanitized filename with special characters replaced

    Examples
    --------
    >>> sanitize_filename("my file, name & stuff.txt")
    'my_file-_name_and_stuff_txt'
    """
    char_map = {
        " ": "_",
        ".": "_",
        ",": "-",
        "&": "and",
        "'": "",
    }
    return "".join([char_map.get(c, c) for c in filename])


def get_most_recent_file(path: Path, ext: str = "") -> Path:
    """Get the most recent file of a given extension in a directory.

    Parameters
    ----------
    path : Path
        Path to the directory containing the files
    ext : str, optional
        File extension to filter by (e.g., ".h5", ".csv"). Default is "",
        which returns the most recent file regardless of extension.

    Returns
    -------
    Path
        Path to the most recently modified file

    Raises
    ------
    FileNotFoundError
        If no files with the given extension are found

    Examples
    --------
    >>> from pathlib import Path
    >>> get_most_recent_file(Path("/path/to/dir"), ext=".h5")
    PosixPath('/path/to/dir/latest_file.h5')
    """
    eval_files = list(Path(path).glob(f"*{ext}"))
    if not eval_files:
        raise FileNotFoundError(f"No {ext} files found in directory: {path}")

    eval_latest, max_modified_time = None, 0
    for eval_path in eval_files:
        modified_time = os.path.getmtime(eval_path)
        if modified_time > max_modified_time:
            eval_latest = eval_path
            max_modified_time = modified_time

    return eval_latest


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
    output_dir = Path(pose_output_dir) / Path(
        f"{nwb_file_name}/{nwb_file_name}_{key['epoch']:02}"
        f"_model_" + key["dlc_model_name"].replace(" ", "-")
    )
    if makedir:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    output_path: Union[str, PosixPath] = pose_video_dir,
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
    logger.info_msg(f"Finished converting {filename}")

    # check packets match orig file
    logger.info_msg(f"Checking packets match orig file: {dest_filename}")
    orig_packets = _check_packets(video_path, count_frames=count_frames)
    dest_packets = _check_packets(dest_path, count_frames=count_frames)
    if orig_packets != dest_packets:  # pragma: no cover
        logger.warning(  # pragma: no cover
            f"Conversion error: {orig_filename} -> {dest_filename}"
        )

    if return_output:
        return dest_path


def _check_packets(file, count_frames=False):
    """Check number of packets/frames in video using ffprobe"""
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
                str(file),
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
