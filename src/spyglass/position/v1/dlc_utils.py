# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
import grp
import logging
import os
import pathlib
import pwd
import subprocess
from collections import abc
from contextlib import redirect_stdout
from itertools import groupby
from operator import itemgetter
from typing import Union

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from ...settings import raw_dir


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

    directory = pathlib.Path(directory)
    assert directory.exists(), f"Target directory: {directory} does not exist"
    uid = pwd.getpwnam(username).pw_uid
    if groupname:
        gid = grp.getgrnam(groupname).gr_gid
    else:
        gid = None
    for dirpath, _, filenames in os.walk(directory):
        os.chown(dirpath, uid, gid)
        os.chmod(dirpath, mode)
        for filename in filenames:
            os.chown(os.path.join(dirpath, filename), uid, gid)
            os.chmod(os.path.join(dirpath, filename), mode)


class OutputLogger:
    """
    A class to wrap a logging.Logger object in order to provide context manager capabilities.

    This class uses contextlib.redirect_stdout to temporarily redirect sys.stdout and thus
    print statements to the log file instead of, or as well as the console.

    Attributes
    ----------
    logger : logging.Logger
        logger object
    name : str
        name of logger
    level : int
        level of logging that the logger is set to handle

    Methods
    -------
    setup_logger(name_logfile, path_logfile, print_console=False)
        initialize or get logger object with name_logfile
        that writes to path_logfile

    Examples
    --------
    >>> with OutputLogger(name, path, print_console=True) as logger:
    ...    print("this will print to logfile")
    ...    logger.logger.info("this will log to the logfile")
    ... print("this will print to the console")
    ... logger.logger.info("this will log to the logfile")

    """

    def __init__(self, name, path, level="INFO", **kwargs):
        self.logger = self.setup_logger(name, path, **kwargs)
        self.name = self.logger.name
        self.level = getattr(logging, level)

    def setup_logger(
        self, name_logfile, path_logfile, print_console=False
    ) -> logging.Logger:
        """
        Sets up a logger for that outputs to a file, and optionally, the console

        Parameters
        ----------
        name_logfile : str
            name of the logfile to use
        path_logfile : str
            path to the file that should be used as the file handler
        print_console : bool, default-False
            if True, prints to console as well as log file.

        Returns
        -------
        logger : logging.Logger
            the logger object with specified handlers
        """

        logger = logging.getLogger(name_logfile)
        # check to see if handlers already exist for this logger
        if logger.handlers:
            for handler in logger.handlers:
                # if it's a file handler
                # type is used instead of isinstance,
                # which doesn't work properly with logging.StreamHandler
                if type(handler) == logging.FileHandler:
                    # if paths don't match, change file handler path
                    if not os.path.samefile(handler.baseFilename, path_logfile):
                        handler.close()
                        logger.removeHandler(handler)
                        file_handler = self._get_file_handler(path_logfile)
                        logger.addHandler(file_handler)
                # if a stream handler exists and
                # if print_console is False remove streamHandler
                if type(handler) == logging.StreamHandler:
                    if not print_console:
                        handler.close()
                        logger.removeHandler(handler)
            if print_console and not any(
                type(handler) == logging.StreamHandler
                for handler in logger.handlers
            ):
                logger.addHandler(self._get_stream_handler())

        else:
            file_handler = self._get_file_handler(path_logfile)
            logger.addHandler(file_handler)
            if print_console:
                logger.addHandler(self._get_stream_handler())
        logger.setLevel(logging.INFO)
        return logger

    def _get_file_handler(self, path):
        output_dir = pathlib.Path(os.path.dirname(path))
        if not os.path.exists(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(self._get_formatter())
        return file_handler

    def _get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self._get_formatter())
        return stream_handler

    def _get_formatter(self):
        return logging.Formatter(
            "[%(asctime)s] in %(pathname)s, line %(lineno)d: %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
        )

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self):
        pass

    def __enter__(self):
        self._redirector = redirect_stdout(self)
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)


def get_dlc_root_data_dir():
    if "custom" in dj.config:
        if "dlc_root_data_dir" in dj.config["custom"]:
            dlc_root_dirs = dj.config.get("custom", {}).get("dlc_root_data_dir")
    if not dlc_root_dirs:
        return [
            "/nimbus/deeplabcut/projects/",
            "/nimbus/deeplabcut/output/",
            "/cumulus/deeplabcut/",
        ]
    elif not isinstance(dlc_root_dirs, abc.Sequence):
        return list(dlc_root_dirs)
    else:
        return dlc_root_dirs


def get_dlc_processed_data_dir() -> str:
    """Returns session_dir relative to custom 'dlc_output_dir' root"""
    if "custom" in dj.config:
        if "dlc_output_dir" in dj.config["custom"]:
            dlc_output_dir = dj.config.get("custom", {}).get("dlc_output_dir")
    if dlc_output_dir:
        return pathlib.Path(dlc_output_dir)
    else:
        return pathlib.Path("/nimbus/deeplabcut/output/")


def find_full_path(root_directories, relative_path):
    """
    from Datajoint Elements - unused
    Given a relative path, search and return the full-path
     from provided potential root directories (in the given order)
        :param root_directories: potential root directories
        :param relative_path: the relative path to find the valid root directory
        :return: full-path (pathlib.Path object)
    """
    relative_path = _to_Path(relative_path)

    if relative_path.exists():
        return relative_path

    # Turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [_to_Path(root_directories)]

    for root_dir in root_directories:
        if (_to_Path(root_dir) / relative_path).exists():
            return _to_Path(root_dir) / relative_path

    raise FileNotFoundError(
        f"No valid full-path found (from {root_directories})"
        f" for {relative_path}"
    )


def find_root_directory(root_directories, full_path):
    """
    From datajoint elements - unused
    Given multiple potential root directories and a full-path,
    search and return one directory that is the parent of the given path
        :param root_directories: potential root directories
        :param full_path: the full path to search the root directory
        :return: root_directory (pathlib.Path object)
    """
    full_path = _to_Path(full_path)

    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} does not exist!")

    # Turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, pathlib.Path)):
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
    # TODO: add check to make sure interval_list_name refers to a single epoch
    # Or make key include epoch in and of itself instead of interval_list_name
    nwb_file_name = key["nwb_file_name"].split("_.")[0]
    output_dir = pathlib.Path(os.getenv("DLC_OUTPUT_PATH")) / pathlib.Path(
        f"{nwb_file_name}/{nwb_file_name}_{key['epoch']:02}"
        f"_model_" + key["dlc_model_name"].replace(" ", "-")
    )
    if makedir is True:
        if not os.path.exists(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_Path(path):
    """
    Convert the input "path" into a pathlib.Path object
    Handles one odd Windows/Linux incompatibility of the "\\"
    """
    return pathlib.Path(str(path).replace("\\", "/"))


def get_video_path(key):
    """
    Given nwb_file_name and interval_list_name returns specified
    video file filename and path

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
    """
    import pynwb

    from ...common.common_behav import VideoFile

    vf_key = {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}
    VideoFile()._no_transaction_make(vf_key, verbose=False)
    video_query = VideoFile & vf_key

    if len(video_query) != 1:
        print(f"Found {len(video_query)} videos for {vf_key}")
        return None, None, None, None

    video_info = video_query.fetch1()
    nwb_path = f"{raw_dir}/{video_info['nwb_file_name']}"

    with pynwb.NWBHDF5IO(path=nwb_path, mode="r") as in_out:
        nwb_file = in_out.read()
        nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
        video_filepath = VideoFile.get_abs_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}
        )
        video_dir = os.path.dirname(video_filepath) + "/"
        video_filename = video_filepath.split(video_dir)[-1]
        meters_per_pixel = nwb_video.device.meters_per_pixel
        timestamps = np.asarray(nwb_video.timestamps)

    return video_dir, video_filename, meters_per_pixel, timestamps


def check_videofile(
    video_path: Union[str, pathlib.PosixPath],
    output_path: Union[str, pathlib.PosixPath] = os.getenv("DLC_VIDEO_PATH"),
    video_filename: str = None,
    video_filetype: str = "h264",
):
    """
    Checks the file extension of a video file to make sure it is .mp4 for
    DeepLabCut processes. Converts to MP4 if not already.

    Parameters
    ----------
    video_path : str or PosixPath object
        path to directory of the existing video file without filename
    output_path : str or PosixPath object
        path to directory where converted video will be saved
    video_filename : str, Optional
        filename of the video to convert, if not provided, video_filetype must be
        and all video files of video_filetype in the directory will be converted
    video_filetype : str or List, Default 'h264', Optional
        If video_filename is not provided,
        all videos of this filetype will be converted to .mp4

    Returns
    -------
    output_files : List of PosixPath objects
        paths to converted video file(s)
    """

    if not video_filename:
        video_files = pathlib.Path(video_path).glob(f"*.{video_filetype}")
    else:
        video_files = [pathlib.Path(f"{video_path}/{video_filename}")]
    output_files = []
    for video_filepath in video_files:
        if video_filepath.exists():
            if video_filepath.suffix == ".mp4":
                output_files.append(video_filepath)
                continue
        video_file = (
            video_filepath.as_posix()
            .rsplit(video_filepath.parent.as_posix(), maxsplit=1)[-1]
            .split("/")[-1]
        )
        output_files.append(
            _convert_mp4(video_file, video_path, output_path, videotype="mp4")
        )
    return output_files


def _convert_mp4(
    filename: str,
    video_path: str,
    dest_path: str,
    videotype: str,
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

    orig_filename = filename
    video_path = pathlib.PurePath(
        pathlib.Path(video_path), pathlib.Path(filename)
    )
    if videotype not in ["mp4"]:
        raise NotImplementedError
    dest_filename = os.path.splitext(filename)[0]
    if ".1" in dest_filename:
        dest_filename = os.path.splitext(dest_filename)[0]
    dest_path = pathlib.Path(f"{dest_path}/{dest_filename}.{videotype}")
    convert_command = [
        "ffmpeg",
        "-vsync",
        "passthrough",
        "-i",
        f"{video_path.as_posix()}",
        "-codec",
        "copy",
        f"{dest_path.as_posix()}",
    ]
    try:
        convert_process = subprocess.Popen(
            convert_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"command {err.cmd} return with error (code {err.returncode}): {err.output}"
        ) from err
    out, _ = convert_process.communicate()
    print(out.decode("utf-8"))
    print(f"finished converting {filename}")
    print(
        f"Checking that number of packets match between {orig_filename} and {dest_filename}"
    )
    num_packets = []
    for file in [video_path, dest_path]:
        packets_command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            file.as_posix(),
        ]
        frames_command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            file.as_posix(),
        ]
        if count_frames:
            try:
                check_process = subprocess.Popen(
                    frames_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"command {err.cmd} return with error (code {err.returncode}): {err.output}"
                ) from err
        else:
            try:
                check_process = subprocess.Popen(
                    packets_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"command {err.cmd} return with error (code {err.returncode}): {err.output}"
                ) from err
        out, _ = check_process.communicate()
        num_packets.append(int(out.decode("utf-8").split("\n")[0]))
    print(
        f"Number of packets in {orig_filename}: {num_packets[0]}, {dest_filename}: {num_packets[1]}"
    )
    assert num_packets[0] == num_packets[1]
    if return_output:
        return dest_path


def get_gpu_memory():
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
            f"command {err.cmd} return with error (code {err.returncode}): "
            + f"{err.output}"
        ) from err
    memory_use_values = {
        i: int(x.split()[0]) for i, x in enumerate(memory_use_info)
    }
    return memory_use_values


def get_span_start_stop(indices):
    """_summary_

    Parameters
    ----------
    indices : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    span_inds = []
    # Get start and stop index of spans of consecutive indices
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        span_inds.append((group[0], group[-1]))
    return span_inds


def interp_pos(dlc_df, spans_to_interp, **kwargs):
    idx = pd.IndexSlice
    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        if (span_stop + 1) >= len(dlc_df):
            dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
            dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
            print(f"ind: {ind} has no endpoint with which to interpolate")
            continue
        if span_start < 1:
            dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
            dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
            print(f"ind: {ind} has no startpoint with which to interpolate")
            continue
        x = [dlc_df["x"].iloc[span_start - 1], dlc_df["x"].iloc[span_stop + 1]]
        y = [dlc_df["y"].iloc[span_start - 1], dlc_df["y"].iloc[span_stop + 1]]
        span_len = int(span_stop - span_start + 1)
        start_time = dlc_df.index[span_start]
        stop_time = dlc_df.index[span_stop]
        if "max_pts_to_interp" in kwargs:
            if span_len > kwargs["max_pts_to_interp"]:
                dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
                dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
                print(
                    f"inds {span_start} to {span_stop} "
                    f"length: {span_len} not interpolated"
                )
        if "max_cm_to_interp" in kwargs:
            if (
                np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
                > kwargs["max_cm_to_interp"]
            ):
                dlc_df.loc[idx[start_time:stop_time], idx["x"]] = np.nan
                dlc_df.loc[idx[start_time:stop_time], idx["y"]] = np.nan
                change = np.linalg.norm(
                    np.array([x[0], y[0]]) - np.array([x[1], y[1]])
                )
                print(
                    f"inds {span_start} to {span_stop + 1} "
                    f"with change in position: {change:.2f} not interpolated"
                )
                continue

        xnew = np.interp(
            x=dlc_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[x[0], x[-1]],
        )
        ynew = np.interp(
            x=dlc_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[y[0], y[-1]],
        )
        dlc_df.loc[idx[start_time:stop_time], idx["x"]] = xnew
        dlc_df.loc[idx[start_time:stop_time], idx["y"]] = ynew

    return dlc_df


def smooth_moving_avg(
    interp_df, smoothing_duration: float, sampling_rate: int, **kwargs
):
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


def fill_nan(variable, video_time, variable_time):
    video_ind = np.digitize(variable_time, video_time[1:])

    n_video_time = len(video_time)
    try:
        n_variable_dims = variable.shape[1]
        filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
    except IndexError:
        filled_variable = np.full((n_video_time,), np.nan)
    filled_variable[video_ind] = variable

    return filled_variable


def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
    """Converts from cm to pixels and flips the y-axis.
    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    cm_to_pixels : float

    Returns
    -------
    converted_data : ndarray, shape (n_time, 2)
    """
    return data / cm_to_pixels


def make_video(
    video_filename,
    video_frame_inds,
    position_mean,
    centroids,
    likelihoods,
    position_time,
    video_time=None,
    processor="opencv",
    frames=None,
    percent_frames=1,
    output_video_filename="output.mp4",
    cm_to_pixels=1.0,
    disable_progressbar=False,
    crop=None,
    arrow_radius=15,
    circle_radius=8,
):
    import cv2

    RGB_PINK = (234, 82, 111)
    RGB_YELLOW = (253, 231, 76)
    RGB_WHITE = (255, 255, 255)
    RGB_BLUE = (30, 144, 255)
    RGB_ORANGE = (255, 127, 80)
    #     "#29ff3e",
    #     "#ff0073",
    #     "#ff291a",
    #     "#1e2cff",
    #     "#b045f3",
    #     "#ffe91a",
    # ]
    if processor == "opencv":
        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        if frames is not None:
            n_frames = len(frames)
        else:
            n_frames = int(len(video_frame_inds) * percent_frames)
            frames = np.arange(0, n_frames)
        print(
            f"video save path: {output_video_filename}\n{n_frames} frames in total."
        )
        if crop:
            crop_offset_x = crop[0]
            crop_offset_y = crop[2]
            frame_size = (crop[1] - crop[0], crop[3] - crop[2])
        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )
        print(f"video_output: {output_video_filename}")

        # centroids = {
        #     color: self.fill_nan(data, video_time, position_time)
        #     for color, data in centroids.items()
        # }
        if video_time:
            position_mean = {
                key: fill_nan(
                    position_mean[key]["position"], video_time, position_time
                )
                for key in position_mean.keys()
            }
            orientation_mean = {
                key: fill_nan(
                    position_mean[key]["orientation"], video_time, position_time
                )
                for key in orientation_mean.keys()
            }
        print(
            f"frames start: {frames[0]}\nvideo_frames start: {video_frame_inds[0]}\ncv2 frame ind start: {int(video.get(1))}"
        )
        for time_ind in tqdm(
            frames, desc="frames", disable=disable_progressbar
        ):
            if time_ind == 0:
                video.set(1, time_ind + 1)
            elif int(video.get(1)) != time_ind - 1:
                video.set(1, time_ind - 1)
            is_grabbed, frame = video.read()

            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop:
                    frame = frame[crop[2] : crop[3], crop[0] : crop[1]].copy()
                if time_ind < video_frame_inds[0] - 1:
                    cv2.putText(
                        img=frame,
                        text=f"time_ind: {int(time_ind)} video frame: {int(video.get(1))}",
                        org=(10, 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=RGB_YELLOW,
                        thickness=1,
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                    continue
                cv2.putText(
                    img=frame,
                    text=f"time_ind: {int(time_ind)} video frame: {int(video.get(1))}",
                    org=(10, 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=RGB_YELLOW,
                    thickness=1,
                )
                pos_ind = time_ind - video_frame_inds[0]
                # red_centroid = centroids["red"][time_ind]
                # green_centroid = centroids["green"][time_ind]
                for key in position_mean.keys():
                    position = position_mean[key][pos_ind]
                    # if crop:
                    #     position = np.hstack(
                    #         (
                    #             convert_to_pixels(
                    #                 position[0, np.newaxis],
                    #                 frame_size,
                    #                 cm_to_pixels,
                    #             )
                    #             - crop_offset_x,
                    #             convert_to_pixels(
                    #                 position[1, np.newaxis],
                    #                 frame_size,
                    #                 cm_to_pixels,
                    #             )
                    #             - crop_offset_y,
                    #         )
                    #     )
                    # else:
                    #     position = convert_to_pixels(position, frame_size, cm_to_pixels)
                    position = convert_to_pixels(
                        position, frame_size, cm_to_pixels
                    )
                    orientation = orientation_mean[key][pos_ind]
                    if key == "DLC":
                        color = RGB_BLUE
                    if key == "Trodes":
                        color = RGB_ORANGE
                    if key == "Common":
                        color = RGB_PINK
                    if np.all(~np.isnan(position)) & np.all(
                        ~np.isnan(orientation)
                    ):
                        arrow_tip = (
                            int(
                                position[0] + arrow_radius * np.cos(orientation)
                            ),
                            int(
                                position[1] + arrow_radius * np.sin(orientation)
                            ),
                        )
                        cv2.arrowedLine(
                            img=frame,
                            pt1=tuple(position.astype(int)),
                            pt2=arrow_tip,
                            color=color,
                            thickness=4,
                            line_type=8,
                            shift=cv2.CV_8U,
                            tipLength=0.25,
                        )

                    if np.all(~np.isnan(position)):
                        cv2.circle(
                            img=frame,
                            center=tuple(position.astype(int)),
                            radius=circle_radius,
                            color=color,
                            thickness=-1,
                            shift=cv2.CV_8U,
                        )
                # if np.all(~np.isnan(red_centroid)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(red_centroid.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_YELLOW,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                # if np.all(~np.isnan(green_centroid)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(green_centroid.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_PINK,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                # if np.all(~np.isnan(head_position)) & np.all(
                #     ~np.isnan(head_orientation)
                # ):
                #     arrow_tip = (
                #         int(head_position[0] + arrow_radius * np.cos(head_orientation)),
                #         int(head_position[1] + arrow_radius * np.sin(head_orientation)),
                #     )
                #     cv2.arrowedLine(
                #         img=frame,
                #         pt1=tuple(head_position.astype(int)),
                #         pt2=arrow_tip,
                #         color=RGB_WHITE,
                #         thickness=4,
                #         line_type=8,
                #         shift=cv2.CV_8U,
                #         tipLength=0.25,
                #     )

                # if np.all(~np.isnan(head_position)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(head_position.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_WHITE,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                print("not grabbed")
                break
        print("releasing video")
        video.release()
        out.release()
        print("destroying cv2 windows")
        cv2.destroyAllWindows()
        print("finished making video with opencv")
        return

    elif processor == "matplotlib":
        import matplotlib.animation as animation
        import matplotlib.font_manager as fm

        position_mean = position_mean["DLC"]
        orientation_mean = orientation_mean["DLC"]
        frame_offset = -1
        time_slice = []
        video_slowdown = 1
        vmax = 0.07  # ?
        # Set up formatting for the movie files

        window_size = 501
        if likelihoods:
            plot_likelihood = True
        elif likelihoods is None:
            plot_likelihood = False

        window_ind = np.arange(window_size) - window_size // 2
        # Get video frames
        assert pathlib.Path(
            video_filename
        ).exists(), f"Path to video: {video_filename} does not exist"
        color_swatch = [
            "#29ff3e",
            "#ff0073",
            "#ff291a",
            "#1e2cff",
            "#b045f3",
            "#ffe91a",
        ]
        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        Writer = animation.writers["ffmpeg"]
        if frames is not None:
            n_frames = len(frames)
        else:
            n_frames = int(len(video_frame_inds) * percent_frames)
            frames = np.arange(0, n_frames)
        print(
            f"video save path: {output_video_filename}\n{n_frames} frames in total."
        )
        fps = int(np.round(frame_rate / video_slowdown))
        writer = Writer(fps=fps, bitrate=-1)
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop:
            frame = frame[crop[2] : crop[3], crop[0] : crop[1]].copy()
            crop_offset_x = crop[0]
            crop_offset_y = crop[2]
        frame_ind = 0
        with plt.style.context("dark_background"):
            # Set up plots
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [8, 1]},
                constrained_layout=False,
            )

            axes[0].tick_params(colors="white", which="both")
            axes[0].spines["bottom"].set_color("white")
            axes[0].spines["left"].set_color("white")
            image = axes[0].imshow(frame, animated=True)
            print(f"frame after init plot: {video.get(1)}")
            centroid_plot_objs = {
                bodypart: axes[0].scatter(
                    [],
                    [],
                    s=2,
                    zorder=102,
                    color=color,
                    label=f"{bodypart} position",
                    animated=True,
                    alpha=0.6,
                )
                for color, bodypart in zip(color_swatch, centroids.keys())
            }
            centroid_position_dot = axes[0].scatter(
                [],
                [],
                s=5,
                zorder=102,
                color="#b045f3",
                label="centroid position",
                animated=True,
                alpha=0.6,
            )
            (orientation_line,) = axes[0].plot(
                [],
                [],
                color="cyan",
                linewidth=1,
                animated=True,
                label="Orientation",
            )
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
            ratio = frame_size[1] / frame_size[0]
            if crop:
                ratio = (crop[3] - crop[2]) / (crop[1] - crop[0])
            x_left, x_right = axes[0].get_xlim()
            y_low, y_high = axes[0].get_ylim()
            axes[0].set_aspect(
                abs((x_right - x_left) / (y_low - y_high)) * ratio
            )
            axes[0].spines["top"].set_color("black")
            axes[0].spines["right"].set_color("black")
            time_delta = pd.Timedelta(
                position_time[0] - position_time[0]
            ).total_seconds()
            axes[0].legend(loc="lower right", fontsize=4)
            title = axes[0].set_title(
                f"time = {time_delta:3.4f}s\n frame = {frame_ind}",
                fontsize=8,
            )
            fontprops = fm.FontProperties(size=12)
            #     scalebar = AnchoredSizeBar(axes[0].transData,
            #                                20, '20 cm', 'lower right',
            #                                pad=0.1,
            #                                color='white',
            #                                frameon=False,
            #                                size_vertical=1,
            #                                fontproperties=fontprops)

            #     axes[0].add_artist(scalebar)
            axes[0].axis("off")
            if plot_likelihood:
                likelihood_objs = {
                    bodypart: axes[1].plot(
                        [],
                        [],
                        color=color,
                        linewidth=1,
                        animated=True,
                        clip_on=False,
                        label=bodypart,
                    )[0]
                    for color, bodypart in zip(color_swatch, likelihoods.keys())
                }
                axes[1].set_ylim((0.0, 1))
                print(f"frame_rate: {frame_rate}")
                axes[1].set_xlim(
                    (
                        window_ind[0] / frame_rate,
                        window_ind[-1] / frame_rate,
                    )
                )
                axes[1].set_xlabel("Time [s]")
                axes[1].set_ylabel("Likelihood")
                axes[1].set_facecolor("black")
                axes[1].spines["top"].set_color("black")
                axes[1].spines["right"].set_color("black")
                axes[1].legend(loc="upper right", fontsize=4)
            progress_bar = tqdm(leave=True, position=0)
            progress_bar.reset(total=n_frames)

            def _update_plot(time_ind):
                if time_ind == 0:
                    video.set(1, time_ind + 1)
                else:
                    video.set(1, time_ind - 1)
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if crop:
                        frame = frame[
                            crop[2] : crop[3], crop[0] : crop[1]
                        ].copy()
                    image.set_array(frame)
                pos_ind = np.where(video_frame_inds == time_ind)[0]
                if len(pos_ind) == 0:
                    centroid_position_dot.set_offsets((np.NaN, np.NaN))
                    for bodypart in centroid_plot_objs.keys():
                        centroid_plot_objs[bodypart].set_offsets(
                            (np.NaN, np.NaN)
                        )
                    orientation_line.set_data((np.NaN, np.NaN))
                    title.set_text(f"time = {0:3.4f}s\n frame = {time_ind}")
                else:
                    pos_ind = pos_ind[0]
                    dlc_centroid_data = convert_to_pixels(
                        position_mean[pos_ind], frame, cm_to_pixels
                    )
                    if crop:
                        dlc_centroid_data = np.hstack(
                            (
                                convert_to_pixels(
                                    position_mean[pos_ind, 0, np.newaxis],
                                    frame,
                                    cm_to_pixels,
                                )
                                - crop_offset_x,
                                convert_to_pixels(
                                    position_mean[pos_ind, 1, np.newaxis],
                                    frame,
                                    cm_to_pixels,
                                )
                                - crop_offset_y,
                            )
                        )
                    for bodypart in centroid_plot_objs.keys():
                        centroid_plot_objs[bodypart].set_offsets(
                            convert_to_pixels(
                                centroids[bodypart][pos_ind],
                                frame,
                                cm_to_pixels,
                            )
                        )
                    centroid_position_dot.set_offsets(dlc_centroid_data)
                    r = 30
                    orientation_line.set_data(
                        [
                            dlc_centroid_data[0],
                            dlc_centroid_data[0]
                            + r * np.cos(orientation_mean[pos_ind]),
                        ],
                        [
                            dlc_centroid_data[1],
                            dlc_centroid_data[1]
                            + r * np.sin(orientation_mean[pos_ind]),
                        ],
                    )
                    # Need to convert times to datetime object probably.

                    time_delta = pd.Timedelta(
                        pd.to_datetime(position_time[pos_ind] * 1e9, unit="ns")
                        - pd.to_datetime(position_time[0] * 1e9, unit="ns")
                    ).total_seconds()
                    title.set_text(
                        f"time = {time_delta:3.4f}s\n frame = {time_ind}"
                    )
                    likelihood_inds = pos_ind + window_ind
                    neg_inds = np.where(likelihood_inds < 0)[0]
                    over_inds = np.where(
                        likelihood_inds
                        > (len(likelihoods[list(likelihood_objs.keys())[0]]))
                        - 1
                    )[0]
                    if len(neg_inds) > 0:
                        likelihood_inds[neg_inds] = 0
                    if len(over_inds) > 0:
                        likelihood_inds[neg_inds] = -1
                    for bodypart in likelihood_objs.keys():
                        likelihood_objs[bodypart].set_data(
                            window_ind / frame_rate,
                            np.asarray(likelihoods[bodypart][likelihood_inds]),
                        )
                progress_bar.update()

                return (
                    image,
                    centroid_position_dot,
                    orientation_line,
                    title,
                    # redC_likelihood,
                    # green_likelihood,
                    # redL_likelihood,
                    # redR_likelihood,
                )

            movie = animation.FuncAnimation(
                fig,
                _update_plot,
                frames=frames,
                interval=1000 / fps,
                blit=True,
            )
            movie.save(output_video_filename, writer=writer, dpi=400)
            video.release()
            print("finished making video with matplotlib")
            return
