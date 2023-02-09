# Convenience functions
# DLC-utils copied from datajoint element-interface utils.py

import pathlib
import os
import pwd
import grp
import subprocess
from collections import abc
from typing import Union
from contextlib import redirect_stdout
import logging
import numpy as np
import datajoint as dj


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
                type(handler) == logging.StreamHandler for handler in logger.handlers
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
        f"No valid full-path found (from {root_directories})" f" for {relative_path}"
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
    from ..common.common_behav import VideoFile
    import pynwb

    video_info = (
        VideoFile() & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}
    ).fetch1()
    nwb_path = f"{os.getenv('SPYGLASS_BASE_DIR')}/raw/{video_info['nwb_file_name']}"
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
    video_path = pathlib.PurePath(pathlib.Path(video_path), pathlib.Path(filename))
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
                    frames_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"command {err.cmd} return with error (code {err.returncode}): {err.output}"
                ) from err
        else:
            try:
                check_process = subprocess.Popen(
                    packets_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
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

    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    query_cmd = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(
            subprocess.check_output(query_cmd.split(), stderr=subprocess.STDOUT)
        )[1:]
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"command {err.cmd} return with error (code {err.returncode}): {err.output}"
        ) from err
    memory_use_values = {i: int(x.split()[0]) for i, x in enumerate(memory_use_info)}
    return memory_use_values
