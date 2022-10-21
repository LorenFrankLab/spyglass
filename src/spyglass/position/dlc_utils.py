# Convenience functions
# DLC-utils copied from datajoint element-interface utils.py

import pathlib
import os
import pwd
import grp
from collections import abc
from typing import Union
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
        "No valid full-path found (from {})"
        " for {}".format(root_directories, relative_path)
    )


def find_root_directory(root_directories, full_path):
    """
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

    except StopIteration:
        raise FileNotFoundError(
            "No valid root directory found (from {})"
            " for {}".format(root_directories, full_path)
        )


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
    from ..common.nwb_helper_fn import get_nwb_file
    from ..common.common_nwbfile import Nwbfile
    import pynwb

    video_info = (
        VideoFile() & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}
    ).fetch1()
    nwb_path = f"{os.getenv('SPYGLASS_BASE_DIR')}/raw/{video_info['nwb_file_name']}"
    with pynwb.NWBHDF5IO(path=nwb_path, mode="r") as io:
        nwb_file = io.read()
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

    import subprocess

    orig_filename = filename
    video_path = pathlib.PurePath(pathlib.Path(video_path), pathlib.Path(filename))
    if videotype not in ["mp4"]:
        raise NotImplementedError
    dest_filename = os.path.splitext(filename)[0]
    if ".1" in dest_filename:
        dest_filename = os.path.splitext(dest_filename)[0]
    dest_path = pathlib.Path(f"{dest_path}/{dest_filename}.{videotype}")
    convert_command = (
        f"ffmpeg -vsync passthrough -i {video_path.as_posix()} "
        f"-codec copy {dest_path.as_posix()}"
    )
    os.system(convert_command)
    print(f"finished converting {filename}")
    print(
        f"Checking that number of packets match between {orig_filename} and {dest_filename}"
    )
    num_packets = []
    for ind, file in enumerate([video_path, dest_path]):
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
            p = subprocess.Popen(
                frames_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        else:
            p = subprocess.Popen(
                packets_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        out, err = p.communicate()
        num_packets.append(int(out.decode("utf-8").split("\n")[0]))
    print(
        f"Number of packets in {orig_filename}: {num_packets[0]}, {dest_filename}: {num_packets[1]}"
    )
    assert num_packets[0] == num_packets[1]
    if return_output:
        return dest_path
