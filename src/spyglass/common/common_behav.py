import os
import datajoint as dj
import ndx_franklab_novela
import pandas as pd
import pynwb
import pathlib
from typing import Dict

from .common_ephys import Raw  # noqa: F401
from .common_interval import IntervalList, interval_list_contains
from .common_nwbfile import Nwbfile
from .common_session import Session  # noqa: F401
from .common_task import TaskEpoch
from ..utils.dj_helper_fn import fetch_nwb
from ..utils.nwb_helper_fn import (
    get_all_spatial_series,
    get_data_interface,
    get_nwb_file,
)

schema = dj.schema("common_behav")


@schema
class PositionSource(dj.Manual):
    definition = """
    -> Session
    -> IntervalList
    ---
    source: varchar(200)            # source of data; current options are "trodes" and "dlc" (deep lab cut)
    import_file_name: varchar(2000)  # path to import file if importing position data
    """

    @classmethod
    def insert_from_nwbfile(cls, nwb_file_name):
        """Given an NWB file name, get the spatial series and interval lists from the file, add the interval
        lists to the IntervalList table, and populate the RawPosition table if possible.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        pos_dict = get_all_spatial_series(nwbf, verbose=True)
        if pos_dict is not None:
            for epoch in pos_dict:
                pdict = pos_dict[epoch]
                pos_interval_list_name = cls.get_pos_interval_name(epoch)

                # create the interval list and insert it
                interval_dict = dict()
                interval_dict["nwb_file_name"] = nwb_file_name
                interval_dict["interval_list_name"] = pos_interval_list_name
                interval_dict["valid_times"] = pdict["valid_times"]
                IntervalList().insert1(interval_dict, skip_duplicates=True)

                # add this interval list to the table
                key = dict()
                key["nwb_file_name"] = nwb_file_name
                key["interval_list_name"] = pos_interval_list_name
                key["source"] = "trodes"
                key["import_file_name"] = ""
                cls.insert1(key)

    @staticmethod
    def get_pos_interval_name(pos_epoch_num):
        return f"pos {pos_epoch_num} valid times"


@schema
class RawPosition(dj.Imported):
    """

    Notes
    -----
    The position timestamps come from: .pos_cameraHWSync.dat.
    If PTP is not used, the position timestamps are inferred by finding the
    closest timestamps from the neural recording via the trodes time.

    """

    definition = """
    -> PositionSource
    ---
    raw_position_object_id: varchar(40)    # the object id of the spatial series for this epoch in the NWB file
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # TODO refactor this. this calculates sampling rate (unused here) and is expensive to do twice
        pos_dict = get_all_spatial_series(nwbf)
        for epoch in pos_dict:
            if key["interval_list_name"] == PositionSource.get_pos_interval_name(epoch):
                pdict = pos_dict[epoch]
                key["raw_position_object_id"] = pdict["raw_position_object_id"]
                self.insert1(key)
                break

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)

    def fetch1_dataframe(self):
        raw_position_nwb = self.fetch_nwb()[0]["raw_position"]
        return pd.DataFrame(
            data=raw_position_nwb.data,
            index=pd.Index(raw_position_nwb.timestamps, name="time"),
            columns=raw_position_nwb.description.split(", "),
        )


@schema
class StateScriptFile(dj.Imported):
    definition = """
    -> TaskEpoch
    ---
    file_object_id: varchar(40)  # the object id of the file object
    """

    def make(self, key):
        """Add a new row to the StateScriptFile table. Requires keys "nwb_file_name", "file_object_id"."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        associated_files = nwbf.processing.get(
            "associated_files"
        ) or nwbf.processing.get("associated files")
        if associated_files is None:
            print(
                f'Unable to import StateScriptFile: no processing module named "associated_files" '
                f"found in {nwb_file_name}."
            )
            return

        for associated_file_obj in associated_files.data_interfaces.values():
            if not isinstance(associated_file_obj, ndx_franklab_novela.AssociatedFiles):
                print(
                    f'Data interface {associated_file_obj.name} within "associated_files" processing module is not '
                    f"of expected type ndx_franklab_novela.AssociatedFiles\n"
                )
                return
            # parse the task_epochs string
            # TODO update associated_file_obj.task_epochs to be an array of 1-based ints,
            # not a comma-separated string of ints
            epoch_list = associated_file_obj.task_epochs.split(",")
            # only insert if this is the statescript file
            print(associated_file_obj.description)
            if (
                "statescript".upper() in associated_file_obj.description.upper()
                or "state_script".upper() in associated_file_obj.description.upper()
                or "state script".upper() in associated_file_obj.description.upper()
            ):
                # find the file associated with this epoch
                if str(key["epoch"]) in epoch_list:
                    key["file_object_id"] = associated_file_obj.object_id
                    self.insert1(key)
            else:
                print("not a statescript file")

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)


@schema
class VideoFile(dj.Imported):
    """

    Notes
    -----
    The video timestamps come from: videoTimeStamps.cameraHWSync if PTP is used.
    If PTP is not used, the video timestamps come from videoTimeStamps.cameraHWFrameCount .

    """

    definition = """
    -> TaskEpoch
    video_file_num = 0: int
    ---
    video_file_object_id: varchar(40)  # the object id of the file object
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        video = get_data_interface(nwbf, "video", pynwb.behavior.BehavioralEvents)

        if video is None:
            print(f"No video data interface found in {nwb_file_name}\n")
            return

        # get the interval for the current TaskEpoch
        interval_list_name = (TaskEpoch() & key).fetch1("interval_list_name")
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")

        is_found = False
        for video_obj in video.time_series.values():
            # check to see if the times for this video_object are largely overlapping with the task epoch times
            if len(
                interval_list_contains(valid_times, video_obj.timestamps)
                > 0.9 * len(video_obj.timestamps)
            ):
                key["video_file_object_id"] = video_obj.object_id
                self.insert1(key)
                is_found = True

        if not is_found:
            print(f"No video found corresponding to epoch {interval_list_name}")

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)

    @classmethod
    def get_abs_path(cls, key: Dict):
        """Return the absolute path for a stored video file given a key with the nwb_file_name and epoch number

        The SPYGLASS_VIDEO_DIR environment variable must be set.

        Parameters
        ----------
        key : dict
            dictionary with nwb_file_name and epoch as keys

        Returns
        -------
        nwb_video_file_abspath : str
            The absolute path for the given file name.
        """
        video_dir = pathlib.Path(os.getenv("SPYGLASS_VIDEO_DIR", None))
        assert video_dir is not None, "You must set SPYGLASS_VIDEO_DIR"
        if not video_dir.exists():
            raise OSError("SPYGLASS_VIDEO_DIR does not exist")
        video_info = (cls & key).fetch1()
        nwb_path = Nwbfile.get_abs_path(key["nwb_file_name"])
        nwbf = get_nwb_file(nwb_path)
        nwb_video = nwbf.objects[video_info["video_file_object_id"]]
        video_filename = nwb_video.name
        # see if the file exists and is stored in the base analysis dir
        nwb_video_file_abspath = pathlib.Path(
            f"{video_dir}/{pathlib.Path(video_filename)}"
        )
        if nwb_video_file_abspath.exists():
            return nwb_video_file_abspath.as_posix()
        else:
            raise FileNotFoundError(
                f"video file with filename: {video_filename} "
                f"does not exist in {video_dir}/"
            )
