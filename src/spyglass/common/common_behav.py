import os
import pathlib
from functools import reduce
from typing import Dict

import datajoint as dj
import ndx_franklab_novela
import pandas as pd
import pynwb

from ..utils.dj_helper_fn import fetch_nwb
from ..utils.nwb_helper_fn import (
    get_all_spatial_series,
    get_data_interface,
    get_nwb_file,
)
from .common_device import CameraDevice
from .common_ephys import Raw  # noqa: F401
from .common_interval import IntervalList, interval_list_contains
from .common_nwbfile import Nwbfile
from .common_session import Session  # noqa: F401
from .common_task import TaskEpoch

schema = dj.schema("common_behav")


@schema
class PositionSource(dj.Manual):
    definition = """
    -> Session
    -> IntervalList
    ---
    source: varchar(200)             # source of data (e.g., trodes, dlc)
    import_file_name: varchar(2000)  # path to import file if importing 
    """

    class SpatialSeries(dj.Part):
        definition = """
        -> master
        id : int unsigned            # index of spatial series
        ---
        name=null: varchar(32)       # name of spatial series
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
        nwbf = get_nwb_file(nwb_file_name)
        all_pos = get_all_spatial_series(nwbf, verbose=True, old_format=False)
        sess_key = dict(nwb_file_name=nwb_file_name)
        src_key = dict(**sess_key, source="trodes", import_file_name="")

        if all_pos is None:
            return

        sources = []
        intervals = []
        spat_series = []

        for epoch, epoch_list in all_pos.items():
            ind_key = dict(interval_list_name=cls.get_pos_interval_name(epoch))

            sources.append(dict(**src_key, **ind_key))
            intervals.append(
                dict(
                    **sess_key,
                    **ind_key,
                    valid_times=epoch_list[0]["valid_times"],
                )
            )

            for index, pdict in enumerate(epoch_list):
                spat_series.append(
                    dict(
                        **sess_key,
                        **ind_key,
                        id=ndex,
                        name=pdict.get("name"),
                    )
                )

        with cls.connection.transaction:
            IntervalList.insert(intervals)
            cls.insert(sources)
            cls.SpatialSeries.insert(spat_series)

    @staticmethod
    def get_pos_interval_name(epoch_num: int) -> str:
        """Return string of the interval name from the epoch number.

        Parameters
        ----------
        pos_epoch_num : int
            Input epoch number

        Returns
        -------
        str
            Position interval name (e.g., pos 2 valid times)
        """
        try:
            int(epoch_num)
        except ValueError:
            raise ValueError(
                f"Epoch number must must be an integer. Received: {epoch_num}"
            )
        return f"pos {epoch_num} valid times"

    @staticmethod
    def get_epoch_num(name: str) -> int:
        """Return the epoch number from the interval name.

        Parameters
        ----------
        name : str
            Name of position interval (e.g., pos epoch 1 index 2 valid times)

        Returns
        -------
        int
            epoch number
        """
        return int(name.replace("pos ", "").replace(" valid times", ""))


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
    """

    class Object(dj.Part):
        definition = """
        -> master
        -> PositionSource.SpatialSeries.proj('id')
        ---
        raw_position_object_id: varchar(40) # id of spatial series in NWB file
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            INDEX_ADJUST = 1  # adjust 0-index to 1-index (e.g., xloc0 -> xloc1)

            id_rp = [(n["id"], n["raw_position"]) for n in self.fetch_nwb()]

            df_list = [
                pd.DataFrame(
                    data=rp.data,
                    index=pd.Index(rp.timestamps, name="time"),
                    columns=[
                        col  # use existing columns if already numbered
                        if "1" in rp.description or "2" in rp.description
                        # else number them by id
                        else col + str(id + INDEX_ADJUST)
                        for col in rp.description.split(", ")
                    ],
                )
                for id, rp in id_rp
            ]

            return reduce(lambda x, y: pd.merge(x, y, on="time"), df_list)

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        interval_list_name = key["interval_list_name"]

        nwbf = get_nwb_file(nwb_file_name)
        indices = (PositionSource.SpatialSeries & key).fetch("id")

        # incl_times = False -> don't do extra processing for valid_times
        spat_objs = get_all_spatial_series(
            nwbf, old_format=False, incl_times=False
        )[PositionSource.get_epoch_num(interval_list_name)]

        self.insert1(key)
        self.Object.insert(
            [
                dict(
                    nwb_file_name=nwb_file_name,
                    interval_list_name=interval_list_name,
                    id=index,
                    raw_position_object_id=obj["raw_position_object_id"],
                )
                for index, obj in enumerate(spat_objs)
                if index in indices
            ]
        )

    def fetch_nwb(self, *attrs, **kwargs):
        raise NotImplementedError(
            "fetch_nwb now operates on RawPosition.Object"
        )

    def fetch1_dataframe(self):
        raise NotImplementedError(
            "fetch1_dataframe now operates on RawPosition.Object"
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
            if not isinstance(
                associated_file_obj, ndx_franklab_novela.AssociatedFiles
            ):
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
                or "state_script".upper()
                in associated_file_obj.description.upper()
                or "state script".upper()
                in associated_file_obj.description.upper()
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
    camera_name: varchar(80)
    video_file_object_id: varchar(40)  # the object id of the file object
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        videos = get_data_interface(
            nwbf, "video", pynwb.behavior.BehavioralEvents
        )

        if videos is None:
            print(f"No video data interface found in {nwb_file_name}\n")
            return
        else:
            videos = videos.time_series

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
        for ind, video in enumerate(videos.values()):
            if isinstance(video, pynwb.image.ImageSeries):
                video = [video]
            for video_obj in video:
                # check to see if the times for this video_object are largely overlapping with the task epoch times
                if len(
                    interval_list_contains(valid_times, video_obj.timestamps)
                    > 0.9 * len(video_obj.timestamps)
                ):
                    key["video_file_num"] = ind
                    camera_name = video_obj.device.camera_name
                    if CameraDevice & {"camera_name": camera_name}:
                        key["camera_name"] = video_obj.device.camera_name
                    else:
                        raise KeyError(
                            f"No camera with camera_name: {camera_name} found in CameraDevice table."
                        )
                    key["video_file_object_id"] = video_obj.object_id
                    self.insert1(key)
                    is_found = True

        if not is_found:
            print(f"No video found corresponding to epoch {interval_list_name}")

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)

    @classmethod
    def update_entries(cls, restrict={}):
        existing_entries = (cls & restrict).fetch("KEY")
        for row in existing_entries:
            if (cls & row).fetch1("camera_name"):
                continue
            video_nwb = (cls & row).fetch_nwb()[0]
            if len(video_nwb) != 1:
                raise ValueError(
                    f"expecting 1 video file per entry, but {len(video_nwb)} files found"
                )
            row["camera_name"] = video_nwb[0]["video_file"].device.camera_name
            cls.update1(row=row)

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
