import pathlib
import re
from functools import reduce
from typing import Dict

import datajoint as dj
import ndx_franklab_novela
import pandas as pd
import pynwb

from spyglass.common.common_device import CameraDevice
from spyglass.common.common_ephys import Raw  # noqa: F401
from spyglass.common.common_interval import IntervalList, interval_list_contains
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_task import TaskEpoch
from spyglass.settings import video_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import (
    get_all_spatial_series,
    get_data_interface,
    get_nwb_file,
)

schema = dj.schema("common_behav")


@schema
class PositionSource(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    -> IntervalList
    ---
    source: varchar(200)             # source of data (e.g., trodes, dlc)
    import_file_name: varchar(2000)  # path to import file if importing
    """

    class SpatialSeries(SpyglassMixin, dj.Part):
        definition = """
        -> master
        id = 0 : int unsigned            # index of spatial series
        ---
        name=null: varchar(32)       # name of spatial series
        """

    def populate(self, keys=None):
        """Insert position source data from NWB file.

        WARNING: populate method on Manual table is not protected by transaction
                protections like other DataJoint tables.
        """
        if not isinstance(keys, list):
            keys = [keys]
        if isinstance(keys[0], (dj.Table, dj.expression.QueryExpression)):
            keys = [k for tbl in keys for k in tbl.fetch("KEY", as_dict=True)]
        for key in keys:
            nwb_file_name = key.get("nwb_file_name")
            if not nwb_file_name:
                raise ValueError(
                    "PositionSource.populate is an alias for a non-computed table "
                    + "and must be passed a key with nwb_file_name"
                )
            self.insert_from_nwbfile(nwb_file_name, skip_duplicates=True)

    @classmethod
    def insert_from_nwbfile(cls, nwb_file_name, skip_duplicates=False) -> None:
        """Add intervals to ItervalList and PositionSource.

        Given an NWB file name, get the spatial series and interval lists from
        the file, add the interval lists to the IntervalList table, and
        populate the RawPosition table if possible.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        """
        nwbf = get_nwb_file(nwb_file_name)
        all_pos = get_all_spatial_series(nwbf, verbose=True)
        sess_key = Nwbfile.get_file_key(nwb_file_name)
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
                    pipeline="position",
                )
            )

            for index, pdict in enumerate(epoch_list):
                spat_series.append(
                    dict(
                        **sess_key,
                        **ind_key,
                        id=index,
                        name=pdict.get("name"),
                    )
                )

        with cls.connection.transaction:
            IntervalList.insert(intervals, skip_duplicates=skip_duplicates)
            cls.insert(sources, skip_duplicates=skip_duplicates)
            cls.SpatialSeries.insert(
                spat_series, skip_duplicates=skip_duplicates
            )

        # make map from epoch intervals to position intervals
        populate_position_interval_map_session(nwb_file_name)

    @staticmethod
    def get_pos_interval_name(epoch_num: int) -> str:
        """Return string of the interval name from the epoch number.

        Parameters
        ----------
        epoch_num : int
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
    def _is_valid_name(name) -> bool:
        return name.startswith("pos ") and name.endswith(" valid times")

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
        if not PositionSource._is_valid_name(name):
            raise ValueError(f"Invalid interval name: {name}")
        return int(name.replace("pos ", "").replace(" valid times", ""))


@schema
class RawPosition(SpyglassMixin, dj.Imported):
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

    class PosObject(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> PositionSource.SpatialSeries.proj('id')
        ---
        raw_position_object_id: varchar(40) # id of spatial series in NWB file
        """

        _nwb_table = Nwbfile

        def fetch1_dataframe(self):
            id_rp = [(n["id"], n["raw_position"]) for n in self.fetch_nwb()]

            if len(set(rp.interval for _, rp in id_rp)) > 1:
                logger.warn("Loading DataFrame with multiple intervals.")

            df_list = [
                pd.DataFrame(
                    data=rp.data,
                    index=pd.Index(rp.timestamps, name="time"),
                    columns=self._get_column_names(rp, pos_id),
                )
                for pos_id, rp in id_rp
            ]

            return reduce(lambda x, y: pd.merge(x, y, on="time"), df_list)

        @staticmethod
        def _get_column_names(rp, pos_id):
            INDEX_ADJUST = 1  # adjust 0-index to 1-index (e.g., xloc0 -> xloc1)
            n_pos_dims = rp.data.shape[1]
            column_names = [
                (
                    col  # use existing columns if already numbered
                    if "1" in rp.description or "2" in rp.description
                    # else number them by id
                    else col + str(pos_id + INDEX_ADJUST)
                )
                for col in rp.description.split(", ")
            ]
            if len(column_names) != n_pos_dims:
                # if the string split didn't work, use default names
                column_names = ["x", "y", "z"][:n_pos_dims]
            return column_names

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        interval_list_name = key["interval_list_name"]

        nwbf = get_nwb_file(nwb_file_name)
        indices = (PositionSource.SpatialSeries & key).fetch("id")

        # incl_times = False -> don't do extra processing for valid_times
        spat_objs = get_all_spatial_series(nwbf, incl_times=False)[
            PositionSource.get_epoch_num(interval_list_name)
        ]

        self.insert1(key)
        self.PosObject.insert(
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

    def fetch_nwb(self, *attrs, **kwargs) -> list:
        """
        Returns a condatenated list of nwb objects from RawPosition.PosObject
        """
        return (
            self.PosObject()
            .restrict(self.restriction)  # Avoids fetch_nwb on whole table
            .fetch_nwb(*attrs, **kwargs)
        )

    def fetch1_dataframe(self):
        """Returns a dataframe with all RawPosition.PosObject items.

        Uses interval_list_name as column index.
        """
        ret = {}

        pos_obj_set = self.PosObject & self.restriction
        unique_intervals = set(pos_obj_set.fetch("interval_list_name"))

        for interval in unique_intervals:
            ret[interval] = (
                pos_obj_set & {"interval_list_name": interval}
            ).fetch1_dataframe()

        if len(unique_intervals) == 1:
            return next(iter(ret.values()))

        return pd.concat(ret, axis=1)


@schema
class StateScriptFile(SpyglassMixin, dj.Imported):
    definition = """
    -> TaskEpoch
    ---
    file_object_id: varchar(40)  # the object id of the file object
    """

    _nwb_table = Nwbfile

    def make(self, key):
        """Add a new row to the StateScriptFile table."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        associated_files = nwbf.processing.get(
            "associated_files"
        ) or nwbf.processing.get("associated files")
        if associated_files is None:
            logger.info(
                "Unable to import StateScriptFile: no processing module named "
                + '"associated_files" found in {nwb_file_name}.'
            )
            return  # See #849

        for associated_file_obj in associated_files.data_interfaces.values():
            if not isinstance(
                associated_file_obj, ndx_franklab_novela.AssociatedFiles
            ):
                logger.info(
                    f"Data interface {associated_file_obj.name} within "
                    + '"associated_files" processing module is not '
                    + "of expected type ndx_franklab_novela.AssociatedFiles\n"
                )
                return

            # parse the task_epochs string
            # TODO: update associated_file_obj.task_epochs to be an array of
            # 1-based ints, not a comma-separated string of ints

            epoch_list = associated_file_obj.task_epochs.split(",")
            # only insert if this is the statescript file
            logger.info(associated_file_obj.description)
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
                logger.info("not a statescript file")


@schema
class VideoFile(SpyglassMixin, dj.Imported):
    """

    Notes
    -----
    The video timestamps come from: videoTimeStamps.cameraHWSync if PTP is
    used. If PTP is not used, the video timestamps come from
    videoTimeStamps.cameraHWFrameCount .

    """

    definition = """
    -> TaskEpoch
    video_file_num = 0: int
    ---
    camera_name: varchar(80)
    video_file_object_id: varchar(40)  # the object id of the file object
    """

    _nwb_table = Nwbfile

    def make(self, key):
        self._no_transaction_make(key)

    def _no_transaction_make(self, key, verbose=True):
        if not self.connection.in_transaction:
            self.populate(key)
            return

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        videos = get_data_interface(
            nwbf, "video", pynwb.behavior.BehavioralEvents
        )

        if videos is None:
            logger.warn(f"No video data interface found in {nwb_file_name}\n")
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
        cam_device_str = r"camera_device (\d+)"
        is_found = False
        for ind, video in enumerate(videos.values()):
            if isinstance(video, pynwb.image.ImageSeries):
                video = [video]
            for video_obj in video:
                # check to see if the times for this video_object are largely
                # overlapping with the task epoch times

                if len(
                    interval_list_contains(valid_times, video_obj.timestamps)
                    > 0.9 * len(video_obj.timestamps)
                ):
                    nwb_cam_device = video_obj.device.name
                    # returns whatever was captured in the first group (within the parentheses) of the regular expression -- in this case, 0
                    key["video_file_num"] = int(
                        re.match(cam_device_str, nwb_cam_device)[1]
                    )
                    camera_name = video_obj.device.camera_name
                    if CameraDevice & {"camera_name": camera_name}:
                        key["camera_name"] = video_obj.device.camera_name
                    else:
                        raise KeyError(
                            f"No camera with camera_name: {camera_name} found "
                            + "in CameraDevice table."
                        )
                    key["video_file_object_id"] = video_obj.object_id
                    self.insert1(key)
                    is_found = True

        if not is_found and verbose:
            logger.info(
                f"No video found corresponding to file {nwb_file_name}, "
                + f"epoch {interval_list_name}"
            )

    @classmethod
    def update_entries(cls, restrict={}):
        existing_entries = (cls & restrict).fetch("KEY")
        for row in existing_entries:
            if (cls & row).fetch1("camera_name"):
                continue
            video_nwb = (cls & row).fetch_nwb()[0]
            if len(video_nwb) != 1:
                raise ValueError(
                    f"Expecting 1 video file per entry. {len(video_nwb)} found"
                )
            row["camera_name"] = video_nwb[0]["video_file"].device.camera_name
            cls.update1(row=row)

    @classmethod
    def get_abs_path(cls, key: Dict):
        """Return the absolute path for a stored video file given a key.

        Key must include the nwb_file_name and epoch number. The
        SPYGLASS_VIDEO_DIR environment variable must be set.

        Parameters
        ----------
        key : dict
            dictionary with nwb_file_name and epoch as keys

        Returns
        -------
        nwb_video_file_abspath : str
            The absolute path for the given file name.
        """
        video_path_obj = pathlib.Path(video_dir)
        video_info = (cls & key).fetch1()
        nwb_path = Nwbfile.get_abs_path(key["nwb_file_name"])
        nwbf = get_nwb_file(nwb_path)
        nwb_video = nwbf.objects[video_info["video_file_object_id"]]
        video_filename = nwb_video.name
        # see if the file exists and is stored in the base analysis dir
        nwb_video_file_abspath = pathlib.Path(video_path_obj / video_filename)
        if nwb_video_file_abspath.exists():
            return nwb_video_file_abspath.as_posix()
        else:
            raise FileNotFoundError(
                f"video file with filename: {video_filename} "
                f"does not exist in {video_path_obj}/"
            )


@schema
class PositionIntervalMap(SpyglassMixin, dj.Computed):
    definition = """
    -> IntervalList
    ---
    position_interval_name: varchar(200)  # name of the corresponding position interval
    """

    def make(self, key):
        self._no_transaction_make(key)

    def _no_transaction_make(self, key):
        # Find correspondence between pos valid times names and epochs. Use
        # epsilon to tolerate small differences in epoch boundaries across
        # epoch/pos intervals

        if not self.connection.in_transaction:
            # if not called in the context of a make function, call its own make function
            self.populate(key)
            return

        # *** HARD CODED VALUES ***
        EPSILON = 0.51  # tolerated time diff in bounds across epoch/pos
        no_pop_msg = "CANNOT POPULATE PositionIntervalMap"

        nwb_file_name = key["nwb_file_name"]
        pos_intervals = get_pos_interval_list_names(nwb_file_name)

        # Skip populating if no pos interval list names
        if len(pos_intervals) == 0:
            # TODO: Now that populate_all accept errors, raise here?
            logger.error(f"NO POS INTERVALS FOR {key}; {no_pop_msg}")
            return

        valid_times = (IntervalList & key).fetch1("valid_times")
        time_bounds = [
            valid_times[0][0] - EPSILON,
            valid_times[-1][-1] + EPSILON,
        ]

        matching_pos_intervals = []
        restr = (
            f"nwb_file_name='{nwb_file_name}' AND interval_list_name=" + "'{}'"
        )
        for pos_interval in pos_intervals:
            # cbroz: fetch1->fetch. fetch1 would fail w/o result
            pos_times = (IntervalList & restr.format(pos_interval)).fetch(
                "valid_times"
            )

            if len(pos_times) == 0:
                continue

            pos_times = pos_times[0]

            if all(
                [
                    time_bounds[0] <= time <= time_bounds[1]
                    for time in [pos_times[0][0], pos_times[-1][-1]]
                ]
            ):
                matching_pos_intervals.append(pos_interval)

            if len(matching_pos_intervals) > 1:
                break

        # Check that each pos interval was matched to only one epoch
        if len(matching_pos_intervals) != 1:
            # TODO: Now that populate_all accept errors, raise here?
            logger.warning(
                f"Found {len(matching_pos_intervals)} pos intervals for {key}; "
                + f"{no_pop_msg}\n{matching_pos_intervals}"
            )
            return

        # Insert into table
        key["position_interval_name"] = matching_pos_intervals[0]
        self.insert1(key, allow_direct_insert=True)
        logger.info(
            "Populated PosIntervalMap for "
            + f'{nwb_file_name}, {key["interval_list_name"]}'
        )


def get_pos_interval_list_names(nwb_file_name) -> list:
    return [
        interval_list_name
        for interval_list_name in (
            IntervalList & {"nwb_file_name": nwb_file_name}
        ).fetch("interval_list_name")
        if PositionSource._is_valid_name(interval_list_name)
    ]


def convert_epoch_interval_name_to_position_interval_name(
    key: dict, populate_missing: bool = True
) -> str:
    """Converts IntervalList key to the corresponding position interval name.

    Parameters
    ----------
    key : dict
        Lookup key
    populate_missing: bool
        Whether to populate PositionIntervalMap for the key if missing. Should
        be False if this function is used inside of another populate call.
        Defaults to True

    Returns
    -------
    position_interval_name : str
    """
    # get the interval list name if given epoch but not interval list name
    if "interval_list_name" not in key and "epoch" in key:
        key["interval_list_name"] = get_interval_list_name_from_epoch(
            key["nwb_file_name"], key["epoch"]
        )

    pos_query = PositionIntervalMap & key

    if len(pos_query) == 0:
        if populate_missing:
            PositionIntervalMap()._no_transaction_make(key)
        else:
            raise KeyError(
                f"{key} must be populated in the PositionIntervalMap table "
                + "prior to your current populate call"
            )

    if len(pos_query) == 0:
        logger.info(f"No position intervals found for {key}")
        return []

    if len(pos_query) == 1:
        return pos_query.fetch1("position_interval_name")


def get_interval_list_name_from_epoch(nwb_file_name: str, epoch: int) -> str:
    """Returns the interval list name for the given epoch.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file.
    epoch : int
        The epoch number.

    Returns
    -------
    interval_list_name : str
        The interval list name.
    """
    interval_names = (
        TaskEpoch & {"nwb_file_name": nwb_file_name, "epoch": epoch}
    ).fetch("interval_list_name")

    if len(interval_names) != 1:
        logger.info(
            f"Found {len(interval_names)} interval list names found for "
            + f"{nwb_file_name} epoch {epoch}"
        )
        return None

    return interval_names[0]


def populate_position_interval_map_session(nwb_file_name: str):
    for interval_name in (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch(
        "interval_list_name"
    ):
        PositionIntervalMap.populate(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        )
