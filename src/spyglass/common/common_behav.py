import pathlib
import re
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Union

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from pynwb.behavior import CompassDirection

from spyglass.common.common_device import CameraDevice
from spyglass.common.common_ephys import Raw  # noqa: F401
from spyglass.common.common_interval import Interval, IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_task import TaskEpoch
from spyglass.settings import video_dir
from spyglass.utils import SpyglassIngestion, SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import (
    _get_epoch_groups,
    _get_pos_dict,
    estimate_sampling_rate,
    get_nwb_file,
    get_position_obj,
    get_valid_intervals,
)

schema = dj.schema("common_behav")


@schema
class PositionSource(SpyglassIngestion, dj.Manual):
    definition = """
    -> Session
    -> IntervalList
    ---
    source: varchar(200)             # source of data (e.g., trodes, dlc)
    import_file_name: varchar(2000)  # path to import file if importing
    """

    # Position intervals may already exist from a previous ingestion, so an
    # existing entry is validated rather than reinserted. NOTE: the previous
    # implementation used IntervalList.cautious_insert(update=True), which
    # silently overwrote a differing entry; validation prompts instead.
    _expected_duplicates = True

    class SpatialSeries(SpyglassIngestion, dj.Part):
        definition = """
        -> master
        id = 0 : int unsigned            # index of spatial series
        ---
        name=null: varchar(32)       # name of spatial series
        """

        _expected_duplicates = True  # follows the master

    @property
    def _source_nwb_object_type(self):
        return pynwb.behavior.Position

    @property
    def table_key_to_obj_attr(self):
        """Entries are grouped by epoch in the override, not per column."""
        return {"self": dict()}

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """The file's Position interface holds every spatial series.

        `get_position_obj` finds series the default type filter would miss,
        and raises if a file declares more than one Position interface.
        """
        pos_interface = get_position_obj(nwb_file)
        return [pos_interface] if pos_interface is not None else []

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Group the file's spatial series into one source entry per epoch.

        Each epoch yields an IntervalList entry (the parent, first), a source
        entry, and one SpatialSeries part entry per series in that epoch.
        RawPosition and its PosObject part are filled from the same series --
        they hold the object ids this pass already read -- rather than making
        RawPosition parse the file a second time.
        """
        sess_key = dict(base_key or dict())
        nwb_file_name = sess_key["nwb_file_name"]
        src_key = dict(**sess_key, source="imported", import_file_name="")

        all_pos = _get_pos_dict(
            position=nwb_obj.spatial_series,
            epoch_groups=_get_epoch_groups(nwb_obj),
            session_id=nwb_file_name,
            verbose=True,
        )
        if len(all_pos) == 0:
            self._info_msg(
                f"No position data found in {nwb_file_name}. Skipping."
            )
            return {self: []}

        intervals, sources, spat_series = [], [], []
        raw_pos, pos_objects = [], []

        for epoch, epoch_list in all_pos.items():
            ind_key = dict(interval_list_name=self.get_pos_interval_name(epoch))

            sources.append(dict(**src_key, **ind_key))
            raw_pos.append(dict(**sess_key, **ind_key))
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
                pos_objects.append(
                    dict(
                        **sess_key,
                        **ind_key,
                        id=index,
                        raw_position_object_id=pdict["raw_position_object_id"],
                    )
                )

        return {
            IntervalList: intervals,
            self: sources,
            self.SpatialSeries: spat_series,
            RawPosition: raw_pos,
            RawPosition.PosObject: pos_objects,
        }

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Ingest, then map epoch intervals to the position intervals."""
        entries = super().insert_from_nwbfile(nwb_file_name, config, dry_run)

        if entries and not dry_run:
            populate_position_interval_map_session(nwb_file_name)

        return entries

    def make(self, keys: Union[List[Dict], dj.Table]):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "PositionSource.make is deprecated. Use insert_from_nwbfile."
        )

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
class RawPosition(SpyglassIngestion, dj.Imported):
    definition = """
    -> PositionSource
    """

    # Filled by PositionSource, which reads the same spatial series.
    _expected_duplicates = True

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Defer to PositionSource, which generates this table's entries.

        The two tables describe the same spatial series, so parsing the file
        once fills both. Kept so `populate` and direct callers still work.
        """
        return PositionSource().insert_from_nwbfile(
            nwb_file_name, config, dry_run
        )

    class PosObject(SpyglassIngestion, dj.Part):
        definition = """
        -> master
        -> PositionSource.SpatialSeries.proj('id')
        ---
        raw_position_object_id: varchar(40) # id of spatial series in NWB file
        """

        _nwb_table = Nwbfile
        _expected_duplicates = True  # follows the master

        def fetch1_dataframe(self):
            """Return a dataframe with all RawPosition.PosObject items."""
            id_rp = [(n["id"], n["raw_position"]) for n in self.fetch_nwb()]

            if len(set(rp.interval for _, rp in id_rp)) > 1:
                logger.warning("Loading DataFrame with multiple intervals.")

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
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "RawPosition.make is deprecated. Use insert_from_nwbfile."
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
class RawCompassDirection(SpyglassIngestion, dj.Manual):
    """
    Table to store raw CompassDirection data from NWB files.
    """

    definition = """
    -> Session
    -> IntervalList
    ---
    compass_object_id: varchar(40)  # the object id of the compass direction object
    name: varchar(80)              # name of the compass direction object
    """

    _nwb_table = Nwbfile
    _compass_import_enumerator = 1

    @property
    def _source_nwb_object_type(self):
        return CompassDirection

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "name": "name",
                "compass_object_id": "object_id",
                "valid_times": self.generate_valid_intervals_from_timeseries,
                "interval_list_name": self.enumerated_interval_name,
            }
        }

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Get all CompassDirection spatial series from NWB file, ordered by time."""
        compass_objects = super().get_nwb_objects(nwb_file, nwb_file_name)
        spatial_series = sum(
            [list(obj.spatial_series.values()) for obj in compass_objects], []
        )
        start_times = [ss.get_timestamps()[0] for ss in spatial_series]
        order = np.argsort(start_times)
        spatial_series = [spatial_series[i] for i in order]

        return spatial_series

    def enumerated_interval_name(
        self, obj: pynwb.behavior.SpatialSeries
    ) -> str:
        """Generate a unique interval list name for each compass direction object."""
        name = f"compass {self._compass_import_enumerator} valid times"
        self._compass_import_enumerator += 1
        return name

    @staticmethod
    def generate_valid_intervals_from_timeseries(
        nwb_obj: pynwb.behavior.SpatialSeries,
    ):
        """Generate valid intervals from spatial series.

        Parameters
        ----------
        nwb_obj : pynwb.behavior.SpatialSeries
            The pynwb.behavior.SpatialSeries NWB object.
        Returns
        -------
        valid_times : list
            List of valid time intervals.
        """
        timestamps = nwb_obj.get_timestamps()
        sampling_rate = estimate_sampling_rate(
            timestamps, filename=nwb_obj.name
        )
        valid_times = get_valid_intervals(
            timestamps=timestamps,
            sampling_rate=sampling_rate,
            min_valid_len=int(sampling_rate),
        )
        return valid_times

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Add IntervalList entry to the generated entries."""
        super_ins = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = super_ins[self][0]
        interval_insert = {
            k: v for k, v in self_key.items() if k in IntervalList.heading.names
        }
        self_key.pop(
            "valid_times", None
        )  # remove valid_times from the insert to this table
        return {
            IntervalList: [interval_insert],
            **super_ins,
        }

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Insert entries from NWB file, generating interval list names ordered by time."""
        self._compass_import_enumerator = 1  # reset enumerator
        return super().insert_from_nwbfile(nwb_file_name, config, dry_run)


@schema
class StateScriptFile(SpyglassIngestion, dj.Imported):
    definition = """
    -> TaskEpoch
    ---
    file_object_id: varchar(40)  # the object id of the file object
    """

    _nwb_table = Nwbfile
    # Exact class name, as ndx_franklab_novela may not be importable
    _source_nwb_object_type = "AssociatedFiles"

    # An associated file is a state script if its description says so.
    # Matching ignores case and spaces, so "STATE SCRIPT" is covered too.
    _source_nwb_object_description = ("state script", "state_script")

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"file_object_id": "object_id"}}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Expand one associated file into an entry per task epoch.

        The file names its epochs in a comma-separated string. Only epochs
        already present in TaskEpoch yield an entry, matching the previous
        implementation, which ran once per existing TaskEpoch key.
        """
        super_ins = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = super_ins[self][0]

        # TODO: update associated_file_obj.task_epochs to be an array of
        # 1-based ints, not a comma-separated string of ints
        named_epochs = str(nwb_obj.task_epochs).split(",")
        task_epochs = TaskEpoch & {
            "nwb_file_name": self_key.get("nwb_file_name")
        }

        return {
            self: [
                dict(self_key, epoch=epoch)
                for epoch in task_epochs.fetch("epoch")
                if str(epoch) in named_epochs
            ]
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "StateScriptFile.make is deprecated. Use insert_from_nwbfile."
        )


@schema
class VideoFile(SpyglassIngestion, dj.Imported):
    """Video file metadata from NWB ImageSeries.

    Notes
    -----
    The video timestamps come from: videoTimeStamps.cameraHWSync if PTP is
    used. If PTP is not used, the video timestamps come from
    videoTimeStamps.cameraHWFrameCount.

    **Issue #1444 Note:** VideoFile requires TaskEpoch entries to import videos.
    If your NWB file contains ImageSeries (video data) without task metadata,
    a warning will be issued.
    """

    definition = """
    -> TaskEpoch
    video_file_num = 0: int
    ---
    camera_name: varchar(80)
    video_file_object_id: varchar(40)  # the object id of the file object
    """

    _nwb_table = Nwbfile
    _timestamp_overlap_threshold = 0.9  # Min fraction of timestamps in epoch
    _epoch_cache = dict()  # nwb_file_name -> {epoch: valid times}
    _failed_videos = defaultdict(list)  # reset per ingested file
    _video_count = 0  # ImageSeries seen in the file being ingested
    _placed_videos = 0  # ImageSeries that landed in at least one epoch

    @property
    def _source_nwb_object_type(self):
        return pynwb.image.ImageSeries

    @property
    def table_key_to_obj_attr(self):
        """Entries are built per epoch in the override, not per column."""
        return {"self": dict()}

    def _epoch_intervals(self, nwb_file_name) -> dict:
        """Return the valid times of each task epoch, fetched once per file.

        A video belongs to whichever epoch its timestamps fall inside, so
        every epoch's times are needed to place a single video.

        Parameters
        ----------
        nwb_file_name : str
            The file being ingested.

        Returns
        -------
        dict
            Epoch number to that epoch's Interval.
        """
        if nwb_file_name not in self._epoch_cache:
            self._epoch_cache[nwb_file_name] = {
                epoch: (
                    IntervalList
                    & {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": interval_list_name,
                    }
                ).fetch_interval()
                for epoch, interval_list_name in zip(
                    *(TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch(
                        "epoch", "interval_list_name"
                    )
                )
            }
        return self._epoch_cache[nwb_file_name]

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Place one video in whichever epochs its timestamps overlap.

        Failures are collected rather than raised, matching the previous
        implementation, which reported them per file at the end.
        """
        base_key = base_key or dict()
        self._video_count += 1
        entries = []

        for epoch, valid_times in self._epoch_intervals(
            base_key["nwb_file_name"]
        ).items():
            key = dict(base_key, epoch=epoch)
            try:
                rows, failure_reason, overlap_percent = (
                    self._validate_video_timestamps(nwb_obj, valid_times, key)
                )
            except KeyError as err:  # camera device not in CameraDevice
                self._failed_videos["missing_camera"].append(
                    {
                        "name": nwb_obj.name,
                        "camera": getattr(
                            nwb_obj.device, "camera_name", "unknown"
                        ),
                        "error": str(err),
                    }
                )
                break  # the camera is missing for every epoch
            except Exception as err:
                self._failed_videos["other"].append(
                    {
                        "name": nwb_obj.name,
                        "error": f"{type(err).__name__}: {str(err)}",
                    }
                )
                break

            if failure_reason:
                self._failed_videos["timestamp_mismatch"].append(
                    {
                        "name": nwb_obj.name,
                        "reason": failure_reason,
                        "overlap_percent": overlap_percent,
                    }
                )
            else:
                entries.extend(rows)

        # Counted per source series, not per row: one video spanning several
        # epochs yields several rows, so a row count cannot tell whether
        # every series was placed.
        self._placed_videos += bool(entries)

        return {self: entries}

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Ingest, then report on any videos that could not be placed."""
        self._epoch_cache = dict()
        self._failed_videos = defaultdict(list)
        self._video_count = 0
        self._placed_videos = 0

        entries = super().insert_from_nwbfile(nwb_file_name, config, dry_run)

        if not self._video_count:
            self._warn_msg(
                f"No video data interface found in {nwb_file_name}\n"
            )
            return entries

        if self._placed_videos < self._video_count:
            self._report_partial_import(
                nwb_file_name,
                self._failed_videos,
                self._video_count,
                self._placed_videos,
            )

        return entries

    def _prepare_video_entry(
        self, key, video_obj, cam_device_regex: str = r"camera_device (\d+)"
    ):
        """Prepare a VideoFile entry dict for a given video object.

        Parameters
        ----------
        key : dict
            The primary key for the VideoFile entry
        video_obj : pynwb.image.ImageSeries
            The video object from the NWB file
        cam_device_regex : str, optional
            Regular expression pattern to extract camera device number.
            Default: r"camera_device (\\d+)"

        Returns
        -------
        dict
            Prepared entry dict ready for insertion

        Raises
        ------
        KeyError
            If camera_name is not found in CameraDevice table
        """
        nwb_cam_device = video_obj.device.name
        camera_name = video_obj.device.camera_name

        if not (CameraDevice & {"camera_name": camera_name}):
            raise KeyError(
                f"No camera with camera_name: {camera_name} found "
                "in CameraDevice table."
            )

        match = re.match(cam_device_regex, nwb_cam_device)
        if not match:
            raise ValueError(
                f"Camera device name '{nwb_cam_device}' does not match "
                f"expected pattern '{cam_device_regex}'"
            )

        return dict(
            key,
            video_file_num=int(match[1]),
            camera_name=camera_name,
            video_file_object_id=video_obj.object_id,
        )

    def _validate_video_timestamps(self, video_obj, valid_times, key):
        """Validate video timestamps and return entries or failure reason.

        Handles both single-file and multi-file ImageSeries. Validates that
        timestamps meet the overlap threshold with epoch intervals.

        Parameters
        ----------
        video_obj : pynwb.image.ImageSeries
            The video object from the NWB file
        valid_times : Interval
            Valid time intervals for the current epoch
        key : dict
            The primary key for the VideoFile entry

        Returns
        -------
        tuple
            (entries_list, failure_reason_or_None, overlap_percent)
            - If validation passes: ([entry_dicts], None, overlap_percent)
            - If validation fails: ([], "failure reason string", overlap_percent)
        """
        timestamps = video_obj.timestamps
        starting_frame = getattr(video_obj, "starting_frame", None)

        # Multi-file ImageSeries
        if starting_frame is not None and len(starting_frame) > 1:
            entries, overlap_pct = self._validate_multifile_timestamps(
                video_obj, timestamps, starting_frame, valid_times, key
            )
            if not entries:
                threshold_pct = self._timestamp_overlap_threshold * 100
                return (
                    [],
                    (
                        f"No file segments have ≥{threshold_pct:.0f}% "
                        "timestamp overlap with epoch"
                    ),
                    overlap_pct,
                )
            return entries, None, overlap_pct

        # Single-file ImageSeries: valid if >= threshold% of timestamps
        # overlap with epoch intervals (epoch covers the video).
        these_times = valid_times.contains(timestamps)
        overlap_pct = len(these_times) / len(timestamps)

        # Also valid if a single epoch interval is >= threshold% covered
        # by the video timestamps (video covers the whole epoch).
        timestamps_interval = [timestamps[0], timestamps[-1]]
        max_interval_overlap_pct = 0
        for interval in valid_times.times:
            interval_duration = interval[1] - interval[0]
            if interval_duration <= 0:
                continue
            overlap_start = max(interval[0], timestamps_interval[0])
            overlap_end = min(interval[1], timestamps_interval[1])
            overlap_duration = max(0, overlap_end - overlap_start)
            interval_overlap_pct = overlap_duration / interval_duration
            max_interval_overlap_pct = max(
                max_interval_overlap_pct, interval_overlap_pct
            )

        if (
            overlap_pct < self._timestamp_overlap_threshold
            and max_interval_overlap_pct < self._timestamp_overlap_threshold
        ):
            threshold_pct = self._timestamp_overlap_threshold * 100
            return (
                [],
                (
                    f"Only {overlap_pct:.1%} of timestamps overlap with epoch, "
                    f"and the best-covered epoch interval has only "
                    f"{max_interval_overlap_pct:.1%} of its duration covered "
                    f"by the timestamps (need ≥{threshold_pct:.0f}%)"
                ),
                overlap_pct,
            )

        return [self._prepare_video_entry(key, video_obj)], None, overlap_pct

    def _validate_multifile_timestamps(
        self,
        video_obj,
        timestamps,
        starting_frame,
        valid_times,
        key,
    ):
        """Validate each segment of multi-file ImageSeries timestamps.

        Parameters
        ----------
        video_obj : pynwb.image.ImageSeries
            The video object from the NWB file
        timestamps : array
            All timestamps for the ImageSeries
        starting_frame : array
            Frame indices indicating where each external file begins
        valid_times : Interval
            Valid time intervals for the current epoch
        key : dict
            The primary key for the VideoFile entry

        Returns
        -------
        tuple(list, float)
            List of entry dicts for segments with valid timestamps. May be empty
            Maximum overlap percentage across all file segments
        """
        entries = []

        max_overlap_pct = 0
        for file_idx in range(len(starting_frame)):
            # Determine timestamp range for this file segment
            start_idx = starting_frame[file_idx]
            end_idx = (
                starting_frame[file_idx + 1]
                if file_idx + 1 < len(starting_frame)
                else len(timestamps)
            )

            # Extract timestamps for this specific file
            file_timestamps = timestamps[start_idx:end_idx]

            # Check if threshold % of this file's timestamps overlap with epoch
            these_times = valid_times.contains(file_timestamps)
            overlap_pct = len(these_times) / len(file_timestamps)
            max_overlap_pct = max(max_overlap_pct, overlap_pct)

            if len(these_times) < (
                self._timestamp_overlap_threshold * len(file_timestamps)
            ):
                continue

            # This file segment matches the epoch - prepare VideoFile entry
            entry = self._prepare_video_entry(key.copy(), video_obj)
            entries.append(entry)

        return entries, max_overlap_pct

    def make(self, key, verbose=True, skip_duplicates=False):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "VideoFile.make is deprecated. Use insert_from_nwbfile."
        )

    @staticmethod
    def _report_partial_import(
        nwb_file_name, failed_videos, total_videos, imported_count
    ):
        """Report specific reasons for partial video import.

        Issue #1444: Provide detailed diagnostics for each video that wasn't
        imported, categorized by failure reason with specific details.

        Parameters
        ----------
        nwb_file_name : str
            Name of the NWB file
        failed_videos : dict
            Dictionary with keys 'timestamp_mismatch', 'missing_camera',
            'other', each containing list of failure details
        total_videos : int
            Total number of ImageSeries found
        imported_count : int
            Number of ImageSeries successfully imported
        """

        msg_parts = [
            f"{nwb_file_name}: VideoFile Partial Import",
            f"Imported {imported_count}/{total_videos} ImageSeries",
        ]

        if failed_videos["timestamp_mismatch"]:
            msg_parts.append("\nTimestamp mismatches:")
            for item in failed_videos["timestamp_mismatch"]:
                if item["overlap_percent"] == 0:
                    continue  # Don't report videos for other epochs as errors
                msg_parts.append(f"  - {item['name']}: {item['reason']}")

        if failed_videos["missing_camera"]:
            msg_parts.append("\nMissing camera devices:")
            for item in failed_videos["missing_camera"]:
                msg_parts.append(
                    f"  - {item['name']}: camera '{item['camera']}' "
                    "not in CameraDevice table"
                )

        if failed_videos["other"]:
            msg_parts.append("\nOther errors:")
            for item in failed_videos["other"]:
                msg_parts.append(f"  - {item['name']}: {item['error']}")

        logger.warning("\n".join(msg_parts))

    @classmethod
    def update_entries(cls, restrict=True):
        """Update the camera_name field for all entries in the table."""
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
    position_interval_name="": varchar(200)  # corresponding interval name
    """

    # #849 - Insert null to avoid rerun

    def make(self, key):
        """Make without transaction"""
        self._no_transaction_make(key)

    def _no_transaction_make(self, key):
        # Find correspondence between pos valid times names and epochs. Use
        # epsilon to tolerate small differences in epoch boundaries across
        # epoch/pos intervals

        if not self.connection.in_transaction:
            # if called w/o transaction, call add via `populate`
            self.populate(key)
            return
        if self & key:
            return

        # *** HARD CODED VALUES ***
        EPSILON = 0.51  # tolerated time diff in bounds across epoch/pos
        no_pop_msg = "CANNOT POPULATE PositionIntervalMap"

        # Strip extra info from key if not passed via populate call
        key = {k: v for k, v in key.items() if k in self.primary_key}

        nwb_file_name = key["nwb_file_name"]
        pos_intervals = get_pos_interval_list_names(nwb_file_name)
        null_key = dict(key, position_interval_name="")
        insert_opts = dict(allow_direct_insert=True, skip_duplicates=True)

        # Skip populating if no pos interval list names
        if len(pos_intervals) == 0:
            self._err_msg(f"NO POS INTERVALS FOR {key};\n{no_pop_msg}")
            self.insert1(null_key, **insert_opts)
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
            self._warn_msg(
                f"{no_pop_msg}. Found {len(matching_pos_intervals)} pos "
                + f"intervals for\n\t{key}\n\t"
                + f"Matching intervals: {matching_pos_intervals}"
            )
            self.insert1(null_key, **insert_opts)
            return

        # Insert into table
        self.insert1(
            dict(key, position_interval_name=matching_pos_intervals[0]),
            **insert_opts,
        )
        self._info_msg(
            "Populated PosIntervalMap for "
            + f'{nwb_file_name}, {key["interval_list_name"]}'
        )


def get_pos_interval_list_names(nwb_file_name) -> list:
    """Return a list of position interval list names for a given NWB file."""
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
    pos_str = "position_interval_name"

    no_entries = len(pos_query) == 0
    null_entry = pos_query.fetch(pos_str)[0] == "" if len(pos_query) else False

    if populate_missing and (no_entries or null_entry):
        if null_entry:
            pos_query.delete(
                force_permission=True, safemode=False
            )  # no prompt; bypass delete permission check for null placeholder entry
        PositionIntervalMap()._no_transaction_make(key)
        pos_query = PositionIntervalMap & key

    if pos_query.fetch(pos_str)[0] == "":
        logger.info(f"No position intervals found for {key}")
        return []

    if len(pos_query) == 1:
        return pos_query.fetch1("position_interval_name")

    else:
        raise ValueError(f"Multiple intervals found for {key}: {pos_query}")


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


def get_position_interval_epoch(
    nwb_file_name: str, position_interval_name: str
) -> int:
    """Return the epoch number for a given position interval name."""
    # look up the epoch
    key = dict(
        nwb_file_name=nwb_file_name,
        position_interval_name=position_interval_name,
    )
    query = PositionIntervalMap * TaskEpoch & key
    if query:
        return query.fetch1("epoch")
    # if no match, make sure all epoch interval names are mapped
    for epoch_key in (TaskEpoch() & key).fetch(
        "nwb_file_name", "interval_list_name", as_dict=True
    ):
        convert_epoch_interval_name_to_position_interval_name(epoch_key)
    # try again
    query = PositionIntervalMap * TaskEpoch & key
    if query:
        return query.fetch1("epoch")
    return None


def populate_position_interval_map_session(nwb_file_name: str):
    """Populate PositionIntervalMap for all epochs in a given NWB file."""
    # 1. remove redundancy in interval names
    # 2. let PositionIntervalMap handle transaction context
    nwb_dict = dict(nwb_file_name=nwb_file_name)
    intervals = (TaskEpoch & nwb_dict).fetch("interval_list_name")
    for interval_name in set(intervals):
        interval_dict = dict(interval_list_name=interval_name)
        if PositionIntervalMap & interval_dict:
            continue
        PositionIntervalMap().make(dict(nwb_dict, **interval_dict))
