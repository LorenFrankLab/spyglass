import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import datajoint as dj
import hdmf
import numpy as np
import probeinterface as pi
import pynwb
import spikeinterface as si
import spikeinterface.extractors as se
from h5py import File as H5File
from hdmf.data_utils import GenericDataChunkIterator
from tqdm import tqdm

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_device import Probe
from spyglass.common.common_ephys import Electrode, Raw  # noqa: F401
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_lab import LabTeam
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.settings import analysis_dir, test_mode
from spyglass.spikesorting.utils import (
    _get_recording_timestamps,
    get_group_by_shank,
)
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_hash import NwbfileHasher

schema = dj.schema("spikesorting_v1_recording")


@schema
class SortGroup(SpyglassMixin, dj.Manual):
    definition = """
    # Set of electrodes to spike sort together
    -> Session
    sort_group_id: int
    ---
    sort_reference_electrode_id = -1: int  # the electrode to use for referencing
                                           # -1: no reference, -2: common median
    """

    class SortGroupElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> SortGroup
        -> Electrode
        """

    @classmethod
    def set_group_by_shank(
        cls,
        nwb_file_name: str,
        references: dict = None,
        omit_ref_electrode_group=False,
        omit_unitrode=True,
    ):
        """Divides electrodes into groups based on their shank position.

        * Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a
          single group
        * Electrodes from probes with multiple shanks (e.g. polymer probes) are
          placed in one group per shank
        * Bad channels are omitted

        Parameters
        ----------
        nwb_file_name : str
            the name of the NWB file whose electrodes should be put into
            sorting groups
        references : dict, optional
            If passed, used to set references. Otherwise, references set using
            original reference electrodes from config. Keys: electrode groups.
            Values: reference electrode.
        omit_ref_electrode_group : bool
            Optional. If True, no sort group is defined for electrode group of
            reference.
        omit_unitrode : bool
            Optional. If True, no sort groups are defined for unitrodes.
        """
        existing_entries = SortGroup & {"nwb_file_name": nwb_file_name}
        if existing_entries and test_mode:
            return
        elif existing_entries:
            # delete any current groups
            (SortGroup & {"nwb_file_name": nwb_file_name}).delete()

        sg_keys, sge_keys = get_group_by_shank(
            nwb_file_name=nwb_file_name,
            references=references,
            omit_ref_electrode_group=omit_ref_electrode_group,
            omit_unitrode=omit_unitrode,
        )
        cls.insert(sg_keys, skip_duplicates=True)
        cls.SortGroupElectrode().insert(sge_keys, skip_duplicates=True)


@schema
class SpikeSortingPreprocessingParameters(SpyglassMixin, dj.Lookup):
    """Parameters for preprocessing a recording prior to spike sorting.

    Attributes
    ----------
    preproc_param_name : str
        Name of the preprocessing parameters
    preproc_params : dict
        Dictionary of preprocessing parameters
        frequency_min: float
            High pass filter value in Hz
        frequency_max: float
            Low pass filter value in Hz
        margin_ms: float
            Margin in ms on border to avoid border effect
        seed: int
            Random seed for whitening
        min_segment_length: float
            Minimum segment length in seconds
    """

    definition = """
    # Parameters for denoising a recording prior to spike sorting.
    preproc_param_name: varchar(200)
    ---
    preproc_params: blob
    """

    contents = [
        [
            "default",
            {
                "frequency_min": 300,  # high pass filter value
                "frequency_max": 6000,  # low pass filter value
                "margin_ms": 5,  # margin in ms on border to avoid border effect
                "seed": 0,  # random seed for whitening
                "min_segment_length": 1,  # minimum segment length in seconds
            },
        ]
    ]

    @classmethod
    def insert_default(cls):
        """Insert default parameters."""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SpikeSortingRecordingSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Raw voltage traces and parameters. Use `insert_selection` method to insert rows.
    recording_id: uuid
    ---
    -> Raw
    -> SortGroup
    -> IntervalList
    -> SpikeSortingPreprocessingParameters
    -> LabTeam
    """

    _parallel_make = True

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into SpikeSortingRecordingSelection with an
        automatically generated unique recording ID as the sole primary key.

        Parameters
        ----------
        key : dict
            primary key of Raw, SortGroup, IntervalList,
            SpikeSortingPreprocessingParameters, LabTeam tables

        Returns
        -------
            primary key of SpikeSortingRecordingSelection table
        """
        query = cls & key
        if query:
            logger.warning("Similar row(s) already inserted.")
            return query.fetch(as_dict=True)
        key["recording_id"] = uuid.uuid4()
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class SpikeSortingRecording(SpyglassMixin, dj.Computed):
    definition = """
    # Processed recording.
    -> SpikeSortingRecordingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the processed recording in NWB file
    electrodes_id=null: varchar(40) # Object ID for the processed electrodes
    hash=null: varchar(32) # Hash of the NWB file
    """

    _use_transaction, _allow_insert = False, True

    def make(self, key):
        """Populate SpikeSortingRecording.

        1. Get valid times for sort interval from IntervalList
        2. Use spikeinterface to preprocess recording
        3. Write processed recording to NWB file
        4. Insert resulting ...
            - Interval to IntervalList
            - NWB file to AnalysisNwbfile
            - Recording ids to SpikeSortingRecording
        """
        nwb_file_name = (SpikeSortingRecordingSelection & key).fetch1(
            "nwb_file_name"
        )

        key.update(self._make_file(key))

        # INSERT:
        # - valid times into IntervalList
        # - analysis NWB file holding processed recording into AnalysisNwbfile
        # - entry into SpikeSortingRecording
        sort_interval_valid_times = self._get_sort_interval_valid_times(key)
        sort_interval_valid_times.set_key(
            nwb_file_name=nwb_file_name,
            interval_list_name=key["recording_id"],
            pipeline="spikesorting_recording_v1",
        )
        IntervalList.insert1(
            sort_interval_valid_times.as_dict, skip_duplicates=True
        )
        AnalysisNwbfile().add(nwb_file_name, key["analysis_file_name"])

        self.insert1(key)
        self._record_environment(key)

    def _record_environment(self, key):
        """Record environment details for this recording."""
        from spyglass.spikesorting.v1 import recompute as rcp

        rcp.RecordingRecomputeSelection().insert(key, at_creation=True)

    @classmethod
    def _make_file(
        cls,
        key: dict = None,
        recompute_file_name: str = None,
        save_to: Union[str, Path] = None,
        rounding: int = 4,
    ):
        """Preprocess recording and write to NWB file.

        All `_make_file` methods should exit early if the file already exists.

        - Get valid times for sort interval from IntervalList
        - Preprocess recording
        - Write processed recording to NWB file

        Parameters
        ----------
        key : dict
            primary key of SpikeSortingRecordingSelection table
        recompute_file_name : str, Optional
            If specified, recompute this file. Use as resulting file name.
            If none, generate a new file name. Used for recomputation after
            typical deletion.
        save_to : Union[str,Path], Optional
            Default None, save to analysis directory. If provided, save to
            specified path. Used for recomputation prior to deletion.
        """
        if not key and not recompute_file_name:
            raise ValueError(
                "Either key or recompute_file_name must be specified."
            )
        if isinstance(key, dict):
            key = {k: v for k, v in key.items() if k in cls.primary_key}

        hash = None
        recompute = recompute_file_name and not key and not save_to

        if recompute or save_to:  # if we expect file to exist
            file_path = AnalysisNwbfile.get_abs_path(
                recompute_file_name, from_schema=True
            )
        if recompute:  # If recompute, check if file exists
            if Path(file_path).exists():  # No need to recompute
                return
            logger.info(f"Recomputing {recompute_file_name}.")
            query = cls & {"analysis_file_name": recompute_file_name}
            # Use deleted file's ids and hash for recompute
            key, recompute_object_id, recompute_electrodes_id = query.fetch1(
                "KEY", "object_id", "electrodes_id"
            )
        elif save_to:  # recompute prior to deletion, save copy to temp_dir
            elect_id = cls._validate_file(file_path)
            obj_id = (cls & key).fetch1("object_id")
            recompute_object_id, recompute_electrodes_id = obj_id, elect_id
        else:
            recompute_object_id, recompute_electrodes_id = None, None

        parent = SpikeSortingRecordingSelection & key
        (recording_nwb_file_name, recording_object_id, electrodes_id) = (
            _write_recording_to_nwb(
                **cls()._get_preprocessed_recording(key),
                nwb_file_name=parent.fetch1("nwb_file_name"),
                recompute_file_name=recompute_file_name,
                recompute_object_id=recompute_object_id,
                recompute_electrodes_id=recompute_electrodes_id,
                save_to=save_to,
            )
        )

        hash = AnalysisNwbfile().get_hash(
            recording_nwb_file_name,
            from_schema=True,
            precision_lookup=dict(ProcessedElectricalSeries=rounding),
            return_hasher=bool(save_to),
        )

        if recompute:
            AnalysisNwbfile()._update_external(recompute_file_name, hash)

        return dict(
            analysis_file_name=recording_nwb_file_name,
            object_id=recording_object_id,
            electrodes_id=electrodes_id,
            hash=hash,
        )

    @classmethod
    def _validate_file(self, file_path: str) -> str:
        """Validate the NWB file exists and contains required upstream data.

        Parameters
        ----------
        file_path : str
            path to the NWB file to validate

        Returns
        -------
        elect_id : str
            ProcessedElectricalSeries/electrodes object ID

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        KeyError
            If the file does not contain electrodes or upstream objects
        """
        elect_attr = "acquisition/ProcessedElectricalSeries/electrodes"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        # ensure partial objects are present in existing file
        with H5File(file_path, "r") as f:
            elect_parts = elect_attr.split("/")
            for i in range(len(elect_parts)):
                mid_path = "/".join(elect_parts[: i + 1])
                if mid_path in f.keys():
                    continue
                raise KeyError(f"H5 object missing, {mid_path}: {file_path}")
            elect_id = f[elect_attr].attrs["object_id"]

        return elect_id

    @classmethod
    def get_recording(cls, key: dict) -> si.BaseRecording:
        """Get recording related to this curation as spikeinterface BaseRecording

        Parameters
        ----------
        key : dict
            primary key of SpikeSorting table
        """
        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )

        if not Path(analysis_file_abs_path).exists():
            cls._make_file(key, recompute_file_name=analysis_file_name)

        recording = se.read_nwb_recording(
            analysis_file_abs_path, load_time_vector=True
        )

        return recording

    @staticmethod
    def _get_recording_timestamps(recording):
        return _get_recording_timestamps(recording)

    def _get_sort_interval_valid_times(self, key: dict):
        """Identifies the intersection between sort interval specified by the user
        and the valid times (times for which neural data exist, excluding e.g. dropped packets).

        Parameters
        ----------
        key: dict
            primary key of SpikeSortingRecordingSelection table

        Returns
        -------
        sort_interval_valid_times: ndarray of tuples
            (start, end) times for valid intervals in the sort interval

        """
        # FETCH: - sort interval - valid times - preprocessing parameters
        nwb_file_name, sort_interval_name, params = (
            SpikeSortingPreprocessingParameters * SpikeSortingRecordingSelection
            & key
        ).fetch1("nwb_file_name", "interval_list_name", "preproc_params")

        sort_interval = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": sort_interval_name,
            }
        ).fetch_interval()

        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch_interval()

        # DO: - take intersection between sort interval and valid times
        return sort_interval.intersect(
            valid_interval_times, min_length=params["min_segment_length"]
        )

    def _get_preprocessed_recording(self, key: dict):
        """Filters and references a recording.

        - Loads the NWB file created during insertion as a spikeinterface Recording
        - Slices recording in time (interval) and space (channels);
          recording chunks from disjoint intervals are concatenated
        - Applies referencing and bandpass filtering

        Parameters
        ----------
        key: dict
            primary key of SpikeSortingRecordingSelection table

        Returns
        -------
        recording: si.Recording
        """
        # FETCH:
        # - full path to NWB file
        # - channels to be included in the sort
        # - the reference channel
        # - probe type
        # - filter parameters

        # TODO: Reduce number of fetches
        nwb_file_name = (SpikeSortingRecordingSelection & key).fetch1(
            "nwb_file_name"
        )
        sort_group_id = (SpikeSortingRecordingSelection & key).fetch1(
            "sort_group_id"
        )
        nwb_file_abs_path = Nwbfile().get_abs_path(nwb_file_name)
        channel_ids = (
            SortGroup.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch("electrode_id")
        ref_channel_id = (
            SortGroup
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch1("sort_reference_electrode_id")
        recording_channel_ids = np.setdiff1d(channel_ids, ref_channel_id)
        all_channel_ids = np.unique(np.append(channel_ids, ref_channel_id))

        # Electrode's fk to probe is nullable, so we need to check if present
        query = Electrode * Probe & {"nwb_file_name": nwb_file_name}
        if len(query) == 0:
            raise ValueError(f"No probe info found for {nwb_file_name}.")

        probe_type_by_channel = []
        electrode_group_by_channel = []
        for channel_id in channel_ids:
            # TODO: limit to one unique fetch. use set
            c_query = query & {"electrode_id": channel_id}
            probe_type_by_channel.append(c_query.fetch1("probe_type"))
            electrode_group_by_channel.append(
                c_query.fetch1("electrode_group_name")
            )

        probe_type = np.unique(probe_type_by_channel)
        filter_params = (
            SpikeSortingPreprocessingParameters * SpikeSortingRecordingSelection
            & key
        ).fetch1("preproc_params")

        # DO:
        # - load NWB file as a spikeinterface Recording
        # - slice the recording object in time and channels
        # - apply referencing depending on the option chosen by the user
        # - apply bandpass filter
        # - set probe to recording
        recording = se.read_nwb_recording(
            nwb_file_abs_path, load_time_vector=True
        )
        all_timestamps = recording.get_times()

        # Note: _consolidate_intervals is only used in spike sorting.v1
        valid_sort_times = self._get_sort_interval_valid_times(key).times
        valid_sort_times_indices = _consolidate_intervals(
            valid_sort_times, all_timestamps
        )

        # slice in time; concatenate disjoint sort intervals
        if len(valid_sort_times_indices) > 1:
            recordings_list = []
            timestamps = []
            for interval_indices in valid_sort_times_indices:
                recording_single = recording.frame_slice(
                    start_frame=interval_indices[0],
                    end_frame=interval_indices[1],
                )
                recordings_list.append(recording_single)
                timestamps.extend(
                    all_timestamps[interval_indices[0] : interval_indices[1]]
                )
            recording = si.concatenate_recordings(recordings_list)
        else:
            recording = recording.frame_slice(
                start_frame=valid_sort_times_indices[0][0],
                end_frame=valid_sort_times_indices[0][1],
            )
            timestamps = all_timestamps[
                valid_sort_times_indices[0][0] : valid_sort_times_indices[0][1]
            ]

        # slice in channels; include ref channel in first slice, then exclude it in second slice
        spyglass_ids = (
            all_channel_ids if ref_channel_id >= 0 else recording_channel_ids
        )
        spikeinterface_ids = self._get_spikeinterface_channel_ids(
            nwb_file_name, spyglass_ids
        )
        recording = recording.channel_slice(
            channel_ids=spikeinterface_ids,
            renamed_channel_ids=spyglass_ids,
        )

        if ref_channel_id >= 0:
            recording = si.preprocessing.common_reference(
                recording,
                reference="single",
                ref_channel_ids=ref_channel_id,
                dtype=np.float64,
            )
            recording = recording.channel_slice(
                channel_ids=recording_channel_ids
            )
        elif ref_channel_id == -2:
            recording = si.preprocessing.common_reference(
                recording,
                reference="global",
                operator="median",
                dtype=np.float64,
            )
        elif ref_channel_id != -1:
            raise ValueError(
                "Invalid reference channel ID. Use -1 to skip referencing. Use "
                + "-2 to reference via global median. Use positive integer to "
                + "reference to a specific channel."
            )

        recording = si.preprocessing.bandpass_filter(
            recording,
            freq_min=filter_params["frequency_min"],
            freq_max=filter_params["frequency_max"],
            dtype=np.float64,
        )

        # if the sort group is a tetrode, change the channel location
        # (necessary because the channel location for tetrodes are not set properly)
        if (
            len(probe_type) == 1
            and probe_type[0] == "tetrode_12.5"
            and len(recording_channel_ids) == 4
            and len(np.unique(electrode_group_by_channel)) == 1
        ):
            tetrode = pi.Probe(ndim=2)
            position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
            tetrode.set_contacts(
                position, shapes="circle", shape_params={"radius": 6.25}
            )
            tetrode.set_contact_ids(channel_ids)
            tetrode.set_device_channel_indices(np.arange(4))
            recording = recording.set_probe(tetrode, in_place=True)

        return dict(recording=recording, timestamps=np.asarray(timestamps))

    def update_ids(self):
        """Update electrodes_id, and hash in SpikeSortingRecording table.

        Only used for transitioning to recompute NWB files, see #1093.
        """
        elect_attr = "acquisition/ProcessedElectricalSeries/electrodes"
        needs_update = self & "electrodes_id is NULL or hash is NULL"

        for key in tqdm(needs_update):
            try:
                analysis_file_path = AnalysisNwbfile.get_abs_path(
                    key["analysis_file_name"]
                )
            except dj.DataJointError:  # file checksum error
                continue  # pragma: no cover
            with H5File(analysis_file_path, "r") as f:
                elect_id = f[elect_attr].attrs["object_id"]

            updated = dict(
                key,
                electrodes_id=elect_id,
                hash=NwbfileHasher(analysis_file_path).hash,
            )

            self.update1(updated)

    def recompute(self, key: dict):
        """Recompute the processed recording.

        Parameters
        ----------
        key : dict
            primary key of SpikeSortingRecording table
        """
        raise NotImplementedError("Recompute not implemented.")

    @staticmethod
    def _get_spikeinterface_channel_ids(
        nwb_file_name: str, channel_ids: List[Union[int, str]]
    ):
        """Given a file name and channel ids, return channel names.

        SpikeInterface uses channel_names instead of index number if present in
        nwb electrodes table. This function ensures match in channel_id values
        for indexing.

        Parameters
        ----------
        nwb_file_name : str
            file name of the NWB file
        channel_ids : List[int]
            list of channel indexes (electrode_id) to be converted

        Returns
        -------
        List[Union[int, str]]
            list of channel_id values used by spikeinterface
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
            nwbfile = io.read()
            electrodes_table = nwbfile.electrodes
            if "channel_name" not in electrodes_table.colnames:
                return channel_ids
            channel_names = electrodes_table["channel_name"]
            return [channel_names[ch] for ch in channel_ids]


def _consolidate_intervals(intervals, timestamps):
    """Convert a list of intervals (start_time, stop_time)
    to a list of intervals (start_index, stop_index) by comparing to a list of
    timestamps; then consolidates overlapping or adjacent intervals

    Parameters
    ----------
    intervals : iterable of tuples
    timestamps : numpy.ndarray

    """
    # Convert intervals to a numpy array if it's not
    intervals = np.array(intervals)
    if intervals.ndim == 1:
        intervals = intervals.reshape(-1, 2)
    if intervals.shape[1] != 2:
        raise ValueError("Input array must have shape (N_Intervals, 2).")

    # Check if intervals are sorted. If not, sort them.
    if not np.all(intervals[:-1] <= intervals[1:]):
        intervals = np.sort(intervals, axis=0)

    # Initialize an empty list to store the consolidated intervals
    consolidated = []

    # Convert start and stop times to indices
    start_indices = np.searchsorted(timestamps, intervals[:, 0], side="left")
    stop_indices = (
        np.searchsorted(timestamps, intervals[:, 1], side="right") - 1
    )

    # Start with the first interval
    start, stop = start_indices[0], stop_indices[0]

    # Loop through the rest of the intervals to join them if needed
    for next_start, next_stop in zip(start_indices, stop_indices):
        # If the stop time of the current interval is equal to or greater than
        # the next start time minus 1
        if stop >= next_start - 1:
            # Extend the current interval to include the next one
            stop = max(stop, next_stop)
        else:
            # Add the current interval to the consolidated list
            consolidated.append((start, stop))
            start, stop = next_start, next_stop  # Start a new interval

    # Add the last interval to the consolidated list
    consolidated.append((start, stop))

    # Convert the consolidated list to a NumPy array and return
    return np.array(consolidated)


def _write_recording_to_nwb(
    recording: si.BaseRecording,
    timestamps: Iterable,
    nwb_file_name: str,
    recompute_file_name: Optional[str] = None,
    recompute_object_id: Optional[str] = None,
    recompute_electrodes_id: Optional[str] = None,
    save_to: Union[str, Path] = None,
):
    """Write a recording in NWB format

    Parameters
    ----------
    recording : si.Recording
    timestamps : iterable
    nwb_file_name : str
        name of NWB file the recording originates
    recompute_file_name : str, optional
        name of the NWB file to recompute
    recompute_object_id : str, optional
        object ID for recomputed processed electrical series object,
        acquisition/ProcessedElectricalSeries.
    recompute_electrodes_id : str, optional
        object ID for recomputed electrodes sub-object,
        acquisition/ProcessedElectricalSeries/electrodes.
    save_to : Union[str, Path], optional
        Default None, save to analysis directory. If provided, save to specified
        path. For use in recompute prior to deletion.

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the preprocessed recording
    """

    recompute_args = (
        recompute_file_name,
        recompute_object_id,
        recompute_electrodes_id,
    )
    recompute = any(recompute_args)
    if recompute and not all(recompute_args):
        raise ValueError(
            "If recomputing, must specify all of recompute_file_name, "
            "recompute_object_id, and recompute_electrodes_id."
        )

    series_name = "ProcessedElectricalSeries"
    series_attr = "acquisition/" + series_name
    elect_attr = series_attr + "/electrodes"

    analysis_nwb_file = AnalysisNwbfile().create(
        nwb_file_name=nwb_file_name,
        recompute_file_name=recompute_file_name,
        alternate_dir=save_to,
    )
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(
        analysis_nwb_file,
        from_schema=recompute,
    )

    abs_obj = Path(analysis_nwb_file_abs_path)
    if save_to and abs_obj.is_relative_to(analysis_dir):
        relative = abs_obj.relative_to(analysis_dir)
        analysis_nwb_file_abs_path = Path(save_to) / relative

    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        table_region = nwbfile.create_electrode_table_region(
            region=[i for i in recording.get_channel_ids()],
            description="Sort group",
        )
        data_iterator = SpikeInterfaceRecordingDataChunkIterator(
            recording=recording, return_scaled=False, buffer_gb=5
        )
        timestamps_iterator = TimestampsDataChunkIterator(
            recording=TimestampsExtractor(timestamps), buffer_gb=5
        )
        processed_electrical_series = pynwb.ecephys.ElectricalSeries(
            name="ProcessedElectricalSeries",
            data=data_iterator,
            electrodes=table_region,
            timestamps=timestamps_iterator,
            filtering="Bandpass filtered for spike band",
            description="Referenced and filtered recording from "
            + f"{nwb_file_name} for spike sorting",
            conversion=np.unique(recording.get_channel_gains())[0] * 1e-6,
        )

        nwbfile.add_acquisition(processed_electrical_series)

        recording_object_id = nwbfile.acquisition[series_name].object_id
        electrodes_id = nwbfile.acquisition[series_name].electrodes.object_id

        io.write(nwbfile)

    if recompute_object_id:
        # logger.info(f"Recomputed {recompute_file_name}, fixing object IDs.")
        with H5File(analysis_nwb_file_abs_path, "a") as f:
            f[series_attr].attrs["object_id"] = recompute_object_id
            f[elect_attr].attrs["object_id"] = recompute_electrodes_id
        recording_object_id = recompute_object_id
        electrodes_id = recompute_electrodes_id
        analysis_nwb_file = recompute_file_name

    return analysis_nwb_file, recording_object_id, electrodes_id


# For writing recording to NWB file


class SpikeInterfaceRecordingDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator specifically for use on RecordingExtractor objects."""

    def __init__(
        self,
        recording: si.BaseRecording,
        segment_index: int = 0,
        return_scaled: bool = False,
        buffer_gb: Optional[float] = None,
        buffer_shape: Optional[tuple] = None,
        chunk_mb: Optional[float] = None,
        chunk_shape: Optional[tuple] = None,
        display_progress: bool = False,
        progress_bar_options: Optional[dict] = None,
    ):
        """
        Initialize an Iterable object which returns DataChunks with data and their selections on each iteration.

        Parameters
        ----------
        recording : si.BaseRecording
            The SpikeInterfaceRecording object which handles the data access.
        segment_index : int, optional
            The recording segment to iterate on.
            Defaults to 0.
        return_scaled : bool, optional
            Whether to return the trace data in scaled units (uV, if True) or in the raw data type (if False).
            Defaults to False.
        buffer_gb : float, optional
            The upper bound on size in gigabytes (GB) of each selection from the iteration.
            The buffer_shape will be set implicitly by this argument.
            Cannot be set if `buffer_shape` is also specified.
            The default is 1GB.
        buffer_shape : tuple, optional
            Manual specification of buffer shape to return on each iteration.
            Must be a multiple of chunk_shape along each axis.
            Cannot be set if `buffer_gb` is also specified.
            The default is None.
        chunk_mb : float, optional
            The upper bound on size in megabytes (MB) of the internal chunk for the HDF5 dataset.
            The chunk_shape will be set implicitly by this argument.
            Cannot be set if `chunk_shape` is also specified.
            The default is 1MB, as recommended by the HDF5 group. For more details, see
            https://support.hdfgroup.org/HDF5/doc/TechNotes/TechNote-HDF5-ImprovingIOPerformanceCompressedDatasets.pdf
        chunk_shape : tuple, optional
            Manual specification of the internal chunk shape for the HDF5 dataset.
            Cannot be set if `chunk_mb` is also specified.
            The default is None.
        display_progress : bool, optional
            Display a progress bar with iteration rate and estimated completion time.
        progress_bar_options : dict, optional
            Dictionary of keyword arguments to be passed directly to tqdm.
            See https://github.com/tqdm/tqdm#parameters for options.
        """
        self.recording = recording
        self.segment_index = segment_index
        self.return_scaled = return_scaled
        self.channel_ids = recording.get_channel_ids()
        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_options=progress_bar_options,
        )

    def _get_data(self, selection: Tuple[slice]) -> Iterable:
        return self.recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self.channel_ids[selection[1]],
            start_frame=selection[0].start,
            end_frame=selection[0].stop,
            return_scaled=self.return_scaled,
        )

    def _get_dtype(self):
        return self.recording.get_dtype()

    def _get_maxshape(self):
        return (
            self.recording.get_num_samples(segment_index=self.segment_index),
            self.recording.get_num_channels(),
        )


class TimestampsExtractor(si.BaseRecording):
    def __init__(
        self,
        timestamps,
        sampling_frequency=30e3,
    ):
        si.BaseRecording.__init__(
            self, sampling_frequency, channel_ids=[0], dtype=np.float64
        )
        rec_segment = TimestampsSegment(
            timestamps=timestamps,
            sampling_frequency=sampling_frequency,
            t_start=None,
            dtype=np.float64,
        )
        self.add_recording_segment(rec_segment)


class TimestampsSegment(si.BaseRecordingSegment):
    def __init__(self, timestamps, sampling_frequency, t_start, dtype):
        si.BaseRecordingSegment.__init__(
            self, sampling_frequency=sampling_frequency, t_start=t_start
        )
        self._timeseries = timestamps

    def get_num_samples(self) -> int:
        """Return the number of samples in the segment."""
        return self._timeseries.shape[0]

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        """Return the traces for the segment for given start/end frames."""
        return np.squeeze(self._timeseries[start_frame:end_frame])


class TimestampsDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator specifically for use on RecordingExtractor objects."""

    def __init__(
        self,
        recording: si.BaseRecording,
        segment_index: int = 0,
        return_scaled: bool = False,
        buffer_gb: Optional[float] = None,
        buffer_shape: Optional[tuple] = None,
        chunk_mb: Optional[float] = None,
        chunk_shape: Optional[tuple] = None,
        display_progress: bool = False,
        progress_bar_options: Optional[dict] = None,
    ):
        """
        Initialize an Iterable object which returns DataChunks with data and their selections on each iteration.

        Parameters
        ----------
        recording : SpikeInterfaceRecording
            The SpikeInterfaceRecording object (RecordingExtractor or BaseRecording) which handles the data access.
        segment_index : int, optional
            The recording segment to iterate on.
            Defaults to 0.
        return_scaled : bool, optional
            Whether to return the trace data in scaled units (uV, if True) or in the raw data type (if False).
            Defaults to False.
        buffer_gb : float, optional
            The upper bound on size in gigabytes (GB) of each selection from the iteration.
            The buffer_shape will be set implicitly by this argument.
            Cannot be set if `buffer_shape` is also specified.
            The default is 1GB.
        buffer_shape : tuple, optional
            Manual specification of buffer shape to return on each iteration.
            Must be a multiple of chunk_shape along each axis.
            Cannot be set if `buffer_gb` is also specified.
            The default is None.
        chunk_mb : float, optional
            The upper bound on size in megabytes (MB) of the internal chunk for the HDF5 dataset.
            The chunk_shape will be set implicitly by this argument.
            Cannot be set if `chunk_shape` is also specified.
            The default is 1MB, as recommended by the HDF5 group. For more details, see
            https://support.hdfgroup.org/HDF5/doc/TechNotes/TechNote-HDF5-ImprovingIOPerformanceCompressedDatasets.pdf
        chunk_shape : tuple, optional
            Manual specification of the internal chunk shape for the HDF5 dataset.
            Cannot be set if `chunk_mb` is also specified.
            The default is None.
        display_progress : bool, optional
            Display a progress bar with iteration rate and estimated completion time.
        progress_bar_options : dict, optional
            Dictionary of keyword arguments to be passed directly to tqdm.
            See https://github.com/tqdm/tqdm#parameters for options.
        """
        self.recording = recording
        self.segment_index = segment_index
        self.return_scaled = return_scaled
        self.channel_ids = recording.get_channel_ids()
        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_options=progress_bar_options,
        )

    # change channel id to always be first channel
    def _get_data(self, selection: Tuple[slice]) -> Iterable:
        return self.recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=[0],
            start_frame=selection[0].start,
            end_frame=selection[0].stop,
            return_scaled=self.return_scaled,
        )

    def _get_dtype(self):
        return self.recording.get_dtype()

    # remove the last dim for the timestamps since it is always just a 1D vector
    def _get_maxshape(self):
        return (
            self.recording.get_num_samples(segment_index=self.segment_index),
        )
