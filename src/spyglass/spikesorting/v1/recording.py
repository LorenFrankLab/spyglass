import uuid
from time import time
from typing import Iterable, List, Optional, Tuple, Union

import datajoint as dj
import numpy as np
import probeinterface as pi
import pynwb
import spikeinterface as si
import spikeinterface.extractors as se
from hdmf.data_utils import GenericDataChunkIterator

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_device import Probe
from spyglass.common.common_ephys import Electrode, Raw  # noqa: F401
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_intersect,
)
from spyglass.common.common_lab import LabTeam
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.utils import SpyglassMixin, logger

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
        # delete any current groups
        # (SortGroup & {"nwb_file_name": nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (
            Electrode()
            & {"nwb_file_name": nwb_file_name}
            & {"bad_channel": "False"}
        ).fetch()
        e_groups = list(np.unique(electrodes["electrode_group_name"]))
        e_groups.sort(key=int)  # sort electrode groups numerically
        sort_group = 0
        sg_key = dict()
        sge_key = dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name
        for e_group in e_groups:
            # for each electrode group, get a list of the unique shank numbers
            shank_list = np.unique(
                electrodes["probe_shank"][
                    electrodes["electrode_group_name"] == e_group
                ]
            )
            sge_key["electrode_group_name"] = e_group
            # get the indices of all electrodes in this group / shank and set their sorting group
            for shank in shank_list:
                sg_key["sort_group_id"] = sge_key["sort_group_id"] = sort_group
                # specify reference electrode. Use 'references' if passed, otherwise use reference from config
                if not references:
                    shank_elect_ref = electrodes[
                        "original_reference_electrode"
                    ][
                        np.logical_and(
                            electrodes["electrode_group_name"] == e_group,
                            electrodes["probe_shank"] == shank,
                        )
                    ]
                    if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                        sg_key["sort_reference_electrode_id"] = shank_elect_ref[
                            0
                        ]
                    else:
                        ValueError(
                            f"Error in electrode group {e_group}: reference "
                            + "electrodes are not all the same"
                        )
                else:
                    if e_group not in references.keys():
                        raise Exception(
                            f"electrode group {e_group} not a key in "
                            + "references, so cannot set reference"
                        )
                    else:
                        sg_key["sort_reference_electrode_id"] = references[
                            e_group
                        ]
                # Insert sort group and sort group electrodes
                reference_electrode_group = electrodes[
                    electrodes["electrode_id"]
                    == sg_key["sort_reference_electrode_id"]
                ][
                    "electrode_group_name"
                ]  # reference for this electrode group
                if (
                    len(reference_electrode_group) == 1
                ):  # unpack single reference
                    reference_electrode_group = reference_electrode_group[0]
                elif (int(sg_key["sort_reference_electrode_id"]) > 0) and (
                    len(reference_electrode_group) != 1
                ):
                    raise Exception(
                        "Should have found exactly one electrode group for "
                        + "reference electrode, but found "
                        + f"{len(reference_electrode_group)}."
                    )
                if omit_ref_electrode_group and (
                    str(e_group) == str(reference_electrode_group)
                ):
                    logger.warn(
                        f"Omitting electrode group {e_group} from sort groups "
                        + "because contains reference."
                    )
                    continue
                shank_elect = electrodes["electrode_id"][
                    np.logical_and(
                        electrodes["electrode_group_name"] == e_group,
                        electrodes["probe_shank"] == shank,
                    )
                ]
                if (
                    omit_unitrode and len(shank_elect) == 1
                ):  # omit unitrodes if indicated
                    logger.warn(
                        f"Omitting electrode group {e_group}, shank {shank} "
                        + "from sort groups because unitrode."
                    )
                    continue
                cls.insert1(sg_key, skip_duplicates=True)
                for elect in shank_elect:
                    sge_key["electrode_id"] = elect
                    cls.SortGroupElectrode().insert1(
                        sge_key, skip_duplicates=True
                    )
                sort_group += 1


@schema
class SpikeSortingPreprocessingParameters(SpyglassMixin, dj.Lookup):
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
            logger.warn("Similar row(s) already inserted.")
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
    """

    def make(self, key):
        AnalysisNwbfile()._creation_times["pre_create_time"] = time()
        # DO:
        # - get valid times for sort interval
        # - proprocess recording
        # - write recording to NWB file
        sort_interval_valid_times = self._get_sort_interval_valid_times(key)
        recording, timestamps = self._get_preprocessed_recording(key)
        recording_nwb_file_name, recording_object_id = _write_recording_to_nwb(
            recording,
            timestamps,
            (SpikeSortingRecordingSelection & key).fetch1("nwb_file_name"),
        )
        key["analysis_file_name"] = recording_nwb_file_name
        key["object_id"] = recording_object_id

        # INSERT:
        # - valid times into IntervalList
        # - analysis NWB file holding processed recording into AnalysisNwbfile
        # - entry into SpikeSortingRecording
        IntervalList.insert1(
            {
                "nwb_file_name": (SpikeSortingRecordingSelection & key).fetch1(
                    "nwb_file_name"
                ),
                "interval_list_name": key["recording_id"],
                "valid_times": sort_interval_valid_times,
                "pipeline": "spikesorting_recording_v1",
            }
        )
        AnalysisNwbfile().add(
            (SpikeSortingRecordingSelection & key).fetch1("nwb_file_name"),
            key["analysis_file_name"],
        )
        AnalysisNwbfile().log(
            recording_nwb_file_name, table=self.full_table_name
        )
        self.insert1(key)

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
        recording = se.read_nwb_recording(
            analysis_file_abs_path, load_time_vector=True
        )

        return recording

    @staticmethod
    def _get_recording_timestamps(recording):
        if recording.get_num_segments() > 1:
            frames_per_segment = [0]
            for i in range(recording.get_num_segments()):
                frames_per_segment.append(
                    recording.get_num_frames(segment_index=i)
                )

            cumsum_frames = np.cumsum(frames_per_segment)
            total_frames = np.sum(frames_per_segment)

            timestamps = np.zeros((total_frames,))
            for i in range(recording.get_num_segments()):
                timestamps[cumsum_frames[i] : cumsum_frames[i + 1]] = (
                    recording.get_times(segment_index=i)
                )
        else:
            timestamps = recording.get_times()
        return timestamps

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
        ).fetch1("valid_times")

        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch1("valid_times")

        # DO: - take intersection between sort interval and valid times
        return interval_list_intersect(
            sort_interval,
            valid_interval_times,
            min_length=params["min_segment_length"],
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

        probe_type_by_channel = []
        electrode_group_by_channel = []
        for channel_id in channel_ids:
            probe_type_by_channel.append(
                (
                    Electrode * Probe
                    & {
                        "nwb_file_name": nwb_file_name,
                        "electrode_id": channel_id,
                    }
                ).fetch1("probe_type")
            )
            electrode_group_by_channel.append(
                (
                    Electrode
                    & {
                        "nwb_file_name": nwb_file_name,
                        "electrode_id": channel_id,
                    }
                ).fetch1("electrode_group_name")
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

        # TODO: make sure the following works for recordings that don't have explicit timestamps
        valid_sort_times = self._get_sort_interval_valid_times(key)
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
        if ref_channel_id >= 0:
            recording = recording.channel_slice(channel_ids=all_channel_ids)
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
            recording = recording.channel_slice(
                channel_ids=recording_channel_ids
            )
            recording = si.preprocessing.common_reference(
                recording,
                reference="global",
                operator="median",
                dtype=np.float64,
            )
        elif ref_channel_id == -1:
            recording = recording.channel_slice(
                channel_ids=recording_channel_ids
            )
        else:
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

        return recording, np.asarray(timestamps)


def _consolidate_intervals(intervals, timestamps):
    """Convert a list of intervals (start_time, stop_time)
    to a list of intervals (start_index, stop_index) by comparing to a list of timestamps;
    then consolidates overlapping or adjacent intervals

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
        raise ValueError(
            "Input array must have shape (N, 2) where N is the number of intervals."
        )
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
        # If the stop time of the current interval is equal to or greater than the next start time minus 1
        if stop >= next_start - 1:
            stop = max(
                stop, next_stop
            )  # Extend the current interval to include the next one
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
):
    """Write a recording in NWB format

    Parameters
    ----------
    recording : si.Recording
    timestamps : iterable
    nwb_file_name : str
        name of NWB file the recording originates

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the preprocessed recording
    """

    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
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
            description=f"Referenced and filtered recording from {nwb_file_name} for spike sorting",
            conversion=np.unique(recording.get_channel_gains())[0] * 1e-6,
        )
        nwbfile.add_acquisition(processed_electrical_series)
        recording_object_id = nwbfile.acquisition[
            "ProcessedElectricalSeries"
        ].object_id
        io.write(nwbfile)
    return analysis_nwb_file, recording_object_id


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
        return self._timeseries.shape[0]

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
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
