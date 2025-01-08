import os
import shutil
from functools import reduce
from pathlib import Path

import datajoint as dj
import numpy as np
import probeinterface as pi
import spikeinterface as si
import spikeinterface.extractors as se

from spyglass.common.common_device import Probe, ProbeType  # noqa: F401
from spyglass.common.common_ephys import Electrode, ElectrodeGroup
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_intersect,
    intervals_by_length,
    union_adjacent_index,
)
from spyglass.common.common_lab import LabTeam  # noqa: F401
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.settings import recording_dir
from spyglass.spikesorting.utils import (
    _get_recording_timestamps,
    get_group_by_shank,
)
from spyglass.utils import SpyglassMixin
from spyglass.utils.dj_helper_fn import dj_replace

schema = dj.schema("spikesorting_recording")


@schema
class SortGroup(SpyglassMixin, dj.Manual):
    definition = """
    # Set of electrodes that will be sorted together
    -> Session
    sort_group_id: int  # identifier for a group of electrodes
    ---
    sort_reference_electrode_id = -1: int  # the electrode to use for reference. -1: no reference, -2: common median
    """

    class SortGroupElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> SortGroup
        -> Electrode
        """

    def set_group_by_shank(
        self,
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
        (SortGroup & {"nwb_file_name": nwb_file_name}).delete()

        sg_keys, sge_keys = get_group_by_shank(
            nwb_file_name=nwb_file_name,
            references=references,
            omit_ref_electrode_group=omit_ref_electrode_group,
            omit_unitrode=omit_unitrode,
        )
        self.insert(sg_keys, skip_duplicates=False)
        self.SortGroupElectrode().insert(sge_keys, skip_duplicates=False)

    def set_group_by_electrode_group(self, nwb_file_name: str):
        """Assign groups to all non-bad channel electrodes based on their electrode group
        and sets the reference for each group to the reference for the first channel of the group.

        Parameters
        ----------
        nwb_file_name: str
            the name of the nwb whose electrodes should be put into sorting groups
        """
        # delete any current groups
        (SortGroup & {"nwb_file_name": nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (
            Electrode()
            & {"nwb_file_name": nwb_file_name}
            & {"bad_channel": "False"}
        ).fetch()
        e_groups = np.unique(electrodes["electrode_group_name"])
        sg_key = dict()
        sge_key = dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name
        sort_group = 0
        for e_group in e_groups:
            sge_key["electrode_group_name"] = e_group
            # sg_key['sort_group_id'] = sge_key['sort_group_id'] = sort_group
            # TEST
            sg_key["sort_group_id"] = sge_key["sort_group_id"] = int(e_group)
            # get the list of references and make sure they are all the same
            shank_elect_ref = electrodes["original_reference_electrode"][
                electrodes["electrode_group_name"] == e_group
            ]
            if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                sg_key["sort_reference_electrode_id"] = shank_elect_ref[0]
            else:
                ValueError(
                    f"Error in electrode group {e_group}: reference electrodes are not all the same"
                )
            self.insert1(sg_key)

            shank_elect = electrodes["electrode_id"][
                electrodes["electrode_group_name"] == e_group
            ]
            for elect in shank_elect:
                sge_key["electrode_id"] = elect
                self.SortGroupElectrode().insert1(sge_key)
            sort_group += 1

    def set_reference_from_list(self, nwb_file_name, sort_group_ref_list):
        """
        Set the reference electrode from a list containing sort groups and reference electrodes
        :param: sort_group_ref_list - 2D array or list where each row is [sort_group_id reference_electrode]
        :param: nwb_file_name - The name of the NWB file whose electrodes' references should be updated
        :return: Null
        """
        key = dict()
        key["nwb_file_name"] = nwb_file_name
        sort_group_list = (SortGroup() & key).fetch1()
        for sort_group in sort_group_list:
            key["sort_group_id"] = sort_group
            self.insert(
                dj_replace(
                    sort_group_list,
                    sort_group_ref_list,
                    "sort_group_id",
                    "sort_reference_electrode_id",
                ),
                replace="True",
            )

    def get_geometry(self, sort_group_id, nwb_file_name):
        """
        Returns a list with the x,y coordinates of the electrodes in the sort group
        for use with the SpikeInterface package.

        Converts z locations to y where appropriate.

        Parameters
        ----------
        sort_group_id : int
        nwb_file_name : str

        Returns
        -------
        geometry : list
            List of coordinate pairs, one per electrode
        """

        # create the channel_groups dictiorary
        channel_group = dict()
        key = dict()
        key["nwb_file_name"] = nwb_file_name
        electrodes = (Electrode() & key).fetch()

        key["sort_group_id"] = sort_group_id
        sort_group_electrodes = (SortGroup.SortGroupElectrode() & key).fetch()
        electrode_group_name = sort_group_electrodes["electrode_group_name"][0]
        probe_id = (
            ElectrodeGroup
            & {
                "nwb_file_name": nwb_file_name,
                "electrode_group_name": electrode_group_name,
            }
        ).fetch1("probe_id")
        channel_group[sort_group_id] = dict()
        channel_group[sort_group_id]["channels"] = sort_group_electrodes[
            "electrode_id"
        ].tolist()

        n_chan = len(channel_group[sort_group_id]["channels"])

        geometry = np.zeros((n_chan, 2), dtype="float")
        tmp_geom = np.zeros((n_chan, 3), dtype="float")
        for i, electrode_id in enumerate(
            channel_group[sort_group_id]["channels"]
        ):
            # get the relative x and y locations of this channel from the probe table
            probe_electrode = int(
                electrodes["probe_electrode"][
                    electrodes["electrode_id"] == electrode_id
                ]
            )
            rel_x, rel_y, rel_z = (
                Probe().Electrode()
                & {"probe_id": probe_id, "probe_electrode": probe_electrode}
            ).fetch("rel_x", "rel_y", "rel_z")
            # TODO: Fix this HACK when we can use probeinterface:
            rel_x = float(rel_x)
            rel_y = float(rel_y)
            rel_z = float(rel_z)
            tmp_geom[i, :] = [rel_x, rel_y, rel_z]

        # figure out which columns have coordinates
        n_found = 0
        for i in range(3):
            if np.any(np.nonzero(tmp_geom[:, i])):
                if n_found < 2:
                    geometry[:, n_found] = tmp_geom[:, i]
                    n_found += 1
                else:
                    Warning(
                        "Relative electrode locations have three coordinates; "
                        + "only two are currently supported"
                    )
        return np.ndarray.tolist(geometry)


@schema
class SortInterval(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    sort_interval_name: varchar(64) # name for this interval
    ---
    sort_interval: longblob # 1D numpy array with start and end time for a single interval to be used for spike sorting
    """

    # NOTE: See #630, #664. Excessive key length.


@schema
class SpikeSortingPreprocessingParameters(SpyglassMixin, dj.Manual):
    definition = """
    preproc_params_name: varchar(32)
    ---
    preproc_params: blob
    """
    # NOTE: Reduced key less than 2 existing entries
    # All existing entries are below 48

    def insert_default(self):
        """Inserts the default preprocessing parameters for spike sorting."""
        # set up the default filter parameters
        freq_min = 300  # high pass filter value
        freq_max = 6000  # low pass filter value
        margin_ms = 5  # margin in ms on border to avoid border effect
        seed = 0  # random seed for whitening

        key = dict()
        key["preproc_params_name"] = "default"
        key["preproc_params"] = {
            "frequency_min": freq_min,
            "frequency_max": freq_max,
            "margin_ms": margin_ms,
            "seed": seed,
        }
        self.insert1(key, skip_duplicates=True)


@schema
class SpikeSortingRecordingSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Defines recordings to be sorted
    -> SortGroup
    -> SortInterval
    -> SpikeSortingPreprocessingParameters
    -> LabTeam
    ---
    -> IntervalList
    """


@schema
class SpikeSortingRecording(SpyglassMixin, dj.Computed):
    use_transaction, _allow_insert = False, True

    definition = """
    -> SpikeSortingRecordingSelection
    ---
    recording_path: varchar(1000)
    -> IntervalList.proj(sort_interval_list_name='interval_list_name')
    """

    _parallel_make = True

    def make(self, key):
        """Populates the SpikeSortingRecording table with the recording data.

        1. Fetches ...
            - Sort interval and parameters from SpikeSortingRecordingSelection
                and SpikeSortingPreprocessingParameters
            - Channel IDs and reference electrode from SortGroup, filtered by
                filtereing parameters
        2. Saves the recording data to the recording directory
        3. Inserts the path to the recording data into SpikeSortingRecording
        """
        sort_interval_valid_times = self._get_sort_interval_valid_times(key)
        recording = self._get_filtered_recording(key)
        recording_name = self._get_recording_name(key)

        # recording_dir = Path("/home/cbroz/wrk/temp_ssr0/")

        # Path to files that will hold the recording extractors
        recording_path = str(recording_dir / Path(recording_name))
        if os.path.exists(recording_path):
            shutil.rmtree(recording_path)

        recording.save(
            folder=recording_path, chunk_duration="10000ms", n_jobs=8
        )

        IntervalList.insert1(
            {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": recording_name,
                "valid_times": sort_interval_valid_times,
                "pipeline": "spikesorting_recording_v0",
            },
            replace=True,
        )

        self.insert1(
            {
                **key,
                # store the list of valid times for the sort
                "sort_interval_list_name": recording_name,
                "recording_path": recording_path,
            }
        )

    @staticmethod
    def _get_recording_name(key):
        return "_".join(
            [
                key["nwb_file_name"],
                key["sort_interval_name"],
                str(key["sort_group_id"]),
                key["preproc_params_name"],
                # key["team_name"], # TODO: add team name, reflect PK structure
            ]
        )

    @staticmethod
    def _get_recording_timestamps(recording):
        return _get_recording_timestamps(recording)

    def _get_sort_interval_valid_times(self, key):
        """Identifies the intersection between sort interval specified by the user
        and the valid times (times for which neural data exist)

        Parameters
        ----------
        key: dict
            specifies a (partially filled) entry of SpikeSorting table

        Returns
        -------
        sort_interval_valid_times: ndarray of tuples
            (start, end) times for valid stretches of the sorting interval

        """
        nwb_file_name, sort_interval_name, params, interval_list_name = (
            SpikeSortingPreprocessingParameters * SpikeSortingRecordingSelection
            & key
        ).fetch1(
            "nwb_file_name",
            "sort_interval_name",
            "preproc_params",
            "interval_list_name",
        )

        sort_interval = (
            SortInterval
            & {
                "nwb_file_name": nwb_file_name,
                "sort_interval_name": sort_interval_name,
            }
        ).fetch1("sort_interval")

        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")

        valid_sort_times = interval_list_intersect(
            sort_interval, valid_interval_times
        )
        # Exclude intervals shorter than specified length
        if "min_segment_length" in params:
            valid_sort_times = intervals_by_length(
                valid_sort_times, min_length=params["min_segment_length"]
            )
        return valid_sort_times

    def _get_filtered_recording(self, key: dict):
        """Filters and references a recording
        * Loads the NWB file created during insertion as a spikeinterface Recording
        * Slices recording in time (interval) and space (channels);
          recording chunks from disjoint intervals are concatenated
        * Applies referencing and bandpass filtering

        Parameters
        ----------
        key: dict,
            primary key of SpikeSortingRecording table

        Returns
        -------
        recording: si.Recording
        """

        nwb_file_abs_path = Nwbfile().get_abs_path(key["nwb_file_name"])
        recording = se.read_nwb_recording(
            nwb_file_abs_path, load_time_vector=True
        )

        valid_sort_times = self._get_sort_interval_valid_times(key)
        # shape is (N, 2)
        valid_sort_times_indices = np.array(
            [
                np.searchsorted(recording.get_times(), interval)
                for interval in valid_sort_times
            ]
        )
        # join intervals of indices that are adjacent
        valid_sort_times_indices = reduce(
            union_adjacent_index, valid_sort_times_indices
        )
        if valid_sort_times_indices.ndim == 1:
            valid_sort_times_indices = np.expand_dims(
                valid_sort_times_indices, 0
            )

        # create an AppendRecording if there is more than one disjoint sort interval
        if len(valid_sort_times_indices) > 1:
            recordings_list = []
            for interval_indices in valid_sort_times_indices:
                recording_single = recording.frame_slice(
                    start_frame=interval_indices[0],
                    end_frame=interval_indices[1],
                )
                recordings_list.append(recording_single)
            recording = si.append_recordings(recordings_list)
        else:
            recording = recording.frame_slice(
                start_frame=valid_sort_times_indices[0][0],
                end_frame=valid_sort_times_indices[0][1],
            )

        channel_ids = (
            SortGroup.SortGroupElectrode
            & {
                "nwb_file_name": key["nwb_file_name"],
                "sort_group_id": key["sort_group_id"],
            }
        ).fetch("electrode_id")
        ref_channel_id = (
            SortGroup
            & {
                "nwb_file_name": key["nwb_file_name"],
                "sort_group_id": key["sort_group_id"],
            }
        ).fetch1("sort_reference_electrode_id")
        channel_ids = np.setdiff1d(channel_ids, ref_channel_id)

        # include ref channel in first slice, then exclude it in second slice
        if ref_channel_id >= 0:
            channel_ids_ref = np.append(channel_ids, ref_channel_id)
            recording = recording.channel_slice(channel_ids=channel_ids_ref)

            recording = si.preprocessing.common_reference(
                recording, reference="single", ref_channel_ids=ref_channel_id
            )
            recording = recording.channel_slice(channel_ids=channel_ids)
        elif ref_channel_id == -2:
            recording = recording.channel_slice(channel_ids=channel_ids)
            recording = si.preprocessing.common_reference(
                recording, reference="global", operator="median"
            )
        else:
            raise ValueError("Invalid reference channel ID")
        filter_params = (SpikeSortingPreprocessingParameters & key).fetch1(
            "preproc_params"
        )
        recording = si.preprocessing.bandpass_filter(
            recording,
            freq_min=filter_params["frequency_min"],
            freq_max=filter_params["frequency_max"],
        )

        # if the sort group is a tetrode, change the channel location
        # note that this is a workaround that would be deprecated when spikeinterface uses 3D probe locations
        probe_type = []
        electrode_group = []
        for channel_id in channel_ids:
            probe_type.append(
                (
                    Electrode * Probe
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "electrode_id": channel_id,
                    }
                ).fetch1("probe_type")
            )
            electrode_group.append(
                (
                    Electrode
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "electrode_id": channel_id,
                    }
                ).fetch1("electrode_group_name")
            )
        if (
            all(p == "tetrode_12.5" for p in probe_type)
            and len(probe_type) == 4
            and all(eg == electrode_group[0] for eg in electrode_group)
        ):
            tetrode = pi.Probe(ndim=2)
            position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
            tetrode.set_contacts(
                position, shapes="circle", shape_params={"radius": 6.25}
            )
            tetrode.set_contact_ids(channel_ids)
            tetrode.set_device_channel_indices(np.arange(4))
            recording = recording.set_probe(tetrode, in_place=True)

        return recording
