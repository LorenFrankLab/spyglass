from typing import Iterable

import os
import tempfile
import time

import datajoint as dj
import numpy as np
import pynwb
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sip
import spikeinterface.sorters as sis
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spyglass.common.common_lab import LabMember, LabTeam
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from .recording import _consolidate_intervals
from .utils import generate_nwb_uuid

schema = dj.schema("spikesorting_v1_sorting")


@schema
class SpikeSorterParameters(dj.Lookup):
    definition = """
    sorter: varchar(200)
    sorter_param_name: varchar(200)
    ---
    sorter_params: blob
    """
    contents = [
        [
            "mountainsort4",
            "franklab_tetrode_hippocampus_30KHz",
            {
                "detect_sign": -1,
                "adjacency_radius": 100,
                "freq_min": 600,
                "freq_max": 6000,
                "filter": False,
                "whiten": True,
                "num_workers": 1,
                "clip_size": 40,
                "detect_threshold": 3,
                "detect_interval": 10,
            },
        ],
        [
            "mountainsort4",
            "franklab_probe_ctx_30KHz",
            {
                "detect_sign": -1,
                "adjacency_radius": 100,
                "freq_min": 300,
                "freq_max": 6000,
                "filter": False,
                "whiten": True,
                "num_workers": 1,
                "clip_size": 40,
                "detect_threshold": 3,
                "detect_interval": 10,
            },
        ],
        [
            "clusterless_thresholder",
            "default_clusterless",
            {
                "detect_threshold": 100.0,  # uV
                # Locally exclusive means one unit per spike detected
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 100,
                # noise levels needs to be 1.0 so the units are in uV and not MAD
                "noise_levels": np.asarray([1.0]),
                "random_chunk_kwargs": {},
                # output needs to be set to sorting for the rest of the pipeline
                "outputs": "sorting",
            },
        ],
    ]
    contents.extend(
        [
            [sorter, "default", sis.get_default_sorter_params(sorter)]
            for sorter in sis.available_sorters()
        ]
    )

    @classmethod
    def insert_default(cls):
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SpikeSortingSelection(dj.Manual):
    definition = """
    # Processed recording and parameters. Use `insert_selection` method to insert rows.
    sorting_id: varchar(50)
    ---
    -> SpikeSortingRecording
    -> SpikeSorterParameters
    -> IntervalList
    """

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into SpikeSortingSelection with an
        automatically generated unique sorting ID as the sole primary key.

        Parameters
        ----------
        key : dict
            primary key of SpikeSortingRecording, SpikeSorterParameters, IntervalList tables

        Returns
        -------
        sorting_id : str
            the unique sorting ID serving as primary key for SpikeSorting
        """
        if len((cls & key).fetch()) > 0:
            print(
                "This row has already been inserted into SpikeSortingSelection."
            )
            return (cls & key).fetch1()
        key["sorting_id"] = generate_nwb_uuid(
            key["nwb_file_name"],
            "S",
            6,
        )
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class SpikeSorting(dj.Computed):
    definition = """
    -> SpikeSortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)          # Object ID for the sorting in NWB file
    time_of_sort: int               # in Unix time, to the nearest second
    """

    def make(self, key: dict):
        """Runs spike sorting on the data and parameters specified by the
        SpikeSortingSelection table and inserts a new entry to SpikeSorting table.
        """
        # FETCH:
        # - information about the recording
        # - artifact free intervals
        # - spike sorter and sorter params
        recording_key = (
            SpikeSortingRecording * SpikeSortingSelection & key
        ).fetch1()
        artifact_removed_intervals = (
            IntervalList
            & {
                "nwb_file_name": (SpikeSortingSelection & key).fetch1(
                    "nwb_file_name"
                ),
                "interval_list_name": (SpikeSortingSelection & key).fetch1(
                    "interval_list_name"
                ),
            }
        ).fetch1("valid_times")
        sorter, sorter_params = (
            SpikeSorterParameters * SpikeSortingSelection & key
        ).fetch1("sorter", "sorter_params")

        # DO:
        # - load recording
        # - concatenate artifact removed intervals
        # - run spike sorting
        # - save output to NWB file
        recording_analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(
            recording_key["analysis_file_name"]
        )
        recording = se.read_nwb_recording(
            recording_analysis_nwb_file_abs_path, load_time_vector=True
        )

        timestamps = recording.get_times()

        artifact_removed_intervals_ind = _consolidate_intervals(
            artifact_removed_intervals, timestamps
        )

        recording_list = []
        for interval in artifact_removed_intervals_ind:
            recording_list.append(
                recording.frame_slice(interval[0], interval[1])
            )
        recording = si.concatenate_recordings(recording_list)

        if sorter == "clusterless_thresholder":
            # need to remove tempdir and whiten from sorter_params
            sorter_params.pop("tempdir", None)
            sorter_params.pop("whiten", None)
            sorter_params.pop("outputs", None)

            # Detect peaks for clusterless decoding
            detected_spikes = detect_peaks(recording, **sorter_params)
            sorting = si.NumpySorting.from_times_labels(
                times_list=detected_spikes["sample_ind"],
                labels_list=np.zeros(len(detected_spikes), dtype=np.int),
                sampling_frequency=recording.get_sampling_frequency(),
            )
        else:
            # Specify tempdir (expected by some sorters like mountainsort4)
            sorter_temp_dir = tempfile.TemporaryDirectory(
                dir=os.getenv("SPYGLASS_TEMP_DIR")
            )
            sorter_params["tempdir"] = sorter_temp_dir.name
            # if whitening is specified in sorter params, apply whitening separately
            # prior to sorting and turn off "sorter whitening"
            if sorter_params["whiten"]:
                recording = sip.whiten(recording, dtype=np.float64)
                sorter_params["whiten"] = False
            sorting = sis.run_sorter(
                sorter,
                recording,
                output_folder=sorter_temp_dir.name,
                **sorter_params,
            )
        key["time_of_sort"] = int(time.time())

        key["analysis_file_name"], key["object_id"] = _write_sorting_to_nwb(
            sorting,
            timestamps,
            artifact_removed_intervals,
            (SpikeSortingSelection & key).fetch1("nwb_file_name"),
        )

        # INSERT
        # - new entry to AnalysisNwbfile
        # - new entry to SpikeSorting
        AnalysisNwbfile().add(
            (SpikeSortingSelection & key).fetch1("nwb_file_name"),
            key["analysis_file_name"],
        )
        self.insert1(key)

    # def delete(self):
    #     """Extends the delete method of base class to implement permission checking.
    #     Note that this is NOT a security feature, as anyone that has access to source code
    #     can disable it; it just makes it less likely to accidentally delete entries.
    #     """
    #     current_user_name = dj.config["database.user"]
    #     entries = self.fetch()
    #     permission_bool = np.zeros((len(entries),))
    #     print(
    #         f"Attempting to delete {len(entries)} entries, checking permission..."
    #     )

    #     for entry_idx in range(len(entries)):
    #         # check the team name for the entry, then look up the members in that team,
    #         # then get their datajoint user names
    #         team_name = (
    #             SpikeSortingRecordingSelection
    #             & (SpikeSortingRecordingSelection & entries[entry_idx]).proj()
    #         ).fetch1()["team_name"]
    #         lab_member_name_list = (
    #             LabTeam.LabTeamMember & {"team_name": team_name}
    #         ).fetch("lab_member_name")
    #         datajoint_user_names = []
    #         for lab_member_name in lab_member_name_list:
    #             datajoint_user_names.append(
    #                 (
    #                     LabMember.LabMemberInfo
    #                     & {"lab_member_name": lab_member_name}
    #                 ).fetch1("datajoint_user_name")
    #             )
    #         permission_bool[entry_idx] = (
    #             current_user_name in datajoint_user_names
    #         )
    #     if np.sum(permission_bool) == len(entries):
    #         print("Permission to delete all specified entries granted.")
    #         super().delete()
    #     else:
    #         raise Exception(
    #             "You do not have permission to delete all specified"
    #             "entries. Not deleting anything."
    #         )

    @classmethod
    def get_sorting(cls, key: dict) -> si.BaseSorting:
        """Get sorting in the analysis NWB file as spikeinterface BaseSorting

        Parameters
        ----------
        key : dict
            primary key of SpikeSorting

        Returns
        -------
        sorting : si.BaseSorting

        """

        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        sorting = se.read_nwb_sorting(analysis_file_abs_path)

        return sorting


def _write_sorting_to_nwb(
    sorting: si.BaseSorting,
    timestamps: np.ndarray,
    sort_interval: Iterable,
    nwb_file_name: str,
):
    """Write a sorting in NWB format.

    Parameters
    ----------
    sorting : si.BaseSorting
        spike times are in samples
    timestamps: np.ndarray
        the absolute time of each sample, in seconds
    sort_interval : Iterable
    nwb_file_name : str
        Name of NWB file the recording originates from

    Returns
    -------
    analysis_nwb_file : str
        Name of analysis NWB file containing the sorting
    """

    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        nwbf.add_unit_column(
            name="curation_label",
            description="curation label applyed to a unit",
        )
        for unit_id in sorting.get_unit_ids():
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=timestamps[spike_times],
                id=unit_id,
                obs_intervals=sort_interval,
                curation_label="uncurated",
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
