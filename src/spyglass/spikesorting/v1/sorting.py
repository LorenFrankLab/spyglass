from typing import Iterable

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

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
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.spikesorting.v1.recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from .recording import _consolidate_intervals

schema = dj.schema("spikesorting_v1_sorting")


@schema
class SpikeSorterParameter(dj.Lookup):
    definition = """
    sorter: varchar(200)
    sorter_param_name: varchar(200)
    ---
    sorter_param: blob
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
    sorters = sis.available_sorters()
    contents.extend(
        [
            [sorter, "default", sis.get_default_sorter_params(sorter)]
            for sorter in sorters
        ]
    )

    def insert_default(self):
        """Default params from spike sorters available via spikeinterface"""
        sorters = sis.available_sorters()
        for sorter in sorters:
            sorter_params = sis.get_default_sorter_params(sorter)
            self.insert1(
                [sorter, "default", sorter_params], skip_duplicates=True
            )

        # Insert Frank lab defaults
        # Hippocampus tetrode default
        sorter = "mountainsort4"
        sorter_params_name = "franklab_tetrode_hippocampus_30KHz"
        sorter_params = {
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
        }
        self.insert1(
            [sorter, sorter_params_name, sorter_params], skip_duplicates=True
        )

        # Cortical probe default
        sorter = "mountainsort4"
        sorter_params_name = "franklab_probe_ctx_30KHz"
        sorter_params = {
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
        }
        self.insert1(
            [sorter, sorter_params_name, sorter_params], skip_duplicates=True
        )

        # clusterless defaults
        sorter = "clusterless_thresholder"
        sorter_params_name = "default_clusterless"
        sorter_params = dict(
            detect_threshold=100.0,  # uV
            # Locally exclusive means one unit per spike detected
            method="locally_exclusive",
            peak_sign="neg",
            exclude_sweep_ms=0.1,
            local_radius_um=100,
            # noise levels needs to be 1.0 so the units are in uV and not MAD
            noise_levels=np.asarray([1.0]),
            random_chunk_kwargs={},
            # output needs to be set to sorting for the rest of the pipeline
            outputs="sorting",
        )
        self.insert1(
            [sorter, sorter_params_name, sorter_params], skip_duplicates=True
        )


@schema
class SpikeSortingSelection(dj.Manual):
    definition = """
    # Processed recording and parameter
    -> SpikeSortingRecording
    -> SpikeSorterParameter
    -> IntervalList
    """


@schema
class SpikeSorting(dj.Computed):
    definition = """
    sorting_id: varchar(50)
    ---
    -> SpikeSortingSelection
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the sorting in NWB file
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
        recording_key = (SpikeSortingRecording & key).fetch1()
        artifact_removed_intervals = (
            IntervalList
            & {
                "nwb_file_name": recording_key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1("artifact_times")
        sorter, sorter_params = (SpikeSorterParameter & key).fetch1(
            "sorter", "sorter_params"
        )

        # DO:
        # - load recording
        # - remove artifacts
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

        sorter_temp_dir = tempfile.TemporaryDirectory(
            dir=os.getenv("SPYGLASS_TEMP_DIR")
        )
        # add tempdir option for mountainsort
        sorter_params["tempdir"] = sorter_temp_dir.name

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
            # turn off whitening by sorter (if that is an option) because
            # recording will be whitened before feeding it to sorter
            if "whiten" in sorter_params.keys():
                sorter_params["whiten"] = False
            recording = sip.whiten(recording, dtype="float64")
            sorting = sis.run_sorter(
                sorter,
                recording,
                output_folder=sorter_temp_dir.name,
                delete_output_folder=True,
                **sorter_params,
            )
        key["time_of_sort"] = int(time.time())

        print("Saving sorting results...")
        _write_sorting_to_nwb(sorting)
        # sorting_folder = Path(os.getenv("SPYGLASS_SORTING_DIR"))
        # sorting_name = self._get_sorting_name(key)
        # key["sorting_path"] = str(sorting_folder / Path(sorting_name))
        # if os.path.exists(key["sorting_path"]):
        #     shutil.rmtree(key["sorting_path"])
        # sorting = sorting.save(folder=key["sorting_path"])
        self.insert1(key)

    def delete(self):
        """Extends the delete method of base class to implement permission checking.
        Note that this is NOT a security feature, as anyone that has access to source code
        can disable it; it just makes it less likely to accidentally delete entries.
        """
        current_user_name = dj.config["database.user"]
        entries = self.fetch()
        permission_bool = np.zeros((len(entries),))
        print(
            f"Attempting to delete {len(entries)} entries, checking permission..."
        )

        for entry_idx in range(len(entries)):
            # check the team name for the entry, then look up the members in that team,
            # then get their datajoint user names
            team_name = (
                SpikeSortingRecordingSelection
                & (SpikeSortingRecordingSelection & entries[entry_idx]).proj()
            ).fetch1()["team_name"]
            lab_member_name_list = (
                LabTeam.LabTeamMember & {"team_name": team_name}
            ).fetch("lab_member_name")
            datajoint_user_names = []
            for lab_member_name in lab_member_name_list:
                datajoint_user_names.append(
                    (
                        LabMember.LabMemberInfo
                        & {"lab_member_name": lab_member_name}
                    ).fetch1("datajoint_user_name")
                )
            permission_bool[entry_idx] = (
                current_user_name in datajoint_user_names
            )
        if np.sum(permission_bool) == len(entries):
            print("Permission to delete all specified entries granted.")
            super().delete()
        else:
            raise Exception(
                "You do not have permission to delete all specified"
                "entries. Not deleting anything."
            )

    def fetch_nwb(self, *attrs, **kwargs):
        raise NotImplementedError
        return None
        # return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def nightly_cleanup(self):
        """Clean up spike sorting directories that are not in the SpikeSorting table.
        This should be run after AnalysisNwbFile().nightly_cleanup()
        """
        # get a list of the files in the spike sorting storage directory
        dir_names = next(os.walk(os.environ["SPYGLASS_SORTING_DIR"]))[1]
        # now retrieve a list of the currently used analysis nwb files
        analysis_file_names = self.fetch("analysis_file_name")
        for dir in dir_names:
            if dir not in analysis_file_names:
                full_path = str(Path(os.environ["SPYGLASS_SORTING_DIR"]) / dir)
                print(f"removing {full_path}")
                shutil.rmtree(
                    str(Path(os.environ["SPYGLASS_SORTING_DIR"]) / dir)
                )

    @staticmethod
    def _get_sorting_name(key):
        recording_name = SpikeSortingRecording._get_recording_name(key)
        sorting_name = (
            recording_name + "_" + str(uuid.uuid4())[0:8] + "_spikesorting"
        )
        return sorting_name


def _write_sorting_to_nwb(
    sorting: si.BaseSorting,
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
        nwbf = io.read()
        # processed_electrical_series = pynwb.ElectricalSeries(
        #     name="ProcessedElectricalSeries",
        #     data=data_iterator,
        #     electrodes=table_region,
        #     timestamps=timestamps_iterator,
        #     filtering="Bandpass filtered for spike band",
        #     description=f"Referenced and filtered recording from {nwb_file_name} for spike sorting",
        #     conversion=np.unique(recording.get_channel_gains())[0] * 1e-6,
        # )
        nwbf.add_acquisition(processed_electrical_series)
        recording_object_id = nwbf.acquisition[
            "ProcessedElectricalSeries"
        ].object_id
    return analysis_nwb_file, recording_object_id
