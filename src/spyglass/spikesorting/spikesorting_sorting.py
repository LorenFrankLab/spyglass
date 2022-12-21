import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.sorters as sis
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from ..common.common_lab import LabMember, LabTeam
from ..common.common_nwbfile import AnalysisNwbfile
from ..utils.dj_helper_fn import fetch_nwb
from .spikesorting_artifact import ArtifactRemovedIntervalList
from .spikesorting_recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)

schema = dj.schema("spikesorting_sorting")


@schema
class SpikeSorterParameters(dj.Manual):
    definition = """
    sorter: varchar(200)
    sorter_params_name: varchar(200)
    ---
    sorter_params: blob
    """

    def insert_default(self):
        """Default params from spike sorters available via spikeinterface"""
        sorters = sis.available_sorters()
        for sorter in sorters:
            sorter_params = sis.get_default_params(sorter)
            self.insert1([sorter, "default", sorter_params], skip_duplicates=True)

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
        self.insert1([sorter, sorter_params_name, sorter_params], skip_duplicates=True)

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
        self.insert1([sorter, sorter_params_name, sorter_params], skip_duplicates=True)

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
        self.insert1([sorter, sorter_params_name, sorter_params], skip_duplicates=True)


@schema
class SpikeSortingSelection(dj.Manual):
    definition = """
    # Table for holding selection of recording and parameters for each spike sorting run
    -> SpikeSortingRecording
    -> SpikeSorterParameters
    -> ArtifactRemovedIntervalList
    ---
    import_path = "": varchar(200)  # optional path to previous curated sorting output
    """


@schema
class SpikeSorting(dj.Computed):
    definition = """
    -> SpikeSortingSelection
    ---
    sorting_path: varchar(1000)
    time_of_sort: int   # in Unix time, to the nearest second
    """

    def make(self, key: dict):
        """Runs spike sorting on the data and parameters specified by the
        SpikeSortingSelection table and inserts a new entry to SpikeSorting table.

        Specifically,
        1. Loads saved recording and runs the sort on it with spikeinterface
        2. Saves the sorting with spikeinterface
        3. Creates an analysis NWB file and saves the sorting there
           (this is redundant with 2; will change in the future)

        """

        recording_path = (SpikeSortingRecording & key).fetch1("recording_path")
        recording = si.load_extractor(recording_path)

        # first, get the timestamps
        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)
        fs = recording.get_sampling_frequency()
        # then concatenate the recordings
        # Note: the timestamps are lost upon concatenation,
        # i.e. concat_recording.get_times() doesn't return true timestamps anymore.
        # but concat_recording.recoring_list[i].get_times() will return correct
        # timestamps for ith recording.
        if recording.get_num_segments() > 1 and isinstance(
            recording, si.AppendSegmentRecording
        ):
            recording = si.concatenate_recordings(recording.recording_list)
        elif recording.get_num_segments() > 1 and isinstance(
            recording, si.BinaryRecordingExtractor
        ):
            recording = si.concatenate_recordings([recording])

        # load artifact intervals
        artifact_times = (
            ArtifactRemovedIntervalList
            & {
                "artifact_removed_interval_list_name": key[
                    "artifact_removed_interval_list_name"
                ]
            }
        ).fetch1("artifact_times")
        if len(artifact_times):
            if artifact_times.ndim == 1:
                artifact_times = np.expand_dims(artifact_times, 0)

            # convert artifact intervals to indices
            list_triggers = []
            for interval in artifact_times:
                list_triggers.append(
                    np.arange(
                        np.searchsorted(timestamps, interval[0]),
                        np.searchsorted(timestamps, interval[1]),
                    )
                )
            list_triggers = [list(np.concatenate(list_triggers))]
            recording = si.preprocessing.remove_artifacts(
                recording=recording,
                list_triggers=list_triggers,
                ms_before=None,
                ms_after=None,
                mode="zeros",
            )

        print(f"Running spike sorting on {key}...")
        sorter, sorter_params = (SpikeSorterParameters & key).fetch1(
            "sorter", "sorter_params"
        )

        sorter_temp_dir = tempfile.TemporaryDirectory(
            dir=os.getenv("SPYGLASS_TEMP_DIR")
        )
        # add tempdir option for mountainsort
        sorter_params["tempdir"] = sorter_temp_dir.name

        if sorter == "clusterless_thresholder":
            # Detect peaks for clusterless decoding
            # need to remove tempdir
            sorter_params.pop("tempdir", None)
            detected_spikes = detect_peaks(recording, **sorter_params)
            sorting = si.NumpySorting.from_times_labels(
                times_list=detected_spikes["sample_ind"],
                labels_list=np.zeros(len(detected_spikes), dtype=np.int),
                sampling_frequency=recording.get_sampling_frequency(),
            )
        else:
            sorting = sis.run_sorter(
                sorter,
                recording,
                output_folder=sorter_temp_dir.name,
                delete_output_folder=True,
                **sorter_params,
            )
        key["time_of_sort"] = int(time.time())

        print("Saving sorting results...")
        sorting_folder = Path(os.getenv("SPYGLASS_SORTING_DIR"))
        sorting_name = self._get_sorting_name(key)
        key["sorting_path"] = str(sorting_folder / Path(sorting_name))
        if os.path.exists(key["sorting_path"]):
            shutil.rmtree(key["sorting_path"])
        sorting = sorting.save(folder=key["sorting_path"])
        self.insert1(key)

    def delete(self):
        """Extends the delete method of base class to implement permission checking.
        Note that this is NOT a security feature, as anyone that has access to source code
        can disable it; it just makes it less likely to accidentally delete entries.
        """
        current_user_name = dj.config["database.user"]
        entries = self.fetch()
        permission_bool = np.zeros((len(entries),))
        print(f"Attempting to delete {len(entries)} entries, checking permission...")

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
                        LabMember.LabMemberInfo & {"lab_member_name": lab_member_name}
                    ).fetch1("datajoint_user_name")
                )
            permission_bool[entry_idx] = current_user_name in datajoint_user_names
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
                shutil.rmtree(str(Path(os.environ["SPYGLASS_SORTING_DIR"]) / dir))

    @staticmethod
    def _get_sorting_name(key):
        recording_name = SpikeSortingRecording._get_recording_name(key)
        sorting_name = recording_name + "_" + str(uuid.uuid4())[0:8] + "_spikesorting"
        return sorting_name

    # TODO: write a function to import sortings done outside of dj

    def _import_sorting(self, key):
        raise NotImplementedError
