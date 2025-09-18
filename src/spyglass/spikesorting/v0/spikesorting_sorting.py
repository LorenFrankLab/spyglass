import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as sip
import spikeinterface.sorters as sis
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from tqdm import tqdm

from spyglass.common.common_lab import LabMember, LabTeam
from spyglass.settings import sorting_dir, temp_dir
from spyglass.spikesorting.v0.spikesorting_artifact import (
    ArtifactRemovedIntervalList,
)
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_sorting")


@schema
class SpikeSorterParameters(SpyglassMixin, dj.Manual):
    """Parameters for spike sorting algorithms.

    Attributes
    ----------
    sorter: str
        Name of the spike sorting algorithm.
    sorter_params_name: str
        Name of the parameter set for the spike sorting algorithm.
    sorter_params: dict
        Dictionary of parameters for the spike sorting algorithm.
        The keys and values depend on the specific algorithm being used.
        For example, for the "mountainsort4" algorithm, the parameters are...
            detect_sign: int
                Sign of the detected spikes. 1 for positive, -1 for negative.
            adjacency_radius: int
                Radius for adjacency graph. Determines which channels are
                considered neighbors.
            freq_min: int
                Minimum frequency for bandpass filter.
            freq_max: int
                Maximum frequency for bandpass filter.
            filter: bool
                Whether to apply bandpass filter.
            whiten: bool
                Whether to whiten the data.
            num_workers: int
                Number of workers to use for parallel processing.
            clip_size: int
                Size of the clips to extract for spike detection.
            detect_threshold: float
                Threshold for spike detection.
            detect_interval: int
                Minimum interval between detected spikes.
        For the "clusterless_thresholder" algorithm, the parameters are...
            detect_threshold: float
                microvolt detection threshold for spike detection.
            method: str
                Method for spike detection. Options are "locally_exclusive" or
                "global".
            peak_sign: enum ("neg", "pos")
                Sign of the detected peaks.
            exclude_sweep_ms: float
                Exclusion time in milliseconds for detected spikes.
            local_radius_um: int
                Local radius in micrometers for spike detection.
            noise_levels: np.ndarray
                Noise levels for spike detection.
            random_chunk_kwargs: dict
                Additional arguments for random chunk processing.
            outputs: str
                Output type for spike detection. Options are "sorting" or
                "labels".
    """

    definition = """
    sorter: varchar(32)
    sorter_params_name: varchar(64)
    ---
    sorter_params: blob
    """

    # NOTE: See #630, #664. Excessive key length.

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
class SpikeSortingSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Table for holding selection of recording and parameters for each spike sorting run
    -> SpikeSortingRecording
    -> SpikeSorterParameters
    -> ArtifactRemovedIntervalList
    ---
    import_path = "": varchar(200)  # optional path to previous curated sorting output
    """


@schema
class SpikeSorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingSelection
    ---
    sorting_path: varchar(1000)
    time_of_sort: int   # in Unix time, to the nearest second
    """

    _parallel_make = True

    def make(self, key: dict):
        """Runs spike sorting on the data and parameters specified by the
        SpikeSortingSelection table and inserts a new entry to SpikeSorting table.

        Specifically,
        1. Loads saved recording and runs the sort on it with spikeinterface
        2. Saves the sorting with spikeinterface
        3. Creates an analysis NWB file and saves the sorting there
           (this is redundant with 2; will change in the future)

        """
        recording = SpikeSortingRecording().load_recording(key)

        # first, get the timestamps
        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)
        _ = recording.get_sampling_frequency()

        # then concatenate the recordings
        # Note: the timestamps are lost upon concatenation,
        # i.e. concat_recording.get_times() doesn't return true timestamps anymore.
        # but concat_recording.recoring_list[i].get_times() will return correct
        # timestamps for ith recording.
        if recording.get_num_segments() > 1:
            if isinstance(recording, si.AppendSegmentRecording):
                recording = si.concatenate_recordings(recording.recording_list)
            elif isinstance(recording, si.BinaryRecordingExtractor):
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
            recording = sip.remove_artifacts(
                recording=recording,
                list_triggers=list_triggers,
                ms_before=None,
                ms_after=None,
                mode="zeros",
            )

        logger.info(f"Running spike sorting on {key}...")
        sorter, sorter_params = (SpikeSorterParameters & key).fetch1(
            "sorter", "sorter_params"
        )

        sorter_temp_dir = tempfile.TemporaryDirectory(dir=temp_dir)
        # add tempdir option for mountainsort
        sorter_params["tempdir"] = sorter_temp_dir.name

        if sorter == "clusterless_thresholder":
            # need to remove tempdir and whiten from sorter_params
            sorter_params.pop("tempdir", None)
            sorter_params.pop("whiten", None)
            sorter_params.pop("outputs", None)
            if "local_radius_um" in sorter_params:
                sorter_params["radius_um"] = sorter_params.pop(
                    "local_radius_um"
                )  # correct existing parameter sets for spikeinterface>=0.99.1

            # Detect peaks for clusterless decoding
            detected_spikes = detect_peaks(recording, **sorter_params)
            sorting = si.NumpySorting.from_times_labels(
                times_list=detected_spikes["sample_index"],
                labels_list=np.zeros(len(detected_spikes), dtype=np.int32),
                sampling_frequency=recording.get_sampling_frequency(),
            )
        else:
            if "whiten" in sorter_params.keys():
                if sorter_params["whiten"]:
                    sorter_params["whiten"] = False  # set whiten to False
            # whiten recording separately; make sure dtype is float32
            # to avoid downstream error with svd
            recording = sip.whiten(recording, dtype="float32")
            sorting = sis.run_sorter(
                sorter,
                recording,
                output_folder=sorter_temp_dir.name,
                remove_existing_folder=True,
                delete_output_folder=True,
                **sorter_params,
            )
        key["time_of_sort"] = int(time.time())

        logger.info("Saving sorting results...")

        sorting_folder = Path(sorting_dir)

        sorting_name = self._get_sorting_name(key)
        key["sorting_path"] = str(sorting_folder / Path(sorting_name))
        if os.path.exists(key["sorting_path"]):
            shutil.rmtree(key["sorting_path"])
        sorting = sorting.save(folder=key["sorting_path"])
        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        """Placeholder to override mixin method"""
        raise NotImplementedError

    def cleanup(self, dry_run=False, verbose=True):
        """Clean up spike sorting directories that are not in the table."""
        sort_dir = Path(sorting_dir)
        tracked = set(self.fetch("sorting_path"))
        all_dirs = {str(f) for f in sort_dir.iterdir() if f.is_dir()}
        untracked = all_dirs - tracked

        if dry_run:
            return untracked

        for folder in tqdm(
            untracked, desc="Removing untracked folders", disable=not verbose
        ):
            try:
                shutil.rmtree(folder)
            except PermissionError:
                logger.warning(f"Permission denied: {folder}")

    @staticmethod
    def _get_sorting_name(key):
        recording_name = SpikeSortingRecording._get_recording_name(key)
        sorting_name = (
            recording_name + "_" + str(uuid.uuid4())[0:8] + "_spikesorting"
        )
        return sorting_name

    def _import_sorting(self, key):
        raise NotImplementedError("Not supported in V0. Use V1 instead.")
