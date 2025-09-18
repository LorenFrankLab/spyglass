import os
import tempfile
import time
import uuid
from typing import Iterable

import datajoint as dj
import numpy as np
import pynwb
import spikeinterface as si
import spikeinterface.curation as sic
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sip
import spikeinterface.sorters as sis
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting.v1.recording import (  # noqa: F401
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
    _consolidate_intervals,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v1_sorting")


@schema
class SpikeSorterParameters(SpyglassMixin, dj.Lookup):
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
    # Spike sorting algorithm and associated parameters.
    sorter: varchar(200)
    sorter_param_name: varchar(200)
    ---
    sorter_params: blob
    """
    mountain_default = {
        "detect_sign": -1,
        "adjacency_radius": 100,
        "filter": False,
        "whiten": True,
        "num_workers": 1,
        "clip_size": 40,
        "detect_threshold": 3,
        "detect_interval": 10,
    }
    contents = [
        [
            "mountainsort4",
            "franklab_tetrode_hippocampus_30KHz",
            {**mountain_default, "freq_min": 600, "freq_max": 6000},
        ],
        [
            "mountainsort4",
            "franklab_probe_ctx_30KHz",
            {**mountain_default, "freq_min": 300, "freq_max": 6000},
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
        """Insert default sorter parameters into SpikeSorterParameters table."""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SpikeSortingSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Processed recording and spike sorting parameters. See `insert_selection`.
    sorting_id: uuid
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
        sorting_id : uuid
            the unique sorting ID serving as primary key for SpikeSorting
        """
        query = cls & key
        if query:
            logger.info("Similar row(s) already inserted.")
            return query.fetch(as_dict=True)
        key["sorting_id"] = uuid.uuid4()
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class SpikeSorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)          # Object ID for the sorting in NWB file
    time_of_sort: int               # in Unix time, to the nearest second
    """

    _use_transaction, _allow_insert = False, True
    _parallel_make = True  # True if n_workers > 1

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
        recording_analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(
            recording_key["analysis_file_name"]
        )

        # DO:
        # - load recording
        # - concatenate artifact removed intervals
        # - run spike sorting
        # - save output to NWB file
        recording = se.read_nwb_recording(
            recording_analysis_nwb_file_abs_path, load_time_vector=True
        )

        timestamps = recording.get_times()

        artifact_removed_intervals_ind = _consolidate_intervals(
            artifact_removed_intervals, timestamps
        )

        # if the artifact removed intervals do not span the entire time range
        if (
            (len(artifact_removed_intervals_ind) > 1)
            or (artifact_removed_intervals_ind[0][0] > 0)
            or (artifact_removed_intervals_ind[-1][1] < len(timestamps))
        ):
            # set the artifact intervals to zero
            list_triggers = []
            if artifact_removed_intervals_ind[0][0] > 0:
                list_triggers.append(
                    np.arange(0, artifact_removed_intervals_ind[0][0])
                )
            for interval_ind in range(len(artifact_removed_intervals_ind) - 1):
                list_triggers.append(
                    np.arange(
                        (artifact_removed_intervals_ind[interval_ind][1] + 1),
                        artifact_removed_intervals_ind[interval_ind + 1][0],
                    )
                )
            if artifact_removed_intervals_ind[-1][1] < len(timestamps):
                list_triggers.append(
                    np.arange(
                        artifact_removed_intervals_ind[-1][1],
                        len(timestamps) - 1,
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
            # Specify tempdir (expected by some sorters like mountainsort4)
            sorter_temp_dir = tempfile.TemporaryDirectory(dir=temp_dir)
            sorter_params["tempdir"] = sorter_temp_dir.name
            os.chmod(sorter_params["tempdir"], 0o777)

            if sorter == "mountainsort5":
                _ = sorter_params.pop("tempdir", None)

            # if whitening is specified in sorter params, apply whitening separately
            # prior to sorting and turn off "sorter whitening"
            if sorter_params.get("whiten", False):
                recording = sip.whiten(recording, dtype=np.float64)
                sorter_params["whiten"] = False

            common_sorter_items = {
                "sorter_name": sorter,
                "recording": recording,
                "output_folder": sorter_temp_dir.name,
                "remove_existing_folder": True,
            }

            if sorter.lower() in ["kilosort2_5", "kilosort3", "ironclust"]:
                sorter_params = {
                    k: v
                    for k, v in sorter_params.items()
                    if k
                    not in ["tempdir", "mp_context", "max_threads_per_process"]
                }
                sorting = sis.run_sorter(
                    **common_sorter_items,
                    singularity_image=True,
                    **sorter_params,
                )
            else:
                sorting = sis.run_sorter(
                    **common_sorter_items,
                    **sorter_params,
                )
        key["time_of_sort"] = int(time.time())
        sorting = sic.remove_excess_spikes(sorting, recording)
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
        self.insert1(key, skip_duplicates=True)

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

        recording_id = (
            SpikeSortingRecording * SpikeSortingSelection & key
        ).fetch1("recording_id")
        recording = SpikeSortingRecording.get_recording(
            {"recording_id": recording_id}
        )
        sampling_frequency = recording.get_sampling_frequency()
        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            units = nwbf.units.to_dataframe()
        units_dict_list = [
            {
                unit_id: np.searchsorted(recording.get_times(), spike_times)
                for unit_id, spike_times in zip(
                    units.index, units["spike_times"]
                )
            }
        ]

        sorting = si.NumpySorting.from_unit_dict(
            units_dict_list, sampling_frequency=sampling_frequency
        )

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
        if sorting.get_num_units() == 0:
            nwbf.units = pynwb.misc.Units(
                name="units", description="Empty units table."
            )
        else:
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label applied to a unit",
            )
            obs_interval = (
                sort_interval
                if sort_interval.ndim == 2
                else sort_interval.reshape(1, 2)
            )
            for unit_id in sorting.get_unit_ids():
                spike_times = sorting.get_unit_spike_train(unit_id)
                nwbf.add_unit(
                    spike_times=timestamps[spike_times],
                    id=unit_id,
                    obs_intervals=obs_interval,
                    curation_label="uncurated",
                )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
