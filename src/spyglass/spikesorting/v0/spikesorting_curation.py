from __future__ import annotations

import json
import pynwb

import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as sip

# SI >= 0.10x split qualitymetrics into the ``metrics`` package; the parent
# namespace re-exports both metrics v0/v1 use (``compute_isi_violations`` from
# metrics.quality, ``compute_num_spikes`` from metrics.spiketrain), so import
# it rather than the quality submodule (which lacks compute_num_spikes).
try:
    import spikeinterface.metrics as sq
except ModuleNotFoundError:  # SI 0.99 (legacy v0/v1 runtime env)
    import spikeinterface.qualitymetrics as sq

from spyglass.common import BrainRegion, Electrode
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import waveforms_dir
from spyglass.spikesorting._legacy_runtime import (
    _require_legacy_si_environment,
)
from spyglass.spikesorting.v0.merged_sorting_extractor import (
    MergedSortingExtractor,
)
from spyglass.spikesorting.v0.spikesorting_recording import (
    SortInterval,
    SpikeSortingRecording,
)
from spyglass.utils import SpyglassMixin, logger

from .spikesorting_recording import SortGroup
from .spikesorting_sorting import SpikeSorting

schema = dj.schema("spikesorting_curation")

valid_labels = ["reject", "noise", "artifact", "mua", "accept"]


def apply_merge_groups_to_sorting(
    sorting: si.BaseSorting, merge_groups: List[List[int]]
):
    """Apply merge groups to a sorting extractor."""
    # return a new sorting where the units are merged according to merge_groups
    # merge_groups is a list of lists of unit_ids.
    # for example: merge_groups = [[1, 2], [5, 8, 4]]]

    return MergedSortingExtractor(
        parent_sorting=sorting, merge_groups=merge_groups
    )


@schema
class Curation(SpyglassMixin, dj.Manual):
    definition = """
    # Stores each spike sorting; similar to IntervalList
    curation_id: int # a number corresponding to the index of this curation
    -> SpikeSorting
    ---
    parent_curation_id=-1: int
    curation_labels: blob # a dictionary of labels for the units
    merge_groups: blob # a list of merge groups for the units
    quality_metrics: blob # a list of quality metrics for the units (if available)
    description='': varchar(1000) #optional description for this curated sort
    time_of_creation: int   # in Unix time, to the nearest second
    """

    _nwb_table = AnalysisNwbfile

    @staticmethod
    def insert_curation(
        sorting_key: dict,
        parent_curation_id: int = -1,
        labels=None,
        merge_groups=None,
        metrics=None,
        description="",
    ):
        """Given a SpikeSorting key and the parent_sorting_id (and optional
        arguments) insert an entry into Curation.


        Parameters
        ----------
        sorting_key : dict
            The key for the original SpikeSorting
        parent_curation_id : int, optional
            The id of the parent sorting
        labels : dict or None, optional
        merge_groups : dict or None, optional
        metrics : dict or None, optional
            Computed metrics for sorting
        description : str, optional
            text description of this sort

        Returns
        -------
        curation_key : dict

        """
        if parent_curation_id == -1:
            # check to see if this sorting with a parent of -1 has already been
            # inserted and if so, warn the user
            inserted_curation = (Curation & sorting_key).fetch("KEY")
            if len(inserted_curation) > 0:
                Warning(
                    "Sorting has already been inserted, returning key to previously"
                    "inserted curation"
                )
                return inserted_curation[0]

        if labels is None:
            labels = {}
        if merge_groups is None:
            merge_groups = []
        if metrics is None:
            metrics = {}

        # generate a unique number for this curation
        id = (Curation & sorting_key).fetch("curation_id")
        if len(id) > 0:
            curation_id = max(id) + 1
        else:
            curation_id = 0

        # convert unit_ids in labels to integers for labels from sortingview.
        new_labels = {int(unit_id): labels[unit_id] for unit_id in labels}

        sorting_key.update(
            {
                "curation_id": curation_id,
                "parent_curation_id": parent_curation_id,
                "description": description,
                "curation_labels": new_labels,
                "merge_groups": merge_groups,
                "quality_metrics": metrics,
                "time_of_creation": int(time.time()),
            }
        )

        # mike: added skip duplicates
        Curation.insert1(sorting_key, skip_duplicates=True)

        # get the primary key for this curation
        curation_key = {
            item: sorting_key[item] for item in Curation.primary_key
        }

        return curation_key

    @staticmethod
    def get_recording(key: dict):
        """Returns the recording extractor for the recording related to this curation

        Parameters
        ----------
        key : dict
            SpikeSortingRecording key

        Returns
        -------
        recording_extractor : spike interface recording extractor

        """
        return SpikeSortingRecording().load_recording(key)

    def _load_sorting_info(self, key: dict) -> Tuple[str, List[List[int]]]:
        """Returns the sorting path and merge groups for this curation

        Parameters
        ----------
        key : dict
            Curation key

        Returns
        -------
        sorting_path : str
        merge_groups : List[List[int]]
        """
        sorting_path = (SpikeSorting & key).fetch1("sorting_path")
        merge_groups = (Curation & key).fetch1("merge_groups")
        return sorting_path, merge_groups

    def _load_sorting(self, sorting_path: str, merge_groups: List[List[int]]):
        """Returns the sorting extractor with merges applied

        Parameters
        ----------
        sorting_path : str
        merge_groups : List[List[int]]

        Returns
        -------
        sorting_extractor: spike interface sorting extractor

        """
        sorting = si.load_extractor(sorting_path)
        if len(merge_groups) != 0:
            return MergedSortingExtractor(
                parent_sorting=sorting, merge_groups=merge_groups
            )
        return sorting

    def get_curated_sorting(self, key: dict):
        """Returns the sorting extractor related to this curation,
        with merges applied.

        Parameters
        ----------
        key : dict
            Curation key

        Returns
        -------
        sorting_extractor: spike interface sorting extractor

        """
        sorting_path, merge_groups = self._load_sorting_info(key)
        return self._load_sorting(sorting_path, merge_groups)

    @staticmethod
    def save_sorting_nwb(
        key,
        sorting,
        timestamps,
        sort_interval,
        sort_interval_list_name: str = None,
        sort_interval_valid_times: np.ndarray = None,
        labels=None,
        metrics=None,
        unit_ids=None,
    ):
        """Store a sorting in a new AnalysisNwbfile

        Parameters
        ----------
        key : dict
            key to SpikeSorting table
        sorting : si.Sorting
            sorting
        timestamps : array_like
            Time stamps of the sorted recoridng;
            used to convert the spike timings from index to real time
        sort_interval_list_name : str
            name of sort interval
        sort_interval : list
            interval for start and end of sort
        labels : dict, optional
            curation labels, by default None
        metrics : dict, optional
            quality metrics, by default None
        unit_ids : list, optional
            IDs of units whose spiketrains to save, by default None

        Returns
        -------
        analysis_file_name : str
        units_object_id : str

        """
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        if (
            sort_interval_valid_times is None
            and sort_interval_list_name is not None
        ):
            sort_interval_valid_times = (
                IntervalList & {"interval_list_name": sort_interval_list_name}
            ).fetch1("valid_times")

        if sort_interval_valid_times is None:
            raise ValueError(
                "Either sort_interval_valid_times or "
                "sort_interval_list_name must be provided."
            )

        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()

        for unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(
                unit_id=unit_id
            )
            units[unit_id] = timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        object_ids = AnalysisNwbfile().add_units(
            analysis_file_name,
            units,
            units_valid_times,
            units_sort_interval,
            metrics=metrics,
            labels=labels,
        )
        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        if object_ids == "":
            logger.warning(
                "Sorting contains no units."
                "Created an empty analysis nwb file anyway."
            )
            units_object_id = ""
        else:
            units_object_id = object_ids[0]

        return analysis_file_name, units_object_id


@schema
class WaveformParameters(SpyglassMixin, dj.Manual):
    """Parameters for extracting waveforms from a sorting extractor

    Attributes
    ----------
    waveform_params_name : str
        Name of the waveform extraction parameters
    waveform_params : dict
        Dictionary of waveform extraction parameters, including...
        ms_before : float
            Number of milliseconds before the spike time to include
        ms_after : float
            Number of milliseconds after the spike time to include
        max_spikes_per_unit : int
            Maximum number of spikes to extract for each unit
        n_jobs : int
            Number of parallel jobs to use for extraction
        total_memory : str
            Total memory to use for extraction (e.g. "5G")
        whiten : bool
            Whether to whiten the waveforms or not
    """

    definition = """
    waveform_params_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """

    def insert_default(self):
        """Inserts default waveform parameters"""
        default = {
            "ms_before": 0.5,
            "ms_after": 0.5,
            "max_spikes_per_unit": 5000,
            "n_jobs": 5,
            "total_memory": "5G",
        }
        self.insert(
            (
                ["default_whitened", dict(**default, whiten=True)],
                ["default_not_whitened", dict(**default, whiten=False)],
            ),
            skip_duplicates=True,
        )


@schema
class WaveformSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Curation
    -> WaveformParameters
    """


@schema
class Waveforms(SpyglassMixin, dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    waveform_extractor_path: varchar(400)
    -> AnalysisNwbfile
    waveforms_object_id: varchar(40)   # Object ID for the waveforms in NWB file
    """

    def make_fetch(self, key):
        """Populate Waveforms table with waveform extraction results

        1. Fetches ...
            - Recording and sorting from Curation table
            - Parameters from WaveformParameters table
        """
        waveform_params = (WaveformParameters & key).fetch1("waveform_params")
        waveform_extractor_name = self._get_waveform_extractor_name(key)
        waveform_extractor_path = Path(waveforms_dir) / Path(
            waveform_extractor_name
        )

        recording_path = SpikeSortingRecording()._fetch_recording_path(key)
        sorting_path, merge_groups = Curation()._load_sorting_info(key)

        return [
            waveform_params,
            waveform_extractor_path,
            recording_path,
            sorting_path,
            merge_groups,
        ]

    def make_compute(
        self,
        key,
        waveform_params,
        waveform_extractor_path,
        recording_path,
        sorting_path,
        merge_groups,
    ):
        """Computes waveforms and returns information for insertion

        2. Uses spikeinterface to extract waveforms
        3. Generates an analysis NWB file with the waveforms
        """
        _require_legacy_si_environment("v0 Waveforms.make")
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        recording = si.load_extractor(recording_path)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])

        sorting = Curation()._load_sorting(sorting_path, merge_groups)

        logger.info("Extracting waveforms...")
        if "whiten" in waveform_params:
            if waveform_params.pop("whiten"):
                recording = sip.whiten(recording, dtype="float32")

        if os.path.exists(waveform_extractor_path):
            shutil.rmtree(waveform_extractor_path)

        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveform_extractor_path,
            **waveform_params,
        )

        object_id = AnalysisNwbfile().add_units_waveforms(
            analysis_file_name, waveform_extractor=waveforms
        )
        return [
            analysis_file_name,
            waveform_extractor_path,
            object_id,
        ]

    def make_insert(
        self, key, analysis_file_name, waveform_extractor_path, object_id
    ):
        """Inserts the computed waveforms into the Waveforms table

        4. Inserts the key into Waveforms table
        """
        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        self.insert1(
            dict(
                key,
                analysis_file_name=analysis_file_name,
                waveform_extractor_path=str(waveform_extractor_path),
                waveforms_object_id=object_id,
            )
        )

    def _get_waveform_path(self, key: dict) -> str:
        return (self & key).fetch1("waveform_extractor_path")

    def load_waveforms(self, key: dict):
        """Returns a spikeinterface waveform extractor specified by key

        Parameters
        ----------
        key : dict
            Could be an entry in Waveforms, or some other key that uniquely defines
            an entry in Waveforms

        Returns
        -------
        we : spikeinterface.WaveformExtractor
        """
        _require_legacy_si_environment("v0 Waveforms.load_waveforms")
        we_path = self._get_waveform_path(key)
        we = si.WaveformExtractor.load_from_folder(we_path)
        return we

    def fetch_nwb(self, key):
        """Fetches the NWB file path for the waveforms. NOT YET IMPLEMENTED."""
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError

    def _get_waveform_extractor_name(self, key):
        waveform_params_name = (WaveformParameters & key).fetch1(
            "waveform_params_name"
        )

        # prev used uuid, but dj.hash is deterministic
        rand_str = dj.hash.key_hash(key)[0:8]

        return (
            f'{key["nwb_file_name"]}_{rand_str}_'
            f'{key["curation_id"]}_{waveform_params_name}_waveforms'
        )


@schema
class MetricParameters(SpyglassMixin, dj.Manual):
    """Parameters for computing quality metrics of sorted units

    See MetricParameters.get_available_metrics() for a list of available metrics
    """

    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_params_name: varchar(64)
    ---
    metric_params: blob
    """

    # NOTE: See #630, #664. Excessive key length.

    metric_default_params = {
        "snr": {
            "peak_sign": "neg",
            "random_chunk_kwargs_dict": {
                "num_chunks_per_segment": 20,
                "chunk_size": 10000,
                "seed": 0,
            },
        },
        "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0.0},
        "nn_isolation": {
            "max_spikes": 1000,
            "min_spikes": 10,
            "n_neighbors": 5,
            "n_components": 7,
            "radius_um": 100,
            "seed": 0,
        },
        "nn_noise_overlap": {
            "max_spikes": 1000,
            "min_spikes": 10,
            "n_neighbors": 5,
            "n_components": 7,
            "radius_um": 100,
            "seed": 0,
        },
        "peak_channel": {"peak_sign": "neg"},
        "num_spikes": {},
    }
    # Example of peak_offset parameters 'peak_offset': {'peak_sign': 'neg'}
    available_metrics = [
        "snr",
        "isi_violation",
        "nn_isolation",
        "nn_noise_overlap",
        "peak_offset",
        "peak_channel",
        "num_spikes",
    ]

    def get_metric_default_params(self, metric: str):
        "Returns default params for the given metric"
        return self.metric_default_params(metric)

    def insert_default(self) -> None:
        """Inserts default metric parameters"""
        self.insert1(
            ["franklab_default3", self.metric_default_params],
            skip_duplicates=True,
        )

    def get_available_metrics(self):
        """Log available metrics and their descriptions"""
        for metric in _metric_name_to_func:
            if metric in self.available_metrics:
                func = _metric_name_to_func[metric]
                if func is None:
                    # The metric's SpikeInterface callable is absent on this SI
                    # version; report it rather than crash on ``None.__doc__``.
                    logger.info(
                        f"{metric} : (unavailable on this SpikeInterface)\n"
                    )
                    continue
                metric_doc = (func.__doc__ or "").split("\n")[0]
                metric_string = ("{metric_name} : {metric_doc}").format(
                    metric_name=metric, metric_doc=metric_doc
                )
                logger.info(metric_string + "\n")

    # TODO
    def _validate_metrics_list(self, key):
        """Checks whether a row to be inserted contains only available metrics"""
        # get available metrics list
        # get metric list from key
        # compare
        return NotImplementedError


@schema
class MetricSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Waveforms
    -> MetricParameters
    """

    def insert1(self, key, **kwargs):
        """Overriding insert1 to add warnings for peak_offset and peak_channel"""
        waveform_params = (WaveformParameters & key).fetch1("waveform_params")
        metric_params = (MetricParameters & key).fetch1("metric_params")
        if "peak_offset" in metric_params:
            if waveform_params["whiten"]:
                warnings.warn(
                    "Calculating 'peak_offset' metric on "
                    "whitened waveforms may result in slight "
                    "discrepancies"
                )
        if "peak_channel" in metric_params:
            if waveform_params["whiten"]:
                Warning(
                    "Calculating 'peak_channel' metric on "
                    "whitened waveforms may result in slight "
                    "discrepancies"
                )
        super().insert1(key, **kwargs)


@schema
class QualityMetrics(SpyglassMixin, dj.Computed):

    definition = """
    -> MetricSelection
    ---
    quality_metrics_path: varchar(500)
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    def make_fetch(self, key):
        """Populate QualityMetrics table with quality metric results.

        1. Fetches ...
            - Waveform extractor from Waveforms table
            - Parameters from MetricParameters table
        """
        wf_path = Waveforms()._get_waveform_path(key)
        params = (MetricParameters & key).fetch1("metric_params")
        qm_name = self._get_quality_metrics_name(key)
        quality_metrics_path = Path(waveforms_dir) / Path(qm_name + ".json")

        return [
            wf_path,
            params,
            qm_name,
            quality_metrics_path,
        ]

    def make_compute(
        self,
        key,
        wf_path,
        params,
        qm_name,
        quality_metrics_path,
    ):
        """Computes quality metrics and returns information for insertion

        2. Computes metrics, including SNR, ISI violation, NN isolation,
            NN noise overlap, peak offset, peak channel, and number of spikes.
        3. Generates an analysis NWB file with the metrics.
        """
        _require_legacy_si_environment("v0 QualityMetrics.make")
        # File name involves random string. Can't pass it through make_fetch.
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        waveform_extractor = si.WaveformExtractor.load_from_folder(wf_path)

        qm = {}
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(
                waveform_extractor, metric_name, **metric_params
            )
            qm[metric_name] = metric

        self._info_msg(f"Computed all metrics: {qm}")
        self._dump_to_json(qm, quality_metrics_path)  # save dict as json

        object_id = AnalysisNwbfile().add_units_metrics(
            analysis_file_name, metrics=qm
        )

        return [
            analysis_file_name,
            quality_metrics_path,
            object_id,
        ]

    def make_insert(
        self, key, analysis_file_name, quality_metrics_path, object_id
    ):
        """Inserts the computed quality metrics into the QualityMetrics table

        4. Inserts the key into QualityMetrics table
        """
        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        self.insert1(
            dict(
                key,
                analysis_file_name=analysis_file_name,
                quality_metrics_path=str(quality_metrics_path),
                object_id=object_id,
            )
        )

    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms()._get_waveform_extractor_name(key)
        qm_name = wf_name + "_qm"
        return qm_name

    def _compute_metric(self, waveform_extractor, metric_name, **metric_params):
        metric_func = _metric_name_to_func[metric_name]

        peak_sign_metrics = ["snr", "peak_offset", "peak_channel"]
        if metric_name == "isi_violation":
            return metric_func(waveform_extractor, **metric_params)
        elif metric_name in peak_sign_metrics:
            if "peak_sign" not in metric_params:
                raise Exception(
                    f"{peak_sign_metrics} metrics require peak_sign",
                    "to be defined in the metric parameters",
                )
            return metric_func(
                waveform_extractor,
                peak_sign=metric_params.pop("peak_sign"),
                **metric_params,
            )

        metric = {}
        num_spikes = sq.compute_num_spikes(waveform_extractor)

        is_nn_iso = metric_name == "nn_isolation"
        is_nn_overlap = metric_name == "nn_noise_overlap"
        min_spikes = metric_params.get("min_spikes", 10)

        for unit_id in waveform_extractor.sorting.get_unit_ids():
            # checks to avoid bug in spikeinterface 0.98.2
            if num_spikes[unit_id] < min_spikes and (
                is_nn_iso or is_nn_overlap
            ):
                if is_nn_iso:
                    metric[str(unit_id)] = (np.nan, np.nan)
                elif is_nn_overlap:
                    metric[str(unit_id)] = np.nan

            else:
                metric[str(unit_id)] = metric_func(
                    waveform_extractor,
                    this_unit_id=int(unit_id),
                    **metric_params,
                )
            # nn_isolation returns tuple with isolation and unit number.
            # We only want isolation.
            if is_nn_iso:
                metric[str(unit_id)] = metric[str(unit_id)][0]
        return metric

    def _dump_to_json(self, qm_dict, save_path):
        new_qm = {}
        for key, value in qm_dict.items():
            m = {}
            for unit_id, metric_val in value.items():
                m[str(unit_id)] = np.float64(metric_val)
            new_qm[str(key)] = m
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(new_qm, f, ensure_ascii=False, indent=4)


def _compute_isi_violation_fractions(waveform_extractor, **metric_params):
    """Computes the per unit fraction of interspike interval violations to total spikes."""
    isi_threshold_ms = metric_params["isi_threshold_ms"]
    min_isi_ms = metric_params["min_isi_ms"]

    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = sq.compute_isi_violations(
        waveform_extractor,
        isi_threshold_ms=isi_threshold_ms,
        min_isi_ms=min_isi_ms,
    ).isi_violations_count

    # Extract the total number of spikes from each unit. The number of ISIs is one less than this
    num_spikes = sq.compute_num_spikes(waveform_extractor)

    # Calculate the fraction of ISIs that are violations
    isi_viol_frac_metric = {
        str(unit_id): isi_violation_counts[unit_id] / (num_spikes[unit_id] - 1)
        for unit_id in waveform_extractor.sorting.get_unit_ids()
    }
    return isi_viol_frac_metric


def _get_peak_offset(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the shift of the waveform peak from center of window."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_offset_inds = si.core.get_template_extremum_channel_peak_shift(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_offset = {key: int(abs(val)) for key, val in peak_offset_inds.items()}
    return peak_offset


def _get_peak_channel(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the electrode_id of the channel with the extremum peak for each unit."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_channel_dict = si.core.get_template_extremum_channel(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_channel = {key: int(val) for key, val in peak_channel_dict.items()}
    return peak_channel


def _get_num_spikes(
    waveform_extractor: si.WaveformExtractor, this_unit_id: int
):
    """Computes the number of spikes for each unit."""
    all_spikes = sq.compute_num_spikes(waveform_extractor)
    cluster_spikes = all_spikes[this_unit_id]
    return cluster_spikes


# Quality-metric callables are resolved lazily so this module imports under
# SpikeInterface 0.104, which renamed/removed some legacy entry points. v0
# metric computation only runs under SI 0.99; any missing entries surface as a
# clear error only when the metric is actually invoked.
_metric_name_to_func = {
    "snr": getattr(sq, "compute_snrs", None),
    "isi_violation": _compute_isi_violation_fractions,
    "nn_isolation": getattr(sq, "nearest_neighbors_isolation", None),
    "nn_noise_overlap": getattr(sq, "nearest_neighbors_noise_overlap", None),
    "peak_offset": _get_peak_offset,
    "peak_channel": _get_peak_channel,
    "num_spikes": _get_num_spikes,
}


@schema
class AutomaticCurationParameters(SpyglassMixin, dj.Manual):
    """Parameters for automatic curation of spike sorting

    Attributes
    ----------
    auto_curation_params_name : str
        Name of the automatic curation parameters
    merge_params : dict, optional
        Dictionary of parameters for merging units. May include nn_noise_overlap
        List[comparison operator: str, threshold: float, labels: List[str]]
    label_params : dict, optional
        Dictionary of parameters for labeling units
    """

    definition = """
    auto_curation_params_name: varchar(36)   # name of this parameter set
    ---
    merge_params: blob   # dictionary of params to merge units
    label_params: blob   # dictionary params to label units
    """

    # NOTE: No existing entries impacted by this change

    def insert1(self, key, **kwargs):
        """Overriding insert1 to validats label_params and merge_params"""
        # validate the labels and then insert
        # TODO: add validation for merge_params
        for metric in key["label_params"]:
            if metric not in _metric_name_to_func:
                raise Exception(f"{metric} not in list of available metrics")
            comparison_list = key["label_params"][metric]
            if comparison_list[0] not in _comparison_to_function:
                raise Exception(
                    f'{metric}: "{comparison_list[0]}" '
                    f"not in list of available comparisons"
                )
            if not isinstance(comparison_list[1], (int, float)):
                raise Exception(
                    f"{metric}: {comparison_list[1]} is of type "
                    f"{type(comparison_list[1])} and not a number"
                )
            for label in comparison_list[2]:
                if label not in valid_labels:
                    raise Exception(
                        f'{metric}: "{label}" '
                        f"not in list of valid labels: {valid_labels}"
                    )
        super().insert1(key, **kwargs)

    def insert_default(self):
        """Inserts default automatic curation parameters"""
        # label_params parsing: Each key is the name of a metric,
        # the contents are a three value list with the comparison, a value,
        # and a list of labels to apply if the comparison is true
        default_params = {
            "auto_curation_params_name": "default",
            "merge_params": {},
            "label_params": {
                "nn_noise_overlap": [">", 0.1, ["noise", "reject"]]
            },
        }
        self.insert1(default_params, skip_duplicates=True)

        # Second default parameter set for not applying any labels,
        # or merges, but adding metrics
        no_label_params = {
            "auto_curation_params_name": "none",
            "merge_params": {},
            "label_params": {},
        }
        self.insert1(no_label_params, skip_duplicates=True)


@schema
class AutomaticCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> QualityMetrics
    -> AutomaticCurationParameters
    """


_comparison_to_function = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
}


@schema
class AutomaticCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> AutomaticCurationSelection
    ---
    auto_curation_key: blob # the key to the curation inserted by make
    """

    def make(self, key):
        """Populate AutomaticCuration table with automatic curation results.

        1. Fetches ...
            - Quality metrics from QualityMetrics table
            - Parameters from AutomaticCurationParameters table
            - Parent curation/sorting from Curation table
        2. Curates the sorting based on provided merge and label parameters
        3. Inserts IDs into  AutomaticCuration and Curation tables
        """
        metrics_path = (QualityMetrics & key).fetch1("quality_metrics_path")
        with open(metrics_path) as f:
            quality_metrics = json.load(f)

        # get the curation information and the curated sorting
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_merge_groups = parent_curation["merge_groups"]
        parent_labels = parent_curation["curation_labels"]
        parent_curation_id = parent_curation["curation_id"]
        parent_sorting = Curation().get_curated_sorting(key)

        merge_params = (AutomaticCurationParameters & key).fetch1(
            "merge_params"
        )
        merge_groups, units_merged = self.get_merge_groups(
            parent_sorting, parent_merge_groups, quality_metrics, merge_params
        )

        label_params = (AutomaticCurationParameters & key).fetch1(
            "label_params"
        )
        labels = self.get_labels(
            parent_sorting, parent_labels, quality_metrics, label_params
        )

        # keep the quality metrics only if no merging occurred.
        metrics = quality_metrics if not units_merged else None

        # insert this sorting into the CuratedSpikeSorting Table
        # first remove keys that aren't part of the Sorting (the primary key of curation)
        c_key = (SpikeSorting & key).fetch("KEY")[0]
        curation_key = {item: key[item] for item in key if item in c_key}
        key["auto_curation_key"] = Curation.insert_curation(
            curation_key,
            parent_curation_id=parent_curation_id,
            labels=labels,
            merge_groups=merge_groups,
            metrics=metrics,
            description="auto curated",
        )

        self.insert1(key)

    @staticmethod
    def get_merge_groups(
        sorting, parent_merge_groups, quality_metrics, merge_params
    ):
        """Identifies units to be merged based on the quality_metrics and
        merge parameters and returns an updated list of merges for the curation.

        Parameters
        ---------
        sorting : spikeinterface.sorting
        parent_merge_groups : list
            Information about previous merges
        quality_metrics : list
        merge_params : dict

        Returns
        -------
        merge_groups : list of lists
        merge_occurred : bool

        """

        # overview:
        # 1. Use quality metrics to determine merge groups for units
        # 2. Combine merge groups with current merge groups to produce union of merges

        if not merge_params:
            return parent_merge_groups, False
        else:
            # TODO: use the metrics to identify clusters that should be merged
            # new_merges should then reflect those merges and the line below should be deleted.
            new_merges = []
            # append these merges to the parent merge_groups
            for new_merge in new_merges:
                # check to see if the first cluster listed is in a current merge group
                for previous_merge in parent_merge_groups:
                    if new_merge[0] == previous_merge[0]:
                        # add the additional units in new_merge to the identified merge group.
                        previous_merge.extend(new_merge[1:])
                        previous_merge.sort()
                        break
                else:
                    # append this merge group to the list if no previous merge
                    parent_merge_groups.append(new_merge)
            return parent_merge_groups.sort(), True

    @staticmethod
    def get_labels(sorting, parent_labels, quality_metrics, label_params):
        """Returns a dictionary of labels using quality_metrics and label
        parameters.

        Parameters
        ---------
        sorting : spikeinterface.sorting
        parent_labels : dict
            Dictionary of labels keyed by unit_id, from previous merges
        quality_metrics : dict
            Dictionary keyed by metric name, each value a dictionary of
            metric values keyed by unit_id:
            ``{metric_name: {unit_id: value}}``
        label_params : dict

        Returns
        -------
        parent_labels : dict
            Dictionary of labels keyed by unit_id

        """
        # overview:
        # 1. Use quality metrics to determine labels for units
        # 2. Append labels to current labels, checking for inconsistencies
        if not label_params:
            return parent_labels

        for metric in label_params:
            if metric not in quality_metrics:
                logger.warning(
                    f"{metric} not found in quality metrics; skipping"
                )
                continue

            compare = _comparison_to_function[label_params[metric][0]]

            for unit_id in quality_metrics[metric]:

                # compare the quality metric to the threshold with the
                # specified operator note that label_params[metric] is a three
                # element list with a comparison operator as a string, the
                # threshold value, and a list of labels to be applied if the
                # comparison is true

                label = label_params[metric]

                if compare(quality_metrics[metric][unit_id], label[1]):
                    if int(unit_id) not in parent_labels:
                        parent_labels[int(unit_id)] = label[2].copy()
                    # check if the label is already there, and if not, add it
                    else:
                        # remove 'accept' label if it exists
                        if "accept" in parent_labels[int(unit_id)]:
                            parent_labels[int(unit_id)].remove("accept")
                        for element in label[2].copy():
                            if element not in parent_labels[int(unit_id)]:
                                parent_labels[int(unit_id)].append(element)

        return parent_labels

    @staticmethod
    def _normalize_labels(labels):
        """Return labels dict with all keys cast to int.

        Quality metrics loaded from JSON have string keys, while
        the fixed ``get_labels`` uses ``int(unit_id)``.  Normalize
        to int so comparisons are consistent regardless of source.
        """
        return {int(k): v for k, v in labels.items()}


@schema
class CuratedSpikeSortingSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Curation
    """


@schema
class CuratedSpikeSorting(SpyglassMixin, dj.Computed):
    definition = """
    -> CuratedSpikeSortingSelection
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40)
    """

    class Unit(SpyglassMixin, dj.Part):
        definition = """
        # Table for holding sorted units
        -> CuratedSpikeSorting
        unit_id: int   # ID for each unit
        ---
        label='': varchar(200)   # optional set of labels for each unit
        nn_noise_overlap=-1: float   # noise overlap metric for each unit
        nn_isolation=-1: float   # isolation score metric for each unit
        isi_violation=-1: float   # ISI violation score for each unit
        snr=0: float            # SNR for each unit
        firing_rate=-1: float   # firing rate
        num_spikes=-1: int   # total number of spikes
        peak_channel=null: int # channel of maximum amplitude for each unit
        """

    def make_fetch(self, key):
        """Populate CuratedSpikeSorting table with curated sorting results.

        1. Fetches metrics and sorting from the Curation table
        """
        # check that the Curation has metrics
        metrics, unit_labels = (Curation & key).fetch1(
            "quality_metrics", "curation_labels"
        )
        if metrics == {}:
            self._warn_msg(
                f"Metrics for Curation {key} should normally be calculated "
                + "before insertion here"
            )

        sorting_path, merge_groups = Curation()._load_sorting_info(key)
        recording_path = SpikeSortingRecording()._fetch_recording_path(key)

        # get the sort_interval and sorting interval list
        sort_interval = (SortInterval & key).fetch1("sort_interval")
        sort_interval_list_name = (SpikeSorting & key).fetch1(
            "artifact_removed_interval_list_name"
        )
        sort_interval_valid_times = (
            IntervalList & {"interval_list_name": sort_interval_list_name}
        ).fetch1("valid_times")

        return [
            metrics,
            unit_labels,
            sorting_path,
            merge_groups,
            recording_path,
            sort_interval,
            sort_interval_valid_times,
        ]

    def make_compute(
        self,
        key,
        metrics,
        unit_labels,
        sorting_path,
        merge_groups,
        recording_path,
        sort_interval,
        sort_interval_valid_times,
    ):
        """Computes curated sorting and returns information for insertion

        2. Saves the sorting in an analysis NWB file
        3. Inserts key into CuratedSpikeSorting table and units into part table.
        """
        sorting = Curation()._load_sorting(sorting_path, merge_groups)
        unit_ids = sorting.get_unit_ids()

        # Get the labels for the units, add only those units that do not have
        # 'reject' or 'noise' labels
        unit_labels_to_remove = ["reject"]
        accepted_units = []
        for unit_id in unit_ids:
            if unit_id in unit_labels:
                if (
                    len(set(unit_labels_to_remove) & set(unit_labels[unit_id]))
                    == 0
                ):
                    accepted_units.append(unit_id)
            else:
                accepted_units.append(unit_id)

        # get the labels for the accepted units
        labels = {}
        for unit_id in accepted_units:
            if unit_id in unit_labels:
                labels[unit_id] = ",".join(unit_labels[unit_id])

        # convert unit_ids in metrics to integers, including only accepted units.
        #  TODO: convert to int this somewhere else
        final_metrics = {}
        for metric in metrics:
            final_metrics[metric] = {
                int(unit_id): metrics[metric][unit_id]
                for unit_id in metrics[metric]
                if int(unit_id) in accepted_units
            }

        logger.info(f"Found {len(accepted_units)} accepted units")

        recording = si.load_extractor(recording_path)
        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)

        analysis_file_name, units_object_id = Curation().save_sorting_nwb(
            key=key,
            sorting=sorting,
            timestamps=timestamps,
            sort_interval=sort_interval,
            sort_interval_valid_times=sort_interval_valid_times,
            metrics=final_metrics,
            unit_ids=accepted_units,
            labels=labels,
        )

        unit_inserts = []
        metric_fields = self.metrics_fields()
        for unit_id in accepted_units:
            this_key = dict(key, unit_id=unit_id)
            if unit_id in labels:
                this_key["label"] = labels[unit_id]
            for field in metric_fields:
                if field in final_metrics:
                    this_key[field] = final_metrics[field][unit_id]
                else:
                    Warning(
                        f"No metric named {field} in computed unit quality "
                        + "metrics; skipping"
                    )
            unit_inserts.append(this_key)

        return [
            analysis_file_name,
            units_object_id,
            unit_inserts,
        ]

    def make_insert(
        self, key, analysis_file_name, units_object_id, unit_inserts
    ):
        """Inserts the computed curated sorting into CuratedSpikeSorting

        4. Inserts the key into CuratedSpikeSorting table
        """
        self.insert1(
            dict(
                key,
                analysis_file_name=analysis_file_name,
                units_object_id=units_object_id,
            )
        )
        CuratedSpikeSorting.Unit.insert(unit_inserts)

    def metrics_fields(self):
        """Returns a list of the metrics that are currently in the Units table."""
        unit_info = self.Unit().fetch(limit=1, format="frame")
        unit_fields = [column for column in unit_info.columns]
        unit_fields.remove("label")
        return unit_fields

    @classmethod
    def get_recording(cls, key):
        """Returns the recording related to this curation. Useful for operations downstream of merge table"""
        # expand the key
        recording_key = (cls & key).fetch1("KEY")
        return SpikeSortingRecording().load_recording(recording_key)

    @classmethod
    def get_sorting(cls, key):
        """Returns the sorting related to this curation. Useful for operations downstream of merge table"""
        # expand the key
        sorting_key = (cls & key).fetch1("KEY")
        return Curation().get_curated_sorting(sorting_key)

    @classmethod
    def get_sort_group_info(cls, key):
        """Returns the sort group information for the curation
        (e.g. brain region, electrode placement, etc.)

        Parameters
        ----------
        key : dict
            restriction on CuratedSpikeSorting table

        Returns
        -------
        sort_group_info : Table
            Table with information about the sort groups
        """
        table = cls & key

        electrode_restrict_list = []
        for entry in table:
            # Just take one electrode entry per sort group
            electrode_restrict_list.extend(
                ((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(
                    limit=1
                )
            )
        # Run joins with the tables with info and return
        sort_group_info = (
            (Electrode & electrode_restrict_list)
            * table
            * SortGroup.SortGroupElectrode()
        ) * BrainRegion()
        return sort_group_info


@schema
class Fix1513Status(SpyglassMixin, dj.Computed):
    """Interactive review and repair table for AutomaticCuration Bug A.

    PR #1281 (2025-04-22) introduced an early-return bug (Bug A) in
    ``AutomaticCuration.get_labels``: only the first metric in
    ``label_params`` was processed. This table surfaces a per-unit
    before/after comparison and lets the data owner decide the
    appropriate action per entry.

    Usage
    -----
    **Recommended workflow**:

    Step 1 — Admin fast pass: populate all out-of-scope entries as
    none_needed. In-scope entries are silently skipped and remain for
    Step 2::

        Fix1513Status.populate(make_kwargs={"action": "none_needed_only"})

    Step 2 — Each data owner reviews their own entries. Using
    ``pending_for_member`` restricts populate to sessions you own,
    avoiding ``PermissionError`` on other members' curations::

        unreviewed, _ = Fix1513Status.pending_for_member("Alice")
        Fix1513Status.populate(unreviewed)

    Batch update (safe only for Case A; raises ValueError for Cases B
    and C, which require ``action="repopulate"``)::

        Fix1513Status.populate(unreviewed, make_kwargs={"action": "update"})

    If running unrestricted and permission errors should not halt the run,
    use DataJoint's ``suppress_errors`` flag (errors are still logged)::

        Fix1513Status.populate(suppress_errors=True)

    Step 3 — After any populate() call, finish deferred work that could
    not safely run inside that populate's transaction:

    * ``action="update"`` stages NWB label edits to temp files but does
      not activate them (``os.replace`` + checksum) until the DB
      transaction has committed::

          Fix1513Status.activate_pending_nwb_repairs()

    * ``action="repopulate"`` cannot call
      ``CuratedSpikeSorting.populate()`` itself (DataJoint refuses
      ``populate()`` from within another populate's transaction)::

          Fix1513Status.run_pending_repopulates()

    Outstanding work::

        # Not yet reviewed (includes entries skipped by none_needed_only):
        AutomaticCuration - Fix1513Status
        # Deferred entries (explicitly skipped by user):
        Fix1513Status & "action='skip'"
        # Repopulate requested but not confirmed:
        Fix1513Status & "action='repopulate' AND repopulated=0"
        # Per-member breakdown:
        Fix1513Status.pending_summary()
    """

    definition = """
    -> AutomaticCuration
    ---
    action        : enum('keep','update','repopulate','skip','none_needed')
    reviewed_by   : varchar(80)   # LabMember.lab_member_name of acting user
    owner_member  : varchar(80)   # Session experimenter (curation owner)
    reviewed_at   : datetime
    repopulated=0 : tinyint       # 1 after CuratedSpikeSorting confirmed
    notes=''      : varchar(500)
    label_diff=NULL : longblob    # dict of old_labels/new_labels for audit
    nwb_staged=NULL : longblob    # [(temp_path, real_path), ...] pending activation
    """

    # Class-level permission cache — persists across populate() calls.
    # Keyed by nwb_file_name; all AutomaticCuration entries sharing the
    # same nwb_file_name belong to the same Session (and experimenter),
    # so one lookup is sufficient per Python session (one user per session).
    _perm_pass = {}  # {nwb_file_name: owner_member} — permitted
    _perm_fail = set()  # {nwb_file_name} — denied

    # ------------------------------------------------------------------
    # Core diff computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_label_diff(key):
        """Compute the label diff for a single AutomaticCuration entry.

        Returns a dict describing the change, or None if the entry is
        out of scope (no change needed / no Bug A impact).

        Parameters
        ----------
        key : dict
            Primary key for AutomaticCuration.

        Returns
        -------
        dict or None
            Keys: auto_curation_key, curation_key, old_labels,
            new_labels, reject_status_changed.  None if entry is
            out of scope or unchanged.
        """
        from copy import deepcopy
        from datetime import datetime, timezone

        bug_date = datetime(2025, 4, 22, tzinfo=timezone.utc).timestamp()

        # --- Early return 1: empty label_params ---
        label_params = (
            (AutomaticCuration & key) * AutomaticCurationParameters
        ).fetch1("label_params")
        if not label_params:
            return None

        # --- Early return 2: created before bug date ---
        auto_curation_key = (AutomaticCuration & key).fetch1(
            "auto_curation_key"
        )
        time_created = (Curation & auto_curation_key).fetch1("time_of_creation")
        if time_created < bug_date:
            return None

        # --- Early return 3: ≤1 overlapping metric (Bug A scope only) ---
        metrics_path = (QualityMetrics & key).fetch1("quality_metrics_path")
        try:
            with open(metrics_path) as f:
                quality_metrics = json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Fix1513Status: metrics file not found: "
                f"{metrics_path}; entry will remain unpopulated for retry"
            )
            raise

        overlap = sum(1 for m in label_params if m in quality_metrics)
        if overlap <= 1:
            return None

        # --- Recompute labels ---
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_labels = AutomaticCuration._normalize_labels(
            parent_curation["curation_labels"]
        )
        stored_labels = AutomaticCuration._normalize_labels(
            (Curation & auto_curation_key).fetch1("curation_labels")
        )

        new_labels = AutomaticCuration._normalize_labels(
            AutomaticCuration.get_labels(
                sorting=None,
                parent_labels=deepcopy(parent_labels),
                quality_metrics=quality_metrics,
                label_params=label_params,
            )
        )

        if stored_labels == new_labels:
            return None

        # --- Compute reject-status changes ---
        all_uids = set(stored_labels.keys()) | set(new_labels.keys())
        reject_status_changed = []
        for uid in all_uids:
            old_reject = uid in stored_labels and "reject" in stored_labels[uid]
            new_reject = uid in new_labels and "reject" in new_labels[uid]
            if old_reject != new_reject:
                reject_status_changed.append(
                    {
                        "unit": uid,
                        "was_rejected": old_reject,
                        "should_reject": new_reject,
                    }
                )

        return {
            "auto_curation_key": auto_curation_key,
            "curation_key": {
                k: auto_curation_key[k]
                for k in Curation.primary_key
                if k in auto_curation_key
            },
            "old_labels": stored_labels,
            "new_labels": new_labels,
            "reject_status_changed": reject_status_changed,
        }

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------

    def _check_permission(self, key, user_name, is_admin):
        """Return owner_member str. Raise PermissionError if denied.

        Parameters
        ----------
        key : dict
            Primary key for AutomaticCuration (must include
            ``nwb_file_name``).
        user_name : str
            LabMember name of the current user.
        is_admin : bool
            True if the DataJoint user is an admin.

        Returns
        -------
        str
            owner_member name (``"unknown"`` only for admins, when no
            Session link / experimenter is found).

        Raises
        ------
        PermissionError
            If the user is not on a team with the curation owner, or
            (for non-admins) if ownership cannot be determined because
            no ``Session`` link exists or ``Session.Experimenter`` is
            missing/NULL.
        """
        from spyglass.common import LabTeam

        nwb_file = key["nwb_file_name"]
        if nwb_file in self._perm_pass:
            return self._perm_pass[nwb_file]
        # Skip the cached-fail short-circuit for admins so the admin
        # bypass below can override a prior non-admin denial.
        if not is_admin and nwb_file in self._perm_fail:
            raise PermissionError(
                f"Permission denied for nwb_file_name='{nwb_file}' "
                f"(cached). User '{user_name}' is not on a team with "
                f"the curation owner."
            )

        # Cache miss — run full chain
        exp_summary = (self & key)._get_exp_summary()
        owners = exp_summary.fetch(self._member_pk) if exp_summary else []

        # No Session link at all, or a Session with no
        # Session.Experimenter (NULL owner): ownership cannot be
        # determined. Admins may proceed (e.g. the none_needed_only fast
        # pass); non-admins are denied to fail closed for a destructive
        # repair.
        if not owners or None in owners:
            if is_admin:
                self._perm_pass[nwb_file] = "unknown"
                return "unknown"
            self._perm_fail.add(nwb_file)
            if not owners:
                raise PermissionError(
                    f"Fix1513Status: no Session link found for {key}; "
                    f"cannot verify the curation owner. An admin can "
                    f"review this entry."
                )
            raise PermissionError(
                f"Fix1513Status: Session.Experimenter is missing for "
                f"{key}; cannot verify the curation owner. Please "
                f"ensure all Sessions have an experimenter, or have an "
                f"admin review this entry."
            )

        owner_member = owners[0]

        if is_admin:
            self._perm_pass[nwb_file] = owner_member
            return owner_member

        permitted = any(
            user_name in LabTeam().get_team_members(o, reload=True)
            for o in set(owners)
        )
        if permitted:
            self._perm_pass[nwb_file] = owner_member
            return owner_member

        self._perm_fail.add(nwb_file)
        raise PermissionError(
            f"User '{user_name}' is not on a team with curation "
            f"owner(s) {set(owners)} for nwb_file_name='{nwb_file}'."
        )

    # ------------------------------------------------------------------
    # Repair helpers (migrated from AutomaticCuration)
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_unit_labels(curation_key, new_labels, verbose=True):
        """Update CuratedSpikeSorting.Unit labels for a curation.

        Updates the label column on existing Unit rows. Does NOT
        add or remove rows — if accept/reject status changed, a
        full repopulation of CuratedSpikeSorting is needed.

        Parameters
        ----------
        curation_key : dict
            Primary key to the Curation entry.
        new_labels : dict
            Corrected labels dict with int keys
            ``{unit_id: [label, ...]}``.
        verbose : bool
            If True, log changes.
        """
        unit_rows = (CuratedSpikeSorting.Unit & curation_key).fetch(
            as_dict=True
        )
        for row in unit_rows:
            uid = int(row["unit_id"])
            old_label = row["label"]
            new_label = ",".join(new_labels.get(uid, []))
            if old_label != new_label:
                if verbose:
                    logger.info(f"  unit {uid}: '{old_label}' -> '{new_label}'")
                CuratedSpikeSorting.Unit.update1(
                    {
                        **{
                            k: row[k]
                            for k in CuratedSpikeSorting.Unit.primary_key
                        },
                        "label": new_label,
                    }
                )

    @staticmethod
    def _repair_nwb_labels(curation_key, new_labels, verbose=True):
        """Update labels in all CuratedSpikeSorting NWB analysis files.

        Does NOT add or remove unit rows. If accept/reject status
        changed such that units need to be added, a full
        repopulation of CuratedSpikeSorting is required.

        Parameters
        ----------
        curation_key : dict
            Primary key to the Curation entry.
        new_labels : dict
            Corrected labels dict ``{unit_id: [label, ...]}``.
        verbose : bool
            If True, log file path and changes.
        """
        css_rows = (CuratedSpikeSorting & curation_key).fetch(as_dict=True)
        for css_row in css_rows:
            abs_path = AnalysisNwbfile().get_abs_path(
                css_row["analysis_file_name"]
            )
            Fix1513Status._patch_nwb_file(abs_path, new_labels, verbose)

    @staticmethod
    def _apply_nwb_label_edits(nwb_path, new_labels, verbose):
        """Open nwb_path in-place ('a' mode), edit unit labels, write if changed.

        Returns True if any label was modified.
        """
        with pynwb.NWBHDF5IO(
            path=nwb_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            if nwbf.units is None or "label" not in nwbf.units:
                if verbose:
                    logger.info("  NWB: no units/label column; skip")
                return False

            unit_ids = list(nwbf.units.id.data[:])
            label_col = nwbf.units["label"]
            changed = False

            for idx, uid in enumerate(unit_ids):
                new_val = ",".join(new_labels.get(int(uid), []))
                old_val = label_col[idx]
                if old_val != new_val:
                    label_col.data[idx] = new_val
                    changed = True
                    if verbose:
                        logger.info(
                            f"  NWB unit {uid}: '{old_val}' -> '{new_val}'"
                        )

            if changed:
                io.write(nwbf)
        return changed

    @staticmethod
    def _update_nwb_checksum(abs_path, verbose):
        """Update the external-table checksum for the given NWB file.

        Delegates to ``dj_helper_fn._update_analysis_file_checksum``;
        Fix1513 is a user-driven retroactive repair, so the admin gate
        on the sibling ``_resolve_external_table`` is intentionally
        skipped.
        """
        from spyglass.utils.dj_helper_fn import (
            _update_analysis_file_checksum,
        )

        _update_analysis_file_checksum(abs_path)
        if verbose:
            logger.info("  NWB: checksum updated")

    @staticmethod
    def _patch_nwb_file(abs_path, new_labels, verbose):
        """Edit the label column in one NWB file and update its checksum.

        For direct / legacy use.  ``make()`` uses ``_stage_nwb_patch`` +
        ``_activate_nwb_repairs`` instead so that HDF5 writes stay outside
        the DataJoint transaction.

        Parameters
        ----------
        abs_path : str or Path
            Absolute path to the NWB analysis file.
        new_labels : dict
            Corrected labels dict ``{unit_id: [label, ...]}``.
        verbose : bool
            If True, log changes.
        """
        if verbose:
            logger.info(f"  NWB: editing {abs_path}")
        changed = Fix1513Status._apply_nwb_label_edits(
            abs_path, new_labels, verbose
        )
        if changed:
            Fix1513Status._update_nwb_checksum(abs_path, verbose)

    @staticmethod
    def _stage_nwb_patch(abs_path, new_labels, verbose):
        """Copy abs_path to a temp file in the same directory and patch it.

        Returns ``(temp_path, abs_path)`` if any label changed, else
        ``None`` (and the temp file is immediately deleted).
        The temp file lives on the same filesystem as the original so that
        ``os.replace`` is atomic when the caller activates the patch.
        """
        abs_path = Path(abs_path)
        tmp = tempfile.NamedTemporaryFile(
            dir=abs_path.parent, suffix=".tmp", delete=False
        )
        tmp.close()
        temp_path = Path(tmp.name)
        try:
            shutil.copy2(abs_path, temp_path)
            changed = Fix1513Status._apply_nwb_label_edits(
                temp_path, new_labels, verbose
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        if not changed:
            temp_path.unlink(missing_ok=True)
            return None
        return (str(temp_path), str(abs_path))

    @staticmethod
    def _stage_nwb_repairs(curation_key, new_labels, verbose=True):
        """Stage NWB label patches for all CuratedSpikeSorting analysis files.

        Returns a list of ``(temp_path, real_path)`` pairs — one per NWB
        file that actually has label changes.  Pass this list to
        ``_activate_nwb_repairs`` after the DB transaction commits.
        """
        css_rows = (CuratedSpikeSorting & curation_key).fetch(as_dict=True)
        staged = []
        for css_row in css_rows:
            abs_path = AnalysisNwbfile().get_abs_path(
                css_row["analysis_file_name"]
            )
            result = Fix1513Status._stage_nwb_patch(
                abs_path, new_labels, verbose
            )
            if result is not None:
                staged.append(result)
        return staged

    @staticmethod
    def _activate_nwb_repairs(staged, verbose=True):
        """Atomically rename staged temp NWB files into place and update checksums.

        Call this only after the DB transaction has committed successfully.
        Each ``os.replace`` is atomic on the same filesystem, so on-disk
        state changes only after the DB is already consistent.

        Safe to re-run on a partially-completed ``staged`` list: a pair
        whose ``temp_path`` no longer exists is assumed to have already
        been activated (by ``os.replace`` in a prior, interrupted call)
        and is skipped.
        """
        for temp_path, real_path in staged:
            if not Path(temp_path).exists():
                if verbose:
                    logger.info(f"  NWB: {real_path} already activated, skip")
                continue
            os.replace(temp_path, real_path)
            if verbose:
                logger.info(f"  NWB: activated patch for {real_path}")
            Fix1513Status._update_nwb_checksum(real_path, verbose)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_diff(diff, owner_member):
        """Print a per-unit before/after label comparison table.

        Parameters
        ----------
        diff : dict
            Return value of ``_compute_label_diff``.
        owner_member : str
            Lab member name of the curation owner.
        """
        ack = diff["auto_curation_key"]
        sorting_id = ack.get("sorting_id", ack.get("sort_group_id", str(ack)))
        curation_id = ack.get("curation_id", "?")

        old_labels = diff["old_labels"]
        new_labels = diff["new_labels"]
        reject_changed_units = {
            r["unit"] for r in diff["reject_status_changed"]
        }

        changed_units = sorted(
            u
            for u in set(old_labels) | set(new_labels)
            if old_labels.get(u) != new_labels.get(u)
        )

        print(
            f"\nEntry: {sorting_id} / curation_id={curation_id}"
            f"  owner: {owner_member}"
        )
        print(f"Units with label changes: {len(changed_units)}\n")

        col_w = 23
        hdr = (
            f"  {'unit':>4}  {'stored_labels':<{col_w}}"
            f"  {'expected_labels':<{col_w}}  reject_changed"
        )
        sep = f"  {'----':>4}  {'-' * col_w}  {'-' * col_w}  {'-' * 14}"
        print(hdr)
        print(sep)

        for uid in changed_units:
            old_str = str(old_labels.get(uid, []))
            new_str = str(new_labels.get(uid, []))
            if uid in reject_changed_units:
                rc = diff["reject_status_changed"]
                entry = next(r for r in rc if r["unit"] == uid)
                if entry["should_reject"]:
                    rc_str = "YES -> rejected"
                else:
                    rc_str = "YES -> accepted"
            else:
                rc_str = ""
            print(
                f"  {uid:>4}  {old_str:<{col_w}}"
                f"  {new_str:<{col_w}}  {rc_str}"
            )

    @staticmethod
    def _get_case(reject_status_changed):
        """Classify the reject-status change to determine safe actions.

        Parameters
        ----------
        reject_status_changed : list of dict
            ``{unit, was_rejected, should_reject}`` entries from diff.

        Returns
        -------
        str
            ``'A'`` — no reject-status change.
            ``'B'`` — only newly-rejected units (must repopulate so
              CuratedSpikeSorting drops them).
            ``'C'`` — any newly-accepted units (must repopulate).
        """
        if not reject_status_changed:
            return "A"
        if any(not r["should_reject"] for r in reject_status_changed):
            return "C"
        return "B"

    @staticmethod
    def _prompt_action(case):
        """Interactively prompt the user for the repair action.

        Interactive only: blocks on ``input()`` while ``make()`` runs
        inside the ``Fix1513Status.populate()`` transaction.
        Only reachable via ``action="report"`` (the default), so it must
        not be used in headless/CI/notebook contexts without stdin —
        ``input()`` raises ``EOFError`` there. Pass an explicit
        ``make_kwargs={"action": ...}`` to ``populate()`` to bypass this
        prompt entirely.

        Parameters
        ----------
        case : str
            One of ``'A'``, ``'B'``, ``'C'`` from ``_get_case``.

        Returns
        -------
        str
            One of ``'keep'``, ``'update'``, ``'repopulate'``,
            ``'skip'``.
        """
        if case == "C":
            warning = (
                "WARNING: Some units that should now be accepted are "
                "absent from the NWB file. 'update' is not available; "
                "use 'repopulate' to rebuild from scratch."
            )
            print(warning)
            prompt = "(k)eep  (r)epopulate  (s)kip  [k/r/s]: "
            valid = {"k": "keep", "r": "repopulate", "s": "skip"}
        elif case == "B":
            warning = (
                "WARNING: Some units are newly rejected.  'update' is "
                "not available because CuratedSpikeSorting excludes "
                "rejected units, so patching labels would leave stale "
                "unit rows behind.  Use 'repopulate' to rebuild."
            )
            print(warning)
            prompt = "(k)eep  (r)epopulate  (s)kip  [k/r/s]: "
            valid = {"k": "keep", "r": "repopulate", "s": "skip"}
        else:  # case A
            prompt = "(k)eep  (u)pdate  (s)kip  [k/u/s]: "
            valid = {"k": "keep", "u": "update", "s": "skip"}

        while True:
            choice = input(prompt).strip().lower()
            if choice in valid:
                return valid[choice]
            print(
                f"Invalid choice '{choice}'. "
                f"Please enter one of: {list(valid.keys())}"
            )

    # ------------------------------------------------------------------
    # Pending entries queries
    # ------------------------------------------------------------------

    @classmethod
    def pending_summary(cls):
        """Print and return pending entry counts grouped by lab member.

        Aggregates unreviewed and skipped entries across all lab members
        so a PI or admin can see the full repair workload at a glance.

        Returns
        -------
        dict[str, dict]
            ``{lab_member_name: {"unreviewed": int, "skipped": int}}``

        Examples
        --------
        ::

            Fix1513Status.pending_summary()
            # lab_member_name    unreviewed    skipped
            # -----------------  ------------  -------
            # Alice              5             2
            # Bob                3             0
        """
        from spyglass.common import Session

        unreviewed_all = (AutomaticCuration * Session.Experimenter) - cls()
        unreviewed_counts = (
            dj.U("lab_member_name")
            .aggr(unreviewed_all, unreviewed="count(*)")
            .fetch(as_dict=True)
        )

        skipped_all = cls() & "action='skip'"
        skipped_counts = (
            dj.U("owner_member")
            .aggr(skipped_all, skipped="count(*)")
            .fetch(as_dict=True)
        )

        # Merge into a single dict keyed by member name
        summary = {}
        for row in unreviewed_counts:
            summary[row["lab_member_name"]] = {
                "unreviewed": row["unreviewed"],
                "skipped": 0,
            }
        for row in skipped_counts:
            member = row["owner_member"]
            if member not in summary:
                summary[member] = {"unreviewed": 0, "skipped": 0}
            summary[member]["skipped"] = row["skipped"]

        # Print aligned table
        if not summary:
            print("No pending entries.")
            return summary

        col_w = max(len(m) for m in summary) + 2
        header = f"  {'lab_member_name':<{col_w}}  unreviewed  skipped"
        print(header)
        print(f"  {'-' * col_w}  ----------  -------")
        for member, counts in sorted(summary.items()):
            print(
                f"  {member:<{col_w}}  "
                f"{counts['unreviewed']:<10}  "
                f"{counts['skipped']}"
            )

        return summary

    @classmethod
    def pending_for_member(cls, lab_member_name):
        """Return unreviewed and skipped Fix1513Status entries for a member.

        Useful for lab members to see what curation entries they own that
        still need a repair decision.

        Parameters
        ----------
        lab_member_name : str
            ``LabMember.lab_member_name`` to filter by (e.g. ``"Alice"``)

        Returns
        -------
        tuple[dj.expression.QueryExpression, dj.expression.QueryExpression]
            ``(unreviewed, skipped)``

            ``unreviewed`` — ``AutomaticCuration`` entries not yet in
            ``Fix1513Status`` that belong to sessions owned by the member.
            These have never been processed (or were left unpopulated by a
            ``none_needed_only`` pass because they are in-scope).

            ``skipped`` — ``Fix1513Status`` entries with ``action='skip'``
            where the member is the curation owner.  These need revisiting.

        Examples
        --------
        ::

            unreviewed, skipped = Fix1513Status.pending_for_member("Alice")
            print(f"Unreviewed: {len(unreviewed)}, Skipped: {len(skipped)}")
            # Populate only Alice's unreviewed entries interactively:
            Fix1513Status.populate(unreviewed)
        """
        from spyglass.common import Session

        member_filter = {"lab_member_name": lab_member_name}

        # AutomaticCuration entries owned by member and not yet reviewed.
        # The natural join on nwb_file_name connects AutomaticCuration to
        # Session.Experimenter through the primary-key chain.
        unreviewed = (
            AutomaticCuration * Session.Experimenter & member_filter
        ) - cls()

        # Fix1513Status entries the member deferred with action='skip'.
        skipped = cls() & {"action": "skip", "owner_member": lab_member_name}

        n_unreviewed = len(unreviewed)
        n_skipped = len(skipped)
        print(
            f"Pending for '{lab_member_name}': "
            f"{n_unreviewed} unreviewed, {n_skipped} skipped"
        )
        if n_unreviewed:
            print(
                "  Run Fix1513Status.populate(unreviewed) "
                "to process interactively."
            )
        if n_skipped:
            print(
                "  Re-open skipped entries by deleting their rows "
                "then re-populating:\n"
                "    skipped.delete()\n"
                "    Fix1513Status.populate(unreviewed)"
            )

        return unreviewed, skipped

    # ------------------------------------------------------------------
    # make
    # ------------------------------------------------------------------

    def make(self, key, action="report"):
        """Review and optionally repair one AutomaticCuration entry.

        Called by ``Fix1513Status.populate()``.  Pass ``action`` via
        ``make_kwargs`` to batch-process without interactive prompts.

        Parameters
        ----------
        key : dict
            AutomaticCuration primary key supplied by DataJoint.
        action : str
            ``'report'`` (default) — compute diff and prompt
              interactively via ``input()``. Interactive only: this
              blocks on stdin while holding the ``populate()``
              transaction open and raises ``EOFError`` in
              headless/CI/notebook contexts. Pass an explicit action
              below for batch use.
            ``'none_needed_only'`` — insert ``none_needed`` for
              out-of-scope entries; silently pass in-scope entries
              so they remain unpopulated for a later interactive run.
              Useful for a fast first pass to clear trivial entries.
            ``'update'`` — apply label fix in place. Safe only for
              Case A (no reject-status change); raises ValueError for
              Cases B and C, which require ``action='repopulate'``.
            ``'repopulate'`` — delete + repopulate CuratedSpikeSorting.
            ``'keep'`` — record decision, no label changes.
            ``'skip'`` — defer; row inserted with action='skip'.
        """
        from datetime import datetime

        from spyglass.common import LabMember

        # --- Step 0: clusterless thresholder is out of scope for Bug A
        # (no per-cluster curation), so auto-insert none_needed before
        # computing any diff or checking permissions.
        if key.get("sorter") == "clusterless_thresholder":
            self.insert1(
                {
                    **key,
                    "action": "none_needed",
                    "reviewed_by": "system",
                    "owner_member": "n/a",
                    "reviewed_at": datetime.now(),
                    "notes": "clusterless sorter — Bug A inapplicable",
                }
            )
            return

        # --- Steps 1–3: cheap scope + diff check (no permission needed)
        diff = self._compute_label_diff(key)
        if diff is None:
            self.insert1(
                {
                    **key,
                    "action": "none_needed",
                    "reviewed_by": "system",
                    "owner_member": "n/a",
                    "reviewed_at": datetime.now(),
                }
            )
            return

        # --- none_needed_only fast path: skip in-scope entries ---
        # Key remains unpopulated; populate() will retry on the next call.
        if action == "none_needed_only":
            return

        # --- Step 4: permission check ---
        dj_user = dj.config["database.user"]
        user_name = LabMember().get_djuser_name(dj_user)
        is_admin = dj_user in LabMember().admin
        owner_member = self._check_permission(key, user_name, is_admin)

        # --- Step 5: display diff ---
        self._print_diff(diff, owner_member)
        case = self._get_case(diff["reject_status_changed"])

        # --- Step 6: determine action ---
        if action == "report":
            action = self._prompt_action(case)
        elif action == "update" and case == "C":
            raise ValueError(
                "action='update' is not safe for this entry: some units "
                "that should now be accepted are absent from the NWB file "
                "(spike data was never written). Use action='repopulate'."
            )
        elif action == "update" and case == "B":
            raise ValueError(
                "action='update' is not safe for this entry: some units "
                "are newly rejected and CuratedSpikeSorting excludes "
                "rejected units at populate time, so patching labels "
                "would leave stale unit rows in the NWB/part-table. "
                "Use action='repopulate'."
            )

        # --- Step 7: execute ---
        auto_curation_key = diff["auto_curation_key"]
        new_labels = diff["new_labels"]
        # Use computed table (not Selection) to detect existing downstream rows
        has_downstream = len(CuratedSpikeSorting & auto_curation_key) > 0
        repopulated = 0
        nwb_staged = None

        if action == "update":
            # make() already runs inside the transaction opened by
            # Fix1513Status.populate(), so DB writes below share that
            # transaction (a nested `with ...connection.transaction:`
            # would raise "Nested connections are not supported").
            # Stage NWB patches first so HDF5 writes happen before the
            # transaction commits; on failure the temp files are discarded.
            staged = []
            if has_downstream:
                staged = self._stage_nwb_repairs(auto_curation_key, new_labels)
            try:
                Curation.update1(
                    {**auto_curation_key, "curation_labels": new_labels}
                )
                if has_downstream:
                    self._repair_unit_labels(auto_curation_key, new_labels)
            except Exception:
                for temp_path, _ in staged:
                    Path(temp_path).unlink(missing_ok=True)
                raise
            # Activation (os.replace + checksum) must happen only after
            # this transaction commits, otherwise a later failure (e.g.
            # the insert1 below) would roll back the DB while the NWB
            # files were already swapped. Persist the staged paths and
            # activate them via activate_pending_nwb_repairs(), called
            # after populate() returns.
            nwb_staged = staged or None
        elif action == "repopulate":
            Curation.update1(
                {**auto_curation_key, "curation_labels": new_labels}
            )
            if has_downstream:
                # safemode=False: avoid an interactive prompt and allow
                # this delete to run inside the populate() transaction.
                (CuratedSpikeSorting & auto_curation_key).delete(safemode=False)
            # CuratedSpikeSorting.populate() cannot run here: DataJoint
            # refuses populate() while a transaction is open (the one
            # Fix1513Status.populate() started). Repopulation is deferred
            # to run_pending_repopulates(), called after populate() returns.

        # diff is never None here (the diff is None branch already
        # inserted 'none_needed' and returned above), so label_diff is
        # recorded for every action — including 'update'/'repopulate',
        # where it is the only audit trail of the labels actually written.
        label_diff = {
            "old_labels": diff["old_labels"],
            "new_labels": diff["new_labels"],
        }
        self.insert1(
            {
                **key,
                "action": action,
                "reviewed_by": user_name,
                "owner_member": owner_member,
                "reviewed_at": datetime.now(),
                "repopulated": repopulated,
                "label_diff": label_diff,
                "nwb_staged": nwb_staged,
            }
        )

    # ------------------------------------------------------------------
    # Deferred repopulation (action='repopulate')
    # ------------------------------------------------------------------

    @classmethod
    def run_pending_repopulates(cls, restriction=True, verbose=True):
        """Run ``CuratedSpikeSorting.populate()`` for pending repopulates.

        ``make(action="repopulate")`` updates ``Curation.curation_labels``
        and deletes the stale ``CuratedSpikeSorting`` rows, but cannot call
        ``CuratedSpikeSorting.populate()`` itself: DataJoint refuses to start
        a ``populate()`` while a transaction is open, and ``make()`` runs
        inside the transaction opened by ``Fix1513Status.populate()``. Call
        this method afterwards, once that transaction has committed, to
        finish the repopulation and mark each entry ``repopulated=1``.

        Parameters
        ----------
        restriction : dict or str, optional
            Additional restriction on the pending ``Fix1513Status`` entries.
        verbose : bool, optional
            Print the ``AutomaticCuration`` key for each entry repopulated.

        Returns
        -------
        int
            Number of entries repopulated.
        """
        pending = cls() & "action='repopulate'" & "repopulated=0" & restriction
        n_done = 0
        for key in pending.fetch("KEY"):
            auto_curation_key = (AutomaticCuration & key).fetch1(
                "auto_curation_key"
            )
            CuratedSpikeSorting.populate(auto_curation_key)
            cls.update1({**key, "repopulated": 1})
            n_done += 1
            if verbose:
                print(f"Repopulated {auto_curation_key}")
        return n_done

    # ------------------------------------------------------------------
    # Deferred NWB activation (action='update')
    # ------------------------------------------------------------------

    @classmethod
    def activate_pending_nwb_repairs(cls, restriction=True, verbose=True):
        """Activate NWB label patches staged by ``make(action="update")``.

        ``make()`` stages NWB edits to temp files but cannot swap them
        into place (``os.replace`` + checksum update) before the
        ``Fix1513Status.populate()`` transaction commits — doing so would
        risk on-disk/DB inconsistency if a later step in that transaction
        failed. Call this afterwards, once that transaction
        has committed, to activate the staged patches.

        Safe to re-run: pairs already activated by a prior, interrupted
        call are skipped (see ``_activate_nwb_repairs``).

        Parameters
        ----------
        restriction : dict or str, optional
            Additional restriction on the pending ``Fix1513Status`` entries.
        verbose : bool, optional
            Print the ``AutomaticCuration`` key for each entry activated.

        Returns
        -------
        int
            Number of entries activated.
        """
        pending = cls() & "nwb_staged IS NOT NULL" & restriction
        n_done = 0
        for key in pending.fetch("KEY"):
            staged = (cls() & key).fetch1("nwb_staged")
            cls._activate_nwb_repairs(staged, verbose=verbose)
            cls.update1({**key, "nwb_staged": None})
            n_done += 1
            if verbose:
                print(f"Activated NWB repairs for {key}")
        return n_done
