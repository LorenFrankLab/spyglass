import json
import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import List

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as sip
import spikeinterface.qualitymetrics as sq

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.metric_utils import (
    get_num_spikes,
    get_peak_channel,
    get_peak_offset,
    compute_isi_violation_fractions,
)
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.spikesorting.v1.sorting import SpikeSorting
from spyglass.spikesorting.merge import SpikeSortingOutput

schema = dj.schema("spikesorting_v1_metric_curation")


_metric_name_to_func = {
    "snr": sq.compute_snrs,
    "isi_violation": compute_isi_violation_fractions,
    "nn_isolation": sq.nearest_neighbors_isolation,
    "nn_noise_overlap": sq.nearest_neighbors_noise_overlap,
    "peak_offset": get_peak_offset,
    "peak_channel": get_peak_channel,
    "num_spikes": get_num_spikes,
}

_comparison_to_function = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
}


@schema
class WaveformParameter(dj.Lookup):
    definition = """
    waveform_param_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_param: blob # a dict of waveform extraction parameters
    """

    contents = [
        [
            "default_not_whitened",
            {
                "ms_before": 0.5,
                "ms_after": 0.5,
                "max_spikes_per_unit": 5000,
                "n_jobs": 5,
                "total_memory": "5G",
                "whiten": False,
            },
        ],
        [
            "default_whitened",
            {
                "ms_before": 0.5,
                "ms_after": 0.5,
                "max_spikes_per_unit": 5000,
                "n_jobs": 5,
                "total_memory": "5G",
                "whiten": True,
            },
        ],
    ]

    @classmethod
    def insert_default(cls):
        cls.insert1(cls.contents, skip_duplicates=True)


@schema
class MetricParameter(dj.Lookup):
    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_param_name: varchar(200)
    ---
    metric_param: blob
    """
    metric_default_param_name = "franklab_default"
    metric_default_param = {
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
    contents = [[metric_default_param_name, metric_default_param]]

    @classmethod
    def insert_default(cls):
        cls.insert1(
            cls.contents,
            skip_duplicates=True,
        )

    @classmethod
    def show_available_metrics(self):
        for metric in _metric_name_to_func:
            metric_doc = _metric_name_to_func[metric].__doc__.split("\n")[0]
            print(f"{metric} : {metric_doc}\n")


@schema
class AutomaticCurationParameter(dj.Lookup):
    definition = """
    auto_curation_param_name: varchar(200)
    ---
    merge_params: blob   # dict of param to merge units
    label_params: blob   # dict of param to label units
    """

    contents = [
        ["default", {}, {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}],
        ["none", {}, {}],
    ]

    @classmethod
    def insert_default(cls):
        cls.insert1(cls.contents, skip_duplicates=True)


@schema
class MetricCurationSelection(dj.Manual):
    definition = """
    -> Curation
    -> WaveformParameter
    -> MetricParameter
    -> AutomaticCurationParameter
    """


@schema
class MetricCuration(dj.Computed):
    definition = """
    metric_curation_id: varchar(32)
    ---
    -> AutomaticCurationSelection
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    metrics: longblob
    labels: longblob
    merge_groups: longblob
    """

    def make(self, key):
        # FETCH
        # load recording and sorting

        # DO
        # create uuid for this metric curation
        # extract waveforms
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=os.environ.get("SPYGLASS_TEMP_DIR"),
            **waveform_params,
        )
        # compute metrics
        params = (MetricParameters & key).fetch1("metric_params")
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(
                waveform_extractor, metric_name, **metric_params
            )
        qm[metric_name] = metric
        # generate labels and merge groups
        labels = self.get_labels(
            parent_sorting, parent_labels, quality_metrics, label_params
        )

        merge_groups, units_merged = self.get_merge_groups(
            parent_sorting, parent_merge_groups, quality_metrics, merge_params
        )
        # save everything in NWB

        # INSERT

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
        parent_labels : list
            Information about previous merges
        quality_metrics : list
        label_params : dict

        Returns
        -------
        parent_labels : list

        """
        # overview:
        # 1. Use quality metrics to determine labels for units
        # 2. Append labels to current labels, checking for inconsistencies
        if not label_params:
            return parent_labels
        else:
            for metric in label_params:
                if metric not in quality_metrics:
                    Warning(f"{metric} not found in quality metrics; skipping")
                else:
                    compare = _comparison_to_function[label_params[metric][0]]

                    for unit_id in quality_metrics[metric].keys():
                        # compare the quality metric to the threshold with the specified operator
                        # note that label_params[metric] is a three element list with a comparison operator as a string,
                        # the threshold value, and a list of labels to be applied if the comparison is true
                        if compare(
                            quality_metrics[metric][unit_id],
                            label_params[metric][1],
                        ):
                            if unit_id not in parent_labels:
                                parent_labels[unit_id] = label_params[metric][2]
                            # check if the label is already there, and if not, add it
                            elif (
                                label_params[metric][2]
                                not in parent_labels[unit_id]
                            ):
                                parent_labels[unit_id].extend(
                                    label_params[metric][2]
                                )
            return parent_labels


@schema
class Waveforms(dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    waveform_extractor_path: varchar(400)
    -> AnalysisNwbfile
    waveforms_object_id: varchar(40)   # Object ID for the waveforms in NWB file
    """

    def make(self, key):
        recording = Curation.get_recording(key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])

        sorting = Curation.get_curated_sorting(key)

        print("Extracting waveforms...")
        waveform_params = (WaveformParameters & key).fetch1("waveform_params")
        if "whiten" in waveform_params:
            if waveform_params.pop("whiten"):
                recording = sip.whiten(recording, dtype="float32")

        waveform_extractor_name = self._get_waveform_extractor_name(key)
        key["waveform_extractor_path"] = str(
            Path(os.environ["SPYGLASS_WAVEFORMS_DIR"])
            / Path(waveform_extractor_name)
        )
        if os.path.exists(key["waveform_extractor_path"]):
            shutil.rmtree(key["waveform_extractor_path"])
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=key["waveform_extractor_path"],
            **waveform_params,
        )

        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
        object_id = AnalysisNwbfile().add_units_waveforms(
            key["analysis_file_name"], waveform_extractor=waveforms
        )
        key["waveforms_object_id"] = object_id
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

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
        we_path = (self & key).fetch1("waveform_extractor_path")
        we = si.WaveformExtractor.load_from_folder(we_path)
        return we

    def fetch_nwb(self, key):
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError

    def _get_waveform_extractor_name(self, key):
        waveform_params_name = (WaveformParameters & key).fetch1(
            "waveform_params_name"
        )

        return (
            f'{key["nwb_file_name"]}_{str(uuid.uuid4())[0:8]}_'
            f'{key["curation_id"]}_{waveform_params_name}_waveforms'
        )


@schema
class MetricSelection(dj.Manual):
    definition = """
    -> Waveforms
    -> MetricParameters
    ---
    """

    def insert1(self, key, **kwargs):
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
class QualityMetrics(dj.Computed):
    definition = """
    -> MetricSelection
    ---
    quality_metrics_path: varchar(500)
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    def make(self, key):
        waveform_extractor = Waveforms().load_waveforms(key)
        qm = {}
        params = (MetricParameters & key).fetch1("metric_params")
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(
                waveform_extractor, metric_name, **metric_params
            )
            qm[metric_name] = metric
        qm_name = self._get_quality_metrics_name(key)
        key["quality_metrics_path"] = str(
            Path(os.environ["SPYGLASS_WAVEFORMS_DIR"]) / Path(qm_name + ".json")
        )
        # save metrics dict as json
        print(f"Computed all metrics: {qm}")
        self._dump_to_json(qm, key["quality_metrics_path"])

        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
        key["object_id"] = AnalysisNwbfile().add_units_metrics(
            key["analysis_file_name"], metrics=qm
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms()._get_waveform_extractor_name(key)
        qm_name = wf_name + "_qm"
        return qm_name

    def _compute_metric(self, waveform_extractor, metric_name, **metric_params):
        peak_sign_metrics = ["snr", "peak_offset", "peak_channel"]
        metric_func = _metric_name_to_func[metric_name]
        # TODO clean up code below
        if metric_name == "isi_violation":
            metric = metric_func(waveform_extractor, **metric_params)
        elif metric_name in peak_sign_metrics:
            if "peak_sign" in metric_params:
                metric = metric_func(
                    waveform_extractor,
                    peak_sign=metric_params.pop("peak_sign"),
                    **metric_params,
                )
            else:
                raise Exception(
                    f"{peak_sign_metrics} metrics require peak_sign",
                    f"to be defined in the metric parameters",
                )
        else:
            metric = {}
            for unit_id in waveform_extractor.sorting.get_unit_ids():
                metric[str(unit_id)] = metric_func(
                    waveform_extractor, this_unit_id=unit_id, **metric_params
                )
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
