import os
from typing import List, Union, Dict, Any

import datajoint as dj
import numpy as np
import pynwb
import spikeinterface as si
import spikeinterface.preprocessing as sp
import spikeinterface.qualitymetrics as sq

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.spikesorting.v1.sorting import SpikeSorting
from spyglass.spikesorting.v1.curation import (
    CurationV1,
    _list_to_merge_dict,
    _merge_dict_to_list,
)
from spyglass.spikesorting.v1.metric_utils import (
    get_num_spikes,
    get_peak_channel,
    get_peak_offset,
    compute_isi_violation_fractions,
)
from spyglass.utils.misc import generate_nwb_uuid

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
        cls.insert(cls.contents, skip_duplicates=True)


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
        cls.insert(cls.contents, skip_duplicates=True)

    @classmethod
    def show_available_metrics(self):
        for metric in _metric_name_to_func:
            metric_doc = _metric_name_to_func[metric].__doc__.split("\n")[0]
            print(f"{metric} : {metric_doc}\n")


@schema
class MetricCurationParameter(dj.Lookup):
    definition = """
    metric_curation_param_name: varchar(200)
    ---
    label_param: blob   # dict of param to label units
    merge_param: blob   # dict of param to merge units
    """

    contents = [
        ["default", {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}, {}],
        ["none", {}, {}],
    ]

    @classmethod
    def insert_default(cls):
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class MetricCurationSelection(dj.Manual):
    definition = """
    -> CurationV1
    -> WaveformParameter
    -> MetricParameter
    -> MetricCurationParameter
    """


@schema
class MetricCuration(dj.Computed):
    definition = """
    metric_curation_id: varchar(32)
    ---
    -> AutomaticCurationSelection
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    def make(self, key):
        # FETCH
        recording_id = (SpikeSorting & key).fetch1("recording_id")
        nwb_file_name = (SpikeSortingRecording & recording_id).fetch1(
            "nwb_file_name"
        )
        waveform_param = (WaveformParameter & key).fetch1("waveform_param")
        metric_param = (MetricParameter & key).fetch1("metric_param")
        label_param = (MetricCurationParameter & key).fetch1("label_param")
        merge_param = (MetricCurationParameter & key).fetch1("merge_param")

        # DO
        # create uuid for this metric curation
        key["metric_curation_id"] = generate_nwb_uuid(
            nwb_file_name=nwb_file_name, initial="MC", len_uuid=6
        )
        # load recording and sorting
        recording = CurationV1.get_recording(key)
        sorting = CurationV1.get_sorting(key)
        # extract waveforms
        if "whiten" in waveform_param:
            if waveform_param.pop("whiten"):
                recording = sp.whiten(recording, dtype=np.float64)

        waveforms_dir = (
            os.environ.get("SPYGLASS_TEMP_DIR")
            + "/"
            + key["metric_curation_id"]
        )
        try:
            os.mkdir(waveforms_dir)
        except FileExistsError:
            pass
        print("Extracting waveforms...")
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_dir,
            **waveform_param,
        )
        # compute metrics
        print("Computing metrics...")
        metrics = {}
        for metric_name, metric_param_dict in metric_param.items():
            metrics[metric_name] = self._compute_metric(
                nwb_file_name, waveforms, metric_name, **metric_param_dict
            )

        print("Applying curation...")
        labels = self._compute_labels(metrics, label_param)
        merge_groups = self._compute_merge_groups(metrics, merge_param)

        print("Saving to NWB...")
        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_metric_curation_to_nwb(
            waveforms, metrics, labels, merge_groups
        )

        # INSERT
        self.insert1(key)

    @classmethod
    def get_waveforms(cls):
        return NotImplementedError

    @classmethod
    def get_metrics(cls, key: dict):
        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        object_id = (cls & key).fetch1("object_id")
        metric_param_name = (cls & key).fetch1("metric_param_name")
        metric_param = (
            MetricParameter & {"metric_param_name": metric_param_name}
        ).fetch1("metric_param")
        metric_names = list(metric_param.keys())

        metrics = {}
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id]
            for metric_name in metric_names:
                metrics[metric_name] = dict(
                    zip(units[id][:], units[metric_name][:])
                )

        return metrics

    @classmethod
    def get_labels(cls, key: dict):
        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        object_id = (cls & key).fetch1("object_id")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id]
            labels = dict(zip(units[id][:], units["curation_labels"][:]))

        return labels

    @classmethod
    def get_merge_groups(cls, key: dict):
        analysis_file_name = (cls & key).fetch1("analysis_file_name")
        object_id = (cls & key).fetch1("object_id")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id]
            labels = dict(zip(units[id][:], units["merge_groups"][:]))

        return _merge_dict_to_list(labels)

    @staticmethod
    def _compute_metric(waveform_extractor, metric_name, **metric_params):
        peak_sign_metrics = ["snr", "peak_offset", "peak_channel"]
        metric_func = _metric_name_to_func[metric_name]
        # Not sure what this is doing; leaving in case someone is using them
        if metric_name in peak_sign_metrics:
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

    @staticmethod
    def _compute_labels(
        metrics: Dict[str, Dict[str, Union[float, List[float]]]],
        label_param: Dict[str, List[Any]],
    ) -> Dict[str, List[str]]:
        """Computes the labels based on the metric and label parameters.

        Parameters
        ---------
        quality_metrics : dict
            Example: {"snr" : {"1" : 2, "2" : 0.1, "3" : 2.3}}
            This indicates that the values of the "snr" quality metric
            for the units "1", "2", "3" are 2, 0.1, and 2.3, respectively.

        label_params : dict
            Example: {
                        "snr" : [(">", 1, ["good", "mua"]),
                                 ("<", 1, ["noise"])]
                     }
            This indicates that units with values of the "snr" quality metric
            greater than 1 should be given the labels "good" and "mua" and values
            less than 1 should be given the label "noise".

        Returns
        -------
        labels : dict
            Example: {"1" : ["good", "mua"], "2" : ["noise"], "3" : ["good", "mua"]}

        """
        if not label_param:
            return {}
        else:
            unit_ids = list(metrics[list(metrics.keys())[0]].keys())
            labels = {unit_id: [] for unit_id in unit_ids}
            for metric in label_param:
                if metric not in metrics:
                    Warning(f"{metric} not found in quality metrics; skipping")
                else:
                    for condition in label_param[metric]:
                        assert (
                            len(condition) == 3
                        ), f"Condition {condition} must be of length 3"
                        compare = _comparison_to_function[condition[0]]
                        for unit_id in unit_ids:
                            if compare(
                                metrics[metric][unit_id],
                                condition[1],
                            ):
                                labels[unit_id].extend(label_param[metric][2])
            return labels

    @staticmethod
    def _compute_merge_groups(
        metrics: Dict[str, Dict[str, Union[float, List[float]]]],
        merge_param: Dict[str, List[Any]],
    ) -> Dict[str, List[str]]:
        """Identifies units to be merged based on the metrics and merge parameters.

        Parameters
        ---------
        quality_metrics : dict
            Example: {"cosine_similarity" : {
                                             "1" : {"1" : 1.00, "2" : 0.10, "3": 0.95},
                                             "2" : {"1" : 0.10, "2" : 1.00, "3": 0.70},
                                             "3" : {"1" : 0.95, "2" : 0.70, "3": 1.00}
                                            }
            This shows the pairwise values of the "cosine_similarity" quality metric
            for the units "1", "2", "3" as a nested dict.

        merge_param : dict
            Example: {"cosine_similarity" : [">", 0.9]}
            This indicates that units with values of the "cosine_similarity" quality metric
            greater than 0.9 should be placed in the same merge group.


        Returns
        -------
        merge_groups : dict
            Example: {"1" : ["3"], "2" : [], "3" : ["1"]}

        """

        if not merge_param:
            return []
        else:
            unit_ids = list(metrics[list(metrics.keys())[0]].keys())
            merge_groups = {unit_id: [] for unit_id in unit_ids}
            for metric in merge_param:
                if metric not in metrics:
                    Warning(f"{metric} not found in quality metrics; skipping")
                else:
                    compare = _comparison_to_function[merge_param[metric][0]]
                    for unit_id in unit_ids:
                        other_unit_ids = [
                            other_unit_id
                            for other_unit_id in unit_ids
                            if other_unit_id != unit_id
                        ]
                        for other_unit_id in other_unit_ids:
                            if compare(
                                metrics[metric][unit_id][other_unit_id],
                                merge_param[metric][1],
                            ):
                                merge_groups[unit_id].extend(other_unit_id)
            return merge_groups


def _write_metric_curation_to_nwb(
    nwb_file_name: str,
    waveforms: si.WaveformExtractor,
    metrics: Union[None, Dict[str, Dict[str, float]]] = None,
    labels: Union[None, Dict[str, List[str]]] = None,
    merge_groups: Union[None, List[List[str]]] = None,
):
    """Save waveforms, metrics, labels, and merge groups to NWB

    Parameters
    ----------
    sorting_id : str
        key for the sorting
    labels : dict or None, optional
        curation labels (e.g. good, noise, mua)
    merge_groups : list or None, optional
        groups of unit IDs to be merged
    metrics : dict or None, optional
        Computed quality metrics, one for each cell
    apply_merge : bool, optional
        whether to apply the merge groups to the sorting before saving, by default False

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting and curation information
    object_id : str
        object_id of the units table in the analysis NWB file
    """

    unit_ids = [int(i) for i in waveforms.sorting.get_unit_ids()]

    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # write waveforms to the nwb file
        for unit_id in unit_ids:
            nwbf.add_unit(
                spike_times=waveforms.sorting.get_unit_spike_train(unit_id),
                id=unit_id,
                electrodes=waveforms.recording.get_channel_ids(),
                waveforms=waveforms.get_waveforms(unit_id),
                waveform_mean=waveforms.get_template(unit_id),
            )

        # add labels, merge groups, metrics
        if labels is not None:
            label_values = []
            for unit_id in unit_ids:
                if unit_id not in labels:
                    label_values.append([])
                else:
                    label_values.append(labels[unit_id])
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label",
                data=label_values,
            )
        if merge_groups is not None:
            merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
            nwbf.add_unit_column(
                name="merge_groups",
                description="merge groups",
                data=list(merge_groups_dict.values()),
            )
        if metrics is not None:
            for metric, metric_dict in metrics.items():
                metric_values = []
                for unit_id in unit_ids:
                    if unit_id not in metric_dict:
                        metric_values.append([])
                    else:
                        metric_values.append(metric_dict[unit_id])
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
