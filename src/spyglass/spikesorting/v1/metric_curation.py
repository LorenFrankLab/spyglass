import os
import uuid
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
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.spikesorting.v1.sorting import SpikeSorting
from spyglass.spikesorting.v1.curation import Curation
from spyglass.utils.misc import generate_nwb_uuid
from spyglass.utils.dj_helper_fn import fetch_nwb

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
class MetricCurationParameter(dj.Lookup):
    definition = """
    metric_curation_param_name: varchar(200)
    ---
    merge_param: blob   # dict of param to merge units
    label_param: blob   # dict of param to label units
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
        merge_param = (MetricCurationParameter & key).fetch1("merge_param")
        label_param = (MetricCurationParameter & key).fetch1("label_param")

        # DO
        # create uuid for this metric curation
        key["metric_curation_id"] = generate_nwb_uuid(
            nwb_file_name=nwb_file_name, initial="MC"
        )
        # load recording and sorting
        recording = Curation.get_recording(key)
        sorting = Curation.get_sorting(key)
        # extract waveforms
        if "whiten" in waveform_param:
            if waveform_param.pop("whiten"):
                recording = sip.whiten(recording, dtype=np.float64)
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=os.environ.get("SPYGLASS_TEMP_DIR"),
            **waveform_param,
        )
        # compute metrics
        quality_metrics = {}
        for metric_name, metric_param_dict in metric_param.items():
            quality_metrics[metric_name] = self._compute_metric(
                waveforms, metric_name, **metric_param_dict
            )
        # generate labels and merge groups
        labels = self._compute_labels(
            parent_sorting, parent_labels, quality_metrics, label_params
        )

        merge_groups, units_merged = self._compute_merge_groups(
            parent_sorting, parent_merge_groups, quality_metrics, merge_params
        )
        # save everything in NWB
        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_metric_curation_to_nwb()

        # INSERT
        self.insert1(key)

    @classmethod
    def get_waveforms(cls):
        return NotImplementedError

    @classmethod
    def get_metrics(cls):
        return NotImplementedError

    @classmethod
    def get_labels(cls):
        return NotImplementedError

    @classmethod
    def get_merge_groups(cls):
        return NotImplementedError

    @staticmethod
    def _compute_metric(waveform_extractor, metric_name, **metric_params):
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

    @staticmethod
    def _compute_merge_groups(
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
    def _compute_labels(sorting, parent_labels, quality_metrics, label_params):
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


def _write_metric_curation_to_nwb(
    sorting_id: str,
    labels: Union[None, Dict[str, List[str]]] = None,
    merge_groups: Union[None, List[List[str]]] = None,
    metrics: Union[None, Dict[str, Dict[str, float]]] = None,
    apply_merge: bool = False,
):
    """Save sorting to NWB with curation information.
    Curation information is saved as columns in the units table of the NWB file.

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
    # FETCH:
    # - primary key for the associated sorting and recording
    sorting_key = (SpikeSorting & {"sorting_id": sorting_id}).fetch1()
    recording_key = (SpikeSortingRecording & sorting_key).fetch1()

    # original nwb file
    nwb_file_name = recording_key["nwb_file_name"]

    # get sorting
    sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
        sorting_key["analysis_file_name"]
    )
    sorting = se.read_nwb_sorting(sorting_analysis_file_abs_path)
    if apply_merge:
        sorting = sc.MergeUnitsSorting(
            parent_sorting=sorting, units_to_merge=merge_groups
        )
        merge_groups = None

    unit_ids = sorting.get_unit_ids()

    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()

        # add labels, merge groups, metrics
        if labels is not None:
            label_values = []
            for unit_id in unit_ids:
                if unit_id not in labels:
                    label_values.append("")
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
            for metric, metric_dict in zip(metrics):
                metric_values = []
                for unit_id in unit_ids:
                    if unit_id not in metric_dict:
                        metric_values.append(np.nan)
                    else:
                        metric_values.append(metric_dict[unit_id])
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        # write sorting to the nwb file
        for unit_id in unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
