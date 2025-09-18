import uuid
from pathlib import Path
from typing import Any, Dict, List, Union

import datajoint as dj
import numpy as np
import pynwb
import spikeinterface as si
import spikeinterface.preprocessing as sp
import spikeinterface.qualitymetrics as sq

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting.v1.curation import (
    CurationV1,
    _list_to_merge_dict,
    _merge_dict_to_list,
)
from spyglass.spikesorting.v1.metric_utils import (
    compute_isi_violation_fractions,
    get_num_spikes,
    get_peak_channel,
    get_peak_offset,
)
from spyglass.spikesorting.v1.sorting import SpikeSortingSelection
from spyglass.utils import SpyglassMixin, logger

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
class WaveformParameters(SpyglassMixin, dj.Lookup):
    """Parameters for extracting waveforms from the recording based on sorting.

    Attributes
    ----------
    waveform_param_name : str
        Name of the waveform extraction parameters.
    waveform_params : dict
        A dictionary of waveform extraction parameters, including...
        ms_before : float
            Number of milliseconds before the spike time to include in the
            waveform.
        ms_after : float
            Number of milliseconds after the spike time to include in the
            waveform.
        max_spikes_per_unit : int
            Maximum number of spikes to include in the waveform for each unit.
        n_jobs : int
            Number of parallel jobs to use for waveform extraction.
        total_memory : str
            Total memory available for waveform extraction e.g. "5G".
        whiten : bool
            Whether to whiten the waveforms or not.
    """

    definition = """
    # Parameters for extracting waveforms from the recording based on the sorting.
    waveform_param_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """

    default_params = {
        "ms_before": 0.5,
        "ms_after": 0.5,
        "max_spikes_per_unit": 5000,
        "n_jobs": 5,
        "total_memory": "5G",
    }
    contents = [
        ["default_not_whitened", {**default_params, "whiten": False}],
        ["default_whitened", {**default_params, "whiten": True}],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default waveform parameters."""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class MetricParameters(SpyglassMixin, dj.Lookup):
    """Parameters for computing quality metrics of sorted units.

    See MetricParameters().show_available_metrics() for a list of available
    metrics and their descriptions.
    """

    definition = """
    # Parameters for computing quality metrics of sorted units.
    metric_param_name: varchar(200)
    ---
    metric_params: blob
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
        """Insert default metric parameters."""
        cls.insert(cls.contents, skip_duplicates=True)

    @classmethod
    def show_available_metrics(self):
        """Prints the available metrics and their descriptions."""
        for metric in _metric_name_to_func:
            metric_doc = _metric_name_to_func[metric].__doc__.split("\n")[0]
            logger.info(f"{metric} : {metric_doc}\n")


@schema
class MetricCurationParameters(SpyglassMixin, dj.Lookup):
    """Parameters for automatic curation of spike sorting

    Attributes
    ----------
    metric_curation_params_name : str
        Name of the automatic curation parameters
    label_params : dict, optional
        Dictionary of parameters for labeling units
    merge_params : dict, optional
        Dictionary of parameters for merging units. May include nn_noise_overlap
        List[comparison operator: str, threshold: float, labels: List[str]]
    """

    definition = """
    # Parameters for curating a spike sorting based on the metrics.
    metric_curation_param_name: varchar(200)
    ---
    label_params: blob   # dict of param to label units
    merge_params: blob   # dict of param to merge units
    """

    contents = [
        ["default", {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}, {}],
        ["none", {}, {}],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default metric curation parameters."""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class MetricCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Spike sorting and parameters for metric curation. Use `insert_selection` to insert a row into this table.
    metric_curation_id: uuid
    ---
    -> CurationV1
    -> WaveformParameters
    -> MetricParameters
    -> MetricCurationParameters
    """

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into MetricCurationSelection with an
        automatically generated unique metric curation ID as the sole primary key.

        Parameters
        ----------
        key : dict
            primary key of CurationV1, WaveformParameters, MetricParameters MetricCurationParameters

        Returns
        -------
        key : dict
            key for the inserted row
        """
        if cls & key:
            logger.warning("This row has already been inserted.")
            return (cls & key).fetch1()
        key["metric_curation_id"] = uuid.uuid4()
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class MetricCuration(SpyglassMixin, dj.Computed):
    definition = """
    # Results of applying curation based on quality metrics. To do additional curation, insert another row in `CurationV1`
    -> MetricCurationSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    _use_transaction, _allow_insert = False, True
    _waves_cache = {}  # Cache waveforms for burst merge

    def make(self, key):
        """Populate MetricCuration table.

        1. Fetches...
            - Waveform parameters from WaveformParameters
            - Metric parameters from MetricParameters
            - Label and merge parameters from MetricCurationParameters
            - Sorting ID and curation ID from MetricCurationSelection
        2. Loads the recording and sorting from CurationV1.
        3. Optionally whitens the recording with spikeinterface
        4. Extracts waveforms from the recording based on the sorting.
        5. Optionally computes quality metrics for the units.
        6. Applies curation based on the metrics, computing labels and merge
            groups.
        7. Saves the waveforms, metrics, labels, and merge groups to an
            analysis NWB file and inserts into MetricCuration table.
        """
        # FETCH
        upstream = (
            SpikeSortingSelection
            * WaveformParameters
            * MetricParameters
            * MetricCurationParameters
            * MetricCurationSelection
            & key
        ).fetch1()

        nwb_file_name = upstream["nwb_file_name"]
        metric_params = upstream["metric_params"]
        label_params = upstream["label_params"]
        merge_params = upstream["merge_params"]

        # DO
        logger.info("Extracting waveforms...")
        waveforms = self.get_waveforms(key)

        # compute metrics
        logger.info("Computing metrics...")
        metrics = {}
        for metric_name, metric_param_dict in metric_params.items():
            metrics[metric_name] = self._compute_metric(
                waveforms, metric_name, **metric_param_dict
            )
        if metrics["nn_isolation"]:
            metrics["nn_isolation"] = {
                unit_id: value[0]
                for unit_id, value in metrics["nn_isolation"].items()
            }

        logger.info("Applying curation...")
        labels = self._compute_labels(metrics, label_params)
        merge_groups = self._compute_merge_groups(metrics, merge_params)

        logger.info("Saving to NWB...")
        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_metric_curation_to_nwb(
            nwb_file_name, waveforms, metrics, labels, merge_groups
        )

        # INSERT
        AnalysisNwbfile().add(
            nwb_file_name,
            key["analysis_file_name"],
        )
        self.insert1(key)

    def get_waveforms(
        self, key: dict, overwrite: bool = True, fetch_all: bool = False
    ):
        """Returns waveforms identified by metric curation.

        Parameters
        ----------
        key : dict
            primary key to MetricCuration
        overwrite : bool, optional
            whether to overwrite existing waveforms, by default True
        fetch_all : bool, optional
            fetch all spikes for units, by default False. Overrides
            max_spikes_per_unit in waveform_params
        """
        key_hash = dj.hash.key_hash(key)
        if cached := self._waves_cache.get(key_hash):
            return cached

        query = (MetricCurationSelection & key) * WaveformParameters
        if len(query) != 1:
            raise ValueError(f"Found {len(query)} entries for: {key}")

        sort_key = query.fetch("sorting_id", "curation_id", as_dict=True)[0]
        recording = CurationV1.get_recording(sort_key)
        sorting = CurationV1.get_sorting(sort_key)

        # extract waveforms
        waveform_params = query.fetch1("waveform_params")
        if "whiten" in waveform_params:
            if waveform_params.pop("whiten"):
                recording = sp.whiten(recording, dtype=np.float64)

        waveforms_dir = temp_dir + "/" + str(key["metric_curation_id"])
        wf_dir_obj = Path(waveforms_dir)
        wf_dir_obj.mkdir(parents=True, exist_ok=True)
        if not any(wf_dir_obj.iterdir()):  # if the directory is empty
            overwrite = True

        if fetch_all:
            waveform_params["max_spikes_per_unit"] = None
            waveforms_dir += "_all"

        # Extract non-sparse waveforms by default
        waveform_params.setdefault("sparse", False)
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_dir,
            overwrite=overwrite,
            load_if_exists=not overwrite,
            **waveform_params,
        )

        self._waves_cache[key_hash] = waveforms

        return waveforms

    @classmethod
    def get_metrics(cls, key: dict):
        """Returns metrics identified by metric curation

        Parameters
        ----------
        key : dict
            primary key to MetricCuration
        """
        analysis_file_name, object_id, metric_param_name, metric_params = (
            cls * MetricCurationSelection * MetricParameters & key
        ).fetch1(
            "analysis_file_name",
            "object_id",
            "metric_param_name",
            "metric_params",
        )
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id].to_dataframe()
        return {
            name: dict(zip(units.index, units[name])) for name in metric_params
        }

    @classmethod
    def get_labels(cls, key: dict):
        """Returns curation labels identified by metric curation

        Parameters
        ----------
        key : dict
            primary key to MetricCuration
        """
        analysis_file_name, object_id = (cls & key).fetch1(
            "analysis_file_name", "object_id"
        )
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id].to_dataframe()
        return dict(zip(units.index, units["curation_label"]))

    @classmethod
    def get_merge_groups(cls, key: dict):
        """Returns merge groups identified by metric curation

        Parameters
        ----------
        key : dict
            primary key to MetricCuration
        """
        analysis_file_name, object_id = (cls & key).fetch1(
            "analysis_file_name", "object_id"
        )
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path,
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            units = nwbf.objects[object_id].to_dataframe()
        merge_group_dict = dict(zip(units.index, units["merge_groups"]))

        return _merge_dict_to_list(merge_group_dict)

    @staticmethod
    def _compute_metric(waveform_extractor, metric_name, **metric_params):
        metric_func = _metric_name_to_func[metric_name]

        peak_sign_metrics = ["snr", "peak_offset", "peak_channel"]
        if metric_name in peak_sign_metrics:
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

        return {
            unit_id: metric_func(waveform_extractor, this_unit_id=unit_id)
            for unit_id in waveform_extractor.sorting.get_unit_ids()
        }

    @staticmethod
    def _compute_labels(
        metrics: Dict[str, Dict[str, Union[float, List[float]]]],
        label_params: Dict[str, List[Any]],
    ) -> Dict[str, List[str]]:
        """Computes the labels based on the metric and label parameters.

        Parameters
        ----------
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
        if not label_params:
            return {}

        unit_ids = [
            unit_id for unit_id in metrics[list(metrics.keys())[0]].keys()
        ]
        labels = {unit_id: [] for unit_id in unit_ids}

        for metric in label_params:
            if metric not in metrics:
                Warning(f"{metric} not found in quality metrics; skipping")
                continue

            condition = label_params[metric]
            if not len(condition) == 3:
                raise ValueError(f"Condition {condition} must be of length 3")

            compare = _comparison_to_function[condition[0]]
            for unit_id in unit_ids:
                if compare(
                    metrics[metric][unit_id],
                    condition[1],
                ):
                    labels[unit_id].extend(label_params[metric][2])
        return labels

    @staticmethod
    def _compute_merge_groups(
        metrics: Dict[str, Dict[str, Union[float, List[float]]]],
        merge_params: Dict[str, List[Any]],
    ) -> Dict[str, List[str]]:
        """Identifies units to be merged based on the metrics and merge parameters.

        Parameters
        ---------
        quality_metrics : dict
            Example: {"cosine_similarity" : {
                                             "1" : {"1" : 1.00, "2" : 0.10, "3": 0.95},
                                             "2" : {"1" : 0.10, "2" : 1.00, "3": 0.70},
                                             "3" : {"1" : 0.95, "2" : 0.70, "3": 1.00}
                                            }}
            This shows the pairwise values of the "cosine_similarity" quality metric
            for the units "1", "2", "3" as a nested dict.

        merge_params : dict
            Example: {"cosine_similarity" : [">", 0.9]}
            This indicates that units with values of the "cosine_similarity" quality metric
            greater than 0.9 should be placed in the same merge group.


        Returns
        -------
        merge_groups : dict
            Example: {"1" : ["3"], "2" : [], "3" : ["1"]}

        """

        if not merge_params:
            return []

        unit_ids = list(metrics[list(metrics.keys())[0]].keys())
        merge_groups = {unit_id: [] for unit_id in unit_ids}
        for metric in merge_params:
            if metric not in metrics:
                Warning(f"{metric} not found in quality metrics; skipping")
                continue
            compare = _comparison_to_function[merge_params[metric][0]]
            for unit_id in unit_ids:
                other_unit_ids = [
                    other_unit_id
                    for other_unit_id in unit_ids
                    if other_unit_id != unit_id
                ]
                for other_unit_id in other_unit_ids:
                    if compare(
                        metrics[metric][unit_id][other_unit_id],
                        merge_params[metric][1],
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
    """Save waveforms, metrics, labels, and merge groups to NWB in the units table.

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
    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)

    unit_ids = [int(i) for i in waveforms.sorting.get_unit_ids()]

    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # Write waveforms to the nwb file
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
                index=True,
            )
        if merge_groups is not None:
            merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
            merge_groups_list = [
                [""] for i in merge_groups_dict.values() if i == []
            ]
            nwbf.add_unit_column(
                name="merge_groups",
                description="merge groups",
                data=merge_groups_list,
                index=True,
            )
        if metrics is not None:
            for metric, metric_dict in metrics.items():
                metric_values = [
                    metric_dict[unit_id] if unit_id in metric_dict else []
                    for unit_id in unit_ids
                ]
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
