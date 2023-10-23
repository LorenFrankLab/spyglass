from typing import Any, Union, List, Dict

import datajoint as dj
import pynwb

import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.sorting import SpikeSorting
from spyglass.spikesorting.v1.curation import CurationV1, _merge_dict_to_list

import kachery_cloud as kcl
import sortingview.views as vv
from sortingview.SpikeSortingView import SpikeSortingView

schema = dj.schema("spikesorting_v1_figurl_curation")


@schema
class FigURLCurationSelection(dj.Manual):
    definition = """
    -> CurationV1
    curation_uri: varchar(1000)     # GitHub-based URI to a file to which the manual curation will be saved
    ---
    metrics_figurl: blob            # metrics to display in the figURL
    """

    @staticmethod
    def generate_curation_uri(key: Dict) -> str:
        """Generates a kachery-cloud URI containing curation info from a row in CurationV1 table

        Parameters
        ----------
        key : dict
            primary key from CurationV1
        """
        curation_key = (CurationV1 & key).fetch1()
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            curation_key["analysis_file_name"]
        )
        with pynwb.NWBHDF5IO(
            analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            nwb_sorting = nwbfile.objects[curation_key["object_id"]]
            unit_ids = nwb_sorting["id"][:]
            labels = nwb_sorting["labels"][:]
            merge_groups = nwb_sorting["merge_groups"][:]

        unit_ids = [str(unit_id) for unit_id in unit_ids]

        if labels:
            labels_dict = dict(zip(unit_ids, labels))
        else:
            labels_dict = {}

        if merge_groups:
            merge_groups_list = _merge_dict_to_list(merge_groups)
            merge_groups_list = [
                [str(unit_id) for unit_id in merge_group]
                for merge_group in merge_groups_list
            ]
        else:
            merge_groups_list = []

        curation_dict = {
            "labelsByUnit": labels_dict,
            "mergeGroups": merge_groups_list,
        }
        curation_uri = kcl.store_json(curation_dict)

        return curation_uri


@schema
class FigURLCuration(dj.Computed):
    definition = """
    -> FigURLCurationSelection
    ---
    url: varchar(1000)
    """

    def make(self, key: dict):
        # FETCH
        sorting_analysis_file_name = (CurationV1 & key).fetch1(
            "analysis_file_name"
        )
        object_id = (CurationV1 & key).fetch1("object_id")
        recording_label = (SpikeSorting & key).fetch1("recording_id")

        # DO
        sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            sorting_analysis_file_name
        )
        recording = CurationV1.get_recording(key)
        sorting = CurationV1.get_sorting(key)
        sorting_label = key["sorting_id"]
        curation_uri = key["curation_uri"]

        metric_dict = {}
        with pynwb.NWBHDF5IO(
            sorting_analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            nwb_sorting = nwbf.objects[object_id]
            unit_ids = nwb_sorting["id"][:]
            for metric in key["metrics_figurl"]:
                metric_dict[metric] = dict(
                    zip(unit_ids, nwb_sorting[metric][:])
                )

        unit_metrics = _reformat_metrics(metric_dict)

        # TODO: figure out a way to specify the similarity metrics

        # Generate the figURL
        key["url"] = _generate_figurl(
            R=recording,
            S=sorting,
            initial_curation_uri=curation_uri,
            recording_label=recording_label,
            sorting_label=sorting_label,
            unit_metrics=unit_metrics,
        )

        # INSERT
        self.insert1(key, skip_duplicates=True)

    @classmethod
    def get_labels(cls):
        return NotImplementedError

    @classmethod
    def get_merge_groups(cls):
        return NotImplementedError


def _generate_figurl(
    R: si.BaseRecording,
    S: si.BaseSorting,
    initial_curation_uri: str,
    recording_label: str,
    sorting_label: str,
    unit_metrics: Union[List[Any], None] = None,
    segment_duration_sec=1200,
    snippet_ms_before=1,
    snippet_ms_after=1,
    max_num_snippets_per_segment=1000,
    channel_neighborhood_size=5,
    raster_plot_subsample_max_firing_rate=50,
    spike_amplitudes_subsample_max_firing_rate=50,
):
    print("Preparing spikesortingview data")
    sampling_frequency = R.get_sampling_frequency()
    X = SpikeSortingView.create(
        recording=R,
        sorting=S,
        segment_duration_sec=segment_duration_sec,
        snippet_len=(
            int(snippet_ms_before * sampling_frequency / 1000),
            int(snippet_ms_after * sampling_frequency / 1000),
        ),
        max_num_snippets_per_segment=max_num_snippets_per_segment,
        channel_neighborhood_size=channel_neighborhood_size,
    )

    # create a fake unit similarity matrix (for future reference)
    # similarity_scores = []
    # for u1 in X.unit_ids:
    #     for u2 in X.unit_ids:
    #         similarity_scores.append(
    #             vv.UnitSimilarityScore(
    #                 unit_id1=u1,
    #                 unit_id2=u2,
    #                 similarity=similarity_matrix[(X.unit_ids==u1),(X.unit_ids==u2)]
    #             )
    #         )
    # # Create the similarity matrix view
    # unit_similarity_matrix_view = vv.UnitSimilarityMatrix(
    #    unit_ids=X.unit_ids,
    #    similarity_scores=similarity_scores
    #    )

    # Assemble the views in a layout
    # You can replace this with other layouts
    raster_plot_subsample_max_firing_rate = (
        raster_plot_subsample_max_firing_rate
    )
    spike_amplitudes_subsample_max_firing_rate = (
        spike_amplitudes_subsample_max_firing_rate
    )
    view = vv.MountainLayout(
        items=[
            vv.MountainLayoutItem(
                label="Summary", view=X.sorting_summary_view()
            ),
            vv.MountainLayoutItem(
                label="Units table",
                view=X.units_table_view(
                    unit_ids=X.unit_ids, unit_metrics=unit_metrics
                ),
            ),
            vv.MountainLayoutItem(
                label="Raster plot",
                view=X.raster_plot_view(
                    unit_ids=X.unit_ids,
                    _subsample_max_firing_rate=raster_plot_subsample_max_firing_rate,
                ),
            ),
            vv.MountainLayoutItem(
                label="Spike amplitudes",
                view=X.spike_amplitudes_view(
                    unit_ids=X.unit_ids,
                    _subsample_max_firing_rate=spike_amplitudes_subsample_max_firing_rate,
                ),
            ),
            vv.MountainLayoutItem(
                label="Autocorrelograms",
                view=X.autocorrelograms_view(unit_ids=X.unit_ids),
            ),
            vv.MountainLayoutItem(
                label="Cross correlograms",
                view=X.cross_correlograms_view(unit_ids=X.unit_ids),
            ),
            vv.MountainLayoutItem(
                label="Avg waveforms",
                view=X.average_waveforms_view(unit_ids=X.unit_ids),
            ),
            vv.MountainLayoutItem(
                label="Electrode geometry", view=X.electrode_geometry_view()
            ),
            # vv.MountainLayoutItem(
            #    label='Unit similarity matrix',
            #    view=unit_similarity_matrix_view
            # ),
            vv.MountainLayoutItem(
                label="Curation", view=vv.SortingCuration2(), is_control=True
            ),
        ]
    )
    url_state = {
        "initialSortingCuration": initial_curation_uri,
        "sortingCuration": initial_curation_uri,
    }
    label = f"{recording_label} {sorting_label}"
    url = view.url(label=label, state=url_state)
    return url


def _reformat_metrics(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    for metric_name in metrics:
        metrics[metric_name] = {
            str(unit_id): metric_value
            for unit_id, metric_value in metrics[metric_name].items()
        }
    new_external_metrics = [
        {
            "name": metric_name,
            "label": metric_name,
            "tooltip": metric_name,
            "data": metric,
        }
        for metric_name, metric in metrics.items()
    ]
    return new_external_metrics
