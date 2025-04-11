import uuid
from typing import Any, Dict, List, Union

import datajoint as dj
import kachery_cloud as kcl
import pynwb
import sortingview.views as vv
import spikeinterface as si
from sortingview.SpikeSortingView import SpikeSortingView

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.utils import _reformat_metrics
from spyglass.spikesorting.v1.curation import CurationV1, _merge_dict_to_list
from spyglass.spikesorting.v1.sorting import SpikeSortingSelection
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v1_figurl_curation")


@schema
class FigURLCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Use `insert_selection` method to insert a row. Use `generate_curation_uri` method to generate a curation uri.
    figurl_curation_id: uuid
    ---
    -> CurationV1
    curation_uri: varchar(1000)     # GitHub-based URI to a file to which the manual curation will be saved
    metrics_figurl: blob            # metrics to display in the figURL
    """

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into FigURLCurationSelection.

        Parameters
        ----------
        key : dict
            primary key of `CurationV1`, `curation_uri`, and `metrics_figurl`.
            - If `curation_uri` is not provided, it will be generated from `generate_curation_uri` method.
            - If `metrics_figurl` is not provided, it will be set to [].

        Returns
        -------
        key : dict
            primary key of `FigURLCurationSelection` table.
        """
        if "curation_uri" not in key:
            key["curation_uri"] = cls.generate_curation_uri(key)
        if "metrics_figurl" not in key:
            key["metrics_figurl"] = []
        if "figurl_curation_id" in key:
            query = cls & {"figurl_curation_id": key["figurl_curation_id"]}
            if query:
                logger.warning("Similar row(s) already inserted.")
                return query.fetch(as_dict=True)
        key["figurl_curation_id"] = uuid.uuid4()
        cls.insert1(key, skip_duplicates=True)
        return key

    @staticmethod
    def generate_curation_uri(key: Dict) -> str:
        """Generates a kachery-cloud URI from a row in CurationV1 table

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
            nwb_sorting = nwbfile.objects[
                curation_key["object_id"]
            ].to_dataframe()
            unit_ids = list(nwb_sorting.index)
            labels = list(nwb_sorting["curation_label"])
            merge_groups = list(nwb_sorting["merge_groups"])

        unit_ids = [str(unit_id) for unit_id in unit_ids]

        labels_dict = (
            {unit_id: list(label) for unit_id, label in zip(unit_ids, labels)}
            if labels
            else {}
        )

        merge_groups_list = (
            [
                [str(unit_id) for unit_id in merge_group]
                for merge_group in _merge_dict_to_list(
                    dict(zip(unit_ids, merge_groups))
                )
            ]
            if merge_groups
            else []
        )

        return kcl.store_json(
            {
                "labelsByUnit": labels_dict,
                "mergeGroups": merge_groups_list,
            }
        )


@schema
class FigURLCuration(SpyglassMixin, dj.Computed):
    definition = """
    # URL to the FigURL for manual curation of a spike sorting.
    -> FigURLCurationSelection
    ---
    url: varchar(1000)
    """

    _use_transaction, _allow_insert = False, True

    def make(self, key: dict):
        """Generate a FigURL for manual curation of a spike sorting."""
        # FETCH
        query = (
            FigURLCurationSelection * CurationV1 * SpikeSortingSelection & key
        )
        (
            sorting_fname,
            object_id,
            recording_label,
            metrics_figurl,
        ) = query.fetch1(
            "analysis_file_name", "object_id", "recording_id", "metrics_figurl"
        )

        # DO
        sel_query = FigURLCurationSelection & key
        sel_key = sel_query.fetch1()
        sorting_fpath = AnalysisNwbfile.get_abs_path(sorting_fname)
        recording = CurationV1.get_recording(sel_key)
        sorting = CurationV1.get_sorting(sel_key)
        sorting_label = sel_query.fetch1("sorting_id")
        curation_uri = sel_query.fetch1("curation_uri")

        metric_dict = {}
        with pynwb.NWBHDF5IO(sorting_fpath, "r", load_namespaces=True) as io:
            nwbf = io.read()
            nwb_sorting = nwbf.objects[object_id].to_dataframe()
            unit_ids = nwb_sorting.index
            for metric in metrics_figurl:
                metric_dict[metric] = dict(zip(unit_ids, nwb_sorting[metric]))

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
    def get_labels(cls, curation_json) -> Dict[int, List[str]]:
        """Uses kachery cloud to load curation json. Returns labelsByUnit."""

        labels_by_unit = kcl.load_json(curation_json).get("labelsByUnit")
        return (
            {
                int(unit_id): curation_label_list
                for unit_id, curation_label_list in labels_by_unit.items()
            }
            if labels_by_unit
            else {}
        )

    @classmethod
    def get_merge_groups(cls, curation_json) -> Dict:
        """Uses kachery cloud to load curation json. Returns mergeGroups."""
        return kcl.load_json(curation_json).get("mergeGroups", {})


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
) -> str:
    logger.info("Preparing spikesortingview data")
    recording = R
    sorting = S

    sampling_frequency = recording.get_sampling_frequency()

    this_view = SpikeSortingView.create(
        recording=recording,
        sorting=sorting,
        segment_duration_sec=segment_duration_sec,
        snippet_len=(
            int(snippet_ms_before * sampling_frequency / 1000),
            int(snippet_ms_after * sampling_frequency / 1000),
        ),
        max_num_snippets_per_segment=max_num_snippets_per_segment,
        channel_neighborhood_size=channel_neighborhood_size,
    )

    # Assemble the views in a layout. Can be replaced with other layouts.
    raster_max_fire = raster_plot_subsample_max_firing_rate
    spike_amp_max_fire = spike_amplitudes_subsample_max_firing_rate

    sort_items = [
        vv.MountainLayoutItem(
            label="Summary", view=this_view.sorting_summary_view()
        ),
        vv.MountainLayoutItem(
            label="Units table",
            view=this_view.units_table_view(
                unit_ids=this_view.unit_ids, unit_metrics=unit_metrics
            ),
        ),
        vv.MountainLayoutItem(
            label="Raster plot",
            view=this_view.raster_plot_view(
                unit_ids=this_view.unit_ids,
                _subsample_max_firing_rate=raster_max_fire,
            ),
        ),
        vv.MountainLayoutItem(
            label="Spike amplitudes",
            view=this_view.spike_amplitudes_view(
                unit_ids=this_view.unit_ids,
                _subsample_max_firing_rate=spike_amp_max_fire,
            ),
        ),
        vv.MountainLayoutItem(
            label="Autocorrelograms",
            view=this_view.autocorrelograms_view(unit_ids=this_view.unit_ids),
        ),
        vv.MountainLayoutItem(
            label="Cross correlograms",
            view=this_view.cross_correlograms_view(unit_ids=this_view.unit_ids),
        ),
        vv.MountainLayoutItem(
            label="Avg waveforms",
            view=this_view.average_waveforms_view(unit_ids=this_view.unit_ids),
        ),
        vv.MountainLayoutItem(
            label="Electrode geometry",
            view=this_view.electrode_geometry_view(),
        ),
        vv.MountainLayoutItem(
            label="Curation", view=vv.SortingCuration2(), is_control=True
        ),
    ]

    return vv.MountainLayout(items=sort_items).url(
        label=f"{recording_label} {sorting_label}",
        state={
            "initialSortingCuration": initial_curation_uri,
            "sortingCuration": initial_curation_uri,
        },
    )
