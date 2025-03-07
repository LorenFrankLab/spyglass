from typing import Any, List, Union

import datajoint as dj
import kachery_cloud as kcl
import sortingview.views as vv
import spikeinterface as si
from sortingview.SpikeSortingView import SpikeSortingView

from spyglass.spikesorting.utils import _reformat_metrics
from spyglass.spikesorting.v0.spikesorting_curation import Curation
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)
from spyglass.spikesorting.v0.spikesorting_sorting import SpikeSorting
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_curation_figurl")

# A curation figURL is a link to a visualization of a curation.
# Optionally you can specify a new_curation_uri which will be
# the location of the new manually-edited curation. The
# new_curation_uri should be a github uri of the form
# gh://user/repo/branch/path/to/curation.json
# and ideally the path should be determined by the primary key
# of the curation. The new_curation_uri can also be blank if no
# further manual curation is planned.


@schema
class CurationFigurlSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Curation
    ---
    new_curation_uri: varchar(2000)
    """


@schema
class CurationFigurl(SpyglassMixin, dj.Computed):
    definition = """
    -> CurationFigurlSelection
    ---
    url: varchar(2000)
    initial_curation_uri: varchar(2000)
    new_curation_uri: varchar(2000)
    """

    def make(self, key: dict):
        """Create a Curation Figurl
        Parameters
        ----------
        key : dict
            primary key of an entry from CurationFigurlSelection table
        """

        # get new_curation_uri from selection table
        new_curation_uri = (CurationFigurlSelection & key).fetch1(
            "new_curation_uri"
        )

        # fetch
        sorting_path = (SpikeSorting & key).fetch1("sorting_path")
        recording_label = SpikeSortingRecording._get_recording_name(key)
        sorting_label = SpikeSorting._get_sorting_name(key)
        unit_metrics = _reformat_metrics(
            (Curation & key).fetch1("quality_metrics")
        )
        initial_labels = (Curation & key).fetch1("curation_labels")
        initial_merge_groups = (Curation & key).fetch1("merge_groups")

        # new_curation_uri = key["new_curation_uri"]

        # Create the initial curation and store it in kachery
        for k, v in initial_labels.items():
            new_list = []
            for item in v:
                if item not in new_list:
                    new_list.append(item)
            initial_labels[k] = new_list
        initial_curation = {
            "labelsByUnit": initial_labels,
            "mergeGroups": initial_merge_groups,
        }
        initial_curation_uri = kcl.store_json(initial_curation)

        # Get the recording/sorting extractors
        R = SpikeSortingRecording().load_recording(key)
        if R.get_num_segments() > 1:
            R = si.concatenate_recordings([R])
        S = si.load_extractor(sorting_path)

        # Generate the figURL
        url = _generate_the_figurl(
            R=R,
            S=S,
            initial_curation_uri=initial_curation_uri,
            new_curation_uri=new_curation_uri,
            recording_label=recording_label,
            sorting_label=sorting_label,
            unit_metrics=unit_metrics,
        )

        # insert
        self.insert1(
            dict(
                key,
                url=url,
                initial_curation_uri=initial_curation_uri,
                new_curation_uri=new_curation_uri,
            )
        )


def _generate_the_figurl(
    *,
    R: si.BaseRecording,
    S: si.BaseSorting,
    unit_metrics: Union[List[Any], None] = None,
    initial_curation_uri: str,
    recording_label: str,
    sorting_label: str,
    new_curation_uri: str,
):
    logger.info("Preparing spikesortingview data")
    X = SpikeSortingView.create(
        recording=R,
        sorting=S,
        segment_duration_sec=60 * 20,
        snippet_len=(20, 20),
        max_num_snippets_per_segment=100,
        channel_neighborhood_size=7,
    )

    # Assemble the views in a layout
    # You can replace this with other layouts
    raster_plot_subsample_max_firing_rate = 50
    spike_amplitudes_subsample_max_firing_rate = 50
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
    url_state = (
        {
            "initialSortingCuration": initial_curation_uri,
            "sortingCuration": new_curation_uri,
        }
        if new_curation_uri
        else {"sortingCuration": initial_curation_uri}
    )
    label = f"{recording_label} {sorting_label}"
    url = view.url(label=label, state=url_state)
    return url
