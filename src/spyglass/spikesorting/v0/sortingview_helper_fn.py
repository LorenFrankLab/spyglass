"Sortingview helper functions"

from typing import Any, List, Tuple, Union

import kachery_cloud as kcl
import sortingview as sv
import sortingview.views as vv
import spikeinterface as si
from sortingview.SpikeSortingView import SpikeSortingView

from spyglass.spikesorting.v0.merged_sorting_extractor import (
    MergedSortingExtractor,
)
from spyglass.utils import logger


def _create_spikesortingview_workspace(
    recording_path: str,
    sorting_path: str,
    merge_groups: List[List[int]],
    workspace_label: str,
    recording_label: str,
    sorting_label: str,
    metrics: dict = None,
    google_user_ids: List = None,
    curation_labels: dict = None,
    similarity_matrix: Union[List[List[float]], None] = None,
    raster_plot_subsample_max_firing_rate=50,
    spike_amplitudes_subsample_max_firing_rate=50,
):
    workspace = sv.create_workspace(label=workspace_label)

    recording = si.load_extractor(recording_path)
    if recording.get_num_segments() > 1:
        recording = si.concatenate_recordings([recording])
    recording_id = workspace.add_recording(
        label=recording_label, recording=recording
    )

    sorting = si.load_extractor(sorting_path)
    if len(merge_groups) != 0:
        sorting = MergedSortingExtractor(
            parent_sorting=sorting, merge_groups=merge_groups
        )
    sorting_id = workspace.add_sorting(
        recording_id=recording_id, label=sorting_label, sorting=sorting
    )

    if google_user_ids is not None:
        workspace.create_curation_feed_for_sorting(sorting_id=sorting_id)
        workspace.set_sorting_curation_authorized_users(
            sorting_id=sorting_id, user_ids=google_user_ids
        )

    if curation_labels is not None:
        for unit_id, label in curation_labels.items():
            workspace.sorting_curation_add_label(
                sorting_id=sorting_id, label=label, unit_ids=[int(unit_id)]
            )

    return workspace.uri, recording_id, sorting_id


def _generate_url(
    *,
    recording: si.BaseRecording,
    sorting: si.BaseSorting,
    label: str,
    initial_curation: dict = {},
    raster_plot_subsample_max_firing_rate=50,
    spike_amplitudes_subsample_max_firing_rate=50,
    unit_metrics: Union[List[Any], None] = None,
) -> Tuple[str, str]:
    # moved figURL creation to function called trythis_URL in sosrtingview.py
    logger.info("Preparing spikesortingview data")
    X = SpikeSortingView.create(
        recording=recording,
        sorting=sorting,
        segment_duration_sec=60 * 20,
        snippet_len=(20, 20),
        max_num_snippets_per_segment=100,
        channel_neighborhood_size=7,
    )

    # Assemble the views in a layout
    # You can replace this with other layouts

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

    if initial_curation is not None:
        logger.warning("found initial curation")
        sorting_curation_uri = kcl.store_json(initial_curation)
    else:
        sorting_curation_uri = None
    url_state = (
        {"sortingCuration": sorting_curation_uri}
        if sorting_curation_uri is not None
        else None
    )
    url = view.url(label=label, state=url_state)

    logger.info(url)

    return url
