"Sortingview helper functions"

from typing import Dict, List, Union

import spikeinterface as si

import sortingview as sv
import sortingview.views as vv
from sortingview.SpikeSortingView import SpikeSortingView

from .merged_sorting_extractor import MergedSortingExtractor


def _set_workspace_permission(
    workspace: sv.Workspace,
    google_user_ids: List[str],
    sortingview_sorting_id: str = None,
):
    """Set permission to curate specified sorting on sortingview workspace based
    on google ID

    Parameters
    ----------
    workspace_uri : sv.Workspace
    team_members : List[str]
        list of team members to be given permission to edit the workspace
    sortingview_sorting_id : str
    """
    if sortingview_sorting_id is None:
        sortingview_sorting_id = workspace.sorting_ids[0]
    workspace.set_sorting_curation_authorized_users(
        sorting_id=sortingview_sorting_id, user_ids=google_user_ids
    )
    print(
        f"Permissions to curate sorting {sortingview_sorting_id} given to {google_user_ids}."
    )
    return workspace


def _add_metrics_to_sorting_in_workspace(
    workspace: sv.Workspace,
    metrics: Dict[str, Dict[str, float]],
    sortingview_sorting_id: str = None,
):
    """Adds a metrics to the specified sorting.

    Parameters
    ----------
    workspace : sv.Workspace
    metrics : dict
        Quality metrics.
        Key: name of quality metric
        Value: another dict in which key: unit ID (str),
                                     value: metric value (float)
    sortingview_sorting_id : str, optional
        if not specified, just uses the first sorting ID of the workspace

    Returns
    -------
    workspace : sv.Workspace
        Workspace after adding the metrics to the sorting
    """

    # make sure the unit IDs are str
    for metric_name in metrics:
        metrics[metric_name] = {
            str(unit_id): metric_value
            for unit_id, metric_value in metrics[metric_name].items()
        }

    # the metrics must be in this form (List[Dict]) to be added to sortingview
    external_metrics = [
        {
            "name": metric_name,
            "label": metric_name,
            "tooltip": metric_name,
            "data": metric,
        }
        for metric_name, metric in metrics.items()
    ]

    if sortingview_sorting_id is None:
        print(
            "sortingview sorting ID not specified, adding metrics to the first"
            "sorting in the workspace..."
        )
        sortingview_sorting_id = workspace.sorting_ids[0]

    workspace.set_unit_metrics_for_sorting(
        sorting_id=sortingview_sorting_id, metrics=external_metrics
    )

    return workspace


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
    recording_id = workspace.add_recording(label=recording_label, recording=recording)

    sorting = si.load_extractor(sorting_path)
    if len(merge_groups) != 0:
        sorting = MergedSortingExtractor(
            parent_sorting=sorting, merge_groups=merge_groups
        )
    sorting_id = workspace.add_sorting(
        recording_id=recording_id, label=sorting_label, sorting=sorting
    )

    if metrics is not None:
        workspace = _add_metrics_to_sorting_in_workspace(
            workspace=workspace, metrics=metrics, sortingview_sorting_id=sorting_id
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

    unit_metrics = workspace.get_unit_metrics_for_sorting(sorting_id)

    print("Preparing spikesortingview data")
    X = SpikeSortingView.create(
        recording=recording,
        sorting=sorting,
        segment_duration_sec=60 * 20,
        snippet_len=(20, 20),
        max_num_snippets_per_segment=100,
        channel_neighborhood_size=7,
    )

    # create a fake unit similiarity matrix
    similarity_scores = []
    # for u1 in X.unit_ids:
    #     for u2 in X.unit_ids:
    #         similarity_scores.append(
    #             vv.UnitSimilarityScore(
    #                 unit_id1=u1,
    #                 unit_id2=u2,
    #                 similarity=similarity_matrix[(X.unit_ids==u1),(X.unit_ids==u2)]
    #             )
    #         )
    # Create the similarity matrix view
    # unit_similarity_matrix_view = vv.UnitSimilarityMatrix(
    #    unit_ids=X.unit_ids,
    #    similarity_scores=similarity_scores
    #    )

    # Assemble the views in a layout
    # You can replace this with other layouts
    view = vv.MountainLayout(
        items=[
            vv.MountainLayoutItem(label="Summary", view=X.sorting_summary_view()),
            vv.MountainLayoutItem(
                label="Units table",
                view=X.units_table_view(unit_ids=X.unit_ids, unit_metrics=unit_metrics),
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
                label="Curation", view=vv.SortingCuration(), is_control=True
            ),
        ]
    )

    sorting_curation_uri = workspace.get_sorting_curation_uri(sorting_id)
    url = view.url(label=recording_label, sorting_curation_uri=sorting_curation_uri)

    print(f"figurl: {url}")

    return workspace.uri, recording_id, sorting_id
