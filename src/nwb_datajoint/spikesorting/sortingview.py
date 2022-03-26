from pathlib import Path

import datajoint as dj
import numpy as np
import sortingview as sv
import spikeinterface as si

from ..common.common_lab import LabTeam
from .sortingview_helper_fn import set_workspace_permission
from .spikesorting_curation import Curation
from .spikesorting_recording import SpikeSortingRecording

schema = dj.schema('spikesorting_sortingview')


@schema
class SortingviewWorkspaceSelection(dj.Manual):
    definition = """
    -> Curation
    """


@schema
class SortingviewWorkspace(dj.Computed):
    definition = """
    -> SortingviewWorkspaceSelection
    ---
    workspace_uri: varchar(1000)
    sortingview_recording_id: varchar(30)
    sortingview_sorting_id: varchar(30)
    channel = 'franklab2' : varchar(80) # the name of the kachery channel for data sharing
    """

    def make(self, key: dict):
        # Load recording, wrap it as old spikeextractors recording extractor,
        # then save as h5 (sortingview requires this format)
        # import Curation here to avoid circular import
        recording = Curation.get_recording_extractor(key)
        old_recording = si.create_extractor_from_new_recording(recording)

        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')

        h5_recording = sv.LabboxEphysRecordingExtractor.store_recording_link_h5(old_recording,
                                                                                str(Path(
                                                                                    recording_path) / 'recording.h5'),
                                                                                dtype='int16')

        workspace_name = SpikeSortingRecording._get_recording_name(key)
        workspace = sv.create_workspace(label=workspace_name)
        key['workspace_uri'] = workspace.uri
        key['sortingview_recording_id'] = workspace.add_recording(recording=h5_recording,
                                                                  label=workspace_name)
        sorting = Curation.get_curated_sorting_extractor(key)
        sorting = si.create_extractor_from_new_sorting(sorting)
        h5_sorting = sv.LabboxEphysSortingExtractor.store_sorting_link_h5(
            sorting, str(Path(recording_path) / 'sorting.h5'))

        key['sortingview_sorting_id'] = workspace.add_sorting(recording_id=workspace.recording_ids[0],
                                                              sorting=h5_sorting, label=key['curation_id'])
        self.insert1(key)

        # add metrics to the sorting if they exist
        metrics = (Curation & key).fetch1('quality_metrics')
        self.add_metrics_to_sorting(
            key, metrics, key['sortingview_sorting_id'])

        labels = (Curation & key).fetch1('curation_labels')
        print(labels)
        for unit_id in labels:
            for label in labels[unit_id]:
                workspace.sorting_curation_add_label(
                    sorting_id=key['sortingview_sorting_id'], label=label, unit_ids=[int(unit_id)])

        # set the permissions
        team_name = (SpikeSortingRecording & key).fetch1()['team_name']
        team_members = (LabTeam.LabTeamMember & {
                        'team_name': team_name}).fetch('lab_member_name')
        set_workspace_permission(
            workspace.uri, team_members, key['sortingview_sorting_id'])

    def remove_sorting_from_workspace(self, key):
        return NotImplementedError

    def add_metrics_to_sorting(self, key: dict, metrics: dict,
                               sortingview_sorting_id: str = None):
        """Adds a metrics to the specified sorting.

        Parameters
        ----------
        key : dict
        metrics : dict
            Quality metrics.
            Key: name of quality metric
            Value: another dict in which key: unit ID (must be str),
                                         value: metric value (float)
        sortingview_sorting_id : str, optional
            if not specified, just uses the first sorting ID of the workspace
        """

        # check that the unit ids are str
        for metric_name in metrics:
            unit_ids = metrics[metric_name].keys()
            assert np.all([isinstance(unit_id, int)] for unit_id in unit_ids)

        # the metrics must be in this form to be added to sortingview
        external_metrics = [{'name': metric_name,
                             'label': metric_name,
                             'tooltip': metric_name,
                             'data': metric} for metric_name, metric in metrics.items()]

        workspace_uri = (self & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri)
        if sortingview_sorting_id is None:
            print(
                'sortingview sorting ID not specified, using the first sorting in the workspace...')
            sortingview_sorting_id = workspace.sorting_ids[0]

        workspace.set_unit_metrics_for_sorting(sorting_id=sortingview_sorting_id,
                                               metrics=external_metrics)

    def set_snippet_len(self, key: dict, snippet_len: int):
        """Sets the snippet length of a workspace specified by the key

        Parameters
        ----------
        key : dict
            defines a workspace
        """
        workspace_uri = (self & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri)
        workspace.set_snippet_len(snippet_len)

    def url(self, key, sortingview_sorting_id: str = None):
        """Generate a URL for visualizing the sorting on the browser

        Parameters
        ----------
        key : dict
            defines a workspace
        sortingview_sorting_id : str, optional
            sortingview sorting ID to visualize; if None then chooses the first one

        Returns
        -------
        url : str
        """

        workspace_uri = (self & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri)
        recording_id = workspace.recording_ids[0]
        if sortingview_sorting_id is None:
            sortingview_sorting_id = workspace.sorting_ids[0]
        url = workspace.spikesortingview(recording_id=recording_id,
                                         sorting_id=sortingview_sorting_id,
                                         label=workspace.label,
                                         include_curation=True)
        return url

    @staticmethod
    def insert_manual_curation(key: dict, description=''):
        """ Based on information in key for an SortingviewWorkspace, loads the curated sorting from sortingview,
        saves it (with labels and the optional description) and inserts it to CuratedSorting

        Assumes that the workspace corresponding to the recording and (original) sorting exists

        Parameters
        ----------
        key : dict
            primary key of AutomaticCuration
        description: str
            optional description of curated sort
        """
        print(key)
        workspace_uri = (SortingviewWorkspace & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri=workspace_uri)

        sortingview_sorting_id = (SortingviewWorkspace & key).fetch1(
            'sortingview_sorting_id')

        # get the labels and remove the non-primary merged units
        labels = workspace.get_sorting_curation(
            sorting_id=sortingview_sorting_id)
        # turn labels to list of str, only including accepted units.

        unit_labels = labels['labelsByUnit']

        if bool(labels['mergeGroups']):
            # clusters were merged, so we empty out metrics
            metrics = {}
        else:
            # get the metrics from the parent curation
            metrics = (Curation & key).fetch1('metrics')

        # insert this curation into the  Table
        return Curation.insert_curation(
            key,
            parent_curation_id=key['curation_id'],
            labels=labels['labelsByUnit'],
            merge_groups=labels['mergeGroups'],
            metrics=metrics,
            description='manually curated')
