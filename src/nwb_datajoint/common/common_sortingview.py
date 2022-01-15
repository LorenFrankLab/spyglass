import os
from pathlib import Path

import datajoint as dj
import kachery_client as kc
import numpy as np
import pynwb
import sortingview as sv
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st

from .common_spikesorting import SpikeSortingRecordingSelection, SpikeSortingRecording
from .common_lab import LabTeam
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .sortingview_helper_fn import set_workspace_permission

schema = dj.schema('common_sortingview')

@schema
class SortingviewWorkspace(dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    channel = 'franklab2' : varchar(80) # the name of the kachery channel for data sharing
    sortingview_recording_id: varchar(30)
    workspace_uri: varchar(1000)
    """
    class SortingID(dj.Part):
        definition = """
        # Information about lab member in the context of Frank lab network
        -> master
        sorting_id: varchar(30)
        sortingview_sorting_id: varchar(30)
        """

    def make(self, key: dict):
        # load recording, wrap it as old spikeextractors recording, then save as h5 (sortingview requires this format)
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        recording = si.load_extractor(recording_path)
        old_recording = si.create_extractor_from_new_recording(recording)
        h5_recording = sv.LabboxEphysRecordingExtractor.store_recording_link_h5(old_recording, 
                                                                                recording_path,
                                                                                dtype='int16')

        workspace_name = SpikeSortingRecording()._get_recording_name(key)
        workspace = sv.create_workspace(label=workspace_name)
        key['workspace_uri'] = workspace.uri
        key['sortingview_recording_id'] = workspace.add_recording(recording=h5_recording,
                                                                  label=workspace_name)

        # Give permission to workspace based on Google account
        team_name = (SpikeSortingRecordingSelection & key).fetch1('team_name')
        team_members = (LabTeam.LabTeamMember & {'team_name': team_name}).fetch('lab_member_name')
        set_workspace_permission(workspace_name, team_members)    
        
        self.insert1(key)
        
    # TODO: finish
    def add_sorting_to_workspace(self, key: dict, sorting_id: str):
        """Add a sorting (defined by sorting_id) to the 
        sortingview workspace (defined by key).
        
        Parameters
        ----------
        key : dict
        sorting_id : spikeinterface Sorting object
            sorting id given during spike sorting, but the same as sortingview sorting_id
        
        Returns
        -------
        sortingview_sorting_id
        """
        parent_key = (SortingviewWorkspace & key).fetch1()
        workspace = sv.load_workspace(parent_key['workspace_uri'])
        parent_key['sorting_id'] = sorting_id
        # parent_key['sortingview_sorting_id'] = 
        old_recording = si.create_extractor_from_new_recording(recording)
        h5_recording = sv.LabboxEphysRecordingExtractor.store_recording_link_h5(old_recording, 
                                                                                recording_path,
                                                                                dtype='int16')
        SortingviewWorkspace.SortingID.insert1(key)
        # get the path from the Recording
        sorting_h5_path  = SpikeSortingRecording.get_sorting_extractor_save_path(key, path_suffix=path_suffix)
        if os.path.exists(sorting_h5_path):
            Warning(f'{sorting_h5_path} exists; overwriting')
        h5_sorting = sv.LabboxEphysSortingExtractor.store_sorting_link_h5(sorting, sorting_h5_path)
        s_key = (SpikeSortingRecording & key).fetch1("KEY")
        sorting_object = s_key['sorting_extractor_object'] = h5_sorting.object()
        
        # add the sorting to the workspace
        workspace_uri = (self & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri)
        # sorting = si.create_extractor_from_new_sorting(sorting)
        sorting = sv.LabboxEphysSortingExtractor(sorting_object)
        
        sorting_name = SpikeSorting()._get_sorting_name
        # note that we only ever have one recording per workspace
        sorting_id = s_key['sorting_id'] = workspace.add_sorting(recording_id=workspace.recording_ids[0], 
                                                                 sorting=sorting,
                                                                 label=sorting_name)

        return sorting_id
    
    # TODO: check
    def add_metrics_to_sorting(self, key: dict, *, sorting_id, metrics, workspace=None):
        """Adds a metrics to the specified sorting
        :param key: key for a Workspace
        :type key: dict
        :param sorting_id: the sorting_id
        :type sorting_id: str
        :param metrics: dictionary of computed metrics
        :type dict
        :param workspace: SortingView workspace (default None; this will be loaded if it is not specified)
        :type SortingView workspace object
        """
        # convert the metrics for storage in the workspace 
        external_metrics = [{'name': metric, 'label': metric, 'tooltip': metric,
                            'data': metrics[metric].to_dict()} for metric in metrics.columns]    
        # change unit id to string
        for metric_ind in range(len(external_metrics)):
            for old_unit_id in metrics.index:
                external_metrics[metric_ind]['data'][str(
                    old_unit_id)] = external_metrics[metric_ind]['data'].pop(old_unit_id)
                # change nan to none so that json can handle it
                if np.isnan(external_metrics[metric_ind]['data'][str(old_unit_id)]):
                    external_metrics[metric_ind]['data'][str(old_unit_id)] = None
        # load the workspace if necessary
        if workspace is None:
            workspace_uri = (self & key).fetch1('workspace_uri')
            workspace = sv.load_workspace(workspace_uri)    
        workspace.set_unit_metrics_for_sorting(sorting_id=sorting_id, 
                                               metrics=external_metrics)  
    
    # TODO: check
    def set_snippet_len(self, key, snippet_len):
        """Sets the snippet length based on the sorting parameters for the workspaces specified by the key

        :param key: SpikeSortingWorkspace key
        :type key: dict
        :param snippet_len: the number of points to display before and after the peak or trough
        :type snippet_len: tuple (int, int)
        """
        workspace_list = (self & key).fetch()
        if len(workspace_list) == 0:
            return 
        for ss_workspace in workspace_list:
            # load the workspace
            workspace = sv.load_workspace(ss_workspace['workspace_uri'])
            workspace.set_snippet_len(snippet_len)

    # TODO: check
    def url(self, key):
        """generate a single url or a list of urls for curation from the key

        :param key: key to select one or a set of workspaces
        :type key: dict
        :return: url or a list of urls
        :rtype: string or a list of strings
        """
        # generate a single website or a list of websites for the selected key
        workspace_list = (self & key).fetch()
        if len(workspace_list) == 0:
            return None
        url_list = []
        for ss_workspace in workspace_list:
            # load the workspace
            workspace = sv.load_workspace(ss_workspace['workspace_uri'])
            F = workspace.figurl()
            url_list.append(F.url(label=workspace.label, channel=ss_workspace['channel']))
        if len(url_list) == 1:
            return url_list[0]
        else:
            return url_list
    
    # TODO: check
    def precalculate(self, key):
        """For each workspace specified by the key, this will run the snipped precalculation code

        Args:
            key ([dict]): key to one or more SpikeSortingWorkspaces
        Returns:
            None
        """
        workspace_uri_list = (self & key).fetch('workspace_uri')
        #TODO: consider running in parallel
        for workspace_uri in workspace_uri_list:
            try:
                workspace = sv.load_workspace(workspace_uri)
                workspace.precalculate()
            except:
                Warning(f'Error precomputing for workspace {workspace_uri}')
