import datajoint as dj
import numpy as np
import sortingview as sv
import spikeinterface as si

from .common_spikesorting import (SpikeSortingRecordingSelection, SpikeSortingRecording,
                                  SpikeSorting, SortingList)
from .common_lab import LabTeam
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
        -> master
        sorting_id: varchar(30)
        ---
        sortingview_sorting_id: varchar(30)
        """

    def make(self, key: dict):
        # Load recording, wrap it as old spikeextractors recording extractor, 
        # then save as h5 (sortingview requires this format)
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
        
    def add_sorting_to_workspace(self, key: dict, sorting_id: str):
        """Add a sorting (defined by sorting_id) to the sortingview workspace (defined by key).
        
        Parameters
        ----------
        key : dict
        sorting_id : str
            sorting id given during spike sorting, NOT the same as sortingview sorting_id
        
        Returns
        -------
        sortingview_sorting_id : str
            unique id given to each sorting by sortingview
        """
        sorting_path = (SortingList & {'sorting_id': sorting_id}).fetch1('sorting_path')
        sorting = si.load_extractor(sorting_path)
        # convert to old sorting extractor
        sorting = si.create_extractor_from_new_sorting(sorting)
        h5_sorting = sv.LabboxEphysSortingExtractor.store_sorting_link_h5(sorting, sorting_path)
        workspace_uri = (self & key).fetch1('workspace_uri') 
        workspace = sv.load_workspace(workspace_uri)
        sorting_key = (SpikeSorting & {'sorting_id':sorting_id}).fetch1()
        sorting_name = SpikeSorting()._get_sorting_name(sorting_key)
        sortingview_sorting_id = workspace.add_sorting(recording_id=workspace.recording_ids[0], 
                                                       sorting=h5_sorting,
                                                       label=sorting_name)
        key['sorting_id'] = sorting_id
        key['sortingview_sorting_id'] = sortingview_sorting_id
        self.SortingID.insert1(key)
        
        return sortingview_sorting_id
    
    def add_metrics_to_sorting(self, key: dict, metrics: dict,
                               sortingview_sorting_id: str=None):
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
            print('sortingview sorting ID not specified, using the first sorting in the workspace...')
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

    def url(self, key, sortingview_sorting_id: str=None):
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
        url = workspace.experimental_spikesortingview(recording_id=recording_id, 
                                                      sorting_id=sortingview_sorting_id,
                                                      label=workspace.label,
                                                      include_curation=True)
        return url
    
    # TODO: check
    def precalculate(self, key):
        """For each workspace specified by the key, this will run the snipped precalculation code
        """
        workspace_uri_list = (self & key).fetch('workspace_uri')
        #TODO: consider running in parallel
        for workspace_uri in workspace_uri_list:
            try:
                workspace = sv.load_workspace(workspace_uri)
                workspace.precalculate()
            except:
                Warning(f'Error precomputing for workspace {workspace_uri}')
