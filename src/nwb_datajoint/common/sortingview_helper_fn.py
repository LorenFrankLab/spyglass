"Sortingview helper functions"

from typing import List
import os
import kachery_client as kc
import spikeinterface as si
import spikeinterface.toolkit as st
import sortingview as sv
from .common_lab import LabMember

import tempfile

def set_workspace_permission(workspace_name: str, team_members: List[str]):
    """Sets permission to sortingview workspace based on google ID
    
    Parameters
    ----------
    workspace_name : str
        name of workspace
    team_members : List[str]
        list of team members to be given permission to edit the workspace
    
    Returns
    ------
    workspace_uri : str
        URI of workspace
    """
    workspace_uri = kc.get(workspace_name)
    # check if workspace exists
    if not workspace_uri:
        raise ValueError('Workspace with given name does not exist. Create it first before setting permission.')
    workspace = sv.load_workspace(workspace_uri)
    # check if team_members is empty
    if len(team_members)==0:
        raise ValueError('The specified team does not exist or there are no members in the team;\
                            create or change the entry in LabTeam table first')
    for team_member in team_members:
        google_user_id = (LabMember.LabMemberInfo & {'lab_member_name': team_member}).fetch('google_user_name')  
        if len(google_user_id)!=1:
            print(f'Google user ID for {team_member} does not exist or more than one ID detected;\
                    permission not given to {team_member}, skipping...')
            continue          
        workspace.set_user_permissions(google_user_id[0], {'edit': True})
        print(f'Permissions for {google_user_id[0]} set to: {workspace.get_user_permissions(google_user_id[0])}')
    return workspace_uri

def add_sorting_to_workspace(workspace_uri: str, sorting):
    """Adds a sorting to an existing sortingview workspace

    Parameters
    ----------
    workspace_uri : str
        workspace URI
    sorting : si.Sorting object
        
    Returns
    -------
    sorting_id : str
        sorting ID associated with 
    external_metrics :  dict
        metrics that were added to workspace
    """

    workspace = sv.load_workspace(workspace_uri)
    
    recording_id = workspace.recording_ids[0]
    recording = workspace.get_recording_extractor(recording_id=recording_id)

    new_recording = si.create_recording_from_old_extractor(recording)
    new_recording.annotate(is_filtered=True)
    new_recording = st.preprocessing.whiten(recording=new_recording, seed=0)
    new_recording.is_dumpable=False 
    
    if sorting_id is None:
        sorting_id = workspace.sorting_ids[0]
    sorting = workspace.get_sorting_extractor(sorting_id=sorting_id)
    new_sorting = si.create_sorting_from_old_extractor(sorting)

    tmpdir = tempfile.TemporaryDirectory(dir=os.environ['KACHERY_TEMP_DIR'])
    waveforms = si.extract_waveforms(new_recording, new_sorting, folder=tmpdir.name,
                                     ms_before=1, ms_after=1, max_spikes_per_unit=2000,
                                     overwrite=True, return_scaled=True, n_jobs=5, total_memory='5G')

    isolation = {}
    noise_overlap = {}
    for unit_id in sorting.get_unit_ids():
        isolation[str(unit_id)] = st.qualitymetrics.pca_metrics.nearest_neighbors_isolation(waveforms, this_unit_id=unit_id)
        noise_overlap[str(unit_id)] = st.qualitymetrics.pca_metrics.nearest_neighbors_noise_overlap(waveforms, this_unit_id=unit_id)
    
    # external metrics must be in this format to be added to workspace
    external_metrics = [{'name': 'isolation', 'label': 'isolation', 'tooltip': 'isolation',
                        'data': isolation},
                        {'name': 'noise_overlap', 'label': 'noise_overlap', 'tooltip': 'noise_overlap',
                        'data': noise_overlap}]
    workspace.set_unit_metrics_for_sorting(sorting_id=sorting_id, metrics=external_metrics)
    
    if user_ids is not None:
        workspace.set_sorting_curation_authorized_users(sorting_id=sorting_id, user_ids=user_ids)
    url = workspace.experimental_spikesortingview(recording_id=recording_id, sorting_id=sorting_id,
                                                  label=workspace.label, include_curation=True)

    print(f'URL for sortingview: {url}')
    
    return url, external_metrics

def add_metrics_to_workspace(workspace_uri: str, sorting_id: str=None,
                             user_ids: List[str]=None):
    """Computes nearest neighbor isolation and noise overlap metrics and inserts them
    in the specified sorting of the workspace. This bypasses the spyglass pipeline and
    should be used only for spike sortings done in the old nwb_datajoint pipeline.

    Parameters
    ----------
    workspace_uri : str
        workspace URI
    sorting_id : str, optional
        sorting id, by default None
    user_ids : list of str, optional
        google ids to confer curation permission, by default None
        
    Returns
    -------
    url : str
        new URL for the view
    external_metrics :  dict
        metrics that were added to workspace
    """

    workspace = sv.load_workspace(workspace_uri)
    
    recording_id = workspace.recording_ids[0]
    recording = workspace.get_recording_extractor(recording_id=recording_id)

    new_recording = si.create_recording_from_old_extractor(recording)
    new_recording.annotate(is_filtered=True)
    new_recording = st.preprocessing.whiten(recording=new_recording, seed=0)
    new_recording.is_dumpable=False 
    
    if sorting_id is None:
        sorting_id = workspace.sorting_ids[0]
    sorting = workspace.get_sorting_extractor(sorting_id=sorting_id)
    new_sorting = si.create_sorting_from_old_extractor(sorting)

    tmpdir = tempfile.TemporaryDirectory(dir=os.environ['KACHERY_TEMP_DIR'])
    waveforms = si.extract_waveforms(new_recording, new_sorting, folder=tmpdir.name,
                                     ms_before=1, ms_after=1, max_spikes_per_unit=2000,
                                     overwrite=True, return_scaled=True, n_jobs=5, total_memory='5G')

    isolation = {}
    noise_overlap = {}
    for unit_id in sorting.get_unit_ids():
        isolation[str(unit_id)] = st.qualitymetrics.pca_metrics.nearest_neighbors_isolation(waveforms, this_unit_id=unit_id)
        noise_overlap[str(unit_id)] = st.qualitymetrics.pca_metrics.nearest_neighbors_noise_overlap(waveforms, this_unit_id=unit_id)
    
    # external metrics must be in this format to be added to workspace
    external_metrics = [{'name': 'isolation', 'label': 'isolation', 'tooltip': 'isolation',
                        'data': isolation},
                        {'name': 'noise_overlap', 'label': 'noise_overlap', 'tooltip': 'noise_overlap',
                        'data': noise_overlap}]
    workspace.set_unit_metrics_for_sorting(sorting_id=sorting_id, metrics=external_metrics)
    
    if user_ids is not None:
        workspace.set_sorting_curation_authorized_users(sorting_id=sorting_id, user_ids=user_ids)
    url = workspace.experimental_spikesortingview(recording_id=recording_id, sorting_id=sorting_id,
                                                  label=workspace.label, include_curation=True)

    print(f'URL for sortingview: {url}')
    
    return url, external_metrics