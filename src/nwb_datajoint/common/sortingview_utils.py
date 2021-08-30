"Functions that are commonly used"

from typing import List
import numpy as np
import kachery_client as kc
import sortingview
from .common_lab import LabMember

def add_to_sortingview_workspace(workspace_name: str, recording_label: str, sorting_label: str, 
                                recording : sortingview.LabboxEphysRecordingExtractor, 
                                sorting: sortingview.LabboxEphysRecordingExtractor, metrics=None):
    """
    Adds recording and sorting to a sortingview workspace
    
    Parameters
    ----------
    workspace_name: str
        name of workspace to appear in sortingview; default is encouraged to be the name of analysis NWB file
    recording_label: str
        label of recording to appear in sortingview
    sorting_label: str
        label of sorting to appear in sortingview
    recording: LabBoxEphysRecordingExtractor
         recording extractor
    sorting: LabBoxEphysSortingExtractor
         sorting extractor
    metrics: pandas.Dataframe 
        external metrics to add to sortingview workspace
    
    Outputs
    ------
        workspace_uri: str
            sortingview workspace uri containing recording and sorting
        sorting_id: str
            sorting_id of sorting that was added
    """               
    workspace_uri: str
    # get uri for workspace; if does not exist, create new one
    workspace_uri = kc.get(workspace_name)
    if not workspace_uri:
        workspace_uri = sortingview.create_workspace(label=workspace_name).uri
        kc.set(workspace_name, workspace_uri)
    workspace = sortingview.load_workspace(workspace_uri)
    print(f'Workspace URI: {workspace.uri}')
    # add the recording and sorting info to kachery
    # recording_uri = kc.store_json(recording_dict)
    # sorting_uri = kc.store_json(sorting_dict)
    # load sorting and recording
    # sorting = sortingview.LabboxEphysSortingExtractor(sorting_uri)
    # recording = sortingview.LabboxEphysRecordingExtractor(recording_uri, download=True)
    # add to workspace
    R_id = workspace.add_recording(recording=recording, label=recording_label)
    S_id = workspace.add_sorting(sorting=sorting, recording_id=R_id, label=sorting_label)
    # Set external metrics that will appear in the units table
    if metrics is not None:
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
        workspace.set_unit_metrics_for_sorting(sorting_id=S_id, metrics=external_metrics)
    # add workspace to sortingview
    workspace_list = sortingview.WorkspaceList(list_name='default')
    workspace_list.add_workspace(name=workspace_name, workspace=workspace)
    print('Workspace added to sortingview')
    print(f'To curate the spike sorting, go to https://sortingview.vercel.app/workspace?workspace={workspace.uri}&channel=franklab')
    
    return workspace.uri, S_id

def set_workspace_permission(workspace_name: str, team_members: List[str]):
    """
    Sets permission to sortingview workspace based on google ID
    
    Parameters
    ----------
    workspace_name: str
        name of workspace
    team_members: List[str]
        list of team members to be given permission to edit the workspace
    
    Output
    ------
    workspace_uri: str
        URI of workspace
    """
    workspace_uri = kc.get(workspace_name)
    # check if workspace exists
    if not workspace_uri:
        raise ValueError('Workspace with given name does not exist. Create it first before setting permission.')
    workspace = sortingview.load_workspace(workspace_uri)
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