"Functions that are commonly used"

from typing import List
import numpy as np
import os
import kachery_client as kc
import sortingview as sv
from .common_lab import LabMember
from .common_interval import IntervalList
from .common_nwbfile import AnalysisNwbfile
from .common_spikesorting import SpikeSortingRecording, SpikeSortingWorkspace, SortingID

def store_sorting_nwb(key, *, sorting, sort_interval_list_name, sort_interval, metrics=None, unit_ids=None):
    """store a sorting in a new AnalysisNwbfile
    :param key: key to SpikeSortingRecoring
    :type key: dict
    :param sorting: sorting extractor
    :type sorting: BaseSortingExtractor
    :param sort_interval_list_name: name of interval list for sort
    :type sort_interval_list_name: str
    :param sort_interval: interval for start and end of sort
    :type sort_interval: list
    :param path_suffix: string to append to end of sorting extractor file name
    :type path_suffix: str
    :param metrics: spikesorting metrics, optional, defaults to None
    :type metrics: dict
    :param unit_ids: the ids of the units to save, optional, defaults to None
    :type list
    :returns: analysis_file_name, units_object_id
    :rtype (str, str)
    """

    sort_interval_valid_times = (IntervalList & \
            {'interval_list_name': sort_interval_list_name}).fetch1('valid_times')

    recording_timestamps = SpikeSortingRecording.get_recording_timestamps(key)
    units = dict()
    units_valid_times = dict()
    units_sort_interval = dict()
    all_unit_ids = sorting.get_unit_ids()
    if unit_ids is None:
        unit_ids = all_unit_ids
    for unit_id in all_unit_ids:
        if unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(
                unit_id=unit_id)
            units[unit_id] = recording_timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

    analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
    units_object_id, _ = AnalysisNwbfile().add_units(analysis_file_name,
                                                    units, units_valid_times,
                                                    units_sort_interval,
                                                    metrics=metrics)

    AnalysisNwbfile().add(key['nwb_file_name'], analysis_file_name)
    return analysis_file_name,  units_object_id


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