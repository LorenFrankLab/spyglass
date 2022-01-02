"Sortingview helper functions"

from typing import List
import numpy as np
import os
import kachery_client as kc
import spikeinterface as si
import spikeinterface.toolkit as st
import sortingview as sv
import spikeextractors
from .common_lab import LabMember
from .common_interval import IntervalList
from .common_nwbfile import AnalysisNwbfile
from .common_spikesorting import SpikeSortingRecording, SpikeSortingWorkspace, SortingID

from pathlib import Path
import tempfile

def store_sorting_nwb(key:dict, *, sorting, sort_interval_list_name:str, sort_interval,
                      metrics=None, unit_ids=None):
    """Store a sorting in a new AnalysisNwbfile

    Parameters
    ----------
    key : dict
        [description]
    sorting : [type]
        [description]
    sort_interval_list_name : str
        [description]
    sort_interval : list
        interval for start and end of sort
    metrics : dict, optional
        quality metrics, by default None
    unit_ids : list, optional
        [description], by default None

    Returns
    -------
    analysis_file_name : str
    units_object_id : str
    """

    sort_interval_valid_times = (IntervalList & \
            {'interval_list_name': sort_interval_list_name}).fetch1('valid_times')

    times = SpikeSortingRecording.get_recording_timestamps(key)
    # times = sorting.recording.get_times()
    units = dict()
    units_valid_times = dict()
    units_sort_interval = dict()
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    for unit_id in unit_ids:
        spike_times_in_samples = sorting.get_unit_spike_train(unit_id=unit_id)
        units[unit_id] = times[spike_times_in_samples]
        units_valid_times[unit_id] = sort_interval_valid_times
        units_sort_interval[unit_id] = [sort_interval]

    analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
    u = AnalysisNwbfile().add_units(analysis_file_name,
                                                     units, units_valid_times,
                                                     units_sort_interval,
                                                     metrics=metrics)
    if u=='':
        print('Sorting contains no units. Created an empty analysis nwb file anyway.')
        units_object_id = ''
    else:
        units_object_id = u[0]
    AnalysisNwbfile().add(key['nwb_file_name'], analysis_file_name)
    return analysis_file_name,  units_object_id

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

def add_metrics_to_workspace(workspace_uri: str, sorting_id: str=None,
                             user_ids: List[str]=None):
    """Computes nearest neighbor isolation and noise overlap metrics and inserts them
    in the specified sorting of the workspace.

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

    # first cache as binary using old spikeextractors to make it compatible with new si
    tmpdir = tempfile.TemporaryDirectory(dir=os.environ['KACHERY_TEMP_DIR'])
    cached_recording = spikeextractors.CacheRecordingExtractor(recording, 
                                                               save_path=str(Path(tmpdir.name) / 'r.dat'))

    # then load with spikeinterface
    new_recording = si.read_binary(file_paths=str(Path(tmpdir.name) / 'r.dat'),
                                   sampling_frequency=cached_recording.get_sampling_frequency(),
                                   num_chan=cached_recording.get_num_channels(),
                                   dtype=cached_recording.get_dtype(),
                                   time_axis=0,
                                   is_filtered=True)
    new_recording.set_channel_locations(locations=recording.get_channel_locations())
    new_recording = st.preprocessing.whiten(recording=new_recording, seed=0)
    
    if sorting_id is None:
        sorting_id = workspace.sorting_ids[0]
    sorting = workspace.get_sorting_extractor(sorting_id=sorting_id)
    new_sorting = si.core.create_sorting_from_old_extractor(sorting)
    # save sorting to enable parallelized waveform extraction
    new_sorting = new_sorting.save(folder=str(Path(tmpdir.name) / 's'),)

    waveforms = si.extract_waveforms(new_recording, new_sorting, folder=str(Path(tmpdir.name) / 'wf'),
                                     ms_before=1, ms_after=1, max_spikes_per_unit=2000,
                                     overwrite=True, return_scaled=True, n_jobs=5, total_memory='5G')

    isolation = {}
    noise_overlap = {}
    for unit_id in sorting.get_unit_ids():
        isolation[unit_id] = st.qualitymetrics.pca_metrics.nearest_neighbors_isolation(waveforms, this_unit_id=unit_id)
        noise_overlap[unit_id] = st.qualitymetrics.pca_metrics.nearest_neighbors_noise_overlap(waveforms, this_unit_id=unit_id)
    
    # external metrics must be in this format to be added to workspace
    external_metrics = [{'name': 'isolation', 'label': 'isolation', 'tooltip': 'isolation',
                        'data': isolation},
                        {'name': 'noise_overlap', 'label': 'noise_overlap', 'tooltip': 'noise_overlap',
                        'data': noise_overlap}
                        ] 
    workspace.set_unit_metrics_for_sorting(sorting_id=sorting_id, metrics=external_metrics)

    if user_ids is not None:
        for user_id in user_ids:
            workspace.set_user_permissions(user_id, {'edit': True})

    F = workspace.figurl()
    url = F.url(label=workspace.label, channel='franklab2')

    # in the future (once new view works fully):
    # workspace.set_sorting_curation_authorized_users(sorting_id=sorting_id, user_ids=user_ids)
    # url = workspace.experimental_spikesortingview(recording_id=recording_id, sorting_id=sorting_id,
    #                                               label=workspace.label, include_curation=True)

    print(f'URL for sortingview: {url}')
    
    return url, external_metrics


