"Functions that are commonly used"

import os
import tempfile
from pathlib import Path
from typing import List

import kachery_client as kc
import numpy as np
import sortingview as sv
import spikeextractors
import spikeinterface as si
import spikeinterface.toolkit as st
from spikeextractors import MdaSortingExtractor, NumpySortingExtractor
from spikesorters.basesorter import BaseSorter
from spikesorters.sorter_tools import recover_recording
from spiketoolkit.sortingcomponents import detect_spikes

from .common_interval import IntervalList
from .common_lab import LabMember
from .common_nwbfile import AnalysisNwbfile


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
    from .common_spikesorting import SpikeSortingRecording

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
                                     ms_before=1,ms_after=1, max_spikes_per_unit=2000,
                                     overwrite=True,return_scaled=True, n_jobs=5, total_memory='5G')

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
    
    return url, external_metricsclass ClusterlessThresholder(BaseSorter):
    sorter_name = 'clusterless_thresholder'  # convenience for reporting
    SortingExtractor_Class = NumpySortingExtractor  # convenience to get the extractor
    compatible_with_parallel = {'loky': False,
                                'multiprocessing': False,
                                'threading': False}
    _default_params = dict(
        detect_threshold=100,
        channel_ids=None,
        detect_sign=-1,
        n_shifts=2,
        n_snippets_for_threshold=10,
        snippet_size_sec=1,
        start_frame=None,
        end_frame=None,
        n_jobs=1,
        joblib_backend='loky',
        chunk_size=None,
        chunk_mb=500,
        verbose=False,
    )
    _params_description = {}
    sorter_description = ""
    requires_locations = False
    installation_mesg = ""  # error message when not installed

    @classmethod
    def is_installed(cls):
        return True

    @staticmethod
    def get_sorter_version():
        return 'unknown'

    @classmethod
    def _setup_recording(cls, recording, output_folder):
        pass

    def _run(self, recording, output_folder):

        recording = recover_recording(recording)
        params = self.params.copy()
        threshold = params.pop('detect_threshold')  # in mV

        # convert to median absolute deviation units
        MAD_threshold = np.max(
            threshold / np.median(np.abs(recording.get_traces()) / 0.6745, 1))

        sorting = detect_spikes(
            recording,
            detect_threshold=MAD_threshold,
            **params
        )

        output_folder = Path(output_folder)
        MdaSortingExtractor.write_sorting(
            sorting, str(output_folder / 'firings.mda'))

        samplerate = recording.get_sampling_frequency()
        samplerate_fname = str(output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as file:
            file.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        tmpdir = output_folder

        result_fname = str(tmpdir / 'firings.mda')
        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'r') as file:
            samplerate = float(file.read())

        return MdaSortingExtractor(
            file_path=result_fname, sampling_frequency=samplerate)
