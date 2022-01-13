from copy import Error
import json
import os
import pathlib
import time
from pathlib import Path
import shutil
from functools import reduce
import tempfile

import spikeextractors
import datajoint as dj
import kachery_client as kc
import numpy as np
import pynwb
import scipy.stats as stats
import sortingview as sv
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st

from .common_device import Probe
from .common_lab import LabMember, LabTeam
from .common_ephys import Electrode, ElectrodeGroup, Raw
from .common_interval import (IntervalList, SortInterval,
                              interval_list_intersect, union_adjacent_index)
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .dj_helper_fn import dj_replace, fetch_nwb
from .nwb_helper_fn import get_valid_intervals
from .sortingview_helper_fn import set_workspace_permission

class Timer:
    """
    Timer context manager for measuring time taken by each sorting step
    """

    def __init__(self, *, label='', verbose=False):
        self._label = label
        self._start_time = None
        self._stop_time = None
        self._verbose = verbose

    def elapsed(self):
        if self._stop_time is None:
            return time.time() - self._start_time
        else:
            return self._stop_time - self._start_time

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stop_time = time.time()
        if self._verbose:
            print(f"Elapsed time for {self._label}: {self.elapsed()} sec")

schema = dj.schema('common_spikesorting')

@schema
class SortGroup(dj.Manual):
    definition = """
    # Set of electrodes that will be sorted together
    -> Session
    sort_group_id: int  # identifier for a group of electrodes
    ---
    sort_reference_electrode_id = -1: int  # the electrode to use for reference. -1: no reference, -2: common median
    """

    class SortGroupElectrode(dj.Part):
        definition = """
        -> master
        -> Electrode
        """

    def set_group_by_shank(self, nwb_file_name: str, references: dict=None):
        """Divides electrodes into groups based on their shank position.
        * Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a
          single group
        * Electrodes from probes with multiple shanks (e.g. polymer probes) are
          placed in one group per shank
        * Bad channels are omitted

        Parameters
        ----------
        nwb_file_name : str
            the name of the NWB file whose electrodes should be put into sorting groups
        references : dict, optional
            If passed, used to set references. Otherwise, references set using
            original reference electrodes from config. Keys: electrode groups. Values: reference electrode.
        """
        # delete any current groups
        (SortGroup & {'nwb_file_name': nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (Electrode() & {'nwb_file_name': nwb_file_name} & {
                      'bad_channel': 'False'}).fetch()
        e_groups = list(np.unique(electrodes['electrode_group_name']))
        e_groups.sort(key=int)  # sort electrode groups numerically
        sort_group = 0
        sg_key = dict()
        sge_key = dict()
        sg_key['nwb_file_name'] = sge_key['nwb_file_name'] = nwb_file_name
        for e_group in e_groups:
            # for each electrode group, get a list of the unique shank numbers
            shank_list = np.unique(
                electrodes['probe_shank'][electrodes['electrode_group_name'] == e_group])
            sge_key['electrode_group_name'] = e_group
            # get the indices of all electrodes in this group / shank and set their sorting group
            for shank in shank_list:
                sg_key['sort_group_id'] = sge_key['sort_group_id'] = sort_group
                # specify reference electrode. Use 'references' if passed, otherwise use reference from config
                if not references:
                    shank_elect_ref = electrodes['original_reference_electrode'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                                            electrodes['probe_shank'] == shank)]
                    if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                        sg_key['sort_reference_electrode_id'] = shank_elect_ref[0]
                    else:
                        ValueError(
                            f'Error in electrode group {e_group}: reference electrodes are not all the same')
                else:
                    if e_group not in references.keys():
                        raise Exception(f"electrode group {e_group} not a key in references, so cannot set reference")
                    else:
                        sg_key['sort_reference_electrode_id'] = references[e_group]
                self.insert1(sg_key)

                shank_elect = electrodes['electrode_id'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                        electrodes['probe_shank'] == shank)]
                for elect in shank_elect:
                    sge_key['electrode_id'] = elect
                    self.SortGroupElectrode().insert1(sge_key)
                sort_group += 1

    def set_group_by_electrode_group(self, nwb_file_name: str):
        """Assign groups to all non-bad channel electrodes based on their electrode group
        and sets the reference for each group to the reference for the first channel of the group.
        
        Parameters
        ----------
        nwb_file_name: str
            the name of the nwb whose electrodes should be put into sorting groups
        """
        # delete any current groups
        (SortGroup & {'nwb_file_name': nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (Electrode() & {'nwb_file_name': nwb_file_name} & {
                      'bad_channel': 'False'}).fetch()
        e_groups = np.unique(electrodes['electrode_group_name'])
        sg_key = dict()
        sge_key = dict()
        sg_key['nwb_file_name'] = sge_key['nwb_file_name'] = nwb_file_name
        sort_group = 0
        for e_group in e_groups:
            sge_key['electrode_group_name'] = e_group
            #sg_key['sort_group_id'] = sge_key['sort_group_id'] = sort_group
            #TEST
            sg_key['sort_group_id'] = sge_key['sort_group_id'] = int(e_group)
            # get the list of references and make sure they are all the same
            shank_elect_ref = electrodes['original_reference_electrode'][electrodes['electrode_group_name'] == e_group]
            if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                sg_key['sort_reference_electrode_id'] = shank_elect_ref[0]
            else:
                ValueError(
                    f'Error in electrode group {e_group}: reference electrodes are not all the same')
            self.insert1(sg_key)

            shank_elect = electrodes['electrode_id'][electrodes['electrode_group_name'] == e_group]
            for elect in shank_elect:
                sge_key['electrode_id'] = elect
                self.SortGroupElectrode().insert1(sge_key)
            sort_group += 1

    def set_reference_from_list(self, nwb_file_name, sort_group_ref_list):
        '''
        Set the reference electrode from a list containing sort groups and reference electrodes
        :param: sort_group_ref_list - 2D array or list where each row is [sort_group_id reference_electrode]
        :param: nwb_file_name - The name of the NWB file whose electrodes' references should be updated
        :return: Null
        '''
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        sort_group_list = (SortGroup() & key).fetch1()
        for sort_group in sort_group_list:
            key['sort_group_id'] = sort_group
            self.insert(dj_replace(sort_group_list, sort_group_ref_list,
                                    'sort_group_id', 'sort_reference_electrode_id'),
                                             replace="True")

    def get_geometry(self, sort_group_id, nwb_file_name):
        """
        Returns a list with the x,y coordinates of the electrodes in the sort group
        for use with the SpikeInterface package. Converts z locations to y where appropriate
        :param sort_group_id: the id of the sort group
        :param nwb_file_name: the name of the nwb file for the session you wish to use
        :param prb_file_name: the name of the output prb file
        :return: geometry: list of coordinate pairs, one per electrode
        """

        # create the channel_groups dictiorary
        channel_group = dict()
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        sort_group_list = (SortGroup() & key).fetch('sort_group_id')
        max_group = int(np.max(np.asarray(sort_group_list)))
        electrodes = (Electrode() & key).fetch()

        key['sort_group_id'] = sort_group_id
        sort_group_electrodes = (SortGroup.SortGroupElectrode() & key).fetch()
        electrode_group_name = sort_group_electrodes['electrode_group_name'][0]
        probe_type = (ElectrodeGroup & {'nwb_file_name': nwb_file_name,
                                        'electrode_group_name': electrode_group_name}).fetch1('probe_type')
        channel_group[sort_group_id] = dict()
        channel_group[sort_group_id]['channels'] = sort_group_electrodes['electrode_id'].tolist()

        label = list()
        n_chan = len(channel_group[sort_group_id]['channels'])

        geometry = np.zeros((n_chan, 2), dtype='float')
        tmp_geom = np.zeros((n_chan, 3), dtype='float')
        for i, electrode_id in enumerate(channel_group[sort_group_id]['channels']):
            # get the relative x and y locations of this channel from the probe table
            probe_electrode = int(
                electrodes['probe_electrode'][electrodes['electrode_id'] == electrode_id])
            rel_x, rel_y, rel_z = (Probe().Electrode() & {'probe_type': probe_type,
                                                          'probe_electrode': probe_electrode}).fetch('rel_x', 'rel_y', 'rel_z')
            # TODO: Fix this HACK when we can use probeinterface:
            rel_x = float(rel_x)
            rel_y = float(rel_y)
            rel_z = float(rel_z)
            tmp_geom[i, :] = [rel_x, rel_y, rel_z]

        # figure out which columns have coordinates
        n_found = 0
        for i in range(3):
            if np.any(np.nonzero(tmp_geom[:, i])):
                if n_found < 2:
                    geometry[:, n_found] = tmp_geom[:, i]
                    n_found += 1
                else:
                    Warning(
                        f'Relative electrode locations have three coordinates; only two are currenlty supported')
        return np.ndarray.tolist(geometry)

@schema
class SpikeSortingPreprocessingParameters(dj.Manual):
    definition = """
    preproc_params_name: varchar(200)
    ---
    preproc_params: blob 
    """
    def insert_default(self):
        # set up the default filter parameters
        freq_min = 300  # high pass filter value
        freq_max = 6000  # low pass filter value
        margin_ms = 5 # margin in ms on border to avoid border effect
        seed = 0 # random seed for whitening
        
        key = dict()
        key['preproc_params_name'] = 'default'
        key['preproc_params'] = {'frequency_min': freq_min,
                                 'frequency_max': freq_max,
                                 'margin_ms': margin_ms,
                                 'seed': seed}
        self.insert1(key, skip_duplicates=True)
    
@schema
class SpikeSortingArtifactDetectionParameters(dj.Manual):
    definition = """
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters for get_no_artifact_times() function
    """
    # TODO: use function calls to new spikeinterface (e.g. remove_artifact)
    def insert_default(self):
        """Insert the default artifact parameters with a appropriate parameter dict.
        """
        artifact_params = {}
        artifact_params['skip'] = True
        artifact_params['zscore_thresh'] = -1.0 
        artifact_params['amplitude_thresh'] = -1.0
        artifact_params['proportion_above_thresh'] = -1.0
        artifact_params['zero_window_len'] = 30 # 1 ms at 30 KHz, but this is of course skipped
        self.insert1(['none', artifact_params], skip_duplicates=True)

    # TODO: check this function
    def get_no_artifact_times(self, recording, zscore_thresh=-1.0, amplitude_thresh=-1.0,
                              proportion_above_thresh=1.0, zero_window_len=1.0, skip: bool=True):
        """returns an interval list of valid times, excluding detected artifacts found in data within recording extractor.
        Artifacts are defined as periods where the absolute amplitude of the signal exceeds one
        or both specified thresholds on the proportion of channels specified, with the period extended
        by the zero_window/2 samples on each side
        Threshold values <0 are ignored.

        :param recording: recording extractor
        :type recording: SpikeInterface recording extractor object
        :param zscore_thresh: Stdev threshold for exclusion, defaults to -1.0
        :type zscore_thresh: float, optional
        :param amplitude_thresh: Amplitude threshold for exclusion, defaults to -1.0
        :type amplitude_thresh: float, optional
        :param proportion_above_thresh:
        :type float, optional
        :param zero_window_len: the width of the window in milliseconds to zero out (window/2 on each side of threshold crossing)
        :type int, optional
        :return: [array of valid times]
        :type: [numpy array]
        """

        # if no thresholds were specified, we return an array with the timestamps of the first and last samples
        if zscore_thresh <= 0 and amplitude_thresh <= 0:
            return np.asarray([[recording._timestamps[0], recording._timestamps[recording.get_num_frames()-1]]])

        half_window_points = np.round(
            recording.get_sampling_frequency() * 1000 * zero_window_len / 2)
        nelect_above = np.round(proportion_above_thresh * data.shape[0])
        # get the data traces
        data = recording.get_traces()

        # compute the number of electrodes that have to be above threshold based on the number of rows of data
        nelect_above = np.round(
            proportion_above_thresh * len(recording.get_channel_ids()))

        # apply the amplitude threshold
        above_a = np.abs(data) > amplitude_thresh

        # zscore the data and get the absolute value for thresholding
        dataz = np.abs(stats.zscore(data, axis=1))
        above_z = dataz > zscore_thresh

        above_both = np.ravel(np.argwhere(
            np.sum(np.logical_and(above_z, above_a), axis=0) >= nelect_above))
        valid_timestamps = recording._timestamps
        # for each above threshold point, set the timestamps on either side of it to -1
        for a in above_both:
            valid_timestamps[a - half_window_points:a +
                             half_window_points] = -1

        # use get_valid_intervals to find all of the resulting valid times.
        return get_valid_intervals(valid_timestamps[valid_timestamps != -1], recording.get_sampling_frequency(), 1.5, 0.001)

@schema
class SpikeSortingRecordingSelection(dj.Manual):
    definition = """
    # Defines recordings to be sorted
    -> SortGroup
    -> SortInterval
    -> SpikeSortingPreprocessingParameters
    ---
    -> SpikeSortingArtifactDetectionParameters
    -> IntervalList
    -> LabTeam
    """

@schema
class SpikeSortingRecording(dj.Computed):
    definition = """
    -> SpikeSortingRecordingSelection
    ---
    recording_path: varchar(1000)
    """
    def make(self, key):
        sort_interval_valid_times = self._get_sort_interval_valid_times(key)
        recording = self._get_filtered_recording_extractor(key)

        # get the artifact detection parameters and apply artifact detection to zero out artifacts
        artifact_params_name = (SpikeSortingRecordingSelection & key).fetch1('artifact_params_name')
        artifact_params = (SpikeSortingArtifactDetectionParameters & 
                               {'artifact_params_name': artifact_params_name}).fetch1('artifact_params')
        
        # TODO: write function to get list_triggers
        # if not artifact_params['skip']:
            # recording = st.remove_artifacts(recording, list_triggers, **artifact_params)
        
        # find valid time for the sort interval and add to IntervalList
        recording_name = self._get_recording_name(key)

        tmp_key = {}
        tmp_key['nwb_file_name'] = key['nwb_file_name']
        tmp_key['interval_list_name'] = recording_name
        tmp_key['valid_times'] = sort_interval_valid_times;
        IntervalList.insert1(tmp_key, replace=True)

        # store the list of valid times for the sort
        key['sort_interval_list_name'] = tmp_key['interval_list_name']

        # Path to files that will hold the recording extractors
        recording_folder = Path(os.getenv('RECORDING_DIR'))
        key['recording_path'] = str(recording_folder / Path(recording_name))
        recording = recording.save(folder=key['recording_extractor_path'])

        # with Timer(label=f'writing to h5 recording extractor at {key["recording_extractor_path"]}',
        #            verbose=True):
        #     old_recording = si.create_extractor_from_new_recording(recording)
        #     h5_recording = sv.LabboxEphysRecordingExtractor.store_recording_link_h5(old_recording, 
        #                                          key["recording_extractor_path"], dtype='int16')
        self.insert1(key)
    
    def _get_recording_name(self, key):
        recording_name = key['nwb_file_name'] + '_' \
        + key['sort_interval_name'] + '_' \
        + str(key['sort_group_id']) + '_' \
        + key['preproc_params_name']
        return recording_name

    def _get_sort_interval_valid_times(self, key):
        """Identifies the intersection between sort interval specified by the user
        and the valid times (times for which neural data exist)

        Parameters
        ----------
        key: dict
            specifies a (partially filled) entry of SpikeSorting table

        Returns
        -------
        sort_interval_valid_times: ndarray of tuples
            (start, end) times for valid stretches of the sorting interval
        """
        sort_interval = (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                         'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        valid_interval_times = (IntervalList & {'nwb_file_name': key['nwb_file_name'],
                                                'interval_list_name': 'raw data valid times'}).fetch1('valid_times')
        valid_sort_times = interval_list_intersect(sort_interval, valid_interval_times)
        return valid_sort_times

    def _get_filtered_recording_extractor(self, key: dict):
        """Filters and references a recording
        * Loads the NWB file created during insertion as a spikeinterface Recording
        * Slices recording in time (interval) and space (channels);
          recording chunks fromdisjoint intervals are concatenated
        * Applies referencing and bandpass filtering

        Parameters
        ----------
        key: dict, 
            primary key of SpikeSortingRecording table

        Returns
        -------
        recording: si.Recording
        """
        
        nwb_file_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])   
        recording = se.read_nwb_recording(nwb_file_abs_path, load_time_vector=True)
        
        valid_sort_times = SpikeSortingRecording().get_sort_interval_valid_times(key)
        # shape is (N, 2)
        valid_sort_times_indices = np.array([np.searchsorted(recording.get_times(), interval) \
                                             for interval in valid_sort_times])
        # join intervals of indices that are adjacent
        valid_sort_times_indices = reduce(union_adjacent_index, valid_sort_times_indices)
        if valid_sort_times_indices.ndim==1:
            valid_sort_times_indices = np.expand_dims(valid_sort_times_indices, 0)
            
        # concatenate if there is more than one disjoint sort interval
        # TODO: this doens't work; fix
        if len(valid_sort_times_indices)>1:
            recordings_list = []
            for interval_indices in valid_sort_times_indices:
                recording_single = recording.frame_slice(start_frame=interval_indices[0],
                                                        end_frame=interval_indices[1])
                recordings_list.append(recording_single)
            recording = si.concatenate_recordings(recordings_list)
        else:
            recording = recording.frame_slice(start_frame=valid_sort_times_indices[0][0],
                                              end_frame=valid_sort_times_indices[0][1])
        
        channel_ids = (SortGroup.SortGroupElectrode & {'nwb_file_name': key['nwb_file_name'],
                                                       'sort_group_id': key['sort_group_id']}).fetch('electrode_id')
        channel_ids = channel_ids.tolist()
        ref_channel_id = (SortGroup & {'nwb_file_name': key['nwb_file_name'],
                                       'sort_group_id': key['sort_group_id']}).fetch('sort_reference_electrode_id')
        ref_channel_id = ref_channel_id.tolist()
        
        # include ref channel in first slice, then exclude it in second slice
        if ref_channel_id[0] >= 0:
            channel_ids_ref = channel_ids + ref_channel_id
            recording = recording.channel_slice(channel_ids=channel_ids_ref)
            
            recording = st.preprocessing.common_reference(recording, reference='single',
                                                          ref_channel_ids=ref_channel_id)
            recording = recording.channel_slice(channel_ids=channel_ids)
        elif ref_channel_id[0] == -2:
            recording = recording.channel_slice(channel_ids=channel_ids)
            recording = st.preprocessing.common_reference(recording, reference='global',
                                                          operator='median')

        filter_params = (SpikeSortingPreprocessingParameters & key).fetch1('preproc_params')
        recording = st.preprocessing.bandpass_filter(recording, freq_min=filter_params['frequency_min'],
                                                     freq_max=filter_params['frequency_max'])
       
        return recording

    @staticmethod
    def get_recording_timestamps(key: dict):
        """Returns the timestamps for the specified SpikeSortingRecording entry

        Parameters
        ---------
        key: dict
        
        Returns
        -------
        timestamps: np.array, (N, )
        """
        nwb_file_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])
        # TODO fix to work with any electrical series object
        with pynwb.NWBHDF5IO(nwb_file_abs_path, 'r', load_namespaces=True) as io:
            nwbfile = io.read()
            timestamps = nwbfile.acquisition['e-series'].timestamps[:]

        sort_interval = (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                         'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')

        sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))
        timestamps = timestamps[sort_indices[0]:sort_indices[1]]
        return timestamps

@schema
class SpikeSortingWorkspace(dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    channel = 'franklab2' : varchar(80) # the name of the kachery channel for data sharing
    workspace_uri: varchar(1000)
    """

    def make(self, key: dict):
        # load recording, wrap it as old spikeextractors recording, then save as h5 (sortingview requires this format)
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        recording = si.load_extractor(recording_path)
        old_recording = si.create_extractor_from_new_recording(recording)
        h5_recording = sv.LabboxEphysRecordingExtractor.store_recording_link_h5(old_recording, 
                                                recording_path, dtype='int16')

        workspace_name = SpikeSortingRecording()._get_recording_name(key)
        workspace = sv.create_workspace(label=workspace_name)
        workspace.add_recording(recording=h5_recording, label=workspace_name)
        
        # Give permission to workspace based on Google account
        team_name = (SpikeSortingRecordingSelection & key).fetch1('team_name')
        team_members = (LabTeam.LabTeamMember & {'team_name': team_name}).fetch('lab_member_name')
        set_workspace_permission(workspace_name, team_members)    
        
        key['workspace_uri'] = workspace.uri
        self.insert1(key)

    # TODO: check this function
    def store_sorting(self, key, *, sorting, sorting_label, path_suffix = '', sorting_id=None, metrics=None):
        """store a sorting in the Workspace given by the key and in the SortingID table. 
        :param key: key to SpikeSortingWorkspace
        :type key: dict
        :param sorting: sorting extractor
        :type sorting: BaseSortingExtractor
        :param sorting_label: label for sort
        :type sorting_label: str
        :param path_suffix: string to append to end of sorting extractor file name
        :type path_suffix: str
        :param metrics: spikesorting metrics, defaults to None
        :type metrics: dict
        :returns: sorting_id 
        :type: str
        """
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
        sorting = sv.LabboxEphysSortingExtractor(sorting_object)
        # note that we only ever have one recording per workspace
        sorting_id = s_key['sorting_id'] = workspace.add_sorting(recording_id=workspace.recording_ids[0], 
                                    sorting=sorting, label=sorting_label)
        SortingID.insert1(s_key)
        if metrics is not None:
            self.add_metrics_to_sorting(key, sorting_id=sorting_id, metrics=metrics, workspace=workspace)

        return s_key['sorting_id']

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

@schema
class SpikeSorter(dj.Manual):
    definition = """
    # Table that holds the list of spike sorters avaialbe through spikeinterface
    sorter_name: varchar(80) # the name of the spike sorting algorithm
    """
    def insert_default(self):
        """Add each of the sorters from spikeinterface.sorters
        """
        sorters = ss.available_sorters()
        for sorter in sorters:
            self.insert1({'sorter_name': sorter}, skip_duplicates=True)

@schema
class SpikeSorterParameters(dj.Manual):
    definition = """
    -> SpikeSorter
    spikesorter_parameter_set_name: varchar(80) # label for this set of parameters
    ---
    parameter_dict: blob # dictionary of parameter names and values
    """

    def insert_default(self):
        """Insert default parameters for each sorter
        """
        # set up the default filter parameters
        sort_param_dict = dict()
        sort_param_dict['spikesorter_parameter_set_name'] = 'default'
        sorters = ss.available_sorters()
        for sorter in sorters:
            if len((SpikeSorter() & {'sorter_name': sorter}).fetch()):
                sort_param_dict['sorter_name'] = sorter
                sort_param_dict['parameter_dict'] = ss.get_default_params(sorter)
                self.insert1(sort_param_dict, replace=True)
            else:
                print(
                    f'Error in SpikeSorterParameter: sorter {sorter} not in SpikeSorter schema')
                continue

@schema
class SpikeSortingSelection(dj.Manual):
    definition = """
    # Table for holding selection of recording and parameters for each spike sorting run
    -> SpikeSortingRecording
    -> SpikeSorterParameters
    ---
    import_path = '': varchar(200) # optional path to previous curated sorting output
    """

@schema
class SpikeSorting(dj.Computed):
    definition = """
    # Table for holding spike sorting runs
    -> SpikeSortingSelection
    ---
    sorting_path: varchar(200)
    time_of_sort: int   # in Unix time, to the nearest second
    -> AnalysisNwbfile
    units_object_id: varchar(40)   # Object ID for the units in NWB file
    """

    def make(self, key: dict):
        """Runs spike sorting on the data and parameters specified by the
        SpikeSortingSelection table and inserts a new entry to SpikeSorting table.
        Specifically,
        1. Loads saved recording and runs the sort on it with spikeinterface
        2. Saves the sorting with spikeinterface and adds it to the workspace (if it exists)
        3. Creates an analysis NWB file and saves the sorting there
           (this is redundant with 2; will change in the future)
        """

        # TODO: check that this works for formats other than h5_v1
        recording_object = (SpikeSortingRecording & key).fetch1('recording_extractor_object')
        recording = sv.LabboxEphysRecordingExtractor(recording_object)
        
        # currently, sorting cannot use multiple workers if it is not first saved to disk
        # in a format that can be read by new spikeinterface. as a workaround, we save this
        # to binary first and then reload with new spikeinterface. in the future we will not
        # use h1v1 recording extractor; instead we will use nwb recording, which can directly
        # be loaded with new si and will not need this step
        tmpdir = tempfile.TemporaryDirectory(dir=os.environ['KACHERY_TEMP_DIR'])
        cached_recording = spikeextractors.CacheRecordingExtractor(recording, 
                                                                   save_path=str(Path(tmpdir.name) / 'r.dat'))

        # reload with new spikeinterface
        new_recording = si.read_binary(file_paths=str(Path(tmpdir.name) / 'r.dat'),
                                       sampling_frequency=cached_recording.get_sampling_frequency(),
                                       num_chan=cached_recording.get_num_channels(),
                                       dtype=cached_recording.get_dtype(),
                                       time_axis=0,
                                       is_filtered=True)
        new_recording.set_channel_locations(locations=recording.get_channel_locations())
        
        # TODO: turn SpikeSortingFilterParameters to SpikeSortingPreprocessingParameters
        # and include seed, chunk size etc
        recording = st.preprocessing.whiten(recording=new_recording, seed=0)
        
        print(f'Running spike sorting on {key}...')
        sort_parameters = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                                    'spikesorter_parameter_set_name': key['spikesorter_parameter_set_name']}).fetch1()
        sorting = ss.run_sorter(key['sorter_name'], recording,
                                output_folder=os.getenv('SORTING_TEMP_DIR', None),
                                **sort_parameters['parameter_dict'])
        key['time_of_sort'] = int(time.time())

        print('Saving sorting results...')
        sorting_label = self.get_sorting_name(key)
        key['sorting_id'] = SpikeSortingWorkspace().store_sorting(key, sorting=sorting,
                                                                  sorting_label=sorting_label)
        sort_interval_list_name = (SpikeSortingRecording & key).fetch1('sort_interval_list_name')
        sort_interval = (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                         'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        key['analysis_file_name'], key['units_object_id'] = \
            SortingID.store_sorting_nwb(key, sorting=sorting, sort_interval_list_name=sort_interval_list_name, 
                                                sort_interval=sort_interval)
        self.insert1(key)
    
    def delete(self):
        """Extends the delete method of base class to implement permission checking
        Note that this is NOT a security feature, as anyone that has access to source code
        can disable it; it just makes it less likely for the user to accidentally delete entries
        """
        current_user_name = dj.config['database.user']
        entries = self.fetch()
        permission_bool = np.zeros((len(entries),))
        print(f'Attempting to delete {len(entries)} entries, checking permission...')
    
        for entry_idx in range(len(entries)):
            # check the team name for the entry, then look up the members in that team, then get their datajoint user names
            team_name = (SpikeSortingRecordingSelection & (SpikeSortingRecordingSelection & entries[entry_idx]).proj()).fetch1()['team_name']
            lab_member_name_list = (LabTeam.LabTeamMember & {'team_name': team_name}).fetch('lab_member_name')
            datajoint_user_names = []
            for lab_member_name in lab_member_name_list:
                datajoint_user_names.append((LabMember.LabMemberInfo & {'lab_member_name': lab_member_name}).fetch1('datajoint_user_name'))
            permission_bool[entry_idx] = current_user_name in datajoint_user_names
        if np.sum(permission_bool)==len(entries):
            print('Permission to delete all specified entries granted.')
            super().delete()
        else:
            raise Exception('You do not have permission to delete all specified entries. Not deleting anything.')

    def _get_sorting_name(self, key):
        recording_name = SpikeSortingRecording()._get_recording_name(key)
        sorting_name = recording_name + '_' \
                       + key['sorter_name'] + '_' \
                       + key['spikesorter_parameter_set_name']
        return sorting_name
    
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)
 
    def get_sorting_extractor(self, key, sort_interval):
        # TODO: replace with spikeinterface call if possible
        """Generates a numpy sorting extractor given a key that retrieves a SpikeSorting and a specified sort interval

        :param key: key for a single SpikeSorting
        :type key: dict
        :param sort_interval: [start_time, end_time]
        :type sort_interval: numpy array
        :return: a spikeextractors sorting extractor with the sorting information
        """
        # get the units object from the NWB file that the data are stored in.
        units = (SpikeSorting & key).fetch_nwb()[0]['units'].to_dataframe()
        unit_timestamps = []
        unit_labels = []

        raw_data_obj = (Raw() & {'nwb_file_name': key['nwb_file_name']}).fetch_nwb()[
            0]['raw']
        # get the indices of the data to use. Note that spike_extractors has a time_to_frame function,
        # but it seems to set the time of the first sample to 0, which will not match our intervals
        timestamps = np.asarray(raw_data_obj.timestamps)
        sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))

        unit_timestamps_list = []
        # TODO: do something more efficient here; note that searching for maching sort_intervals within pandas doesn't seem to work
        for index, unit in units.iterrows():
            if np.ndarray.all(np.ravel(unit['sort_interval']) == sort_interval):
                # unit_timestamps.extend(unit['spike_times'])
                unit_frames = np.searchsorted(
                    timestamps, unit['spike_times']) - sort_indices[0]
                unit_timestamps.extend(unit_frames)
                # unit_timestamps_list.append(unit_frames)
                unit_labels.extend([index] * len(unit['spike_times']))

        output = se.NumpySortingExtractor()
        output.set_times_labels(times=np.asarray(
            unit_timestamps), labels=np.asarray(unit_labels))
        return output

    # TODO: write a function to import sorted data
    def import_sorted_data(self, key):
        # Check if spikesorting has already been run on this dataset;
        # if import_path is not empty, that means there exists a previous spikesorting run
        import_path = (SpikeSortingSelection() & key).fetch1('import_path')
        if import_path != '':
            sort_path = Path(import_path)
            assert sort_path.exists(
            ), f'Error: import_path {import_path} does not exist when attempting to import {(SpikeSortingSelection() & key).fetch1()}'
            # the following assumes very specific file names from the franklab, change as needed
            firings_path = sort_path / 'firings_processed.mda'
            assert firings_path.exists(
            ), f'Error: {firings_path} does not exist when attempting to import {(SpikeSortingSelection() & key).fetch1()}'
            # The firings has three rows, the electrode where the peak was detected, the sample count, and the cluster ID
            firings = readmda(str(firings_path))
            # get the clips
            clips_path = sort_path / 'clips.mda'
            assert clips_path.exists(
            ), f'Error: {clips_path} does not exist when attempting to import {(SpikeSortingSelection() & key).fetch1()}'
            clips = readmda(str(clips_path))
            # get the timestamps corresponding to this sort interval
            # TODO: make sure this works on previously sorted data
            # timestamps = timestamps[np.logical_and(
            #     timestamps >= sort_interval[0], timestamps <= sort_interval[1])]
            # # get the valid times for the sort_interval
            # sort_interval_valid_times = interval_list_intersect(
            #     np.array([sort_interval]), valid_times)

            # # get a list of the cluster numbers
            # unit_ids = np.unique(firings[2, :])
            # for index, unit_id in enumerate(unit_ids):
            #     unit_indices = np.ravel(np.argwhere(firings[2, :] == unit_id))
            #     units[unit_id] = timestamps[firings[1, unit_indices]]
            #     units_templates[unit_id] = np.mean(
            #         clips[:, :, unit_indices], axis=2)
            #     units_valid_times[unit_id] = sort_interval_valid_times
            #     units_sort_interval[unit_id] = [sort_interval]

            # TODO: Process metrics and store in Units table.
            metrics_path = (sort_path / 'metrics_processed.json').exists()
            assert metrics_path.exists(
            ), f'Error: {metrics_path} does not exist when attempting to import {(SpikeSortingSelection() & key).fetch1()}'
            metrics_processed = json.load(metrics_path)

    def nightly_cleanup(self):
        """Clean up spike sorting directories that are not in the SpikeSorting table. 
        This should be run after AnalysisNwbFile().nightly_cleanup()
        """
        # get a list of the files in the spike sorting storage directory
        dir_names = next(os.walk(os.environ['SPIKE_SORTING_STORAGE_DIR']))[1]
        # now retrieve a list of the currently used analysis nwb files
        analysis_file_names = self.fetch('analysis_file_name')
        for dir in dir_names:
            if not dir in analysis_file_names:
                full_path = str(pathlib.Path(os.environ['SPIKE_SORTING_STORAGE_DIR']) / dir)
                print(f'removing {full_path}')
                shutil.rmtree(str(pathlib.Path(os.environ['SPIKE_SORTING_STORAGE_DIR']) / dir))


# @schema
# class ModifySortingParameters(dj.Manual):
#     definition = """
#     # Table for holding parameters for modifying the sorting (e.g. merging clusters, )
#     modify_sorting_parameter_set_name: varchar(200)   #name of this parameter set
#     ---
#     modify_sorting_parameter_dict: BLOB         #dictionary of variables and values for cluster merging
#     """

# @schema
# class ModifySortingSelection(dj.Manual):
#     definition = """
#     # Table for selecting a sorting and cluster merge parameters for merging
#     -> SpikeSortingRecording
#     -> ModifySortingParameters
#     """
#     class SortingsIDs(dj.Part):
#         # one entry for each sorting ID that should be used.
#         definition = """
#         -> SortingID
#         """
   

# @schema
# class ModifySorting(dj.Computed):
#     definition = """
#     # Table for merging / modifying clusters based on parameters
#     -> ModifySortingSelection
#     ---
#     -> SortingID
#     -> AnalysisNwbfile
#     units_object_id: varchar(40)           # Object ID for the units in NWB file
#     """
    #TODO: implement cluster merge, burst merge

    # if 'delete_duplicate_spikes' in acpd:
    #     if acpd['delete_duplicate_spikes']:
    #         print('deleting duplicate spikes')
    #         # look up the detection interval 
    #         param_dict = (SpikeSorterParameters & key).fetch1('parameter_dict')
    #         if 'detect_interval' not in param_dict:
    #             Warning(f'delete_duplicate_spikes enabled, but detect_interval is not specified in the spike sorter parameters {key["parameter_set_name"]}; skipping')
    #         else:
    #             unit_samples = np.empty((0,),dtype='int64')
    #             unit_labels = np.empty((0,),dtype='int64')
    #             for unit in sorting.get_unit_ids():
    #                 tmp_samples = sorting.get_unit_spike_train(unit)
    #                 invalid = np.where(np.diff(tmp_samples) < param_dict['detect_interval'])[0]
    #                 tmp_samples = np.delete(tmp_samples, invalid)    
    #                 print(f'Unit {unit}: {len(invalid)} spikes deleted')
    #                 tmp_labels = np.asarray([unit]*len(tmp_samples))
    #                 unit_samples = np.hstack((unit_samples, tmp_samples))
    #                 unit_labels = np.hstack((unit_labels, tmp_labels))
    #             # sort the times and labels
    #             sort_ind = np.argsort(unit_samples)
    #             unit_samples = unit_samples[sort_ind]
    #             unit_labels = unit_labels[sort_ind]
    #             # create a numpy sorting extractor
    #             new_sorting = se.NumpySortingExtractor()
    #             new_sorting.set_times_labels(times=unit_samples, labels=unit_labels.tolist())
    #             new_sorting.set_sampling_frequency((Raw & key).fetch1('sampling_rate'))
    #             sorting_modified = True
    # if 'merge_clusters' in acpd:
    #     # do merge stuff here
    #     # initialize the labels dictionary
    #     labels = dict()
    #     labels['mergeGroups'] = []
    #     # format: labels['mergeGroups'] = [[1, 2, 5], [3, 4]] would merge units 1,2,and 5 and, separately, 3 and4
    #     labels['labelsByUnit'] = dict()
    #     # format: labels['labelsByUnit'] = {1:'accept', 2:'noise,reject'}] would label unit 1 as 'accept' and unit 2 as 'noise' and 'reject'
        
    #     # List of available functions for spike waveform extraction: 
    #     # https://spikeinterface.readthedocs.io/en/0.13.0/api.html#module-spiketoolkit.postprocessing

    #     sorting_modified = True