from csv import list_dialects
import os
import pathlib
import time
from pathlib import Path
import shutil
import uuid
from functools import reduce

import datajoint as dj
import numpy as np
import pynwb
import scipy.stats as stats
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st

from .common_device import Probe
from .common_lab import LabMember, LabTeam
from .common_ephys import Electrode, ElectrodeGroup
from .common_interval import (IntervalList, SortInterval,
                              interval_list_intersect, union_adjacent_index)
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .dj_helper_fn import dj_replace, fetch_nwb
from .nwb_helper_fn import get_valid_intervals


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
        -> SortGroup
        -> Electrode
        """

    def set_group_by_shank(self, nwb_file_name: str, references: dict=None, omit_ref_electrode_group=False):
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
        omit_ref_electrode_group : bool
            Optional. If True, no sort group is defined for electrode group of reference.
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

                # If not omitting electrode group that reference electrode is a part of, or if doing this but current
                # electrode group not same as reference electrode group, insert sort group and procceed to define sort
                # group electrodes
                reference_electrode_group = \
                electrodes[electrodes["electrode_id"] == sg_key['sort_reference_electrode_id']][
                    "electrode_group_name"]
                # If reference electrode corresponds to a real electrode (and not for example the flag for common
                # average referencing),
                # check that exactly one electrode group was found for it, and take that electrode group.
                if len(reference_electrode_group) == 1:
                    reference_electrode_group = reference_electrode_group[0]
                elif (int(sg_key['sort_reference_electrode_id']) > 0) and (len(reference_electrode_group) != 1):
                    raise Exception(
                        f"Should have found exactly one electrode group for reference electrode,"
                        f"but found {len(reference_electrode_group)}.")
                if not omit_ref_electrode_group or (str(e_group) != str(reference_electrode_group)):
                    self.insert1(sg_key)
                    shank_elect = electrodes['electrode_id'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                            electrodes['probe_shank'] == shank)]
                    for elect in shank_elect:
                        sge_key['electrode_id'] = elect
                        self.SortGroupElectrode().insert1(sge_key)
                    sort_group += 1
                else:
                    print(f"Omitting electrode group {e_group} from sort groups because contains reference.")

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
                                 'whiten': True,
                                 'seed': seed}
        self.insert1(key, skip_duplicates=True)
        
@schema
class SpikeSortingRecordingSelection(dj.Manual):
    definition = """
    # Defines recordings to be sorted
    -> SortGroup
    -> SortInterval
    -> SpikeSortingPreprocessingParameters
    -> LabTeam
    ---
    -> IntervalList
    """

@schema
class SpikeSortingRecording(dj.Computed):
    definition = """
    -> SpikeSortingRecordingSelection
    ---
    recording_path: varchar(1000)
    -> IntervalList.proj(sort_interval_list_name='interval_list_name')
    """
    def make(self, key):
        sort_interval_valid_times = self._get_sort_interval_valid_times(key)
        recording = self._get_filtered_recording(key)

        recording_name = self._get_recording_name(key)

        tmp_key = {}
        tmp_key['nwb_file_name'] = key['nwb_file_name']
        tmp_key['interval_list_name'] = recording_name
        tmp_key['valid_times'] = sort_interval_valid_times
        IntervalList.insert1(tmp_key, replace=True)

        # store the list of valid times for the sort
        key['sort_interval_list_name'] = tmp_key['interval_list_name']

        # Path to files that will hold the recording extractors
        recording_folder = Path(os.getenv('NWB_DATAJOINT_RECORDING_DIR'))
        key['recording_path'] = str(recording_folder / Path(recording_name))
        if os.path.exists(key['recording_path']):
            shutil.rmtree(key['recording_path'])
        recording = recording.save(folder=key['recording_path'], n_jobs=4,
                                   total_memory='5G')        
        self.insert1(key)
    
    @staticmethod
    def _get_recording_name(self, key):
        recording_name = key['nwb_file_name'] + '_' \
        + key['sort_interval_name'] + '_' \
        + str(key['sort_group_id']) + '_' \
        + key['preproc_params_name']
        return recording_name
    
    @staticmethod
    def _get_recording_timestamps(recording):
        if recording.get_num_segments()>1:
            frames_per_segment = [0]
            for i in range(len(recording.recording_list)):
                frames_per_segment.append(recording.get_num_frames(segment_index=i))

            cumsum_frames = np.cumsum(frames_per_segment)
            total_frames = np.sum(frames_per_segment)

            timestamps = np.zeros((total_frames,))
            for i in range(len(recording.recording_list)):
                timestamps[cumsum_frames[i]:cumsum_frames[i+1]] = recording.get_times(segment_index=i)
        else:
            timestamps = recording.get_times()
        return timestamps

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

    def _get_filtered_recording(self, key: dict):
        """Filters and references a recording
        * Loads the NWB file created during insertion as a spikeinterface Recording
        * Slices recording in time (interval) and space (channels);
          recording chunks from disjoint intervals are concatenated
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
        
        valid_sort_times = self._get_sort_interval_valid_times(key)
        # shape is (N, 2)
        valid_sort_times_indices = np.array([np.searchsorted(recording.get_times(), interval) \
                                             for interval in valid_sort_times])
        # join intervals of indices that are adjacent
        valid_sort_times_indices = reduce(union_adjacent_index, valid_sort_times_indices)
        if valid_sort_times_indices.ndim==1:
            valid_sort_times_indices = np.expand_dims(valid_sort_times_indices, 0)
            
        # create an AppendRecording if there is more than one disjoint sort interval
        if len(valid_sort_times_indices)>1:
            recordings_list = []
            for interval_indices in valid_sort_times_indices:
                recording_single = recording.frame_slice(start_frame=interval_indices[0],
                                                         end_frame=interval_indices[1])
                recordings_list.append(recording_single)
            recording = si.append_recordings(recordings_list)
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
        else:
            raise ValueError("Invalid reference channel ID")
        filter_params = (SpikeSortingPreprocessingParameters & key).fetch1('preproc_params')
        recording = st.preprocessing.bandpass_filter(recording, freq_min=filter_params['frequency_min'],
                                                     freq_max=filter_params['frequency_max'])
       
        return recording

@schema
class SpikeSorterParameters(dj.Manual):
    definition = """
    sorter: varchar(200)
    sorter_params_name: varchar(200)
    ---
    sorter_params: blob
    """
    def insert_default(self):
        """Default params from spike sorters available via spikeinterface
        """
        sorters = ss.available_sorters()
        for sorter in sorters:
            sorter_params = ss.get_default_params(sorter)
            self.insert1([sorter, 'default', sorter_params], skip_duplicates=True)

from .common_artifact import ArtifactRemovedIntervalList
@schema
class SpikeSortingSelection(dj.Manual):
    definition = """
    # Table for holding selection of recording and parameters for each spike sorting run
    -> SpikeSortingRecording
    -> SpikeSorterParameters
    -> ArtifactRemovedIntervalList
    ---
    import_path = "": varchar(200)  # optional path to previous curated sorting output
    """

@schema
class SpikeSorting(dj.Computed):
    definition = """
    -> SpikeSortingSelection
    ---
    sorting_path: varchar(1000)
    time_of_sort: int   # in Unix time, to the nearest second
    -> AnalysisNwbfile
    units_object_id: varchar(40)   # Object ID for the units in NWB file
    """

    def make(self, key: dict):
        """Runs spike sorting on the data and parameters specified by the
        SpikeSortingSelection table and inserts a new entry to SpikeSorting table.
        Specifically,
        1. Loads saved recording and runs the sort on it with spikeinterface
        2. Saves the sorting with spikeinterface
        3. Creates an analysis NWB file and saves the sorting there
           (this is redundant with 2; will change in the future)
        """
        
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        recording = si.load_extractor(recording_path)
        
        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)

        # load valid times
        artifact_times = (ArtifactRemovedIntervalList & key).fetch1('artifact_times')
        if artifact_times.ndim==1:
            artifact_times = np.expand_dims(artifact_times,0)
        
        if artifact_times:
            # convert valid intervals to indices
            list_triggers = []
            for interval in artifact_times:
                list_triggers.append(np.arange(np.searchsorted(timestamps, interval[0]),
                                            np.searchsorted(timestamps, interval[1])))
            list_triggers = np.asarray(list_triggers).flatten().tolist()
            
            if recording.get_num_segments()>1:
                recording = si.concatenate_recordings(recording.recording_list)
            recording = st.remove_artifacts(recording=recording, list_triggers=list_triggers,
                                            ms_before=0, ms_after=0, mode='zeros')
        
        # NOTE: decided not to do this and instead just use remove_artifacts; keep for now
        # slice valid times and append them
        # rec_list = []
        # for idx_interval in artifact_removed_valid_times_idx_list:
        #     if recording.get_num_segments()>1:
        #         segment_ind = np.ravel(np.argwhere(cumsum_frames <= idx_interval[0]))[-1]
        #         rec = recording.recording_list[segment_ind]
        #         sliced_rec = rec.frame_slice(start_frame=idx_interval[0]-cumsum_frames[segment_ind],
        #                                      end_frame=idx_interval[1]-cumsum_frames[segment_ind])
        #     else:
        #         sliced_rec = recording.frame_slice(start_frame=idx_interval[0], end_frame=idx_interval[1])
        #     rec_list.append(sliced_rec)
        # recording = si.append_recordings(rec_list)
                
        preproc_params = (SpikeSortingPreprocessingParameters & key).fetch1('preproc_params')
        if preproc_params['whiten']:
            recording = st.preprocessing.whiten(recording=recording, seed=preproc_params['seed'])
        
        print(f'Running spike sorting on {key}...')
        sorter, sorter_params = (SpikeSorterParameters & key).fetch1('sorter','sorter_params')
        sorting = ss.run_sorter(sorter, recording,
                                output_folder=os.getenv('KACHERY_TEMP_DIR'),
                                delete_output_folder=True,
                                **sorter_params)
        key['time_of_sort'] = int(time.time())

        print('Saving sorting results...')
        sorting_folder = Path(os.getenv('NWB_DATAJOINT_SORTING_DIR'))
        sorting_name = self._get_sorting_name(key)
        key['sorting_path'] = str(sorting_folder / Path(sorting_name))
        if os.path.exists(key['sorting_path']):
            shutil.rmtree(key['sorting_path'])
        sorting = sorting.save(folder=key['sorting_path'])
        
        # NWB stuff
        sort_interval_list_name = (SpikeSortingRecording & key).fetch1('sort_interval_list_name')
        sort_interval = (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                         'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')        
        key['analysis_file_name'], key['units_object_id'] = self._save_sorting_nwb(key, sorting, timestamps,
                                                                                   sort_interval_list_name, 
                                                                                   sort_interval)
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])       
        
        sorting_key = (self & key).proj().fetch1()
        sorting_key['sorting_id'] = 'S_'+str(uuid.uuid4())[:8]
        sorting_key['sorting_path'] = key['sorting_path']
        sorting_key['parent_sorting_id'] = ''
        
        # add sorting to CuratedSpikeSorting
        CuratedSpikeSorting.insert1(sorting_key, skip_duplicates=True)
        self.insert1(key)
    
    def delete(self):
        """Extends the delete method of base class to implement permission checking.
        Note that this is NOT a security feature, as anyone that has access to source code
        can disable it; it just makes it less likely to accidentally delete entries.
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
        
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)
    
    def nightly_cleanup(self):
        """Clean up spike sorting directories that are not in the SpikeSorting table.
        This should be run after AnalysisNwbFile().nightly_cleanup()
        """
        # get a list of the files in the spike sorting storage directory
        dir_names = next(os.walk(os.environ['NWB_DATAJOINT_SORTING_DIR']))[1]
        # now retrieve a list of the currently used analysis nwb files
        analysis_file_names = self.fetch('analysis_file_name')
        for dir in dir_names:
            if not dir in analysis_file_names:
                full_path = str(pathlib.Path(os.environ['NWB_DATAJOINT_SORTING_DIR']) / dir)
                print(f'removing {full_path}')
                shutil.rmtree(str(pathlib.Path(os.environ['NWB_DATAJOINT_SORTING_DIR']) / dir))
                
    def _get_sorting_name(self, key):
        recording_name = SpikeSortingRecording._get_recording_name(key)
        sorting_name = recording_name + '_' \
                       + key['sorter'] + '_' \
                       + key['sorter_params_name'] + '_' \
                       + key['artifact_removed_interval_list_name']
        return sorting_name
    
    def _save_sorting_nwb(self, key, sorting, timestamps, sort_interval_list_name,
                          sort_interval, metrics=None, unit_ids=None):
        """Store a sorting in a new AnalysisNwbfile
        Parameters
        ----------
        key : dict
            key to SpikeSorting table
        sorting : si.Sorting
            sorting
        timestamps : array_like
            Time stamps of the sorted recoridng;
            used to convert the spike timings from index to real time
        sort_interval_list_name : str
            name of sort interval
        sort_interval : list
            interval for start and end of sort
        metrics : dict, optional
            quality metrics, by default None
        unit_ids : list, optional
            IDs of units whose spiketrains to save, by default None

        Returns
        -------
        analysis_file_name : str
        units_object_id : str
        """

        sort_interval_valid_times = (IntervalList & \
                {'interval_list_name': sort_interval_list_name}).fetch1('valid_times')

        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        for unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(unit_id=unit_id)
            units[unit_id] = timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
        object_ids = AnalysisNwbfile().add_units(analysis_file_name,
                                        units, units_valid_times,
                                        units_sort_interval,
                                        metrics=metrics)
        if object_ids=='':
            print('Sorting contains no units. Created an empty analysis nwb file anyway.')
            units_object_id = ''
        else:
            units_object_id = object_ids[0]
        return analysis_file_name,  units_object_id
    
    # TODO: write a function to import sortings done outside of dj
    def _import_sorting(self, key):
        raise NotImplementedError
                
@schema
class CuratedSpikeSorting(dj.Manual):
    definition = """
    # Has records for every sorting; similar to IntervalList
    sorting_id: varchar(10)
    ---
    -> SpikeSorting
    sorting_path: varchar(1000)
    parent_sorting_id='': varchar(10)
    """ 
