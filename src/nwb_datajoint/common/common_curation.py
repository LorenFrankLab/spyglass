import os
import tempfile
import shutil

import datajoint as dj
import numpy as np
import sortingview as sv
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st

from .common_lab import LabMember, LabTeam
from .common_interval import IntervalList, SortInterval
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .common_metrics import QualityMetrics
from .common_spikesorting import (SortingID, SpikeSortingRecordingSelection, SpikeSortingRecording,
                                  SpikeSortingWorkspace, SpikeSortingFilterParameters)

from .dj_helper_fn import fetch_nwb

schema = dj.schema('common_curation')

@schema
class AutomaticCurationParameters(dj.Manual):
    definition = """
    automatic_curation_params_name: varchar(200)   # name of this parameter set
    ---
    merge_params: BLOB   # params to merge units
    reject_params: BLOB   # params to reject units
    """
    def insert_default(self):
        automatic_curation_params_name = 'default'
        merge_params = {}
        reject_params = {}
        self.insert1([automatic_curation_params_name, merge_params, reject_params])

@schema
class AutomaticCurationSelection(dj.Manual):
    definition = """
    # Table for holding the output
    -> SortingID
    -> AutomaticCurationParameters
    -> QualityMetrics
    """
 
@schema
class AutomaticCuration(dj.Computed):
    definition = """
    # Automated curation applied to sortings
    -> AutomaticCurationSelection
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40)   # Object ID for the units in NWB file
    automatic_curation_results_dict=NULL: BLOB   #dictionary of outputs from automatic curation
    """

    def make(self, key):
        # LOGIC:
        #1. Compute the requested metrics
        #3. Using metrics, add labels for noise clusters, etc. 

        key['automatic_curation_results_dict'] = dict()
        re_key = (SpikeSortingRecording & key).fetch1()
        workspace_uri = (SpikeSortingWorkspace & key).fetch1('workspace_uri')

        # get the sortings to be used. 
        sorting_id = key['sorting_id']
  
        # load the workspace, the sorting, and the recording
        workspace = sv.load_workspace(workspace_uri)
        sorting = workspace.get_sorting_extractor(sorting_id)
        recording = workspace.get_recording_extractor(workspace.recording_ids[0])

        auto_curate_param_name = (AutomaticCurationSelection & key).fetch1('automatic_curation_parameter_set_name')
        acpd = (AutomaticCurationParameters & {'automatic_curation_parameter_set_name': auto_curate_param_name}).fetch1('automatic_curation_parameter_dict')
        # check for defined automatic curation keys / parameters
        
        #1. Get the sorting
        analysis_file_created = False
 
        # get the cluster metrics list name and add a name for this sorting
        cluster_metrics_list_name = (AutomaticCurationSelection & key).fetch1('cluster_metrics_list_name')
 
        # 2. Calculate the metrics
        #First, whiten the recording     
        filter_params = (SpikeSortingFilterParameters & key).fetch1('filter_parameter_dict')
        recording = st.preprocessing.whiten(
            recording, seed=0, chunk_size=filter_params['filter_chunk_size'])
        tmpfile = tempfile.NamedTemporaryFile(dir=os.environ['NWB_DATAJOINT_TEMP_DIR'])
        metrics_recording = se.CacheRecordingExtractor(recording, save_path=tmpfile.name, chunk_mb=10000)
        metrics = QualityMetrics().compute_metrics(cluster_metrics_list_name, metrics_recording, sorting)  
        
        # add labels to the sorting based on metrics?
        if 'noise_reject' in acpd:
                if acpd['noise_reject']:
                    # get the noise rejection parameters
                    noise_reject_param = acpd['noise_reject_param']
                    #TODO write noise/ rejection code

        # Store the sorting with metrics in the NWB file and update the metrics in the workspace
        sort_interval_list_name = (SpikeSortingRecording & key).fetch1('sort_interval_list_name')
 
        sort_interval = (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                         'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
 
        key['analysis_file_name'], key['units_object_id'] = \
            SortingID.store_sorting_nwb(key, sorting=sorting, sort_interval_list_name=sort_interval_list_name, 
            sort_interval=sort_interval, metrics=metrics)

        SpikeSortingWorkspace().add_metrics_to_sorting(key, sorting_id=sorting_id, metrics=metrics)
        self.insert1(key)
        

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def delete(self):
        """
        Extends the delete method of base class to implement permission checking
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
            # delete the sortings from the workspaces
            #for entry_idx in range(len(entries)):
                #print(entries[entry_idx])
                #TODO FIX:
                #key = (self & (self & entries[entry_idx]).proj())
                # workspace_uri = key['curation_feed_uri'] 
                #print(key)
                # # load the workspace and the sorting
                # workspace = sv.load_workspace(workspace_uri)
                # sorting_id = key['sorting_id']
                # workspace.delete_sorting(sorting_id)
            super().delete()
        else:
            raise Exception('You do not have permission to delete all specified entries. Not deleting anything.')

@schema 
class CuratedSpikeSortingSelection(dj.Manual):
    definition = """
    -> AutomaticCuration
    ---
    """

@schema
class CuratedSpikeSorting(dj.Computed):
    definition = """
    # Fully curated spike sorting
    -> CuratedSpikeSortingSelection
    ---
    -> AnalysisNwbfile   # New analysis NWB file to hold unit info
    units_object_id: varchar(40)   # Object ID for the units in NWB file
    """

    class Unit(dj.Part):
        definition = """
        # Table for holding sorted units
        -> master
        unit_id: int   # ID for each unit
        ---
        label='': varchar(80)   # optional label for each unit
        noise_overlap=-1: float   # noise overlap metric for each unit
        nn_isolation=-1: float   # isolation score metric for each unit
        isi_violation=-1: float   # ISI violation score for each unit
        firing_rate=-1: float   # firing rate
        num_spikes=-1: int   # total number of spikes
        """

    def make(self, key):
        # define the list of properties. TODO: get this from table definition.
        non_metric_fields = ['spike_times', 'obs_intervals', 'sort_interval']

        # Creating the curated units table involves 4 steps:
        # 1. Merging units labeled for merge
        # 2. Recalculate metrics
        # 3. Inserting accepted units into new analysis NWB file and into the Curated Units table.

        # 1. Merge
        # Get the new curated soring from the workspace.
        workspace_uri = (SpikeSortingWorkspace & key).fetch1('workspace_uri')
        workspace = sv.load_workspace(workspace_uri=workspace_uri)
        sorting_id = key['sorting_id']
        sorting = workspace.get_curated_sorting_extractor(sorting_id)

        # Get labels
        labels = workspace.get_sorting_curation(sorting_id)

        # turn labels to list of str, only including accepted units.
        accepted_units = []
        unit_labels = labels['labelsByUnit']
        for idx, unitId in enumerate(unit_labels):
            if 'accept' in unit_labels[unitId]:
                accepted_units.append(unitId)            

        # remove non-primary merged units
        clusters_merged = bool(labels['mergeGroups'])
        if clusters_merged:
            clusters_merged = True
            for m in labels['mergeGroups']:
                if set(m[1:]).issubset(accepted_units):
                    for cell in m[1:]:
                        accepted_units.remove(cell)

        # get the labels for the accepted units
        labels_concat = []
        for unitId in accepted_units:
            label_concat = ','.join(unit_labels[unitId])
            labels_concat.append(label_concat)

        print(f'Found {len(accepted_units)} accepted units')

        # exit out if there are no labels or no accepted units
        if len(unit_labels) == 0 or len(accepted_units) == 0:
            print(f'{key}: no curation found or no accepted units')
            return

        # get the original units from the Automatic curation NWB file
        orig_units = (AutomaticCuration & key).fetch_nwb()[0]['units']
        orig_units = orig_units.loc[accepted_units]
        #TODO: fix if unit 0 doesn't exist
        sort_interval = orig_units.iloc[0]['sort_interval']
        sort_interval_list_name = (SpikeSortingRecording & key).fetch1('sort_interval_list_name')

        # 2. Recalculate metrics for curated units to account for merges if there were any
        # get the recording extractor
        if clusters_merged:
            recording = workspace.get_recording_extractor(
                workspace.recording_ids[0])
            tmpfile = tempfile.NamedTemporaryFile(dir=os.environ['NWB_DATAJOINT_TEMP_DIR'])
            recording = se.CacheRecordingExtractor(
                                recording, save_path=tmpfile.name, chunk_mb=10000)
            # whiten the recording
            filter_params = (SpikeSortingFilterParameters & key).fetch1('filter_parameter_dict')
            recording = st.preprocessing.whiten(
                recording, seed=0, chunk_size=filter_params['filter_chunk_size'])
            cluster_metrics_list_name = (CuratedSpikeSortingSelection & key).fetch1('final_cluster_metrics_list_name')
            # metrics = QualityMetrics().compute_metrics(cluster_metrics_list_name, recording, sorting)
        else:
            # create the metrics table
            for col_name in orig_units.columns:
                if col_name in non_metric_fields:
                    orig_units = orig_units.drop(col_name, axis=1)
            metrics = orig_units
 
        # Limit the metrics to accepted units
        metrics = metrics.loc[accepted_units]

        # 3. Save the accepted, merged units and their metrics
        # load the AnalysisNWBFile from the original sort to get the sort_interval_valid times and the sort_interval
        key['analysis_file_name'], key['units_object_id'] = \
            SortingID.store_sorting_nwb(key, sorting=sorting, sort_interval_list_name=sort_interval_list_name, 
                              sort_interval=sort_interval, metrics=metrics, unit_ids=accepted_units)

        # Insert entry to CuratedSpikeSorting table
        self.insert1(key)

        # Remove the non primary key entries.
        del key['units_object_id']
        del key['analysis_file_name']

        units_table = (CuratedSpikeSorting & key).fetch_nwb()[0]['units']

        # Add entries to CuratedSpikeSorting.Units table
        unit_key = key
        for unit_num, unit in units_table.iterrows():
            unit_key['unit_id'] = unit_num
            for property in unit.index:
                if property not in non_metric_fields:
                    unit_key[property] = unit[property]
            CuratedSpikeSorting.Unit.insert1(unit_key)

    def metrics_fields(self):
        """Returns a list of the metrics that are currently in the Units table
        """
        unit_fields = list(self.Unit().fetch(limit=1, as_dict=True)[0].keys())
        parent_fields = list(CuratedSpikeSorting.fetch(limit=1, as_dict=True)[0].keys())
        parent_fields.extend(['unit_id', 'label'])
        return [field for field in unit_fields if field not in parent_fields]

    def delete(self):
        """
        Extends the delete method of base class to implement permission checking
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

    def delete_extractors(self, key):
        """Delete directories with sorting and recording extractors that are no longer needed

        :param key: key to curated sortings where the extractors can be removed
        :type key: dict
        """
        # get a list of the files in the spike sorting storage directory
        dir_names = next(os.walk(os.environ['SPIKE_SORTING_STORAGE_DIR']))[1]
        # now retrieve a list of the currently used analysis nwb files
        analysis_file_names = (self & key).fetch('analysis_file_name')
        delete_list = []
        for dir in dir_names:
            if not dir in analysis_file_names:
                delete_list.append(dir)
                print(f'Adding {dir} to delete list')
        delete = input('Delete all listed directories (y/n)? ')
        if delete == 'y' or delete == 'Y':
            for dir in delete_list:
                shutil.rmtree(dir)
            return
        print('No files deleted')
    # def delete(self, key)
@schema
class SortingID(dj.Manual):
    definition = """
    # Table for holding the a sorting ID and and the sorting extractor object associated with a recording
    -> SpikeSortingRecording
    sorting_id: varchar(20)
    ---
    sorting_extractor_object: BLOB  # to retrieve with kachery
    """  
    @staticmethod
    def store_sorting_nwb(key, *, sorting, sort_interval_list_name,
                          sort_interval, metrics=None, unit_ids=None):
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
@schema
class SelectedUnitsParameters(dj.Manual):
    definition = """
    unit_inclusion_param_name: varchar(80) # the name of the list of thresholds for unit inclusion
    ---
    max_noise_overlap=1:        float   # noise overlap threshold (include below) 
    min_nn_isolation=-1:        float   # isolation score threshold (include above)
    max_isi_violation=100:      float   # ISI violation threshold
    min_firing_rate=0:          float   # minimum firing rate threshold
    max_firing_rate=100000:     float   # maximum fring rate thershold
    min_num_spikes=0:           int     # minimum total number of spikes
    exclude_label_list=NULL:    BLOB    # list of labels to EXCLUDE
    """
    
    def get_included_units(self, curated_sorting_key, unit_inclusion_key):
        """given a reference to a set of curated sorting units and a specific unit inclusion parameter list, returns 
        the units that should be included

        :param curated_sorting_key: key to entries in CuratedSpikeSorting.Unit table
        :type curated_sorting_key: dict
        :param unit_inclusion_key: key to a single unit inclusion parameter set
        :type unit_inclusion_key: dict
        """
        curated_sortings = (CuratedSpikeSorting() & curated_sorting_key).fetch()
        inclusion_key = (self & unit_inclusion_key).fetch1()
     
        units = (CuratedSpikeSorting().Unit() & curated_sortings).fetch()
        # get a list of the metrics in the units table
        metrics_list = CuratedSpikeSorting().metrics_fields()
        # create a list of the units to kepp. 
        #TODO: make this code more flexible
        keep = np.asarray([True] * len(units))
        if 'noise_overlap' in metrics_list and "max_noise_overlap" in inclusion_key:
            keep = np.logical_and(keep, units['noise_overlap'] <= inclusion_key["max_noise_overlap"])
        if 'nn_isolation' in metrics_list and "min_isolation" in inclusion_key:
            keep = np.logical_and(keep, units['nn_isolation'] >= inclusion_key["min_isolation"])
        if 'isi_violation' in metrics_list and "isi_violation" in inclusion_key:
            keep = np.logical_and(keep, units['isi_violation'] <= inclusion_key["isi_violation"])
        if 'firing_rate' in metrics_list and "firing_rate" in inclusion_key:
            keep = np.logical_and(keep, units['firing_rate'] >= inclusion_key["min_firing_rate"])
            keep = np.logical_and(keep, units['firing_rate'] <= inclusion_key["max_firing_rate"])
        if 'num_spikes' in metrics_list and "min_num_spikes" in inclusion_key:
            keep = np.logical_and(keep, units['num_spikes'] >= inclusion_key["min_num_spikes"])
        units = units[keep]
        #now exclude by label if it is specified
        if inclusion_key['exclude_label_list'] is not None:
            included_units = []
            for unit in units:
                labels = unit['label'].split(',')
                exclude = False
                for label in labels:
                    if label in inclusion_key['exclude_label_list']:
                        exclude = True
                if not exclude:
                    included_units.append(unit)   
            return included_units
        else:
            return units

@schema
class SelectedUnits(dj.Computed):
    definition = """
    -> CuratedSpikeSorting
    -> SelectedUnitsParameters    
    ---
    -> AnalysisNwbfile   # New analysis NWB file to hold unit info
    units_object_id: varchar(40)   # Object ID for the units in NWB file
    """
    