import datajoint as dj
import tempfile

from .common_session import Session
from .common_region import BrainRegion
from .common_device import Probe
from .common_interval import IntervalList, SortIntervalList, interval_list_intersect, interval_list_excludes_ind
from .common_ephys import Raw, Electrode, ElectrodeGroup

import spikeinterface as si
import spikeextractors as se
import spiketoolkit as st
import pynwb
import re
import os
import numpy as np
import scipy.signal as signal
import json
import h5py as h5
from tempfile import NamedTemporaryFile
from .common_nwbfile import Nwbfile, AnalysisNwbfile
from .nwb_helper_fn import get_valid_intervals, estimate_sampling_rate, get_electrode_indeces
from .dj_helper_fn import dj_replace, fetch_nwb

used = [Session, BrainRegion, Probe, IntervalList, Raw]

schema = dj.schema('common_spikesorting')
@schema 
class SortGroup(dj.Manual):
    definition = """
    -> Session
    sort_group_id: int  # identifier for a group of electrodes
    ---
    sort_reference_electrode_id=-1: int  # the electrode to use for reference. -1: no reference, -2: common median 
    """
    class SortGroupElectrode(dj.Part):
        definition = """
        -> master
        -> Electrode
        """
    
    def set_group_by_shank(self, nwb_file_name):
        '''
        :param: nwb_file_name - the name of the NWB file whose electrodes should be put into sorting groups
        :return: None
        Assign groups to all non-bad channel electrodes based on their shank:
        Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a single group
        Electrodes from probes with multiple shanks (e.g. polymer probes) are placed in one group per shank
        '''
        # delete any current groups
        (SortGroup() & {'nwb_file_name' : nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (Electrode() & {'nwb_file_name' : nwb_file_name} & {'bad_channel' : 'False'}).fetch()
        e_groups = np.unique(electrodes['electrode_group_name'])
        sort_group = 0
        sg_key = dict()
        sge_key = dict()
        sg_key['nwb_file_name'] = sge_key['nwb_file_name'] = nwb_file_name
        for e_group in e_groups:
            # for each electrode group, get a list of the unique shank numbers
            shank_list = np.unique(electrodes['probe_shank'][electrodes['electrode_group_name'] == e_group])
            sge_key['electrode_group_name'] = e_group
            # get the indices of all electrodes in for this group / shank and set their sorting group
            for shank in shank_list:
                sg_key['sort_group_id'] = sge_key['sort_group_id'] = sort_group
                shank_elect_ref = electrodes['original_reference_electrode'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                    electrodes['probe_shank'] == shank)]                               
                if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                    sg_key['sort_reference_electrode_id'] = shank_elect_ref[0] 
                else: 
                    ValueError(f'Error in electrode group {e_group}: reference electrodes are not all the same')  
                self.insert1(sg_key)                   

                shank_elect = electrodes['electrode_id'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                        electrodes['probe_shank'] == shank)]
                for elect in shank_elect:
                    sge_key['electrode_id'] = elect
                    self.SortGroupElectrode().insert1(sge_key)
                sort_group += 1

    def set_group_by_electrode_group(self, nwb_file_name):
        '''
        :param: nwb_file_name - the name of the nwb whose electrodes should be put into sorting groups
        :return: None
        Assign groups to all non-bad channel electrodes based on their electrode group and sets the reference for each group 
        to the reference for the first channel of the group.
        '''
        # delete any current groups
        (SortGroup() & {'nwb_file_name' : nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (Electrode() & {'nwb_file_name': nwb_file_name} & {'bad_channel': 'False'}).fetch()
        e_groups = np.unique(electrodes['electrode_group_name'])
        sg_key = dict()
        sge_key = dict()
        sg_key['nwb_file_name'] = sge_key['nwb_file_name'] = nwb_file_name
        sort_group = 0
        for e_group in e_groups:
            sge_key['electrode_group_name'] = e_group
            sg_key['sort_group_id'] = sge_key['sort_group_id'] = sort_group
            # get the list of references and make sure they are all the same     
            shank_elect_ref = electrodes['original_reference_electrode'][electrodes['electrode_group_name'] == e_group]                                                           
            if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                sg_key['sort_reference_electrode_id'] = shank_elect_ref[0] 
            else: 
                ValueError(f'Error in electrode group {e_group}: reference electrodes are not all the same')  
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
        sort_group_list = (SortGroup() & key).fetch('sort_group_id')
        for sort_group in sort_group_list:
            key['sort_group_id'] = sort_group
            self.SortGroupElectrode().insert(dj_replace(sort_group_list, sort_group_ref_list, 
                                             'sort_group_id', 'sort_reference_electrode_id'), 
                                             replace="True")
       
    def write_prb(self, sort_group_id, nwb_file_name, prb_file_name):
        '''
        Writes a prb file containing informaiton on the specified sort group and it's geometry for use with the
        SpikeInterface package. See the SpikeInterface documentation for details on prb file format.
        :param sort_group_id: the id of the sort group
        :param nwb_file_name: the name of the nwb file for the session you wish to use
        :param prb_file_name: the name of the output prb file
        :return: None
        '''
        # try to open the output file
        try:
            prbf = open(prb_file_name, 'w')
        except:
            print(f'Error opening prb file {prb_file_name}')
            return

        # create the channel_groups dictiorary
        channel_group = dict()
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        sort_group_list = (SortGroup() & key).fetch('sort_group_id')
        max_group = int(np.max(np.asarray(sort_group_list)))
        electrodes = (Electrode() & key).fetch()

        key['sort_group_id'] = sort_group_id
        sort_group_electrodes = (SortGroup.SortGroupElectrode() & key).fetch()
        channel_group[sort_group_id] = dict()
        channel_group[sort_group_id]['channels'] = sort_group_electrodes['electrode_id'].tolist()
        geometry = list()
        label = list()
        for electrode_id in channel_group[sort_group_id]['channels']:
            # get the relative x and y locations of this channel from the probe table
            probe_electrode = int(electrodes['probe_electrode'][electrodes['electrode_id'] == electrode_id])
            rel_x, rel_y = (Probe().Electrode() & {'probe_electrode' : probe_electrode}).fetch(
                'rel_x','rel_y')
            rel_x = float(rel_x)
            rel_y = float(rel_y)
            geometry.append([rel_x, rel_y])
            label.append(str(electrode_id))
        channel_group[sort_group_id]['geometry'] = geometry
        channel_group[sort_group_id]['label'] = label
        # write the prf file in their odd format. Note that we only have one group, but the code below works for multiple groups
        prbf.write('channel_groups = {\n')
        for group in channel_group.keys():
            prbf.write(f'    {int(group)}:\n')
            prbf.write('        {\n')
            for field in channel_group[group]:
                prbf.write("          '{}': ".format(field))
                prbf.write(json.dumps(channel_group[group][field]) + ',\n')
            if int(group) != max_group:
                prbf.write('        },\n')
            else:
                prbf.write('        }\n')
        prbf.write('    }\n')
        prbf.close()

@schema
class SpikeSorter(dj.Manual):
    definition = """
    sorter_name: varchar(80) # the name of the spike sorting algorithm
    """
    def insert_from_spikeinterface(self):
        '''
        Add each of the sorters from spikeinterface.sorters 
        :return: None
        '''
        sorters = si.sorters.available_sorters()
        for sorter in sorters:
            self.insert1({'sorter_name' : sorter}, skip_duplicates="True")

@schema
class SpikeSorterParameters(dj.Manual):
    definition = """
    -> SpikeSorter 
    parameter_set_name: varchar(80) # label for this set of parameters
    ---
    parameter_dict: blob # dictionary of parameter names and values
    """

    def insert_from_spikeinterface(self):
        '''
        Add each of the default parameter dictionaries from spikeinterface.sorters
        :return: None
        '''
        sorters = si.sorters.available_sorters()
        # check to see that the sorter is listed in the SpikeSorter schema
        sort_param_dict = dict()
        sort_param_dict['parameter_set_name'] = 'default'
        for sorter in sorters:
            if len((SpikeSorter() & {'sorter_name' : sorter}).fetch()):
                sort_param_dict['sorter_name'] = sorter
                sort_param_dict['parameter_dict'] = si.sorters.get_default_params(sorter)
                self.insert1(sort_param_dict, skip_duplicates="True")
            else:
                print(f'Error in SpikeSorterParameter: sorter {sorter} not in SpikeSorter schema')
                continue

# Note: Unit and SpikeSorting need to be developed further and made compatible with spikeinterface

@schema
class SpikeSortingParameters(dj.Manual):
    definition = """
    -> SortGroup
    -> SpikeSorterParameters 
    -> SortIntervalList # the time intervals to be used for sorting
    ---
    -> IntervalList # the valid times for the raw data (excluding artifacts, etc. if desired)
    """

@schema 
class SpikeSorting(dj.Computed):
    definition = """
    -> SpikeSortingParameters
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40) # the object ID for the units for this sort group
    units_waveforms_object_id : varchar(40) # the object ID for the unit waveforms
    """

    def make(self, key):
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        # get the valid times. 
        # NOTE: we will sort independently between each entry in the valid times list
        sort_intervals =  (SortIntervalList() & {'nwb_file_name' : key['nwb_file_name'],
                                        'sort_interval_list_name' : key['sort_interval_list_name']})\
                                            .fetch1('sort_intervals')
        interval_list_name = (SpikeSortingParameters() & key).fetch1('interval_list_name')
        valid_times =  (IntervalList() & {'nwb_file_name' : key['nwb_file_name'],
                                        'interval_list_name' : interval_list_name})\
                                            .fetch('valid_times')[0]   
        raw_data_obj = (Raw() & {'nwb_file_name' : key['nwb_file_name']}).fetch_nwb()[0]['raw']
        timestamps = np.asarray(raw_data_obj.timestamps)
        sampling_rate = estimate_sampling_rate(timestamps[0:100000], 1.5)

        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()
        units_templates = dict()
        units_waveforms = dict()
        # we will add an offset to the unit_id for each sort interval to avoid duplicating ids
        unit_id_offset = 0
        #interate through the arrays of sort intervals, sorting each interval separately
        for sort_interval in sort_intervals:
               # Get the list of valid times for this sort interval
            recording_extractor, sort_interval_valid_times = self.get_recording_extractor(key, sort_interval)
            sort_parameters = (SpikeSorterParameters() & {'sorter_name': key['sorter_name'],
                                                        'parameter_set_name': key['parameter_set_name']}).fetch1()
            # get a name for the recording extractor for this sort interval
            recording_extractor_path = os.path.join(os.environ['SPIKE_SORTING_STORAGE_DIR'], 
                                                    key['analysis_file_name'], '_', str(sort_interval))
            recording_extractor_cached = se.CacheRecordingExtractor(recording_extractor, save_path=recording_extractor_path)
            print(f'Sorting {key}...')
            sort = si.sorters.run_mountainsort4(recording=recording_extractor_cached, 
                                                **sort_parameters['parameter_dict'], 
                                                grouping_property='group', 
                                                output_folder=os.getenv('SORTING_TEMP_DIR', None))
            # create a stack of labelled arrays of the sorted spike times
            timestamps = np.asarray(raw_data_obj.timestamps)
            unit_ids = sort.get_unit_ids()
            # get the waveforms; we may want to specifiy these parameters more flexibly in the future
            waveform_params = st.postprocessing.get_waveforms_params()
            print(sort_parameters)
            waveform_params['grouping_property'] = 'group'
            # set the window to half of the clip size before and half after
            waveform_params['ms_before'] = sort_parameters['parameter_dict']['clip_size'] / sampling_rate * 1000 / 2
            waveform_params['ms_after'] = waveform_params['ms_before'] 
            waveform_params['max_spikes_per_unit'] = 1000
            waveform_params['dtype'] = 'i2'
            #template_params['n_jobs'] = 7
            waveform_params['verbose'] = False
            #print(f'template_params: {template_params}')
            templates = st.postprocessing.get_unit_templates(recording_extractor_cached, sort, **waveform_params)
            # for the waveforms themselves we only need to change the max_spikes_per_unit:
            waveform_params['max_spikes_per_unit'] = 1e100
            waveforms = st.postprocessing.get_unit_waveforms(recording_extractor_cached, sort, unit_ids, **waveform_params)

            for index, unit_id in enumerate(unit_ids):
                current_index = unit_id + unit_id_offset
                unit_spike_samples = sort.get_unit_spike_train(unit_id=unit_id)  
                #print(f'template for {unit_id}: {unit_templates[unit_id]} ')
                units[current_index] = timestamps[unit_spike_samples]
                # the templates are zero based, so we have to use the index here. 
                units_templates[current_index] = templates[index]
                units_waveforms[current_index] = waveforms[index]
                units_valid_times[current_index] = sort_interval_valid_times
                units_sort_interval[current_index] = [sort_interval]
            if len(unit_ids) > 0:
                unit_id_offset += np.max(unit_ids) + 1
        
        #Add the units to the Analysis file       
        # TODO: consider replacing with spikeinterface call if possible 
        units_object_id, units_waveforms_object_id = AnalysisNwbfile().add_units(key['analysis_file_name'], units, units_templates, units_valid_times,
                                                              units_sort_interval, units_waveforms=units_waveforms)
        key['units_object_id'] = units_object_id
        key['units_waveforms_object_id'] = units_waveforms_object_id
        self.insert1(key)
        # add an offset sufficient to avoid unit number overlap (+1 more to make it easy to find the space)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def get_recording_extractor(self, key, sort_interval):
        """Given a key containing the key fields for a SpikeSorting schema, and the interval to be sorted,
         returns the recording extractor object (see the spikeinterface package for details)

        :param key: key to SpikeSorting schema
        :type key: dict
        :param sort_interval: [start_time, end_time]
        :type sort_interval: 1D array with the start and end times for this sort
        :return: (recording_extractor, sort_interval_valid_times)
        :rtype: tuple with spikeextractor recording extractor object and valid times list
        """
        interval_list_name = (SpikeSortingParameters() & key).fetch1('interval_list_name')
        valid_times =  (IntervalList() & {'nwb_file_name' : key['nwb_file_name'],
                                        'interval_list_name' : interval_list_name})\
                                            .fetch('valid_times')[0]  
        sort_interval_valid_times = interval_list_intersect(np.array([sort_interval]), valid_times)
                                 
        raw_data_obj = (Raw() & {'nwb_file_name' : key['nwb_file_name']}).fetch_nwb()[0]['raw']
        # get the indeces of the data to use. Note that spike_extractors has a time_to_frame function, 
        # but it seems to set the time of the first sample to 0, which will not match our intervals
        timestamps = np.asarray(raw_data_obj.timestamps)
        sort_indeces = np.searchsorted(timestamps, np.ravel(sort_interval))
        #print(f'sample indeces: {sort_indeces}')

        # Use spike_interface to run the sorter on the selected sort group
        raw_data = se.NwbRecordingExtractor(Nwbfile.get_abs_path(key['nwb_file_name']), electrical_series_name='e-series')
        
        # Blank out non-valid times. 
        exclude_inds = interval_list_excludes_ind(sort_interval_valid_times, timestamps)
        exclude_inds = exclude_inds[exclude_inds <= sort_indeces[-1]]
        # TODO: add a blanking function to the preprocessing module 
        raw_data = st.preprocessing.remove_artifacts(raw_data, exclude_inds, ms_before=0.1, ms_after=0.1)

        # create a group id within spikeinterface for the specified electodes
        electrode_ids = (SortGroup.SortGroupElectrode() & {'nwb_file_name' : key['nwb_file_name'], 
                                                        'sort_group_id' : key['sort_group_id']}).fetch('electrode_id')
        raw_data.set_channel_groups([key['sort_group_id']]*len(electrode_ids), channel_ids=electrode_ids)
        
        raw_data.add_epoch(key['sort_interval_list_name'], sort_indeces[0], sort_indeces[1])
        # restrict the raw data to the specific samples
        raw_data_epoch = raw_data.get_epoch(key['sort_interval_list_name'])
        
        # get the reference for this sort group
        sort_reference_electrode_id = (SortGroup() & {'nwb_file_name' : key['nwb_file_name'], 
                                                    'sort_group_id' : key['sort_group_id']}
                                                    ).fetch('sort_reference_electrode_id')           
        if sort_reference_electrode_id >= 0:
            raw_data_epoch_referenced = st.preprocessing.common_reference(raw_data_epoch, reference='single',
                                                            groups=[key['sort_group_id']], ref_channels=sort_reference_electrode_id)
        elif sort_reference_electrode_id == -2:
            raw_data_epoch_referenced = st.preprocessing.common_reference(raw_data, reference='median')
        else:
            raw_data_epoch_referenced = raw_data_epoch

        # create a temporary file for the probe with a .prb extension and write out the channel locations in the prb file
        with tempfile.TemporaryDirectory() as tmp_dir:
            prb_file_name = os.path.join(tmp_dir, 'sortgroup.prb')
            SortGroup().write_prb(key['sort_group_id'], key['nwb_file_name'], prb_file_name)
            # add the probe geometry to the raw_data recording
            raw_data_epoch_referenced.load_probe_file(prb_file_name)

        return se.SubRecordingExtractor(raw_data_epoch_referenced,channel_ids=electrode_ids), sort_interval_valid_times

    def get_sorting_extractor(self, key, sort_interval):
        #TODO: replace with spikeinterface call if possible
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
        
        # TODO: do something more efficient here; note that searching for maching sort_intervals within pandas doesn't seem to work
        for index, unit in units.iterrows():
            if np.ndarray.all(np.ravel(unit['sort_interval']) == sort_interval):
                unit_timestamps.extend(unit['spike_times'])
                unit_labels.extend([index]*len(unit['spike_times']))

        output=se.NumpySortingExtractor()
        output.set_times_labels(times=np.asarray(unit_timestamps),labels=np.asarray(unit_labels))
        return output

