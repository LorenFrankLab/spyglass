import datajoint as dj
import tempfile

from .common_session import Session
from .common_region import BrainRegion
from .common_device import Probe
from .common_interval import IntervalList, SortIntervalList
from .common_filter import FirFilter

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

used = [Session, BrainRegion, Probe, IntervalList]

schema = dj.schema('common_ephys')


@schema
class ElectrodeGroup(dj.Imported):
    definition = """
    # grouping of electrodes corresponding to a physical probe
    -> Session
    electrode_group_name: varchar(80)  # electrode group name from NWBFile
    ---
    -> BrainRegion
    -> Probe
    description: varchar(80) # description of electrode group
    target_hemisphere: enum('Right','Left')
    """
    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            # fill in the groups
            egroups = list(nwbf.electrode_groups.keys())
            
            for eg_name in egroups:
                # for each electrode group, we get the group and add an electrode group entry.
                # as the ElectrodeGroup
                electrode_group = nwbf.get_electrode_group(eg_name)
                key['electrode_group_name'] = eg_name
                # check to see if the location is listed in the region.BrainRegion schema, and if not add it
                region_dict = dict()
                region_dict['region_name'] = electrode_group.location
                region_dict['subregion_name'] = ''
                region_dict['subsubregion_name'] = ''
                query = BrainRegion() & region_dict
                if len(query) == 0:
                    # this region isn't in the list, so add it
                    BrainRegion().insert1(region_dict)
                    query = BrainRegion() & region_dict
                    # we also need to get the region_id for this new region or find the right region_id
                region_id_dict = query.fetch1()
                key['region_id'] = region_id_dict['region_id']
                key['description'] = electrode_group.description
                # the following should probably be a function that returns the probe devices from the file
                probe_re = re.compile("probe")
                for d in nwbf.devices:
                    if probe_re.search(d):
                        if nwbf.devices[d] == electrode_group.device:
                            # this will match the entry in the device schema
                            key['probe_type'] = electrode_group.device.probe_type
                            break
                if 'probe_type' not in key:
                    key['probe_type'] = 'unknown-probe-type'
                self.insert1(key, skip_duplicates=True)

@schema
class Electrode(dj.Imported):
    definition = """
    -> ElectrodeGroup
    electrode_id: int               # the unique number for this electrode
    ---
    -> Probe.Electrode
    -> BrainRegion
    name='': varchar(80)           # unique label for each contact
    original_reference_electrode=-1: int # the configured reference electrode for this electrode 
    x=NULL: float                   # the x coordinate of the electrode position in the brain
    y=NULL: float                   # the y coordinate of the electrode position in the brain
    z=NULL: float                   # the z coordinate of the electrode position in the brain
    filtering: varchar(200)         # description of the signal filtering
    impedance=null: float                # electrode impedance
    bad_channel: enum("True","False")      # if electrode is 'good' or 'bad' as observed during recording
    x_warped=NULL: float                 # x coordinate of electrode position warped to common template brain
    y_warped=NULL: float                 # y coordinate of electrode position warped to common template brain
    z_warped=NULL: float                 # z coordinate of electrode position warped to common template brain
    contacts: varchar(80)           # label of electrode contacts used for a bipolar signal -- current workaround
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            # create the table of electrodes
            electrodes = nwbf.electrodes.to_dataframe()

            # Below it would be better to find the mapping between  nwbf.electrodes.colnames and the schema fields and
            # where possible, assign automatically. It would also help to be robust to missing fields and have them
            # assigned as empty if they don't exist in the nwb file in case people are not using our column names.

            for elect in electrodes.iterrows():
                key['electrode_group_name'] = elect[1].group_name
                key['electrode_id'] = elect[0]
                key['name'] = str(elect[0])
                key['probe_type'] = elect[1].group.device.probe_type
                key['probe_shank'] = elect[1].probe_shank
                key['probe_electrode'] = elect[1].probe_electrode 
                key['bad_channel'] = 'True' if elect[1].bad_channel else 'False'
                # look up the region
                region_dict = dict()
                region_dict['region_name'] = elect[1].group.location
                region_dict['subregion_name'] = ''
                region_dict['subsubregion_name'] = ''
                key['region_id'] = (BrainRegion() & region_dict).fetch1('region_id')
                key['x'] = elect[1].x
                key['y'] = elect[1].y
                key['z'] = elect[1].z
                key['x_warped'] = 0
                key['y_warped'] = 0
                key['z_warped'] = 0
                key['contacts'] = ''
                key['filtering'] = elect[1].filtering
                key['impedance'] = elect[1].imp
                try:
                    key['original_reference_electrode'] = elect[1].ref_elect
                except:
                    key['original_reference_electrode'] = -1
                self.insert1(key, skip_duplicates=True)


#TODO: organize a set of SortGroups into a LinkingGroup for linking across sorts

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
    -> IntervalList # the valid times for the raw data (excluding artifacts, etc. if desired)
    """

@schema 
class SpikeSorting(dj.Computed):
    definition = """
    -> SpikeSortingParameters
    ---
    ->AnalysisNwbfile
    units_object_id: varchar(40) # the object ID for the units for this sort group
    """

    def make(self, key):
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        # get the valid times. 
        # NOTE: we will sort independently between each entry in the valid times list
        sort_intervals = (SortIntervalList() & {'nwb_file_name' : key['nwb_file_name'],
                                        'sort_interval_list_name' : key['sort_interval_list_name']})\
                                            .fetch('sort_intervals')
        units = dict()
        # we will add an offset to the unit_id for each sort interval to avoid duplicating ids
        unit_id_offset = 0
        #interate through the arrays of valid times, sorting separately for each array
        for sort_interval in sort_intervals[0]:
            # get the indeces of the data to use. Note that spike_extractors has a time_to_frame function, 
            # but it seems to set the time of the first sample to 0, which will not match our intervals
            raw_data_obj = (Raw() & {'nwb_file_name' : key['nwb_file_name']}).fetch_nwb()[0]['raw']
            sort_indeces = np.searchsorted(raw_data_obj.timestamps, sort_interval)
            print(f'sample indeces: {sort_indeces}')

            # Use spike_interface to run the sorter on the selected sort group
            raw_data = se.NwbRecordingExtractor(Nwbfile.get_abs_path(key['nwb_file_name']), electrical_series_name='e-series')
            
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

            raw_data_epoch_referenced_subset = se.SubRecordingExtractor(raw_data_epoch_referenced, 
                                                                        channel_ids=electrode_ids)

            sort_parameters = (SpikeSorterParameters() & {'sorter_name': key['sorter_name'],
                                                        'parameter_set_name': key['parameter_set_name']}).fetch1()

            #TODO: Add artifact rejection

            print(f'Sorting {key}...')
            sort = si.sorters.run_mountainsort4(recording=raw_data_epoch_referenced_subset, 
                                                **sort_parameters['parameter_dict'], 
                                                grouping_property='group', 
                                                output_folder=os.getenv('SORTING_TEMP_DIR', None))
            # create a stack of labelled arrays of the sorted spike times
            timestamps = np.asarray(raw_data_obj.timestamps)
            unit_ids = sort.get_unit_ids()
            for unit_id in unit_ids:
                unit_spike_samples = sort.get_unit_spike_train(unit_id=unit_id)  
                #print(f'looking up times for unit {unit_id + unit_id_offset}: {len(unit_spike_samples)} spikes')
                units[unit_id+unit_id_offset] = timestamps[unit_spike_samples]
            unit_id_offset += np.max(unit_ids) + 1
        
        #Add the units to the Analysis file        
        key['units_object_id'] = AnalysisNwbfile().add_units(key['analysis_file_name'], units)
        self.insert1(key)
        # add an offset sufficient to avoid unit number overlap (+1 more to make it easy to find the space)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

@schema
class Raw(dj.Imported):
    definition = """
    # Raw voltage timeseries data, electricalSeries in NWB
    -> Session
    ---
    -> IntervalList
    raw_object_id: varchar(80)      # the NWB object ID for loading this object from the file
    sampling_rate: float                            # Sampling rate calculated from data, in Hz
    comments: varchar(80)
    description: varchar(80)
    """
    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            raw_interval_name = "raw data valid times"
            # get the acquisition object
            try:
                rawdata = nwbf.get_acquisition()
            except:
                print(f'WARNING: Unable to get aquisition object in: {nwb_file_abspath}')
                return
            print('Estimating sampling rate...')
            # NOTE: Only use first 1e6 timepoints to save time
            sampling_rate = estimate_sampling_rate(np.asarray(rawdata.timestamps[:1000000]), 1.5)
            print(f'Estimated sampling rate: {sampling_rate}')
            key['sampling_rate'] = sampling_rate
            # get the list of valid times given the specified sampling rate.
            interval_dict = dict()
            interval_dict['nwb_file_name'] = key['nwb_file_name']
            interval_dict['interval_list_name'] = raw_interval_name
            interval_dict['valid_times'] = get_valid_intervals(np.asarray(rawdata.timestamps), key['sampling_rate'],
                                                                  1.75, 0)
            IntervalList().insert1(interval_dict, skip_duplicates=True)

            # now insert each of the electrodes as an individual row, but with the same nwb_object_id
            key['raw_object_id'] = rawdata.object_id
            key['sampling_rate'] = sampling_rate
            print(f'Importing raw data: Estimated sampling rate:\t{key["sampling_rate"]} Hz')
            print(f'                    Number of valid intervals:\t{len(interval_dict["valid_times"])}')
            key['interval_list_name'] = raw_interval_name
            key['comments'] = rawdata.comments
            key['description'] = rawdata.description
            self.insert1(key, skip_duplicates='True')

    def nwb_object(self, key):
        # return the nwb_object; FIX: this should be replaced with a fetch call. Note that we're using the raw file
        # so we can modify the other one. 
        # NOTE: This leaves the file open, which means that it cannot be appended to. This should be fine normally
        nwb_file_name = key['nwb_file_name']

        # TO DO: This likely leaves the io object in place and the file open. Fix
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        io = pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r')
        nwbf = io.read()
        # get the object id
        raw_object_id = (self & {'nwb_file_name' : key['nwb_file_name']}).fetch1('raw_object_id')
        return nwbf.objects[raw_object_id]
    
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, 'nwb_file_abs_path'), *attrs, **kwargs)


@schema
class LFPSelection(dj.Manual):
    definition = """
     -> Session
     """

    class LFPElectrode(dj.Part):
        definition = """
        -> master
        -> Electrode
        """

    def set_lfp_electrodes(self, nwb_file_name, electrode_list):
        '''
        Removes all electrodes for the specified nwb file and then adds back the electrodes in the list
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of electrodes to be used for LFP
        :return:
        '''
        # remove the session and then recreate the session and Electrode list
        (LFPSelection() & {'nwb_file_name' : nwb_file_name}).delete()
        # check to see if the user allowed the deletion
        if len((LFPSelection() & {'nwb_file_name' : nwb_file_name}).fetch()) == 0:
            LFPSelection().insert1({'nwb_file_name' : nwb_file_name})

            # TO DO: do this in a better way
            all_electrodes = Electrode.fetch(as_dict=True)
            primary_key = Electrode.primary_key
            for e in all_electrodes:
                # create a dictionary so we can insert new elects
                if e['electrode_id'] in electrode_list:
                    lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                    LFPSelection().LFPElectrode.insert1(lfpelectdict, replace='True')

@schema
class LFP(dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> IntervalList         # the valid intervals for the data
    -> FirFilter                 # the filter used for the data
    -> AnalysisNwbfile      # the name of the nwb file with the lfp data
    lfp_object_id: varchar(80)  # the NWB object ID for loading this object from the file
    lfp_sampling_rate: float # the sampling rate, in HZ
    """

    def make(self, key):
       # get the NWB object with the data; FIX: change to fetch with additional infrastructure
        rawdata = Raw().nwb_object(key)
        sampling_rate, interval_list_name = (Raw() & key).fetch1('sampling_rate', 'interval_list_name')
        sampling_rate = int(np.round(sampling_rate))

        key['interval_list_name'] = interval_list_name
        valid_times = (IntervalList() & {'nwb_file_name': key['nwb_file_name'] ,  'interval_list_name': interval_list_name}).fetch1('valid_times')

        nwb_file_name = key['nwb_file_name']
        #target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (FirFilter() & {'filter_name' : 'LFP 0-400 Hz'} & {'filter_sampling_rate':
                                                                                  sampling_rate}).fetch(as_dict=True)

        # there should only be one filter that matches, so we take the first of the dictionaries
        key['filter_name'] = filter[0]['filter_name']
        key['filter_sampling_rate'] = filter[0]['filter_sampling_rate']

        filter_coeff = filter[0]['filter_coeff']
        if len(filter_coeff) == 0:
            print(f'Error in LFP: no filter found with data sampling rate of {sampling_rate}')
            return None
        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPSelection.LFPElectrode & key).fetch('KEY')
        electrode_id_list = list(k['electrode_id'] for k in electrode_keys)

        lfp_file_name = AnalysisNwbfile().create(key['nwb_file_name'])

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        # test:
        lfp_object_id = FirFilter().filter_data_nwb(lfp_file_abspath, rawdata.timestamps, rawdata.data,
                                    filter_coeff, valid_times, electrode_id_list, decimation)

        key['analysis_file_name'] = lfp_file_name
        key['lfp_object_id'] = lfp_object_id
        key['lfp_sampling_rate'] = sampling_rate // decimation
        self.insert1(key)
        
    def nwb_object(self, key):
        # return the nwb_object.
        lfp_file_name = (LFP() & {'nwb_file_name': key['nwb_file_name']}).fetch1('analysis_file_name')
        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        io = pynwb.NWBHDF5IO(path=lfp_file_abspath, mode='r')
        nwbf = io.read()
        # get the object id
        nwb_object_id = (self & {'analysis_file_name' : lfp_file_name}).fetch1('filtered_data_object_id')
        return nwbf.objects[nwb_object_id]

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

@schema
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFP
    -> FirFilter                 # the filter to use for the data    
    ---  
    -> IntervalList # the set of times to be filtered
    lfp_band_sampling_rate: int # the sampling rate for this band
    """

    class LFPBandElectrode(dj.Part):
        definition = """
        -> master
        -> LFPSelection.LFPElectrode # the LFP electrode to be filtered LFP
        reference_elect_id = -1: int # the reference electrode to use; -1 for no reference
        ---
        """

    def set_lfp_band_electrodes(self, nwb_file_name, electrode_list, filter_name, interval_list_name, reference_electrode_list, lfp_band_sampling_rate):
        '''
        Adds an entry for each electrode in the electrode_list with the specified filter, interval_list, and reference electrode.
        Also removes any entries that have the same filter, interval list and reference electrode but are not in the electrode_list.
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of LFP electrodes to be filtered
        :param filter_name: the name of the filter (from the FirFilter schema)
        :param interval_name: the name of the interval list (from the IntervalList schema)
        :param reference_electrode_list: A single electrode id corresponding to the reference to use for all electrodes or a list with one element per entry in the electrode_list
        :param lfp_band_sampling_rate: The output sampling rate to be used for the filtered data; must be an integer divisor of the LFP sampling rate
        :return: none
        '''
        # Error checks on parameters
        # electrode_list
        available_electrodes = (LFPSelection().LFPElectrode() & {'nwb_file_name' : nwb_file_name}).fetch('electrode_id')
        if not np.all(np.isin(electrode_list,available_electrodes)):
            raise ValueError('All elements in electrode_list must be valid electrode_ids in the LFPSelection table')
        #sampling rate
        lfp_sampling_rate = (LFP() & {'nwb_file_name' : nwb_file_name}).fetch1('lfp_sampling_rate')
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(f'lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer divisor of lfp samping rate {lfp_sampling_rate}')
        #filter 
        if not len((FirFilter() & {'filter_name' : filter_name, 'filter_sampling_rate' : lfp_sampling_rate}).fetch()):
            raise ValueError(f'filter {filter_name}, sampling rate {lfp_sampling_rate}is not in the FirFilter table')
        #interval_list
        if not len((IntervalList() & {'interval_name' : interval_list_name}).fetch()):
            raise ValueError(f'interval list {interval_list_name} is not in the IntervalList table; the list must be added before this function is called')
        # reference_electrode_list
        if len(reference_electrode_list) != 1 and len(reference_electrode_list) != len(electrode_list):
            raise ValueError(f'reference_electrode_list must contain either 1 or len(electrode_list) elements')
        # add a -1 element to the list to allow for the no reference option
        available_electrodes = np.append(available_electrodes, [-1])
        if not np.all(np.isin(reference_electrode_list,available_electrodes)):
            raise ValueError('All elements in reference_electrode_list must be valid electrode_ids in the LFPSelection table')
        
        # make a list of all the references
        ref_list = np.zeros((len(electrode_list),))
        ref_list[:] = reference_electrode_list       

        key = dict()
        key['nwb_file_name'] = nwb_file_name
        key['filter_name'] = filter_name
        key['filter_sampling_rate'] = lfp_sampling_rate
        key['interval_list_name'] = interval_list_name
        key['lfp_band_sampling_rate'] = lfp_sampling_rate // decimation
        # insert an entry into the main LFPBandSelectionTable
        self.insert1(key, skip_duplicates='True')

        #remove the keys that are not used for the LFPBandElectrode table
        key.pop('interval_list_name')
        key.pop('lfp_band_sampling_rate')
        #get all of the current entries and delete any that are not in the list
        elect_id, ref_id = (self.LFPBandElectrode() & key).fetch('electrode_id', 'reference_elect_id')
        for e, r in zip(elect_id, ref_id): 
            if not len(np.where((electrode_list == e) & (ref_list == r))[0]):
                key['electrode_id'] = e
                key['reference_elect_id'] = r
                (self.LFPBandElectrode() & key).delete()

        #iterate through all of the new elements and add them
        for e, r in zip(electrode_list, ref_list):
            key['electrode_id'] = e
            key['electrode_group_name'] = (Electrode & {'electrode_id' : e}).fetch1('electrode_group_name')
            key['reference_elect_id'] = r
            self.LFPBandElectrode().insert1(key, skip_duplicates='True')

     

@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFPBandSelection
    ---
    -> AnalysisNwbfile
    filtered_data_object_id: varchar(80)  # the NWB object ID for loading this object from the file
    """
    def make(self, key):
        # get the NWB object with the lfp data; FIX: change to fetch with additional infrastructure
        lfp_object = (LFP() & {'nwb_file_name' : key['nwb_file_name']}).fetch_nwb()[0]['lfp']
        
        # load all the data to speed filtering
        lfp_data = np.asarray(lfp_object.data, dtype=type(lfp_object.data[0][0]))
        lfp_timestamps = np.asarray(lfp_object.timestamps, dtype=type(lfp_object.timestamps[0]))

        #get the electrodes to be filtered and their references
        lfp_band_elect_id, lfp_band_ref_id = (LFPBandSelection().LFPBandElectrode() & key).fetch('electrode_id', 'reference_elect_id')

        # get the indeces of the electrodes to be filtered and the references     
        lfp_band_elect_index = get_electrode_indeces(lfp_object, lfp_band_elect_id)
        lfp_band_ref_index = get_electrode_indeces(lfp_object, lfp_band_ref_id)

        # subtract off the references for the selected channels  
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:,elect_index] = lfp_data[:,elect_index] - lfp_data[:,lfp_band_ref_index] 


        lfp_sampling_rate = (LFP() & {'nwb_file_name': key['nwb_file_name']}).fetch1('lfp_sampling_rate')
        interval_list_name, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1('interval_list_name', 'lfp_band_sampling_rate')
        valid_times = (IntervalList() & {'interval_list_name' : interval_list_name}).fetch1('valid_times')
        filter_name, filter_sampling_rate, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1('filter_name', 'filter_sampling_rate', 'lfp_band_sampling_rate')

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # get the LFP filter that matches the raw data
        filter = (FirFilter() & {'filter_name' : filter_name} & 
                                {'filter_sampling_rate': filter_sampling_rate}).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(f'Filter {filter_name} and sampling_rate {lfp_band_sampling_rate} does not exit in the FirFilter table')
  
        filter_coeff = filter[0]['filter_coeff']
        if len(filter_coeff) == 0:
            print(f'Error in LFPBand: no filter found with data sampling rate of {lfp_band_sampling_rate}')
            return None

        #create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
        lfp_band_file_abspath = AnalysisNwbfile().get_abs_path(lfp_band_file_name)
        # filter the data and write to an the nwb file
        filtered_data_object_id = FirFilter().filter_data_nwb(lfp_band_file_abspath, lfp_timestamps, lfp_data,
                                    filter_coeff, valid_times, lfp_band_elect_id, decimation)

        key['analysis_file_name'] = lfp_band_file_name
        key['filtered_data_object_id'] = filtered_data_object_id
        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)


