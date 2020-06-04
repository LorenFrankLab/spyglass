import datajoint as dj

import common_session
import common_region
import common_interval
import common_device
import common_filter
import spikeinterface as si
import pynwb
import re
import numpy as np
import json
import h5py as h5
import nwb_helper_fn as nh
import dj_helper_fn as dh



schema = dj.schema('common_ephys')


@schema
class ElectrodeConfig(dj.Manual):
    definition = """
    -> common_session.Session
    """

    class ElectrodeGroup(dj.Part):
        definition = """
        # grouping of electrodes corresponding to a physical probe
        -> master
        electrode_group_name: varchar(80)  # electrode group name from NWBFile
        ---
        -> common_region.BrainRegion
        -> common_device.Probe
        description: varchar(80) # description of electrode group
        target_hemisphere: enum('Right','Left')
        """

    class Electrode(dj.Part):
        definition = """
        -> master
        electrode_id: int               # the unique number for this electrode
        ---
        -> master.ElectrodeGroup
        -> common_device.Probe.Electrode
        -> common_region.BrainRegion
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

    def insert_from_nwb(self, nwb_file_name):
        '''
        Insert Electrode groups and electrodes from the NWB file.
        :param nwb_file_name: string - name of NWB file
        :return: None
        '''
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        # insert the session identifier (the name of the nwb file)
        self.insert1(key)
        # now open the NWB file and fill in the groups
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            return

        egroups = list(nwbf.electrode_groups.keys())
        eg_dict = dict()
        eg_dict['nwb_file_name'] = key['nwb_file_name']
        for eg_name in egroups:
            # for each electrode group, we get the group and add an electrode group entry.
            # as the ElectrodeGroup
            electrode_group = nwbf.get_electrode_group(eg_name)
            eg_dict['electrode_group_name'] = eg_name
            # check to see if the location is listed in the region.BrainRegion schema, and if not add it
            region_dict = dict()
            region_dict['region_name'] = electrode_group.location
            region_dict['subregion_name'] = ''
            region_dict['subsubregion_name'] = ''
            query = common_region.BrainRegion & region_dict
            if len(query) == 0:
                # this region isn't in the list, so add it
                common_region.BrainRegion.insert1(region_dict)
                query = common_region.BrainRegion & region_dict
                # we also need to get the region_id for this new region or find the right region_id
            region_id_dict = query.fetch1()
            eg_dict['region_id'] = region_id_dict['region_id']
            eg_dict['description'] = electrode_group.description
            # the following should probably be a function that returns the probe devices from the file
            probe_re = re.compile("probe")
            for d in nwbf.devices:
                if probe_re.search(d):
                    if nwbf.devices[d] == electrode_group.device:
                        # this will match the entry in the device schema
                        eg_dict['probe_type'] = electrode_group.device.probe_type
                        break
            ElectrodeConfig.ElectrodeGroup.insert1(eg_dict)

        # now create the table of electrodes
        elect_dict = dict()
        electrodes = nwbf.electrodes.to_dataframe()
        elect_dict['nwb_file_name'] = key['nwb_file_name']

        # Below it would be better to find the mapping between  nwbf.electrodes.colnames and the schema fields and
        # where possible, assign automatically. It would also help to be robust to missing fields and have them
        # assigned as empty if they don't exist in the nwb file in case people are not using our extensions.

        for elect in electrodes.iterrows():
            # not certain that the assignment below is correct; needs to be checked.
            elect_dict['electrode_id'] = elect[0]
            elect_dict['electrode_group_name'] = elect[1].group_name
            # FIX to get electrode information properly from input or NWB file. Label may not exist, so we should
            # check that and use and empty string if not.
            elect_dict['name'] = str(elect[0])
            elect_dict['probe_type'] = elect[1].group.device.probe_type
            elect_dict['probe_shank'] = elect[1].probe_shank
            # change below when extension name is changed and bug is fixed in fldatamigration
            elect_dict['probe_electrode'] = elect[1].probe_electrode % 128

            elect_dict['bad_channel'] = 'True' if elect[1].bad_channel else 'False'
            elect_dict['region_id'] = eg_dict['region_id']
            elect_dict['x'] = elect[1].x
            elect_dict['y'] = elect[1].y
            elect_dict['z'] = elect[1].z
            elect_dict['x_warped'] = 0
            elect_dict['y_warped'] = 0
            elect_dict['z_warped'] = 0
            elect_dict['contacts'] = ''
            elect_dict['filtering'] = elect[1].filtering
            elect_dict['impedance'] = elect[1].imp
            try:
                elect_dict['original_reference_electrode'] = elect[1].ref_elect
            except:
                elect_dict['original_reference_electrode'] = -1

            ElectrodeConfig.Electrode.insert1(elect_dict)
        # close the file
        io.close()

@schema
class ElectrodeSortingInfo(dj.Manual):
    definition = """
     -> ElectrodeConfig.Electrode
     ---
     sort_group=-1: int  # the number of the sorting group for this electrode
     reference_electrode=-1: int  # the electrode to use for reference. -1: no reference, -2: common median 
    """
    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def initialize(self, nwb_file_name):
        '''
        Initialize the entries for the specified nwb file with the default values
        :param nwb_file_name:
        :return: None
        '''
        # add entries with default values for all fields
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name': nwb_file_name} & {'bad_channel': 'False'}) \
            .fetch()
        electrode_sorting_info = electrodes[['nwb_file_name', 'electrode_id']]
        self.insert(electrode_sorting_info, replace="True")


    def set_group_by_shank(self, nwb_file_name):
        '''
        :param: nwb_file_name - the name of the NWB file whose electrodes should be put into sorting groups
        :return: None
        Assign groups to all non-bad channel electrodes based on their shank:
        Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a single group
        Electrodes from probes with multiple shanks (e.g. polymer probes) are placed in one group per shank
        '''
        # get the electrodes from this NWB file
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name' : nwb_file_name} & {'bad_channel' : 'False'}) \
            .fetch()
        # get the current sorting info
        elect_sort_info  = (ElectrodeSortingInfo() & {'nwb_file_name' : nwb_file_name}).fetch()
        if not len(elect_sort_info):
            print('Error in ElectrodeSortingInfo.set_group_by_shank: no sorting information found. Try initialize()')
        # if the current sorting info exists,
        # get a list of the electrode groups
        e_groups = np.unique(electrodes['electrode_group_name'])
        sort_group = 0
        for e_group in e_groups:
            # for each electrode group, get a list of the unique shank numbers
            shank_list = np.unique(electrodes['probe_shank'][electrodes['electrode_group_name'] == e_group])
            # get the indices of all electrodes in for this group / shank and set their sorting group
            for shank in shank_list:
                shank_elect = electrodes['electrode_id'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                        electrodes['probe_shank'] == shank)]
                 # create list of schema entries for insertion
                new_group = list()
                for elect in shank_elect:
                    new_group.append([elect, sort_group])
                # get the new replacement entries for the table
                self.insert(dh.replace(elect_sort_info, new_group, 'electrode_id', 'sort_group'),replace="True")
                sort_group += 1

    def set_group_by_electrode_group(self, nwb_file_name):
        '''
        :param: nwb_file_name - the name of the nwb whose electrodes should be put into sorting groups
        :return: None
        Assign groups to all non-bad channel electrodes based on their electrode group:
        '''
        # get the electrodes from this NWB file
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name': nwb_file_name} & {'bad_channel': 'False'}) \
            .fetch()
        e_groups = np.unique(electrodes['electrode_group_name'])
        # get the current sorting info
        elect_sort_info  = (ElectrodeSortingInfo() & {'nwb_file_name' : nwb_file_name}).fetch()
        sort_group = 0
        for e_group in e_groups:
            # get the indices of all electrodes in for this group / shank and set their sorting group
            eg_elect = electrodes['electrode_id'][electrodes['electrode_group_name'] == e_group]
            new_group = list()
            for elect in eg_elect:
                new_group.append([ elect, sort_group])
            self.insert(dh.replace(elect_sort_info, new_group, 'electrode_id', 'sort_group'), replace="True")
            sort_group += 1

    def set_reference_from_electrodes(self, nwb_file_name):
        '''
        Set the reference electrode from the original reference from the acquisition system
        :param: nwb_file_name - The name of the NWB file whose electrodes' references should be updated
        :return: Null
        '''
        # get the electrodes from this NWB file
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name': nwb_file_name} & {'bad_channel': 'False'}) \
            .fetch()
        # get the current reference info
        elect_ref_info  = (ElectrodeSortingInfo() & {'nwb_file_name' : nwb_file_name}).fetch()

        ref_elect = []
        for elect in electrodes:
            ref_elect.append([elect['electrode_id'], elect['original_reference_electrode']])
        self.insert(dh.replace(elect_ref_info, ref_elect, 'electrode_id', 'reference_electrode'), replace="True")

    def set_reference_from_list(self, nwb_file_name, elect_ref_list):
        '''
        Set the reference electrode from a list containing electrodes and reference electrodes
        :param: elect_ref_list - 2D array or list where each row is [electrode_id reference_electrode]
        :param: nwb_file_name - The name of the NWB file whose electrodes' references should be updated
        :return: Null
        '''
        # get the current reference info
        elect_ref_info = (ElectrodeSortingInfo() & {'nwb_file_name': nwb_file_name}).fetch()
        self.insert(dh.replace(elect_ref_info, elect_ref_list, 'electrode_id', 'reference_electrode'), replace="True")

    def write_prb(self, nwb_file_name, prb_file_name):
        '''
        Writes a prb file containing informaiton on the sorting groups and their geometry for use with the
        SpikeInterface package. See the SpikeInterface documentation for details on prb file format.
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
        # Get the list of electrodes and their sort groups
        sort_electrodes = (ElectrodeSortingInfo() & {'nwb_file_name': nwb_file_name}).fetch()
        sort_groups = np.unique(sort_electrodes['sort_group']).tolist()
        max_group = int(np.max(np.asarray(sort_groups)))
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name': nwb_file_name}).fetch()
        # get the relative x and y positions of each electrode in each probe.

        for sort_group in sort_groups:
            channel_group[sort_group] = dict()
            channel_group[sort_group]['channels'] = sort_electrodes['electrode_id'][sort_electrodes['sort_group'] ==
                                                                             sort_group].tolist()
            geometry = list()
            label = list()
            for electrode_id in channel_group[sort_group]['channels']:
                # get the relative x and y locations of this channel from the probe table
                probe_electrode = int(electrodes['probe_electrode'][electrodes['electrode_id'] == electrode_id])
                rel_x, rel_y = (common_device.Probe.Electrode() & {'probe_electrode' : probe_electrode}).fetch(
                    'rel_x','rel_y')
                rel_x = float(rel_x)
                rel_y = float(rel_y)
                geometry.append([rel_x, rel_y])
                label.append(str(electrode_id))
            channel_group[sort_group]['geometry'] = geometry
            channel_group[sort_group]['label'] = label
        # write the prf file in their odd format
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
        # use the spikeinterface function to write the prb file

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
        self.delete()
        sorters = si.sorters.available_sorters()
        for sorter in sorters:
            self.insert1({'sorter_name' : sorter}, replace="True")

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
                self.insert1(sort_param_dict, replace="True")
            else:
                print(f'Error in SpikeSorterParameter: sorter {sorter} not in SpikeSorter schema')
                continue

@schema
class SpikeSort(dj.Manual):
    definition = """
    -> common_session.Session
    -> SpikeSorterParameters 
    ---
    nwb_object_id: varchar(256) # the NWB object holding information about this sort. 
    """



@schema
class Units(dj.Imported):
    definition = """
    -> common_session.Session
    unit_id: int # unique identifier for this unit in this session
    ---
    -> ElectrodeConfig.ElectrodeGroup
    -> common_interval.IntervalList
    cluster_name: varchar(80)   # the name for this cluster (e.g. t5 c4)
    nwb_object_id: int      # the object_id for the spikes; once the object is loaded, use obj.get_unit_spike_times
    """

    # should this use the file pointer instead?
    def make(self, key):
        # get the NWB file name from this session
        nwb_file_name = key['nwb_file_name']

        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            return

        interval_list_dict = dict()
        interval_list_dict['nwb_file_name'] = nwb_file_name
        units = nwbf.units.to_dataframe()
        for unum in range(len(units)):
            # for each unit we first need to an an interval list for this unit
            interval_list_dict['interval_name'] = 'unit {} interval list'.format(
                unum)
            interval_list_dict['valid_times'] = np.asarray(
                units.iloc[unum]['obs_intervals'])
            common_interval.IntervalList.insert1(
                interval_list_dict, skip_duplicates=True)
            try:
                key['unit_id'] = units.iloc[unum]['id']
            except:
                key['unit_id'] = unum

            egroup = units.iloc[unum]['electrode_group']
            key['electrode_group_name'] = egroup.name
            key['interval_name'] = interval_list_dict['interval_name']
            key['cluster_name'] = units.iloc[unum]['cluster_name']
            #key['spike_times'] = np.asarray(units.iloc[unum]['spike_times'])
            key['nwb_object_id'] = -1  # FIX
            self.insert1(key)


@schema
class Raw(dj.Manual):
    definition = """
    # Raw voltage timeseries data, electricalSeries in NWB
    -> common_session.Session
    ---
    -> common_interval.IntervalList
    nwb_object_id: varchar(80)  # the NWB object ID for loading this object from the file
    sampling_rate: float                            # Sampling rate calculated from data, in Hz
    comments: varchar(80)
    description: varchar(80)
    """
    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def insert_from_nwb(self, nwb_file_name):
        '''
        Insert the nwb object and interval information for the raw electrophysiological data in the NWB file.
        Assumes the data are returned by nwbf.get_acquisition()
        :param nwb_file_name:
        :return:
        '''
        # get the NWB file name from this session
        key = dict()
        key['nwb_file_name'] = nwb_file_name

        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
            raw_interval_name = "raw data valid times"
            # get the acquisition object
            rawdata = nwbf.get_acquisition()
            key['sampling_rate'] = nh.estimate_sampling_rate(np.asarray(rawdata.timestamps), 1.5)
            # get the list of valid times given the specified sampling rate.
            interval_dict = dict()
            interval_dict['nwb_file_name'] = key['nwb_file_name']
            interval_dict['interval_name'] = raw_interval_name
            interval_dict['valid_times'] = nh.get_valid_intervals(np.asarray(rawdata.timestamps), key['sampling_rate'],
                                                                  1.75, 0)
            common_interval.IntervalList.insert1(interval_dict, skip_duplicates="True")

            # now insert each of the electrodes as an individual row, but with the same nwb_object_id
            key['nwb_object_id'] = rawdata.object_id
            key['sampling_rate'] = nh.estimate_sampling_rate(np.asarray(rawdata.timestamps), 1.5)
            print(f'Importing raw data: Estimated sampling rate:\t{key["sampling_rate"]} Hz')
            print(f'                    Number of valid intervals:\t{len(interval_dict["valid_times"])}')
            key['interval_name'] = raw_interval_name
            key['comments'] = rawdata.comments
            key['description'] = rawdata.description
            self.insert1(key, skip_duplicates='True')
        except:
            print(f'Error: Raw data import from nwbfile {nwb_file_name} failed; file may not exist.')
        finally:
            io.close()

    def nwb_object(self, key):
        # return the nwb_object; FIX: this should be replaced with a fetch call. Note that we're using the raw file
        # so we can modify the other one.
        nwb_file_name = key['nwb_file_name']

        # TO DO: This likely leaves the io object in place and the file open. Fix
        io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
        nwbf = io.read()
        # get the object id
        nwb_object_id = (self & {'nwb_file_name' : key['nwb_file_name']}).fetch1('nwb_object_id')
        return nwbf.objects[nwb_object_id]


@schema
class LFPElectrode(dj.Manual):
    definition = """
     -> ElectrodeConfig.Electrode
     """

    def set_lfp_elect(self, nwb_file_name, electrode_list):
        '''
        Removes all electrodes for the specified nwb file and then adds back the electrodes in the list
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of electrodes to be used for LFP
        :return:
        '''
        # remove the electrodes
        (LFPElectrode() & {'nwb_file_name' : nwb_file_name}).delete()
        # TO DO: do this in a better way
        all_electrodes = ElectrodeConfig.Electrode.fetch(as_dict=True)
        primary_key = ElectrodeConfig.Electrode.primary_key
        for e in all_electrodes:
            # create a dictionary so we can insert new elects
            if e['electrode_id'] in electrode_list:
                lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                self.insert1(lfpelectdict, replace='True')


@schema
class LFP(dj.Computed):
    definition = """
    -> LFPElectrode                         # the LFP electrodes selected
    ---
    -> Raw                                  # the Raw data this LFP is computed from
    -> common_session.LinkedNwbfile    # the linked file the LFP data is stored in
    -> common_interval.IntervalList         # the valid intervals for the data
    -> common_filter.FirFilter                 # the filter used for the data
    nwb_object_id: varchar(80)  # the NWB object ID for loading this object from the linked file
    sampling_rate: float # the sampling rate, in HZ
    """
    @property
    def key_source(self):
        # get a list of the sessions for which we want to compute the LFP
        return common_session.Session()

    def make(self, key):
       # get the NWB object with the data; FIX: change to fetch with additional infrastructure
        rawdata = Raw().nwb_object(key)
        sampling_rate, interval_name = (Raw() & key).fetch1('sampling_rate', 'interval_name')
        key['interval_name'] = interval_name
        sampling_rate = int(np.round(sampling_rate))
        key['sampling_rate'] = sampling_rate


        # TEMPORARY HARD CODED FILTERED DATA PATH
        filtered_data_path = '/data/nwb_builder_test_data/filtered_data/'
        nwb_file_name = key['nwb_file_name']
        #target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        valid_times = (common_interval.IntervalList() & {'nwb_file_name': key['nwb_file_name'] ,
                                                          'interval_name': interval_name}).fetch1('valid_times')
        # get the LFP filter that matches the raw data
        filter = (common_filter.FirFilter() & {'filter_name' : 'LFP 0-400 Hz'} & {'filter_sampling_rate':
                                                                                  sampling_rate}).fetch(as_dict=True)

        # there should only be one filter that matches, so we take the first of the dictionaries
        key['filter_name'] = filter[0]['filter_name']
        key['filter_sampling_rate'] = filter[0]['filter_sampling_rate']

        filter_coeff = filter[0]['filter_coeff']
        if len(filter_coeff) == 0:
            print(f'Error in LFP: no filter found with data sampling rate of {sampling_rate}')
            return None


        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPElectrode & key).fetch('KEY')
        electrode_id_list = list(k['electrode_id'] for k in electrode_keys)

        #TO DO: go back to filter_data_nwb when appending multiple times to NWB files is fixed.

        """
        nwb_out_file_name = common_session.LinkedNwbfile().create(key['nwb_file_name'])
        print(f'output file name {nwb_out_file_name}')

        common_filter.FirFilter().filter_data_nwb(nwb_out_file_name, rawdata.timestamps, rawdata.data,
                                                  filter_coeff, valid_times, electrode_id_list, decimation)
        """
        h5file_name = common_filter.FirFilter().filter_data_hdf5(filtered_data_path, rawdata.timestamps,
                                                                    rawdata.data, filter_coeff, valid_times,
                                                                               electrode_id_list, decimation)

        # create a linked NWB file with a new electrical series and link these new data to it. This is TEMPORARY
        linked_file_name = common_session.LinkedNwbfile().get_name_without_create(nwb_file_name)

        key['linked_file_name'] = linked_file_name
        key['linked_file_location'] = linked_file_name

        io_in = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
        nwbf = io_in.read()
        nwbf_out = nwbf.copy()

       #FIX to be indeces into electrodes, not electrode_ids
        electrode_table_region = nwbf_out.create_electrode_table_region(electrode_id_list,
                                                                        'filtered electrode table')
        print('past append')

        # open the hdf5 file and get the datasets from it
        with h5.File(h5file_name, 'r') as infile:
            filtered_data = infile.get('filtered_data')
            timestamps = infile.get('timestamps')

            lfp_description = f'LFP {filter[0]["filter_low_pass"]} Hz to {filter[0]["filter_high_pass"]} Hz '
            print(f'writing new NWB file {linked_file_name}')
            with pynwb.NWBHDF5IO(linked_file_name, mode='a', manager=io_in.manager) as io:
                es = pynwb.ecephys.ElectricalSeries('LFP', filtered_data, electrode_table_region,
                                                    timestamps=timestamps, description=lfp_description)
                nwbf_out.add_analysis(es)
                io.write(nwbf_out)
                key['nwb_object_id'] = es.object_id

            io_in.close()
        # loop through all of the electrodes and insert a row for each one
        for electrode in electrode_id_list:
            key['electrode_id'] = electrode
            print(key)
            self.insert1(key)


@schema
class LFPBandElectrode(dj.Manual):
    definition = """
    -> LFPElectrode
    ---
    reference_electrode=-1: int     # The reference electrode to use. -1 for none
    -> common_filter.FirFilter # the filter to be used for this LFP Band 
    use_for_band = 'False': enum('True', 'False') # True if this electrode should be filtered in this band
    """

@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFPBandElectrode
    ---
    -> LFP
    -> common_interval.IntervalList
    nwb_object_id: varchar(80)  # the NWB object ID for loading this object from the file
    sampling_rate: float # the sampling rate, in HZ
    """


@schema
class DecompSeries(dj.Computed):
    definition = """
    # Raw power timeseries data
    -> common_session.Session
    -> LFP
    ---
    -> common_interval.IntervalList
    nwb_object_id: varchar(255)  # the NWB object ID for loading this object from the file
    sampling_rate: float                                # Sampling rate, in Hz
    metric: enum("phase","amplitude","power")  # Metric represented in data
    comments: varchar(80)
    description: varchar(80)
    """
