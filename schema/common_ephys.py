import datajoint as dj

import common_session
import common_region
import common_interval
import common_device
import common_filter
import pynwb
import re
import numpy as np
import json
import nwb_helper_fn as nh

schema = dj.schema('common_ephys')


@schema
class ElectrodeConfig(dj.Imported):
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
        -> master.ElectrodeGroup
        electrode_id: int               # the unique number for this electrode
        ---
        -> common_device.Probe.Electrode
        -> common_region.BrainRegion
        name='': varchar(80)           # unique label for each contact
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
            elect_dict['electrode_group_name'] = elect[1].group_name
            # not certain that the assignment below is correct; needs to be checked.
            elect_dict['electrode_id'] = elect[0]

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
            ElectrodeConfig.Electrode.insert1(elect_dict)
        # close the file
        io.close()

@schema
class SpikeSortingGroup(dj.Manual):
    definition = """
     -> ElectrodeConfig.Electrode
     ---
     sort_group=-1: int  # the number of the sorting group for this electrode
     """
    def set_by_electrode_group(self, nwb_file_name):
        '''
        :param: nwb_file_name - the name of the nwb whose electrodes should be put into sorting groups
        :return: None
        Assign groups to all non-bad channel electrodes based on their electrode group and probe:
        Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a single group
        Electrodes from probes with multiple shanks (e.g. polymer probes) are placed in one group per shank
        '''
        # get the electrodes from this NWB file
        electrodes = (ElectrodeConfig.Electrode() & {'nwb_file_name' : nwb_file_name} & {'bad_channel' : 'False'}) \
            .fetch()
        # get a list of the electrode groups
        e_groups = np.unique(electrodes['electrode_group_name'])
        sort_group = 0
        elect_dict = dict()
        elect_dict['nwb_file_name'] = nwb_file_name
        for e_group in e_groups:
            elect_dict['electrode_group_name'] = e_group
            # for each electrode group, get a list of the unique shank numbers
            shank_list = np.unique(electrodes['probe_shank'][electrodes['electrode_group_name'] == e_group])
            # get the indices of all electrodes in for this group / shank and set their sorting group
            for shank in shank_list:
                shank_elect = electrodes['electrode_id'][np.logical_and(electrodes['electrode_group_name'] == e_group,
                                                                        electrodes['probe_shank'] == shank)]
                 # create a 2d array for these electrodes
                sort_insert = list()
                for elect in shank_elect:
                    sort_insert.append([nwb_file_name, e_group, elect, sort_group])
                self.insert(sort_insert, replace="True")
                sort_group += 1

    def write_prb(self, nwb_file_name, prb_file_name):
        '''
        Writes a prb file containing informaiton on the sorting groups and their geometry for use with the
        SpikeInterface package. See the SpikeInterface documentation for details on prb file formats
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
        sort_electrodes = (SpikeSortingGroup() & {'nwb_file_name': nwb_file_name}).fetch()
        sort_groups = np.unique(sort_electrodes['sort_group']).tolist()
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
                geometry.append([rel_x + sort_group, rel_y])
                label.append(str(electrode_id))
            channel_group[sort_group]['geometry'] = geometry
            channel_group[sort_group]['label'] = label
        prbf.write('channel_groups = ')
        prbf.write(json.dumps(channel_group))

        prbf.close()







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
class Raw(dj.Imported):
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

    def make(self, key):
        # get the NWB file name from this session
        nwb_file_name = key['nwb_file_name']

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
        with pynwb.NWBHDF5IO(nwb_file_name, mode='r') as io:
            nwbf = io.read()
        # get the object id
        nwb_object_id = (self & {'nwb_file_name' : key['nwb_file_name']}).fetch1('nwb_object_id')
        return nwbf.objects[nwb_object_id]


@schema
class LFPElectrode(dj.Manual):
    definition = """
     -> ElectrodeConfig.Electrode
     ---
     use_for_lfp = 'False': enum('True', 'False')
     """

    def set_lfp_elect(self, electrode_list):
        # TO DO: do this in a better way
        all_electrodes = ElectrodeConfig.Electrode.fetch(as_dict=True)
        primary_key = ElectrodeConfig.Electrode.primary_key
        for e in all_electrodes:
            use_for_lfp = 'True' if e['electrode_id'] in electrode_list else 'False'
            # create a dictionary so we can insert new elects
            lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
            lfpelectdict['use_for_lfp'] = use_for_lfp
            self.insert1(lfpelectdict, replace='True')


@schema
class LFP(dj.Computed):
    definition = """
    -> Raw                                  # the Raw data this LFP is computed from
    -> LFPElectrode                         # the LFP electrodes selected
    ---
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
        sampling_rate = int(np.round(sampling_rate))

        #target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        valid_times = (common_interval.IntervalList() & {'nwb_file_name': key['nwb_file_name'] ,
                                                          'interval_name': interval_name}).fetch1('valid_times')
        # get the LFP filter that matches the raw data
        filter_coeff = (common_filter.FirFilter() & {'filter_name' : 'LFP 0-400 Hz'} & {'filter_sampling_rate':
                                                                                  sampling_rate}).fetch1('filter_coeff')
        if len(filter_coeff) == 0:
            print(f'Error in LFP: no filter found with data sampling rate of {sampling_rate}')
            return None

        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPElectrode & key & {'use_for_lfp' : 'True'}).fetch('KEY')
        electrode_id_list = list(k['electrode_id'] for k in electrode_keys)

        # get the filename for the linked file that will store the data
        nwb_out_file_name = common_session.LinkedNwbfile().create(key['nwb_file_name'])
        print(f'output file name {nwb_out_file_name}')
        common_filter.FirFilter().filter_data_nwb(nwb_out_file_name, rawdata.timestamps, rawdata.data,
                                               filter_coeff, valid_times, electrode_id_list, decimation)



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
    -> LFPBandParameters
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
