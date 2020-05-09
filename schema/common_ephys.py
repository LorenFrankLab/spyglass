import datajoint as dj

import common_session
import common_region
import common_interval
import common_device
import common_filter
import pynwb

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
        -> common_device.Device
        description: varchar(80) # description of electrode group
        target_hemisphere: enum('Right','Left')
        """

    class Electrode(dj.Part):
        definition = """
        -> master.ElectrodeGroup
        electrode_id: int               # the unique number for this electrode
        label='': varchar(80)           # unique label for each contact
        ---
        -> common_device.Probe.Electrode
        -> common_region.BrainRegion
        x=NULL: float                   # the x coordinate of the electrode position in the brain
        y=NULL: float                   # the y coordinate of the electrode position in the brain
        z=NULL: float                   # the z coordinate of the electrode position in the brain
        filtering: varchar(200)         # description of the signal filtering
        impedance=null: float                # electrode impedance
        good_chan: enum("True","False")      # if electrode is 'good' or 'bad' as observed during recording
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
            # get the device name
            devices = list(nwbf.devices.keys())
            for d in devices:
                if nwbf.devices[d] == electrode_group.device:
                    # this will match the entry in the device schema
                    eg_dict['device_name'] = d
                    break
            ElectrodeConfig.ElectrodeGroup.insert1(eg_dict)

        # now create the table of electrodes
        elect_dict = dict()
        electrodes = nwbf.electrodes.to_dataframe()
        elect_dict['nwb_file_name'] = key['nwb_file_name']
        for elect in range(len(electrodes)):
            # there is probably a more elegant way to do this
            elect_dict['electrode_group_name'] = electrodes.iloc[elect]['group_name']
            elect_dict['electrode_id'] = elect

            # FIX to get electrode information properly from input or NWB file. Label may not exist, so we should
            # check that and use and empty string if not.
            elect_dict['label'] = ''
            elect_dict['probe_type'] = 'tetrode'
            elect_dict['shank_num'] = 0
            elect_dict['probe_electrode'] = 0

            elect_dict['region_id'] = eg_dict['region_id']
            # elect_dict['x'] = electrodes.iloc[elect]['x']
            # elect_dict['y'] = electrodes.iloc[elect]['y']
            # elect_dict['z'] = electrodes.iloc[elect]['z']
            elect_dict['x'] = 0
            elect_dict['y'] = 0
            elect_dict['z'] = 0
            elect_dict['x_warped'] = 0
            elect_dict['y_warped'] = 0
            elect_dict['z_warped'] = 0
            elect_dict['contacts'] = ''

            elect_dict['filtering'] = electrodes.iloc[elect]['filtering']
            elect_dict['impedance'] = electrodes.iloc[elect]['imp']
            # FIX so we can have an NAN value
            # if elect_dict['impedance'] == nan:
            elect_dict['impedance'] = -1
            ElectrodeConfig.Electrode.insert1(elect_dict)
        # close the file
        io.close()


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
        print('nwb_file_name:', nwb_file_name)
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
    -> ElectrodeConfig.Electrode
    ---
    -> common_interval.IntervalList
    nwb_object_id: int  # the NWB object ID for loading this object from the file
    sampling_rate: float                            # Sampling rate, in Hz
    comments: varchar(80)
    description: varchar(80)
    """


@schema
class LFP(dj.Computed):
    definition = """
    -> Raw
    ref_elect: int  # the reference electrode used for this LFP trace, -1 for none
    ---
    -> common_interval.IntervalList
    -> common_filter.Filter
    nwb_object_id: varchar(255)  # the NWB object ID for loading this object from the file
    sampling_rate: float # the sampling rate, in HZ
    """

@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFP
    freq_min: int       #high pass frequency
    freq_max: int       #low pass frequency
    ---
    -> common_interval.IntervalList
    -> common_filter.Filter
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
