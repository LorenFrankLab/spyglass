import datajoint as dj
import numpy as np
import pynwb
import re
import warnings

from .common_device import Probe  # noqa: F401
from .common_filter import FirFilter
from .common_interval import IntervalList    # noqa: F401
# SortInterval, interval_list_intersect, interval_list_excludes_ind
from .common_nwbfile import Nwbfile, AnalysisNwbfile
from .common_region import BrainRegion  # noqa: F401
from .common_session import Session  # noqa: F401
from .nwb_helper_fn import (get_valid_intervals, estimate_sampling_rate, get_electrode_indices, get_data_interface,
                            get_nwb_file)
from .dj_helper_fn import fetch_nwb  # dj_replace

schema = dj.schema('common_ephys')


@schema
class ElectrodeGroup(dj.Imported):
    definition = """
    # Grouping of electrodes corresponding to a physical probe.
    -> Session
    electrode_group_name: varchar(80)  # electrode group name from NWBFile
    ---
    -> BrainRegion
    -> Probe
    description: varchar(80)  # description of electrode group
    target_hemisphere: enum('Right','Left')
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
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
            # TODO check and replace this with
            # if isinstance(electrode_group.device, ndx_franklab_novela.Probe):
            # key['probe_type'] = electrode_group.device.probe_type
            # else:
            # key['probe_type'] = 'unknown-probe-type'
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
    electrode_id: int                      # the unique number for this electrode
    ---
    -> Probe.Electrode
    -> BrainRegion
    name='': varchar(80)                   # unique label for each contact
    original_reference_electrode=-1: int   # the configured reference electrode for this electrode
    x=NULL: float                          # the x coordinate of the electrode position in the brain
    y=NULL: float                          # the y coordinate of the electrode position in the brain
    z=NULL: float                          # the z coordinate of the electrode position in the brain
    filtering: varchar(200)                # description of the signal filtering
    impedance=null: float                  # electrode impedance
    bad_channel: enum("True","False")      # if electrode is 'good' or 'bad' as observed during recording
    x_warped=NULL: float                   # x coordinate of electrode position warped to common template brain
    y_warped=NULL: float                   # y coordinate of electrode position warped to common template brain
    z_warped=NULL: float                   # z coordinate of electrode position warped to common template brain
    contacts: varchar(80)                  # label of electrode contacts used for a bipolar signal -- current workaround
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # create the table of electrodes
        electrodes = nwbf.electrodes.to_dataframe()

        # Below it would be better to find the mapping between
        # nwbf.electrodes.colnames and the schema fields and
        # where possible, assign automatically. It would also help to be
        # robust to missing fields and have them
        # assigned as empty if they don't exist in the nwb file in case
        # people are not using our column names.

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
            except Exception:  # TODO: use more precise error check
                key['original_reference_electrode'] = -1
            self.insert1(key, skip_duplicates=True)


@schema
class Raw(dj.Imported):
    definition = """
    # Raw voltage timeseries data, ElectricalSeries in NWB.
    -> Session
    ---
    -> IntervalList
    raw_object_id: varchar(80)      # the NWB object ID for loading this object from the file
    sampling_rate: float            # Sampling rate calculated from data, in Hz
    comments: varchar(80)
    description: varchar(80)
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        raw_interval_name = "raw data valid times"
        # get the acquisition object
        # TODO this assumes there is a single item in NWBFile.acquisition
        try:
            rawdata = nwbf.get_acquisition()
            assert isinstance(rawdata, pynwb.ecephys.ElectricalSeries)
        except Exception:  # TODO: use more precise error check
            warnings.warn(f'WARNING: Unable to get aquisition object in: {nwb_file_abspath}')
            return
        print('Estimating sampling rate...')
        # NOTE: Only use first 1e6 timepoints to save time
        sampling_rate = estimate_sampling_rate(np.asarray(rawdata.timestamps[:int(1e6)]), 1.5)
        print(f'Estimated sampling rate: {sampling_rate}')
        key['sampling_rate'] = sampling_rate

        interval_dict = dict()
        interval_dict['nwb_file_name'] = key['nwb_file_name']
        interval_dict['interval_list_name'] = raw_interval_name
        # get the list of valid times given the specified sampling rate.
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
        self.insert1(key, skip_duplicates=True)

    def nwb_object(self, key):
        # TODO return the nwb_object; FIX: this should be replaced with a fetch call. Note that we're using the raw file
        # so we can modify the other one.
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        raw_object_id = (self & {'nwb_file_name': key['nwb_file_name']}).fetch1('raw_object_id')
        return nwbf.objects[raw_object_id]

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, 'nwb_file_abs_path'), *attrs, **kwargs)


@schema
class SampleCount(dj.Imported):
    definition = """
    # Sample count :s timestamp timeseries
    -> Session
    ---
    sample_count_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # get the sample count object
        # TODO: change name when nwb file is changed
        sample_count = get_data_interface(nwbf, 'sample_count')
        if sample_count is None:
            warnings.warn(f'Unable to get sample count object in: {nwb_file_abspath}')
            return
        key['sample_count_object_id'] = sample_count.object_id

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
        (LFPSelection() & {'nwb_file_name': nwb_file_name}).delete()
        # check to see if the user allowed the deletion
        if len((LFPSelection() & {'nwb_file_name': nwb_file_name}).fetch()) == 0:
            LFPSelection().insert1({'nwb_file_name': nwb_file_name})

            # TO DO: do this in a better way
            all_electrodes = Electrode.fetch(as_dict=True)
            primary_key = Electrode.primary_key
            for e in all_electrodes:
                # create a dictionary so we can insert new elects
                if e['electrode_id'] in electrode_list:
                    lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                    LFPSelection().LFPElectrode.insert1(lfpelectdict, replace=True)


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

        # TEST
        # interval_list_name = '01_s1'
        key['interval_list_name'] = interval_list_name

        valid_times = (IntervalList() & {'nwb_file_name': key['nwb_file_name'],
                                         'interval_list_name': interval_list_name}).fetch1('valid_times')

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (FirFilter() & {'filter_name': 'LFP 0-400 Hz'} &
                  {'filter_sampling_rate': sampling_rate}).fetch(as_dict=True)

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
        lfp_object_id = FirFilter().filter_data_nwb(lfp_file_abspath, rawdata,
                                                    filter_coeff, valid_times, electrode_id_list, decimation)

        key['analysis_file_name'] = lfp_file_name
        key['lfp_object_id'] = lfp_object_id
        key['lfp_sampling_rate'] = sampling_rate // decimation
        self.insert1(key)

    def nwb_object(self, key):
        # return the nwb_object.
        lfp_file_name = (LFP() & {'nwb_file_name': key['nwb_file_name']}).fetch1('analysis_file_name')
        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        nwbf = get_nwb_file(lfp_file_abspath)
        # get the object id
        nwb_object_id = (self & {'analysis_file_name': lfp_file_name}).fetch1('filtered_data_object_id')
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

    def set_lfp_band_electrodes(self, nwb_file_name, electrode_list, filter_name, interval_list_name,
                                reference_electrode_list, lfp_band_sampling_rate):
        '''
        Adds an entry for each electrode in the electrode_list with the specified filter, interval_list, and
        reference electrode.
        Also removes any entries that have the same filter, interval list and reference electrode but are not
        in the electrode_list.
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of LFP electrodes to be filtered
        :param filter_name: the name of the filter (from the FirFilter schema)
        :param interval_name: the name of the interval list (from the IntervalList schema)
        :param reference_electrode_list: A single electrode id corresponding to the reference to use for all
        electrodes or a list with one element per entry in the electrode_list
        :param lfp_band_sampling_rate: The output sampling rate to be used for the filtered data; must be an
        integer divisor of the LFP sampling rate
        :return: none
        '''
        # Error checks on parameters
        # electrode_list
        available_electrodes = (LFPSelection().LFPElectrode() & {'nwb_file_name': nwb_file_name}).fetch('electrode_id')
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError('All elements in electrode_list must be valid electrode_ids in the LFPSelection table')
        # sampling rate
        lfp_sampling_rate = (LFP() & {'nwb_file_name': nwb_file_name}).fetch1('lfp_sampling_rate')
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(f'lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer divisor of lfp '
                             f'samping rate {lfp_sampling_rate}')
        # filter
        if not len((FirFilter() & {'filter_name': filter_name, 'filter_sampling_rate': lfp_sampling_rate}).fetch()):
            raise ValueError(f'filter {filter_name}, sampling rate {lfp_sampling_rate}is not in the FirFilter table')
        # interval_list
        if not len((IntervalList() & {'interval_name': interval_list_name}).fetch()):
            raise ValueError(f'interval list {interval_list_name} is not in the IntervalList table; the list must be '
                             'added before this function is called')
        # reference_electrode_list
        if len(reference_electrode_list) != 1 and len(reference_electrode_list) != len(electrode_list):
            raise ValueError('reference_electrode_list must contain either 1 or len(electrode_list) elements')
        # add a -1 element to the list to allow for the no reference option
        available_electrodes = np.append(available_electrodes, [-1])
        if not np.all(np.isin(reference_electrode_list, available_electrodes)):
            raise ValueError('All elements in reference_electrode_list must be valid electrode_ids in the LFPSelection '
                             'table')

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
        self.insert1(key, skip_duplicates=True)

        # remove the keys that are not used for the LFPBandElectrode table
        key.pop('interval_list_name')
        key.pop('lfp_band_sampling_rate')
        # get all of the current entries and delete any that are not in the list
        elect_id, ref_id = (self.LFPBandElectrode() & key).fetch('electrode_id', 'reference_elect_id')
        for e, r in zip(elect_id, ref_id):
            if not len(np.where((electrode_list == e) & (ref_list == r))[0]):
                key['electrode_id'] = e
                key['reference_elect_id'] = r
                (self.LFPBandElectrode() & key).delete()

        # iterate through all of the new elements and add them
        for e, r in zip(electrode_list, ref_list):
            key['electrode_id'] = e
            key['electrode_group_name'] = (Electrode & {'electrode_id': e}).fetch1('electrode_group_name')
            key['reference_elect_id'] = r
            self.LFPBandElectrode().insert1(key, skip_duplicates=True)


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
        lfp_object = (LFP() & {'nwb_file_name': key['nwb_file_name']}).fetch_nwb()[0]['lfp']

        # load all the data to speed filtering
        lfp_data = np.asarray(lfp_object.data, dtype=type(lfp_object.data[0][0]))
        # lfp_timestamps = np.asarray(lfp_object.timestamps, dtype=type(lfp_object.timestamps[0]))

        # get the electrodes to be filtered and their references
        lfp_band_elect_id, lfp_band_ref_id = (LFPBandSelection().LFPBandElectrode() & key).fetch('electrode_id',
                                                                                                 'reference_elect_id')

        # get the indices of the electrodes to be filtered and the references
        lfp_band_elect_index = get_electrode_indices(lfp_object, lfp_band_elect_id)
        lfp_band_ref_index = get_electrode_indices(lfp_object, lfp_band_ref_id)

        # subtract off the references for the selected channels
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = lfp_data[:, elect_index] - lfp_data[:, lfp_band_ref_index]

        lfp_sampling_rate = (LFP() & {'nwb_file_name': key['nwb_file_name']}).fetch1('lfp_sampling_rate')
        interval_list_name, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1('interval_list_name',
                                                                                       'lfp_band_sampling_rate')
        valid_times = (IntervalList() & {'interval_list_name': interval_list_name}).fetch1('valid_times')
        filter_name, filter_sampling_rate, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1(
            'filter_name', 'filter_sampling_rate', 'lfp_band_sampling_rate')

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # get the LFP filter that matches the raw data
        filter = (FirFilter() & {'filter_name': filter_name} &
                                {'filter_sampling_rate': filter_sampling_rate}).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(f'Filter {filter_name} and sampling_rate {lfp_band_sampling_rate} does not exit in the '
                             'FirFilter table')

        filter_coeff = filter[0]['filter_coeff']
        if len(filter_coeff) == 0:
            print(f'Error in LFPBand: no filter found with data sampling rate of {lfp_band_sampling_rate}')
            return None

        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
        lfp_band_file_abspath = AnalysisNwbfile().get_abs_path(lfp_band_file_name)
        # filter the data and write to an the nwb file
        filtered_data_object_id = FirFilter().filter_data_nwb(lfp_band_file_abspath, lfp_object, filter_coeff,
                                                              valid_times, lfp_band_elect_id, decimation)

        key['analysis_file_name'] = lfp_band_file_name
        key['filtered_data_object_id'] = filtered_data_object_id
        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)
