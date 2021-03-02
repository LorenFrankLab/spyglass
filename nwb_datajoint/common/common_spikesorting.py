import datajoint as dj
import tempfile

from .common_session import Session
from .common_region import BrainRegion
from .common_device import Probe
from .common_interval import IntervalList, SortInterval, interval_list_intersect, interval_list_excludes_ind
from .common_ephys import Raw, Electrode, ElectrodeGroup

import labbox_ephys as le
import spikeinterface as si
import spikeextractors as se
import spiketoolkit as st
import pynwb
import re
import os
import socket
import time
from pathlib import Path
import numpy as np
import scipy.signal as signal
import json
import h5py as h5
import kachery_p2p as kp
import kachery as ka
from itertools import compress
from tempfile import NamedTemporaryFile
from .common_nwbfile import Nwbfile, AnalysisNwbfile
from .nwb_helper_fn import get_valid_intervals, estimate_sampling_rate, get_electrode_indices
from .dj_helper_fn import dj_replace, fetch_nwb

from mountainlab_pytools.mdaio import readmda

from requests.exceptions import ConnectionError

schema = dj.schema('common_spikesorting')

_OVERWRITE_SORTING_RESULTS = True


@schema
class SortGroup(dj.Manual):
    definition = """
    # Table for holding the set of electrodes that will be sorted together
    -> Session
    sort_group_id : int  # identifier for a group of electrodes
    ---
    sort_reference_electrode_id = -1 : int  # the electrode to use for reference. -1: no reference, -2: common median
    """

    class SortGroupElectrode(dj.Part):
        definition = """
        -> master
        -> Electrode
        """

    def set_group_by_shank(self, nwb_file_name):
        """
        Adds sort group entries in SortGroup table based on shank
        Assigns groups to all non-bad channel electrodes based on their shank:
        - Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a
            single group
        - Electrodes from probes with multiple shanks (e.g. polymer probes) are
            placed in one group per shank

        Parameters
        ----------
        nwb_file_name : str
            the name of the NWB file whose electrodes should be put into sorting groups
        """
        # delete any current groups
        (SortGroup & {'nwb_file_name' : nwb_file_name}).delete()
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
        (SortGroup & {'nwb_file_name' : nwb_file_name}).delete()
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
        """
        Writes a prb file containing informaiton on the specified sort group and
        its geometry for use with the SpikeInterface package. See the
        SpikeInterface documentation for details on prb file format.
        :param sort_group_id: the id of the sort group
        :param nwb_file_name: the name of the nwb file for the session you wish to use
        :param prb_file_name: the name of the output prb file
        :return: None
        """
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
        electrode_group_name = sort_group_electrodes['electrode_group_name'][0]
        probe_type = (ElectrodeGroup & {'nwb_file_name' : nwb_file_name,
                                        'electrode_group_name' : electrode_group_name}).fetch1('probe_type')
        channel_group[sort_group_id] = dict()
        channel_group[sort_group_id]['channels'] = sort_group_electrodes['electrode_id'].tolist()
        geometry = list()
        label = list()
        for electrode_id in channel_group[sort_group_id]['channels']:
            # get the relative x and y locations of this channel from the probe table
            probe_electrode = int(electrodes['probe_electrode'][electrodes['electrode_id'] == electrode_id])
            rel_x, rel_y = (Probe().Electrode() & {'probe_type': probe_type,
                                                    'probe_electrode' : probe_electrode}).fetch('rel_x','rel_y')
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
    # Table that holds the list of spike sorters avaialbe through spikeinterface
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
    frequency_min=300: int # high pass filter value
    frequency_max=6000: int # low pass filter value
    filter_width=1000: int # the number of coefficients in the filter
    filter_chunk_size=30000: int # the size of the chunk for the filtering
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

@schema
class SpikeSortingWaveformParameters(dj.Manual):
    definition = """
    waveform_parameters_name: varchar(80) # the name for this set of waveform extraction parameters
    ---
    waveform_parameter_dict: blob # a dictionary containing the SpikeInterface waveform parameters
    """

@schema
class SpikeSortingMetrics(dj.Manual):
    definition = """
    # Table for holding the parameters for computing quality metrics
    cluster_metrics_list_name: varchar(80) # the name for this list of cluster metrics
    ---
    metrics_dict: blob # a dict of SpikeInterface metrics with True / False elements to indicate whether a given metric should be computed.
    # n_noise_waveforms=1000: int # the number of random noise waveforms to use for the noise overlap
    # n_cluster_waveforms=1000: int # the maximum number of spikes / waveforms from a cluster to use for metric calculations
    isi_threshold = 0.003: float # Interspike interval threshold in s for ISI metric (default 0.003)
    snr_mode = 'mad': enum('mad', 'std') # SNR mode: median absolute deviation ('mad) or standard deviation ('std') (default 'mad')
    snr_noise_duration = 10.0: float # length of data to use for noise estimation (default 10.0)
    max_spikes_per_unit_for_snr = 1000: int # Maximum number of spikes to compute templates for SNR from (default 1000)
    template_mode = 'mean': enum('mean','median') # Use 'mean' or 'median' to compute templates
    max_channel_peak = 'both': enum('both', 'neg', 'pos') # direction of the maximum channel peak: 'both', 'neg', or 'pos' (default 'both')
    max_spikes_per_unit_for_noise_overlap = 1000: int # Maximum number of spikes to compute templates for noise overlap from (default 1000)
    noise_overlap_num_features = 5: int # Number of features to use for PCA for noise overlap
    noise_overlap_num_knn = 1000: int # Number of nearest neighbors for noise overlap
    drift_metrics_interval_s = 60: float # length of period in s for evaluating drift (default 60 s)
    drift_metrics_min_spikes_per_interval = 10: int # minimum number of spikes in an interval for evaluation of drift (default 10)
    max_spikes_for_silhouette = 1000: int # Max spikes to be used for silhouette metric
    num_channels_to_compare = 7: int # number of channels to be used for the PC extraction and comparison (default 7)
    max_spikes_per_cluster = 1000: int # Max spikes to be used from each unit
    max_spikes_for_nn = 1000: int # Max spikes to be used for nearest-neighbors calculation
    n_neighbors = 4: int # number of nearest clusters to use for nearest neighbor calculation (default 4)
    n_jobs = 96: int # Number of parallel jobs (default 96)
    memmap = 0 : tinyint(1) # If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
    max_spikes_per_unit = 2000: int # Max spikes to use for computing waveform
    seed = 47: int # Random seed for reproducibility
    verbose = 1 : tinyint(1) # If nonzero (True), will be verbose in metric computation
    """

    def get_metric_dict(self):
        """Get the current list of metrics from spike interface and create a
        dictionary with all False elemnets.
        Users should set the desired set of metrics to be true and insert a new
        entry for that set.
        """
        metrics_list =  st.validation.get_quality_metrics_list()
        metric_dict = dict()
        return {metric: False for metric in metrics_list}


    @staticmethod
    def selected_metrics_list(metrics_dict):
        return [metric for metric in metrics_dict.keys() if metrics_dict[metric]]

    def validate_metrics_list(self, key):
        """ Checks whether metrics_list contains only valid metric names

        :param key: key for metrics to validate
        :type key: dict
        :return: True or False
        :rtype: boolean
        """
        #TODO: get list of valid metrics from spiketoolkit when available
        valid_metrics = self.get_metric_dict()
        metrics_dict = (self & key).fetch1('metrics_dict')
        valid = True
        for metric in metrics_dict:
            if not metric in valid_metrics.keys():
                print(f'Error: {metric} not in list of valid metrics: {valid_metrics}')
                valid = False
        return valid

    def compute_metrics(self, key, recording, sorting):
        """
        Use spikeinterface to compute the list of selected metrics for a sorting

        Parameters
        ----------
        key: str
            cluster_metrics_list_name from SpikeSortingParameters
        recording: spikeinterface RecordingExtractor
        sorting: spikeinterface SortingExtractor

        Returns
        -------
        metrics: pandas.dataframe
        """
        m = (self & {'cluster_metrics_list_name': key}).fetch(as_dict=True)[0]

        return st.validation.compute_quality_metrics(sorting=sorting,
                                                     recording=recording,
                                                     metric_names=self.selected_metrics_list(m['metrics_dict']),
                                                     as_dataframe=True,
                                                     isi_threshold=m['isi_threshold'],
                                                     snr_mode=m['snr_mode'],
                                                     snr_noise_duration=m['snr_noise_duration'],
                                                     max_spikes_per_unit_for_snr=m['max_spikes_per_unit_for_snr'],
                                                     template_mode=m['template_mode'],
                                                     max_channel_peak=m['max_channel_peak'],
                                                     max_spikes_per_unit_for_noise_overlap=m['max_spikes_per_unit_for_noise_overlap'],
                                                     noise_overlap_num_features=m['noise_overlap_num_features'],
                                                     noise_overlap_num_knn=m['noise_overlap_num_knn'],
                                                     drift_metrics_interval_s=m['drift_metrics_interval_s'],
                                                     drift_metrics_min_spikes_per_interval=m['drift_metrics_min_spikes_per_interval'],
                                                     max_spikes_for_silhouette=m['max_spikes_for_silhouette'],
                                                     num_channels_to_compare=m['num_channels_to_compare'],
                                                     max_spikes_per_cluster=m['max_spikes_per_cluster'],
                                                     max_spikes_for_nn=m['max_spikes_for_nn'],
                                                     n_neighbors=m['n_neighbors'],
                                                     n_jobs=m['n_jobs'],
                                                     memmap=bool(m['memmap']),
                                                     max_spikes_per_unit=m['max_spikes_per_unit'],
                                                     seed=m['seed'],
                                                     verbose=bool(m['verbose']))

@schema
class SpikeSortingParameters(dj.Manual):
    definition = """
    # Table for holding parameters for each spike sorting run
    -> SortGroup
    -> SpikeSorterParameters
    -> SortInterval
    ---
    -> SpikeSortingMetrics
    -> IntervalList
    import_path = '': varchar(200) # optional path to previous curated sorting output
    """

@schema
class SpikeSorting(dj.Computed):
    definition = """
    # Table for holding spike sorting runs
    -> SpikeSortingParameters
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40) # the object ID for the units for this sort group
    time_of_sort = 0: int # This is when the sort was done.
    curation_workspace_name: varchar(1000) # Name of labbox-ephys workspace for curation
    """

    def make(self, key):
        """
        Runs spike sorting on the data and parameters specified by the
        SpikeSortingParameter table and inserts a new entry to SpikeSorting table.

        Parameters
        ----------
        key: dict
            partially filled entity; value of primary keys from key source
            (in this case SpikeSortingParameters)
        """
        print('Getting ready...')

        # Create a new NWB file for holding the results of analysis (e.g. spike sorting).
        # Save the name to the 'key' dict to use later.
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])

        # will use these later
        sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                          'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        # sampling_rate = (Raw & {'nwb_file_name': key['nwb_file_name']}).fetch1('sampling_rate')


        # Create dictionaries to hold output of spike sorting
        units = dict()
        units_valid_times = dict() # why both valid times and sort interval?
        units_sort_interval = dict()
        units_templates = dict()

        # TODO: finish `import_sorted_data` function below
        # import_sorted_data()

        # Prepare a RecordingExtractor to pass into spike sorting
        recording, sort_interval_valid_times = self.get_recording_extractor(key)

        # Path to Nwb file that will hold recording and sorting extractors
        unique_file_name = key['nwb_file_name'] + '_' + key['sort_interval_name'] \
                           + '_' + str(key['sort_group_id']) \
                           + '_' + key['sorter_name'] \
                           + '_' + str(key['parameter_set_name'])
        analysis_path = str(Path(os.environ['SPIKE_SORTING_STORAGE_DIR'])
                            / key['analysis_file_name'])
        os.makedirs(analysis_path, exist_ok=_OVERWRITE_SORTING_RESULTS)
        # os.mkdir(analysis_path)
        extractor_nwb_path = str(Path(analysis_path) / unique_file_name) + '.nwb'

        # Write recording extractor to NWB file
        se.NwbRecordingExtractor.write_recording(recording,
                                                 save_path = extractor_nwb_path,
                                                 overwrite = _OVERWRITE_SORTING_RESULTS)

        # Run spike sorting
        print(f'\nRunning spike sorting on {key}...')
        sort_parameters = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                                    'parameter_set_name': key['parameter_set_name']}).fetch1()
        sorting = si.sorters.run_sorter(key['sorter_name'], recording,
                                        output_folder = os.getenv('SORTING_TEMP_DIR', None),
                                        grouping_property = 'group',
                                        **sort_parameters['parameter_dict'])
        # Save time of sort
        key['time_of_sort'] = int(time.time())
        # Write sorting extractor to NWB file that contains recording extractor
        se.NwbSortingExtractor.write_sorting(sorting, save_path = extractor_nwb_path,
                                             timestamps = recording.timestamps)

        # Compute quality metrics
        print('\nComputing quality metrics...')
        metrics_key = (SpikeSortingParameters & key).fetch1('cluster_metrics_list_name')
        metrics = SpikeSortingMetrics().compute_metrics(metrics_key, recording, sorting)

        # Save sorting results
        print('\nSaving sorting results...')
        # Create a stack of labeled arrays of the sorted spike times
        unit_ids = sorting.get_unit_ids()
        for unit_id in unit_ids:
            # spike times in samples; 0 is first sample
            unit_spike_samples = sorting.get_unit_spike_train(unit_id = unit_id)
            units[unit_id] = recording.timestamps[unit_spike_samples]
            # units[unit_id] = sort_interval[0] + unit_spike_samples/sampling_rate
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        # Add the units to the Analysis Nwb file
        # TODO: consider replacing with spikeinterface call if possible
        units_object_id, _ = AnalysisNwbfile().add_units(key['analysis_file_name'],
                                                         units, units_valid_times,
                                                         units_sort_interval,
                                                         metrics = metrics)

        # Add the new analysis NWB file to the AnalysisNWBFile table
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])

        key['units_object_id'] = units_object_id

        # Check if KACHERY_P2P_API_PORT is set
        kp_port = os.getenv('KACHERY_P2P_API_PORT', False)
        assert kp_port, 'You must set KACHERY_P2P_API_PORT environmental variable'

        # Check if the kachery-p2p daemon is running in the background
        try:
            kp_channel = kp.get_channels()
        except ConnectionError:
            raise RuntimeError(('You must have a kachery-p2p daemon running in'
                                ' the background (kachery-p2p-start-daemon --label'
                                ' franklab --config https://gist.githubuse'
                                'rcontent.com/khl02007/b3a092ba3e590946480fb1267'
                                '964a053/raw/f05eda4789e61980ce630b23ed38a7593f5'
                                '8a7d9/franklab_kachery-p2p_config.yaml)'))

        # Create workspace and feed
        fname, _ = os.path.splitext(key['analysis_file_name'])
        feed = kp.load_feed(fname, create=True)
        print(fname)
        workspace_name = unique_file_name
        print(workspace_name)
        workspace = le.load_workspace(workspace_name=workspace_name, feed=feed)

        recording_uri = ka.store_object({
            'recording_format': 'nwb',
            'data': {
                'path': extractor_nwb_path
            }
        })
        sorting_uri = ka.store_object({
            'sorting_format': 'nwb',
            'data': {
                'path': extractor_nwb_path
            }
        })

        sorting = le.LabboxEphysSortingExtractor(sorting_uri)
        recording = le.LabboxEphysRecordingExtractor(recording_uri, download=True)

        print(f'Feed URI: {feed.get_uri()}')

        recording_label = key['nwb_file_name'] + '_' + key['sort_interval_name'] \
                          + '_' + str(key['sort_group_id'])
        sorting_label = key['sorter_name'] +  '_' + key['parameter_set_name']

        R_id = workspace.add_recording(recording=recording, label=recording_label)
        S_id = workspace.add_sorting(sorting=sorting, recording_id=R_id, label=sorting_label)

        key['curation_workspace_name'] = unique_file_name

        # Finally, insert the entity into table
        self.insert1(key)
        print('\nDone - entry inserted to table!\n')

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def get_recording_extractor(self, key):
        """
        Generates a RecordingExtractor object for to be sorted

        Parameters
        ----------
        key: dict

        Returns
        -------
        sub_R : SubRecordingExtractor
            for the part of recording to be sorted (referenced and filtered)
        sort_interval_valid_times : np.array
            (start, end) times for valid sorting interval
        """
        # Get the timestamps for the recording
        raw_data_obj = (Raw & {'nwb_file_name': key['nwb_file_name']}).fetch_nwb()[0]['raw']
        timestamps = np.asarray(raw_data_obj.timestamps)

        # Get start and end frames for the chunk of recording we are sorting
        sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                          'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        interval_list_name = (SpikeSortingParameters & key).fetch1('interval_list_name')
        valid_times =  (IntervalList & {'nwb_file_name': key['nwb_file_name'],
                                        'interval_list_name': interval_list_name}).fetch1('valid_times')
        sort_interval_valid_times = interval_list_intersect(sort_interval, valid_times)
        sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))
        assert sort_indices[1] - sort_indices[0] > 1000, f'Error in get_recording_extractor: sort indices {sort_indices} are not valid'

        # Get electrodes for the chunk of recording we are sorting
        electrode_ids = (SortGroup.SortGroupElectrode & {'nwb_file_name' : key['nwb_file_name'],
                                                         'sort_group_id' : key['sort_group_id']}).fetch('electrode_id')
        electrode_group_name = (SortGroup.SortGroupElectrode & {'nwb_file_name' : key['nwb_file_name'],
                                                                'sort_group_id' : key['sort_group_id']}).fetch('electrode_group_name')
        electrode_group_name = np.int(electrode_group_name[0])
        probe_type = (Electrode & {'nwb_file_name': key['nwb_file_name'],
                                   'electrode_group_name': electrode_group_name,
                                   'electrode_id': electrode_ids[0]}).fetch1('probe_type')

        # Create a NwbRecordingExtractor
        R = se.NwbRecordingExtractor(Nwbfile.get_abs_path(key['nwb_file_name']),
                                     electrical_series_name = 'e-series')

        # Create a SubRecordingExtractor for the chunk
        sub_R = se.SubRecordingExtractor(R, channel_ids = electrode_ids.tolist(),
                                         start_frame = sort_indices[0],
                                         end_frame = sort_indices[1])
        # Necessary for now
        # sub_R.set_channel_groups([0]*len(electrode_ids),
        #                          channel_ids = electrode_ids.tolist())

        # TODO: add a step where large transients are masked

        # Reference the chunk
        sort_reference_electrode_id = (SortGroup & {'nwb_file_name' : key['nwb_file_name'],
                                                    'sort_group_id' : key['sort_group_id']}).fetch('sort_reference_electrode_id')
        if sort_reference_electrode_id >= 0:
            sub_R = st.preprocessing.common_reference(sub_R, reference='single',
                                                      groups=[key['sort_group_id']],
                                                      ref_channels=sort_reference_electrode_id)
        elif sort_reference_electrode_id == -2:
            sub_R = st.preprocessing.common_reference(sub_R, reference='median')

        # Filter the chunk
        param = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                          'parameter_set_name': key['parameter_set_name']}).fetch1()
        sub_R = st.preprocessing.bandpass_filter(sub_R, freq_min = param['frequency_min'],
                                                 freq_max = param['frequency_max'],
                                                 freq_wid = param['filter_width'],
                                                 chunk_size = param['filter_chunk_size'])

        # If tetrode and location for every channel is (0,0), give new locations
        channel_locations = sub_R.get_channel_locations()
        if np.all(channel_locations==0) and len(channel_locations)==4 and probe_type[:7]=='tetrode':
            print('Tetrode; making up channel locations...')
            channel_locations = [[0,0],[0,1],[1,0],[1,1]]
            sub_R.set_channel_locations(channel_locations)

        # give timestamps for the SubRecordingExtractor
        sub_R.timestamps = timestamps[sort_indices[0]:sort_indices[1]]

        return sub_R, sort_interval_valid_times

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

        raw_data_obj = (Raw() & {'nwb_file_name' : key['nwb_file_name']}).fetch_nwb()[0]['raw']
        # get the indices of the data to use. Note that spike_extractors has a time_to_frame function,
        # but it seems to set the time of the first sample to 0, which will not match our intervals
        timestamps = np.asarray(raw_data_obj.timestamps)
        sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))

        unit_timestamps_list = []
        # TODO: do something more efficient here; note that searching for maching sort_intervals within pandas doesn't seem to work
        for index, unit in units.iterrows():
            if np.ndarray.all(np.ravel(unit['sort_interval']) == sort_interval):
                #unit_timestamps.extend(unit['spike_times'])
                unit_frames = np.searchsorted(timestamps, unit['spike_times']) - sort_indices[0]
                unit_timestamps.extend(unit_frames)
                #unit_timestamps_list.append(unit_frames)
                unit_labels.extend([index]*len(unit['spike_times']))

        output=se.NumpySortingExtractor()
        output.set_times_labels(times=np.asarray(unit_timestamps),labels=np.asarray(unit_labels))
        return output

    # TODO: write a function to import sorted data
    def import_sorted_data():
        # Check if spikesorting has already been run on this dataset;
        # if import_path is not empty, that means there exists a previous spikesorting run
        import_path = (SpikeSortingParameters() & key).fetch1('import_path')
        if import_path != '':
            sort_path = Path(import_path)
            assert sort_path.exists(), f'Error: import_path {import_path} does not exist when attempting to import {(SpikeSortingParameters() & key).fetch1()}'
            # the following assumes very specific file names from the franklab, change as needed
            firings_path = sort_path / 'firings_processed.mda'
            assert firings_path.exists(), f'Error: {firings_path} does not exist when attempting to import {(SpikeSortingParameters() & key).fetch1()}'
            # The firings has three rows, the electrode where the peak was detected, the sample count, and the cluster ID
            firings = readmda(str(firings_path))
            # get the clips
            clips_path = sort_path / 'clips.mda'
            assert clips_path.exists(), f'Error: {clips_path} does not exist when attempting to import {(SpikeSortingParameters() & key).fetch1()}'
            clips = readmda(str(clips_path))
            # get the timestamps corresponding to this sort interval
            # TODO: make sure this works on previously sorted data
            timestamps = timestamps[np.logical_and(timestamps >= sort_interval[0], timestamps <= sort_interval[1])]
            # get the valid times for the sort_interval
            sort_interval_valid_times = interval_list_intersect(np.array([sort_interval]), valid_times)

            # get a list of the cluster numbers
            unit_ids = np.unique(firings[2,:])
            for index, unit_id in enumerate(unit_ids):
                unit_indices = np.ravel(np.argwhere(firings[2,:] == unit_id))
                units[unit_id] = timestamps[firings[1, unit_indices]]
                units_templates[unit_id] = np.mean(clips[:,:,unit_indices], axis=2)
                units_valid_times[unit_id] = sort_interval_valid_times
                units_sort_interval[unit_id] = [sort_interval]

            #TODO: Process metrics and store in Units table.
            metrics_path = (sort_path / 'metrics_processed.json').exists()
            assert metrics_path.exists(), f'Error: {metrics_path} does not exist when attempting to import {(SpikeSortingParameters() & key).fetch1()}'
            metrics_processed = json.load(metrics_path)

    @staticmethod
    def prepare_labbox_curation(recording_label, sorting_label,
                                recording_nwb_path, sorting_nwb_path,
                                feed_name, workspace_name):
        """
        Creates labbox-ephys feed and workspace; adds recording and sorting

        Parameters
        ----------
        recording_label: string
            name given to recording
        sorting_label: str
            name given to sorting
        recording_nwb_path: str
            path to nwb file containing recording
        sorting_nwb_path: str
            path to nwb file containing sorting
        feed_name: str
            default: name of analysisNWB file
        workspace_name: str
            default: concatenated SpikeSortingParameter primary key
        """

        recording_uri = ka.store_object({
            'recording_format': 'nwb',
            'data': {
                'path': recording_nwb_path
            }
        })
        sorting_uri = ka.store_object({
            'sorting_format': 'nwb',
            'data': {
                'path': sorting_nwb_path
            }
        })

        sorting = le.LabboxEphysSortingExtractor(sorting_uri)
        recording = le.LabboxEphysRecordingExtractor(recording_uri, download=True)

        feed = kp.load_feed(feed_name, create=True)
        workspace = le.load_workspace(workspace_name=workspace_name, feed=feed)

        print(f'Feed URI: {feed.get_uri()}')
        R_id = workspace.add_recording(recording=recording, label=recording_label)
        S_id = workspace.add_sorting(sorting=sorting, recording_id=R_id, label=sorting_label)

        return None

    def create_labbox_ephys_feed(self, le_recordings, le_sortings, create_snapshot=False):
        """
        Creates feed to be used by labbox-ephys during curation

        Parameters
        ----------
        create_snapshot: bool
            set to False if want writable feed

        Returns
        -------
        feed uri
        """
        try:
            f = kp.create_feed()
            recordings = f.get_subfeed(dict(documentId='default', key='recordings'))
            sortings = f.get_subfeed(dict(documentId='default', key='sortings'))
            for le_recording in le_recordings:
                recordings.append_message(dict(
                    action=dict(
                        type='ADD_RECORDING',
                        recording=le_recording
                    )
                ))
            for le_sorting in le_sortings:
                sortings.append_message(dict(
                    action=dict(
                        type='ADD_SORTING',
                        sorting=le_sorting
                    )
                ))
            if create_snapshot:
                x = f.create_snapshot([
                    dict(documentId='default', key='recordings'),
                    dict(documentId='default', key='sortings')
                ])
                return x.get_uri()
            else:
                return f.get_uri()
        finally:
            if create_snapshot:
                f.delete()

    def metrics_to_labbox_ephys(self, metrics, unit_ids):
        """Turns the metrics pandas.dataframe to a list of dict to feed to labbox

        Parameters
        ----------
        metrics : pandas.DataFrame
            from spikeinterface
        unit_ids : list
            from SortingExtractor

        Returns
        -------
        external_unit_metrics: list of dict
        """
        external_unit_metrics = []
        for metric in metrics.columns:
            test_metric = dict(
                name=metric,
                label=metric,
                tooltip=metric,
                data=dict(zip([str(uid) for uid in unit_ids], [i for i in metrics[metric]]))
            )
            external_unit_metrics.append(test_metric)
        return external_unit_metrics

@schema
class CuratedSpikeSorting(dj.Computed):
    definition = """
    # Table for holding the output of spike sorting
    -> SpikeSorting
    ---
    -> AnalysisNwbfile    # New analysis NWB file to hold unit info
    """

    class Units(dj.Part):
        definition = """
        # Table for holding sorted units
        -> CuratedSpikeSorting
        unit_id: int            # ID for each unit
        ---
        label: varchar(80)      # label for each unit
        noise_overlap: float    # noise overlap metric for each unit
        isolation_score: float  # isolation score metric for each unit
        """

    def make(self, key):
        # Create a new analysis NWB file that is a copy of the original
        # analysis NWB file
        parent_key = (SpikeSorting & key).fetch1()
        new_analysis_nwb_filename = AnalysisNwbfile.copy(parent_key['analysis_file_name'])

        # Get labels and print
        labels = self.get_labels(parent_key['curation_feed_uri'])
        print('Labels: ' + str(labels))

        # turn labels to list of str
        labels_concat = []
        for idx, unitId in enumerate(labels):
            label_concat = ','.join(labels[unitId])
            labels_concat.append(label_concat)

        # Get metrics from analysis NWB file
        with pynwb.NWBHDF5IO(path = AnalysisNwbfile.get_abs_path(parent_key['analysis_file_name']),
                             mode = "r") as io:
            nwbf = io.read()
            noise_overlap = nwbf.units['noise_overlap'][:]
            isolation_score = nwbf.units['nn_hit_rate'][:]

        # Print metrics
        print('Noise overlap: ' + str(noise_overlap))
        print('Isolation score: ' + str(isolation_score))

        # Add labels to the new analysis NWB file
        print('\nSaving units data to new AnalysisNwb file...')
        with pynwb.NWBHDF5IO(path = AnalysisNwbfile.get_abs_path(new_analysis_nwb_filename),
                             mode = "a") as io:
            nwbf = io.read()
            nwbf.add_unit_column(name = 'label', description='label given during curation',
                                 data = labels_concat)
            print(nwbf.units)
            io.write(nwbf)
        print('Done with AnalysisNwb file.')

        # Insert new file to AnalysisNWBfile table
        AnalysisNwbfile().add(key['nwb_file_name'], new_analysis_nwb_filename)
        # Insert entry to CuratedSpikeSorting table
        self.insert1(dict(key, analysis_file_name = new_analysis_nwb_filename))

        # Add entries to CuratedSpikeSorting.Units table
        print('\nAdding to dj Units table...')
        for idx, unitId in enumerate(labels):
            CuratedSpikeSorting.Units.insert1(dict(key, unit_id = unitId,
                                              label = ','.join(labels[unitId]),
                                              noise_overlap = noise_overlap[idx],
                                              isolation_score = isolation_score[idx]))
        print('Done with dj Units table.')

    def get_labels(self, feed_uri):
        """Parses the curation feed to get a label for each unit.

        Parameters
        ----------
        feed_uri : str

        Returns
        -------
        final_labels : dict
            key is int (unitId), value is list of strings (labels)
        """
        f = kp.load_feed(feed_uri)
        sf = f.get_subfeed(dict(documentId='default', key='sortings'))
        msgs = sf.get_next_messages()
        label_msgs = list(compress(msgs,[(m['action']['type']=='ADD_UNIT_LABEL') or (m['action']['type']=='REMOVE_UNIT_LABEL') for m in msgs]))
        unitIds = list(set([lm['action']['unitId'] for lm in label_msgs]))
        final_labels = dict()
        for cell in unitIds:
            unit_label_msgs = list(compress(label_msgs, [lm['action']['unitId']==cell for lm in label_msgs]))
            adds = list(compress(unit_label_msgs,[i['action']['type']=='ADD_UNIT_LABEL' for i in unit_label_msgs]))
            removes = list(compress(unit_label_msgs,[i['action']['type']=='REMOVE_UNIT_LABEL' for i in unit_label_msgs]))
            labels_added = [k['action']['label'] for k in adds]
            labels_removed = [k['action']['label'] for k in removes]
            final_labels.update({cell: list(set(labels_added) - set(labels_removed))})
        return final_labels

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)


""" for curation feed reading:
import kachery_p2p as kp
a = kp.load_feed('feed://...')
b= a.get_subfeed(dict(documentId='default', key='sortings'))
b.get_next_messages()

result is list of dictionaries


During creation of feed:
feed_uri = create_labbox_ephys_feed(le_recordings, le_sortings, create_snapshot=False)

Pull option-create_snapshot branch
----------
metrics
----------
- add to units table
- isolation score, noise overlap
- waveform samples
- want to recompute metrics after merge

- looking at spikeinterface to figure out which metrics it is computing and where
- SNR, dprime, drift, firing rates, nearest neighbor metrics
- once we have sorting and recording, can just call functions to compute these
- as soon as sorting is done we call these
- spike sorting parameters: need a dictionary for all the metrics we compute
- store in nwb units table
- would have to create new units table after merges
- can get rid of unnecessary waveforms
- ryan will take metrics from units table to labbox ephys
- labboxepys doesnt have noise overlap
- noise overlap: how similar are random waveforms to your waveforms
- mlsm4-alg has feature to toss clusters below noise overlap
- TODO:
- parse the feed; and add labels to units table in analysisNWB file and then maybe datajoint
- nwb file put into kachery
- labbox can read from this file via plugin
- curate
- take that feed back into datajoint
- labels live only in datajoint units table
- when pulling back into dj, create new units table that reflects merges

labbox-launcher command:
labbox-launcher run magland/labbox-ephys:0.4.0 --docker_run_opts "--net host -e KACHERY_P2P_API_PORT=$KACHERY_P2P_API_PORT" --kachery $KACHERY_STORAGE_DIR
"""
