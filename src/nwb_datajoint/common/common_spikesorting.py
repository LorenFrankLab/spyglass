from typing import Dict
import datajoint as dj

from .common_device import Probe
from .common_interval import IntervalList, SortInterval, interval_list_intersect, interval_list_excludes_ind
from .common_ephys import Raw, Electrode, ElectrodeGroup
from .common_session import Session
from .common_nwbfile import Nwbfile, AnalysisNwbfile
from .dj_helper_fn import dj_replace, fetch_nwb
from .nwb_helper_fn import get_valid_intervals

import spikeextractors as se
import spikesorters as ss
import spiketoolkit as st

from pathlib import Path
import os
import time
import numpy as np
import scipy.stats as stats
import json
import pynwb
import tempfile
import getpass

import kachery_p2p as kp
import labbox_ephys as le
import labbox_ephys_widgets_jp as lew
import sortingview

from itertools import compress

from mountainsort4._mdaio_impl import readmda
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
        (SortGroup & {'nwb_file_name': nwb_file_name}).delete()
        # get the electrodes from this NWB file
        electrodes = (Electrode() & {'nwb_file_name': nwb_file_name} & {'bad_channel': 'False'}).fetch()
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
        (SortGroup & {'nwb_file_name': nwb_file_name}).delete()
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
        probe_type = (ElectrodeGroup & {'nwb_file_name' : nwb_file_name,
                                        'electrode_group_name' : electrode_group_name}).fetch1('probe_type')
        channel_group[sort_group_id] = dict()
        channel_group[sort_group_id]['channels'] = sort_group_electrodes['electrode_id'].tolist()
        
        label = list()
        n_chan = len(channel_group[sort_group_id]['channels'])
        
        geometry = np.zeros((n_chan,2), dtype='float')
        tmp_geom = np.zeros((n_chan,3), dtype='float')
        for i, electrode_id in enumerate(channel_group[sort_group_id]['channels']):
            # get the relative x and y locations of this channel from the probe table
            probe_electrode = int(electrodes['probe_electrode'][electrodes['electrode_id'] == electrode_id])
            rel_x, rel_y, rel_z = (Probe().Electrode() & {'probe_type': probe_type,
                                                    'probe_electrode' : probe_electrode}).fetch('rel_x','rel_y', 'rel_z')
            #TODO: Fix this HACK when we can use probeinterface:
            rel_x = float(rel_x)
            rel_y = float(rel_y)
            rel_z = float(rel_z)
            tmp_geom[i,:] = [rel_x, rel_y, rel_z]

        #figure out which columns have coordinates
        n_found = 0
        for i in range(3):
            if np.any(np.nonzero(tmp_geom[:,i])):
                if n_found < 2:
                    geometry[:,n_found] = tmp_geom[:,i]
                    n_found+=1
                else:
                    Warning(f'Relative electrode locations have three coordinates; only two are currenlty supported')
        return np.ndarray.tolist(geometry)

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
        sorters = ss.available_sorters()
        for sorter in sorters:
            self.insert1({'sorter_name' : sorter}, skip_duplicates="True")

@schema
class SpikeSorterParameters(dj.Manual):
    definition = """
    -> SpikeSorter
    parameter_set_name: varchar(80) # label for this set of parameters
    ---
    parameter_dict: blob # dictionary of parameter names and values
    filter_parameter_dict: blob # dictionary of filter parameter names and
    """

    def insert_from_spikeinterface(self):
        '''
        Add each of the default parameter dictionaries from spikeinterface.sorters
        :return: None
        '''
        # set up the default filter parameters
        frequency_min = 300 # high pass filter value
        frequency_max = 6000 # low pass filter value
        filter_width = 1000 # the number of coefficients in the filter
        filter_chunk_size = 2000000 # the size of the chunk for the filtering
        
        sort_param_dict = dict()
        sort_param_dict['parameter_set_name'] = 'default'
        sort_param_dict['filter_parameter_dict'] = {'frequency_min' : frequency_min, 
                                                    'frequency_max' : frequency_max,
                                                    'filter_width' : filter_width,
                                                    'filter_chunk_size' : filter_chunk_size}
        sorters = ss.available_sorters()
        for sorter in sorters:
            if len((SpikeSorter() & {'sorter_name' : sorter}).fetch()):
                sort_param_dict['sorter_name'] = sorter
                sort_param_dict['parameter_dict'] = ss.get_default_params(sorter)
                self.insert1(sort_param_dict, skip_duplicates=True)
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
    metric_dict: blob            # dict of SpikeInterface metrics with True / False elements to indicate whether a given metric should be computed.
    metric_parameter_dict: blob  # dict of parameters for the metrics
    """

    def get_metric_dict(self):
        """Get the current list of metrics from spike interface and create a
        dictionary with all False elemnets.
        Users should set the desired set of metrics to be true and insert a new
        entry for that set.
        """
        metrics_list =  st.validation.get_quality_metrics_list()
        metric_dict = {metric: False for metric in metrics_list}
        return metric_dict

    def get_metric_parameter_dict(self):
        """
        Get params for the metrics specified in the metric dict

        Parameters
        ----------
        metric_dict: dict
          a dictionary in which a key is the name of a quality metric and the value
          is a boolean
        """
        #TODO replace with call to spiketoolkit when available
        metric_params_dict =   {'isi_threshold': 0.003,                 # Interspike interval threshold in s for ISI metric (default 0.003)
                                'snr_mode': 'mad',                      # SNR mode: median absolute deviation ('mad) or standard deviation ('std') (default 'mad')
                                'snr_noise_duration': 10.0,             # length of data to use for noise estimation (default 10.0)
                                'max_spikes_per_unit_for_snr': 1000,    # Maximum number of spikes to compute templates for SNR from (default 1000)
                                'template_mode': 'mean',                # Use 'mean' or 'median' to compute templates
                                'max_channel_peak': 'both',             # direction of the maximum channel peak: 'both', 'neg', or 'pos' (default 'both')
                                'max_spikes_per_unit_for_noise_overlap': 1000, # Maximum number of spikes to compute templates for noise overlap from (default 1000)
                                'noise_overlap_num_features': 5,        # Number of features to use for PCA for noise overlap
                                'noise_overlap_num_knn' : 1,            # Number of nearest neighbors for noise overlap
                                'drift_metrics_interval_s': 60,         # length of period in s for evaluating drift (default 60 s)
                                'drift_metrics_min_spikes_per_interval': 10,    # Minimum number of spikes in an interval for evaluation of drift (default 10)
                                'max_spikes_for_silhouette': 1000,      # Max spikes to be used for silhouette metric
                                'num_channels_to_compare': 7,           # Number of channels to be used for the PC extraction and comparison (default 7)
                                'max_spikes_per_cluster': 1000,         # Max spikes to be used from each unit
                                'max_spikes_for_nn': 1000,              # Max spikes to be used for nearest-neighbors calculation
                                'n_neighbors': 4,                       # number of nearest clusters to use for nearest neighbor calculation (default 4)
                                'n_jobs': 24,                           # number of parallel jobs (default 96 in spiketoolkit, changed to 24)
                                'memmap': False,                        # If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
                                'max_spikes_per_unit': 2000,            # Max spikes to use for computing waveform
                                'seed': 47,                             # Random seed for reproducibility
                                'verbose': True}                        # If nonzero (True), will be verbose in metric computation
        return metric_params_dict
    
    def get_default_metrics_entry(self):
        """
        Re-inserts the entry for Frank lab default parameters
        (run in case it gets accidentally deleted)
        """
        cluster_metrics_list_name = 'franklab_default_cluster_metrics'
        metric_dict = self.get_metric_dict()
        metric_dict['firing_rate'] = True
        metric_dict['nn_hit_rate'] = True
        metric_dict['noise_overlap'] = True
        metric_parameter_dict = self.get_metric_parameter_dict()
        self.insert1([cluster_metrics_list_name,metric_dict,metric_parameter_dict], replace=True)

    @staticmethod
    def selected_metrics_list(metric_dict):
        return [metric for metric in metric_dict.keys() if metric_dict[metric]]

    def validate_metrics_list(self, key):
        """ Checks whether metrics_list contains only valid metric names

        :param key: key for metrics to validate
        :type key: dict
        :return: True or False
        :rtype: boolean
        """
        #TODO: get list of valid metrics from spiketoolkit when available
        valid_metrics = self.get_metric_dict()
        metric_dict = (self & key).fetch1('metric_dict')
        valid = True
        for metric in metric_dict:
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
        m = (self & {'cluster_metrics_list_name': key}).fetch1()

        return st.validation.compute_quality_metrics(sorting=sorting,
                                                     recording=recording,
                                                     metric_names=self.selected_metrics_list(m['metric_dict']),
                                                     as_dataframe=True,
                                                     **m['metric_parameter_dict'])

@schema
class SpikeSortingArtifactParameters(dj.Manual):
    definition = """
    # Table for holding parameters related to artifact detection
    artifact_param_name: varchar(200) #name for this set of parameters
    ---
    parameter_dict: BLOB    # dictionary of parameters for get_no_artifact_times() function
    """

    def get_no_artifact_times(self, recording, zscore_thresh=-1.0, amplitude_thresh=-1.0,
                              proportion_above_thresh = 1.0, zero_window_len = 1.0, skip:bool=True):
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
        if zscore_thresh <= 0 and amplitude_thresh <=0:
            return np.asarray([[recording._timestamps[0], recording._timestamps[recording.get_num_frames()]]])   

        half_window_points = np.round(recording.get_sampling_frequency() * 1000 * zero_window_len / 2)
        nelect_above = np.round(proportion_above_thresh * data.shape[0])
        # get the data traces
        data = recording.get_traces()
        
        # compute the number of electrodes that have to be above threshold based on the number of rows of data
        nelect_above = np.round(proportion_above_thresh * len(recording.get_channel_ids()))

        # apply the amplitude threshold
        above_a = np.abs(data) > amplitude_thresh

        #zscore the data and get the absolute value for thresholding
        dataz = np.abs(stats.zscore(data, axis=1))
        above_z = dataz > zscore_thresh

        above_both = np.ravel(np.argwhere(np.sum(np.logical_and(above_z, above_a), axis=0) >= nelect_above))
        valid_timestamps = recording._timestamps
        # for each above threshold point, set the timestamps on either side of it to -1
        for a in above_both:
            valid_timestamps[a-half_window_points:a+half_window_points] = -1
        
        # use get_valid_intervals to find all of the resulting valid times. 
        return get_valid_intervals(valid_timestamps[valid_timestamps !=-1 ], recording.get_sampling_frequency(), 1.5, 0.001)

@schema
class SpikeSortingParameters(dj.Manual):
    definition = """
    # Table for holding parameters for each spike sorting run
    -> SortGroup
    -> SpikeSorterParameters
    -> SortInterval
    ---
    -> SpikeSortingArtifactParameters
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
    units_object_id: varchar(40)           # Object ID for the units in NWB file
    time_of_sort=0: int                    # This is when the sort was done
    curation_feed_uri='': varchar(1000)    # Labbox-ephys feed for curation
    """

    def make(self, key):
        """
        Runs spike sorting on the data and parameters specified by the
        SpikeSortingParameter table and inserts a new entry to SpikeSorting table.
        Specifically,

        (1) Creates a new NWB file (analysis NWB file) that will hold the results of
            the sort (in .../analysis/)
        (2) Creates a se.RecordingExtractor
        (3) Creates a se.SortingExtractor based on (2) (i.e. runs the sort)
        (4) Saves (2) and (3) to another NWB file (in .../spikesorting/)
        (5) Creates a feed and workspace for curation based on labbox-ephys

        Parameters
        ----------
        key: dict
            partially filled entity; value of primary keys from key source
            (in this case SpikeSortingParameters)
        """
        
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])

        sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                          'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        sort_interval_valid_times = self.get_sort_interval_valid_times(key)

        # TODO: finish `import_sorted_data` function below

        with Timer(label='getting filtered recording extractor', verbose=True):
            recording = self.get_filtered_recording_extractor(key)
            recording_timestamps = recording._timestamps

        # get the artifact detection parameters and apply artifact detection to zero out artifacts
        artifact_key = (SpikeSortingParameters & key).fetch1('artifact_param_name')
        artifact_param_dict = (SpikeSortingArtifactParameters & {'artifact_param_name': artifact_key}).fetch1('parameter_dict')
        if not artifact_param_dict['skip']:
            no_artifact_valid_times = SpikeSortingArtifactParameters.get_no_artifact_times(recording, **artifact_param_dict)
            #update the sort interval valid times to exclude the artifacts
            sort_interval_valid_times = interval_list_intersect(sort_interval_valid_times, no_artifact_valid_times)
            # exclude the invalid times
            mask = np.full(recording.get_num_frames(), True, dtype='bool')
            mask[interval_list_excludes_ind(sort_interval_valid_times, recording_timestamps)] = False
            recording = st.preprocessing.mask(recording, mask)

        # Path to files that will hold recording and sorting extractors
        extractor_file_name = key['nwb_file_name'] \
                           + '_' + key['sort_interval_name'] \
                           + '_' + str(key['sort_group_id']) \
                           + '_' + key['sorter_name'] \
                           + '_' + key['parameter_set_name'] + '.nwb'
        analysis_path = str(Path(os.environ['SPIKE_SORTING_STORAGE_DIR'])
                            / key['analysis_file_name'])
        
        if not os.path.isdir(analysis_path):
            os.mkdir(analysis_path)
        extractor_nwb_path = str(Path(analysis_path) / extractor_file_name)

        metadata = {}
        metadata['Ecephys'] = {'ElectricalSeries': {'name': 'ElectricalSeries',
                                                    'description': key['nwb_file_name'] + \
                                                                   '_' + key['sort_interval_name'] + \
                                                                   '_' + str(key['sort_group_id'])}}
        with Timer(label=f'writing filtered NWB recording extractor to {extractor_nwb_path}', verbose=True):
            # TODO: save timestamps together
            #Caching the extractor GREATLY speeds up the subsequent processing and NWB writing
            tmpfile = tempfile.NamedTemporaryFile(dir='/stelmo/nwb/tmp')
            recording = se.CacheRecordingExtractor(recording, save_path=tmpfile.name, chunk_mb=1000, n_jobs=4) 
            #TODO: consider writing NWB or other recording extractor in a separate process
            se.NwbRecordingExtractor.write_recording(recording, save_path=extractor_nwb_path,
                                                    buffer_mb=10000, overwrite=True, metadata=metadata,
                                                    es_key='ElectricalSeries')

        # whiten the extractor for sorting and metric calculations
        print('\nWhitening recording...')
        with Timer(label=f'whiteneing', verbose=True):
            filter_params = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                                    'parameter_set_name': key['parameter_set_name']}).fetch1('filter_parameter_dict')
            recording = st.preprocessing.whiten(recording, seed=0, chunk_size=filter_params['filter_chunk_size'])

        print(f'\nRunning spike sorting on {key}...')
        sort_parameters = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                                    'parameter_set_name': key['parameter_set_name']}).fetch1()
        
        sorting = ss.run_sorter(key['sorter_name'], recording,
                                output_folder=os.getenv('SORTING_TEMP_DIR', None),
                                **sort_parameters['parameter_dict'])

        key['time_of_sort'] = int(time.time())
        
        # TODO: save timestamps
        se.NwbSortingExtractor.write_sorting(sorting, save_path=extractor_nwb_path)

        with Timer(label='computing quality metrics', verbose=True):
            # tmpfile = tempfile.NamedTemporaryFile(dir='/stelmo/nwb/tmp')
            # metrics_recording = se.CacheRecordingExtractor(recording, save_path=tmpfile.name, chunk_mb=10000)
            metrics_key = (SpikeSortingParameters & key).fetch1('cluster_metrics_list_name')    
            metric_info = (SpikeSortingMetrics & {'cluster_metrics_list_name': metrics_key }).fetch1()
            print(metric_info) 
            metrics = SpikeSortingMetrics().compute_metrics(metrics_key, recording, sorting)

        print('\nSaving sorting results...')
        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()
        unit_ids = sorting.get_unit_ids()
        for unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(unit_id = unit_id)
            units[unit_id] = recording_timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        # TODO: consider replacing with spikeinterface call if possible
        units_object_id, _ = AnalysisNwbfile().add_units(key['analysis_file_name'],
                                                         units, units_valid_times,
                                                         units_sort_interval,
                                                         metrics=metrics)

        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])
        key['units_object_id'] = units_object_id

        print('\nGenerating feed for curation...')
        extractor_nwb_uri = kp.link_file(extractor_nwb_path)
        print(f'kachery URI for symbolic link to extractor NWB file: {extractor_nwb_uri}')
        
        # create workspace
        workspace_uri = kp.get(key['analysis_file_name'])
        if not workspace_uri:
            workspace_uri = le.create_workspace(label=key['analysis_file_name']).uri
            kp.set(key['analysis_file_name'], workspace_uri)
        workspace = le.load_workspace(workspace_uri)
        print(f'Workspace URI: {workspace.uri}')
        
        recording_label = key['nwb_file_name']+'_'+key['sort_interval_name']+'_'+str(key['sort_group_id'])
        sorting_label = key['sorter_name']+'_'+key['parameter_set_name']

        recording_uri = kp.store_json({
            'recording_format': 'nwb',
            'data': {
                'path': extractor_nwb_path,
            }
        })
        sorting_uri = kp.store_json({
            'sorting_format': 'nwb',
            'data': {
                'path': extractor_nwb_path
            }
        })
        labbox_sorting = le.LabboxEphysSortingExtractor(sorting_uri)
        labbox_recording = le.LabboxEphysRecordingExtractor(recording_uri, download=True)

        R_id = workspace.add_recording(recording=labbox_recording, label=recording_label)
        S_id = workspace.add_sorting(sorting=labbox_sorting, recording_id=R_id, label=sorting_label)               
        
        key['curation_feed_uri'] = workspace.uri

        # Set external metrics that will appear in the units table
        external_metrics = [{'name': metric, 'label': metric, 'tooltip': metric,
                             'data': metrics[metric].to_dict()} for metric in metrics.columns]
        # change unit id to string
        for metric_ind in range(len(external_metrics)):
            for old_unit_id in metrics.index:
                external_metrics[metric_ind]['data'][str(old_unit_id)] = external_metrics[metric_ind]['data'].pop(old_unit_id)
        workspace.set_unit_metrics_for_sorting(sorting_id=S_id, metrics=external_metrics)
        
        # add to the web app as well
        workspace_list = sortingview.WorkspaceList(backend_uri='gs://labbox-franklab/sortingview-backends/franklab.json')
        workspace_list.add_workspace(name=key['analysis_file_name'], workspace=workspace)
        
        # TODO: extract labels 

        self.insert1(key)
        print('\nDone - entry inserted to table.')

    def get_stored_recording_sorting(self, key):
        """Retrieves the stored recording and sorting extractors given the key to a SpikeSorting

        Args:
            key (dict): key to retrieve one SpikeSorting entry
        """
        #TODO write this function
    

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def get_sort_interval_valid_times(self, key):
        """
        Identifies the intersection between sort interval specified by the user
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
        sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                          'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')
        interval_list_name = (SpikeSortingParameters & key).fetch1('interval_list_name')
        valid_times =  (IntervalList & {'nwb_file_name': key['nwb_file_name'],
                                        'interval_list_name': interval_list_name}).fetch1('valid_times')
        sort_interval_valid_times = interval_list_intersect(sort_interval, valid_times)
        return sort_interval_valid_times

    def get_filtered_recording_extractor(self, key):
        """
        Generates a RecordingExtractor object based on parameters in key.
        (1) Loads the NWB file created during insertion as a NwbRecordingExtractor
        (2) Slices the NwbRecordingExtractor in time (interval) and space (channels) to
            get a SubRecordingExtractor
        (3) Applies referencing, and bandpass filtering

        Parameters
        ----------
        key: dict

        Returns
        -------
        sub_R: se.SubRecordingExtractor
        """
        #print('In get_filtered_recording_extractor')
        with Timer(label='filtered recording extractor setup', verbose=True):
            nwb_file_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])
            with pynwb.NWBHDF5IO(nwb_file_abs_path,'r', load_namespaces=True) as io:
                nwbfile = io.read()
                timestamps = nwbfile.acquisition['e-series'].timestamps[:]
            
            # raw_data_obj = (Raw & {'nwb_file_name': key['nwb_file_name']}).fetch_nwb()[0]['raw']
            # timestamps = np.asarray(raw_data_obj.timestamps)

            sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                            'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')

            sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))
            assert sort_indices[1] - sort_indices[0] > 1000, f'Error in get_recording_extractor: sort indices {sort_indices} are not valid'

            electrode_ids = (SortGroup.SortGroupElectrode & {'nwb_file_name' : key['nwb_file_name'],
                                                            'sort_group_id' : key['sort_group_id']}).fetch('electrode_id')
            electrode_group_name = (SortGroup.SortGroupElectrode & {'nwb_file_name' : key['nwb_file_name'],
                                                                    'sort_group_id' : key['sort_group_id']}).fetch('electrode_group_name')
            electrode_group_name = np.int(electrode_group_name[0])
            probe_type = (Electrode & {'nwb_file_name': key['nwb_file_name'],
                                    'electrode_group_name': electrode_group_name,
                                    'electrode_id': electrode_ids[0]}).fetch1('probe_type')

        with Timer(label='NWB recording extractor create from file', verbose=True):
            R = se.NwbRecordingExtractor(Nwbfile.get_abs_path(key['nwb_file_name']),
                                     electrical_series_name = 'e-series')
        
        sub_R = se.SubRecordingExtractor(R, start_frame=sort_indices[0], end_frame=sort_indices[1])  
        
        sort_reference_electrode_id = int((SortGroup & {'nwb_file_name': key['nwb_file_name'],
                                                    'sort_group_id': key['sort_group_id']}).fetch('sort_reference_electrode_id'))
        
        # make a list of the channels in the sort group and the reference channel if it exists
        channel_ids = electrode_ids.tolist()

        if sort_reference_electrode_id >= 0:
            # make a list of the channels in the sort group and the reference channel if it exists
            channel_ids.append(sort_reference_electrode_id) 
       
        sub_R = se.SubRecordingExtractor(R, channel_ids=channel_ids,
                                         start_frame=sort_indices[0],
                                         end_frame=sort_indices[1]) 

        #Caching the extractor GREATLY speeds up the subsequent processing and NWB writing
        tmpfile = tempfile.NamedTemporaryFile(dir='/stelmo/nwb/tmp')
        sub_R = se.CacheRecordingExtractor(sub_R, save_path=tmpfile.name, chunk_mb=10000)
         
        if sort_reference_electrode_id >= 0:
            sub_R = st.preprocessing.common_reference(sub_R, reference='single',
                                                        ref_channels=sort_reference_electrode_id)
            # now restrict it to just the electrode IDs in the sort group
            sub_R = se.SubRecordingExtractor(sub_R, channel_ids=electrode_ids.tolist())

        elif sort_reference_electrode_id == -2:
            sub_R = st.preprocessing.common_reference(sub_R, reference='median')
            
        filter_params = (SpikeSorterParameters & {'sorter_name': key['sorter_name'],
                                                'parameter_set_name': key['parameter_set_name']}).fetch1('filter_parameter_dict')
        sub_R = st.preprocessing.bandpass_filter(sub_R, freq_min=filter_params['frequency_min'],
                                                freq_max=filter_params['frequency_max'],
                                                freq_wid=filter_params['filter_width'],
                                                chunk_size=filter_params['filter_chunk_size'],
                                                dtype='float32', )

        
        
        # Make sure the locations are set correctly
        sub_R.set_channel_locations(SortGroup().get_geometry(key['sort_group_id'],key['nwb_file_name']))

        # give timestamps for the SubRecordingExtractor
        # TODO: change this once spikeextractors is updated
        sub_R._timestamps = timestamps[sort_indices[0]:sort_indices[1]]

        return sub_R

    @staticmethod
    def get_recording_timestamps(key):
        """Returns the timestamps for the specified SpikeSorting entry

        Args:
            key (dict): the SpikeSorting key
        Returns:
            timestamps (numpy array)
        """
        nwb_file_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])
        #TODO fix to work with any electrical series object
        with pynwb.NWBHDF5IO(nwb_file_abs_path,'r', load_namespaces=True) as io:
            nwbfile = io.read()
            timestamps = nwbfile.acquisition['e-series'].timestamps[:]
        
        sort_interval =  (SortInterval & {'nwb_file_name': key['nwb_file_name'],
                                            'sort_interval_name': key['sort_interval_name']}).fetch1('sort_interval')

        sort_indices = np.searchsorted(timestamps, np.ravel(sort_interval))
        timestamps = timestamps[sort_indices[0]:sort_indices[1]]
        return timestamps

        
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

@schema
class AutomaticCurationParameters(dj.Manual):
    definition = """
    #Table for holding parameters for automatic aspects of curation
    automatic_curation_param_name: varchar(80)   #name of this parameter set
    ---
    automatic_curation_param_dict: BLOB         #dictionary of variables and values for automatic curation
    """

@schema
class AutomaticCurationSpikeSortingParameters(dj.Manual):
    definition = """
    # Table for holding the output 
    -> AutomaticCurationParameters
    -> SpikeSorting
    """

@schema
class AutomaticCurationSpikeSorting(dj.Computed):
    definition = """
    # Table for holding the output of automated curation applied to each spike sorting
    -> AutomaticCurationSpikeSortingParameters
    ---
    automatic_curation_results_dict=NULL: BLOB       #dictionary of outputs from automatic curation
    """
    def make(self, key):
        print(key)
        #TODO: add burst parent detection and noise waveform detection
        key['automatic_curation_results_dict'] = dict()
        self.insert1(key)

@schema
class CuratedSpikeSorting(dj.Computed):
    definition = """
    # Table for holding the output of fully curated spike sorting
    -> AutomaticCurationSpikeSorting
    ---
    -> AnalysisNwbfile    # New analysis NWB file to hold unit info
    curator:         varchar(80)            # the username of the person who is inserting the curation
    units_object_id: varchar(40)           # Object ID for the units in NWB file
    """

    class Unit(dj.Part):
        definition = """
        # Table for holding sorted units
        -> master
        unit_id: int            # ID for each unit
        ---
        label='' :              varchar(80)      # optional label for each unit
        noise_overlap=-1 :      float    # noise overlap metric for each unit
        nn_hit_rate=-1:         float  # isolation score metric for each unit
        isi_violation=-1:       float # ISI violation score for each unit
        firing_rate=-1:         float   # firing rate
        num_spikes=-1:          int          # total number of spikes
        """

    def make(self, key):
        #define the list of properties. TODO: get this from table definition.
        unit_properties = ['label', 'nn_hit_rate', 'noise_overlap', 'isi_violation', 'firing_rate', 'num_spikes']


        #Creating the curated units table involves 4 steps:
        # 1. Merging units labeled for merge
        # 2. Recalculate metrics 
        # 3. Inserting accepted units into new analysis NWB file and into the Curated Units table.

        #1. Merge
        # We can get the new curated soring from the workspace.
        workspace_uri = (SpikeSorting & key).fetch1('curation_feed_uri')
        workspace = le.load_workspace(workspace_uri=workspace_uri)
        # check that there is exactly one sorting in this workspace
        if len(workspace.sorting_ids) == 0:
            return
        elif len(workspace.sorting_ids) > 1:
            Warning(f'More than one sorting associated with {key}; delete extra sorting and try populate again')
            return

        #sorting = workspace.get_curated_sorting_extractor(workspace.sorting_ids[0])
        sorting = workspace.get_curated_sorting_extractor(workspace.sorting_ids[0])

        # Get labels 
        labels = workspace.get_sorting_curation(workspace.sorting_ids[0])

        # turn labels to list of str, only including accepted units.
        accepted_units = []
        unit_labels = labels['labelsByUnit']
        for idx, unitId in enumerate(unit_labels):
            if 'accept' in unit_labels[unitId]:
                accepted_units.append(unitId)
            if len(unit_labels[unitId]) == 0:
                Warning(f'In CuratedSpikeSorting: unit {unitId} has no curation labels. It will not be included in the table')

        #remove non-primary merged units
        for m in labels['mergeGroups']:
            for unit_id in m[1:]:
                accepted_units.remove(unit_id)

        # get the labels for the accepted units
        labels_concat = []
        for unitID in accepted_units:
            label_concat = ','.join(unit_labels[unitId])
            labels_concat.append(label_concat)

        print(f'Found {len(accepted_units)} accepted units')

        #exit out if there are no labels or no accepted units
        if len(unit_labels) == 0 or len(accepted_units) == 0:
            print(f'{key}: no curation found or no accepted units')
            return

        #2. Recalucate metrics for curated units to account for merges
        # get the recording extractor
        with Timer(label=f'Recomputing metrics', verbose=True):
            recording = workspace.get_recording_extractor(workspace.recording_ids[0])
            tmpfile = tempfile.NamedTemporaryFile(dir='/stelmo/nwb/tmp')
            recording = se.CacheRecordingExtractor(recording, save_path=tmpfile.name, chunk_mb=10000)
            metrics_key = (SpikeSortingParameters & key).fetch1('cluster_metrics_list_name')    
            metrics = SpikeSortingMetrics().compute_metrics(metrics_key, recording, sorting)

        #Limit the metrics to accepted units
        metrics = metrics.loc[accepted_units]
        
        #3. Save the accepted, merged units and their metrics
        # load the AnalysisNWBFile from the original sort to get the sort_interval_valid times and the sort_interval
        orig_units = (SpikeSorting & key).fetch_nwb()[0]['units'].to_dataframe()
        sort_interval = orig_units.iloc[1]['sort_interval']
        sort_interval_valid_times = orig_units.iloc[1]['obs_intervals']

        # add the units with the metrics and labels to the file.
        print('\nSaving curated sorting results...')
        timestamps = SpikeSorting.get_recording_timestamps(key)
        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()
        unit_ids = sorting.get_unit_ids()
        for unit_id in unit_ids:
            if unit_id in accepted_units:
                spike_times_in_samples = sorting.get_unit_spike_train(unit_id = unit_id)
                units[unit_id] = timestamps[spike_times_in_samples]
                units_valid_times[unit_id] = sort_interval_valid_times
                units_sort_interval[unit_id] = [sort_interval]

        # Create a new analysis NWB file 
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
 
        units_object_id, _ = AnalysisNwbfile().add_units(key['analysis_file_name'],
                                                         units, units_valid_times,
                                                         units_sort_interval,
                                                         metrics=metrics, labels=labels_concat)
        # add the analysis file to the table
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])
        key['units_object_id'] = units_object_id
        key['curator'] = getpass.getuser()

        # Insert entry to CuratedSpikeSorting table
        self.insert1(key)

        #Remove the non primary key entries.
        del key['units_object_id']
        del key['analysis_file_name']
        del key['curator']

        units_table = (CuratedSpikeSorting & key).fetch_nwb()[0]['units'].to_dataframe()

        # Add entries to CuratedSpikeSorting.Units table
        print('\nAdding to dj Unit table...')
        unit_key = key
        for unit_num, unit in units_table.iterrows():
            unit_key['unit_id'] = unit_num
            for property in unit_properties:
                if property in unit:
                    unit_key[property] = unit[property]
            CuratedSpikeSorting.Unit.insert1(unit_key)

        print('Done with dj Unit table.')

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)
