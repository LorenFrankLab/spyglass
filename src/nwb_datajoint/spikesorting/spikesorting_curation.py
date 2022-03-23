import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from warnings import WarningMessage
from xmlrpc.client import FastUnmarshaller

import datajoint as dj
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import numpy as np
from typing import List, Dict, Union
import time

from ..common.common_interval import IntervalList 
from ..common.common_lab import LabMember, LabTeam
from ..common.common_nwbfile import AnalysisNwbfile
from ..common.dj_helper_fn import fetch_nwb
from .spikesorting_recording import SortInterval, SpikeSortingRecording
from .spikesorting_sorting import SpikeSorting

schema = dj.schema('spikesorting_curation')

def apply_merge_groups_to_sorting(sorting: si.BaseSorting, merge_groups: List[List[int]]):
    # return a new sorting where the units are merged according to merge_groups
    # merge_groups is a list of lists of unit_ids.
    # for example: merge_groups = [[1, 2], [5, 8, 4]]]

    # A sorting segment in the new sorting
    class NewSortingSegment(si.BaseSortingSegment):
        def __init__(self):
            # Store all the unit spike trains in RAM
            self._unit_spike_trains: Dict[int, np.array] = {}

        def add_unit(self, unit_id: int, spike_times: np.array):
            # Add a unit spike train
            self._unit_spike_trains[unit_id] = spike_times

        def get_unit_spike_train(self,
            unit_id,
            start_frame: Union[int, None] = None,
            end_frame: Union[int, None] = None,
        ) -> np.ndarray:
            # Get a unit spike train
            spike_times = self._unit_spike_trains[unit_id]
            if start_frame is not None:
                spike_times = spike_times[spike_times >= start_frame]
            if end_frame is not None:
                spike_times = spike_times[spike_times < end_frame]
            return spike_times

    # Here's the new sorting we are going to return
    new_sorting = si.BaseSorting(
        sampling_frequency=sorting.get_sampling_frequency(),
        unit_ids=sorting.get_unit_ids()
    )
    # Loop through the sorting segments in the original sorting
    # and add merged versions to the new sorting
    for sorting_segment in sorting._sorting_segments: # Note: sorting_segments should be exposed in spikeinterface
        # initialize the new sorting segment
        new_sorting_segment = NewSortingSegment()
        # keep track of which unit_ids are part of the merge groups
        used_unit_ids = []
        # first loop through the merge groups
        for merge_group in merge_groups:
            # the representative unit_id is the min in the group
            representative_unit_id = min(merge_group)
            # we are going to take the union of all the spike trains for the merge group
            spike_trains_to_concatenate = []
            for unit_id in merge_group:
                spike_trains_to_concatenate.append(
                    sorting_segment.get_unit_spike_train(unit_id)
                )
                # append this unit_id to the used unit_ids
                used_unit_ids.append(unit_id)
            # concatenate the spike trains for this merge group
            spike_train = np.concatenate(spike_trains_to_concatenate)
            # sort the concatenated spike train (chronological)
            spike_train = np.sort(spike_train)
            # add the unit to the new sorting segment
            new_sorting_segment.add_unit(representative_unit_id, spike_train)
        # Now we'll take care of all of the unit_ids that are not part of a merge group
        for unit_id in sorting.get_unit_ids():
            if unit_id not in used_unit_ids:
                new_sorting_segment.add_unit(
                    unit_id,
                    sorting_segment.get_unit_spike_train(unit_id)
                )
        # add the new sorting segment to the new sorting
        new_sorting.add_sorting_segment(new_sorting_segment)
    # finally, return the new sorting
    return new_sorting 

@schema
class Curation(dj.Manual):
    definition = """
    # Stores each spike sorting; similar to IntervalList
    curation_id: varchar(10) #A unique identifier for each curation entry for convenience
    -> SpikeSorting
    ---
    curation_num = -1: int #  
    parent_curation_id='': varchar(10)
    labels: blob # a dictionary of labels for the units
    merge_groups: blob # a list of merge groups for the units
    metrics: blob # a list of quality metrics for the units (if available)
    description='': varchar(1000) #optional description for this curated sort
    time_of_creation: int   # in Unix time, to the nearest second
    """

    @staticmethod
    def insert_curation(sorting_key: dict, parent_curation_id: int = -1, labels=None, merge_groups=None, metrics=None, description=''):
        """Given a SpikeSorting key and the parent_sorting_id (and optional arguments) insert an entry into Curation

        :param sorting_key: the key for the original SpikeSorting
        :type dict
        :param parent_sorting_id: the id of the parent sorting (optional)
        :type parent_sorting_id: str
        :param metrics: computed metrics for sorting (optional)
        :type dict
        :param description: text description of this sort (optional)
        :type str
        :param labels: dictionary of labels for this sort (optional)
        :type dict

        returns:
        :param sorting_id
        :type str

        """
        if labels is None:
            labels = {}
        if merge_groups is None:
            merge_groups = []
        if metrics is None:
            metrics = {}

        # generate a unique ID
        num = (Curation & sorting_key).fetch('curation_id')
        if len(num) > 0:
            num.sort()
            curation_num = num[-1] + 1
        else:
            curation_num = 0

        # generate a unique ID string
        found = True
        while found:
            curation_id = 'C_' + str(uuid.uuid4())[:8]
            if len((Curation & {'curation_id': curation_id}).fetch()) == 0:
                found = False
        
        sorting_key['curation_id'] = curation_id
        sorting_key['curation_num'] = curation_num
        sorting_key['parent_curation_id'] = parent_curation_id
        sorting_key['description'] = description
        sorting_key['labels'] = labels
        sorting_key['merge_groups'] = merge_groups
        sorting_key['metrics'] = metrics
        sorting_key['time_of_creation'] = int(time.time())

        Curation.insert1(sorting_key)
        return sorting_key['curation_id']

    @staticmethod
    def get_recording_extractor(key):
        """Returns the recording extractor for the recording related to this curation

        :param key: key to a single Curation entry
        :type key: dict

        returns
            recording_extractor: spike interface recording extractor
        """
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        return si.load_extractor(recording_path)

    @staticmethod
    def get_curated_sorting_extractor(key):
        """Returns the sorting extractor related to this curation, with merges applied

        :param key: key to a single Curation entry
        :type key: dict

        returns
            sorting_extractor: spike interface sorting extractor
        """
        sorting_path = (SpikeSorting & key).fetch1('sorting_path')
        sorting = si.load_extractor(sorting_path)
        merge_groups = (Curation & key).fetch1('merge_groups')
        # TODO: write code to get merged sorting extractor
        if len(merge_groups) != 0:
            return apply_merge_groups_to_sorting(sorting, merge_groups)
        else:
            return sorting

    @staticmethod
    def save_sorting_nwb(key, sorting, timestamps, sort_interval_list_name,
                         sort_interval, labels=None, metrics=None, unit_ids=None):
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
        labels : dict, optional
            curation labels, by default None
        metrics : dict, optional
            quality metrics, by default None
        unit_ids : list, optional
            IDs of units whose spiketrains to save, by default None

        Returns
        -------
        analysis_file_name : str
        units_object_id : str
        """

        sort_interval_valid_times = (IntervalList &
                                     {'interval_list_name': sort_interval_list_name}).fetch1('valid_times')

        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        for unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(
                unit_id=unit_id)
            units[unit_id] = timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
        object_ids = AnalysisNwbfile().add_units(analysis_file_name,
                                                 units, units_valid_times,
                                                 units_sort_interval,
                                                 metrics=metrics, labels=labels)
        if object_ids == '':
            print('Sorting contains no units. Created an empty analysis nwb file anyway.')
            units_object_id = ''
        else:
            units_object_id = object_ids[0]
        return analysis_file_name,  units_object_id

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)


@schema
class WaveformParameters(dj.Manual):
    definition = """
    waveform_params_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """

    def insert_default(self):
        waveform_params_name = 'default'
        waveform_params = {'ms_before': 0.5, 'ms_after': 0.5, 'max_spikes_per_unit': 5000,
                           'n_jobs': 5, 'total_memory': '5G'}
        self.insert1([waveform_params_name, waveform_params],
                     skip_duplicates=True)


@schema
class WaveformSelection(dj.Manual):
    definition = """
    -> Curation
    -> WaveformParameters
    ---
    """


@schema
class Waveforms(dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    waveform_extractor_path: varchar(400)
    -> AnalysisNwbfile
    waveforms_object_id: varchar(40)   # Object ID for the waveforms in NWB file
    """

    def make(self, key):
        recording = Curation.get_recording_extractor(key)
        sorting = Curation.get_curated_sorting_extractor(key)

        print('Extracting waveforms...')
        waveform_params = (WaveformParameters & key).fetch1('waveform_params')
        if 'whiten' in waveform_params:
            if waveform_params['whiten']:
                recording = st.preprocessing.whiten(recording)
                # remove the 'whiten' dictionary entry as it is not recognized by spike interface
            del waveform_params['whiten']
  
        waveform_extractor_name = self._get_waveform_extractor_name(key)
        key['waveform_extractor_path'] = str(
            Path(os.environ['NWB_DATAJOINT_WAVEFORMS_DIR']) / Path(waveform_extractor_name))
        if os.path.exists(key['waveform_extractor_path']):
            shutil.rmtree(key['waveform_extractor_path'])
        waveforms = si.extract_waveforms(recording=recording,
                                         sorting=sorting,
                                         folder=key['waveform_extractor_path'],
                                         **waveform_params)

        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        object_id = AnalysisNwbfile().add_units_waveforms(key['analysis_file_name'],
                                                          waveform_extractor=waveforms)
        key['waveforms_object_id'] = object_id
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])

        self.insert1(key)

    def load_waveforms(self, key: dict):
        """Returns a spikeinterface waveform extractor specified by key

        Parameters
        ----------
        key : dict
            Could be an entry in Waveforms, or some other key that uniquely defines
            an entry in Waveforms

        Returns
        -------
        we : spikeinterface.WaveformExtractor
        """
        we_path = (self & key).fetch1('waveform_extractor_path')
        we = si.WaveformExtractor.load_from_folder(we_path)
        return we

    def fetch_nwb(self, key):
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError

    def _get_waveform_extractor_name(self, key):
        we_name = str(key['curation_id']) + '_waveform'
        return we_name


@schema
class MetricParameters(dj.Manual):
    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_params_name: varchar(200)
    ---
    metric_params: blob
    """
    metric_default_params = {
        'snr': {'peak_sign': 'neg',
                'num_chunks_per_segment': 20,
                'chunk_size': 10000,
                'seed': 0},
        'isi_violation': {'isi_threshold_ms': 1.5,
                          'min_isi_ms' : 0.0},
        'nn_isolation': {'max_spikes_for_nn': 1000,
                         'n_neighbors': 5,
                         'n_components': 7,
                         'radius_um': 100,
                         'seed': 0},
        'nn_noise_overlap': {'max_spikes_for_nn': 1000,
                             'n_neighbors': 5,
                             'n_components': 7,
                             'radius_um': 100,
                             'seed': 0},
        'peak_offset' : {'peak_sign' : 'neg'}
    }
    available_metrics = list(metric_default_params.keys())

    def get_metric_default_params(self, metric: str):
        "Returns default params for the given metric"
        return self.metric_default_params(metric)

    def insert_default(self):
        self.insert1(
            ['franklab_default', self.metric_default_params], skip_duplicates=True)

    # TODO
    def _validate_metrics_list(self, key):
        """Checks whether a row to be inserted contains only the available metrics
        """
        # get available metrics list
        # get metric list from key
        # compare
        return NotImplementedError


@schema
class MetricSelection(dj.Manual):
    definition = """
    -> Waveforms
    -> MetricParameters
    ---
    """


@schema
class QualityMetrics(dj.Computed):
    definition = """
    -> MetricSelection
    ---
    quality_metrics_path: varchar(500)
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    def make(self, key):
        waveform_extractor = Waveforms().load_waveforms(key)
        qm = {}
        params = (MetricParameters & key).fetch1('metric_params')
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(
                waveform_extractor, metric_name, **metric_params)
            qm[metric_name] = metric
        qm_name = self._get_quality_metrics_name(key)
        key['quality_metrics_path'] = str(
            Path(os.environ['NWB_DATAJOINT_WAVEFORMS_DIR']) / Path(qm_name + '.json'))
        # save metrics dict as json
        print(f'Computed all metrics: {qm}')
        self._dump_to_json(qm, key['quality_metrics_path'])

        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        key['object_id'] = AnalysisNwbfile().add_units_metrics(
            key['analysis_file_name'], metrics=qm)
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])

        self.insert1(key)

    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms()._get_waveform_extractor_name(key)
        qm_name = wf_name + '_qm'
        return qm_name

    def _compute_metric(self, waveform_extractor, metric_name, **metric_params):
        metric_func = _metric_name_to_func[metric_name]
        #TODO clean up code below
        if metric_name == 'isi_violation':
            metric = metric_func(waveform_extractor, **metric_params)
        elif (metric_name == 'snr' or
            metric_name == 'peak_offset'):
            peak_sign = metric_params['peak_sign']
            del metric_params['peak_sign']
            metric = metric_func(waveform_extractor, peak_sign=peak_sign, **metric_params)
        else:
            metric = {str(unit_id) : metric_func(waveform_extractor,
                            this_unit_id=unit_id, **metric_params)
                            for unit_id in 
                            waveform_extractor.sorting.get_unit_ids()}
        return metric

    def _dump_to_json(self, qm_dict, save_path):
        new_qm = {}
        for key, value in qm_dict.items():
            m = {}
            for unit_id, metric_val in value.items():
                m[str(unit_id)] = metric_val
            new_qm[str(key)] = m
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_qm, f, ensure_ascii=False, indent=4)


def _compute_isi_violation_fractions(waveform_extractor, **metric_params):
    isi_threshold_ms = metric_params['isi_threshold_ms']
    min_isi_ms = metric_params['min_isi_ms']

    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = st.qualitymetrics.compute_isi_violations(
        waveform_extractor, isi_threshold_ms=isi_threshold_ms, min_isi_ms=min_isi_ms).isi_violations_count

    # Extract the total number of spikes from each unit
    num_spikes = st.qualitymetrics.compute_num_spikes(waveform_extractor)
    isi_viol_frac_metric = {str(unit_id) : isi_violation_counts[unit_id] /
        num_spikes[unit_id] for unit_id in waveform_extractor.sorting.get_unit_ids()}
    return isi_viol_frac_metric

def _get_peak_offset(waveform_extractor:si.WaveformExtractor, peak_sign:str, **metric_params):
    if 'peak_sign' in metric_params:
        del metric_params['peak_sign']
    peak_offset_inds = st.get_template_extremum_channel_peak_shift(
                        waveform_extractor=waveform_extractor,
                        peak_sign=peak_sign, **metric_params)
    peak_offset = {key : np.int16(val) for key,val in peak_offset_inds.items()}
    return peak_offset

_metric_name_to_func = {
    "snr": st.qualitymetrics.compute_snrs,
    "isi_violation": _compute_isi_violation_fractions,
    'nn_isolation': st.qualitymetrics.nearest_neighbors_isolation,
    'nn_noise_overlap': st.qualitymetrics.nearest_neighbors_noise_overlap,
    'peak_offset' : _get_peak_offset
}


@schema
class AutomaticCurationParameters(dj.Manual):
    definition = """
    auto_curation_params_name: varchar(200)   # name of this parameter set
    ---
    merge_params: blob   # params to merge units
    label_params: blob   # params to label units
    """

    def insert_default(self):
        auto_curation_params_name = 'default'
        merge_params = {}
        # label_params parsing: Each key is the name of a metric, 
        # the contents are a three value list with the comparison, a value, and the label if the comparison is true
        label_params = {'nn_noise_overlap' : ['>', 0.1, 'noise']}
        self.insert1([auto_curation_params_name, merge_params,
                     label_params], skip_duplicates=True)


@schema
class AutomaticCurationSelection(dj.Manual):
    definition = """
    -> QualityMetrics
    -> AutomaticCurationParameters
    """

_comparison_to_function = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}

@schema
class AutomaticCuration(dj.Computed):
    definition = """
    -> AutomaticCurationSelection
    ---
    auto_curation_id: varchar(10) # the ID of this curation, matches curation_id in Curation
    """

    def make(self, key):
        metrics_path = (QualityMetrics & key).fetch1('quality_metrics_path')
        with open(metrics_path) as f:
            quality_metrics = json.load(f)

        # get the curation information and the curated sorting
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_merge_groups = parent_curation['merge_groups']
        parent_labels = parent_curation['labels']
        parent_curation_id = parent_curation['parent_curation_id']
        parent_sorting = Curation.get_curated_sorting_extractor(key)

        merge_params = (AutomaticCurationParameters &
                        key).fetch1('merge_params')
        merge_groups, units_merged = self.get_merge_groups(
            parent_sorting, parent_merge_groups, quality_metrics, merge_params)
        if units_merged:
            # get merged sorting extractor here
            sorting = apply_merge_groups_to_sorting(parent_sorting, merge_groups=merge_groups)
        else:
            sorting = parent_sorting

        label_params = (AutomaticCurationParameters &
                        key).fetch1('label_params')
        labels = self.get_labels(
            sorting, parent_labels, quality_metrics, label_params)

        # keep the quality metrics only if no merging occurred.
        metrics = quality_metrics if not units_merged else None

        # insert this sorting into the CuratedSpikeSorting Table
        # first remove keys that aren't part of the Sorting (the primary key of curation)
        c_key = (SpikeSorting & key).fetch("KEY")[0]
        curation_key = {item : key[item] for item in key if item in c_key}
        key['auto_curation_id'] = Curation.insert_curation(curation_key, parent_curation_id=parent_curation_id,
                                                     labels=labels, merge_groups=merge_groups, metrics=metrics, description='auto curated')

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    @staticmethod
    def get_merge_groups(sorting, parent_merge_groups, quality_metrics, merge_params):
        """ Identifies units to be merged based on the quality_metrics and merge parameters and returns an updated list of merges for the curation

        :param sorting: sorting object
        :type sorting: spike interface sorting object
        :param parent_merge_groups: information about previous merges
        :type quality_metrics: list
        :param quality_metrics: dictionary of quality metrics by unit
        :type quality_metrics: dict
        :param merge_params: dictionary of merge parameters
        :type merge_params: dict
        :return: merge_groups, merge_occurred
        :rtype: list of lists, bool
        """  # overview:
        # 1. Use quality metrics to determine merge groups for units
        # 2. Combine merge groups with current merge groups to produce union of merges

        if not merge_params:
            return parent_merge_groups, False
        else:
            # TODO: use the metrics to identify clusters that should be merged
            new_merges = []
            # append these merges to the parent merge_groups
            for m in new_merges:
                # check to see if the first cluster listed is in a current merge group
                found = False
                for idx, pm in enumerate(parent_merge_groups):
                    if m[0] == pm[0]:
                        # add the additional units in m to the identified merge group.
                        pm.extend(m[1:])
                        pm.sort()
                        found = True
                        break;
                if not found:
                    #append this merge group to the list
                    parent_merge_groups.append(m)
            return parent_merge_groups.sort(), True
   

    @staticmethod
    def get_labels(sorting, parent_labels, quality_metrics, label_params):
        """ Returns a dictionary of labels using quality_metrics and label parameters

        :param sorting: sorting object
        :type sorting: spike interface sorting object
        :param parent_labels: previously applied labels
        :param quality_metrics: dictionary of quality metrics by unit
        :type quality_metrics: dict
        :param label_params: dictionary of label parameters
        :type labe_params: dict
        :return: labels
        :rtype: dict
        """
        # overview:
        # 1. Use quality metrics to determine labels for units
        # 2. Append labels to current labels, checking for inconsistencies
        if not label_params:
            return parent_labels
        else:
            for label in label_params:
                if not label in quality_metrics:
                    Warning(f'{label} not found in quality metrics; skipping')
                else:
                    for unit_id in quality_metrics[label].keys():
                        # compare the quality metric to the threshold with the specified operator
                        # note that label_params[label] is a three element list with a comparison operator as a string, 
                        # the threshold value, and the label to be applied if the comparison is true
                        if _comparison_to_function[label_params[label][0]](quality_metrics[label][unit_id], label_params[label][1]):
                            if unit_id not in parent_labels:
                                parent_labels[unit_id] = [label_params[label][2]]
                            # check if the label is already there, and if not, add it
                            elif label_params[label][2] not in parent_labels[unit_id]:
                                parent_labels[unit_id].extend(label_params[label][2])
            return parent_labels



@schema
class FinalizedSpikeSortingSelection(dj.Manual):
    definition = """
    -> Curation
    """

@schema
class FinalizedSpikeSorting(dj.Manual):
    definition = """
    -> FinalizedSpikeSortingSelection
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40)
    """

    class Unit(dj.Part):
        definition = """
        # Table for holding sorted units
        -> FinalizedSpikeSorting
        unit_id: int   # ID for each unit
        ---
        label='': varchar(200)   # optional set of labels for each unit
        nn_noise_overlap=-1: float   # noise overlap metric for each unit
        nn_isolation=-1: float   # isolation score metric for each unit
        isi_violation=-1: float   # ISI violation score for each unit
        snr=0: float            # SNR for each unit
        firing_rate=-1: float   # firing rate
        num_spikes=-1: int   # total number of spikes
        """

    def make(self, key):
        unit_labels_to_remove = ['reject', 'noise']
        # check that the Curation has metrics
        metrics = (Curation & key).fetch1('metrics')
        if metrics == {}:
            Warning(f'Metrics for Curation {key} should normally be calculated before insertion here')

        sorting = Curation.get_curated_sorting_extractor(key)
        unit_ids = sorting.get_unit_ids()
        # Get the labels for the units, add only those units that do not have 'reject' or 'noise' labels
        unit_labels = (Curation & key).fetch1('labels')
        accepted_units = []
        for unit_id in unit_ids:
            if unit_id in unit_labels:
                if len(set(unit_labels_to_remove) & set(unit_labels[unit_id])) == 0:
                    accepted_units.append(unit_id)
            else:
                accepted_units.append(unit_id)

        # get the labels for the accepted units
        labels = {}
        for unit_id in accepted_units:
            labels[unit_id] = ','.join(unit_labels[unit_id])

        # Limit the metrics to accepted units
        metrics = metrics.loc[accepted_units]

        print(f'Found {len(accepted_units)} accepted units')

        # exit out if there are no labels or no accepted units
        if len(unit_labels) == 0 or len(accepted_units) == 0:
            print(f'{key}: no accepted units found')
            return

        # get the sorting and save it in the NWB file
        sorting = Curation.get_curated_sorting_extractor(key)
        recording = Curation.get_recording_extractor(key)

        # get the original units from the Automatic curation NWB file to get the sort interval information
        orig_units = (SpikeSorting & key).fetch_nwb()[0]['units']
        orig_units = orig_units.loc[accepted_units]
        # TODO: fix if unit 0 doesn't exist
        sort_interval = orig_units.iloc[0]['sort_interval']
        sort_interval_list_name = (SpikeSortingRecording & key).fetch1(
            'sort_interval_list_name')

        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)

        key['analysis_file_name'], key['units_object_id'] = Curation.save_sorting_nwb(self, key, sorting, timestamps, sort_interval_list_name,
                                                                                      sort_interval, metrics=metrics, unit_ids=accepted_units)
        self.insert1(key)

        # now add the units
        # Remove the non primary key entries.
        del key['units_object_id']
        del key['analysis_file_name']

        metric_fields = self.metrics_fields()

        for unit_id in accepted_units:
            key['unit_id'] = unit_id
            key['label'] = labels[unit_id]
            for field in metric_fields:
                if field in metrics[unit_id]:
                    key[field] = metrics[unit_id][field]
                else:
                    WarningMessage(
                        f'No metric named {field} in metrics table; skipping')
        self.Unit.insert1(key)

    def metrics_fields(self):
        """Returns a list of the metrics that are currently in the Units table
        """
        unit_fields = list(self.Unit().fetch(limit=1, as_dict=True)[0].keys())
        parent_fields = list(FinalizedSpikeSorting.fetch(
            limit=1, as_dict=True)[0].keys())
        parent_fields.extend(['unit_id', 'label'])
        return [field for field in unit_fields if field not in parent_fields]

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)


#         # 3. Save the accepted, merged units and their metrics
#         # load the AnalysisNWBFile from the original sort to get the sort_interval_valid times and the sort_interval
#         key['analysis_file_name'], key['units_object_id'] = \
#             SortingID.store_sorting_nwb(key, sorting=sorting, sort_interval_list_name=sort_interval_list_name,
#                               sort_interval=sort_interval, metrics=metrics, unit_ids=accepted_units)

#         # Insert entry to CuratedSpikeSorting table
#         self.insert1(key)


#     def delete(self):
#         """
#         Extends the delete method of base class to implement permission checking
#         """
#         current_user_name = dj.config['database.user']
#         entries = self.fetch()
#         permission_bool = np.zeros((len(entries),))
#         print(f'Attempting to delete {len(entries)} entries, checking permission...')

#         for entry_idx in range(len(entries)):
#             # check the team name for the entry, then look up the members in that team, then get their datajoint user names
#             team_name = (SpikeSortingRecordingSelection & (SpikeSortingRecordingSelection & entries[entry_idx]).proj()).fetch1()['team_name']
#             lab_member_name_list = (LabTeam.LabTeamMember & {'team_name': team_name}).fetch('lab_member_name')
#             datajoint_user_names = []
#             for lab_member_name in lab_member_name_list:
#                 datajoint_user_names.append((LabMember.LabMemberInfo & {'lab_member_name': lab_member_name}).fetch1('datajoint_user_name'))
#             permission_bool[entry_idx] = current_user_name in datajoint_user_names
#         if np.sum(permission_bool)==len(entries):
#             print('Permission to delete all specified entries granted.')
#             super().delete()
#         else:
#             raise Exception('You do not have permission to delete all specified entries. Not deleting anything.')

#     def fetch_nwb(self, *attrs, **kwargs):
#         return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

#     def delete_extractors(self, key):
#         """Delete directories with sorting and recording extractors that are no longer needed

#         :param key: key to curated sortings where the extractors can be removed
#         :type key: dict
#         """
#         # get a list of the files in the spike sorting storage directory
#         dir_names = next(os.walk(os.environ['SPIKE_SORTING_STORAGE_DIR']))[1]
#         # now retrieve a list of the currently used analysis nwb files
#         analysis_file_names = (self & key).fetch('analysis_file_name')
#         delete_list = []
#         for dir in dir_names:
#             if not dir in analysis_file_names:
#                 delete_list.append(dir)
#                 print(f'Adding {dir} to delete list')
#         delete = input('Delete all listed directories (y/n)? ')
#         if delete == 'y' or delete == 'Y':
#             for dir in delete_list:
#                 shutil.rmtree(dir)
#             return
#         print('No files deleted')
#     # def delete(self, key)

# @schema
# class SelectedUnitsParameters(dj.Manual):
#     definition = """
#     unit_inclusion_param_name: varchar(80) # the name of the list of thresholds for unit inclusion
#     ---
#     max_noise_overlap=1:        float   # noise overlap threshold (include below)
#     min_nn_isolation=-1:        float   # isolation score threshold (include above)
#     max_isi_violation=100:      float   # ISI violation threshold
#     min_firing_rate=0:          float   # minimum firing rate threshold
#     max_firing_rate=100000:     float   # maximum fring rate thershold
#     min_num_spikes=0:           int     # minimum total number of spikes
#     exclude_label_list=NULL:    BLOB    # list of labels to EXCLUDE
#     """

#     def get_included_units(self, curated_sorting_key, unit_inclusion_key):
#         """given a reference to a set of curated sorting units and a specific unit inclusion parameter list, returns
#         the units that should be included

#         :param curated_sorting_key: key to entries in CuratedSpikeSorting.Unit table
#         :type curated_sorting_key: dict
#         :param unit_inclusion_key: key to a single unit inclusion parameter set
#         :type unit_inclusion_key: dict
#         """
#         curated_sortings = (CuratedSpikeSorting() & curated_sorting_key).fetch()
#         inclusion_key = (self & unit_inclusion_key).fetch1()

#         units = (CuratedSpikeSorting().Unit() & curated_sortings).fetch()
#         # get a list of the metrics in the units table
#         metrics_list = CuratedSpikeSorting().metrics_fields()
#         # create a list of the units to kepp.
#         #TODO: make this code more flexible
#         keep = np.asarray([True] * len(units))
#         if 'noise_overlap' in metrics_list and "max_noise_overlap" in inclusion_key:
#             keep = np.logical_and(keep, units['noise_overlap'] <= inclusion_key["max_noise_overlap"])
#         if 'nn_isolation' in metrics_list and "min_isolation" in inclusion_key:
#             keep = np.logical_and(keep, units['nn_isolation'] >= inclusion_key["min_isolation"])
#         if 'isi_violation' in metrics_list and "isi_violation" in inclusion_key:
#             keep = np.logical_and(keep, units['isi_violation'] <= inclusion_key["isi_violation"])
#         if 'firing_rate' in metrics_list and "firing_rate" in inclusion_key:
#             keep = np.logical_and(keep, units['firing_rate'] >= inclusion_key["min_firing_rate"])
#             keep = np.logical_and(keep, units['firing_rate'] <= inclusion_key["max_firing_rate"])
#         if 'num_spikes' in metrics_list and "min_num_spikes" in inclusion_key:
#             keep = np.logical_and(keep, units['num_spikes'] >= inclusion_key["min_num_spikes"])
#         units = units[keep]
#         #now exclude by label if it is specified
#         if inclusion_key['exclude_label_list'] is not None:
#             included_units = []
#             for unit in units:
#                 labels = unit['label'].split(',')
#                 exclude = False
#                 for label in labels:
#                     if label in inclusion_key['exclude_label_list']:
#                         exclude = True
#                 if not exclude:
#                     included_units.append(unit)
#             return included_units
#         else:
#             return units

# @schema
# class SelectedUnits(dj.Computed):
#     definition = """
#     -> CuratedSpikeSorting
#     -> SelectedUnitsParameters
#     ---
#     -> AnalysisNwbfile   # New analysis NWB file to hold unit info
#     units_object_id: varchar(40)   # Object ID for the units in NWB file
#     """
# Note this is completely untested

