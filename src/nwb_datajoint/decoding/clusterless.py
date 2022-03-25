import pprint

import datajoint as dj
import labbox_ephys as le
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import xarray as xr
from nwb_datajoint.common.common_interval import IntervalList
from nwb_datajoint.common.common_nwbfile import AnalysisNwbfile
from nwb_datajoint.common.common_position import IntervalPositionInfo
from nwb_datajoint.common.dj_helper_fn import fetch_nwb
from nwb_datajoint.decoding.core import _convert_transitions_to_dict, _to_dict
from nwb_datajoint.spikesorting.spikesorting_curation import \
    CuratedSpikeSorting
from replay_trajectory_classification.classifier import (
    _DEFAULT_CLUSTERLESS_MODEL_KWARGS, _DEFAULT_CONTINUOUS_TRANSITIONS,
    _DEFAULT_ENVIRONMENT)
from replay_trajectory_classification.discrete_state_transitions import \
    DiagonalDiscrete
from replay_trajectory_classification.initial_conditions import \
    UniformInitialConditions
from ripple_detection import (get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)

schema = dj.schema('decoding_clusterless')


@schema
class MarkParameters(dj.Manual):
    definition = """
    mark_param_name : varchar(80) # a name for this set of parameters
    ---
    # the type of mark. Currently only 'amplitude' is supported
    mark_type = 'amplitude':  varchar(40)
    mark_param_dict:    BLOB    # dictionary of parameters for the mark extraction function
    """

    def insert_default(self):
        """insert the default parameter set {'sign': -1, 'threshold' : 0} corresponding to negative going waveforms of at least 100 uV size
        """
        default_dict = {'sign': -1, 'threshold': 0}
        self.insert1({'mark_param_name': 'default',
                      'mark_param_dict': default_dict}, skip_duplicates=True)

    @staticmethod
    def supported_mark_type(mark_type):
        """checks whether the requested mark type is supported. Currently only 'amplitude" is supported

        Args:
            mark_type (str): the requested mark type
        """
        supported_types = ['amplitude']
        if mark_type in supported_types:
            return True
        return False

@schema
class MarkSource(dj.Manual):
    definition = """
    ---
    source
    mark_param_dict:    BLOB    # dictionary of parameters for the mark extraction function
    """


@schema
class UnitMarkParameters(dj.Manual):
    definition = """
    -> MarkParameters
    """


@schema
class UnitMarks(dj.Computed):
    definition = """
    -> UnitMarkParameters
    ---
    -> AnalysisNwbfile
    marks_object_id: varchar(40) # the NWB object that stores the marks
    """

    def make(self, key):
        # get the list of mark parameters
        mark_param = (MarkParameters & key).fetch1()
        mark_param_dict = mark_param['mark_param_dict']

        # check that the mark type is supported
        if not MarkParameters().supported_mark_type(mark_param['mark_type']):
            Warning(
                f'Mark type {mark_param["mark_type"]} not supported; skipping')
            return

        # get the list of units
        units = SelectedUnitsParameters().get_included_units(
            curated_sorting_key=key,
            unit_inclusion_key=key)

        # retrieve the units from the NWB file
        nwb_units = (CuratedSpikeSorting() & key).fetch_nwb()[0]['units']

        # get the labbox workspace so we can get the waveforms from the recording
        workspace_uri = (SortingviewWorkspace & key).fetch1('workspace_uri')
        workspace = le.load_workspace(workspace_uri)
        recording = workspace.get_recording_extractor(
            workspace.recording_ids[0])
        sorting = workspace.get_sorting_extractor(workspace.sorting_ids[0])
        channel_ids = recording.get_channel_ids()
        # assume the channels are all the same for the moment. This would need to be changed for larger probes
        channel_ids_by_unit = [channel_ids] * (max(units['unit_id']) + 1)

        N_WAVEFORM_POINTS = 2
        waveforms = le.get_unit_waveforms(
            recording,
            sorting,
            units['unit_id'],
            channel_ids_by_unit,
            N_WAVEFORM_POINTS)

        if mark_param['mark_type'] == 'amplitude':
            # peak is stored in the middle of the waveform
            peak_ind = N_WAVEFORM_POINTS // 2
            marks = np.concatenate(waveforms, axis=0)[
                :, :, peak_ind].astype(np.int16)
            timestamps = np.concatenate(np.asarray(
                nwb_units.loc[units['unit_id']].spike_times))

            # sort the timestamps to order them properly
            sort_order = np.argsort(timestamps)
            timestamps = timestamps[sort_order]
            marks = marks[sort_order, :]

            if 'threshold' in mark_param_dict:
                print('thresholding')
                # filter the marks by the amplitude threshold
                if mark_param_dict['sign'] == -1:
                    threshold = (mark_param_dict['sign'] *
                                 mark_param_dict['threshold'])
                    include = np.min(marks, axis=1) <= threshold
                elif mark_param_dict['sign'] == 1:
                    threshold = (mark_param_dict['sign'] *
                                 mark_param_dict['threshold'])
                    include = np.max(marks, axis=1) >= threshold
                else:
                    threshold = mark_param_dict['sign']
                    include = np.max(np.abs(marks), axis=1) >= threshold
                timestamps = timestamps[include]
                marks = marks[include, :]

            # create a new AnalysisNwbfile and a timeseries for the marks and save
            key['analysis_file_name'] = AnalysisNwbfile().create(
                key['nwb_file_name'])
            nwb_object = pynwb.TimeSeries(
                name='marks',
                data=marks,
                unit='uV',
                timestamps=timestamps,
                description=f'amplitudes of spikes from electrodes {recording.get_channel_ids()}')
            key['marks_object_id'] = (AnalysisNwbfile().add_nwb_object(
                key['analysis_file_name'], nwb_object))
            AnalysisNwbfile().add(key['nwb_file_name'],
                                  key['analysis_file_name'])
            self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [self._convert_to_dataframe(data) for data in self.fetch_nwb()]

    @staticmethod
    def _convert_to_dataframe(nwb_data):
        n_marks = nwb_data['marks'].data.shape[1]
        columns = [f'amplitude_{ind}' for ind in range(n_marks)]
        return pd.DataFrame(nwb_data['marks'].data,
                            index=pd.Index(nwb_data['marks'].timestamps,
                                           name='time'),
                            columns=columns)


@schema
class UnitMarksIndicatorSelection(dj.Lookup):
    definition = """
    -> UnitMarks
    -> IntervalList
    sampling_rate=500 : float
    ---
    """


@schema
class UnitMarksIndicator(dj.Computed):
    definition = """
    -> UnitMarks
    -> UnitMarksIndicatorSelection
    ---
    -> AnalysisNwbfile
    marks_indicator_object_id: varchar(40)
    """

    def make(self, key):
        pprint.pprint(key)
        # TODO: intersection of sort interval and interval list
        interval_times = (IntervalList &
                          {
                              'nwb_file_name': key['nwb_file_name'],
                              'interval_list_name': key['interval_list_name']
                          }
                          ).fetch1('valid_times')

        sampling_rate = (UnitMarksIndicatorSelection & {
            'nwb_file_name': key['nwb_file_name'],
            'sort_interval_name': key['sort_interval_name'],
            'filter_parameter_set_name': key['filter_parameter_set_name'],
            'sorting_id': key['sorting_id'],
            'unit_inclusion_param_name': key['unit_inclusion_param_name'],
            'mark_param_name': key['mark_param_name'],
            'interval_list_name': key['interval_list_name']
        }).fetch('sampling_rate')

        marks_df = (UnitMarks & {
            'nwb_file_name': key['nwb_file_name'],
            'sort_interval_name': key['sort_interval_name'],
            'filter_parameter_set_name': key['filter_parameter_set_name'],
            'sorting_id': key['sorting_id'],
            'unit_inclusion_param_name': key['unit_inclusion_param_name'],
            'mark_param_name': key['mark_param_name'],
        }).fetch1_dataframe()

        time = self.get_time_bins_from_interval(interval_times, sampling_rate)

        # Bin marks into time bins. No spike bins will have NaN
        marks_df = marks_df.loc[time.min():time.max()]
        time_index = np.digitize(marks_df.index, time[1:-1])
        marks_indicator_df = (marks_df
                              .groupby(time[time_index])
                              .mean()
                              .reindex(index=pd.Index(time, name='time')))

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key['analysis_file_name'] = nwb_analysis_file.create(
            key['nwb_file_name'])

        key['marks_indicator_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=marks_indicator_df.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        self.insert1(key)

    @staticmethod
    def get_time_bins_from_interval(interval_times, sampling_rate):
        start_time, end_time = interval_times[0][0], interval_times[-1][-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

        return np.linspace(start_time, end_time, n_samples)

    @staticmethod
    def plot_all_marks(marks_indicators):
        for electrode_ind in marks_indicators.electrodes:
            marks = marks_indicators.sel(electrodes=electrode_ind).dropna(
                'time', how='all').dropna('marks')
            n_features = len(marks.marks)
            fig, axes = plt.subplots(n_features, n_features,
                                     constrained_layout=True, sharex=True, sharey=True,
                                     figsize=(5 * n_features, 5 * n_features))
            for ax_ind1, feature1 in enumerate(marks.marks):
                for ax_ind2, feature2 in enumerate(marks.marks):
                    try:
                        axes[ax_ind1, ax_ind2].scatter(
                            marks.sel(marks=feature1), marks.sel(marks=feature2), s=10)
                    except TypeError:
                        axes.scatter(marks.sel(marks=feature1),
                                     marks.sel(marks=feature2), s=10)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data['marks_indicator'].set_index('time') for data in self.fetch_nwb()]

    def fetch_xarray(self):
        return (xr.concat([df.to_xarray().to_array('marks') for df in self.fetch_dataframe()], dim='electrodes')
                .transpose('time', 'marks', 'electrodes'))


def _convert_algorithm_params(algo_params):
    try:
        algo_params = algo_params.copy()
        algo_params['model'] = algo_params['model'].__name__
    except KeyError:
        pass

    return algo_params


def make_default_decoding_parameters_cpu():

    classifier_parameters = dict(
        environments=[vars(_DEFAULT_ENVIRONMENT)],
        observation_models=None,
        continuous_transition_types=_convert_transitions_to_dict(
            _DEFAULT_CONTINUOUS_TRANSITIONS),
        discrete_transition_type=_to_dict(DiagonalDiscrete(0.98)),
        initial_conditions_type=_to_dict(UniformInitialConditions()),
        infer_track_interior=True,
        clusterless_algorithm='multiunit_likelihood_integer',
        clusterless_algorithm_params=_convert_algorithm_params(
            _DEFAULT_CLUSTERLESS_MODEL_KWARGS)
    )

    predict_parameters = {
        'is_compute_acausal': True,
        'use_gpu':  False,
        'state_names':  ['Continuous', 'Fragmented']
    }
    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


def make_default_decoding_parameters_gpu():
    classifier_parameters = dict(
        environments=[vars(_DEFAULT_ENVIRONMENT)],
        observation_models=None,
        continuous_transition_types=_convert_transitions_to_dict(
            _DEFAULT_CONTINUOUS_TRANSITIONS),
        discrete_transition_type=_to_dict(DiagonalDiscrete(0.98)),
        initial_conditions_type=_to_dict(UniformInitialConditions()),
        infer_track_interior=True,
        clusterless_algorithm='multiunit_likelihood_integer_gpu',
        clusterless_algorithm_params={
            'mark_std': 24.0,
            'position_std': 6.0
        }
    )

    predict_parameters = {
        'is_compute_acausal': True,
        'use_gpu':  True,
        'state_names':  ['Continuous', 'Uniform']
    }

    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


@schema
class ClusterlessClassifierParameters(dj.Manual):
    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default(self):
        (classifier_parameters, fit_parameters,
         predict_parameters) = make_default_decoding_parameters_cpu()
        self.insert1(
            {'classifier_param_name': 'default_decoding_cpu',
             'classifier_params': classifier_parameters,
             'fit_params': fit_parameters,
             'predict_params': predict_parameters},
            skip_duplicates=True)

        (classifier_parameters, fit_parameters,
         predict_parameters) = make_default_decoding_parameters_gpu()
        self.insert1(
            {'classifier_param_name': 'default_decoding_gpu',
             'classifier_params': classifier_parameters,
             'fit_params': fit_parameters,
             'predict_params': predict_parameters},
            skip_duplicates=True)


@schema
class MultiunitFiringRate(dj.Computed):
    definition = """
    -> UnitMarksIndicator
    ---
    -> AnalysisNwbfile
    multiunit_firing_rate_object_id: varchar(40)
    """

    def make(self, key):

        marks = (UnitMarksIndicator & key).fetch_xarray()
        multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)
                            ).astype(float)
        multiunit_firing_rate = pd.DataFrame(
            get_multiunit_population_firing_rate(
                multiunit_spikes, key['sampling_rate']), index=marks.time,
            columns=['firing_rate'])

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key['analysis_file_name'] = nwb_analysis_file.create(
            key['nwb_file_name'])

        key['multiunit_firing_rate_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=multiunit_firing_rate.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        self.insert1(key)

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

        def fetch1_dataframe(self):
            return self.fetch_dataframe()[0]

        def fetch_dataframe(self):
            return [data['multiunit_firing_rate'].set_index('time') for data in self.fetch_nwb()]


@schema
class MultiunitHighSynchronyEventsParameters(dj.Manual):
    definition = """
    param_name : varchar(80) # a name for this set of parameters
    ---
    minimum_duration = 0.015 :  float # minimum duration of event (in seconds)
    zscore_threshold = 2.0 : float    # threshold event must cross to be considered (in std. dev.)
    close_event_threshold = 0.0 :  float # events closer than this will be excluded (in seconds)
    """


@schema
class MultiunitHighSynchronyEvents(dj.Computed):
    definition = """
    -> MultiunitHighSynchronyEventsParameters
    -> UnitMarksIndicator
    -> IntervalPositionInfo
    ---
    -> AnalysisNwbfile
    multiunit_hse_times_object_id: varchar(40)
    """

    def make(self, key):

        marks = (UnitMarksIndicator & key).fetch_xarray()
        multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)
                            ).astype(float)
        position_info = (IntervalPositionInfo() & key).fetch1_dataframe()

        params = (MultiunitHighSynchronyEventsParameters & key).fetch1()

        multiunit_high_synchrony_times = multiunit_HSE_detector(
            marks.time.values,
            multiunit_spikes,
            position_info.head_speed.values,
            sampling_frequency=key['sampling_rate'],
            **params)

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key['analysis_file_name'] = nwb_analysis_file.create(
            key['nwb_file_name'])

        key['multiunit_hse_times_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=multiunit_high_synchrony_times.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        self.insert1(key)
