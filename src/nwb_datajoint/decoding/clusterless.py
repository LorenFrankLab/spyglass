import pprint

import datajoint as dj
import labbox_ephys as le
import numpy as np
import pandas as pd
import pynwb
import xarray as xr
from nwb_datajoint.common.common_interval import IntervalList
from nwb_datajoint.common.common_nwbfile import AnalysisNwbfile
from nwb_datajoint.common.common_spikesorting import (CuratedSpikeSorting,
                                                      SpikeSorting,
                                                      UnitInclusionParameters)
from nwb_datajoint.common.common_position import TrackGraph
from nwb_datajoint.common.dj_helper_fn import fetch_nwb

from replay_trajectory_classification.classifier import (
    _DEFAULT_ENVIRONMENT, _DEFAULT_CONTINUOUS_TRANSITIONS, _DEFAULT_CLUSTERLESS_MODEL_KWARGS, ClusterlessClassifier)
from replay_trajectory_classification.continuous_state_transitions import RandomWalk, RandomWalkDirection1, RandomWalkDirection2, Uniform, Identity
from replay_trajectory_classification.discrete_state_transitions import DiagonalDiscrete, UniformDiscrete, RandomDiscrete, UserDefinedDiscrete
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions, UniformOneEnvironmentInitialConditions
from replay_trajectory_classification.observation_model import ObservationModel
from replay_trajectory_classification.misc import NumbaKDE


schema = dj.schema('decoding_clusterless')


@schema
class MarkParameters(dj.Manual):
    definition = """
    mark_param_name : varchar(80) # a name for this set of parameters
    ---
    mark_type = 'amplitude':  varchar(40) # the type of mark. Currently only 'amplitude' is supported
    mark_param_dict:    BLOB    # dictionary of parameters for the mark extraction function
    """

    def insert_default_param(self):
        """insert the default parameter set {'sign': -1, 'threshold' : 100} corresponding to negative going waveforms of at least 100 uV size
        """
        default_dict = {'sign': -1, 'threshold': 100}
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
class UnitMarkParameters(dj.Manual):
    definition = """
    -> CuratedSpikeSorting
    -> UnitInclusionParameters
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
        units = UnitInclusionParameters().get_included_units(key, key)

        # retrieve the units from the NWB file
        nwb_units = (CuratedSpikeSorting() & key).fetch_nwb()[
            0]['units'].to_dataframe()

        # get the labbox workspace so we can get the waveforms from the recording
        curation_feed_uri = (SpikeSorting & key).fetch('curation_feed_uri')[0]
        workspace = le.load_workspace(curation_feed_uri)
        recording = workspace.get_recording_extractor(
            workspace.recording_ids[0])
        sorting = workspace.get_sorting_extractor(workspace.sorting_ids[0])
        channel_ids = recording.get_channel_ids()
        # assume the channels are all the same for the moment. This would need to be changed for larger probes
        channel_ids_by_unit = [channel_ids] * (max(units['unit_id']) + 1)
        # here we only get 8 points because that should be plenty to find the minimum/maximum
        waveforms = le.get_unit_waveforms(
            recording, sorting, units['unit_id'], channel_ids_by_unit, 8)

        if mark_param['mark_type'] == 'amplitude':
            # get the marks and timestamps
            marks = np.empty((0, 4), dtype='int16')
            timestamps = np.empty((0), dtype='float64')
            for index, unit in enumerate(waveforms):
                marks = np.concatenate(
                    (marks, np.amin(np.asarray(unit, dtype='int16'), axis=2)), axis=0)
                timestamps = np.concatenate(
                    (timestamps, nwb_units.loc[units['unit_id'][index]].spike_times), axis=0)
            # sort the timestamps to order them properly
            sort_order = np.argsort(timestamps)
            timestamps = timestamps[sort_order]
            marks = marks[sort_order, :]

            if 'threshold' in mark_param_dict:
                print('thresholding')
                # filter the marks by the amplitude threshold
                if mark_param_dict['sign'] == -1:
                    include = np.where(
                        np.amax(marks, axis=1) <= (mark_param_dict['sign'] *
                                                   mark_param_dict['threshold'])
                    )[0]
                elif mark_param_dict['sign'] == -1:
                    include = np.where(np.amax(marks, axis=1)
                                       >= mark_param_dict['threshold'])[0]
                else:
                    include = np.where(
                        np.abs(np.amax(marks, axis=1)) >= mark_param_dict['threshold'])[0]
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

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data['marks_indicator'].set_index('time') for data in self.fetch_nwb()]

    def fetch_xarray(self):
        return (xr.concat([df.to_xarray().to_array('marks') for df in self.fetch_dataframe()], dim='electrodes')
                .transpose('time', 'marks', 'electrodes'))

def _to_dict(transition):
    parameters = vars(transition)
    parameters['class_name'] = type(transition).__name__

    return parameters

def _convert_transitions_to_dict(transitions):
    return [[_to_dict(transition) for transition in transition_rows]
            for transition_rows in transitions]

def _convert_algorithm_params(algo_params):
    try:
        algo_params = algo_params.copy()
        algo_params['model'] = algo_params['model'].__name__
    except KeyError:
        pass
    
    return algo_params

def _convert_dict_to_class(d: dict, class_conversion: dict):
    class_name = d.pop('class_name')
    return class_conversion[class_name](**d)

def _convert_env(env_params):
    if env_params['track_graph'] is not None:
        env_params['track_graph'] = (TrackGraph & {'track_graph_name': env_params['track_graph']}).get_networkx_track_graph()
        
    return env_params

def _restore_classes(params):
    continuous_state_transition_types = {
        'RandomWalk': RandomWalk,
        'RandomWalkDirection1': RandomWalkDirection1,
        'RandomWalkDirection2': RandomWalkDirection2,
        'Uniform': Uniform,
        'Identity': Identity,
    }

    discrete_state_transition_types = {
        'DiagonalDiscrete': DiagonalDiscrete,
        'UniformDiscrete': UniformDiscrete,
        'RandomDiscrete': RandomDiscrete,
        'UserDefinedDiscrete': UserDefinedDiscrete, 
    }

    initial_conditions_types = {
        'UniformInitialConditions': UniformInitialConditions,
        'UniformOneEnvironmentInitialConditions':  UniformOneEnvironmentInitialConditions,
    }

    model_types = {
        'NumbaKDE': NumbaKDE,
    }
    
    params['classifier_params']['continuous_transition_types'] = [
        [_convert_dict_to_class(st, continuous_state_transition_types) for st in sts]
        for sts in params['classifier_params']['continuous_transition_types']]
    params['classifier_params']['environments'] = [Environment(**_convert_env(env_params)) for env_params in params['classifier_params']['environments']]
    params['classifier_params']['discrete_transition_type'] = _convert_dict_to_class(params['classifier_params']['discrete_transition_type'], discrete_state_transition_types)
    params['classifier_params']['initial_conditions_type'] = _convert_dict_to_class(params['classifier_params']['initial_conditions_type'], initial_conditions_types)

    if params['classifier_params']['observation_models'] is not None:
        params['classifier_params']['observation_models'] = [ObservationModel(obs) for obs in params['classifier_params']['observation_models']]

    try:
        params['classifier_params']['clusterless_algorithm_params']['model'] = model_types[params['classifier_params']['clusterless_algorithm_params']['model']]
    except KeyError:
        pass
    
    return params


def make_default_decoding_parameters_cpu():

    classifier_parameters = dict(
        environments=[vars(_DEFAULT_ENVIRONMENT)],
        observation_models=None,
        continuous_transition_types=_convert_transitions_to_dict(_DEFAULT_CONTINUOUS_TRANSITIONS),
        discrete_transition_type=_to_dict(DiagonalDiscrete(0.98)),
        initial_conditions_type=_to_dict(UniformInitialConditions()),
        infer_track_interior=True,
        clusterless_algorithm='multiunit_likelihood',
        clusterless_algorithm_params=_convert_algorithm_params(_DEFAULT_CLUSTERLESS_MODEL_KWARGS)
    )

    predict_parameters = {
        'is_compute_acausal': True,
        'use_gpu':  False,
        'state_names':  ['Continuous', 'Uniform']
    }
    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


def convert_classes(classifier_parameters):
    classifier_parameters = dict(
        environments=[vars(env) for env in classifier_parameters['environments']],
        observation_models=None,
        continuous_transition_types=_convert_transitions_to_dict(classifier_parameters['continuous_transition_types']),
        discrete_transition_type=_to_dict(classifier_parameters['discrete_transition_type']),
        initial_conditions_type=_to_dict(classifier_parameters['initial_conditions_type']),
        infer_track_interior=True,
        clusterless_algorithm='multiunit_likelihood_gpu',
        clusterless_algorithm_params={
            'mark_std': 20.0,
            'position_std': 6.0
        }
    )

def make_default_decoding_parameters_gpu():
    classifier_parameters = dict(
        environments=[vars(_DEFAULT_ENVIRONMENT)],
        observation_models=None,
        continuous_transition_types=_convert_transitions_to_dict(_DEFAULT_CONTINUOUS_TRANSITIONS),
        discrete_transition_type=_to_dict(DiagonalDiscrete(0.98)),
        initial_conditions_type=_to_dict(UniformInitialConditions()),
        infer_track_interior=True,
        clusterless_algorithm='multiunit_likelihood_gpu',
        clusterless_algorithm_params={
            'mark_std': 20.0,
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
class ClassifierParameters(dj.Manual):
    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default_param(self):
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
