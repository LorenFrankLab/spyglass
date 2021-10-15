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
from nwb_datajoint.common.dj_helper_fn import fetch_nwb  # dj_replace
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
                          ).fetch1('valid_times')[0]

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

#         # Exclude times without valid neural data
#         raw_valid_times = (IntervalList() &
#                            {'nwb_file_name': key['nwb_file_name'],
#                             'interval_list_name': 'raw data valid times'}
#                            ).fetch1()['valid_times']

#         marks_indicator_df = pd.concat(
#             [marks_indicator_df.loc[start:end]
#              for start, end in raw_valid_times
#              if marks_indicator_df.loc[start:end].shape[0] > 0], axis=0)

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
        start_time, end_time = interval_times
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


def make_default_decoding_parameters_cpu():
    continuous_transition_types = (
        [['random_walk', 'uniform'],
         ['uniform',     'uniform']])

    clusterless_algorithm = 'multiunit_likelihood_integer'
    clusterless_algorithm_params = {
        'mark_std': 20.0,
        'position_std': 8.0,
    }

    classifier_parameters = {
        'replay_speed': 1,
        'place_bin_size': 2.0,
        'movement_var': 6.0,
        'position_range': None,
        'continuous_transition_types': continuous_transition_types,
        'discrete_transition_type': 'strong_diagonal',
        'initial_conditions_type': 'uniform_on_track',
        'discrete_transition_diag': 0.968,
        'infer_track_interior': True,
        'clusterless_algorithm': clusterless_algorithm,
        'clusterless_algorithm_params': clusterless_algorithm_params
    }

    return classifier_parameters


def make_default_decoding_parameters_gpu():
    continuous_transition_types = (
        [['random_walk', 'uniform'],
         ['uniform',     'uniform']])

    clusterless_algorithm = 'multiunit_likelihood_gpu'
    clusterless_algorithm_params = {
        'mark_std': 20.0,
        'position_std': 8.0,
    }

    classifier_parameters = {
        'replay_speed': 1,
        'place_bin_size': 2.0,
        'movement_var': 6.0,
        'position_range': None,
        'continuous_transition_types': continuous_transition_types,
        'discrete_transition_type': 'strong_diagonal',
        'initial_conditions_type': 'uniform_on_track',
        'discrete_transition_diag': 0.968,
        'infer_track_interior': True,
        'clusterless_algorithm': clusterless_algorithm,
        'clusterless_algorithm_params': clusterless_algorithm_params
    }

    return classifier_parameters


def make_default_forward_reverse_decoding_parameters_gpu():
    continuous_transition_types = [['random_walk_direction2', 'random_walk',            'uniform', 'random_walk',            'random_walk',            'uniform'],  # noqa
                                   ['random_walk',            'random_walk_direction1', 'uniform', 'random_walk',            'random_walk',            'uniform'],  # noqa
                                   ['uniform',                'uniform',                'uniform', 'uniform',                'uniform',                'uniform'],  # noqa
                                   ['random_walk',            'random_walk',            'uniform', 'random_walk_direction1', 'random_walk',            'uniform'],  # noqa
                                   ['random_walk',            'random_walk',            'uniform', 'random_walk',            'random_walk_direction2', 'uniform'],  # noqa
                                   ['uniform',                'uniform',                'uniform', 'uniform',                'uniform',                'uniform'],  # noqa
                                   ]

    clusterless_algorithm = 'multiunit_likelihood_gpu'
    clusterless_algorithm_params = {
        'mark_std': 20.0,
        'position_std': 8.0,
    }

    classifier_parameters = {
        'replay_speed': 1,
        'place_bin_size': 2.0,
        'movement_var': 6.0,
        'position_range': None,
        'continuous_transition_types': continuous_transition_types,
        'discrete_transition_type': 'strong_diagonal',
        'initial_conditions_type': 'uniform_on_track',
        'discrete_transition_diag': 0.968,
        'clusterless_algorithm': clusterless_algorithm,
        'infer_track_interior': True,
        'clusterless_algorithm_params': clusterless_algorithm_params
    }

    return classifier_parameters


@schema
class ClassifierParameters(dj.Manual):
    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_param_dict:    BLOB    # dictionary of parameters
    """

    def insert_default_param(self):
        self.insert1(
            {'classifier_param_name': 'default_decoding_cpu',
             'classifier_param_dict': make_default_decoding_parameters_cpu()},
            skip_duplicates=True)

        self.insert1(
            {'classifier_param_name': 'default_decoding_gpu',
             'classifier_param_dict': make_default_decoding_parameters_gpu()},
            skip_duplicates=True)
