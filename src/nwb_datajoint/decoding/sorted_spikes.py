import pprint

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb

from nwb_datajoint.common.common_interval import IntervalList
from nwb_datajoint.common.common_nwbfile import AnalysisNwbfile
from nwb_datajoint.common.common_position import TrackGraph
from nwb_datajoint.common.common_spikesorting import (CuratedSpikeSorting,
                                                      SpikeSorting)
from nwb_datajoint.common.dj_helper_fn import fetch_nwb

schema = dj.schema('decoding_clusterless')

@schema
class SortedSpikesIndicatorSelection(dj.Lookup):
    definition = """
    -> CuratedSpikeSorting
    -> IntervalList
    sampling_rate=500 : float
    ---
    """


@schema
class SortedSpikesIndicator(dj.Computed):
    definition = """
    -> CuratedSpikeSorting
    -> SortedSpikesIndicatorSelection
    ---
    -> AnalysisNwbfile
    spike_indicator_object_id: varchar(40)
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

        sampling_rate = (SortedSpikesIndicatorSelection & {
            'nwb_file_name': key['nwb_file_name'],
            'sort_interval_name': key['sort_interval_name'],
            'filter_parameter_set_name': key['filter_parameter_set_name'],
            'sorting_id': key['sorting_id'],
            'interval_list_name': key['interval_list_name']
        }).fetch('sampling_rate')
        
        time = self.get_time_bins_from_interval(interval_times, sampling_rate)
        
        spikes_nwb = (CuratedSpikeSorting & {
            'nwb_file_name': key['nwb_file_name'],
            'sort_interval_name': key['sort_interval_name'],
            'filter_parameter_set_name': key['filter_parameter_set_name'],
            'sorting_id': key['sorting_id'],
        }).fetch_nwb()
        
        spikes = np.concatenate(
            [np.asarray(n_trode['units']['spike_times']) for n_trode in spikes_nwb])

        # Bin spikes into time bins
        spike_indicator = []
        for spike_times in spikes:
            spike_times = spike_times[(spike_times > time[0]) & (spike_times <= time[-1])]
            spike_indicator.append(np.bincount(np.digitize(spike_times, time[1:-1]), minlength=time.shape[0]))
        
        column_names = np.concatenate(
            [[f'{n_trode["sort_group_id"]:04d}_{unit_number:04d}'for unit_number in n_trode['units'].index]
             for n_trode in spikes_nwb])
        spike_indicator = pd.DataFrame(np.stack(spike_indicator, axis=1),
            index=pd.Index(time, name='time'),
            columns=column_names)

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key['analysis_file_name'] = nwb_analysis_file.create(
            key['nwb_file_name'])

        key['spike_indicator_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=spike_indicator.reset_index(),
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
        return pd.concat([data['spike_indicator'].set_index('time') for data in self.fetch_nwb()], axis=1)