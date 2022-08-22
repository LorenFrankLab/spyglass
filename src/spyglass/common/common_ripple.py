import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripple_detection import Kay_ripple_detector
from ripple_detection.core import gaussian_smooth, get_envelope
from spyglass.common import (Electrode, IntervalList, IntervalPositionInfo,
                             IntervalPositionInfoSelection, LFPBand, Session)

from .common_nwbfile import AnalysisNwbfile

schema = dj.schema('common_ripple')


def interpolate_to_new_time(df, new_time, upsampling_interpolation_method='linear'):
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate(
        (old_time, new_time))), name='time')
    return (df
            .reindex(index=new_index)
            .interpolate(method=upsampling_interpolation_method)
            .reindex(index=new_time))


def get_Kay_ripple_consensus_trace(ripple_filtered_lfps, sampling_frequency,
                                   smoothing_sigma=0.004):
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)

    ripple_consensus_trace[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null])
    ripple_consensus_trace = np.sum(ripple_consensus_trace ** 2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], smoothing_sigma, sampling_frequency)
    return np.sqrt(ripple_consensus_trace)


@schema
class RippleLFPSelection(dj.Manual):
    definition = """
     -> Session
     """

    class RippleLFPElectrode(dj.Part):
        definition = """
        -> RippleLFPSelection
        -> Electrode
        """

    def set_lfp_electrodes(self, nwb_file_name, id, electrode_list):
        '''Removes all electrodes for the specified nwb file and then adds back the electrodes in the list

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired session
        id : int
        electrode_list : list
            list of electrodes to be used for LFP

        '''
        # remove the session and then recreate the session and Electrode list
        (RippleLFPSelection() & {
         'nwb_file_name': nwb_file_name, 'id': id}).delete()
        # check to see if the user allowed the deletion
        if len((RippleLFPSelection() & {'nwb_file_name': nwb_file_name}).fetch()) == 0:
            RippleLFPSelection().insert1(
                {'nwb_file_name': nwb_file_name, 'id': id})

            # TODO: do this in a better way
            all_electrodes = (
                Electrode() & {'nwb_file_name': nwb_file_name}).fetch(as_dict=True)
            primary_key = Electrode.primary_key
            for e in all_electrodes:
                # create a dictionary so we can insert new elects
                if e['electrode_id'] in electrode_list:
                    lfpelectdict = {k: v for k,
                                    v in e.items() if k in primary_key}
                    RippleLFPSelection().RippleLFPElectrode.insert1(lfpelectdict, replace=True)


@schema
class RippleParameters(dj.Lookup):
    definition = """
    ripple_param_name : varchar(80) # a name for this set of parameters
    ----
    ripple_param_dict : BLOB    # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default parameter set

        Examples
        --------
        {'peak_sign': 'neg', 'threshold' : 100}
        corresponds to negative going waveforms of at least 100 uV size
        """
        default_dict = {
            'filter_name': 'Ripple 150-250 Hz'
        }
        self.insert1({'ripple_param_name': 'default',
                      'ripple_param_dict': default_dict}, skip_duplicates=True)


@schema
class RippleTimes(dj.Manual):
    definition = """
    -> RippleParameters
    -> RippleLFPSelection
    -> IntervalPositionInfo
    ---
    -> AnalysisNwbfile
    ripple_times_object_id : varchar(40)
     """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        interval_list_name = key['interval_list_name']
        position_info_param_name = key['position_info_param_name']
        id = key['id']

        ripple_params = (
            RippleParameters &
            {'ripple_param_name': key['ripple_param_name']}
        ).fetch1('ripple_param_dict')

        filter_name = ripple_params['filter_name']

        electrode_keys = (RippleLFPSelection() &
                          {'nwb_file_name': nwb_file_name,
                          'id': id}
                          ).RippleLFPElectrode().fetch('KEY')

        # warn/validate that there is only one wire per electrode

        ripple_lfp_nwb = (
            LFPBand & {'nwb_file_name': nwb_file_name,
                       'filter_name': filter_name}
        ).fetch_nwb()[0]

        ripple_lfp = pd.DataFrame(
            ripple_lfp_nwb['filtered_data'].data,
            index=pd.Index(ripple_lfp_nwb['filtered_data'].timestamps, name='time'))
        sampling_frequency = ripple_lfp_nwb['lfp_band_sampling_rate']

        electrode_df = (Electrode() & {'nwb_file_name': nwb_file_name}).fetch(
            format="frame")
        electrode_keys = pd.DataFrame(electrode_keys).set_index(
            Electrode.primary_key).index
        ripple_lfp = ripple_lfp.loc[:, electrode_df.index.isin(electrode_keys)]

        position_valid_times = (
            IntervalList & {
                'nwb_file_name': nwb_file_name,
                'interval_list_name': interval_list_name}
        ).fetch1('valid_times')

        position_info = (
            IntervalPositionInfo() &
            {'nwb_file_name': nwb_file_name,
             'interval_list_name': interval_list_name,
             'position_info_param_name': position_info_param_name}
        ).fetch1_dataframe()

        position_info = pd.concat(
            [position_info.loc[slice(valid_time[0], valid_time[1])]
             for valid_time in position_valid_times], axis=1)
        interval_ripple_lfps = pd.concat(
            [ripple_lfp.loc[slice(valid_time[0], valid_time[1])]
             for valid_time in position_valid_times], axis=1)

        position_info = interpolate_to_new_time(
            position_info, interval_ripple_lfps.index)

        ripple_times = Kay_ripple_detector(
            time=np.asarray(interval_ripple_lfps.index),
            filtered_lfps=np.asarray(interval_ripple_lfps),
            speed=np.asarray(position_info.head_speed),
            sampling_frequency=sampling_frequency
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key['ripple_times_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=ripple_times,
        )

        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        self.insert1(key)

    def plot_ripple(lfps, ripple_times, ripple_label=1,  offset=0.100, relative=True):
        lfp_labels = lfps.columns
        n_lfps = len(lfp_labels)
        ripple_start = ripple_times.loc[ripple_label].start_time
        ripple_end = ripple_times.loc[ripple_label].end_time
        time_slice = slice(ripple_start - offset,
                           ripple_end + offset)
        fig, ax = plt.subplots(1, 1, figsize=(12, n_lfps * 0.20))

        start_offset = ripple_start if relative else 0

        for lfp_ind, lfp_label in enumerate(lfp_labels):
            lfp = lfps.loc[time_slice, lfp_label]
            ax.plot(lfp.index - start_offset, lfp_ind + (lfp - lfp.mean()) / (lfp.max() - lfp.min()),
                    color='black')

        ax.axvspan(ripple_start - start_offset, ripple_end -
                   start_offset, zorder=-1, alpha=0.5, color='lightgrey')
        ax.set_ylim((-1, n_lfps))
        ax.set_xlim((time_slice.start - start_offset,
                    time_slice.stop - start_offset))
        ax.set_ylabel('LFPs')
        ax.set_xlabel('Time [s]')
