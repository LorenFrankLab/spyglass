import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pynwb
import scipy.signal
#import ghostipy as gsp


from spyglass.common import (IntervalPositionInfo, IntervalPositionInfoSelection, IntervalList,
                             ElectrodeGroup, LFP, interval_list_intersect, LFPBand, Electrode)

from spyglass.spikesorting.v0 import (SortGroup,
                                    SortInterval,
                                    SpikeSortingPreprocessingParameters,
                                    SpikeSortingRecording,
                                    SpikeSorterParameters,
                                    SpikeSortingRecordingSelection,
                                    ArtifactDetectionParameters, ArtifactDetectionSelection,
                                    ArtifactRemovedIntervalList, ArtifactDetection,
                                      SpikeSortingSelection, SpikeSorting,)

from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,
                                   get_multiunit_population_firing_rate,
                                   threshold_by_zscore,
                                   merge_overlapping_ranges,
                                   exclude_close_events,
                                   extend_threshold_to_mean)
from itertools import chain

from ripple_detection.detectors import _get_event_stats
from ripple_detection.core import segment_boolean_series

from scipy.stats import zscore
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
#from spyglass.shijiegu.load import load_position circular import problem, called directly by full name in code
from spyglass.shijiegu.helpers import interval_union
from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,RippleTimes,HSETimes,MUA
from spyglass.shijiegu.helpers import interval_union,interpolate_to_new_time

MIN_RIPPLE_LENGTH_IN_S = 0.02

def ripple_detection_master(nwb_file_name, epochID, conservative = True):
    nwb_copy_file_name=get_nwb_copy_filename(nwb_file_name)

    # find epoch/session name and position interval name
    key = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'epoch':epochID}).fetch1()
    epoch_name = key['epoch_name']
    position_interval = key['position_interval']

    position_valid_times = (IntervalList & {'nwb_file_name': nwb_copy_file_name,
                                            'interval_list_name': position_interval}).fetch1('valid_times')

    # Ripple Band LFP
    filtered_lfps, filtered_lfps_t, CA1TetrodeInd, CCTetrodeInd = loadRippleLFP_OneChannelPerElectrode(
        nwb_copy_file_name,epoch_name,position_valid_times)

    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch':int(epoch_name[:2])}).fetch1('choice_reward')
    )

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    t_ind = np.logical_and(filtered_lfps_t >= trial_1_t, filtered_lfps_t <= trial_last_t)
    filtered_lfps_t = filtered_lfps_t[t_ind]
    filtered_lfps = filtered_lfps[t_ind,:]

    # load Position
    position_info = spyglass.shijiegu.load.load_position(nwb_copy_file_name,position_interval)
    position_info_upsample = interpolate_to_new_time(position_info, filtered_lfps_t)
    position_info_upsample = removeDataBeforeTrial1(position_info_upsample,trial_1_t,trial_last_t)
    position_info_upsample = removeArtifactTime(position_info_upsample, filtered_lfps)

    """Detecting Ripples"""
    lfp_band_sampling_rate=(LFPBand & {'nwb_file_name':nwb_copy_file_name,
                                    'target_interval_list_name': epoch_name,
                                    'filter_name': 'Ripple 150-250 Hz'}).fetch1('lfp_band_sampling_rate')

    # Ripple detection. intersection of Consensus, Karlsson method, and MUA,
    # with each event onset offset extended to the union of the former 2.
    ## MUA Detector
    """
    _0,_1,mua_time,mua,_2=load_maze_spike(nwb_copy_file_name,epoch_name)
    mua_smooth = gaussian_smooth(mua, 0.004, 30000) # 4ms smoothing, as in Kay, Karlsson, spiking data are in 30000Hz
    mua_ds = mua_smooth[::10]
    mua_time_ds = mua_time[::10]
    position_info_upsample2 = interpolate_to_new_time(position_info_upsample, mua_time_ds)

    hse_times,firing_rate_raw, mua_mean, mua_std = multiunit_HSE_detector(mua_time_ds,mua_ds,np.array(position_info_upsample2.head_speed),3000,speed_threshold=5.0,
                                    zscore_threshold=0,use_speed_threshold_for_zscore=True)

    # Insert into HSETimes table
    animal = nwb_copy_file_name[:5]
    savePath=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                      nwb_copy_file_name+'_'+epoch_name+'_hse_times.nc')
    hse_times.to_csv(savePath)

    key = {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': epoch_name}
    key['hse_times'] = savePath
    HSETimes().insert1(key,replace = True)

    # Insert into MUA table for future plotting
    key = {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': epoch_name}
    mua_df = pd.DataFrame(data=firing_rate_raw, index=mua_time_ds, columns = ['mua'])
    mua_df.index.name='time'
    mua_df=xr.Dataset.from_dataframe(mua_df)

    savePath=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                             nwb_copy_file_name+'_'+epoch_name+'_mua.nc')
    mua_df.to_netcdf(savePath)

    key['mua_trace'] = savePath
    (key['mean'],key['sd']) = (mua_mean,mua_std)
    MUA().insert1(key,replace = True)
    """

    ripple_times = Gu_ripple_detector(
        filtered_lfps_t, filtered_lfps,
        np.array(position_info_upsample.head_speed), lfp_band_sampling_rate,
        speed_threshold=4.0, zscore_threshold=2.0, conservative = conservative)

    return ripple_times, CA1TetrodeInd, CCTetrodeInd

def extended_ripple_detection_master(nwb_file_name,epochID):
    nwb_copy_file_name=get_nwb_copy_filename(nwb_file_name)

    # find epoch/session name and position interval name
    key = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'epoch':epochID}).fetch1()
    epoch_name = key['epoch_name']
    position_interval = key['position_interval']

    position_valid_times = (IntervalList & {'nwb_file_name': nwb_copy_file_name,
                                            'interval_list_name': position_interval}).fetch1('valid_times')

    # Ripple Band LFP
    filtered_lfps, filtered_lfps_t, CA1TetrodeInd, CCTetrodeInd = loadRippleLFP_OneChannelPerElectrode(
        nwb_copy_file_name,epoch_name,position_valid_times)


    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch':int(epoch_name[:2])}).fetch1('choice_reward')
    )
    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    t_ind = np.logical_and(filtered_lfps_t >= trial_1_t, filtered_lfps_t <= trial_last_t)
    filtered_lfps_t = filtered_lfps_t[t_ind]
    filtered_lfps = filtered_lfps[t_ind,:]

    # load Position
    position_info = spyglass.shijiegu.load.load_position(nwb_copy_file_name,position_interval)
    position_info_upsample = interpolate_to_new_time(position_info, filtered_lfps_t)
    position_info_upsample = removeDataBeforeTrial1(position_info_upsample,trial_1_t,trial_last_t)
    position_info_upsample = removeArtifactTime(position_info_upsample, filtered_lfps)

    """Extend Ripples"""
    lfp_band_sampling_rate=(LFPBand & {'nwb_file_name':nwb_copy_file_name,
                                    'target_interval_list_name': epoch_name,
                                    'filter_name': 'Ripple 150-250 Hz'}).fetch1('lfp_band_sampling_rate')

    ripple_times = pd.DataFrame((RippleTimes() & {'nwb_file_name': nwb_copy_file_name,
                                              'interval_list_name': epoch_name}).fetch1('ripple_times'))

    ripple_times = extended_ripple_detector(ripple_times,
        filtered_lfps_t, filtered_lfps,
        np.array(position_info_upsample.head_speed), lfp_band_sampling_rate, speed_threshold=4.0, zscore_threshold=2.0)

    return ripple_times, CA1TetrodeInd, CCTetrodeInd

def loadRippleLFP(nwb_copy_file_name,
            interval_list_name,
            position_valid_times,
            fieldname = 'artifact removed filtered data'):
    """fieldname is either 'filtered data' for first time runnning through LFP not yet artifact removed.
                    or 'artifact removed filtered data'.
    """
    ripple_nwb_file_path = (LFPBand & {'nwb_file_name': nwb_copy_file_name,
                                    'target_interval_list_name': interval_list_name,
                                    'filter_name': 'Ripple 150-250 Hz'}).fetch1('analysis_file_name')
    if nwb_copy_file_name[:5] == "molly":
        analysisNWBFilePath = '/stelmo/nwb/analysis/'+ripple_nwb_file_path
    else:
        analysisNWBFilePath = '/stelmo/nwb/analysis/'+nwb_copy_file_name[:-5]+'/'+ripple_nwb_file_path
    with pynwb.NWBHDF5IO(analysisNWBFilePath, 'r',load_namespaces=True) as io:
        ripple_nwb = io.read()

        filtered_t=np.array(ripple_nwb.scratch[fieldname].timestamps)
        electrodes=ripple_nwb.scratch[fieldname].electrodes.to_dataframe()
        # TO DO: return directly the electrode IDs. Now it is just a nwb file which I do not know how to get IDs from

        if len(position_valid_times) > 0:
            ripple_lfp = np.concatenate(
                [ripple_nwb.scratch[fieldname].data[
                    np.logical_and(filtered_t>=valid_time[0],filtered_t<=valid_time[1]),:]
                    for valid_time in position_valid_times], axis=0)

            ripple_lfp_t = np.concatenate(
                [filtered_t[np.logical_and(filtered_t>=valid_time[0],filtered_t<=valid_time[1])]
                for valid_time in position_valid_times])
        else:
            ripple_lfp = np.array(ripple_nwb.scratch[fieldname].data)
            ripple_lfp_t = filtered_t

    return ripple_lfp, ripple_lfp_t, electrodes

def loadRippleLFP_OneChannelPerElectrode(nwb_copy_file_name,
            interval_list_name,
            position_valid_times,
            fieldname = 'artifact removed filtered data',return_all = False):
    """Building on loadRippleLFP output so that
    we only load ripple from electrodes in the SortGroup.
    In addition, one channel from each CA1 electrode.
    If `return_all` is set to true:
        return CA1 channels + CC channels
    Otherwise:
        only CA1 channels.
    Also return CA1TetrodeInd, CCTetrodeInd for record keeping purpose.

    returns:
    filtered_lfps: (session length) x (number of channels)
    time: (session length), timestamps
    CA1TetrodeInd: column index of filtered_lfps that correspond to a CA1 tetrode. One for each electrode.
    CCTetrodeInd: column index of filtered_lfps that correspond to a reference tetrodes.
    """
    ripple_data, ripple_timestamps, electrodes = loadRippleLFP(nwb_copy_file_name,interval_list_name,
                                                         position_valid_times,
                                                         fieldname)
    electrodes.index = np.arange(len(electrodes))

    ## find tetrodes with signal
    groups_with_cell=(SpikeSortingRecordingSelection & {
        'nwb_file_name' : nwb_copy_file_name}).fetch('sort_group_id')
    groups_with_cell=np.setdiff1d(groups_with_cell,[100,101])
    print("Using LFP from these eletrodes: ")
    print(groups_with_cell)
    print("\n")

    ## only choose one channel
    CA1TetrodeInd = []
    for e in groups_with_cell:
        CA1TetrodeInd.append(electrodes[electrodes['group_name']==str(e)].index[0])

    ## get reference electrode index
    CCTetrodeInd = []
    LFPTetID = np.array([int(a) for a in np.unique(electrodes.group_name)])
    CA1TetrodeReferenceTetID = np.setdiff1d(LFPTetID,groups_with_cell)

    for e in np.unique(np.array(CA1TetrodeReferenceTetID)):
        CCTetrodeInd.append(electrodes[electrodes['group_name']==str(e)].index[0])

    ### get envelope
    not_null = np.all(pd.notnull(ripple_data), axis=1)

    filtered_lfps, time = (
            ripple_data[not_null], ripple_timestamps[not_null])

    #filtered_lfps_env = get_envelope(filtered_lfps)

    #return filtered_lfps, filtered_lfps_env, time, CA1TetrodeInd, CCTetrodeInd
    if return_all:
        return filtered_lfps, time, CA1TetrodeInd, CCTetrodeInd
    else:
        return filtered_lfps[:,CA1TetrodeInd], time, CA1TetrodeInd, CCTetrodeInd

def removeDataBeforeTrial1(position_df,trial_1_time,trial_last_time):
    problem_ind = position_df.index[np.logical_or(position_df.index <= trial_1_time,
                                                  position_df.index >= trial_last_time)]
    position_df.loc[problem_ind] = np.nan
    position_df = position_df.drop(problem_ind)
    return position_df

def removeArrayDataBeforeTrial1(data,data_t,trial_1_time,trial_last_time):
    keep_ind = np.logical_and(data_t > trial_1_time,
                              data_t < trial_last_time)
    return data[keep_ind],data_t[keep_ind]

def removeArtifactTime(position_df,filtered_lfps):
    SMALLEST = np.iinfo(filtered_lfps.dtype).min + 1
    drop_ind = np.unique(np.argwhere(filtered_lfps <= SMALLEST)[:,0])
    position_df.iloc[drop_ind] = np.nan
    return position_df


def get_electrodes(nwb_file_name):
    """Select good electrodes on that day, which are just the ones in the table SpikeSortingRecordingSelection"""
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)

    # find tetrodes with signal
    groups_with_cell=(SpikeSortingRecordingSelection & {'nwb_file_name' : nwb_copy_file_name}).fetch('sort_group_id')
    groups_with_cell
    groups_with_cell=np.setdiff1d(groups_with_cell,[100,101])

    # construct a smaller electrode table with only selected tetrodes
    electrode_keys=[]
    for tetrode in groups_with_cell:
        #print(tetrode)
        tetrode_group=(Electrode() & {'nwb_file_name': nwb_copy_file_name,'electrode_group_name':tetrode}).fetch('KEY')
        for sub in tetrode_group:
            electrode_keys.append(sub)

    electrode_keys = pd.DataFrame(electrode_keys).set_index(Electrode.primary_key).index

    # big electrode table to select a subset
    electrode_df = (Electrode() & {'nwb_file_name': nwb_copy_file_name}).fetch(format="frame")

def get_zscored_Kay_ripple_consensus_trace(filtered_lfps, time, speed, sampling_frequency,
                                   smoothing_sigma=0.004,speed_threshold = 4):
    """
    ripple_filtered_lfps: of size
    """

    print("The shape of LFP input is " + str(filtered_lfps.shape))

    # 1. Save for Step 4
    speed_full = speed.copy()
    time_full = time.copy()

    # 2. Artifact time are nullified in speed or filtered_lfps
    not_null = np.logical_and(np.all(pd.notnull(filtered_lfps), axis=1),pd.notnull(speed))

    filtered_lfps, speed, time = (
        filtered_lfps[not_null], speed[not_null], time[not_null])

    # 3. Remove null before envelope calculation.
    filtered_lfps = get_envelope(filtered_lfps)
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=1)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    # 4. Return to full length data
    combined_filtered_lfps_full = np.zeros_like(speed_full) + np.nan
    combined_filtered_lfps_full[not_null] = combined_filtered_lfps

    # 5. Remove non-stationary moments
    imobility_ind = speed_full>speed_threshold
    combined_filtered_lfps_full[imobility_ind] = np.nan
    speed_full[imobility_ind] = np.nan
    time_full[imobility_ind] = np.nan

    # 6. zscore
    combined_filtered_lfps_full = zscore(combined_filtered_lfps_full, nan_policy = 'omit')

    return np.sqrt(combined_filtered_lfps_full)

def plot_ripple(lfps, ripple_times, ripple_label=1,  offset=0.100, relative=True):
    """
    LFP is an xarray
    relative: if true, x axis start at 0. if not x axis is absolute UNIX time.
    """
    lfp_labels = list(lfps.keys()) #lfps.columns
    n_lfps = len(lfp_labels)
    ripple_start = ripple_times.loc[ripple_label].start_time
    ripple_end = ripple_times.loc[ripple_label].end_time
    time_slice = slice(ripple_start - offset,
                       ripple_end + offset)
    lfp_subset=lfps.sel(
        time=lfps.time[np.logical_and(lfps.time>=ripple_start - offset,lfps.time<=ripple_end + offset)])

    fig, ax = plt.subplots(1, 1, figsize=(8, n_lfps * 0.40)) #plt.subplots(1, 1, figsize=(12, n_lfps * 0.20))

    start_offset = ripple_start if relative else 0

    for lfp_ind, lfp_label in enumerate(lfp_labels):
        lfp = lfp_subset[lfp_label]
        ax.plot(lfp.time - start_offset, lfp_ind + (lfp - lfp.mean()) / (lfp.max() - lfp.min()),
                color='black',lw=0.5)

    ax.axvspan(ripple_start - start_offset, ripple_end - start_offset, zorder=-1, alpha=0.5, color=[0.8,0.9,1])
    ax.set_ylim((-1, n_lfps))
    ax.set_xlim((time_slice.start - start_offset, time_slice.stop - start_offset))
    ax.set_ylabel('LFPs')
    ax.set_xlabel('Time [s]')

    '''
    Also plot shades of nearby SWRs
    '''
    for other_label in ripple_times.index:
        if other_label == ripple_label:
            continue
        oripple_start = np.max([ripple_times.loc[other_label].start_time,ripple_start - offset])
        oripple_end = np.min([ripple_times.loc[other_label].end_time,ripple_end + offset])
        if np.logical_and(oripple_start >= (ripple_start - offset),oripple_end <= (ripple_end + offset)):
            ax.axvspan(oripple_start, oripple_end, zorder=-1, alpha=0.5, color=[0.8,0.8,0.8])



    return lfp_subset,ax

def threshold_by_zscore_Gu(data, time, minimum_duration=0.015,
                        zscore_threshold=2):
    '''Standardize the data and determine whether it is above a given
    number.

    Parameters
    ----------
    data : array_like, shape (n_time,)
    zscore_threshold : int, optional

    Returns
    -------
    candidate_ripple_times : pandas Dataframe

    '''
    zscored_data = zscore(data, nan_policy='omit')
    is_above_mean = zscored_data >= 0
    is_above_threshold = zscored_data >= zscore_threshold

    return extend_threshold_to_mean(
        is_above_mean, is_above_threshold, time,
        minimum_duration=minimum_duration)



def Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency,
                        speed_threshold=4.0, minimum_duration=0.015,
                        zscore_threshold=2.0, smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Kay et al. 2016 [1]. But

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    filtered_lfps = np.asarray(filtered_lfps)
    print("The shape of LFP input is " + str(filtered_lfps.shape))

    # 1. Save for Step 4
    speed_full = speed.copy()
    time_full = time.copy()

    # 2. Artifact time are nullified in speed or filtered_lfps
    not_null = np.logical_and(np.all(pd.notnull(filtered_lfps), axis=1),pd.notnull(speed))

    filtered_lfps, speed, time = (
        filtered_lfps[not_null], speed[not_null], time[not_null])

    # 3. Remove null before envelope calculation.
    filtered_lfps = get_envelope(filtered_lfps)
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=1)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    # 4. Return to full length data
    combined_filtered_lfps_full = np.zeros_like(speed_full) + np.nan
    combined_filtered_lfps_full[not_null] = combined_filtered_lfps

    # 5. Remove non-stationary moments
    imobility_ind = speed_full>speed_threshold
    combined_filtered_lfps_full[imobility_ind] = np.nan
    speed_full[imobility_ind] = np.nan
    time_full[imobility_ind] = np.nan

    # 6. zscore
    combined_filtered_lfps_full = zscore(combined_filtered_lfps_full, nan_policy = 'omit')
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps_full, time_full, minimum_duration, zscore_threshold)

    # 7. other procedures that are not necessary
    ripple_times = exclude_movement(
        candidate_ripple_times, speed_full, time_full,
        speed_threshold=speed_threshold)
    ripple_times = exclude_close_events(
        ripple_times, close_ripple_threshold)

    #_get_event_stats(ripple_times, time, combined_filtered_lfps, speed)

    #index = pd.Index(np.arange(len(ripple_times)) + 1,
    #                 name='ripple_number')
    #return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
    #                    index=index)
    return _get_event_stats(ripple_times, time, combined_filtered_lfps, speed), combined_filtered_lfps_full

def Karlsson_ripple_detector(
    time,
    filtered_lfps,
    speed,
    sampling_frequency,
    numberOfChannel=2,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=3.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
):
    """Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Karlsson et al. 2009 [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    numberOfChannel: int
        Number of channels to cross threshold simultaneously to say it is a ripple.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.


    """
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    # 1. Save for Step 4
    filtered_lfps_full = np.zeros_like(filtered_lfps) + np.nan
    speed_full = np.zeros_like(speed) + np.nan
    time_full = time.copy()

    not_null = np.logical_and(np.all(pd.notnull(filtered_lfps), axis=1), pd.notnull(speed))
    filtered_lfps, speed, time = (
        filtered_lfps[not_null],
        speed[not_null],
        time[not_null],
    )

    filtered_lfps = get_envelope(filtered_lfps)
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )

    # 4. Return to full length data
    filtered_lfps_full[not_null,:] = filtered_lfps
    speed_full[not_null] = speed

    # 5. Remove non-stationary moments
    imobility_ind = speed_full > speed_threshold
    filtered_lfps_full[imobility_ind,:] = np.nan
    speed_full[imobility_ind] = np.nan
    time_full[imobility_ind] = np.nan

    filtered_lfps_full = zscore(filtered_lfps_full, nan_policy="omit")

    is_above_threshold_all = np.sum(filtered_lfps_full >= zscore_threshold, axis=1) >= numberOfChannel

    candidate_ripple_times = []
    for filtered_lfp in filtered_lfps_full.T:
        is_above_mean = filtered_lfp >= 0
        is_above_threshold = np.logical_and(filtered_lfp >= zscore_threshold, is_above_threshold_all)

        candidate_ripple_times.append(extend_threshold_to_mean(
            is_above_mean, is_above_threshold, time_full, minimum_duration)
            )


    #candidate_ripple_times = [
    #    threshold_by_zscore(filtered_lfp, time, minimum_duration, zscore_threshold)
    #    for filtered_lfp in filtered_lfps.T
    #]

    candidate_ripple_times = list(
        merge_overlapping_ranges(chain.from_iterable(candidate_ripple_times))
    )
    ripple_times = exclude_movement(
        candidate_ripple_times, speed_full, time_full, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)

    '''
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')

    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)
    '''
    return _get_event_stats(ripple_times, time_full, filtered_lfps_full.mean(axis=1), speed_full)

def removePositionDataBeforeTrial1(position_df,trial_1_time,trial_last_time):
    problem_ind = position_df.index[np.logical_or(position_df.index <= trial_1_time,
                                                  position_df.index >= trial_last_time)]
    position_df.loc[problem_ind] = np.nan
    return position_df

def checkOverLap(oped1,oped2):
    if oped1[0] <= oped2[0]:
        return oped1[1] > oped2[0]
    elif oped2[0] < oped1[0]:
        return oped2[1] > oped1[0]

def ExtendInterSection(Consensus,Karlsson):

    candidate_ripple_times = []

    for i in range(len(Consensus)):

        op_ed = [Consensus.iloc[i]['start_time'],Consensus.iloc[i]['end_time']]

        for j in range(len(Karlsson)):

            op_ed_K = [Karlsson.iloc[j]['start_time'],Karlsson.iloc[j]['end_time']]

            if checkOverLap(op_ed,op_ed_K):

                candidate_ripple_times.append((np.min([op_ed[0],op_ed_K[0]]),np.max([op_ed[1],op_ed_K[1]])))

    candidate_ripple_list = list(merge_overlapping_ranges(candidate_ripple_times))

    return candidate_ripple_list

def InterSection(Karlsson, MUA):

    candidate_ripple_times = []

    for i in range(len(MUA)):

        op_ed = [MUA.iloc[i]['start_time'],MUA.iloc[i]['end_time']]

        for j in range(len(Karlsson)):

            op_ed_K = [Karlsson.iloc[j]['start_time'],Karlsson.iloc[j]['end_time']]

            if checkOverLap(op_ed,op_ed_K):

                candidate_ripple_times.append(op_ed_K)

    candidate_ripple_list = list(candidate_ripple_times)

    return candidate_ripple_list

def get_event_stats(event_times, time, zscore_metric, speed):
    """knock off of _get_event_stats with no speed_start or speed_end due to sampling"""
    index = pd.Index(np.arange(len(event_times)) + 1, name="event_number")

    mean_zscore = []
    median_zscore = []
    max_zscore = []
    min_zscore = []
    duration = []
    max_speed = []
    min_speed = []
    median_speed = []
    mean_speed = []

    for start_time, end_time in event_times:
        ind = np.logical_and(time >= start_time, time <= end_time)
        event_zscore = zscore_metric[ind]
        mean_zscore.append(np.mean(event_zscore))
        median_zscore.append(np.median(event_zscore))
        max_zscore.append(np.max(event_zscore))
        min_zscore.append(np.min(event_zscore))
        duration.append(end_time - start_time)
        max_speed.append(np.max(speed[ind]))
        min_speed.append(np.min(speed[ind]))
        median_speed.append(np.median(speed[ind]))
        mean_speed.append(np.mean(speed[ind]))

    try:
        event_start_times = event_times[:, 0]
        event_end_times = event_times[:, 1]
    except TypeError:
        event_start_times = []
        event_end_times = []

    return pd.DataFrame(
        {
            "start_time": event_start_times,
            "end_time": event_end_times,
            "duration": duration,
            "mean_zscore": mean_zscore,
            "median_zscore": median_zscore,
            "max_zscore": max_zscore,
            "min_zscore": min_zscore,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "median_speed": median_speed,
            "mean_speed": mean_speed,
        },
        index=index,
    )


    # filtered_lfps = np.asarray(filtered_lfps)
    # print("The shape of LFP input is " + str(filtered_lfps.shape))

    # not_null = np.all(pd.notnull(filtered_lfps), axis=1)
    # filtered_lfps, speed, time = (
    #     filtered_lfps[not_null], speed[not_null], time[not_null])

    # filtered_lfps = get_envelope(filtered_lfps)
    # combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=1)
    # combined_filtered_lfps = gaussian_smooth(
    #     combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    # combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    # combined_filtered_lfps = zscore(combined_filtered_lfps, nan_policy = 'omit')

    # return _get_event_stats(candidate_ripple_list, time, combined_filtered_lfps, speed)

def extended_ripple_detector(ripple_times, time, filtered_lfps, speed, sampling_frequency,
                             speed_threshold=4.0, minimum_duration=0.015,
                             zscore_threshold=2.0, smoothing_sigma=0.004,
                             close_ripple_threshold=0.0):
    # Use Kay only to get Consensus Trace
    _, consensus_trace = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency,
                        speed_threshold, minimum_duration,
                        zscore_threshold, smoothing_sigma,
                        close_ripple_threshold)
    start_end = np.vstack((np.array(ripple_times.start_time)-0.1,
                                np.array(ripple_times.end_time)+0.1))
    start_end_list = [[start_end[0,ci],start_end[1,ci]] for ci in range(start_end.shape[1])]
    extended_start_end = np.array(list(merge_overlapping_ranges(start_end_list)))

    is_low_speed = pd.Series(speed <= speed_threshold, index=time)
    extended_start_end = interval_list_intersect(extended_start_end, np.array(segment_boolean_series(is_low_speed)))

    #extended_start_end_df = pd.DataFrame(
    #    {
    #        "start_time": extended_start_end[:,0],
    #        "end_time": extended_start_end[:,1],
    #    },
    #    index=1+np.arange(np.shape(extended_start_end)[0]),
    #)
    "Due to intersections, some events are very short, exclude those below 20ms"
    delta_t = np.diff(extended_start_end, axis = 1)
    delta_t_ind = np.squeeze(delta_t >= MIN_RIPPLE_LENGTH_IN_S)
    extended_start_end = extended_start_end[delta_t_ind]

    extended_ripple_times = get_event_stats(extended_start_end,
                               time, consensus_trace, speed)

    return extended_ripple_times

def Gu_ripple_detector(time, filtered_lfps, speed, sampling_frequency,
                        speed_threshold=4.0, minimum_duration=0.015,
                        zscore_threshold=2.0, smoothing_sigma=0.004,
                        close_ripple_threshold=0.0,conservative = True):
    # Consensus
    RippleTimes_Kay, consensus_trace = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency,
                        speed_threshold, minimum_duration,
                        zscore_threshold, smoothing_sigma,
                        close_ripple_threshold)

    if conservative:
        print("both Karlsson and Consensus method")
        RippleTimes_Karlsson = Karlsson_ripple_detector(
            time, filtered_lfps, speed, sampling_frequency, numberOfChannel=2,
            speed_threshold=speed_threshold,
            minimum_duration=minimum_duration,
            zscore_threshold=3,#zscore_threshold,
            smoothing_sigma=smoothing_sigma,
            close_ripple_threshold=close_ripple_threshold)

        # Since Karlsson method usually finds a lot of events, intersection this first
        # candidate_ripple_list = InterSection(RippleTimes_Karlsson,HSETimes)

        # Intersect with Kay method
        # index = pd.Index(np.arange(len(candidate_ripple_list)) + 1,
        #                name='ripple_number')
        #MUA_Karlsson = pd.DataFrame(candidate_ripple_list, columns=['start_time', 'end_time'],
        #                    index=index)

        # For each ripple detectded by both Kay and Karlsson/MUA, extend to union of time
        #candidate_ripple_list2 = ExtendInterSection(RippleTimes_Kay,MUA_Karlsson)
        candidate_ripple_list2 = ExtendInterSection(RippleTimes_Kay,RippleTimes_Karlsson)

        RippleTimes = _get_event_stats(np.array(candidate_ripple_list2),
                                time, consensus_trace, speed)

        print("Final events are: " + str(len(RippleTimes)/len(RippleTimes_Kay)) + "from Consensus, due to Karlsson Intersection.\n")
        print("Final events are: " + str(len(RippleTimes)/len(RippleTimes_Karlsson)) + "from Karlsson, due to Kay Intersection.\n")

    else:
        print("consensus only")
        RippleTimes = RippleTimes_Kay

    print("Found " + str(len(RippleTimes)) + " events.")

    return RippleTimes

# class EnvelopeEstimator:
#     def __init__(self, num_signals, bp_order=2, bp_crit_freqs=[150,250], lfp_sampling_rate=1500, env_num_taps=15, env_band_edges=[50,55], env_desired=[1,0]):
#         # set up iir
#         sos = scipy.signal.iirfilter(
#             bp_order,
#             bp_crit_freqs,
#             output='sos',
#             fs=lfp_sampling_rate,
#             btype='bandpass',
#             ftype='butter',
#         )[:, :, None]

#         ns = sos.shape[0]
#         self._b_ripple = sos[:, :3] # (ns, 3, num_signals)
#         self._a_ripple = sos[:, 3:] # (ns, 3, num_signals)

#         self._x_ripple = np.zeros((ns, 3, num_signals))
#         self._y_ripple = np.zeros((ns, 3, num_signals))

#         # set up envelope filter
#         self._b_env = gsp.firdesign(
#             env_num_taps,
#             env_band_edges,
#             env_desired,
#             fs=lfp_sampling_rate,
#         )[:, None]
#         self._x_env = np.zeros((self._b_env.shape[0], num_signals))

#     def add_new_data(self, data):
#         # coming in parallel, data has width of num_signals

#         # IIR ripple bandpass
#         ns = self._a_ripple.shape[0]
#         for ii in range(ns):

#             self._x_ripple[ii, 1:] = self._x_ripple[ii, :-1]
#             if ii == 0: # new input is incoming data
#                 self._x_ripple[ii, 0] = data
#             else: # new input is IIR output of previous stage
#                 self._x_ripple[ii, 0] = self._y_ripple[ii - 1, 0]

#             self._y_ripple[ii, 1:] = self._y_ripple[ii, :-1]
#             ripple_data = (
#                 np.sum(self._b_ripple[ii] * self._x_ripple[ii], axis=0) -
#                 np.sum(self._a_ripple[ii, 1:] * self._y_ripple[ii, 1:], axis=0)
#             )
#             self._y_ripple[ii, 0] = ripple_data


#         # FIR estimate envelope
#         self._x_env[1:] = self._x_env[:-1]
#         self._x_env[0] = ripple_data**2
#         # parallel dot product without saying it's a dot product
#         env = np.sqrt(np.sum(self._b_env * self._x_env, axis=0))

#         return ripple_data, env


def removeArtifactInFilteredData(nwb_copy_file_name, session_interval_name,
                    filter_name = 'Ripple 150-250 Hz', interp = False):
    ripple_nwb_file_path = (LFPBand & {'nwb_file_name': nwb_copy_file_name,
                                   'target_interval_list_name': session_interval_name,
                                   'filter_name': filter_name}).fetch1('analysis_file_name')
    if nwb_copy_file_name[:5] == "molly":
        analysisNWBFilePath = '/stelmo/nwb/analysis/'+ripple_nwb_file_path
    else:
        analysisNWBFilePath = '/stelmo/nwb/analysis/'+nwb_copy_file_name[:-5]+'/'+ripple_nwb_file_path

    with pynwb.NWBHDF5IO(analysisNWBFilePath, 'r+',load_namespaces=True) as io:
        ripple_nwb = io.read()
        filtered_data=np.array(ripple_nwb.scratch['filtered data'].data)
        timestamps=np.array(ripple_nwb.scratch['filtered data'].timestamps)
        electrodes=ripple_nwb.scratch['filtered data'].electrodes
        SMALLEST = np.iinfo(filtered_data.dtype).min #-32768, or -32767

        '''artifact time to nan'''
        artifact_times_100=(ArtifactRemovedIntervalList()
                                & {'nwb_file_name': nwb_copy_file_name,
                                    'sort_interval_name':session_interval_name,
                                    'sort_group_id':100}).fetch1('artifact_times')
        artifact_times_101=(ArtifactRemovedIntervalList()
                                & {'nwb_file_name': nwb_copy_file_name,
                                    'sort_interval_name':session_interval_name,
                                    'sort_group_id':101}).fetch1('artifact_times')

        artifact_time_list=interval_union(artifact_times_100,artifact_times_101)

        for artifact_time in artifact_time_list:
            filtered_data[np.logical_and(timestamps>=artifact_time[0],
                        timestamps<=artifact_time[1]),:] = SMALLEST #the smallest number in int16

        if interp:
            nan_ind=(filtered_data[:,1] == SMALLEST).ravel()

            for t in range(filtered_data.shape[1]):
                filtered_data[nan_ind,t]= np.interp(timestamps[nan_ind], timestamps[~nan_ind], filtered_data[~nan_ind,t])

        # write processed ripple data back to nwb
        eseries_name = 'artifact removed filtered data'
        es = pynwb.ecephys.ElectricalSeries(name=eseries_name,
                                            data=filtered_data,
                                            electrodes=electrodes,
                                            timestamps=timestamps)

        ripple_nwb.add_scratch(es)

        io.write(ripple_nwb)

def multiunit_HSE_detector(
    time,
    firing_rate,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=2.0,
    smoothing_sigma=0.015,
    close_event_threshold=0.0,
    use_speed_threshold_for_zscore=False,
):
    """Multiunit High Synchrony Event detector. Finds times when the multiunit
    population spiking activity is high relative to the average.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    speed : ndarray, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float
        Maximum running speed of animal to be counted as an event
    minimum_duration : float
        Minimum time the z-score has to stay above threshold to be
        considered an event.
    zscore_threshold : float
        Number of standard deviations the multiunit population firing rate must
        exceed to be considered an event
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.
    close_event_threshold : float
        Exclude events that occur within `close_event_threshold` of a
        previously detected event.
    use_speed_threshold_for_zscore : bool
        Use speed thresholded multiunit for mean and std for z-score calculation

    Returns
    -------
    high_synchrony_event_times : pandas.DataFrame, shape (n_events, 2)

    References
    ----------
    .. [1] Davidson, T.J., Kloosterman, F., and Wilson, M.A. (2009).
    Hippocampal Replay of Extended Experience. Neuron 63, 497–507.

    """
    firing_rate = np.asarray(firing_rate)
    speed = np.asarray(speed)
    time = np.asarray(time)

    firing_rate_raw = firing_rate.copy()
    firing_rate_raw[np.isnan(speed)] = 0

    #firing_rate = gaussian_smooth(
    #    firing_rate, 0.004, sampling_frequency)

    if use_speed_threshold_for_zscore:
        mean = np.nanmean(firing_rate[speed < speed_threshold])
        std = np.nanstd(firing_rate[speed < speed_threshold])
    else:
        mean = np.nanmean(firing_rate)
        std = np.nanstd(firing_rate)

    firing_rate = (firing_rate - mean) / std
    candidate_high_synchrony_events = threshold_by_zscore(
        firing_rate, time, minimum_duration, zscore_threshold
    )
    high_synchrony_events = exclude_movement(
        candidate_high_synchrony_events, speed, time, speed_threshold=speed_threshold
    )
    high_synchrony_events = exclude_close_events(
        high_synchrony_events, close_event_threshold
    )

    return _get_event_stats(high_synchrony_events, time, firing_rate, speed), firing_rate_raw, mean, std

def exclude_movement(candidate_ripple_times, speed, time, speed_threshold=4.0):
    """Removes candidate ripples if the animal is moving.

    Parameters
    ----------
    candidate_ripple_times : array_like, shape (n_ripples, 2)
    speed : ndarray, shape (n_time,)
        Speed of animal during recording session.
    time : ndarray, shape (n_time,)
        Time in recording session.
    speed_threshold : float, optional
        Maximum speed for animal to be considered to be moving.

    Returns
    -------
    ripple_times : ndarray, shape (n_ripples, 2)
        Ripple times where the animal is not moving.

    """
    candidate_ripple_times = np.array(candidate_ripple_times)
    is_below_speed_threshold = []
    op_ind = np.argwhere(np.isin(time,candidate_ripple_times[:,0])).ravel()
    ed_ind = np.argwhere(np.isin(time,candidate_ripple_times[:,1])).ravel()
    for i in range(np.shape(candidate_ripple_times)[0]):
        speed_during_ripple = speed[op_ind[i]:ed_ind[i]]
        is_below_speed_threshold.append(np.nanmax(speed_during_ripple) < speed_threshold)

    return candidate_ripple_times[np.array(is_below_speed_threshold)]
