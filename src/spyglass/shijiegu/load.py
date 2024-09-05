import datajoint as dj
import os
import ndx_franklab_novela
import pandas as pd
import pynwb
import numpy as np
import spikeinterface as si
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


from spyglass.common.common_ephys import Raw  # noqa: F401
from spyglass.common.common_interval import IntervalList, interval_list_contains
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_behav import StateScriptFile
from spyglass.common.common_position import TrackGraph
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_position import IntervalPositionInfo#,IntervalLinearizedPosition
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.common import LFPBand, LFP
from spyglass.spikesorting.v0 import (SpikeSortingRecording)


from ripple_detection import Kay_ripple_detector
from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,get_multiunit_population_firing_rate)
from ripple_detection.detectors import Kay_ripple_detector

from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.helpers import mergeIntervals,interpolate_to_new_time
from spyglass.shijiegu.Analysis_SGU import TrialChoice,DecodeResultsLinear,TetrodeNumber,EpochPos,RippleTimesWithDecode
import spyglass.shijiegu as shijiegu
from spyglass.shijiegu.ripple_detection import (loadRippleLFP_OneChannelPerElectrode,
                                                removeDataBeforeTrial1,removeArrayDataBeforeTrial1,
                                                get_zscored_Kay_ripple_consensus_trace,
                                                removeArtifactTime)

map = {}
map[0] = 'arm0'
map[1] = 'center1'
map[2] = 'center2'
map[3] = 'center3'
map[4] = 'center4'
map[5] = 'center0'
map[6] = 'arm1'
map[7] = 'arm2'
map[8] = 'arm3'
map[9] = 'arm4'

def linear_segment2str(linear_segment):
    return map[linear_segment]

def load_session_name(nwb_copy_file_name):
    # get sleep and run epoch
    query = {'nwb_file_name': nwb_copy_file_name}

    session_interval = []
    sleep_interval = []
    for epoch in (TaskEpoch & {'nwb_file_name': nwb_copy_file_name}).fetch('epoch'):
        query['epoch'] = epoch
        n = (TaskEpoch & query).fetch1('interval_list_name')
        if (TaskEpoch & query).fetch1('task_name') == 'sleep':
            sleep_interval.append(n)
        elif (TaskEpoch & query).fetch1('task_name') == 'maze':
            session_interval.append(n)

    return session_interval, sleep_interval

def load_run_sessions(nwb_copy_file_name):
    epochs = (EpochPos() & {'nwb_file_name': nwb_copy_file_name}).fetch('epoch')
    epochPos = EpochPos() & {'nwb_file_name': nwb_copy_file_name}
    print(epochPos)

    run_session_id = []
    run_session_name = []
    pos_session_name = []
    for e in epochs:
        epoch_name = (EpochPos() & {'nwb_file_name': nwb_copy_file_name,'epoch':e}).fetch1('epoch_name')
        if epoch_name.split('_')[1][4:8] == 'Sess':
            run_session_id.append(e)
            run_session_name.append(epoch_name)
            pos_name = (EpochPos() & {'nwb_file_name': nwb_copy_file_name,'epoch':e}).fetch1('position_interval')
            pos_session_name.append(pos_name)
    return run_session_id, run_session_name, pos_session_name

def load_session_pos_names(nwb_copy_file_name):
    IntervalList_pd=pd.DataFrame(IntervalList & {'nwb_file_name': nwb_copy_file_name})
    interval_pd=pd.DataFrame((TaskEpoch & {'nwb_file_name':nwb_copy_file_name}).fetch())
    interval_pd.insert(5, "pos_name", '')

    # select position timestamps, only maze sessions or sleep are selected
    for i in IntervalList_pd.index:
        interval=IntervalList_pd['interval_list_name'][i]
        if interval[0:3]=='pos':
            interval_pd.loc[int(interval[4:6]),'pos_name']=interval

    #position_interval=interval_pd.pos_name #use this line if needed
    #but here I make it into a dictionary
    session_interval={}
    for i in interval_pd.index:
        #if interval_pd.loc[i,"task_name"]=="maze":
        session_interval[i] = [interval_pd.loc[i,"interval_list_name"],interval_pd.loc[i,"pos_name"]]

    return session_interval

def load_epoch_data(nwb_copy_file_name,epoch_num,
                    classifier_param_name = 'default_decoding_gpu_4armMaze',
                    decode_encoding_set = 'mobility_2Dheadspeed_above_6'):
    # Load state script
    key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num}
    log=(TrialChoice & key).fetch1('choice_reward')
    epoch_name=(TrialChoice & key).fetch1('epoch_name')
    epoch_pos_name = (EpochPos() & key).fetch1('position_interval')
    print('epoch name',epoch_name)
    print('epoch_pos_name',epoch_pos_name)
    log_df=pd.DataFrame(log)

    # load decode
    decode=load_decode(nwb_copy_file_name,epoch_name,classifier_param_name,decode_encoding_set)

    # load position
    position_info=load_position(nwb_copy_file_name,epoch_pos_name)

    head_speed=xr.Dataset.from_dataframe(pd.DataFrame(position_info['head_speed']))
    head_orientation=xr.Dataset.from_dataframe(pd.DataFrame(position_info['head_orientation']))

    # load linear position
    linear_position_df = xr.Dataset.from_dataframe((IntervalLinearizedPosition() &
                          {'nwb_file_name': nwb_copy_file_name,
                           'interval_list_name': epoch_pos_name,
                           'position_info_param_name': 'default_decoding'}
                         ).fetch1_dataframe())

    # load LFP, one channel from CA1 tetrodes and CC tetrodes
    lfp,lfp_t = load_LFP(nwb_copy_file_name,epoch_name)
    CA1TetrodeInd = (TetrodeNumber() & {'nwb_file_name': nwb_copy_file_name,
                                        'epoch':epoch_num}).fetch1('ca1_tetrode_ind')
    CCTetrodeInd = (TetrodeNumber() & {'nwb_file_name': nwb_copy_file_name,
                                        'epoch':epoch_num}).fetch1('cc_tetrode_ind')
    TetrodeInd = CA1TetrodeInd + CCTetrodeInd
    lfp_df = pd.DataFrame(data=np.array(lfp)[:,TetrodeInd], index=lfp_t)
    lfp_df.index.name='time'
    lfp_df=xr.Dataset.from_dataframe(lfp_df)

    # load_theta, one channel from each CC channel
    theta_data,theta_timestamps=load_theta(nwb_copy_file_name,epoch_name)
    theta_df = pd.DataFrame(data=theta_data[:,CCTetrodeInd], index=theta_timestamps)
    theta_df.index.name='time'
    theta_df=xr.Dataset.from_dataframe(theta_df)

    # load_ripple
    ripple_data,ripple_timestamps,electrodes=load_ripple(nwb_copy_file_name,epoch_name)
    ripple_df = pd.DataFrame(data=ripple_data[:,CA1TetrodeInd], index=ripple_timestamps)
    ripple_df.index.name='time'
    ripple_df=xr.Dataset.from_dataframe(ripple_df)

    # load spikes
    neural_data,neural_ts,mua_time,mua,channel_IDs=load_spike(nwb_copy_file_name,epoch_name)
    mua_df = pd.DataFrame(data=mua, index=mua_time)
    mua_df.index.name='time'
    mua_df=xr.Dataset.from_dataframe(mua_df)

    neural_df = pd.DataFrame(data=neural_data, index=neural_ts[0],columns=channel_IDs)
    neural_df.index.name='time'
    neural_df=xr.Dataset.from_dataframe(neural_df)

    return epoch_name,log_df,decode,head_speed,head_orientation,linear_position_df,lfp_df,theta_df,ripple_df,neural_df,mua_df

def load_decode(nwb_copy_file_name,interval_list_name,
                classifier_param_name = 'default_decoding_gpu_4armMaze',
                encoding_set = 'mobility_2Dheadspeed_above_6'):
    #decoding_path=(Decode & {'nwb_file_name': nwb_copy_file_name,
    #      'interval_list_name':interval_list_name}).fetch1('posterior')
    decoding_path=(DecodeResultsLinear & {'nwb_file_name': nwb_copy_file_name,
          'interval_list_name': interval_list_name,
          'classifier_param_name': classifier_param_name,
          'encoding_set': encoding_set}).fetch1('posterior')
    decode= xr.open_dataset(decoding_path)
    return decode

def load_position(nwb_copy_file_name,interval_list_name):
    position_info_param_name = 'default'
    position_valid_times = (IntervalList & {'nwb_file_name': nwb_copy_file_name,
                                        'interval_list_name':interval_list_name}).fetch1('valid_times')

    position_info = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                'interval_list_name': interval_list_name,
                'position_info_param_name': position_info_param_name}).fetch1_dataframe()
    position_info = pd.concat(
        [position_info.loc[slice(valid_time[0], valid_time[1])]
         for valid_time in position_valid_times], axis=0)
    return position_info

def load_LFP(nwb_copy_file_name,epoch_name):
    # broad band LFP
    # TO DO. USE ARTIFACT REMOVED
    lfp_nwb=(LFP & {'nwb_file_name': nwb_copy_file_name,
            'target_interval_list_name':epoch_name}).fetch_nwb()

    lfp_data=lfp_nwb[0]['lfp'].data
    lfp_timestamps=np.array(lfp_nwb[0]['lfp'].timestamps)
    return lfp_data,lfp_timestamps

def load_theta(nwb_copy_file_name,epoch_name):
    # TO DO. USE ARTIFACT REMOVED
    theta_nwb=(LFPBand & {'nwb_file_name': nwb_copy_file_name,
            'target_interval_list_name':epoch_name,
            'filter_name':'Theta 5-11 Hz pass, 4.5-12 Hz stop'}).fetch_nwb()

    theta_data=theta_nwb[0]['filtered_data'].data
    theta_timestamps=np.array(theta_nwb[0]['filtered_data'].timestamps)
    return theta_data,theta_timestamps

def load_theta_maze(nwb_copy_file_name,epoch_name):
    """load theta, maze time only"""
    theta_data,theta_timestamps = load_theta(nwb_copy_file_name,epoch_name)

    CCTetrodeInd = (TetrodeNumber() & {'nwb_file_name': nwb_copy_file_name,
                                        'epoch_name':epoch_name}).fetch1('cc_tetrode_ind')

    theta_df = pd.DataFrame(data=theta_data[:,CCTetrodeInd], index=theta_timestamps)
    theta_df.index.name='time'
    theta_df=xr.Dataset.from_dataframe(theta_df)

    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':epoch_name}).fetch1('choice_reward')
    )

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    time_slice = slice(trial_1_t, trial_last_t)
    theta_df_subset=theta_df.sel(time=time_slice)

    return theta_df_subset

def load_ripple(nwb_copy_file_name,epoch_name):

    (ripple_data, ripple_timestamps,
        electrodes) = shijiegu.ripple_detection.loadRippleLFP(nwb_copy_file_name,
        epoch_name,"",'artifact removed filtered data')

    return ripple_data,ripple_timestamps,electrodes

def load_spike(nwb_copy_file_name,interval_list_name,tetrode_list=[100,101]):
    #obtain neural data
    recordings=[]
    neural_ts=[]
    neural_datas=[]
    for tetrode in tetrode_list:
        recording_path = (SpikeSortingRecording & {'nwb_file_name' : nwb_copy_file_name,
                                                   'sort_interval_name' : interval_list_name,
                                                   'sort_group_id' : tetrode}).fetch1('recording_path')
        recording = si.load_extractor(recording_path)
        if recording.get_num_segments() > 1 and isinstance(recording, si.AppendSegmentRecording):
            recording = si.concatenate_recordings(recording.recording_list)
        elif recording.get_num_segments() > 1 and isinstance(recording, si.BinaryRecordingExtractor):
            recording = si.concatenate_recordings([recording])

        neural_data = recording.get_traces()
        neural_t = SpikeSortingRecording._get_recording_timestamps(recording)

        recordings.append(recording)
        neural_ts.append(neural_t)
        neural_datas.append(neural_data)

    neural_data=np.concatenate(neural_datas,axis=1)
    mua_time=neural_ts[0] ### CHECK THIS!!!!
    mua=get_multiunit_population_firing_rate(neural_data<=-60,30000,
                                             smoothing_sigma=100/30000)
    assert len(mua)==len(mua_time)

    channel_IDs = np.concatenate([l.channel_ids for l in recordings])

    return neural_data,neural_ts,mua_time,mua,channel_IDs

def load_maze_spike(nwb_copy_file_name,interval_list_name,tetrode_list=[100,101]):
    # obtain neural data
    neural_data,neural_ts,mua_time,mua,channel_IDs = load_spike(nwb_copy_file_name,
                                                                interval_list_name,tetrode_list=tetrode_list)

    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':interval_list_name}).fetch1('choice_reward')
    )

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    t_ind = np.logical_and(neural_ts[0] >= trial_1_t,neural_ts[0] <= trial_last_t)


    neural_ts[0] = neural_ts[0][t_ind]
    neural_ts[1] = neural_ts[1][t_ind]

    mua_time = mua_time[t_ind]

    neural_data = neural_data[t_ind,:]
    mua = mua[t_ind]

    return neural_data,neural_ts,mua_time,mua,channel_IDs

def load_zscored_ripple_consensus(nwb_copy_file_name, session_name, pos_name):
    position_valid_times = (IntervalList & {'nwb_file_name': nwb_copy_file_name,
                                        'interval_list_name': pos_name}).fetch1('valid_times')

    filtered_lfps, filtered_lfps_t, CA1TetrodeInd, _ = loadRippleLFP_OneChannelPerElectrode(nwb_copy_file_name,
            session_name,position_valid_times,fieldname = 'artifact removed filtered data')

    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}).fetch1('choice_reward')
    )

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    position_info = load_position(nwb_copy_file_name,pos_name)
    position_info_upsample = interpolate_to_new_time(position_info, filtered_lfps_t)
    position_info_upsample = removeDataBeforeTrial1(position_info_upsample,trial_1_t,trial_last_t)
    filtered_lfps, filtered_lfps_t = removeArrayDataBeforeTrial1(filtered_lfps, filtered_lfps_t,trial_1_t,trial_last_t)
    position_info_upsample = removeArtifactTime(position_info_upsample, filtered_lfps)


    lfp_band_sampling_rate=(LFPBand & {'nwb_file_name':nwb_copy_file_name,
                                    'target_interval_list_name': session_name,
                                    'filter_name': 'Ripple 150-250 Hz'}).fetch1('lfp_band_sampling_rate')

    consensus = get_zscored_Kay_ripple_consensus_trace(filtered_lfps, filtered_lfps_t,
                                                   np.array(position_info_upsample.head_speed), lfp_band_sampling_rate,
                                                   smoothing_sigma=0.004)

    return consensus, filtered_lfps_t
