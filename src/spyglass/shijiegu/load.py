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
from spyglass.common.common_position import IntervalPositionInfo,IntervalLinearizedPosition
from spyglass.common import LFPBand
from spyglass.spikesorting import (SpikeSortingRecording)


from ripple_detection import Kay_ripple_detector
from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,get_multiunit_population_firing_rate)
from ripple_detection.detectors import Kay_ripple_detector

from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.helpers import mergeIntervals
from spyglass.shijiegu.Analysis_SGU import TrialChoice,Decode

def load_epoch_data(nwb_copy_file_name,epoch_num):
    # Load state script
    key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num}
    log=(TrialChoice & key).fetch1('choice_reward')
    epoch_name=(TrialChoice & key).fetch1('epoch_name')
    epoch_pos_name='pos '+str(int(epoch_name[:2])-1)+' valid times'
    print('epoch name',epoch_name)
    print('epoch_pos_name',epoch_pos_name)
    log_df=pd.DataFrame(log)
    
    # load decode
    decode=load_decode(nwb_copy_file_name,epoch_name)

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
    
    # load LFP
    
    # load_theta
    theta_data,theta_timestamps=load_theta(nwb_copy_file_name,epoch_name)
    theta_df = pd.DataFrame(data=theta_data, index=theta_timestamps)
    theta_df.index.name='time'
    theta_df=xr.Dataset.from_dataframe(theta_df)

    # load_ripple
    ripple_data,ripple_timestamps=load_ripple(nwb_copy_file_name,epoch_name)
    ripple_df = pd.DataFrame(data=ripple_data, index=ripple_timestamps)
    ripple_df.index.name='time'
    ripple_df=xr.Dataset.from_dataframe(ripple_df)
    
    # load spikes
    neural_data,neural_ts,mua_time,mua,recordings=load_spike(nwb_copy_file_name,epoch_name)
    mua_df = pd.DataFrame(data=mua, index=mua_time)
    mua_df.index.name='time'
    mua_df=xr.Dataset.from_dataframe(mua_df)
    
    neural_df = pd.DataFrame(data=neural_data, index=neural_ts[0])
    neural_df.index.name='time'
    neural_df=xr.Dataset.from_dataframe(neural_df)
    
    return epoch_name,log_df,decode,head_speed,head_orientation,linear_position_df,theta_df,ripple_df,neural_df,mua_df,recordings

def load_decode(nwb_copy_file_name,interval_list_name):
    decoding_path=(Decode & {'nwb_file_name': nwb_copy_file_name,
          'interval_list_name':interval_list_name}).fetch1('posterior')
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

    lfp_data=lfp_nwb[0]['filtered_data'].data
    lfp_timestamps=np.array(lfp_nwb[0]['filtered_data'].timestamps)
    return lfp_data,lfp_timestamps
    
def load_theta(nwb_copy_file_name,epoch_name):
    # TO DO. USE ARTIFACT REMOVED
    theta_nwb=(LFPBand & {'nwb_file_name': nwb_copy_file_name,
            'target_interval_list_name':epoch_name,
            'filter_name':'Theta 5-11 Hz pass, 4.5-12 Hz stop'}).fetch_nwb()

    theta_data=theta_nwb[0]['filtered_data'].data
    theta_timestamps=np.array(theta_nwb[0]['filtered_data'].timestamps)
    return theta_data,theta_timestamps

def load_ripple(nwb_copy_file_name,epoch_name):
    # TO DO. USE ARTIFACT REMOVED
    ripple_nwb=(LFPBand & {'nwb_file_name': nwb_copy_file_name,
            'target_interval_list_name':epoch_name,
            'filter_name':'Ripple 150-250 Hz'}).fetch_nwb()

    ripple_data=ripple_nwb[0]['filtered_data'].data
    ripple_timestamps=np.array(ripple_nwb[0]['filtered_data'].timestamps)

    return ripple_data,ripple_timestamps

def load_spike(nwb_copy_file_name,interval_list_name):
    #obtain neural data
    recordings=[]
    neural_ts=[]
    neural_datas=[]
    for tetrode in [100,101]:
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
    mua=get_multiunit_population_firing_rate(np.abs(neural_data)>=100,30000,
                                             smoothing_sigma=100/30000)
    assert len(mua)==len(mua_time)
    
    return neural_data,neural_ts,mua_time,mua,recordings