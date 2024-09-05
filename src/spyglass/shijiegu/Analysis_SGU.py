import datajoint as dj
import os
import ndx_franklab_novela
import pandas as pd
import pynwb
import numpy as np
import spikeinterface as si
import xarray as xr
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
from spyglass.spikesorting.v0 import (SpikeSortingRecording)
from spyglass.decoding.v0.clusterless import ClusterlessClassifierParameters

from spyglass.spikesorting.v0.spikesorting_curation import Waveforms


from ripple_detection import Kay_ripple_detector
from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,get_multiunit_population_firing_rate)
from ripple_detection.detectors import Kay_ripple_detector

from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.helpers import mergeIntervals

schema = dj.schema('shijiegu_trialanalysis')

@schema
class EpochPos(dj.Manual):
    definition = """
    # trial by trial information of choice
    -> TaskEpoch
    ---
    epoch_name: varchar(200)  # TaskEpoch or IntervalList
    position_interval: varchar(200)  # IntervalPositionInfo
    """
    def make(self,key,replace=False):

        # add in epoch name
        nwb_copy_file_name = key['nwb_file_name']
        # select position timestamps, only maze sessions are selected
        IntervalList_pd=pd.DataFrame(IntervalList & {'nwb_file_name': nwb_copy_file_name})
        position_interval=[]
        session_interval=[]
        for i in IntervalList_pd.index:
            interval=IntervalList_pd['interval_list_name'][i]
            if 'Session' in interval[-9:-1] or 'Sleep' in interval[-7:-1]:
                session_interval.append(interval)
            if interval[:3]=='pos':
                #pos_interval='pos '+str(i)+' valid times'
                position_interval.append(interval)

        # sort them respectively in order
        session_ids = [int(s.split("_")[0]) for s in session_interval]
        session_interval = list(np.array(session_interval)[np.argsort(session_ids)])

        position_ids = [int(p.split(" ")[1]) for p in position_interval]
        position_interval = list(np.array(position_interval)[np.argsort(position_ids)])

        session_pos_map = {}
        for i in range(len(session_interval)):
            session_pos_map[session_interval[i]] = position_interval[i]

        #for epoch in (TaskEpoch & {'nwb_file_name':nwb_copy_file_name}).fetch('epoch'):

        epoch = key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_copy_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name
        key['position_interval'] = session_pos_map[epoch_name]


        self.insert1(key,replace=replace)

@schema
class TrialChoice(dj.Manual):
    definition = """
    # trial by trial information of choice
    -> StateScriptFile
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    choice_reward = NULL: blob  # pandas dataframe, choice
    """
    def make(self,key,replace=False):

        # add in epoch name
        nwb_file_name = key['nwb_file_name']
        epoch=key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name

        # add in tags
        log_df_tagged=get_trial_tags(key['choice_reward'])
        key['choice_reward']=log_df_tagged.to_dict()

        self.insert1(key,replace=replace)

@schema
class TetrodeNumber(dj.Manual):
    definition = """
    # tetrode CHANNEL number for plotting data from various brain regions
    -> TaskEpoch
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    ca1_tetrode_ind = NULL: blob  # 0-based indices of CA1 tetrodes channels. These can be used for plotting
    cc_tetrode_ind = NULL: blob  # 0-based indices of Corpus Collosum tetrodes channels. These can be used for plotting
    """

@schema
class HSETimes(dj.Manual):
    definition = """
    # MUA times
    -> IntervalList
    ---
    hse_times: varchar(200)  # file name for hse_times table
    """


@schema
class MUA(dj.Manual):
    definition = """
    # MUA trace
    -> IntervalList
    ---
    mua_trace: varchar(200)  # file name for MUA trace
    mean = 0: double  # mean
    sd = 0: double #sd
    """

@schema
class Footprint(dj.Manual):
    definition = """
    -> Waveforms
    ---
    foot_print = NULL: blob  # dict
    matfile_path: varchar(1000) #same info as Footprint but .mat location for MATLAB
    """

@schema
class Tracking(dj.Manual):
    """For 2 sessioms, produce a tracking.
    """
    definition = """
    nwb_file_name: varchar(1000)
    sort_group_id: int
    sort_interval_name1: varchar(100)
    sort_interval_name2: varchar(100)
    ---
    matfile_path: varchar(1000) #tracking result from MATLAB
    """


def get_trial_tags(log):
    '''
    This function adds current, past, etc categories to behavior parsing.
    '''
    # load TrialChoice
    #key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num}
    #log_df=pd.DataFrame((TrialChoice & key).fetch1('choice_reward'))
    #epoch_name=(TrialChoice & key).fetch1('epoch_name')

    log_df=pd.DataFrame(log)

    future_H=np.array(log_df['OuterWellIndex'])
    current=np.array(log_df['OuterWellIndex'])
    future_O=np.concatenate((current[1:],[np.nan]))
    past=np.concatenate(([np.nan],current[:-1]))
    past_reward=np.zeros(len(past)+1)+np.nan

    for t in range(2,len(log_df)): # in trial (1 indexing)
        if log_df.loc[t-1,:]['rewardNum']==2:
            past_reward[t]=log_df.loc[t-1,:]['OuterWellIndex']
        else:
            past_reward[t]=past_reward[t-1]
    # go back to zero indexing
    past_reward=past_reward[1:]

    log_df_tagged=log_df.copy()
    log_df_tagged['current']=current
    log_df_tagged['future_H']=future_H
    log_df_tagged['future_O']=future_O
    log_df_tagged['past']=past
    log_df_tagged['past_reward']=past_reward

    return log_df_tagged

@schema
class ExtendedRippleTimes(dj.Manual):
    definition = """
    # ripple times
    -> IntervalList
    ---
    ripple_times = NULL: blob   # ripple times within that interval
    ripple_times_1sd = NULL: blob   # MUA threshold is 1SD
    """

@schema
class ThetaIntervals(dj.Manual):
    definition = """
    # theta intervals during mobility (>4cm/s)
    -> IntervalList
    ---
    theta_times = NULL: blob   # ripple times within that interval
    """

@schema
class DecodeIngredients(dj.Manual):
    definition = """
    # ingredients to do decoded replay content
    -> IntervalList
    ---
    marks = NULL: blob   # valid marks within that interval
    position_1d = NULL: blob   # valid position within that interval (1D), inluding all rat-on-track time
    position_2d = NULL: blob   # valid position within that interval (2D), inluding all rat-on-track time
    """

@schema
class DecodeResultsLinear(dj.Manual):
    definition = """
    # decoded 1D replay content results
    -> IntervalList
    -> ClusterlessClassifierParameters
    encoding_set : varchar(32) # a name for this set of encoding
    ---
    posterior = NULL: blob   # posterior within that interval (1D)
    """

@schema
class ExtendedRippleTimesWithDecode(dj.Manual):
    definition = """
    # ripple times with decode
    -> DecodeResultsLinear
    decode_threshold_method: varchar(64) # a name for this thresholding method
    ---
    ripple_times = NULL: blob   # ripple times within that interval
    """

@schema
class RippleTimesWithDecode(dj.Manual):
    definition = """
    # ripple times with decode
    -> DecodeResultsLinear
    decode_threshold_method: varchar(64) # a name for this thresholding method
    ---
    ripple_times = NULL: blob   # ripple times within that interval
    """

@schema
class TrialChoiceReplay(dj.Manual):
    definition = """
    # trial by trial information of choice NOTE: ripple based
    -> DecodeResultsLinear
    ---
    choice_reward_replay = NULL: blob  # pandas dataframe, choice, reward, ripple time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        #nwb_file_name = key['nwb_file_name']
        #epoch=key['epoch']
        #epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        #key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class ExtendedTrialChoiceReplay(dj.Manual):
    definition = """
    # trial by trial information of choice and ripple based replay
    -> DecodeResultsLinear
    decode_threshold_method: varchar(64) # a name for this thresholding method
    ---
    choice_reward_replay = NULL: blob  # pandas dataframe, choice, reward, ripple time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        # nwb_file_name = key['nwb_file_name']
        # epoch=key['epoch']
        # epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        # key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class NotExtendedTrialChoiceReplay(dj.Manual):
    definition = """
    # trial by trial information of choice and ripple based replay
    # 2024 August onwards, to distinguish from TrialChoiceReplay
    -> DecodeResultsLinear
    decode_threshold_method: varchar(64) # a name for this thresholding method
    ---
    choice_reward_replay = NULL: blob  # pandas dataframe, choice, reward, ripple time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        # nwb_file_name = key['nwb_file_name']
        # epoch=key['epoch']
        # epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        # key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class TrialChoiceRemoteReplay(dj.Manual):
    definition = """
    # trial by trial information of choice NOTE: remote content based
    -> StateScriptFile
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    choice_reward_replay = NULL: blob  # pandas dataframe, choice, reward, remote time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        nwb_file_name = key['nwb_file_name']
        epoch=key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class TrialChoiceDispersedReplay(dj.Manual):
    definition = """
    # trial by trial information of choice NOTE: remote content based
    -> StateScriptFile
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    choice_reward_dispersed_replay = NULL: blob  # pandas dataframe, choice, reward, remote time, replays
    coactivation_matrix = NULL: blob # per matrix per epoch, row i col j: one occurance of arm j representation while the animal is at i
    """

    def make(self,key,replace=False):

        # add in epoch name
        nwb_file_name = key['nwb_file_name']
        epoch=key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class TrialChoiceReplayTransition(dj.Manual):
    definition = """
    # trial by trial information of choice
    -> DecodeResultsLinear
    ---
    transitions = NULL: blob   # ndarray, for all replayed arm transitions
    choice_reward_replay_transition = NULL: blob  # pandas dataframe, choice, reward, replayed arm transitions, ripple time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        # nwb_file_name = key['nwb_file_name']
        # epoch=key['epoch']
        # epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        # key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class RippleTimes(dj.Manual):
    definition = """
    # ripple times
    -> IntervalList
    ---
    ripple_times = NULL: blob   # ripple times within that interval
    ripple_times_04sd = NULL: blob   # ripple times within that interval
    ripple_times_1sd = NULL: blob   # ripple times within that interval
    ripple_times_2sd = NULL: blob   # ripple times within that interval
    """



@schema
class Decode(dj.Manual):
    definition = """
    # decoded replay content
    -> IntervalList
    ---
    posterior = NULL: blob   # posterior within that interval (1D)
    posterior2D = NULL: blob   # posterior within that interval (2D)
    """


def get_linearization_map(track_graph_name='4 arm lumped'):
    graph = TrackGraph() & {'track_graph_name': track_graph_name}
    track_graph = graph.get_networkx_track_graph()

    edge_order=graph.fetch1('linear_edge_order')
    edge_spacing=graph.fetch1('linear_edge_spacing')

    start_positions=[]
    end_positions=[]
    start_node_linear_position=0
    for ind, edge in enumerate(edge_order):
        end_node_linear_position = (start_node_linear_position +
                                    track_graph.edges[edge]["distance"])
        start_positions.append(start_node_linear_position)
        try:
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing[ind])
        except IndexError:
            pass
        end_positions.append(end_node_linear_position)

    linear_map=np.concatenate([np.array(start_positions).reshape((-1,1)),
                               np.array(end_positions).reshape((-1,1))],axis=1)
    #np.diff(linear_map,axis=1)
    label=['home','center',
           'center','arm1',
           'center','arm2',
           'center','arm3',
           'center','arm4',
          ]

    welllocations={}
    welllocations['home']=linear_map[0][0]
    welllocations['well1']=linear_map[3][1]
    welllocations['well2']=linear_map[5][1]
    welllocations['well3']=linear_map[7][1]
    welllocations['well4']=linear_map[9][1]


    return linear_map,welllocations

def segment_to_linear_range(linear_map,segment_id):
    """takes an int segment_id and return the range of linearization of the 4-arm track """
    #linear_map, welllocations = get_linearization_map()
    arm = int(segment_id-5)
    ind = int(2*(arm) + 1)
    return linear_map[ind], arm

def mua_thresholder(ripple_H_,mua,mua_time,mua_threshold):
    ripple_H=[]
    for i in range(np.shape(ripple_H_)[0]):
        ripple_t_ind = np.argwhere(np.logical_and(mua_time >= ripple_H_[i,0],
                                                  mua_time < ripple_H_[i,1])).ravel()
        if np.mean(mua[ripple_t_ind])>=mua_threshold:
            ripple_H.append(ripple_H_[i])
    return np.array(ripple_H)

def find_remote_times(decode,arm_start,arm_end,mask_time,delta_t):
    '''
    find remote (not on the same arm) replay times from mask
    delta_t: if adjacent intervals are within delta_t of each other, they are merged

    RETURN: numpy array, n x 2
    '''
    delta_t=delta_t/2
    acausal_posterior=decode.isel(time=mask_time).acausal_posterior
    time=np.array(acausal_posterior.time)
    positions=np.array(acausal_posterior.position)

    # IF USING MEAN
    # mean_location=np.matmul(np.array(acausal_posterior.sum(dim='state')),np.arange(acausal_posterior.position.size))
    # mask_remote = np.logical_or(mean_location > arm_end,mean_location < arm_start)

    max_location=positions[acausal_posterior.sum(dim='state').argmax(dim='position')]
    mask_remote = np.logical_or(max_location > arm_end,max_location < arm_start)

    remote_segments=np.array(segment_boolean_series(pd.Series(mask_remote,index=time), minimum_duration=0.005))
    if len(remote_segments)>0:
        remote_segments[:,0]=remote_segments[:,0]-delta_t
        remote_segments[:,1]=remote_segments[:,1]+delta_t
        smooth_remote_segments=np.array(mergeIntervals(remote_segments))
        smooth_remote_segments[:,0]=smooth_remote_segments[:,0]+delta_t
        smooth_remote_segments[:,1]=smooth_remote_segments[:,1]-delta_t
        return smooth_remote_segments
    else:
        return []

def find_ripple_times(ripple_times,t0,t1):
    '''
    find ripple times r that are t0 <= r < t1

    RETURN: numpy array, n x 2
    '''

    # find intervals
    ind=np.argwhere(np.logical_and(np.array(ripple_times.start_time)-t0>=0,
                               t1>np.array(ripple_times.end_time))).ravel()+1
    if len(ind)==0:
        return np.array([])

    # ripple has to be longer than 40ms

    ind_final=[]
    for t in ind:
        if (ripple_times.loc[t,'end_time']-ripple_times.loc[t,'start_time'])>0.04:
            ind_final.append(t)

    #ind_final=ind

    if len(ind_final)==0:
        return np.array([])

    # return intervals
    selected=ripple_times.loc[ind_final,:].reset_index(drop=True)
    selected.index=selected.index+1

    return np.array(selected)





def classify_ripples(decode,ripple_times):
    # ripple_times is a list of lists of (2,)'s
    decoded_arms=[]

    # if no ripples, return empty
    if len(ripple_times)==0:
        return decoded_arms

    # for each ripple, classify ripple
    for i in range(len(ripple_times)):
        decoded_arms_i=[]
        for s in range(len(ripple_times[i])):
            decoded_arm=classify_ripple_content(decode,ripple_times[i][s][0],
                                                ripple_times[i][s][1])
            decoded_arms_i.append(decoded_arm)
        decoded_arms.append(decoded_arms_i)

    return decoded_arms



def classify_ripple_content(decode,t0,t1):
    '''
    RETURN:
    if continuous, most likely arm (HOME is 0)
    if not, return nan
    '''
    # Load linear map
    linear_map,_=get_linearization_map()

    # load ripple content

    # only ripple continous state p>=0.5 is considered
    mask_time = ((decode.time >= t0)
                & (decode.time < t1))
    state_posterior=np.array(decode.isel(time=mask_time).acausal_posterior.sum('position'))
    continuous_flag=np.mean(state_posterior[:,0])>=0.5
    if not continuous_flag:
        return  np.nan

    # mean posterior across time bins
    position_posterior=decode.isel(time=mask_time).acausal_posterior.sum('state')
    mean_location=position_posterior.mean(dim='time')

    posterior_by_arm=[]
    for a in [0,3,5,7,9]: #home, arm1, arm2, arm3, arm4
        mask_pos = ((mean_location.position > linear_map[a,0])
                & (mean_location.position <= linear_map[a,1]))
        posterior_arm=mean_location.isel(position=mask_pos).sum()

        posterior_by_arm.append(float(posterior_arm))

    posterior_arm=0
    for a in [1,2,4,6,8]: # all middle area
        mask_pos = ((mean_location.position > linear_map[a,0])
                & (mean_location.position <= linear_map[a,1]))
        posterior_arm=posterior_arm+mean_location.isel(position=mask_pos).sum()
    posterior_by_arm.append(float(posterior_arm))
    '''
    posterior_center=0
    for a in [1,2,4,6,8]:

        mask_pos = ((mean_location.position >= linear_map[a,0])
                & (mean_location.position <= linear_map[a,1]))
        posterior_arm=mean_location.isel(position=mask_pos).sum()

        posterior_center=posterior_center+posterior_arm
    posterior_by_arm.append(float(posterior_center))
    '''
    # return argmax
    arm_id=np.argmax(posterior_by_arm)

    return arm_id

def sort_replays(cont_ripple_H,frag_ripple_H,cont_replay_H):
    replay_H=[]
    ripple_H=[]

    for ri in range(len(cont_ripple_H)):
        if len(cont_ripple_H[ri])>0 and len(frag_ripple_H[ri])>0:
            ripple_H_tmp=np.concatenate([np.array(cont_ripple_H[ri]),
                                         np.array(frag_ripple_H[ri])])
            replay_H_tmp=np.array([replay for
                          replay in cont_replay_H[ri]]+[np.nan for replay
                                                        in frag_ripple_H[ri]])
            ripple_ind=np.argsort(ripple_H_tmp[:,0])

            ripple_H_tmp=ripple_H_tmp[ripple_ind]
            replay_H_tmp=replay_H_tmp[ripple_ind]

        else:
            if len(cont_ripple_H[ri])>0:
                ripple_H_tmp=np.array(cont_ripple_H[ri])
                replay_H_tmp=cont_replay_H[ri]
            elif len(frag_ripple_H[ri])>0:
                ripple_H_tmp=np.array(frag_ripple_H[ri])
                replay_H_tmp=[np.nan for replay in frag_ripple_H[ri]]
            else:
                ripple_H_tmp=[]
                replay_H_tmp=[]
        replay_H.append(replay_H_tmp)
        ripple_H.append(ripple_H_tmp)
    return ripple_H,replay_H

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

def find_max(ripple_time,ripple_consensus_trace,t0,t1):
    time_ind=np.logical_and(ripple_time>=t0,ripple_time<t1)

    time_middle=ripple_time[time_ind]
    consensus_middle=ripple_consensus_trace[time_ind]

    return time_middle[np.argmax(consensus_middle)],np.max(consensus_middle)

def find_ripple_peaks(ripple_time,ripple_consensus_trace,t0,t1):
    # ripple time is the time for the consensus_trace
    # returns peak time in seconds
    time_ind=np.logical_and(ripple_time>=t0,ripple_time<t1)

    time_middle=ripple_time[time_ind]
    consensus_middle=ripple_consensus_trace[time_ind]

    peaks,_=find_peaks(consensus_middle,distance=10) #10ms apart

    return time_middle[peaks],consensus_middle[peaks]

# def session2snippet(t0t1,    #ripple start and end time
#                     linear_position_df, # animal's location
#                     results, #decode
#                     recordings,neural_ts, #spiking data
#                     ripple_data,ripple_timestamps,#ripple
#                     theta_data,theta_timestamps,#theta
#                     headspeed,head_orientation,#data frame, speed
#                     offset=1, #2 second window
#                     title='',
#                     savefolder=[],savename=[],simple=False):






def load_everything(nwb_copy_file_name,interval_list_name,pos_interval_list_name):
    # position
    linear_position_df = (IntervalLinearizedPosition() &
                          {'nwb_file_name': nwb_copy_file_name,
                           'interval_list_name': pos_interval_list_name,
                           'position_info_param_name': 'default_decoding'}
                         ).fetch1_dataframe()

    # load decode
    decode=load_decode(nwb_copy_file_name,interval_list_name)

    # MUA
    neural_data,neural_ts,mua_time,mua,recordings=load_spike(nwb_copy_file_name,interval_list_name)

    # ripple
    ripple_data,ripple_timestamps=load_ripple(nwb_copy_file_name,interval_list_name)

#     ripple_nwb_file_path = (LFPBand & {'nwb_file_name': nwb_copy_file_name,
#                              'target_interval_list_name': interval_list_name,
#                              'filter_name': 'Ripple 150-250 Hz'}).fetch1('analysis_file_name')
#     io=pynwb.NWBHDF5IO('/stelmo/nwb/analysis/'+nwb_copy_file_name[:-5]+'/'+ripple_nwb_file_path, 'r',load_namespaces=True)
#     ripple_nwb_io = io.read()
#     ripple_nwb = ripple_nwb_io.scratch

#     ripple_timestamps=np.array(ripple_nwb['filtered data'].timestamps)

#     # detected ripple times
#     ripple_times=pd.DataFrame((RippleTimes &{'nwb_file_name':nwb_copy_file_name,
#                                             'interval_list_name': interval_list_name
#                                             }).fetch1('ripple_times_04sd')) #1sd

    ripple_times=[]




    return linear_position_df,decode,ripple_data,ripple_timestamps,ripple_times,recordings,neural_ts,mua,mua_time
