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
from spyglass.common.common_position import IntervalLinearizedPosition
from spyglass.common import LFPBand
from spyglass.spikesorting import (SpikeSortingRecording)


from ripple_detection import Kay_ripple_detector
from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,get_multiunit_population_firing_rate)
from ripple_detection.detectors import Kay_ripple_detector

schema = dj.schema('shijiegu_trialanalysis')


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
class TrialChoiceReplay(dj.Manual):
    definition = """
    # trial by trial information of choice
    -> StateScriptFile
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    choice_reward_replay = NULL: blob  # pandas dataframe, choice, reward, ripple time, replays
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
    -> StateScriptFile
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    transitions = NULL: blob   # ndarray, for all replayed arm transitions
    choice_reward_replay_transition = NULL: blob  # pandas dataframe, choice, reward, replayed arm transitions, ripple time, replays
    """
    def make(self,key,replace=False):

        # add in epoch name
        nwb_file_name = key['nwb_file_name']
        epoch=key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name

        self.insert1(key,replace=replace)

@schema
class RippleTimes(dj.Manual):
    definition = """
    # ripple times
    -> IntervalList
    ---
    ripple_times = NULL: blob   # ripple times within that interval
    ripple_times_1sd = NULL: blob   # ripple times within that interval
    ripple_times_2sd = NULL: blob   # ripple times within that interval
    """

@schema
class Decode(dj.Manual):
    definition = """
    # decoded replay content
    -> IntervalList
    ---
    posterior = NULL: blob   # posterior within that interval
    """

'''
@schema
class TrialChoiceDecode(dj.Manual):
    definition = """
    # trial by trial information of choice and decoding
    -> TrialChoice
    ---
    epoch_name: varchar(200)  # session name, get from IntervalList
    choice_reward = NULL: blob  # pandas dataframe, choice
    replay_decode = NULL: blob  # pandas dataframe, decoded replay content
    theta_decode = NULL: blob   # pandas dataframe, decoded theta content
    choice_reward_decode = NULL: blob # combined pandas dataframe
    """

    def make(self,key,replace=False):
        nwb_file_name = key['nwb_file_name']
        epoch=key['epoch']
        epoch_name=(TaskEpoch() & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
        key['epoch_name']=epoch_name
        #key['choice_reward']=None
        #key['replay_decode']=None
        #key['theta_decode']=None
        #key['choice_reward_decode']=None
        self.insert1(key,replace=replace)

'''

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

def mua_thresholder(ripple_H_,mua,mua_time,mua_threshold):
    ripple_H=[]
    for i in range(np.shape(ripple_H_)[0]):
        ripple_t_ind = np.argwhere(np.logical_and(mua_time >= ripple_H_[i,0],
                                                  mua_time < ripple_H_[i,1])).ravel()
        if np.mean(mua[ripple_t_ind])>=mua_threshold:
            ripple_H.append(ripple_H_[i])
    return np.array(ripple_H)

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

def find_start_end(binary_timebins):
    binary_timebins=np.concatenate(([0],binary_timebins,[0]))
    start=np.argwhere(np.diff(binary_timebins)==1).ravel()
    end=np.argwhere(np.diff(binary_timebins)==-1).ravel()-1
    start_end=np.concatenate((start.reshape((-1,1)),end.reshape((-1,1))),axis=1)
    return start_end

def segment_ripples(decode,ripple_t0t1):
    continuous_all=[]
    frag_all=[]

    for i in range(np.shape(ripple_t0t1)[0]):
        t0=ripple_t0t1[i,0]
        t1=ripple_t0t1[i,1]

        mask_time = ((decode.time >= t0) & (decode.time < t1))
        acausal_posterior=decode.isel(time=mask_time).acausal_posterior
        time=np.array(acausal_posterior.time)

        state_posterior=np.array(acausal_posterior.sum('position'))

        snippets_conti=find_start_end(state_posterior[:,0]>=0.5) #continuous
        snippets_frag=find_start_end(state_posterior[:,1]>=0.5)  #fragment

        snippets=[time[s] for s in snippets_conti if np.diff(time[s])[0]>0.02]
        continuous_all.append(snippets)

        snippets=[time[s] for s in snippets_frag if np.diff(time[s])[0]>0.02]
        frag_all.append(snippets)

    return continuous_all,frag_all

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

    # only ripple continous state p>=0.8 is considered
    mask_time = ((decode.time >= t0)
                & (decode.time < t1))
    state_posterior=np.array(decode.isel(time=mask_time).acausal_posterior.sum('position'))
    continuous_flag=np.mean(state_posterior[:,0])>=0.5
    if not continuous_flag:
        return  np.nan

    # mean posterior of each time bin
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

def find_ripple_peaks(ripple_time,ripple_consensus_trace,t0,t1):
    # ripple time is the time for the consensus_trace
    # returns peak time in seconds
    time_ind=np.logical_and(ripple_time>=t0,ripple_time<t1)

    time_middle=ripple_time[time_ind]
    consensus_middle=ripple_consensus_trace[time_ind]

    peaks,_=find_peaks(consensus_middle,distance=40) #40ms apart

    return time_middle[peaks],consensus_middle[peaks]

def plot_decode_spiking(t0t1,    #ripple start and end time
                        linear_position_df, # animal's location
                        results, #decode
                        recordings,neural_ts, #spiking data
                        ripple_nwb,ripple_timestamps,#ripple
                        offset=1, #2 second window
                        title='',
                        savefolder=[],savename=[]):
    lfp_band_sampling_rate=1000
    plottimes=[t0t1[0]-offset,t0t1[1]+offset]

    fig, axes = plt.subplots(6, 1, figsize=(25, 25), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [3, 1,3,1,3,1]},)

    time_slice = slice(plottimes[0], plottimes[1])

    '''Decode and Position data'''
    results.sel(time=time_slice).acausal_posterior.sum('state').plot(x='time', y='position', ax=axes[0], robust=True, cmap='bone_r')
    axes[0].scatter(linear_position_df.loc[time_slice].index,
                    linear_position_df.loc[time_slice].linear_position.values,
                    s=1, color='magenta', zorder=10)

    axes[0].set_title(title+'\n'+'ripple start time (s):'+str(t0t1[0]),size=40)

    '''decode state data'''
    results.sel(time=time_slice).acausal_posterior.sum('position').plot(x='time',
                                                                         hue='state',
                                                                         ax=axes[1])
    # add 50 ms scale bar
    ymin,ymax = axes[0].get_ylim()
    axes[0].plot([plottimes[0]+0.01,plottimes[0]+0.01+0.05],[ymax-20,ymax-20],linewidth=10,color='firebrick',alpha=0.5)
    axes[0].text(plottimes[0]+0.01,ymax-50,'50 ms',fontsize=20)

    '''spike data'''
    neural_times=[]
    neural_datas=[]
    for g in range(len(neural_ts)):
        neural_t=neural_ts[g]
        recording=recordings[g]
        orig_time_ind = np.argwhere(np.logical_and(neural_t >= plottimes[0], neural_t < plottimes[1])).ravel()
        neural_data = recording.get_traces(start_frame=orig_time_ind[0], end_frame=orig_time_ind[-1]+1)
        neural_time=neural_t[orig_time_ind]
        neural_times.append(neural_time)
        neural_datas.append(neural_data)
    neural_data=np.concatenate(neural_datas,axis=1)
    neural_time=neural_times[0] ### CHECK THIS!!!!
    numtrodes=np.shape(neural_data)[1]
    for i in range(numtrodes):
        axes[2].plot(neural_time, neural_data[:,i]+i*300)
    axes[2].set_title('spiking, each row is a tetrode channel',size=30)

    '''mua'''
    # 2 ms * 30 sample/ms
    mua=get_multiunit_population_firing_rate(np.abs(neural_data)>=100,30000,
                                             smoothing_sigma=60/30000)
    ripple_t_ind = np.argwhere(np.logical_and(neural_time >= t0t1[0],
                                              neural_time < t0t1[1])).ravel()
    axes[3].plot(neural_time,mua)
    ymin,ymax = axes[3].get_ylim()
    # add 50 ms scale bar
    axes[3].plot([t0t1[0],t0t1[0]+0.05],[ymax,ymax],linewidth=10,color='firebrick',alpha=0.5)
    axes[3].text(t0t1[0],ymax-150,'50 ms',fontsize=20)
    axes[3].set_title('mua, smooth 2ms, \n'+'max of mua during ripple '+str(np.mean(mua[ripple_t_ind])),size=30)

    '''Ripple band data'''
    ripple_t_ind = np.argwhere(np.logical_and(ripple_timestamps >= plottimes[0],
                                              ripple_timestamps < plottimes[1])).ravel()
    ripple_data = ripple_nwb['artifact removed filtered data'].data[ripple_t_ind,:].astype('int32')
    ripple_time = ripple_timestamps[ripple_t_ind]
    axes[4].plot(ripple_time,ripple_data)
    axes[4].set_title('ripple LFP',size=30)


    '''Consensus'''
    ripple_consensus_trace = get_Kay_ripple_consensus_trace(
        ripple_data, lfp_band_sampling_rate,smoothing_sigma=0.004)

    '''find peaks of Consensus'''
    peak_time,peak_value=find_ripple_peaks(ripple_time,ripple_consensus_trace,t0t1[0],t0t1[1])

    axes[5].plot(ripple_time,ripple_consensus_trace)
    axes[5].plot(peak_time,peak_value,"x")

    axes[5].set_title('consensus ripple',size=30)

    for i in range(1,6):
        axes[i].axvspan(t0t1[0],t0t1[1], zorder=-1, alpha=0.5, color='paleturquoise')


    if len(savefolder)>0:
        plt.savefig(os.path.join(savefolder,savename+'.png'),bbox_inches='tight',dpi=300)
    #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)
    return [(peak_time[i],peak_value[i]) for i in range(len(peak_time))]


def load_everything(nwb_copy_file_name,interval_list_name,pos_interval_list_name):
    # position
    linear_position_df = (IntervalLinearizedPosition() &
                          {'nwb_file_name': nwb_copy_file_name,
                           'interval_list_name': pos_interval_list_name,
                           'position_info_param_name': 'default_decoding'}
                         ).fetch1_dataframe()

    # decode
    decoding_path=(Decode & {'nwb_file_name': nwb_copy_file_name,
          'interval_list_name':interval_list_name}).fetch1('posterior')
    decode= xr.open_dataset(decoding_path)

    # ripple
    ripple_nwb_file_path = (LFPBand & {'nwb_file_name': nwb_copy_file_name,
                             'target_interval_list_name': interval_list_name,
                             'filter_name': 'Ripple 150-250 Hz'}).fetch1('analysis_file_name')
    io=pynwb.NWBHDF5IO('/stelmo/nwb/analysis/'+ripple_nwb_file_path, 'r',load_namespaces=True)
    ripple_nwb_io = io.read()
    ripple_nwb = ripple_nwb_io.scratch

    ripple_timestamps=np.array(ripple_nwb['filtered data'].timestamps)

    # detected ripple times
    ripple_times=pd.DataFrame((RippleTimes &{'nwb_file_name':nwb_copy_file_name,
                                            'interval_list_name': interval_list_name
                                            }).fetch1('ripple_times_1sd'))


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
                                             smoothing_sigma=60/30000)
    assert len(mua)==len(mua_time)
    #mua_std=np.std(mua)

    return linear_position_df,decode,ripple_nwb,ripple_timestamps,ripple_times,recordings,neural_ts,mua,mua_time
