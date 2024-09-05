import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from spyglass.shijiegu.Analysis_SGU import (TrialChoice,
                                   TrialChoiceReplay,ExtendedTrialChoiceReplay,NotExtendedTrialChoiceReplay,
                                   RippleTimes,ExtendedRippleTimes,
                                   ExtendedRippleTimesWithDecode,RippleTimesWithDecode,
                                   MUA,
                                   Decode,get_linearization_map,EpochPos,TetrodeNumber)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_epoch_data, linear_segment2str
from spyglass.common.common_position import TrackGraph
from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.singleUnit import electrode_unit


POSTERIOR_THRESHOLD = 0.4
MINIMUM_DECODE_LENGTH = 10 #2 ms decode bin, 20 ms

def replay_parser_master(nwb_file_name, epoch_num,
                        classifier_param_name = 'default_decoding_gpu_4armMaze',
                        encoding_set = 'all_maze',
                        decode_threshold_method = 'MUA_0SD',
                        extended = False,
                        causal = False,
                        replace = True):
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    # 1. Load state script
    key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num}
    epoch_name=(TrialChoice & key).fetch1('epoch_name')
    print('epoch name',epoch_name)

    # 2. load data
    linear_map,node_location=get_linearization_map()
    (epoch_name,log_df,decode,head_speed,head_orientation,linear_position_df,
        lfp_df,theta_df,ripple_df,neural_df,_) = load_epoch_data(nwb_copy_file_name,epoch_num,
                                                  classifier_param_name,
                                                  encoding_set)
    # load MUA
    mua_path=(MUA & {'nwb_file_name': nwb_copy_file_name,
                 'interval_list_name':epoch_name}).fetch1('mua_trace')
    mua_xr = xr.open_dataset(mua_path)
    if decode_threshold_method == 'MUA_0SD':
        mua_threshold=(MUA & {'nwb_file_name': nwb_copy_file_name,
                 'interval_list_name':epoch_name}).fetch1('mean')
    elif decode_threshold_method == 'MUA_05SD':
        mua_threshold = (MUA & {'nwb_file_name': nwb_copy_file_name,
                 'interval_list_name':epoch_name}).fetch1('mean') + 0.5 * (MUA & {'nwb_file_name': nwb_copy_file_name,
                 'interval_list_name':epoch_name}).fetch1('sd')
    else:
        mua_threshold = 0

    ripple_times = pd.DataFrame((RippleTimes() & {'nwb_file_name': nwb_copy_file_name,
                                              'interval_list_name': epoch_name}).fetch1('ripple_times'))
    if extended:
        ripple_times = pd.DataFrame((ExtendedRippleTimes() & {'nwb_file_name': nwb_copy_file_name,
                                              'interval_list_name': epoch_name}).fetch1('ripple_times'))

    # IF THIS STEP HAS RUN BEFORE
    if 'animal_location' in ripple_times.columns:
        ripple_times = ripple_times.drop(columns = ['animal_location','trial_number',
                                 'cont_intvl','frag_intvl','cont_intvl_replay'])

    # 3. add replay and trial time data to ripple data
    ripple_times = add_location_replay(ripple_times,
                                       linear_position_df,node_location,linear_map,
                                       log_df,
                                       mua_xr,mua_threshold,
                                       decode, causal = causal)

    # reinsert ripple data with added information
    epoch_name = (EpochPos() & {'nwb_file_name':
        nwb_copy_file_name ,'epoch':epoch_num}).fetch1('epoch_name')
    key = {'nwb_file_name': nwb_copy_file_name,
           'interval_list_name': epoch_name,
           'classifier_param_name': classifier_param_name,
           'encoding_set': encoding_set,
           'decode_threshold_method': decode_threshold_method}
    key['ripple_times'] = ripple_times.to_dict()
    if extended:
        ExtendedRippleTimesWithDecode().insert1(key,replace = replace)
    else:
        RippleTimesWithDecode().insert1(key,replace = replace)

    # 2. Pre-expand result: Augment choice_reward table to be choice_reward_replay
    log_df_replay=log_df.copy()
    log_df_replay.insert(5,'ripple_H',[[] for i in range(len(log_df))]) #hold ripple times
    log_df_replay.insert(6,'ripple_O',[[] for i in range(len(log_df))])
    log_df_replay.insert(7,'replay_H',[[] for i in range(len(log_df))]) #hold decoded replay
    log_df_replay.insert(8,'replay_O',[[] for i in range(len(log_df))])
    log_df_replay.insert(9,'ripple_ID_H',[[]  for i in range(len(log_df))])
    log_df_replay.insert(10,'ripple_ID_O',[[]  for i in range(len(log_df))])
    #hold in ripple name

    # 3. Fill in info from ripple_times to the trial table
    for t in np.array(log_df.index)[:-2]:
        ripple_times_subset = ripple_times.loc[ripple_times.trial_number == t]
        for r_ind in range(len(ripple_times_subset)):
            entry = ripple_times_subset.iloc[r_ind]
            if len(entry.animal_location) < 2:
                # in one of the arm segments
                continue
            elif entry.animal_location == 'home':
                log_df_replay.at[t,'ripple_H'].append(entry.cont_intvl)
                log_df_replay.at[t,'replay_H'].append(entry.cont_intvl_replay)
                log_df_replay.at[t,'ripple_ID_H'].append(entry.name)
            elif entry.animal_location[:4] == 'well':
                log_df_replay.at[t,'ripple_O'].append(entry.cont_intvl)
                log_df_replay.at[t,'replay_O'].append(entry.cont_intvl_replay)
                log_df_replay.at[t,'ripple_ID_O'].append(entry.name)


    # 4. insert into Spyglass
    print('Done parsing. Insert into Spyglass')

    key={'nwb_file_name':nwb_copy_file_name,
         'interval_list_name':epoch_name,
         'classifier_param_name':classifier_param_name,
         'encoding_set':encoding_set,
         'decode_threshold_method':decode_threshold_method,
         'choice_reward_replay':log_df_replay.to_dict()}
    if extended:
        ExtendedTrialChoiceReplay().make(key,replace = replace)
    else:
        NotExtendedTrialChoiceReplay().make(key,replace = replace)
    return epoch_name,log_df_replay,ripple_times,decode,head_speed,head_orientation,linear_position_df,lfp_df,theta_df,ripple_df,neural_df,mua_xr


def add_location_replay(ripple_times,linear_position_df,
                        node_location,linear_map,log_df,mua_xr,mua_threshold,decode,causal = True):
    '''
    for each ripple, find where the animal is.
    If it is around well, add it to well (0 - Home, 1 - Well 1, ...), SWR that occurs on track will be in its segment number
    '''
    ripple_times = add_location(ripple_times,linear_position_df,node_location)
    ripple_times = add_trial(ripple_times,log_df)
    ripple_times = add_replay(ripple_times,mua_xr,mua_threshold,decode,linear_map,causal = causal)
    return ripple_times

def add_location(ripple_times,linear_position_df,node_location,colind = 2):
    ripple_times.insert(colind,'animal_location',[[] for i in range(len(ripple_times))]) #hold in ripple times

    for r_ind in ripple_times.index:
        (t0,t1) = (ripple_times.loc[r_ind,'start_time'],ripple_times.loc[r_ind,'end_time'])
        linear_position_subset = linear_position_df.isel(time=np.logical_and(linear_position_df.time >= t0,
                                                                                        linear_position_df.time <= t1))
        linear_position = np.array(np.mean(linear_position_subset.linear_position))
        linear_segment = linear_segment2str(np.array(linear_position_subset.track_segment_id)[0])
        ripple_times.loc[r_ind,'animal_location'] = linear_segment
        for k in node_location.keys():
            if np.abs(node_location[k] - linear_position) < 15:
                ripple_times.loc[r_ind,'animal_location'] = k

    return ripple_times

def add_head_orientation(ripple_times,head_orientation,colind =  5):
    ripple_times.insert(colind,'head_orientation',[[] for i in range(len(ripple_times))]) #hold in ripple times

    for r_ind in ripple_times.index:
        (t0,t1) = (ripple_times.loc[r_ind,'start_time'],ripple_times.loc[r_ind,'end_time'])
        head_orientation_subset = head_orientation.isel(time=np.logical_and(
            head_orientation.time >= t0,
            head_orientation.time <= t1))
        hd = np.array(np.mean(head_orientation_subset.head_orientation))
        ripple_times.loc[r_ind,'head_orientation'] = hd

    return ripple_times

def add_trial(ripple_times,log_df):
    '''
    for each ripple, find which trial it occured.
    '''
    ripple_times.insert(3,'trial_number',[[] for i in range(len(ripple_times))]) #hold in ripple times

    # find legal trials
    rewardNum=np.array(log_df.rewardNum)
    legal_trials=(np.argwhere(rewardNum[:-1]>=1)+1).ravel()#(np.argwhere(np.logical_and(rewardNum[1:]>=1,rewardNum[:-1]>=1))+1).ravel()
    trial_ind=np.array(log_df.index)

    for r_ind in ripple_times.index:
        (t0,t1) = (ripple_times.loc[r_ind,'start_time'],ripple_times.loc[r_ind,'end_time'])
        trial_number = np.argwhere(log_df.timestamp_H >= t0).ravel()[0]
        # this way only works if the trial is legal.
        if np.isin(trial_number,legal_trials):
            ripple_times.loc[r_ind,'trial_number'] = trial_number
        else:
            # if it is a nonlegal trial, use outer well time
            trial_number = np.argwhere(log_df.timestamp_O-1 >= t0).ravel()[0] #assume the rat takes 1 second to go between places
            ripple_times.loc[r_ind,'trial_number'] = trial_number
    return ripple_times

def add_replay(ripple_times,mua_xr,mua_threshold,decode,linear_map,causal = False):
    """also invalidates low mua time"""

    mua_downsample = (interpolate_to_new_time(mua_xr.to_dataframe(), decode.time)).to_xarray()

    ripple_times.insert(4,'cont_intvl',[[] for i in range(len(ripple_times))]) #hold in ripple times
    ripple_times.insert(5,'frag_intvl',[[]  for i in range(len(ripple_times))]) #hold in ripple times
    ripple_times.insert(6,'cont_intvl_replay',[[]  for i in range(len(ripple_times))]) #hold in ripple times
    '''
    for each ripple, segment into fragment and cont time intervals
    '''
    for r_ind in ripple_times.index:
        (t0,t1) = (ripple_times.loc[r_ind,'start_time'],ripple_times.loc[r_ind,'end_time'])
        continuous_t,frag_t =segment_ripples(decode,np.array([[t0,t1]]),causal = causal)
        if len(continuous_t[0]) > 0:
            #print(r_ind)
            ripple_times.at[r_ind,"cont_intvl"] = np.array(continuous_t[0])
        if len(frag_t[0]) > 0:
            ripple_times.at[r_ind,"frag_intvl"] = np.array(frag_t[0])

    '''
    for each ripple, decode cont time intervals
    '''
    for r_ind in ripple_times.index:
        cont_intvl = ripple_times.loc[r_ind,"cont_intvl"]
        if len(cont_intvl) == 0:
            continue

        replays = []
        for i in range(cont_intvl.shape[0]):
            (t0,t1) = cont_intvl[i,:]
            mask_time = ((decode.time >= t0)
                     & (decode.time < t1))
            if causal:
                position_posterior=decode.isel(time=mask_time).causal_posterior.sum('state')
            else:
                position_posterior=decode.isel(time=mask_time).acausal_posterior.sum('state')
            posterior_by_arm = position_posterior2arm_posterior(position_posterior,linear_map)
            mua_portion = mua_downsample.isel(time=mask_time).mua
            valid_ind = np.array(mua_portion >= mua_threshold)

            # only mua > threshold time
            posterior_by_arm = posterior_by_arm[:, valid_ind]
            #position_time = np.array(position_posterior.time)[valid_ind]

            """find argmax over time and invalidate mua time"""
            # ensure only one arm larger than threshold
            monomodal_ind = np.sum(posterior_by_arm >= POSTERIOR_THRESHOLD, axis = 0) <= 1
            posterior_by_arm = posterior_by_arm[:, monomodal_ind]
            #position_time = position_time[:, monomodal_ind]

            # find max arm
            position_posterior_arm_ind = np.argmax(posterior_by_arm, axis = 0)

            # decode in an arm needs to be larger than a value to be considered
            position_posterior_arm_ind,posterior_by_arm = remove_weak_decode(position_posterior_arm_ind,posterior_by_arm)

            # remove short decode that are less than 20 ms
            if len(position_posterior_arm_ind) < MINIMUM_DECODE_LENGTH:
                replays.append([])
                continue
            position_posterior_arm_ind,posterior_by_arm = remove_short_decode(position_posterior_arm_ind,posterior_by_arm)

            replays_ = remove_adjacent(list(position_posterior_arm_ind))

            replays.append(replays_)

        ripple_times.at[r_ind,'cont_intvl_replay'] = replays

    return ripple_times


def remove_short_decode(position_posterior_arm_ind,posterior_by_arm):
    valid_ind = np.arange(len(position_posterior_arm_ind))>=0
    for a in [0,1,2,3,4]: #home, arm1, arm2, arm3, arm4
        condition = position_posterior_arm_ind == a
        lengths = find_boolean_length(condition)
        # if some lengths are short
        if np.sum(lengths >= MINIMUM_DECODE_LENGTH) < len(lengths):
            start_ends = find_start_end(condition)
            problem_inds = start_ends[(start_ends[:,1]-start_ends[:,0] + 1) <= MINIMUM_DECODE_LENGTH,:]
            for r in range(np.shape(problem_inds)[0]):
                valid_ind[problem_inds[r,0]:problem_inds[r,1]+1] = False
    return position_posterior_arm_ind[valid_ind],posterior_by_arm[:,valid_ind]

def remove_weak_decode(position_posterior_arm_ind,posterior_by_arm):
    position_posterior_value = [posterior_by_arm[position_posterior_arm_ind[t_],
                                                 t_] for t_ in range(posterior_by_arm.shape[1])]
    valid_ind = np.array(position_posterior_value) >= POSTERIOR_THRESHOLD
    return position_posterior_arm_ind[valid_ind],posterior_by_arm[:,valid_ind]




def find_boolean_length(condition):
    return np.diff(np.where(np.concatenate(([condition[0]],
                                     condition[:-1] != condition[1:],
                                     [True])))[0])[::2]

def find_start_end(binary_timebins):
    binary_timebins=np.concatenate(([0],binary_timebins,[0]))
    start=np.argwhere(np.diff(binary_timebins)==1).ravel()
    end=np.argwhere(np.diff(binary_timebins)==-1).ravel()-1
    start_end=np.concatenate((start.reshape((-1,1)),end.reshape((-1,1))),axis=1)
    return start_end

def segment_ripples(decode,ripple_t0t1,causal = False):
    continuous_all=[]
    frag_all=[]

    for i in range(np.shape(ripple_t0t1)[0]):
        t0=ripple_t0t1[i,0]
        t1=ripple_t0t1[i,1]

        mask_time = ((decode.time >= t0) & (decode.time < t1))
        if causal:
            posterior=decode.isel(time=mask_time).causal_posterior
        else:
            posterior=decode.isel(time=mask_time).acausal_posterior
        time=np.array(posterior.time)

        state_posterior=np.array(posterior.sum('position'))

        snippets_conti=find_start_end(state_posterior[:,0]>0.5) #continuous
        snippets_frag=find_start_end(state_posterior[:,1]>=0.5)  #fragment

        snippets=[time[s] for s in snippets_conti if np.diff(time[s])[0]>0.02]
        continuous_all.append(snippets)

        snippets=[time[s] for s in snippets_frag if np.diff(time[s])[0]>0.02]
        frag_all.append(snippets)

    return continuous_all,frag_all

def position_posterior2arm_posterior(position_posterior,linear_map):
    """lump sums bins of one arm"""
    posterior_by_arm=[]
    for a in [0,3,5,7,9]: #home, arm1, arm2, arm3, arm4
        mask_pos = ((position_posterior.position >= linear_map[a,0])
                    & (position_posterior.position <= linear_map[a,1]))
        posterior_arm=position_posterior.isel(position=mask_pos).sum('position')

        posterior_by_arm.append(np.array(posterior_arm))
    posterior_by_arm = np.array(posterior_by_arm)

    return posterior_by_arm

def remove_adjacent(nums):
  a = nums[:1]
  for item in nums[1:]:
    if item != a[-1]:
      a.append(item)
  return a

def normalize_signal(s):
    return (s - s.mean()) / (s.max() - s.min())

def binarize_signal_positive(s):
    return s >= 100

def binarize_signal_negative(s):
    return s <= -100

def binarize_signal_vector(s):
    s_ = (s >= 50)/2
    return (np.sum(s_, axis=1) > 0)/2

def plot_ripples_replays(ripple_times,ripples,ripple_table_inds,linear_position_df,decode,spikeColInd,lfp_df,theta_df,
                neural_df,mua_df,head_speed,head_orientation,save_name,
                savefolder,
                wholeTrial = 1,plottimes=[],mua_thresh = 0,
                causal = False):

    ripples = [r for r in ripples if len(r)>0]
    for ripple_ind in range(len(ripples)):
        t0t1 = np.vstack(ripples) #all possible shadings, just np.vstack(ripples), remove empty segments
        ripple_table_ind = ripple_table_inds[ripple_ind]

        if wholeTrial:
            if len(plottimes) == 0:
                plottimes = [t0t1.ravel()[0]-1,t0t1.ravel()[-1]+1]
            else:
                plottimes = [np.min([t0t1.ravel()[0]-1,plottimes[0]]),np.max([t0t1.ravel()[-1]+1,plottimes[1]])]
            title = f"{save_name} all ripples"
            savename = save_name
        else:
            savename = save_name + '_' + str(ripple_table_ind)
            current_t0t1 = ripples[ripple_ind]
            plottimes = [current_t0t1.ravel()[0]-1,current_t0t1.ravel()[-1]+1]

            duration = int(ripple_times.loc[ripple_table_ind].duration * 1000)
            max_zscore = round(ripple_times.loc[ripple_table_ind].max_zscore,2)
            title = f"{savename} \n duration: {duration} ms, max_zscore: {max_zscore}"
            title += '\n Decoded arms:' + str(ripple_times.loc[ripple_table_ind].cont_intvl_replay)

        plot_decode_spiking(plottimes,t0t1,linear_position_df,decode,lfp_df,theta_df,
                neural_df,mua_df,head_speed,head_orientation,
                ripple_consensus_trace=None,
                title=title,savefolder=savefolder,savename=savename,
                simple=True,tetrode2ind = spikeColInd,mua_thresh = mua_thresh, causal = causal)
        if wholeTrial:
            break

def plot_decode_spiking(plottimes,t0t1,linear_position_xr,decode,lfp_xr,theta_xr,
          neural_xr,mua_xr,head_speed,head_orientation,
          ripple_consensus_trace=None,
          title='',savefolder=[],savename=[],
          simple=False,tetrode2ind=None,likelihood=False,mua_thresh=0,causal = False,plot_spiking=True):
    '''
    plotting submodule for plot_decode_spiking()
    plottimes: list of 2 numbers
    t0t1: numpy array of x x 2 numbers, shading region start and end times
    '''

    fig, axes = plt.subplots(6, 1, figsize=(18, 12), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [3,1,1,3,0.5,0.5]},)
    time_slice = slice(plottimes[0], plottimes[1])

    (posterior_position_subset,posterior_state_subset,
     linear_position_subset,lfp_subset,
     theta_subset,
     neural_subset,
     head_orientation_subset,head_speed_subset)=select_subset(plottimes,linear_position_xr,
                                                              decode,lfp_xr,
                                                              theta_xr,
                                                              neural_xr,
                                                              head_speed,head_orientation,likelihood,causal = causal)
    # mua is added later, so it is not streamlined into the select_subset function
    # but same function
    if mua_xr is not None:
        mua_subset=mua_xr.sel(
            time=mua_xr.time[
                np.logical_and(mua_xr.time>=plottimes[0],mua_xr.time<=plottimes[1])])

    '''Decode and Position data'''
    x_axis=np.arange(plottimes[0],plottimes[1],100)
    if likelihood: #Not finished yet
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.acausal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',origin='lower')
    elif causal:
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.causal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')
    else:
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.acausal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')
    axes[0].set_xticks(x_axis, x_axis)
    axes[0].set_aspect('auto')
    # decode.sel(time=time_slice).acausal_posterior.sum('state').plot(x='time', y='position',
    #                                                                  ax=axes[0], robust=True, cmap='bone_r')
    axes[0].scatter(linear_position_subset.time,
                    np.array(linear_position_subset.linear_position),
                    s=1, color='magenta', zorder=10)

    if len(t0t1) > 0:
        t0t1_ = t0t1.ravel()
        axes[0].set_title(title+'\n'+'first shade start time (s):'+str(t0t1_[0])+'\n'+'last shade end time (s):'+str(t0t1_[-1]),size=15)
    else:
        axes[0].set_title(title,size=15)

    '''decode state data'''
    posterior_state_subset.plot(x='time',hue='state',ax=axes[1])

    # add 50 ms scale bar
    ymin,ymax = axes[0].get_ylim()
    axes[0].plot([plottimes[0]+0.01,plottimes[0]+0.01+0.05],[ymax-20,ymax-20],linewidth=10,color='firebrick',alpha=0.5)
    axes[0].text(plottimes[0]+0.01,ymax,'50 ms',fontsize=20)

    '''theta band LFP'''
    theta_d=np.array(theta_subset.to_array()).astype('int32').T
    theta_t=np.array(theta_subset.time)
    axes[2].plot(theta_t,theta_d)
    axes[2].set_title('theta LFP')

    '''spike data and broad band LFP'''
    if plot_spiking:
        spike_d=np.array(neural_subset.to_array()).astype('int32').T
        spike_t=np.array(neural_subset.time)

        lfp_labels = list(lfp_subset.keys()) #lfps.columns
        n_lfps = len(lfp_labels)
        for lfp_ind, lfp_label in enumerate(lfp_labels):
            lfp = lfp_subset[lfp_label]
            axes[3].plot(lfp.time, lfp_ind + normalize_signal(lfp),color='black',lw=0.5)

        if simple: #each tetrode plot all channel's spiking data
            tetrodes = list(tetrode2ind.keys())
            numtrodes = len(tetrodes)
            for i in range(numtrodes):
                tetrode = tetrodes[i]

                # plot negative signal
                spike_data = binarize_signal_negative(spike_d[:,tetrode2ind[tetrode]])/6
                for chi in range(spike_data.shape[1]):
                    spike_time = spike_t[np.argwhere(spike_data[:,chi]).ravel()]
                    axes[3].scatter(spike_time, np.ones(len(spike_time)) + i + chi/4,rasterized=True,marker='|',color = 'salmon',alpha = 0.5)

                # plot positive signal
                spike_data = binarize_signal_positive(spike_d[:,tetrode2ind[tetrode]])/6
                for chi in range(spike_data.shape[1]):
                    spike_time = spike_t[np.argwhere(spike_data[:,chi]).ravel()]
                    axes[3].scatter(spike_time, np.ones(len(spike_time)) + i + chi/4,rasterized=True,marker='|',color = 'navy')

                #axes[3].plot(spike_t, normalize_signal(spike_d[:,tetrode_ind[i]])+i,linewidth=0.3,rasterized=True,color = 'k')
                axes[3].set_title('spiking, each row comes from one channel per tetrode',size=10)
        else:
            numtrodes=np.shape(spike_d)[1]
            for i in range(numtrodes):
                axes[3].plot(spike_t, normalize_signal(spike_d[:,i])+i,linewidth=0.3,rasterized=True,color = 'k')
                axes[3].set_title('spiking, each row is a tetrode channel',size=10)


    '''mua strip at the bottom of decoding'''

    ymin,ymax = axes[0].get_ylim()

    if mua_xr is not None:
        mua_d = np.array(mua_subset.to_array()).astype('int32').T
        mua_t = np.array(mua_subset.time)
        is_above_mean = pd.Series(np.squeeze(mua_d)>mua_thresh, index = mua_t)
        above_mean_segments = np.array(segment_boolean_series(
            is_above_mean, minimum_duration=0.010))
        segmentNum = np.shape(above_mean_segments)[0]
        if segmentNum > 0:
            for i in range(segmentNum):
                axes[0].axvspan(above_mean_segments[i,0],above_mean_segments[i,1],ymin=0,ymax=0.04,color='firebrick',alpha=0.4)
    # axes[4].plot(mua_t,mua_d)
    # axes[4].set_ylim([0, float(mua_xr.quantile(0.95)[0])])
    # ymin,ymax = axes[4].get_ylim()
    # # add 50 ms scale bar
    # axes[4].axhline(y=ymax,xmin=t0t1_[0],xmax=t0t1_[0]+0.05,linewidth=10,color='firebrick',alpha=0.5)
    # axes[4].set_title('MUA')


    # '''ripple band LFP consensus'''
    # ripple_d=np.array(ripple_subset.to_array()).astype('int32').T
    # ripple_t=np.array(ripple_subset.time)

    # # '''Consensus'''
    # if ripple_consensus_trace is None:
    #     ripple_consensus_trace = get_Kay_ripple_consensus_trace(
    #         ripple_d, sampling_frequency=1000,smoothing_sigma=0.004)

    # axes[5].plot(ripple_t,ripple_consensus_trace)
    # axes[5].set_title('ripple LFP')


    '''speed information'''
    axes[4].plot(np.array(head_speed_subset.time),np.array(head_speed_subset.to_array()).ravel())
    axes[4].set_ylim([0, 10])
    axes[4].set_title('animal head speed cm/s')

    phi=head_orientation_subset.diff("head_orientation")
    axes[5].plot(np.array(phi.time),np.array(phi.head_orientation))
    axes[5].set_title('animal head angular speed')

    if len(t0t1) > 0:
        for i in range(1,6):
            for j in range(np.shape(t0t1)[0]):
                axes[i].axvspan(t0t1[j,0],t0t1[j,1], zorder=-1, alpha=0.5, color='paleturquoise')


    if len(savefolder)>0:
        plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200)
        #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)

    return axes

def plot_decode_sortedSpikes(nwb_copy_file_name,session_name,
                             plottimes,t0t1,linear_position_xr,decode,lfp_xr,theta_xr,
                             neural_xr,cell_peak,head_speed,head_orientation,
                             ripple_consensus_trace=None,
                             title='',savefolder=[],savename=[],
                             likelihood=False,mua_thresh=0,causal = False,cell_color={},
                             replay_type_time = None, replay_type = None):
    '''
    plotting submodule for plot_decode_spiking()
    plottimes: list of 2 numbers
    t0t1: numpy array of x x 2 numbers, shading region start and end times
    '''
    if len(list(cell_color.keys())) == 0:
        for k in cell_peak.keys():
            cell_color[k] = 'C0'

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [3,0.5,3]},)
    time_slice = slice(plottimes[0], plottimes[1])

    (posterior_position_subset,posterior_state_subset,
     linear_position_subset,lfp_subset,
     theta_subset,
     neural_subset,
     head_orientation_subset,head_speed_subset)=select_subset(plottimes,linear_position_xr,
                                                              decode,lfp_xr,
                                                              theta_xr,
                                                              neural_xr,
                                                              head_speed,head_orientation,likelihood,causal = causal)

    '''Decode and Position data'''
    x_axis=np.arange(plottimes[0],plottimes[1],100)
    if likelihood: #Not finished yet
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.acausal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',origin='lower')
    elif causal:
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.causal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')
    else:
        axes[0].imshow(posterior_position_subset.T,extent=(
            plottimes[0],plottimes[1],0,np.array(decode.acausal_posterior.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')
    axes[0].set_xticks(x_axis, x_axis)
    axes[0].set_aspect('auto')
    # decode.sel(time=time_slice).acausal_posterior.sum('state').plot(x='time', y='position',
    #                                                                  ax=axes[0], robust=True, cmap='bone_r')
    axes[0].scatter(linear_position_subset.time,
                    np.array(linear_position_subset.linear_position),
                    s=1, color='magenta', zorder=10)

    t0t1_ = t0t1.ravel()
    axes[0].set_title(title+'\n'+'first shade start time (s):'+str(t0t1_[0])+'\n'+'last shade end time (s):'+str(t0t1_[-1]),size=15)

    '''decode state data'''
    posterior_state_subset.plot(x='time',hue='state',ax=axes[1])

    # add 50 ms scale bar
    ymin,ymax = axes[0].get_ylim()
    axes[0].plot([plottimes[0]+0.01,plottimes[0]+0.01+0.05],[ymax-20,ymax-20],linewidth=10,color='firebrick',alpha=0.5)
    axes[0].text(plottimes[0]+0.01,ymax,'50 ms',fontsize=20)

    '''theta band LFP'''
    #theta_d=np.array(theta_subset.to_array()).astype('int32').T
    #theta_t=np.array(theta_subset.time)
    #axes[2].plot(theta_t,theta_d)
    #axes[2].set_title('theta LFP')

    '''spike data and broad band LFP'''

    cells = list(cell_peak.keys())
    e_old = -1
    for cell in cells:
        e = cell[0]
        u = cell[1]
        if e != e_old:
            nwb_units = electrode_unit(nwb_copy_file_name,session_name,e)

        spike_times = nwb_units.loc[u].spike_times
        spike_times = spike_times[np.logical_and(spike_times >= plottimes[0],spike_times <= plottimes[1])]

        axes[2].scatter(spike_times, np.zeros(len(spike_times)) + cell_peak[(e,u)],rasterized=True, marker='|', color = cell_color[(e,u)],alpha = 0.5)
        axes[2].set_title('spiking, each row is a cell',size=10)
        e_old = e.copy()

    """well locations"""
    linear_map,node_location=get_linearization_map()
    for junctions in linear_map[:,1]:
        axes[2].axhline(junctions, linewidth = 0.2, linestyle = '-.',color = [0.2,0.2,0.2])
    for n in node_location.keys():
        axes[2].axhline(node_location[n], linewidth = 1, linestyle = '-')

    graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
    graph.plot_track_graph_as_1D(
    ax=axes[2],
    axis='y',
    other_axis_start = plottimes[0]-0.1)




    # '''ripple band LFP consensus'''
    # ripple_d=np.array(ripple_subset.to_array()).astype('int32').T
    # ripple_t=np.array(ripple_subset.time)

    # # '''Consensus'''
    # if ripple_consensus_trace is None:
    #     ripple_consensus_trace = get_Kay_ripple_consensus_trace(
    #         ripple_d, sampling_frequency=1000,smoothing_sigma=0.004)

    # axes[5].plot(ripple_t,ripple_consensus_trace)
    # axes[5].set_title('ripple LFP')

    '''
    #speed information
    axes[4].plot(np.array(head_speed_subset.time),np.array(head_speed_subset.to_array()).ravel())
    axes[4].set_ylim([0, 10])
    axes[4].set_title('animal head speed cm/s')

    phi=head_orientation_subset.diff("head_orientation")
    axes[5].plot(np.array(phi.time),np.array(phi.head_orientation))
    axes[5].set_title('animal head angular speed')
    '''
    if replay_type_time is not None:
        axes[2].text(replay_type_time,300,replay_type)
        savename = 'type'+replay_type+savename
        print('here 666')


    for i in range(3):
        for j in range(np.shape(t0t1)[0]):
            axes[i].axvspan(t0t1[j,0],t0t1[j,1], zorder=-1, alpha=0.5, color='paleturquoise')


    if len(savefolder)>0:
        plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200)
        #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)

def select_subset_helper(xr_ob,plottimes):
    xr_ob=xr_ob.sel(
        time=xr_ob.time[
            np.logical_and(xr_ob.time>=plottimes[0],xr_ob.time<=plottimes[1])])
    return xr_ob


def select_subset(plottimes,linear_position_xr,results,lfp_xr,theta_xr,
                  neural_xr,head_speed,head_orientation,likelihood=False,causal = False):

    time_slice = slice(plottimes[0], plottimes[1])
    if likelihood:
        posterior_position_subset=results.sel(time=time_slice).likelihood.sum(
            dim='state')
        posterior_position_subset = posterior_position_subset/posterior_position_subset.sum(dim='position')
        posterior_state_subset=results.sel(time=time_slice).likelihood.sum('position')
    elif causal:
        results_subset = select_subset_helper(results,plottimes)
        posterior_position_subset=results_subset.causal_posterior.sum(
            dim='state')
        posterior_state_subset=results_subset.causal_posterior.sum('position')
    else:
        results_subset = select_subset_helper(results,plottimes)
        posterior_position_subset=results_subset.acausal_posterior.sum(
            dim='state')
        posterior_state_subset=results_subset.acausal_posterior.sum('position')


    linear_position_subset= select_subset_helper(linear_position_xr,plottimes)

    lfp_subset = select_subset_helper(lfp_xr,plottimes)

    theta_subset= select_subset_helper(theta_xr, plottimes)
    # ripple_subset=ripple_xr.sel(time=slice(plottimes[0],plottimes[1]))

    neural_subset = None
    if neural_xr is not None:
        neural_subset=neural_xr.sel(
            time=neural_xr.time[
                np.logical_and(neural_xr.time>=plottimes[0],neural_xr.time<=plottimes[1])])


    # if mua_xr is not None:
    #     mua_subset=mua_xr.sel(
    #         time=mua_xr.time[
    #             np.logical_and(mua_xr.time>=plottimes[0],mua_xr.time<=plottimes[1])])
    # else:
    #     mua_subset = None

    head_orientation_subset=head_orientation.sel(
        time=head_orientation.time[
            np.logical_and(head_orientation.time>=plottimes[0],head_orientation.time<=plottimes[1])])

    head_speed_subset=head_speed.sel(
        time=head_speed.time[
            np.logical_and(head_speed.time>=plottimes[0],head_speed.time<=plottimes[1])])

    #head_speed_subset=head_speed.sel(time=slice(plottimes[0],plottimes[1]))

    return posterior_position_subset,posterior_state_subset,linear_position_subset,lfp_subset,theta_subset,neural_subset,head_orientation_subset,head_speed_subset
