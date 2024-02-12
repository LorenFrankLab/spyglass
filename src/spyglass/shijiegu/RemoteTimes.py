import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import entropy
from skimage import measure
from skimage.morphology import area_closing,remove_small_holes
from spyglass.shijiegu.Analysis_SGU import (TrialChoice,EpochPos,
                                   TrialChoiceReplay,TrialChoiceRemoteReplay,
                                   RippleTimes,TrialChoiceDispersedReplay,
                                   Decode,get_linearization_map,
                                   find_ripple_times,find_remote_times,mua_thresholder,
                                   segment_ripples,sort_replays,select_subset,
                                   classify_ripples,classify_ripple_content,plot_decode_spiking,
                                  )


def cal_arm_posterior(posterior_position,maze='4 arm lumped 2023'):
    '''lump posteriors into position bins, 0:home arm, 1-4 arm 1-4, 5: center area'''
    
    linear_map,node_location=get_linearization_map(maze)
    arm_start_end={}
    arm_start_end[0]=linear_map[0]
    arm_start_end[1]=linear_map[3]
    arm_start_end[2]=linear_map[5]
    arm_start_end[3]=linear_map[7]
    arm_start_end[4]=linear_map[9]

    # sum of posterior for an arm for each time bin and entropy within each arm
    posterior_arms=np.zeros((6,posterior_position.shape[0])) #posterior_position.shape[0] is time bin number
    posterior_arms_entropy=np.zeros((5,posterior_position.shape[0])) #posterior_position.shape[0] is time bin number
    for arm_id in [0,1,2,3,4]:
        position_slice=slice(arm_start_end[arm_id][0],arm_start_end[arm_id][1])
        posterior_arms[arm_id,:]=posterior_position.sel(position=position_slice).sum(dim='position')
        posterior_arms_entropy[arm_id,:]=entropy(posterior_position.sel(position=position_slice),axis=1)
        
    posterior_arms[5,:]=1-np.sum(posterior_arms[:5,:],axis=0)
    posterior_arms[posterior_arms < 0] = 0

    return posterior_arms, posterior_arms_entropy

def find_dispersed_time(posterior):
    '''return time windows and binary flag of the same length in time indicating whether each time bin is a dispersed event'''
    times=[]
    posterior_arm_subsets=[]
    posterior_armsIDs=[]
    
    # posterior of arm by summing out the state variable
    posterior_position=posterior.sum(dim='state')
    posterior_state=posterior.sum(dim='position')
    
    # lump by arms
    posterior_arms,posterior_arms_entropy=cal_arm_posterior(posterior_position)
    
    '''
    # find dispersed time windows through entropy
    H = entropy(posterior_arms,axis=0)
    # plt.hist(H)

    # we will use 1 as a cutoff
    
    # find time windows where the following conditions are met, the last MUA requirement is enforced outside this function
    # (1) simultaneous decode (defined as sum of posterior in the arm >=1/6=0.16) in 2 arms for 20 ms 
    # (2) continuous state sum > 0.5 overall in the time window
    # (3) the decode must be clear, have an entropy <3, 
    #      note: uniform entropy at outer arms is [print(entropy(np.ones(29)),entropy(np.ones(43)))]= [3.4 3.8]
    # (4) above mean mua, this requirement is checked outside of this function
    
    f=H>=0.8
    f_closed = area_closing(1-f, 10, connectivity=1) #close small 1's, decode is done at 500 Hz, 2 ms, fullfilling requirement 2
    c_labels,c_num = measure.label(1-f_closed,return_num=True)
    '''
    
    # requirement 1,3: find dispersed time windows through simultaneous arm segments (including home) for 20ms
    f2=(np.sum((posterior_arms[:5,:]>=0.16).astype(int),axis=0)>=2).astype(int)
    f3=(np.sum((posterior_arms_entropy[:5,:]<=3).astype(int),axis=0)>=2).astype(int)
    f=f2+f3>=2 #both conditions are met
    
    f2_closed = area_closing(1-f, 10, connectivity=1) #close small 1's, decode is done at 500 Hz, 2 ms, fullfilling requirement 2
    c_labels,c_num = measure.label(1-f2_closed,return_num=True)
    
    eligible_c=[]
    ineligible_c=[]
    # check continuous state, fullfilling requirement 2
    for c in range(1,c_num+1):
        c_ind=np.argwhere(c_labels==c).ravel()
        c_start=np.array(posterior.time[c_ind[0]])
        c_end=np.array(posterior.time[c_ind[-1]])
        posterior_state_mean=posterior_state.sel(time=slice(c_start,c_end)).mean(dim='time')    
        cont_mean=np.array(posterior_state_mean.sel(state='Continuous'))
        if cont_mean>0.5:
            eligible_c.append(c)
        else:
            ineligible_c.append(c)
            
    eligible_c_copy = eligible_c.copy()
    ineligible_c_copy = ineligible_c.copy()
    
    for c in eligible_c: #eligible for now!
        c_ind=np.argwhere(c_labels==c).ravel()
        c_start=np.array(posterior.time[c_ind[0]])
        c_end=np.array(posterior.time[c_ind[-1]])
        
        # lump posterior by arm for requirement 4
        posterior_position_subset=posterior_position.sel(time=slice(c_start,c_end))
        posterior_arms_subset,posterior_entropy_subset=cal_arm_posterior(posterior_position_subset)
        
        posterior_armsID = np.argwhere(np.logical_and(
            np.mean(posterior_arms_subset,axis=1)[:5]>=0.16,np.mean(posterior_entropy_subset,axis=1)<=3))
            
        posterior_arm_subset_pd = pd.DataFrame({'home':posterior_arms_subset[0,:],
                                                'arm1':posterior_arms_subset[1,:],
                                                'arm2':posterior_arms_subset[2,:],
                                                'arm3':posterior_arms_subset[3,:],
                                                'arm4':posterior_arms_subset[4,:],
                                                'center':posterior_arms_subset[5,:]}, index=np.array(posterior_position_subset.time))
        posterior_arm_subset_pd.index.name='time'
        posterior_arm_subset=xr.Dataset.from_dataframe(posterior_arm_subset_pd)
            
        posterior_arm_subsets.append(posterior_arm_subset)
        times.append([c_start,c_end])
        posterior_armsIDs.append(posterior_armsID)
    # else:
    #     eligible_c_copy.remove(c)
    #     ineligible_c_copy.append(c)
    # eligible_c=eligible_c_copy.copy()
    # ineligible_c=ineligible_c_copy.copy()
        

    for c in ineligible_c:
        c_labels[c_labels==c]=0

    multimodal_flag=c_labels>=1
    multimodal_df = pd.DataFrame({'multimodal_flag':multimodal_flag}, index=np.array(posterior.time))
    multimodal_df.index.name='time'
    multimodal_xr=xr.Dataset.from_dataframe(multimodal_df)
        
    
    return np.array(times),multimodal_xr,posterior_arm_subsets,posterior_armsIDs