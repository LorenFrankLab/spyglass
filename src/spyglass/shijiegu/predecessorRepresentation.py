import numpy as np
import os
import pickle
import pandas as pd

from spyglass.shijiegu.ripple_add_replay import remove_adjacent

from spyglass.shijiegu.ripple_add_replay import remove_adjacent
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import TrialChoice,TrialChoiceReplayTransition

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

def transitions_to_matrix(transitions):
    M = np.zeros((4,4))
    for rt in transitions:
        i = int(rt[0]-1)
        j = int(rt[1]-1)
        M[i,j] = M[i,j] + 1
    return M

def normalize_by_row(M):
    for i in range(M.shape[0]):
        rowsum = np.nansum(M[i,:])
        if rowsum != 0:
            M[i,:] = M[i,:]/rowsum
    return M
    
def normalize_by_col(M):
    for i in range(M.shape[1]):
        colsum = np.nansum(M[:,i])
        if colsum != 0:
            M[:,i] = M[:,i]/colsum
    return M

def find_past_reward_behavior_transitions(table,t,LOOK_BACK_NUM = 2, reverse = False):

    trials = table.index[:-1]
    first_trial = trials[0]

    trials_from = np.max([t - 80,first_trial])
    trials_to = t

    transitions = []
    for tt in np.arange(trials_from,trials_to-1):
        i = int(table.loc[tt].OuterWellIndex)
        j = int(table.loc[tt+1].OuterWellIndex)
        if table.loc[tt+1].rewardNum == 2:
            if reverse:
                transitions.append((j,i))
            else:
                transitions.append((i,j))
    if len(transitions) > LOOK_BACK_NUM:
        transitions = transitions[len(transitions)-LOOK_BACK_NUM:]

    return transitions

def replay_PR_correlation(table,look_back,reverse):
    totalTrial = len(table.index)-1
    
    rewarded_transition = np.zeros((4,4,totalTrial))
    replay = np.zeros((4,4,totalTrial))

    t_ind = 0
    replay_trial_ind= []
    for t in table.index[:-1]:
        
        past_reward_transition = find_past_reward_behavior_transitions(table,t,LOOK_BACK_NUM = look_back, reverse = reverse)
        replay_transitions_t = table.loc[t].replayed_transitions
        
        rewarded_transition[:,:,t_ind] = transitions_to_matrix(past_reward_transition)
        replay[:,:,t_ind] = transitions_to_matrix(replay_transitions_t)
        if np.sum(replay[:,:,t_ind]) > 0:
            replay_trial_ind.append(t_ind)
        
        rewarded_transition[:,:,t_ind] = normalize_by_col(rewarded_transition[:,:,t_ind])

        t_ind += 1
        
    R = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            R[i,j] = np.corrcoef(rewarded_transition[i,j,replay_trial_ind],replay[i,j,replay_trial_ind])[0,1]
    
    return R, rewarded_transition[:,:,replay_trial_ind], replay[:,:,replay_trial_ind]

def replay_PR_correlation_day(animal,day,encoding_set,classifier_param_name,look_back,reverse):
    nwb_copy_file_name = animal.lower() + day + '_.nwb'
    epoch_names = (TrialChoice & {'nwb_file_name': nwb_copy_file_name}).fetch('epoch_name')
    
    rewarded_transitions = []
    replays =[]
    for epoch_name in epoch_names:
        table = pd.DataFrame((TrialChoiceReplayTransition & {
            'nwb_file_name': nwb_copy_file_name,
            'interval_list_name':epoch_name,
            'classifier_param_name':classifier_param_name,
            'encoding_set':encoding_set}).fetch1('choice_reward_replay_transition'))
        _, rewarded_transition, replay = replay_PR_correlation(table,look_back,reverse)
        rewarded_transitions.append(rewarded_transition)
        replays.append(replay)
    rewarded_transition_all = np.concatenate(rewarded_transitions,axis = 2)
    replay_all = np.concatenate(replays,axis = 2)
    
    R = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            R[i,j] = np.corrcoef(rewarded_transition_all[i,j,:],replay_all[i,j,:])[0,1]
        
    return R