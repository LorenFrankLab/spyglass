import numpy as np
import os
import pickle
import pandas as pd

from spyglass.shijiegu.ripple_add_replay import remove_adjacent

from spyglass.shijiegu.ripple_add_replay import remove_adjacent
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import TrialChoice,TrialChoiceReplayTransition

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

def merge_overlapping_ranges(ranges):
    '''
    return interals and indices of intervals that got merged
    >>> list(merge_overlapping_ranges([(3.1, 5.3), (4.2, 7.5), (10, 11)]))
    [(3.1, 7.5, 0, 1),(10,11, 2, 2)]
    '''

    ranges =sorted(ranges)
    if len(ranges)==0:
        return []
    current_start, current_stop=ranges[0]
    previous_j=0
    for j in range(len(ranges)):
        start, stop=ranges[j]
        if start>current_stop:
            # Gap between segments: output current segment and start a new
            # one.
            yield current_start, current_stop, previous_j,j-1
            current_start, current_stop = start, stop
            previous_j = j
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop, previous_j, len(ranges)-1

def removeEmptyDecode(arms):
    # within each ripple event, there can be periods where there is no decode due to ambiguity
    # also concatenated arms
    """
    Example:
    input: [[[4], [0], [3]], [[4], [0], [0]], [[], [], [], [0]], [[4], []], [[0], [3]]]
    output: arms_ = [[[4], [0], [3]], [[4], [0], [0]], [[0]], [[4]], [[0], [3]]],
            ind_ = [[0, 1, 2], [0, 1, 2], [3], [0], [0, 1]]
    """

    ind_ = []
    arms_ = []
    for a in arms:
        i = 0
        arms_a = []
        ind_a = []
        for ai in a:
            if len(ai)>0:
                ind_a.append(i)
                arms_a.append(ai)
            i = i + 1
        ind_.append(ind_a)
        arms_.append(arms_a)
    return arms_,ind_

def findReplayPair(T):
    """
    T is choice_reward_replay table
    """
    #length of T.loc[ind,'ripple_H'] is the number of seperate ripple events
    #within each entry, there are N number of continous segments

    transitions_all=np.zeros((6,6))
    T_transition=T.copy()
    T_transition.insert(4,'replayed_transitions',[[] for i in range(len(T))]) #hold replayed transitions

    for t in T.index:
        arms=T.loc[t,'replay_H'].copy()
        ripple_times=T.loc[t,'ripple_H'].copy()

        if len(arms)==0:
            continue
        # remove empty entries (ripples that are pure fragmented)
        replays_ind=[i for i in range(len(arms)) if len(arms[i])>0]
        ripple_times=[ripple_times[i] for i in replays_ind]
        arms=[arms[i] for i in replays_ind]

        if len(ripple_times)==0:
            continue

        arms, nonemptyInd = removeEmptyDecode(arms)
        ripple_times_nonempty = []
        for r_ind in range(len(ripple_times)):
            ripple_times_nonempty.append(ripple_times[r_ind][nonemptyInd[r_ind]])

        #ripple_times = np.concatenate(ripple_times)
        #ripple_times = ripple_times[nonemptyInd,:]
        #arms = np.array(arms)

        for ripple_ind_this_trial in range(len(ripple_times_nonempty)):
            ripple_times_ = ripple_times_nonempty[ripple_ind_this_trial]
            arms_ = arms[ripple_ind_this_trial]

            ripple_times_[:,0]=ripple_times_[:,0]-0.1
            ripple_times_[:,1]=ripple_times_[:,1]+0.1

            ripple_times_list=[list(a) for a in ripple_times_]
            ripple_bouts=list(merge_overlapping_ranges(ripple_times_list))
            for bout in ripple_bouts:

                # move all arms between the first and the last segment into one bout
                arms_bout = np.concatenate([arms_[i] for i in np.arange(int(bout[2]),int(bout[3])+1)])

                # remove home and center platform
                arm_ind=~np.isin(arms_bout,[0,5])
                arms_bout=arms_bout[arm_ind]

                # remove nans
                notnan_ind=~np.isnan(arms_bout)
                arms_bout=arms_bout[notnan_ind]

                if len(arms_bout)==0:
                    continue
                if len(arms_bout)>1:
                    # get rid of adjacent duplicates
                    arms_bout=np.array(remove_adjacent(list(arms_bout)))
                    for m in range(len(arms_bout)-1):
                        i=int(arms_bout[m])
                        j=int(arms_bout[m+1])
                        transitions_all[i,j]+=1
                        if i!=j:
                            T_transition.at[t,'replayed_transitions'].append((i,j))
                else:
                    transitions_all[int(arms_bout[0]),int(arms_bout[0])]+=1

    transitions_all_ = transitions_all[1:5,:]
    transitions_all_ = transitions_all_[:,1:5]
    return T_transition, transitions_all_


# replay transitions
def replay_day_transitions(nwb_copy_file_name,encoding_set,classifier_param_name = ''):

    #nwb_copy_file_name=get_nwb_copy_filename(nwb_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)

    transitions_all=np.zeros((4,4))
    for epoch_name in session_interval:
        print(epoch_name)

        key = {'nwb_file_name': nwb_copy_file_name,
               'interval_list_name': epoch_name,
               'classifier_param_name':classifier_param_name,
               'encoding_set':encoding_set}

        transitions_session = (TrialChoiceReplayTransition & key).fetch1('transitions')
        transitions_all = transitions_all + transitions_session

    np.fill_diagonal(transitions_all,np.nan)
    count = np.nansum(transitions_all)
    #transitions_all_percentage = transitions_all/count
    return count, (transitions_all, normalize_offdiag_rowwise(transitions_all), normalize_offdiag_colwise(transitions_all))

def replay_transitions(animal,dates_to_plot,encoding_set,classifier_param_name=''):
    P_replay_all = {}
    count_all = {}
    for d in dates_to_plot:
        nwb_file_name = animal.lower() + d + '_.nwb'
        count_all[d], P_replay_all[d] = replay_day_transitions(nwb_file_name,encoding_set,classifier_param_name)
    return count_all, P_replay_all

def matrix_correlation(M1,M2):
    return np.sum(np.multiply(M1,M2))/(np.linalg.norm(M1,ord='fro')*np.linalg.norm(M2,ord='fro'))

def normalize_offdiag_rowwise(T_):
    T=T_.copy()
    for ti in range(4):
        if np.nansum(T[ti])!=0:
            T[ti]=T[ti]/np.nansum(T[ti])
    return T

def normalize_offdiag_colwise(T_):
    T=T_.copy()
    for ti in range(4):
        if np.nansum(T[:,ti])!=0:
            T[:,ti]=T[:,ti]/np.nansum(T[:,ti])
    return T

def find_past_behavior_transitions(table,t,LOOK_BACK_NUM = 10):

    trials = table.index[:-1]
    first_trial = trials[0]

    trials_from = np.max([t - LOOK_BACK_NUM,first_trial])
    trials_to = t

    transitions = []
    for tt in np.arange(trials_from,trials_to-1):
        transitions.append((int(table.loc[tt].OuterWellIndex),int(table.loc[tt+1].OuterWellIndex)))

    return set(transitions)

def find_future_behavior_transitions(table,t,LOOK_NUM = 10):

    trials = table.index[:-1]
    last_trial = trials[-1]

    trials_to = np.min([t + LOOK_NUM,last_trial])
    trials_from = t

    transitions = []
    for tt in np.arange(trials_from,trials_to-1):
        transitions.append((int(table.loc[tt].OuterWellIndex),int(table.loc[tt+1].OuterWellIndex)))

    return set(transitions)

def find_random_behavior_transitions(T_1d,LEN):

    ij = np.array([(j,i) for j in [1,2,3,4] for i in [1,2,3,4]])
    ij_ind = np.arange(16)

    ij_ind_random = np.random.choice(ij_ind, size = LEN, replace = True, p = T_1d)
    return set([tuple(ij[ij_i]) for ij_i in ij_ind_random])

def find_past_reward_behavior_transitions(table,t,LOOK_BACK_NUM = 2):

    trials = table.index[:-1]
    first_trial = trials[0]

    trials_from = np.max([t - 80,first_trial])
    trials_to = t

    transitions = []
    for tt in np.arange(trials_from,trials_to-1):
        i = int(table.loc[tt].OuterWellIndex)
        j = int(table.loc[tt+1].OuterWellIndex)
        if table.loc[tt+1].rewardNum == 2:
            transitions.append((i,j))
    if len(transitions) > LOOK_BACK_NUM:
        transitions = transitions[len(transitions)-LOOK_BACK_NUM:]

    return set(transitions)

def find_past_nonreward_behavior_transitions(table,t,LOOK_BACK_NUM = 2):

    trials = table.index[:-1]
    first_trial = trials[0]

    trials_from = np.max([t - 80,first_trial])
    trials_to = t

    transitions = []
    for tt in np.arange(trials_from,trials_to-1):
        i = int(table.loc[tt].OuterWellIndex)
        j = int(table.loc[tt+1].OuterWellIndex)
        if table.loc[tt+1].rewardNum == 1:
            transitions.append((i,j))
    if len(transitions) > LOOK_BACK_NUM:
        transitions = transitions[len(transitions)-LOOK_BACK_NUM:]

    return set(transitions)

def find_reward_reward_behavior_transitions(table,t,LOOK_BACK_NUM = 2):
    bridge_subset = table.loc[table.rewardNum == 2]

    trials = table.index[:-1]
    first_trial = trials[0]

    trials_from = np.max([t - 80,first_trial])
    trials_to = t
    trials_to_consider = np.arange(trials_from,trials_to)
    trials_to_consider = np.intersect1d(trials_to_consider,bridge_subset.index)
    if len(trials_to_consider) > LOOK_BACK_NUM + 1:
        trials_to_consider = trials_to_consider[(len(trials_to_consider)-(LOOK_BACK_NUM + 1)):]

    transitions = []
    for tt_ind in range(len(trials_to_consider)-1):
        tt = trials_to_consider[tt_ind]
        tt_plus1= trials_to_consider[tt_ind+1]
        i = int(table.loc[tt].OuterWellIndex)
        j = int(table.loc[tt_plus1].OuterWellIndex)
        transitions.append((i,j))

    return set(transitions)


def trial_by_trial_behavior_replay_pairs(animal,dates_to_plot,encoding_set,
                                             classifier_param_name,LOOK_BACK_NUM,
                                             replay_mode = 'normal',mode = 'PAST',random = False):
    """
    replay_mode: if 'reverse', it will swap (swap arm_a,arm_b) to (arm_b,arm_a)
    """
    count_replay = {}
    p_replay_all = {}
    count_transition = {}
    p_transition_all = {}
    for d in dates_to_plot:
        nwb_file_name = animal.lower() + d + '_.nwb'
        (p_replay_all[d], count_replay[d],
         p_transition_all[d], count_transition[d]) = trial_by_trial_behavior_replay_pairs_day(nwb_file_name,encoding_set,
                                             classifier_param_name,LOOK_BACK_NUM,
                                             replay_mode,mode,random)
    return p_replay_all, count_replay, p_transition_all, count_transition

def trial_by_trial_behavior_replay_pairs_day(nwb_copy_file_name,encoding_set,
                                             classifier_param_name,LOOK_BACK_NUM,
                                             replay_mode,mode,random):
    epoch_names = (TrialChoice & {'nwb_file_name': nwb_copy_file_name}).fetch('epoch_name')
    denominator_replay = 0
    nominator_replay = 0
    denominator_transition = 0
    nominator_transition = 0
    for epoch_name in epoch_names:
        (nominator_replay_, denominator_replay_,
         nominator_transition_, denominator_transition_
         ) = trial_by_trial_behavior_replay_pairs_session(nwb_copy_file_name,epoch_name,
                                                 encoding_set,classifier_param_name,
                                                 LOOK_BACK_NUM,
                                                 replay_mode,mode,
                                                 random)
        nominator_replay += nominator_replay_
        denominator_replay += denominator_replay_
        nominator_transition += nominator_transition_
        denominator_transition += denominator_transition_

    if denominator_transition == 0:
        p_transition = 0
    else:
        p_transition = nominator_transition/denominator_transition
    return (nominator_replay/denominator_replay,denominator_replay,
            p_transition,denominator_transition)


def trial_by_trial_behavior_replay_pairs_session(nwb_copy_file_name,epoch_name,
                                                 encoding_set,classifier_param_name,
                                                 LOOK_BACK_NUM,replay_mode,mode,random):
    table = pd.DataFrame((TrialChoice & {'nwb_file_name': nwb_copy_file_name,
                             'epoch_name':epoch_name}).fetch1('choice_reward'))
    key = {'nwb_file_name': nwb_copy_file_name,
       'interval_list_name': epoch_name,
       'classifier_param_name':classifier_param_name,
       'encoding_set':encoding_set}
    replay_table = pd.DataFrame((TrialChoiceReplayTransition & key).fetch1('choice_reward_replay_transition'))

    """This section is for random selection"""
    #T,_ = behavior_transitions_count_session(nwb_copy_file_name,epoch_name)
    T = np.ones(4)-np.eye(4)
    T = T/np.sum(T)
    T_1d = np.reshape(T,(1,-1)).ravel()


    trials = table.index[:-1]
    denominator_replay = 0
    nominator_replay = 0

    nominator_transition = 0
    denominator_transition = 0
    for t in trials:
        if mode == 'PAST':
            transitions = find_past_behavior_transitions(table,t,LOOK_BACK_NUM)
        elif mode == 'PAST_REWARD':
            transitions = find_past_reward_behavior_transitions(table,t,LOOK_BACK_NUM)
        elif mode == 'REWARD_REWARD':
            transitions = find_reward_reward_behavior_transitions(table,t,LOOK_BACK_NUM)
        elif mode == 'NONREWARD':
            transitions = find_past_nonreward_behavior_transitions(table,t,LOOK_BACK_NUM)
        elif mode == 'RANDOM':
            transitions = find_random_behavior_transitions(T_1d,LOOK_BACK_NUM)
        elif mode == 'FUTURE':
            transitions = find_future_behavior_transitions(table,t,LOOK_BACK_NUM)
        replay_t = replay_table.loc[t].replayed_transitions
        for replay_t_ in replay_t:
            if replay_mode == 'reverse':
                replay_t_totest = replay_t_[::-1]
            else:
                replay_t_totest = replay_t_
            denominator_replay += 1
            nominator_replay += int(set([replay_t_totest]).issubset(transitions))

        for transition in transitions:
            if replay_mode == 'reverse':
                transition_totest = transition[::-1]
            else:
                transition_totest = transition
            denominator_transition += 1
            nominator_transition += int( set([transition_totest]).issubset(set(replay_t)) )
    return nominator_replay, denominator_replay, nominator_transition, denominator_transition

def trial_by_trial_random_behavior_replay_pairs(animal,dates_to_plot,encoding_set,
                                                classifier_param_name,LOOK_BACK_NUM,replay_mode,behavior_mode,p = 0.05):
    p_replay_random_all = {}
    p_transition_random_all = {}
    for d in dates_to_plot:
        p_replay_random_all[d] = []
        p_transition_random_all[d] = []
    for i in range(100):
        print("working on bootstrap iteration:",i)
        p_replay, _ , p_transition, _ = trial_by_trial_behavior_replay_pairs(animal,dates_to_plot,encoding_set,
                                             classifier_param_name,LOOK_BACK_NUM,replay_mode,behavior_mode,True)
        for d in dates_to_plot:
            p_replay_random_all[d].append(p_replay[d])
            p_transition_random_all[d].append(p_transition[d])

    p_replay_random_mean = {}
    p_replay_random_25 = {}
    p_replay_random_975 = {}
    for d in dates_to_plot:
        p_replay_random_mean[d] = np.mean(np.array(p_replay_random_all[d]))
        p_replay_random_25[d] = np.quantile(np.array(p_replay_random_all[d]),p/2)
        p_replay_random_975[d] = np.quantile(np.array(p_replay_random_all[d]),1-p/2)

    p_transition_random_mean = {}
    p_transition_random_25 = {}
    p_transition_random_975 = {}
    for d in dates_to_plot:
        p_transition_random_mean[d] = np.mean(np.array(p_transition_random_all[d]))
        p_transition_random_25[d] = np.quantile(np.array(p_transition_random_all[d]),p/2)
        p_transition_random_975[d] = np.quantile(np.array(p_transition_random_all[d]),1-p/2)
    return (p_replay_random_mean, p_replay_random_25, p_replay_random_975,
            p_transition_random_mean, p_transition_random_25, p_transition_random_975)

def behavior_transitions_count(animal,dates_to_plot):
    C_all = {}
    C_reward_all = {}
    for d in dates_to_plot:
        nwb_file_name = animal.lower() + d + '_.nwb'
        C_all[d], C_reward_all[d] = behavior_transitions_count_day(nwb_file_name)
    return C_all, C_reward_all

def behavior_transitions_count_day(nwb_copy_file_name):
    epoch_names = (TrialChoice & {'nwb_file_name': nwb_copy_file_name}).fetch('epoch_name')
    C = np.zeros((4,4))
    C_reward = np.zeros((4,4))
    for epoch_name in epoch_names:
        C_,C_rewarded_ = behavior_transitions_count_session(nwb_copy_file_name,epoch_name)
        C += C_
        C_reward += C_rewarded_
    return C, C_reward


def behavior_transitions_count_session(nwb_copy_file_name,epoch_name):
    table = pd.DataFrame((TrialChoice & {'nwb_file_name': nwb_copy_file_name,
                             'epoch_name':epoch_name}).fetch1('choice_reward'))
    T = np.zeros((4,4))
    T_rewarded = np.zeros((4,4))

    for t in table.index[:-2]:
        if (table.loc[t+1].name-table.loc[t].name)==1: #adjacent trials
            i = int(table.loc[t].OuterWellIndex)-1
            j = int(table.loc[t+1].OuterWellIndex)-1
            T[i,j] += 1 # minus 1 due to python indexing
            if (table.loc[t+1].rewardNum == 2): # rewarded trial
                T_rewarded[i,j] += 1
    return T,T_rewarded

seq1=[1,3,2,4]
seq2=[1,3,4,2]
seq3=[1,2,3,4]

rev1=[1,4,2,3]
rev2=[1,2,4,3]
rev3=[1,4,3,2]
seqs=np.vstack((seq1,seq2,seq3,rev1,rev2,rev3))
P_task=np.zeros((4,4,6))
for s in range(6):
    for a in range(4):
        P_task[seqs[s,a%4]-1,seqs[s,(a+1)%4]-1,s]=1

def findXCorrAllDays(transition_dict):
    xcorr = {}
    for d in transition_dict.keys():
        np.fill_diagonal(transition_dict[d],0)
        xcorr_behavior_task=[matrix_correlation(transition_dict[d],P_task[:,:,s]) for s in range(P_task.shape[2])]
        xcorr[d]=xcorr_behavior_task

    xcorr_replay_plot = []
    for d in transition_dict.keys():
        xcorr_replay_plot.append(xcorr[d])
    xcorr_replay_plot = np.array(xcorr_replay_plot)

    return xcorr_replay_plot

# behavior transitions
def behavior_transitions(animal,dates_to_plot):
    # load behavior meta data, use dates as key

    datafolder = '/cumulus/shijie/behavior_pilot/Batch1'
    data_pair2=pickle.load(open(os.path.join(datafolder,animal,'behavior_metaPairwise_'+animal+'.p'), "rb"))

    dates = data_pair2['dates']
    P_behavior_all = data_pair2['P_behavior_all']
    xcorr = data_pair2['xcorr']
    xcorr25_ = data_pair2['xcorr25']
    xcorr975_ = data_pair2['xcorr975']

    xcorr25 = {}
    xcorr975 = {}
    for d_ind in range(len(dates_to_plot)):
        d = dates_to_plot[d_ind]
        xcorr25[d] = xcorr25_[d_ind]
        xcorr975[d] = xcorr975_[d_ind]

    xcorr_plot = []
    xcorr25_plot = []
    xcorr975_plot = []
    for d_ind in range(len(dates_to_plot)):
        d = dates_to_plot[d_ind]
        xcorr_plot.append(xcorr[d])
        xcorr25_plot.append(xcorr25[d])
        xcorr975_plot.append(xcorr975[d])
    xcorr_plot = np.array(xcorr_plot)
    xcorr25_plot = np.array(xcorr25_plot)
    xcorr975_plot = np.array(xcorr975_plot)

    return P_behavior_all,xcorr_plot,xcorr25_plot,xcorr975_plot
