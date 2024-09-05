import numpy as np
import pandas as pd
from spyglass.common.common_interval import _intersection

def interpolate_to_new_time(df, new_time, upsampling_interpolation_method='linear'):
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate(
        (old_time, new_time))), name='time')
    tmp = df.reindex(index=
                     new_index
                     ).interpolate(
                         method=upsampling_interpolation_method).reindex(index=new_time)
    tmp.index.name = df.index.name
    return tmp

def interval_union(interval_list1,interval_list2):
    '''
    interval_list1 : np.array, (N,2) where N = number of intervals
    interval_list2 : np.array, (N,2) where N = number of intervals

    e.g.
    interval_list1 = np.array([[1,11],[6,8],[12,20],[25,30]])
    interval_list2 = np.array([[1,10]])

    Returns
    -------
    interval_list: np.array, (N,2)

    '''
    interval_list1=interval_list1.tolist()
    interval_list2=interval_list2.tolist()

    # find all pairwise intersections
    intersect_tally=np.zeros((len(interval_list1),len(interval_list2)))
    for i in range(len(interval_list1)):
        for j in range(len(interval_list2)):
            if _intersection(interval_list1[i],interval_list2[j]) is not None:
                intersect_tally[i,j]=1

    union_set=[]
    union_list2_ind=np.argwhere(np.sum(intersect_tally,axis=0)==0).ravel()
    union_list1_ind=np.argwhere(np.sum(intersect_tally,axis=1)==0).ravel()

    # for those in set 1 or set 2 that has no intersections, append zero row sum or column sum
    for i in union_list1_ind:
        union_set.append(interval_list1[i])
    for i in union_list2_ind:
        union_set.append(interval_list2[i])

    # for those that has intersections, find mutually intersecting interva;s
    for i in np.argwhere(np.sum(intersect_tally,axis=1)!=0).ravel():
        intvl_1=np.array([interval_list1[i]])

        intvl_1s=[]
        # first, find the intersecting one in set 2
        j_all=np.argwhere(intersect_tally[i,:]).ravel()
        intvl_2s=np.array([interval_list2[j] for j in j_all])

        # then track back in set 1, to find all the ones in set 1 intersecting this one in set 2
        for j in j_all:
            i_all=np.argwhere(intersect_tally[:,j]).ravel()
            for i in i_all:
                intvl_1s.append(interval_list1[i])

        # put all the related/intersecting intervals together, find union
        union_tmp=np.concatenate([np.array(intvl_1s),intvl_2s],axis=0)

        union_set.append([np.min(union_tmp[:,0]),np.max(union_tmp[:,1])])
    union_set=np.unique(union_set,axis=0)
    return union_set


def mergeIntervals(intervals):
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)

    return stack

def find_trial_id(c_end,log_df):
    '''
    find the trial in which an event time c_end occurs

    c_end is a float or np array
    log_df is pd datafrmae
    '''
    trial_id = None

    # if it is between outerwell (trial t) and home poke (trial t+1)
    tmp=c_end>log_df.timestamp_O
    trial_id_min=tmp[::-1].idxmax()
    trial_id_max=(c_end<log_df.timestamp_H).idxmax()
    if (trial_id_max-trial_id_min)==1:
        trial_id=trial_id_min
        return trial_id
    elif (trial_id_max-trial_id_min)>1: #trial t+1 did not poke home
        if (c_end-log_df.loc[trial_id_min].timestamp_O)<=5: #less than 5 seconds from outer poke:
            trial_id=trial_id_min
            return trial_id

    # if it is between home poke (trial t) and outerwell (trial t+1)
    tmp=c_end>log_df.timestamp_H
    trial_id_min=tmp[::-1].idxmax()
    trial_id_max=(c_end<log_df.timestamp_O).idxmax()
    if (trial_id_max-trial_id_min)==0:
        trial_id=trial_id_min
        return trial_id
    elif (trial_id_max-trial_id_min)>1: #trial t did not poke home
        if (c_end-log_df.loc[trial_id_min].timestamp_H)<=5: #less than 5 seconds from outer poke:
            trial_id=trial_id_min
            return trial_id