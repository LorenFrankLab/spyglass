import numpy as np
import spyglass as nd
import pandas as pd
from spyglass.shijiegu.Analysis_SGU import (TrialChoice,TrialChoiceReplay)


def count_replay_by_category(replay_df,tally_df,category_names):
    '''
    replay_df is a list (of length trials) of list (arm replays on each trial)
    tally_df is a dataframe that has corresponence between arm and category
    '''
    assert len(replay_df)==len(tally_df)
    # initialize 
    category={}
    for n in category_names:
        category[n]=0
    # add in nan
    nan_count=0
    home_count=0
    
    # do counting
    for t in range(len(replay_df)):
        replay=replay_df[t]
        nan_count+=np.sum(np.isnan(replay))
        if len(replay)==0:
            continue
        home_count+=np.sum(np.array(replay)==0)
        for c in list(category.keys()):
            if c=='home':  
                continue
            else:
                category[c]+=np.sum(replay==tally_df.loc[tally_df.index[t],c])
                
    category['nan']=nan_count
    if np.isin('home',category_names):
        category['home']=home_count
    return category


def proportion(replay_dict):
    replay_dict_proportion={}
    total_replays=np.sum(np.array(list(replay_dict.values())))
    for c in list(replay_dict.keys()):
        replay_dict_proportion[c]=replay_dict[c]/total_replays
    return replay_dict_proportion,total_replays

def unravel_replay(replay_list):
    # first level is trial, then
    #  for each trial, there can be multiple events.
    #    within each event, there are segments of replay
    # This function only keeps the first level.
    replay_df=[]
    for t in range(len(replay_list)):
        replay_t=replay_list[t]
        replay_t_unraveled=[]
        for ri in range(len(replay_t)):
            for seg in replay_t[ri]:
                replay_t_unraveled.append(seg)
        replay_df.append(replay_t_unraveled)
    return replay_df


def find_distinct_subset(nwb_copy_file_name,epoch_num,category_names):
    '''find unique trials with distinct categories'''
    # e.g. category_names=['current','future_O','past','past_reward']
    
    log_df_tagged=pd.DataFrame((TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                                               'epoch':epoch_num}).fetch1('choice_reward'))
    cfpp=np.array(log_df_tagged.loc[:,category_names]) #'current','future','past','past_reward'

    unique_trials=np.argwhere([len(np.unique(t[~np.isnan(t)]))==len(category_names) for t in cfpp]).ravel()
    #print('number of valid trials: '+str(len(unique_trials)))

    log_df_subset=log_df_tagged.iloc[unique_trials,:].copy()
    #tally_df.index=np.arange(len(tally_df))
    return log_df_subset


def simulate_random_replay(replay_df):
    '''
    simulate random replay
    '''
    replay_df_permuted=np.concatenate(replay_df)
        
    replay_df_random=[np.random.choice(replay_df_permuted, # 4 arms
                                       len(replay_df[t])) for t in range(len(replay_df))]
    return replay_df_random


def replay_in_categories(nwb_copy_file_name,epoch_num,categories_H,
                        simulate_random_flag=False):
    '''
    the MASTER function of this script.
    '''

    # Behavior: find trials where categories are distinct outer arms
    behav_df_all=find_distinct_subset(nwb_copy_file_name,epoch_num,np.setdiff1d(categories_H,['home']))
    
    # Replay:
    key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num}
    replay_df_all=pd.DataFrame((TrialChoiceReplay & key).fetch1('choice_reward_replay'))
    subset_ind=np.intersect1d(behav_df_all.index,replay_df_all.index)
    
    # Restrict to the subset where we have replay
    behav_df=behav_df_all.loc[subset_ind,:].copy()
    print('number of valid trials: '+str(len(behav_df)))
    
    assert np.all(behav_df.OuterWellIndex==replay_df_all.loc[subset_ind,:]['OuterWellIndex'])
    
    if len(np.intersect1d(categories_H,['current']))>0:
        replay_H=list(replay_df_all.loc[subset_ind,'replay_O'])
    else:
        replay_H=list(replay_df_all.loc[subset_ind,'replay_H'])
    replay_df_H=unravel_replay(replay_H)
    
    if simulate_random_flag:
        replay_df_H=simulate_random_replay(replay_df_H)
    counts=count_replay_by_category(replay_df_H,behav_df,categories_H)
    
    return counts

def category_day(nwb_copy_file_name,epochs,
                 categories_H,
                 plot_categories_H,
                 simulate_random_flag=False):
    count_H_day={}
    for c in categories_H:
        count_H_day[c]=0

    for epoch_num in epochs:
        count_H=replay_in_categories(nwb_copy_file_name,
                                     epoch_num,categories_H,simulate_random_flag)
        for c in categories_H:
            count_H_day[c]+=count_H[c]

    count_H_day_hat={plot_categories_H[k]:
                          count_H_day[categories_H[k]] for k in range(len(categories_H))}     
    return count_H_day_hat