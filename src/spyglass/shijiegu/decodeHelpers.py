import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import logging
import os
import cupy as cp

from spyglass.common import (Session, IntervalList,IntervalPositionInfo,
                             LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common.common_interval import _intersection
from spyglass.common.common_position import IntervalLinearizedPosition

import spyglass.spikesorting.v0 as ss
from spyglass.spikesorting.v0 import (SortGroup,
                                    SortInterval,
                                    SpikeSortingPreprocessingParameters,
                                    SpikeSortingRecording,
                                    SpikeSorterParameters,
                                    SpikeSortingRecordingSelection,
                                    ArtifactDetectionParameters, ArtifactDetectionSelection,
                                    ArtifactRemovedIntervalList, ArtifactDetection,
                                      SpikeSortingSelection, SpikeSorting,
                                   CuratedSpikeSortingSelection,CuratedSpikeSorting,Curation)
from spyglass.decoding.v0.clusterless import (UnitMarks,
                                           UnitMarkParameters,UnitMarksIndicatorSelection,
                                          UnitMarksIndicator)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo, IntervalPositionInfoSelection

from replay_trajectory_classification.environments import Environment
from spyglass.common.common_position import TrackGraph
from spyglass.decoding.v0.clusterless import ClusterlessClassifierParameters
from replay_trajectory_classification import ClusterlessClassifier
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1

import pprint
# Here are the analysis tables specific to Shijie Gu
from spyglass.shijiegu.Analysis_SGU import EpochPos,TrialChoice,Decode,DecodeIngredients
from spyglass.shijiegu.helpers import interval_union

def runSessionNames(nwb_copy_file_name):
    # select position timestamps, only maze sessions are selected

    epochNames = (EpochPos & {'nwb_file_name':nwb_copy_file_name}).fetch('epoch_name')
    positionNames = (EpochPos & {'nwb_file_name':nwb_copy_file_name}).fetch('position_interval')

    position_interval=[]
    session_interval=[]

    for i in range(len(epochNames)):
        interval=epochNames[i]
        if interval[7:10]=='Ses':
            session_interval.append(interval)
            position_interval.append(positionNames[i])
    return session_interval, position_interval

def decodePrepMasterSession(nwb_copy_file_name,e,populate = True):

    session_interval, position_interval = runSessionNames(nwb_copy_file_name)

    """
    Ephys: UnitMarksIndicator
    """
    sorting_keys = thresholder_sort(nwb_copy_file_name,session_interval[e],populate)
    mark_parameters_keys = populateUnitMarks(sorting_keys)

    positionIntervalList = (
        IntervalList &
        {'nwb_file_name': nwb_copy_file_name,
         'interval_list_name': position_interval[e]})

    marks_selection = ((UnitMarks & mark_parameters_keys) * positionIntervalList)
    marks_selection = (pd.DataFrame(marks_selection)
                    .loc[:, marks_selection.primary_key]
                    .to_dict('records'))
    UnitMarksIndicatorSelection.insert(marks_selection, skip_duplicates=True)
    UnitMarksIndicator.populate(marks_selection)

    marks = (UnitMarksIndicator & mark_parameters_keys).fetch_xarray()

    """
    Position: UnitMarksIndicator
    """
    linear_position_df = (IntervalLinearizedPosition() &
        {'nwb_file_name': nwb_copy_file_name,
        'interval_list_name': position_interval[e],
        'position_info_param_name': 'default_decoding'}
        ).fetch1_dataframe()
    position_df = (IntervalPositionInfo &
        {'nwb_file_name': nwb_copy_file_name,
        'interval_list_name': position_interval[e],
        'position_info_param_name': 'default_decoding'}
                    ).fetch1_dataframe()

    # Remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':session_interval[e]}).fetch1('choice_reward'))

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
    linear_position_df = removeDataBeforeTrial1(linear_position_df,trial_1_t,trial_last_t)
    position_df = removeDataBeforeTrial1(position_df,trial_1_t,trial_last_t)

    """
    Intervals
    """
    intersect_interval = intersectValidIntervals(nwb_copy_file_name,
                                                session_interval[e],position_interval[e])
    marks_=[]
    linear_position_df_=[]
    position_df_ =[]
    for i in range(len(intersect_interval)):
        valid_time_slice = slice(intersect_interval[i][0], intersect_interval[i][1])

        linear_position_df_.append(linear_position_df.loc[valid_time_slice])
        position_df_.append(position_df.loc[valid_time_slice])
        marks_.append(marks.sel(time=valid_time_slice))

    marks=xr.concat(marks_,dim='time')
    linear_position_df=pd.concat(linear_position_df_)
    position_df=pd.concat(position_df_)
    print('final shape of marks and linear position df is:')
    print(marks.shape)
    print(linear_position_df.shape)

    """save result"""
    animal = nwb_copy_file_name[:5]
    marks_path=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                             nwb_copy_file_name+'_'+session_interval[e]+'_marks.nc')
    marks.to_netcdf(marks_path)

    position1d_path=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                             nwb_copy_file_name+'_'+session_interval[e]+'_1dposition.csv')
    linear_position_df.to_csv(position1d_path)

    position2d_path=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                             nwb_copy_file_name+'_'+session_interval[e]+'_2dposition.csv')
    position_df.to_csv(position2d_path)

    key={'nwb_file_name':nwb_copy_file_name,
     'interval_list_name':session_interval[e],
     'marks':marks_path,
     'position_1d':position1d_path,
     'position_2d':position2d_path}
    DecodeIngredients().insert1(key,replace=True)

    return marks, linear_position_df

def thresholder_sort(nwb_copy_file_name,sort_interval_name,populate=True):


    """fetch artifact"""
    artifact_params_name='ampl_100_prop_05_2ms'
    artifact_key=(ArtifactDetection() & {'nwb_file_name' : nwb_copy_file_name}
                              & {'artifact_params_name': artifact_params_name}
                              & {'sort_interval_name':sort_interval_name})

    #, and there should be 2 entries, one from left cannula (group 100), and one from right cannula (group 101)
    artifact_removed_name_list=artifact_key.fetch('artifact_removed_interval_list_name')
    assert len(artifact_removed_name_list)==2

    artifact_time_list=[]
    artifact_removed_time_list=[]
    for artifact_removed_name in artifact_removed_name_list:
        artifact_time_list.append((ArtifactDetection() & {'nwb_file_name' : nwb_copy_file_name,'artifact_removed_interval_list_name':artifact_removed_name}).fetch1('artifact_times'))
        artifact_removed_time_list.append((IntervalList() & {'nwb_file_name' : nwb_copy_file_name,'interval_list_name':artifact_removed_name}).fetch1('valid_times'))

    artifact_time_list=interval_union(artifact_time_list[0],artifact_time_list[1])

    artifact_removed_time_list=interval_list_intersect(
        np.array(artifact_removed_time_list[0]),np.array(artifact_removed_time_list[1]))

    """remove time before the rat is on track"""
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':sort_interval_name}).fetch1('choice_reward')
    )
    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    artifact_removed_time_list=interval_list_intersect(artifact_removed_time_list,np.array([trial_1_t,trial_last_t]))

    # find tetrodes
    tetrode_with_cell=np.unique((SpikeSortingRecordingSelection & {'nwb_file_name':nwb_copy_file_name}).fetch('sort_group_id'))
    tetrode_with_cell=np.setdiff1d(tetrode_with_cell,[100,101])

    # insert into individual tetrodes ArtifactRemovedIntervalList
    artifact_params_name='ampl_1500_prop_075_1ms'
    sorting_keys=[]
    for tetrode in tetrode_with_cell:
        print(tetrode)
        artifact_key = {'nwb_file_name' : nwb_copy_file_name,
                    'sort_interval_name' : sort_interval_name,
                    'sort_group_id' : tetrode,
                    'preproc_params_name': 'franklab_tetrode_hippocampus',
                    'team_name': 'Shijie Gu'}
        artifact_key['artifact_params_name'] = artifact_params_name

        if populate:
            ArtifactDetectionSelection.insert1(artifact_key, skip_duplicates=True)

        artifact_key['artifact_times']=artifact_time_list
        artifact_key['artifact_removed_valid_times']=artifact_removed_time_list



        artifact_name=(nwb_copy_file_name+'_'+sort_interval_name+'_'+str(artifact_key['sort_group_id'])+'_'+
                       artifact_key['preproc_params_name']+'_'+artifact_params_name+'_artifact_removed_valid_times_track_time_only')

        artifact_key['artifact_removed_interval_list_name']=artifact_name

        if populate:
            ArtifactRemovedIntervalList().insert1(artifact_key,skip_duplicates=True)

            print('inserting into IntervalList')
            IntervalList().insert1({'nwb_file_name' : nwb_copy_file_name,
                               'interval_list_name':artifact_name,
                               'valid_times':artifact_removed_time_list},skip_duplicates=True)

        print('done inserting into IntervalList')

        artifact_key.pop('artifact_params_name')
        artifact_key.pop('artifact_times')
        artifact_key.pop('artifact_removed_valid_times')

        artifact_key['sorter'] = 'clusterless_thresholder'
        artifact_key['sorter_params_name'] = 'default'

        if populate:
            SpikeSortingSelection.insert1(artifact_key, skip_duplicates=True)

        sorting_keys.append(artifact_key.copy())
    if populate:
        #SpikeSorting.populate({'nwb_file_name':nwb_copy_file_name,
        #                       'artifact_params_name': artifact_params_name, # make sure to inlude this
        #                       'sort_interval_name':sort_interval_name,
        #                       'artifact_removed_interval_list_name':artifact_name})
        SpikeSorting.populate(sorting_keys)
    return sorting_keys

def populateUnitMarks(sorting_keys):
    """
    sorting_keys: is the SpikeSortingSelection entries for thresholder sort:
        Get it from thresholder_sort().

    """
    curation_keys = [Curation.insert_curation(key) for key in sorting_keys]

    '''
    keys = {}
    keys['nwb_file_name'] = key['nwb_file_name']
    keys['sort_interval_name'] = key['sort_interval_name']
    curation_keys = (Curation & keys).fetch(as_dict=True)

    for key in curation_keys:
        key.pop('parent_curation_id')
        key.pop('curation_labels')
        key.pop('merge_groups')
        key.pop('quality_metrics')
        key.pop('description')
        key.pop('time_of_creation')
    '''

    CuratedSpikeSortingSelection().insert(
        curation_keys,
        skip_duplicates=True)

    CuratedSpikeSorting.populate(
        CuratedSpikeSortingSelection() & curation_keys)

    mark_parameters_keys = pd.DataFrame(CuratedSpikeSorting & curation_keys)
    mark_parameters_keys['mark_param_name'] = 'default'
    mark_parameters_keys = (mark_parameters_keys.loc[:, UnitMarkParameters.primary_key].to_dict('records'))

    UnitMarkParameters().insert(
        mark_parameters_keys,
        skip_duplicates=True)

    UnitMarks.populate(UnitMarkParameters & mark_parameters_keys)
    return mark_parameters_keys

def intersectValidIntervals(nwb_copy_file_name,sort_interval_name,pos_interval_name):

    key = {}
    key['interval_list_name'] = sort_interval_name
    key['nwb_file_name'] = nwb_copy_file_name

    interval = (IntervalList &
                {'nwb_file_name': key['nwb_file_name'],
                'interval_list_name': key['interval_list_name']}
            ).fetch1('valid_times')

    valid_ephys_times = (IntervalList &
                        {'nwb_file_name': key['nwb_file_name'],
                        'interval_list_name': 'raw data valid times'}
                        ).fetch1('valid_times')

    valid_pos_times = (IntervalList &
                    {'nwb_file_name': key['nwb_file_name'],
                        'interval_list_name': pos_interval_name}
                    ).fetch1('valid_times')

    StateScript = pd.DataFrame(
            (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                            'epoch_name':sort_interval_name}).fetch1('choice_reward'))

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    intersect_interval_ = interval_list_intersect(
        interval_list_intersect(interval, valid_ephys_times), valid_pos_times)
    intersect_interval = interval_list_intersect(intersect_interval_,np.array([trial_1_t,trial_last_t]))

    return intersect_interval
