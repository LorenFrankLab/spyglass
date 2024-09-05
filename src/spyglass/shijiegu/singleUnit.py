import numpy as np
import pandas as pd
import uuid
from pathlib import Path
from spyglass.settings import waveforms_dir
import spikeinterface as si
import json
import os
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
from spyglass.spikesorting.v0.curation_figurl import CurationFigurl,CurationFigurlSelection
from spyglass.spikesorting.v0.spikesorting_curation import MetricParameters,MetricSelection,QualityMetrics
from spyglass.spikesorting.v0.spikesorting_curation import WaveformParameters,WaveformSelection,Waveforms
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from pprint import pprint
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import norm

from spyglass.shijiegu.helpers import interval_union
from spyglass.shijiegu.Analysis_SGU import TrialChoice,RippleTimes,EpochPos,get_linearization_map
from spyglass.shijiegu.load import load_run_sessions
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.shijiegu.helpers import interpolate_to_new_time

SNR_THRESHOLD = 8

def do_mountainSort(nwb_copy_file_name,session_name):
    tetrode_with_cell=np.unique((SpikeSortingRecordingSelection & {'nwb_file_name':nwb_copy_file_name}).fetch('sort_group_id'))
    tetrode_with_cell=np.setdiff1d(tetrode_with_cell,[100,101])

    for tetrode in tetrode_with_cell:
        artifacts_keys_tmp = (ArtifactRemovedIntervalList & {'nwb_file_name' : nwb_copy_file_name,
                            'sort_interval_name' : session_name,
                            'sort_group_id' : tetrode,
                            'preproc_params_name': 'franklab_tetrode_hippocampus'}).fetch(as_dict=True)
        for k in artifacts_keys_tmp:
            if k['artifact_removed_interval_list_name'][-15:] == 'track_time_only':
                artifact_key = k

        artifact_key.pop('artifact_params_name')
        artifact_key.pop('artifact_times')
        artifact_key.pop('artifact_removed_valid_times')
        artifact_key['sorter'] = "mountainsort4"
        artifact_key['sorter_params_name'] = "CA1_tet_Shijie"

        SpikeSortingSelection.insert1(artifact_key, skip_duplicates=True)
        SpikeSorting.populate(artifact_key)

def session_unit(nwb_copy_file_name,session_name):
    """return nwb_units for only large SNR units"""
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "sort_interval_name":session_name}

    nwb_units_all = {}
    sort_group_ids = np.unique((QualityMetrics & key).fetch("sort_group_id"))

    sort_group_ids_with_good_cell = []
    for sort_group_id in sort_group_ids:
        nwb_units = electrode_unit(nwb_copy_file_name,session_name,sort_group_id)
        if len(nwb_units)==0:
            continue
        nwb_units_all[sort_group_id] = nwb_units
        sort_group_ids_with_good_cell.append(sort_group_id)
    return nwb_units_all, sort_group_ids_with_good_cell

def electrode_unit(nwb_copy_file_name,session_name,sort_group_id):
    """return nwb_units for only large SNR units"""

    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "sort_interval_name":session_name}

    key["sort_group_id"] = sort_group_id

    metrics_json_path=(QualityMetrics & key).fetch1("quality_metrics_path")

    # Opening JSON file
    with open(metrics_json_path) as json_file:
        metrics_json = json.load(json_file)
    snr = metrics_json['snr']

    nwb_units = (CuratedSpikeSorting() & key).fetch_nwb()[0]["units"]
    if len(nwb_units) == 0:
        return None
    SPIKECOUNT_THRESHOLD = 14 * np.diff(nwb_units.iloc[0].sort_interval.ravel()) #14Hz for half an hour session

    accepted_units1=[unit_id for unit_id in nwb_units.index if snr[str(unit_id)] >= SNR_THRESHOLD] #SNR_THRESHOLD = 10
    accepted_units = [unit_id for unit_id in accepted_units1 if len(nwb_units.loc[unit_id].spike_times)< SPIKECOUNT_THRESHOLD ]
    #print('Accepted units: ', accepted_units)

    rejected_units = np.setdiff1d(nwb_units.index,accepted_units)
    nwb_units = nwb_units.drop(rejected_units)

    return nwb_units

def find_spikes(electrodes_units,cell_list,axis):
    """
    Get firing rate matrix over axis
    electrodes_units is a dictionary where keys are electrode
    cell_list is a list of cells, each is a tuple (electrode, unit),
        the returned matrix will be in this order in column
    """
    DELTA_T = axis[1] - axis[0]
    firing_matrix = np.zeros((len(axis)-1,len(cell_list)))
    for i in range(len(cell_list)):
        (e, u) = cell_list[i]
        firing_matrix[:,i], _ = np.histogram(electrodes_units[e].loc[u].spike_times,axis)
    firing_matrix = firing_matrix / DELTA_T

    return firing_matrix

def xcorr(firing_matrix):
    xcorr_matrix = []
    pairs = []
    (time_num, cell_num) = np.shape(firing_matrix)
    lag = signal.correlation_lags(time_num, time_num, mode='full')
    for i in range(cell_num):
        for j in range(i+1,cell_num):
            x = firing_matrix[:,i]
            y = firing_matrix[:,j]
            x = (x-np.mean(x))/norm(x)
            y = (y-np.mean(y))/norm(y)
            corr = signal.correlate(x, y, mode='full')
            xcorr_matrix.append(corr)
            pairs.append((i,j))
    xcorr_matrix = np.array(xcorr_matrix)
    # xcorr_matrix is number of pairs x time lag

    non_zeros_rows = np.argwhere(np.sum(xcorr_matrix>0,axis=1) > 0).ravel()
    pairs = np.array(pairs)[non_zeros_rows]
    xcorr_matrix = xcorr_matrix[non_zeros_rows,:].T

    # for output, we make xcorr_matrix into shape time lag x pairs, so it matches that of firing_matrix

    return xcorr_matrix,pairs,lag



def ParticipatedRippleIndex(nwb_copy_file_name,session_name,
                            ripple_times,electrode,unit,fragFlag):
    nwb_units = electrode_unit(nwb_copy_file_name,session_name,electrode)
    spike_times = nwb_units.loc[unit].spike_times
    ind = []
    for ripple_id in ripple_times.index:
        if fragFlag:
            intervals = ripple_times.loc[ripple_id].frag_intvl
        else:
            intervals = ripple_times.loc[ripple_id].cont_intvl
        for interval in intervals:
            spike_count=float(np.sum(
                np.logical_and(spike_times>=interval[0],spike_times<=interval[1])))
            if spike_count > 0:
                ind.append(ripple_id)
    return ind

def spikeLocation(nwb_copy_file_name, session_name, pos_name, electrode,unit):
    nwb_units = electrode_unit(nwb_copy_file_name,session_name,electrode)
    spike_time = nwb_units.loc[unit].spike_times

    pos1d = (IntervalLinearizedPosition() & {'nwb_file_name': nwb_copy_file_name,
                                                     'track_graph_name':'4 arm lumped 2023',
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    pos2d = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    # select track time only, remove Data before 1st trial and after last trial
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':session_name}).fetch1('choice_reward'))

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
    pos1d = removeDataBeforeTrial1(pos1d,trial_1_t,trial_last_t)
    pos2d = removeDataBeforeTrial1(pos2d,trial_1_t,trial_last_t)
    assert len(pos1d) == len(pos2d)
    spike_time = spike_time[np.logical_and(spike_time>=trial_1_t,
                                           spike_time<=trial_last_t)]
    total_spike_count = len(spike_time)

    # immobility only
    pos2d_spike_time_all = interpolate_to_new_time(pos2d,spike_time)
    pos1d_spike_time_all = interpolate_to_new_time(pos1d,spike_time)
    immobility_index = np.argwhere(pos2d_spike_time_all.head_speed < 4).ravel() # >4cm/s
    pos1d_spike_time = pos1d_spike_time_all.iloc[immobility_index]
    imm_spike_time = spike_time[immobility_index]

    # location to closet well
    linear_map,node_location=get_linearization_map()

    linear_position = np.array(pos1d_spike_time.linear_position)
    linear_segment = np.array(pos1d_spike_time.track_segment_id)
    for l in linear_segment:
        if l == 6:
            l = 'arm 1'
        elif l == 7:
            l = 'arm 2'
        elif l == 8:
            l = 'arm 3'
        elif l == 9:
            l = 'arm 4'
    linear_segment = [str(l) for l in linear_segment]
    for t in range(len(linear_position)):
        for k in node_location.keys():
            if np.abs(node_location[k] - linear_position[t]) < 15:
                linear_segment[t] = k
    return imm_spike_time,linear_segment




def RippleTime2FiringRate(nwb_units,ripple_times,accepted_units,fragFlag = True):
    """filters spikes according to their time, whether cont/frag ripple times"""

    if len(nwb_units)==0:
        return None
    firing_rate_by_ripple = {}
    firing_rate = {}
    for u in accepted_units:
        #### THIS IS HOW I GET SPIKE TIMES
        spike_times = nwb_units.loc[u].spike_times

        ind = []
        for ripple_id in ripple_times.index:
            if fragFlag:
                intervals = ripple_times.loc[ripple_id].frag_intvl
            else:
                intervals = ripple_times.loc[ripple_id].cont_intvl


            for interval in intervals:
                duration = interval[1]-interval[0] # in seconds
                spike_count=float(np.sum(
                    np.logical_and(spike_times>=interval[0],spike_times<=interval[1])))
                ind.append((spike_count/duration,spike_count,duration))

        firing_rate_by_ripple[u] = ind

    for u in accepted_units:
        total_spike_count = 0
        total_ripple_duration = 0
        for ripples in firing_rate_by_ripple[u]:
            total_spike_count += ripples[1]
            total_ripple_duration += ripples[2]
        firing_rate[u] = (total_spike_count/total_ripple_duration, total_spike_count, total_ripple_duration)

    return firing_rate_by_ripple, firing_rate

def findWaveForms(nwb_copy_file_name,epoch_name,eletrode,curation_id = 0):
    """returns waveform_extractor, which has memmap for waveforms"""

    #artifact_name = f'{nwb_copy_file_name}_{epoch_name}_{eletrode}_franklab_tetrode_hippocampus_ampl_1500_prop_075_1ms_artifact_removed_valid_times_track_time_only'
    artifact_name = f'{nwb_copy_file_name}_{epoch_name}_{eletrode}_franklab_tetrode_hippocampus_ampl_1500_prop_075_1ms_artifact_removed_valid_times'
    key = {'nwb_file_name':nwb_copy_file_name,
           'sort_interval_name':epoch_name,
           "sorter":"mountainsort4",
           'sort_group_id':eletrode,
           'artifact_removed_interval_list_name':artifact_name,
           "curation_id":curation_id
           }

    nwb_units = (CuratedSpikeSorting() & key).fetch_nwb()[0]["units"]
    waveform_extractor = Waveforms().load_waveforms(key)

    return nwb_units,waveform_extractor

def RippleTime2Index(spike_times,ripple_times,fragFlag = True):
    """filters spikes according to their time, whether cont/frag ripple times"""
    #### THIS IS HOW I GET SPIKE TIMES
    #spike_times = nwb_units.spike_times[0]
    ####
    ind = []

    for ripple_id in ripple_times.index:
        if fragFlag:
            intervals = ripple_times.loc[ripple_id].frag_intvl
        else:
            intervals = ripple_times.loc[ripple_id].cont_intvl

        for interval in intervals:
            ind_subset=np.argwhere(
                np.logical_and(spike_times>=interval[0],spike_times<=interval[1])).ravel()
            ind.append(ind_subset)
    return np.concatenate(ind)

def plot_firing_rate(StateScript,nwb_copy_file_name,e,u,
                     DELTA_T,trials,firing_rate_matrix,ripple_times,savefolder,savename):
    """
    trials is the y axis of the firing_rate_matrix,
        row trial t will have data from trial t's outer well poke to t+1's outerwell
    """
    fig, axes = plt.subplots(1,2,figsize = (10,10), sharey= True, gridspec_kw={"width_ratios": [1,10]})
    for t_ind in range(len(StateScript.index[:-2])):
        axes[1].plot(firing_rate_matrix[t_ind,:] - t_ind, color = 'k',linewidth = 0.5,)

    for r_ind in ripple_times.index:
        trial = ripple_times.loc[r_ind].trial_number
        animal_location = ripple_times.loc[r_ind].animal_location
        if animal_location == 'home':
            trial = trial - 1
        t0 = StateScript.loc[trial].timestamp_O
        t1 = StateScript.loc[trial+1].timestamp_O
        axis = np.concatenate((np.arange(t0,t1,DELTA_T),np.array([t1])))

        plot_y = - np.argwhere(trials == trial).ravel()[0] + 0.5 #row to plot for y


        cont_intvls = ripple_times.loc[r_ind].cont_intvl
        for cont_intvl in cont_intvls:
            ripple_t0_ind = np.argwhere(cont_intvl[0] <= axis).ravel()[0]
            ripple_t1_ind_ = np.argwhere(cont_intvl[1] <= axis).ravel()
            if len(ripple_t1_ind_) == 0:
                ripple_t1_ind = len(axis) - 1 #in this case, the ripple extends after outer poke
            else:
                ripple_t1_ind = ripple_t1_ind_[0]
            axes[1].plot([ripple_t0_ind,ripple_t1_ind],[plot_y,plot_y],linewidth = 5, color = 'C1', alpha = 0.3)


        frag_intvls = ripple_times.loc[r_ind].frag_intvl
        for frag_intvl in frag_intvls:
            ripple_t0_ind = np.argwhere(frag_intvl[0] <= axis).ravel()[0]
            ripple_t1_ind_ = np.argwhere(frag_intvl[1] <= axis).ravel()
            if len(ripple_t1_ind_) == 0:
                ripple_t1_ind = len(axis) - 1 #in this case, the ripple extends after outer poke
            else:
                ripple_t1_ind = ripple_t1_ind_[0]
            axes[1].plot([ripple_t0_ind,ripple_t1_ind],[plot_y,plot_y],linewidth = 5, color = 'C0', alpha = 0.3)

    # choice data
    seq = [3, 4, 2, 1]
    outerwell = np.array(StateScript.OuterWellIndex)
    legal_ind = np.argwhere(np.array(StateScript.rewardNum) > 0)
    for a_ind in range(4):
        a = seq[a_ind]
        trials_arm = np.argwhere(outerwell == a).ravel()
        for t in trials_arm:
            if np.isin(t,legal_ind):
                axes[0].scatter(a_ind, -t, color = 'C'+str(a))
            else:
                axes[0].scatter(a_ind, -t, color = [1,1,1], edgecolors = 'C'+str(a), linewidth = 2)
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xticklabels(seq)
    axes[1].set_yticks(-(StateScript.index[:-2] -1));
    axes[1].set_yticklabels(StateScript.index[:-2]);
    axes[0].set_ylabel('trial');

    xticks_locations = np.arange(0,firing_rate_matrix.shape[1],100)
    axes[1].set_xticks(xticks_locations)
    xtick_labels = [str(x) for x in xticks_locations]
    xtick_labels[0] = 'outer poke trial t'
    xtick_labels[1] = ''
    xtick_labels[-2] = ''
    xtick_labels[-1] = 'outer poke trial t+1'
    axes[1].set_xticklabels(xtick_labels);
    axes[1].set_xlabel('seconds');
    axes[1].set_title(nwb_copy_file_name+' electrode '+str(e)+' unit '+str(u));

    if len(savefolder)>0:
        plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200)