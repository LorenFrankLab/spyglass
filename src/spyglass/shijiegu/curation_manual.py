import spyglass.spikesorting.v0 as sgs
import json
from spyglass.spikesorting.v0.spikesorting_curation import Waveforms
from spyglass.decoding.v0.clusterless import UnitMarks #not using data from the table but some functions associated with it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn import svm
from spyglass.common.common_interval import IntervalList

pair=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]  #tetrode channel pairs
color_palet = 1-np.array([[1,0.6,0],[0.7,0.6,0.4],[0.6,0.8,0.3],
                          [0,0.6,.3],[0,0,1],[0,0.6,1],[0,0.7,0.7],
                          [0.7,0,0.7],[0.7,0.4,1]]);
color_palet = color_palet[np.concatenate((np.arange(0,9,2),np.arange(1,9,2)))] # scramble slightly

def auto_noise_removal_session(nwb_copy_file_name, session_name,
                               curation_id = 0, sorter = "mountainsort4",
                               sort_group_ids = None):

    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":sorter,
       "sort_interval_name":session_name}

    session_duration = get_session_duration(nwb_copy_file_name,session_name)

    if sort_group_ids is None:
        sort_group_ids = np.unique((sgs.QualityMetrics & key).fetch("sort_group_id"))

    status = {}
    noiseUnits_spikeNum = {}
    noiseUnits_isi = {}
    metrics = {}
    fig_sp = {}
    fig_m = {}
    peak_amps = {}

    for sort_group_id in sort_group_ids:
        print("\n sort_group_id: ",sort_group_id)
        metrics_json = load_metric(nwb_copy_file_name,session_name,sort_group_id,curation_id)
        metrics_pd = pd.DataFrame(metrics_json)
        peak_amp, timestamps = load_peak_amp(nwb_copy_file_name,session_name,sort_group_id,curation_id)
        peak_amps[sort_group_id] = (peak_amp, timestamps)

        fig_sp[sort_group_id], _, color_map = plot_spray_window(peak_amp)
        fig_m[sort_group_id], _2 = plot_metric(metrics_pd,color_map,session_duration)
        (status[sort_group_id],
         noiseUnits_spikeNum[sort_group_id],
         noiseUnits_isi[sort_group_id]) = find_noise_units(metrics_pd)
        metrics[sort_group_id] = metrics_pd

    return fig_sp, fig_m, metrics, status, noiseUnits_spikeNum, noiseUnits_isi, peak_amps

def get_session_duration(nwb_copy_file_name,session_name):
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter": "mountainsort4",
       "sort_interval_name": session_name}
    sort_interval_name = (sgs.SpikeSorting() & key).fetch("artifact_removed_interval_list_name")[0]

    session_valid_t = (IntervalList() & {"interval_list_name":
        sort_interval_name}).fetch1("valid_times").ravel()
    session_duration = session_valid_t[-1] - session_valid_t[0]
    return session_duration

def load_metric(nwb_copy_file_name,session_name,sort_group_id,curation_id):
    '''
    load metric
    '''
    ss_key = {"nwb_file_name": nwb_copy_file_name,
              "sorter":"mountainsort4",
              "sort_interval_name":session_name,
              "sort_group_id":sort_group_id,
              "curation_id":curation_id}

    metrics_json_path=(sgs.QualityMetrics & ss_key).fetch1("quality_metrics_path")
    # Opening JSON file
    with open(metrics_json_path) as json_file:
        metrics_json = json.load(json_file)

    #print("metrics are: ",metrics_json.keys())
    return metrics_json #just a dictionary

def load_waveforms(nwb_copy_file_name,session_name,
                  sort_group_id,curation_id):
    '''
    load waveforms and peak amp across 4 tetrode channels into a dict called waves
    each key of the dict is a unit_id
    '''
    ss_key = {"nwb_file_name": nwb_copy_file_name,
              "sorter":"mountainsort4",
              "sort_interval_name":session_name,
              "sort_group_id":sort_group_id,
              "curation_id":curation_id}

    we = Waveforms.load_waveforms(Waveforms,ss_key) #extractor

    curatedss_entry = (sgs.CuratedSpikeSorting() & ss_key).fetch_nwb()[0]
    if "units" in curatedss_entry:
        nwb_units = curatedss_entry["units"]
    else:
        return we, {}, {}

    peak={}
    timestamps={}
    for unit_id in nwb_units.index:
        wave=UnitMarks._get_peak_amplitude(
            waveform=we.get_waveforms(unit_id),
            peak_sign="neg",
            estimate_peak_time=True)
        timestamp = np.asarray(nwb_units["spike_times"][unit_id])
        sorted_timestamp_ind = np.argsort(timestamp)
        marks = wave[sorted_timestamp_ind]
        timestamp = timestamp[sorted_timestamp_ind]

        peak[unit_id]= marks
        timestamps[unit_id] = timestamp
    return we, peak, timestamps

def load_peak_amp(nwb_copy_file_name,session_name,
                  sort_group_id,curation_id):
    '''
    load peak amplitude across 4 tetrode channels into a dict called waves
    each key of the dict is a unit_id
    '''
    ss_key = {"nwb_file_name": nwb_copy_file_name,
              "sorter":"mountainsort4",
              "sort_interval_name":session_name,
              "sort_group_id":sort_group_id,
              "curation_id":curation_id}

    we = Waveforms.load_waveforms(Waveforms,ss_key) #extractor

    nwb_units = (sgs.CuratedSpikeSorting() & ss_key).fetch_nwb()[0]["units"]

    waves={}
    timestamps={}
    for unit_id in nwb_units.index:
        wave=UnitMarks._get_peak_amplitude(
            waveform=we.get_waveforms(unit_id),
            peak_sign="neg",
            estimate_peak_time=True)
        timestamp = np.asarray(nwb_units["spike_times"][unit_id])
        sorted_timestamp_ind = np.argsort(timestamp)
        marks = wave[sorted_timestamp_ind]
        timestamp = timestamp[sorted_timestamp_ind]

        waves[unit_id]= marks
        timestamps[unit_id] = timestamp
    return waves, timestamps

def plot_spray_window(waves):
    color_map = {}
    fig,axes=plt.subplots(1,6,figsize=(24,4),sharex=True,sharey=True)

    ## get amplitude
    color_ind=0
    for u in waves.keys():#noiseUnits:#nwb_units.index:1,3,5,12,6,
        color_map[u] = color_palet[color_ind]

        for p in range(6):
            e1,e2=pair[p]
            # plot every other spike
            _ = axes[p].scatter(-waves[u][::2,e1],-waves[u][::2,e2], color=color_palet[color_ind], s=4, alpha = 0.2)

            # plot unit name
            centroid_x = np.mean(-waves[u][:,e1])
            centroid_y = np.mean(-waves[u][:,e2])
            txt = axes[p].text(centroid_x,centroid_y,str(u))
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w',alpha = 0.8)])

        color_ind = (color_ind+1) % color_palet.shape[0]

    for p in range(6):
        axes[p].set_xlabel("uV")
        axes[p].set_ylabel("uV")
    plt.close()

    return fig, axes, color_map

NN_NOISE_OVERLAP = 0.1
ISI_VIOLATION = 0.004
NN_ISOLATION = 0.9
FIRING_RATE = 7 #Hz

def plot_metric(metric_pd, color_map = None, duration = None):
    """duration is session duration in seconds"""

    fig, axes=plt.subplots(1,4,figsize=(24,4),sharex=False,sharey=False)

    color_ind = 0
    for i in metric_pd.index:
        unit_name = i
        if color_map is not None:
            color = color_map[int(i)]
        else:
            color = color_palet[color_ind]

        axes[0].axhspan(NN_NOISE_OVERLAP,axes[0].get_ylim()[1],color = 'salmon',alpha = 0.01)
        axes[0].scatter(metric_pd.loc[unit_name]['num_spikes'],metric_pd.loc[unit_name]['nn_noise_overlap'],color = color)
        txt = axes[0].text(metric_pd.loc[unit_name]['num_spikes'],metric_pd.loc[unit_name]['nn_noise_overlap'],unit_name)
        if duration is not None:
            NUM = FIRING_RATE * duration
            axes[0].axvspan(NUM,axes[0].get_xlim()[1],color = 'salmon',alpha = 0.01)


        axes[1].axhspan(0,NN_ISOLATION,color = 'salmon',alpha = 0.01)
        axes[1].scatter(metric_pd.loc[unit_name]['num_spikes'],metric_pd.loc[unit_name]['nn_isolation'], color = color)
        txt = axes[1].text(metric_pd.loc[unit_name]['num_spikes'],metric_pd.loc[unit_name]['nn_isolation'],unit_name)
        if duration is not None:
            NUM = FIRING_RATE * duration
            axes[1].axvspan(NUM,axes[1].get_xlim()[1],color = 'salmon',alpha = 0.01)


        axes[2].axhspan(ISI_VIOLATION,axes[2].get_ylim()[1],color = 'salmon',alpha = 0.01)
        axes[2].axvspan(0,NN_ISOLATION,color = 'salmon',alpha = 0.01)
        axes[2].scatter(metric_pd.loc[unit_name]['nn_isolation'],metric_pd.loc[unit_name]['isi_violation'], color = color)
        txt = axes[2].text(metric_pd.loc[unit_name]['nn_isolation'],metric_pd.loc[unit_name]['isi_violation'],unit_name)

        axes[3].axhspan(ISI_VIOLATION,axes[3].get_ylim()[1],color = 'salmon',alpha = 0.01)
        axes[3].scatter(metric_pd.loc[unit_name]['snr'],metric_pd.loc[unit_name]['isi_violation'], color = color)
        txt = axes[3].text(metric_pd.loc[unit_name]['snr'],metric_pd.loc[unit_name]['isi_violation'],unit_name)

        color_ind = color_ind + 1


    axes[0].set_xlabel('num_spikes')
    axes[0].set_ylabel('nn_noise_overlap')

    axes[1].set_xlabel('num_spikes')
    axes[1].set_ylabel('nn_isolation')

    axes[2].set_xlabel('nn_isolation')
    axes[2].set_ylabel('isi_violation')

    axes[3].set_xlabel('snr')
    axes[3].set_ylabel('isi_violation')
    plt.close()
    return fig, axes

#SECOND_PER_PANEL = 4000
def plot_peak_amp_overtime(peak_amps,sort_group_id,units,overlap = False):

    peak_amp = peak_amps[sort_group_id]
    peak_amp_V = peak_amp[0]
    peak_amp_t = peak_amp[1]

    u = units[0]
    start_ind = 0 #starting from unit index 0
    if overlap:
        fig, axes = plot_peak_amp_overtime_unit(peak_amp_V[u],peak_amp_t[u],row_duration = 100, show_plot = True)
        start_ind = 1
    else:
        fig = {}
        axes = {}

    for u_ind in range(start_ind,len(units)):
        u = units[u_ind]
        if overlap:
            fig, axes = plot_peak_amp_overtime_unit(peak_amp_V[u],peak_amp_t[u], fig, axes, row_duration = 100, show_plot = True)
        else:
            fig[u], axes[u] = plot_peak_amp_overtime_unit(peak_amp_V[u],peak_amp_t[u],show_plot = False)
    return fig, axes

def plot_peak_amp_overtime_unit(V, t, fig = None, axes = None, row_duration = 600,show_plot = False):
    # row_duration is how many seconds in each panel

    max_channel = np.argmax(-np.mean(V,0))
    time_since_session = t - t[0]
    row_num = int(np.ceil(time_since_session[-1]/(row_duration)))

    if axes is None:
        fig, axes = plt.subplots(row_num,1,
                             figsize=(20,2*row_num),sharex=True,sharey=True,squeeze = False)
    else:
        row_num = axes.shape[0]

    for ind in range(row_num):
        (t0, t1) = (ind * row_duration, (ind + 1) * row_duration)
        sub_ind = np.argwhere(np.logical_and(time_since_session >= t0,
                                            time_since_session <= t1)).ravel()
        axes[ind,0].scatter(time_since_session[sub_ind]-t0,
                            V[sub_ind,max_channel])

    if not show_plot:
        plt.close()
    return fig, axes


def find_noise_units(metric_pd):
    success = 0
    success_spike_num, noiseUnits_spike_num = find_noise_units_by_spike_num(metric_pd)
    success_isi, noiseUnits_isi = find_noise_units_by_ISI(metric_pd)
    if (success_spike_num and success_isi):
        if np.all(noiseUnits_spike_num == noiseUnits_isi):
            success = 1
            print("Auto noise exclusion success!")
    return success, noiseUnits_spike_num, noiseUnits_isi


def find_noise_units_by_spike_num(metric_pd):
    """
    Train a Support Vector Machine to seperate units's num_spikes by
    nn_isolation, isi_violation, snr
    """
    success = 0

    snr = np.array(metric_pd.snr) #of shape number of units
    nn_isolation = np.array(metric_pd.nn_isolation)
    isi_violation = np.array(metric_pd.isi_violation)
    num_spikes = np.array(metric_pd.num_spikes)

    threshold = 5000

    # adaptive thresholding
    X = np.hstack((nn_isolation.reshape((-1,1)),isi_violation.reshape((-1,1)),snr.reshape((-1,1))))
    for m in np.arange(100):
        threshold = threshold + 1000
        y = (num_spikes >= threshold).astype('int')
        if len(np.unique(y)) < 2:
            # less than 2 classes
            continue
        clf = svm.LinearSVC(C=100)
        clf.fit(X, y)
        yPred = clf.predict(X)
        if np.sum(np.abs(y - yPred)) == 0:
            print("spike_num_threshold",threshold)
            success = 1
            noiseUnits = np.argwhere(y).ravel()+1
            return success,noiseUnits

    return success,[]

def find_noise_units_by_ISI(metric_pd):
    """
    Train a Support Vector Machine to seperate units's isi_violation by
    nn_isolation, num_spikes, snr
    """

    success = 0

    snr = np.array(metric_pd.snr) #of shape number of units
    nn_isolation = np.array(metric_pd.nn_isolation)
    isi_violation = np.array(metric_pd.isi_violation)
    num_spikes = np.array(metric_pd.num_spikes)

    # pick out num_spikes for units with SNR >= 15
    #   and use these unit's average isi_violation score as a starting threshold for SVM

    # largeAmp_ind = np.argwhere(snr >= 15).ravel()
    # if len(largeAmp_ind) == 0:
    #    largeAmp_ind = np.argwhere(snr >= 8).ravel()
    # if len(largeAmp_ind) == 0:
    #    return success,[]

    #baseThreshold = np.mean(isi_violation[largeAmp_ind]) + 0.0001 #make sure it is not 0
    threshold = 0.0001

    # adaptive thresholding
    #multipliers = np.ones(10)
    X = np.hstack((nn_isolation.reshape((-1,1)),num_spikes.reshape((-1,1)),snr.reshape((-1,1))))
    for m in np.arange(100):
        threshold = threshold + 0.0005

        y = (isi_violation >= threshold).astype('int')
        if len(np.unique(y)) < 2:
            # less than 2 classes
            continue
        clf = svm.LinearSVC(C=100, dual=False)
        clf.fit(X, y)
        yPred = clf.predict(X)
        if np.sum(np.abs(y - yPred)) == 0:
            print("ISI threshold",threshold)
            success = 1
            noiseUnits = np.argwhere(y).ravel()+1
            return success,noiseUnits
    return success, []

def insert_CuratedSpikeSorting(nwb_copy_file_name, session_name, sort_group_id,
                               metrics_sort_group_id,
                               parent_curation_id, good_units, noise_units, mua_unit):

    target_curation_id = parent_curation_id + 1

    # make new labels
    labels = {}
    good_units = np.setdiff1d(good_units,noise_units)
    good_units = np.setdiff1d(good_units,mua_unit)
    for gu in good_units:
        labels[gu] = ['accept']
    for nu in noise_units:
        labels[nu] = ['reject']
    for mua in mua_unit:
        labels[mua] =['mua']

    # make new metrics table, drop bad units
    updated_metrics = (metrics_sort_group_id).drop(
    labels = [str(nu) for nu in noise_units],axis = 'index').to_dict()

    key = {"nwb_file_name": nwb_copy_file_name,
        "sort_group_id": sort_group_id,
        "sorter": "mountainsort4",
        "sort_interval_name": session_name}

    ssk = (sgs.SpikeSorting & key).fetch1("KEY")

    # double check Curation has no entry in the table before
    tmp_key = key.copy()
    tmp_key["curation_id"] = target_curation_id
    assert len(sgs.Curation & tmp_key) == 0, "target curation id entry already exists. Delete the target entry first."

    id = (sgs.Curation & ssk).fetch("curation_id")
    assert max(id) + 1 == target_curation_id, "Spyglass will not insert into target curation id entry"

    description = "removal of noise units ONLY, by SVM and manual inspection"

    ck = sgs.Curation.insert_curation(sorting_key = ssk,
                                parent_curation_id = parent_curation_id,
                                labels=labels,
                                merge_groups=None,
                                metrics=updated_metrics,
                                description=description)

    sgs.CuratedSpikeSortingSelection.insert1(ck)
    sgs.CuratedSpikeSorting().populate(ck)

    ##### This section filling the waveform table is needed for further curation
    key = {"nwb_file_name":nwb_copy_file_name,
       "sorter":"mountainsort4",
       "sort_interval_name":session_name,
       "sort_group_id":sort_group_id,
       "curation_id":target_curation_id}

    key_full = (sgs.Curation() & key).fetch1("KEY")
    key_full["waveform_params_name"] = "default_clusterless"
    sgs.WaveformSelection.insert1(key_full)
    sgs.Waveforms.populate(key_full)

    return ck

def show_plot(f, title):
    # given a plot f, plot it inline.
    managed_fig, _ = plt.subplots(1,4,figsize=(24,4),sharex=False,sharey=False)
    canvas_manager = managed_fig.canvas.manager
    canvas_manager.canvas.figure = f
    managed_fig.set_canvas(canvas_manager.canvas)
    plt.suptitle(title,fontsize = 20)

def show_waveform(extractor,units):
    if len(units) == 0:
        return
    fig, axes = plt.subplots(len(units),4,
                         figsize = (10,3*len(units)),sharex = 1,sharey = 1,squeeze = False)

    row_ind = 0
    for row_ind in range(len(units)):
        print(row_ind)
        u = units[row_ind]
        waveforms = extractor.get_waveforms(u)

        (sample_num,clip_size,channel_num) = waveforms.shape

        sub_ind = np.arange(0,sample_num,int(sample_num/10))#np.random.choice(sample_num, 50) #plot ~ 10 spikes

        for s in sub_ind: #plot every 1000 spikes
            for ch in range(channel_num):
                axes[row_ind,ch].plot(waveforms[s,:,ch].squeeze(),'C1',alpha = 0.3)
                axes[row_ind,ch].set_ylim([-500,500])

        axes[row_ind,1].set_title('     unit '+str(u))

def end_of_session_check(nwb_copy_file_name,session_name,parent_curation_id):
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "sort_interval_name":session_name,
       "curation_id": parent_curation_id}
    if parent_curation_id == 0:
        sort_group_ids = np.unique((sgs.QualityMetrics & key).fetch("sort_group_id"))
    else:
        sort_group_ids_ = np.unique((sgs.CuratedSpikeSorting & key).fetch("sort_group_id"))

        # remove sort groups with all rejected units
        sort_group_ids = []
        for sort_group_id in sort_group_ids_:
            key["sort_group_id"] = sort_group_id
            curation_labels = (sgs.Curation() & key).fetch1("curation_labels")
            not_empty = False
            for k in curation_labels.keys():
                if 'accept' in curation_labels[k]:
                    sort_group_ids.append(sort_group_id)
                    break

        sort_group_ids = np.array(sort_group_ids)

    key.pop("sort_group_id")
    key["curation_id"] = parent_curation_id + 1
    sort_group_ids_processed = np.unique((sgs.CuratedSpikeSorting() & key).fetch("sort_group_id"))

    if len(sort_group_ids_processed) == len(sort_group_ids):
        print(session_name + " all done!")
        return 1
    else:
        missing_electrode = np.setdiff1d(sort_group_ids,
                                         sort_group_ids_processed)
        print(f"{session_name} missing sort group {missing_electrode}")
        return 0

def end_of_day_check(nwb_copy_file_name,parent_curation_id):
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "curation_id": parent_curation_id}
    if parent_curation_id == 0:
        intervals = np.unique((sgs.QualityMetrics & key).fetch("sort_interval_name"))
    else:
        intervals = np.unique((sgs.CuratedSpikeSorting & key).fetch("sort_interval_name"))

    key["curation_id"] = parent_curation_id + 1
    intervals_processed_tmp = np.unique((sgs.CuratedSpikeSorting() & key).fetch("sort_interval_name"))

    intervals_processed = []
    for session_name in intervals_processed_tmp:
        if end_of_session_check(nwb_copy_file_name,session_name,parent_curation_id):
            intervals_processed.append(session_name)
    intervals_processed = np.array(intervals_processed)

    if len(intervals_processed) == len(intervals):
        print(nwb_copy_file_name + " all done!")
        return 1
    else:
        missing_electrode = np.setdiff1d(intervals,
                                         intervals_processed)
        print(f"missing session {missing_electrode}")
        return 0
