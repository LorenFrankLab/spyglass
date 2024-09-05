import numpy as np
from itertools import combinations
from scipy import stats
from spyglass.shijiegu.curation_manual import load_waveforms
from spyglass.shijiegu.placefield import place_field
from spikeinterface.postprocessing.correlograms import compute_correlograms
from spyglass.spikesorting.v0 import Curation
import spyglass.spikesorting.v0 as sgs
import matplotlib.pyplot as plt

def auto_burst_pairs_session(nwb_copy_file_name, session_name, pos_name,
                               curation_id = 1, sorter = "mountainsort4",
                               sort_group_ids = None):
    key = {"nwb_file_name": nwb_copy_file_name,
           "waveform_params_name":"default_clusterless",
           "sorter":sorter,
           "sort_interval_name":session_name,
           "curation_id":curation_id}

    if sort_group_ids is None:
        sort_group_ids = np.unique((Curation & key).fetch("sort_group_id"))

    fig_m = {}
    fig_place = {}
    wf_r_all = {}
    pf_r_all = {}
    ccgs_all = {}
    isi_violation_all = {}
    peak_amps_all = {}
    for e in sort_group_ids:
        print("working on electrode "+str(e))
        (units,
         wf_r, #waveform
         pf_r,
         peak_amps, peak_amps_t, placefields, spike_count, #place field
         isi_violation, ca, ccgs, bins) = auto_burst_pairs_electrode(
             nwb_copy_file_name,session_name,pos_name,
             e,curation_id)

        if len(units) > 0:
            fig_m[e], _ = plot_metrics(wf_r, pf_r, isi_violation, ca)
            fig_place[e], _ = plot_placefield(placefields, units)
            pf_r_all[e] = pf_r
            wf_r_all[e] = wf_r
            ccgs_all[e] = ccgs
            isi_violation_all[e] = isi_violation
            peak_amps_all[e] = (peak_amps,peak_amps_t)

    return fig_m, fig_place, wf_r_all, pf_r_all, isi_violation_all, peak_amps_all, ccgs_all, bins


def auto_burst_pairs_electrode(nwb_copy_file_name,session_name,pos_name,
                                sort_group_id,curation_id):
    # average waveform correlation r
    wf_r, peak_amps, peak_timestamps = get_mean_waveform_r(
        nwb_copy_file_name,session_name,
        sort_group_id,curation_id)
    units = peak_amps.keys()

    # pairwise place field correlation r
    pf_r, placefields, spike_count = get_place_field_r(
        nwb_copy_file_name,session_name, pos_name,
        sort_group_id, curation_id, units)

    # calculate isi violation suppose they are merged
    isi_violation = isi_all_pairs(peak_timestamps)

    # calulate cross-correlogram and asymmetry
    ca,ccgs,bins = get_xcorr_asymmetry(
        nwb_copy_file_name,session_name,
        sort_group_id = sort_group_id, curation_id = curation_id, units = units)

    return units, wf_r, pf_r, peak_amps, peak_timestamps, placefields, spike_count, isi_violation, ca,ccgs,bins

def plot_metrics(wf_r, pf_r, isi_violation, ca):
    """parameters are 4 metrics to be plotted against each other.

    Parameters
    ----------
    wf_r : dict
        waveform similarities
    pf_r : dict
        placefield similarities
    isi_violation : dict
        isi violation
    ca : dict
        spike cross correlogram assymmetry

    Returns
    -------
    figure for plotting later
    """

    fig,axes = plt.subplots(1,2,figsize=(12,5))

    color_ind = 0
    for p in pf_r.keys():
        (u1,u2) = p
        #isi_score = round(isi_violation[p], 2)

        axes[0].scatter(wf_r[p],pf_r[p],color = "C"+str(color_ind))
        axes[1].scatter(wf_r[p],ca[p],color = "C"+str(color_ind))


        #axes[0].text(wf_r[p],pf_r[p],"(" + str(u1) + "," + str(u2) + "), "+ str(isi_score),color = "C"+str(color_ind))
        axes[0].text(wf_r[p],pf_r[p],"(" + str(u1) + "," + str(u2) + ")", color = "C"+str(color_ind))
        axes[1].text(wf_r[p],ca[p],"(" + str(u1) + "," + str(u2) + ")", color = "C"+str(color_ind))

        color_ind += 1
    axes[0].set_xlabel("waveform similarity")
    axes[0].set_ylabel("placefield similarity")
    axes[1].set_xlabel("waveform similarity")
    axes[1].set_ylabel("cross-correlogram assymmetry")

    plt.close()
    return fig, axes

def plot_placefield(placefields, units):
    cells_to_plot = units

    col_num = int(np.ceil(len(cells_to_plot)/2))
    fig, axes = plt.subplots(2,col_num, figsize = (col_num * 3,5), squeeze = True)

    ind = 0
    for p in cells_to_plot:
        axes[np.unravel_index(ind, axes.shape)].imshow(placefields[p],
                                                    vmax = np.nanquantile(placefields[p],0.999),
                                                    vmin = np.nanquantile(placefields[p],0.4)
                                                    )
        axes[np.unravel_index(ind, axes.shape)].set_title(p)
        axes[np.unravel_index(ind, axes.shape)].set_axis_off()
        #axes[np.unravel_index(ind, axes.shape)].text(200,15,round(peak_frs[(e,u)],1))
        ind = ind + 1

    plt.close()
    return fig, axes

def get_mean_waveforms(waves,units):
    """ mean waveforms in a dict: each one is of spike number x 4
    output 2 is all channels concatenated

    Parameters
    ----------
    waves : waveform extractor, use load_waveforms() in curation_manual.py
    """
    #
    #get mean waveform
    waves_mean = {}
    for u in units:
        waveforms = waves.get_waveforms(u) #(sample_num,clip_size,channel_num) = waveforms.shape
        waves_mean[u] = np.mean(waveforms,axis = 0)

    waves_mean_1d = {}
    for u in units:
        waves_mean_1d[u] = np.reshape(waves_mean[u].T,(1,-1)).ravel()
    return waves_mean, waves_mean_1d

def get_mean_waveform_r(nwb_copy_file_name,session_name,
                                  sort_group_id,curation_id):
    waves, peak_amps, timestamps = load_waveforms(nwb_copy_file_name,session_name,
                                  sort_group_id,curation_id)
    units = peak_amps.keys()
    waves_mean, waves_mean_1d = get_mean_waveforms(waves,units)

    all_pairs = list(combinations(units,2))

    r_all = {}
    for p in all_pairs:
        (u1,u2) = p
        r_all[u1,u2] = stats.pearsonr(waves_mean_1d[u1],waves_mean_1d[u2]).statistic

    return r_all, peak_amps, timestamps

def get_place_field_r(nwb_copy_file_name,session_name, pos_name,
                      sort_group_id, curation_id, units):
    """Get correlation of cells' place field for all pairs of units

    Parameters
    ----------
    units : list
        unit that you wish to calculate correlations of
    """

    smoothed_placefield = {}
    spike_count = {}
    r_all = {}

    for u in units:
        (smoothed_placefield[u], peak_firing_rate,
        xbins, ybins, spike_count[u], total_spike_count) = place_field(nwb_copy_file_name,
                                                                    session_name, pos_name,
                                                                    sort_group_id, u,
                                                                    BINWIDTH = 2, sigma = 2,
                                                                    curation_id = curation_id)
    all_pairs = list(combinations(units,2))

    for p in all_pairs:
        (u1,u2) = p
        if (spike_count[u1] == 0 or spike_count[u2] == 0):
            continue
        placefield1 = choose_not_nan(smoothed_placefield[u1])
        placefield2 = choose_not_nan(smoothed_placefield[u2])
        r_all[u1,u2] = stats.pearsonr(placefield1,placefield2).statistic

    return r_all, smoothed_placefield, spike_count

def get_xcorr_asymmetry(nwb_copy_file_name,session_name,
                      sort_group_id, curation_id, units):
    """Get correlation of cells' place field for all pairs of units

    Parameters
    ----------
    units : list
        unit that you wish to calculate correlations of
    """

    key = {"nwb_file_name":nwb_copy_file_name,
       "waveform_params_name":"default_clusterless",
       "sorter":"mountainsort4",
       "sort_interval_name":session_name,
       "sort_group_id":sort_group_id,
       "curation_id": curation_id}
    sorting = Curation.get_curated_sorting(key)
    ccgs, bins = compute_correlograms(sorting,load_if_exists=False,
        window_ms = 100.0,
        bin_ms = 5.0,
        method = "numba",
    )
    ca = {} #correlogram asymmetry
    all_pairs = list(combinations(units,2))

    for p in all_pairs:
        (u1,u2) = p
        ca[p] = calculate_ca(bins[1:],ccgs[u1-1,u2-1,:])

    return ca,ccgs,bins


def choose_not_nan(data):
    return data[~np.isnan(data)]

def isi_all_pairs(peak_timestamps):
    """workhorse is isi_violations"""
    isi_violation = {}

    units = peak_timestamps.keys()
    all_pairs = list(combinations(units,2))

    for p in all_pairs:
        (u1,u2) = p
        spike_train = np.sort(np.concatenate(
            (peak_timestamps[u1],peak_timestamps[u2])))
        isi_violation[p] = isi_violations(spike_train)

    return isi_violation

def isi_violations(spike_train,isi_threshold_s = 1.5):
    """
    This function is a simplified version of
    spikeinterface.qualitymetrics.misc_metrics.isi_violations

    Parameters
    ----------
    spike_train : numpy array of spike time, in seconds
    isi_threshold_s is in units of ms
    """
    isis = np.diff(spike_train)
    num_spikes = len(spike_train)
    num_violations = np.sum(isis < (isi_threshold_s * 1e-3))

    return num_violations/num_spikes

def calculate_ca(bins,c):
    """
    calculate Correlogram Asymmetry (CA),
        defined as the contrast ratio of the area of the correlogram right and left of coincident activity (zero).
        http://www.psy.vanderbilt.edu/faculty/roeaw/edgeinduction/Fig-W6.htm
    """
    assert len(bins) == len(c)
    R = np.sum(c[bins > 0])
    L = np.sum(c[bins < 0])
    if (R + L) == 0:
        return 0
    return (R - L)/(R + L)

def make_new_labels(good_units,noise_units,mua_unit):
    labels = {}
    good_units = np.setdiff1d(good_units,noise_units)
    good_units = np.setdiff1d(good_units,mua_unit)
    for gu in good_units:
        labels[gu] = ['accept']
    for nu in noise_units:
        labels[nu] = ['reject']
    for mua in mua_unit:
        labels[mua] =['mua']
    return labels

def insert_CuratedSpikeSorting_burst(nwb_copy_file_name, session_name, sort_group_id,
                               merge_groups,
                               parent_curation_id,
                               good_units = [], noise_units = [], mua_unit = []):

    target_curation_id = parent_curation_id + 1

    key = {"nwb_file_name": nwb_copy_file_name,
        "sort_group_id": sort_group_id,
        "sorter": "mountainsort4",
        "sort_interval_name": session_name,
        "curation_id": parent_curation_id}

    if (len(good_units) + len(noise_units) + len(mua_unit)) > 0:
        # make new labels
        labels = make_new_labels(good_units,noise_units,mua_unit)
    else:
        # load old labels
        labels = (sgs.Curation & key).fetch1("curation_labels")
    metrics_sort_group_id = (sgs.Curation & key).fetch1("quality_metrics")

    # make new metrics table, drop bad units
    if len(noise_units) > 0:
        metrics_sort_group_id = (metrics_sort_group_id).drop(
            labels = [str(nu) for nu in noise_units],axis = 'index').to_dict()

    ssk = (sgs.SpikeSorting & key).fetch1("KEY")

    # double check Curation has no entry in the table before
    tmp_key = key.copy()
    tmp_key["curation_id"] = target_curation_id
    assert len(sgs.Curation & tmp_key) == 0, "target curation id entry already exists. Delete the target entry first."

    id = (sgs.Curation & ssk).fetch("curation_id")
    assert max(id) + 1 == target_curation_id, "Spyglass will not insert into target curation id entry"

    description = "burst merge by waveform, place field, asymmetric correlogram, and low ISI violation"

    # Up to here
    ck = sgs.Curation.insert_curation(sorting_key = ssk,
                                parent_curation_id = parent_curation_id,
                                labels = labels,
                                merge_groups = merge_groups,
                                metrics = metrics_sort_group_id,
                                description = description)

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
