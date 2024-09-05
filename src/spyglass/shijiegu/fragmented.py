import numpy as np
from spyglass.shijiegu.singleUnit import (electrode_unit,RippleTime2FiringRate,
                                          find_spikes,xcorr)
from scipy import signal

def fragmented_triggered(ripple_times, DELTA_T, RESOLUTION, cont = True):
    """return fragmented interval onset triggered continuous or frag interval onset and duration"""
    intvl_cont_all = concatenate_by_type(ripple_times, cont = cont)

    cont_onset_all = [] #this is the list of onset
    cont_all = [] #this is the list that will keep all cont time

    for i in ripple_times.index:
        intvls = ripple_times.loc[i].frag_intvl
        for intvl in intvls:
            if (intvl[1] - intvl[0]) < 0.02:
                # skip these short fragmented intervals
                continue
            (t0,t1) = (intvl[0] - DELTA_T, intvl[0] + DELTA_T)

            # find all cont ripple intervals that are within t0-t1
            ripple_ind_start = np.argwhere(intvl_cont_all[:,1] >= t0).ravel()[0]
            ripple_ind_end = np.argwhere(intvl_cont_all[:,0] <= t1).ravel()[-1]

            intvl_cont_ = intvl_cont_all[np.arange(ripple_ind_start,ripple_ind_end+1)]
            for intvl_cont in intvl_cont_:
                if intvl_cont[0] == intvl[0]: #this affects frag triggered frag only
                    continue
                cont_onset_all.append(intvl_cont[0]-intvl[0]) #center around the fragmented
                cont_all.append(np.arange(intvl_cont[0],intvl_cont[1],RESOLUTION) - intvl[0]) #center around the fragmented
    return cont_onset_all, cont_all

def fragmented_triggered_replay(ripple_times, DELTA_T, RESOLUTION, random = False):
    # if random = True, it will randomly shift each fragment interval left and right in time

    replay_cont = []
    intvl_cont = []

    for i in ripple_times.index:
        intvls = ripple_times.loc[i].cont_intvl
        replays = ripple_times.loc[i].cont_intvl_replay
        for ind in range(len(intvls)):
            if len(replays[ind])>0:
                replay_cont.append(replays[ind])
                intvl_cont.append(intvls[ind])

    replay_cont_all = replay_cont #np.array(replay_cont)
    intvl_cont_all = np.array(intvl_cont)

    replay_all = []
    for i in ripple_times.index:
        intvls = ripple_times.loc[i].frag_intvl
        for intvl in intvls:

            skip = False
            if (intvl[1] - intvl[0]) < 0.02:
                continue
            if random:
                intvl = intvl+np.random.uniform(0,0.6)
            (t0,t1) = (intvl[0] - DELTA_T, intvl[0] + DELTA_T)

            # find all cont ripple intervals that are within t0-t1
            # find all cont ripple intervals that are within t0-t1
            ripple_ind_start = np.argwhere(intvl_cont_all[:,1] >= t0).ravel()
            if len(ripple_ind_start)>0:
                ripple_ind_start = ripple_ind_start[0]
            else:
                continue
            ripple_ind_end = np.argwhere(intvl_cont_all[:,0] <= t1).ravel()
            if len(ripple_ind_end)>0:
                ripple_ind_end = ripple_ind_end[-1]
            else:
                continue

            ripple_ind_list = np.arange(ripple_ind_start,ripple_ind_end+1)
            intvl_cont_ = intvl_cont_all[ripple_ind_list]
            replay_cont_ = [replay_cont_all[r_ind] for r_ind in ripple_ind_list]
            for replay in replay_cont_:
                if len(replay) > 1:
                    skip = True

            if skip:
                continue

            axis = np.arange(t0,t1+RESOLUTION,RESOLUTION)
            replay_ = np.zeros_like(axis) + np.nan
            for intvl_cont_ind in range(len(ripple_ind_list)):
                intvl_cont = intvl_cont_[intvl_cont_ind]
                replay_cont = replay_cont_[intvl_cont_ind]
                ind1 = np.argwhere(axis >= intvl_cont[0]).ravel()[0]
                ind2 = np.argwhere(axis <= intvl_cont[1]).ravel()[-1]
                replay_[ind1:ind2] = np.zeros(ind2-ind1)+replay_cont
            replay_all.append(replay_)

    replay_all = np.array(replay_all)
    return replay_all

def concatenate_by_type(ripple_times, cont = True):
    """concatenate all cont (if cont = True) intervals in one numpy array"""
    intvl_all = []
    for i in ripple_times.index:
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl
        if len(intvls)>0:
            intvl_all.append(intvls)
    intvl_all = np.concatenate(intvl_all)
    return intvl_all

def get_nwb_units(nwb_copy_file_name,session_name,sort_group_ids_with_good_cell):
    """kept for legacy, but use the function session_unit() in singleUnit instead"""
    nwb_units_all = {}
    for sort_group_id in sort_group_ids_with_good_cell:
        nwb_units = electrode_unit(nwb_copy_file_name,session_name,sort_group_id)
        nwb_units_all[sort_group_id] = nwb_units
    return nwb_units_all

def find_spike_count_ratio(nwb_units_all,ripple_times, use_rate= False):
    """
    find the firing ratio between frag and cont, define RATIO_THRESHOLD using this histogram
    each cell is one data point
    """
    sort_group_ids_with_good_cell = list(nwb_units_all.keys())
    firing_rate_F = {}
    firing_rate_C = {}

    for sort_group_id in sort_group_ids_with_good_cell:
        nwb_units = nwb_units_all[sort_group_id]

        (_,firing_rate_F[sort_group_id]) = RippleTime2FiringRate(nwb_units,ripple_times,nwb_units.index,fragFlag = True)
        (_,firing_rate_C[sort_group_id]) = RippleTime2FiringRate(nwb_units,ripple_times,nwb_units.index,fragFlag = False)

    # find the firing ratio between frag and cont, define RATIO_THRESHOLD using this histogram
    ratio = []
    for e in sort_group_ids_with_good_cell:
        for u in list(firing_rate_F[e].keys()):
            (rate_F, count_F, _) = firing_rate_F[e][u]
            (rate_C, count_C, _) = firing_rate_C[e][u]
            if use_rate:
                if rate_C > 0:
                    ratio.append(rate_F/rate_C)
                elif rate_C == 0 and rate_F > 0:
                    ratio.append(100)
                else:
                    ratio.append(0)
            else:
                if count_C > 0:
                    ratio.append(count_F/count_C)
                elif count_C == 0 and count_F > 0:
                    ratio.append(100)
                else:
                    ratio.append(0)
    return ratio, firing_rate_F, firing_rate_C


def sum_place_field_by_weight(normalized_placefields,activate_cell_cont,weights):
    cell_num = 0
    valid_index = []
    for cell_ind in range(len(activate_cell_cont)):
        cell = activate_cell_cont[cell_ind]
        tmp = normalized_placefields[cell]
        if np.sum(~np.isnan(tmp)) != 0:
            valid_index.append(cell_ind)
            cell_num = cell_num + 1
        else:
            weights[cell_ind] = 0

    weights = weights/np.sum(weights)
    placefields_intvl_sum = np.sum(
        np.array(
            [normalized_placefields[activate_cell_cont[ind]] * weights[ind] for ind in valid_index]
        ),
        axis = 0)
    return placefields_intvl_sum, cell_num

def classify_cells(firing_rate_F, firing_rate_C):
    # into fragmented or continuous
    RATIO_FRAG = 1.5
    RATIO_CONT = 0.5
    cells_cont = []
    cells_frag = []

    sort_group_ids_with_good_cell = list(firing_rate_F.keys())
    for e in sort_group_ids_with_good_cell:
        for u in list(firing_rate_F[e].keys()):

            (rate_F, count_F, _) = firing_rate_F[e][u]
            print('electrode '+ str(e) + ' unit ' + str(u))
            print('during fragmented replay: ',rate_F, count_F)

            (rate_C, count_C, _) = firing_rate_C[e][u]
            print('during cont replay: ',rate_C, count_C)
            if count_C == 0 and count_F > 2:
                cells_frag.append((e,u))
                continue

            if count_C > 0 and count_F/count_C > RATIO_FRAG:
                cells_frag.append((e,u))
            elif count_C > 0 and count_F/count_C < RATIO_CONT:
                cells_cont.append((e,u))
    return cells_frag, cells_cont

def find_max_length(ripple_times,cont,DELTA_T):
    """helper function for getting cross-correlation"""
    length_max = 0
    for i in ripple_times.index:
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl

        for intvl in intvls:
            axis1 = np.arange(intvl[0], intvl[1] + DELTA_T, DELTA_T)

            if len(axis1) > length_max:
                length_max = len(axis1)
    return length_max-1

def find_firing_cross_correlation(ripple_times, electrodes_units, cell_list, cont = True, DELTA_T = 0.04):
    length_max = find_max_length(ripple_times,cont,DELTA_T)
    lag = signal.correlation_lags(length_max, length_max, mode='full')

    length_max_plot = np.ceil(0.2/DELTA_T)
    lag_plot = signal.correlation_lags(length_max_plot, length_max_plot, mode='full')

    firing_correlation_all = []
    pairs_all = []
    ripple_index_all = []

    for i in ripple_times.index:

        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl

        for intvl in intvls:
            axis1 = np.arange(intvl[0], intvl[1]+DELTA_T, DELTA_T)
            firing_matrix = find_spikes(electrodes_units,cell_list,axis1)
            xcorr_matrix_,pairs,lag_ = xcorr(firing_matrix)

            # match shape (the firing matrix could be shorter)
            ind1 = int(np.argwhere(lag == lag_[0]).ravel())
            ind2 = int(np.argwhere(lag == lag_[-1]).ravel())
            xcorr_matrix = np.zeros((len(lag),xcorr_matrix_.shape[1]))
            xcorr_matrix[ind1:ind2+1,:] = xcorr_matrix_
            firing_correlation_all.append(xcorr_matrix)
            pairs_all.append(pairs)
            ripple_index_all.append(np.zeros(xcorr_matrix_.shape[1]) + i)

    firing_correlation_all = np.concatenate(firing_correlation_all,axis=1)
    if length_max_plot < length_max:
        ind1 = int(np.argwhere(lag == lag_plot[0]).ravel())
        ind2 = int(np.argwhere(lag == lag_plot[-1]).ravel())
    firing_correlation_all = firing_correlation_all[ind1:ind2+1,:]

    return firing_correlation_all,np.concatenate(pairs_all),np.concatenate(ripple_index_all),lag[ind1:ind2+1]

def permute_frag_cont(ripple_times):
    # first find marginal distribution of inverval identity
    cont_all = concatenate_by_type(ripple_times, cont = True)
    frag_all = concatenate_by_type(ripple_times, cont = False)
    p_cont = len(cont_all)/(len(cont_all) + len(frag_all))

    # empty ripple_times_random
    ripple_times_random = ripple_times.copy()
    ripple_times_random.cont_intvl = [[] for i in range(len(ripple_times))]
    ripple_times_random.frag_intvl = [[] for i in range(len(ripple_times))]

    for i in ripple_times.index:
        # put all intervals in one!
        intvls = [c for c in ripple_times.loc[i].cont_intvl]
        for c in ripple_times.loc[i].frag_intvl:
            intvls.append(c)

        # permute identity
        for intvl in intvls:
            if np.random.binomial(1, p_cont):
                ripple_times_random.loc[i].cont_intvl.append(intvl)
            else:
                ripple_times_random.loc[i].frag_intvl.append(intvl)
    return ripple_times_random
