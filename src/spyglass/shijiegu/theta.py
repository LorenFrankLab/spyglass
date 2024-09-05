import os
import numpy as np
import pandas as pd
import xarray as xr
from spyglass.shijiegu.Analysis_SGU import TrialChoice, ThetaIntervals
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.load import load_theta_maze, load_position
import scipy.signal as ss
import scipy.stats as stats
from spyglass.shijiegu.Analysis_SGU import get_linearization_map,segment_to_linear_range
from spyglass.shijiegu.fragmented_graph import cell_name_to_ind,find_active_cells_intvl,make_graph, plot_graph
from scipy.signal import find_peaks
from spyglass.shijiegu.ripple_add_replay import add_trial,add_location,add_head_orientation
import matplotlib.pyplot as plt


linear_map, welllocations = get_linearization_map()

def theta_parser_master(nwb_copy_file_name, pos_name, session_name, nwb_units_all,cell_list):
    # to do: put nwb_units and cell_list in a table


    """Section 1: parse theta cycle by cycle"""
    theta_df, pos1d, pos2d = return_skaggs_theta(nwb_copy_file_name,pos_name,session_name,
                                             nwb_units_all,cell_list)
    pos1d = interpolate_to_new_time(pos1d,theta_df.index)
    pos2d = interpolate_to_new_time(pos2d,theta_df.index)

    # mobility only
    mobility_index = np.argwhere(pos2d.head_speed > 4).ravel() # >4cm/s
    pos1d_mobility = pos1d.iloc[mobility_index]
    pos2d_mobility = pos2d.iloc[mobility_index]

    # split into outbound and inbound
    # outbound = pos2d_mobility.head_orientation > 0
    # inbound = pos2d_mobility.head_orientation < 0

    # pos1d_mobility_out = pos1d_mobility[outbound]
    # pos2d_mobility_out = pos2d_mobility[outbound]
    # pos1d_mobility_in = pos1d_mobility[inbound]
    # pos2d_mobility_in = pos2d_mobility[inbound]

    theta_peaks,_ = find_peaks(np.array(np.cos(theta_df.phase0)),
                        distance = 60) # 1000 samples/s * 0.125/2 s

    theta_intervals = []
    print("parsing thetas cycle by cycle.")
    for i in range(len(theta_peaks)-1):
        ind0 = theta_peaks[i]
        ind1 = theta_peaks[i+1]
        theta_time_i = theta_df.iloc[ind0:ind1].phase0.index
        start_end = [theta_time_i[0],theta_time_i[-1]]
        if np.isin(theta_time_i[0],pos1d_mobility.index): # make sure animal is moving
            theta_intervals.append(start_end)
    print("Done parsing thetas cycle by cycle.")

    """Section 2: add meta data"""
    theta_intervals_df = pd.DataFrame({'theta_interval':theta_intervals})
    # add active cells
    theta_intervals_df.insert(1,'active_cells_ind',[[] for i in range(len(theta_intervals_df))]) #hold ripple times
    theta_intervals_df.insert(2,'active_cells',[[] for i in range(len(theta_intervals_df))]) #hold ripple times

    for i in theta_intervals_df.index:
        active_group = find_active_cells_intvl(theta_intervals_df.loc[i].theta_interval,
                                            nwb_units_all, cell_list)
        if len(active_group) > 0:
            theta_intervals_df.at[i,'active_cells_ind'] = list(active_group)
            theta_intervals_df.at[i,'active_cells'] = [cell_list[p] for p in active_group]

    # add trial number, animal location, and head orientation
    # the start time and end time is added so that we can directly use functions we wrote for ripple
    # load linear position and head orientation
    position_info=load_position(nwb_copy_file_name,pos_name)

    head_speed=xr.Dataset.from_dataframe(pd.DataFrame(position_info['head_speed']))
    head_orientation=xr.Dataset.from_dataframe(pd.DataFrame(position_info['head_orientation']))

    # load linear position
    linear_position_df = xr.Dataset.from_dataframe((IntervalLinearizedPosition() &
                          {'nwb_file_name': nwb_copy_file_name,
                           'interval_list_name': pos_name,
                           'position_info_param_name': 'default_decoding'}
                         ).fetch1_dataframe())

    # load state script
    key={'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}
    log=(TrialChoice & key).fetch1('choice_reward')
    log_df=pd.DataFrame(log)

    theta_intervals_df.insert(1,'start_time',[theta_intervals_df.loc[i].theta_interval[0] for i in theta_intervals_df.index]) #hold ripple times
    theta_intervals_df.insert(2,'end_time',[theta_intervals_df.loc[i].theta_interval[1] for i in theta_intervals_df.index]) #hold ripple times

    if 'trial_number' in theta_intervals_df.columns:
        theta_intervals_df = theta_intervals_df.drop(columns = ['trial_number'])
    if 'animal_location' in theta_intervals_df.columns:
        theta_intervals_df = theta_intervals_df.drop(columns = ['animal_location'])
    if 'head_orientation' in theta_intervals_df.columns:
        theta_intervals_df = theta_intervals_df.drop(columns = ['head_orientation'])

    theta_intervals_df = add_trial(theta_intervals_df,log_df)
    theta_intervals_df = add_location(theta_intervals_df,
                                    linear_position_df,welllocations,colind = 3)
    theta_intervals_df = add_head_orientation(theta_intervals_df,
                                              head_orientation,colind =  5)

    """Section 3: Insert into ThetaIntervals table """
    animal = nwb_copy_file_name[:5]
    savePath=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                        nwb_copy_file_name+'_'+session_name+'_theta_intervals.nc')
    theta_intervals_df.to_csv(savePath)

    key = {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': session_name}
    key['theta_times'] = savePath
    ThetaIntervals().insert1(key,replace = True)
    print("Done inserting into Spyglass.")

    """
    # to load
    loadPath = (ThetaIntervals() & {'nwb_file_name': nwb_copy_file_name,
                                    'interval_list_name': session_name}).fetch1('theta_times')
    theta_intervals_df = pd.read_csv(loadPath)
    """
    return theta_intervals_df, pos1d



def return_skaggs_theta(nwb_copy_file_name,pos_name,session_name,
                        nwb_units_all,cell_list):
    """
    pos1d and 2d should be nonmaze time removed
    shift theta like that in (Skaggs et al., 1996).
    """
    # get linearized location and 2D position
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

    # load theta
    theta_xr = load_theta_maze(nwb_copy_file_name,session_name)
    theta_xr = theta_xr.assign(phase0=('time',np.angle(ss.hilbert(theta_xr[0]))))
    theta_xr = theta_xr.assign(phase1=('time',np.angle(ss.hilbert(theta_xr[1]))))
    theta_df = theta_xr.to_dataframe()

    # shift by Skaggs
    delta = return_theta_offset(pos1d,pos2d,theta_df,nwb_units_all,cell_list)

    theta_df.phase0 = theta_df.phase0 - delta
    theta_df.phase1 = theta_df.phase1 - delta

    # make sure the final range is within -np.pi to np.pi
    theta_df.phase0 = np.angle(np.cos(theta_df.phase0) + np.sin(theta_df.phase0)*1j)
    theta_df.phase1 = np.angle(np.cos(theta_df.phase1) + np.sin(theta_df.phase1)*1j)

    return theta_df, pos1d, pos2d

def return_theta_phase_histogram(pos1d,pos2d,theta_df,nwb_units_all,unitID):
    """NOTE: pos1d and 2d should be nonmaze time removed
    returns the phase of maximal CA1 firing (Skaggs et al., 1996).
    substract this value from theta_df to make plots/analysis
    """
    (e,u) = unitID #(30,3)

    nwb_e = nwb_units_all[e]
    spike_time = nwb_e.loc[u].spike_times

    spike_time = spike_time[np.logical_and(spike_time>=pos1d.index[0],
                                           spike_time<=pos1d.index[-1])]

    # only movement data
    pos2d = interpolate_to_new_time(pos2d,spike_time)
    mobility_index = np.argwhere(pos2d.head_speed > 4).ravel() # >4cm/s
    pos1d = pos1d.iloc[mobility_index]
    spike_time = spike_time[mobility_index]

    theta_df_spike_time_all = interpolate_to_new_time(theta_df,spike_time)

    count, bins = np.histogram(theta_df_spike_time_all.phase0,24,density = True)

    return count, bins

def return_theta_offset(pos1d,pos2d,theta_df,nwb_units_all,cell_list):
    """
    define the phase of maximal CA1 firing to be 0 (Skaggs et al., 1996) degree.
    """

    counts = []
    for i in range(len(cell_list)):
        unitID = cell_list[i]
        count, bins = return_theta_phase_histogram(pos1d,pos2d,theta_df,nwb_units_all,unitID)
        counts.append(count)
    counts = np.array(counts)

    population_firing = np.sum(counts, axis = 0)
    max_firing_ind = np.argmax(population_firing)
    delta = (bins[max_firing_ind] + bins[max_firing_ind + 1])/2

    return delta

def return_theta_phase_location(pos1d,pos2d,theta_df,nwb_units_all,unitID):
    """
    (for phase precession plots)
    for each unit (unitID, electrode - unit tuple)'s every spike,
    return tuples of theta phase and animal location, split among inbound and outbound
    """
    # delta is the shift that allows
    (e,u) = unitID

    nwb_e = nwb_units_all[e]
    spike_time = nwb_e.loc[u].spike_times

    spike_time = spike_time[np.logical_and(spike_time>=pos1d.index[0],
                                           spike_time<=pos1d.index[-1])]

    pos1d_spike_time_all = interpolate_to_new_time(pos1d,spike_time)
    pos2d_spike_time_all = interpolate_to_new_time(pos2d,spike_time)
    theta_df_spike_time_all = interpolate_to_new_time(theta_df,spike_time)

    mobility_index = np.argwhere(pos2d_spike_time_all.head_speed > 4).ravel() # >4cm/s

    # the 2 tables for phase precession plot
    pos1d_spike_time = pos1d_spike_time_all.iloc[mobility_index]
    pos2d_spike_time = pos2d_spike_time_all.iloc[mobility_index]
    theta_df_spike_time = theta_df_spike_time_all.iloc[mobility_index]

    # split into inbound and outbound
    pos_phase = {}
    arms = {}
    direction_names = ['outbound','inbound']
    for i in range(2):
        direction = (pos2d_spike_time.head_orientation > 0,pos2d_spike_time.head_orientation < 0)

        # restrict to a direction
        pos1d_spike_time_dir = pos1d_spike_time[direction[i]]
        theta_df_spike_time_dir = theta_df_spike_time[direction[i]]

        if len(pos1d_spike_time_dir) == 0:
            pos_phase[direction_names[i]] = (pos1d_spike_time_dir['linear_position'],
                                             theta_df_spike_time_dir['phase0'])
            arms[direction_names[i]] = np.nan
            continue

        # restrict to one maze segment
        mode, _ = stats.mode(pos1d_spike_time_dir.track_segment_id)
        place_range, arms[direction_names[i]] = segment_to_linear_range(linear_map,mode)
        time_ind = np.logical_and(pos1d_spike_time_dir['linear_position']>=place_range[0],
                              pos1d_spike_time_dir['linear_position']<=place_range[1])

        pos_phase[direction_names[i]] = (pos1d_spike_time_dir['linear_position'][time_ind],theta_df_spike_time_dir['phase0'][time_ind])

    return pos_phase,arms

def find_cell_connectivity_theta(cell_list,theta_intervals_df):
    G, weights = make_graph(cell_list,list((theta_intervals_df.active_cells_ind)))
    weightMatrix = np.zeros((len(cell_list),len(cell_list)))
    for key in weights.keys():
        w = weights[key]
        weightMatrix[key[0],key[1]] = weights[key]

    for row in range(weightMatrix.shape[0]):
        weightMatrix[row] = weightMatrix[row] / weightMatrix[row][row]

    """
    from sklearn.preprocessing import normalize
    unnormalizedW = weightMatrix.copy()
    weightMatrix = normalize(weightMatrix, axis=1, norm='l1')
    weightMatrix = normalize(weightMatrix, axis=0, norm='l1')
    """

    print("The shape of the weightMatrix is: ", weightMatrix.shape)
    return weightMatrix

def sort_connectivity(weightMatrix, ref_cell_name, cell_list):
    """Given a weight matrix and a cell, find its connected cells, sorted from top to low"""
    # cell_list for translating between cell index and name

    ref_cell_ind = cell_name_to_ind(ref_cell_name,cell_list)

    cell_connected_ind = np.argwhere(weightMatrix[ref_cell_ind,:]).ravel()
    weight_connected = weightMatrix[ref_cell_ind,cell_connected_ind]

    ascending_indices = np.argsort(weight_connected)
    descending_indices = ascending_indices[::-1]

    weight_connected = weight_connected[descending_indices]
    cell_connected_ind = cell_connected_ind[descending_indices]

    cell_connected_name = np.array(cell_list)[cell_connected_ind]
    return cell_connected_ind, cell_connected_name, weight_connected

def find_theta_cycle(theta_intervals_df, session_cell_list, # session info
                     required_cell_name, #required cell name to be in the theta cycle
                     animal_location = None, # required animal location
                     criteron_num = None, ref_cell = None):
    """
    (for finding theta cycles that have cofiring of cells)
    return theta cycle indices that have cofiring of criteron_num number of cells in the required_cell_ind
    """

    # convert to index
    ref_ind = -1
    candidates_ind = [cell_name_to_ind(p,session_cell_list) for p in required_cell_name]
    if ref_cell is not None:
        ref_ind = cell_name_to_ind(ref_cell,session_cell_list)

    cofiring_theta_ind = []
    for i in theta_intervals_df.index:
        this_cycle_cells = theta_intervals_df.loc[i].active_cells_ind
        this_cycle_location = theta_intervals_df.loc[i].animal_location
        if (ref_cell is not None) and (not np.isin(ref_ind,this_cycle_cells)):
                continue
        if (np.sum(np.isin(candidates_ind,this_cycle_cells)) >= criteron_num):
            if (animal_location is not None) and (this_cycle_location == animal_location):
                cofiring_theta_ind.append(i)
            elif animal_location is None:
                cofiring_theta_ind.append(i)
    return cofiring_theta_ind

def find_cofiring_theta_cycle(theta_intervals_df, candidates, cell_list, criteron_num, ref_cell):
    """
    (for finding theta cycles that have cofiring of cells)
    return theta cycle indices that have cofiring of criteron_num number of cells in the candidates
    """

    # convert to index
    ref_ind = -1
    candidates_ind = [cell_name_to_ind(p,cell_list) for p in candidates]
    if ref_cell is not None:
        ref_ind = cell_name_to_ind(ref_cell,cell_list)

    cofiring_theta_ind = []
    for i in theta_intervals_df.index:
        this_cycle_cells = theta_intervals_df.loc[i].active_cells_ind
        if (ref_cell is not None) and (not np.isin(ref_ind,this_cycle_cells)):
                continue
        if np.sum(np.isin(candidates_ind,this_cycle_cells)) >= criteron_num:
            cofiring_theta_ind.append(i)
    return cofiring_theta_ind

def plot_place_field_group(cells,placefields,cell_list,
                           peak_frs = None,
                           mobility_spike_counts = None,
                           count_ratio = None, #the spike count ratio between continuous and frag replays
                           ref_cell_ind = None,
                           weightMatrix = None # weight matrix entry from ref_cell_ind
                           ):
    #pairs_fragmented = [(13,4),(14,21),(5,11),(17,13),(14,19),(16,8),(23,4),(23,5)]
    #pairs_fragmented = [(30,3),(13,2),(13,8),(26,7),(5,11),(16,11),(13,9)]
    #pairs_fragmented = [(17,4),(5,5),(13,2),(13,8),(26,7),(16,9),(0,6),(5,11),(13,10),(20,7),
    #                    (20,12),(14,19),(17,14),(16,11),(23,4),(17,5),(31,2),(16,8),(30,3)]
    #pairs_fragmented = [(17,4),(5,5),(13,2),(0,6),(20,7),(14,19)]

    col_num = int(np.ceil(len(cells)/2))
    fig, axes = plt.subplots(2,col_num, figsize = (col_num * 3,5), squeeze = True)

    ind = 0
    for p in cells:
        (e,u) = p
        cell_ind = cell_name_to_ind(p,cell_list)
        place_field_plot = placefields[(e,u)]
        axes[np.unravel_index(ind, axes.shape)].imshow(place_field_plot,
                                                    vmax = np.nanquantile(place_field_plot,0.999),
                                                    vmin = np.nanquantile(place_field_plot,0.4)
                                                    )
        max_loc = np.unravel_index(np.nanargmax(place_field_plot),np.shape(place_field_plot))
        axes[np.unravel_index(ind, axes.shape)].scatter(max_loc[1],max_loc[0],20,marker = '+',color = 'C1')

        axes[np.unravel_index(ind, axes.shape)].set_title(p)
        if peak_frs is not None:
            axes[np.unravel_index(ind, axes.shape)].text(80,-27,round(peak_frs[(e,u)],1))
        if mobility_spike_counts is not None:
            axes[np.unravel_index(ind, axes.shape)].text(80,-20,'mob spike num:'+str(mobility_spike_counts[(e,u)]))
        if count_ratio is not None:
            axes[np.unravel_index(ind, axes.shape)].text(80,-13,'count ratio:'+str(round(count_ratio[cell_ind],1)))
        if weightMatrix is not None:
            #axes[np.unravel_index(ind, axes.shape)].text(80,-5,'weight:'+str(weightMatrix[ref_cell_ind][cell_ind]))
            axes[np.unravel_index(ind, axes.shape)].text(80,-5,'weight from ref:'+str(np.round(weightMatrix[ref_cell_ind][cell_ind],4)))
            axes[np.unravel_index(ind, axes.shape)].text(80,3,'weight to ref:'+str(np.round(weightMatrix[cell_ind][ref_cell_ind],4)))
        axes[np.unravel_index(ind, axes.shape)].set_axis_off()

        ind = ind + 1
    return fig, axes

def scatter_place_field_group(cells,rasters,cell_list,pos2d,
                              peak_frs = None,
                              mobility_spike_counts = None,
                              count_ratio = None, #the spike count ratio between continuous and frag replays
                              ref_cell_ind = None,
                              weightMatrix = None # weight matrix entry from ref_cell_ind
                              ):
    """
    rasters and pos2d can be made from the following code:

    from spyglass.shijiegu.placefield import raster_field

    rasters = {}
    for row_ind in range(len(cell_list)):
        print(row_ind)
        e = cell_list[row_ind][0]
        u = cell_list[row_ind][1]
        pos2d, pos2d_spike_time, _ = raster_field(
            nwb_copy_file_name, session_name, pos_name, e, u)
        rasters[(e,u)] = pos2d_spike_time

    """

    col_num = int(np.ceil(len(cells)/2))
    fig, axes = plt.subplots(2,col_num, figsize = (col_num * 3,5), squeeze = True)

    ind = 0
    for p in cells:
        (e,u) = p
        cell_ind = cell_name_to_ind(p,cell_list)
        raster_plot = rasters[(e,u)]

        axes[np.unravel_index(ind, axes.shape)].scatter(pos2d.head_position_x, pos2d.head_position_y, s = 1, color = [0.5, 0.5, 0.5])
        axes[np.unravel_index(ind, axes.shape)].scatter(raster_plot.head_position_x, raster_plot.head_position_y, s = 1, edgecolor = 'k')
        axes[np.unravel_index(ind, axes.shape)].invert_yaxis()
        axes[np.unravel_index(ind, axes.shape)].set_aspect('equal', adjustable='box')

        #max_loc = np.unravel_index(np.nanargmax(place_field_plot),np.shape(place_field_plot))
        #axes[np.unravel_index(ind, axes.shape)].scatter(max_loc[1],max_loc[0],20,marker = '+',color = 'C1')

        axes[np.unravel_index(ind, axes.shape)].set_title(p)
        if peak_frs is not None:
            axes[np.unravel_index(ind, axes.shape)].text(280,-35,round(peak_frs[(e,u)],1))
        if mobility_spike_counts is not None:
            axes[np.unravel_index(ind, axes.shape)].text(280,-20,'mob spike num:'+str(mobility_spike_counts[(e,u)]))
        if count_ratio is not None:
            axes[np.unravel_index(ind, axes.shape)].text(280,-5,'count ratio:'+str(round(count_ratio[cell_ind],1)))
        if weightMatrix is not None:
            axes[np.unravel_index(ind, axes.shape)].text(280,10,'weight from ref:'+str(np.round(weightMatrix[ref_cell_ind][cell_ind],4)))
            axes[np.unravel_index(ind, axes.shape)].text(280,25,'weight to ref:'+str(np.round(weightMatrix[cell_ind][ref_cell_ind],4)))

        axes[np.unravel_index(ind, axes.shape)].set_axis_off()

        ind = ind + 1

    return fig, axes

def plot_cell_spikes(axes,cells,t0,t1,pos1d,nwb_units_all,ind = 0):
    # plot spikes of cells in an axes
    # peak_frs,mobility_spike_counts are dictionaries
    # ind is the starting row for plotting

    color_ind = 0

    for cell in cells:
        (e,u) = cell
        nwb_units = nwb_units_all[e]
        spike_times = nwb_units.loc[u].spike_times
        #if peak_frs[(e,u)] > 20 or mobility_spike_counts[(e,u)] > 5000:
        #    continue
        spike_times = spike_times[np.logical_and(spike_times>=t0,
                                                     spike_times<=t1)]

        #pos1d_spike_time_all = interpolate_to_new_time(pos1d,spike_times,'nearest')


        axes.scatter(pos1d.index,pos1d.linear_position,s = 1,color = 'k')
        axes.scatter(spike_times, np.zeros_like(spike_times) + ind*20 + 100,
                         rasterized=True,marker='|',color = 'C' + str(color_ind),s = 5)
        #axes.scatter(spike_times,pos1d_spike_time_all.linear_position,rasterized=True,marker='|',color = 'C' + str(ind))
        # add cell name:
        axes.text(t0 + 0.1*(ind % 3), ind*20 + 100, str((e,u)),
                  fontsize = 7, color = 'C' + str(color_ind))
        ind = ind + 1
        color_ind = color_ind + 1