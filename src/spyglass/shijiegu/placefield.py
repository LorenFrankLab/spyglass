import numpy as np

from scipy.ndimage.filters import gaussian_filter
import track_linearization as ti
from spyglass.common.common_position import TrackGraph

import pandas as pd

from spyglass.common.common_position import IntervalPositionInfo

from spyglass.linearization.v0.main import IntervalLinearizedPosition

from spyglass.shijiegu.singleUnit import findWaveForms
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.Analysis_SGU import TrialChoice, get_linearization_map
from spyglass.common.common_position import TrackGraph

def raster_field(nwb_copy_file_name, session_name, pos_name, electrode, unit):
    # default is 2 cm bins, smoothed by 2 bins = 4 cm

    nwb_units,extractor = findWaveForms(nwb_copy_file_name,session_name,electrode)

    # get spike time
    spike_time = nwb_units.loc[unit].spike_times

    # get linearized location and 2D position
    pos1d = (IntervalLinearizedPosition() & {'nwb_file_name': nwb_copy_file_name,
                                                     'track_graph_name':'4 arm lumped 2023',
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    pos2d = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    #delta_t = np.mean(np.diff(pos2d.index)) #for later, translating the unit of time number bins to time in seconds

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

    # mobility only
    mobility_index = np.argwhere(pos2d.head_speed > 4).ravel() # >4cm/s
    pos1d_mobility = pos1d.iloc[mobility_index]
    pos2d_mobility = pos2d.iloc[mobility_index]

    pos2d_spike_time_all = interpolate_to_new_time(pos2d,spike_time,
                                                   upsampling_interpolation_method = 'nearest')
    mobility_index = np.argwhere(pos2d_spike_time_all.head_speed > 4).ravel() # >4cm/s
    pos2d_spike_time = pos2d_spike_time_all.iloc[mobility_index]

    return pos2d, pos2d_spike_time, total_spike_count

def place_field(nwb_copy_file_name, session_name, pos_name, electrode, unit, BINWIDTH = 2, sigma = 2, curation_id = 0):
    # default is 2 cm bins, smoothed by 2 bins = 4 cm

    nwb_units,extractor = findWaveForms(nwb_copy_file_name,session_name,electrode,curation_id)

    # get spike time
    spike_time = nwb_units.loc[unit].spike_times

    # get linearized location and 2D position
    pos1d = (IntervalLinearizedPosition() & {'nwb_file_name': nwb_copy_file_name,
                                                     'track_graph_name':'4 arm lumped 2023',
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    pos2d = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                                                     'interval_list_name':pos_name,
                                                     'position_info_param_name':'default'}).fetch1_dataframe()

    delta_t = np.mean(np.diff(pos2d.index)) #for later, translating the unit of time number bins to time in seconds

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

    # mobility only
    mobility_index = np.argwhere(pos2d.head_speed > 4).ravel() # >4cm/s
    pos1d_mobility = pos1d.iloc[mobility_index]
    pos2d_mobility = pos2d.iloc[mobility_index]

    pos2d_spike_time_all = interpolate_to_new_time(pos2d,spike_time,
                                                   upsampling_interpolation_method = 'nearest')
    mobility_index = np.argwhere(pos2d_spike_time_all.head_speed > 4).ravel() # >4cm/s
    pos2d_spike_time = pos2d_spike_time_all.iloc[mobility_index]

    #Define bins for all position
    xmin = np.min(pos2d_mobility.head_position_x)-10
    xmax = np.max(pos2d_mobility.head_position_x)+10
    ymin = np.min(pos2d_mobility.head_position_y)-10
    ymax = np.max(pos2d_mobility.head_position_y)+10
    xbins = np.arange(xmin,xmax,BINWIDTH)
    ybins = np.arange(ymin,ymax,BINWIDTH)

    # place field, aka occupancy normalized firing rate, aka P(spike | location)
    occupancy, xe, ye = np.histogram2d(pos2d_mobility.head_position_y,
                                       pos2d_mobility.head_position_x, bins = [ybins,xbins])
    smoothed_occupancy = gaussian_filter(occupancy, sigma = sigma) #1 bins = 1cm
    occupancy = occupancy * delta_t # number of entry * delta_t second / entry
    smoothed_occupancy = smoothed_occupancy * delta_t
    old_nan_index = smoothed_occupancy < 0.01
    occupancy[occupancy < 0.001] = np.nan # for numerical stability

    spike, xe, ye = np.histogram2d(pos2d_spike_time.head_position_y,
                                   pos2d_spike_time.head_position_x, bins = [ybins,xbins])


    pf = spike/occupancy
    pf = np.nan_to_num(pf, nan = 0, posinf = 0)
    smoothed_placefield = gaussian_filter(pf, sigma = sigma) #1 bin = 1cm
    smoothed_placefield[old_nan_index] = np.nan
    peak_firing_rate = np.nanmax(smoothed_placefield)

    spike_count = len(pos2d_spike_time)

    return smoothed_placefield, peak_firing_rate, xbins, ybins, spike_count, total_spike_count

def placefield_to_peak1dloc(track_graph_name, placefield, ybins, xbins):
    """
    graph is
    ybin: placefield axis in the y dimenstion
    xbin: placefield axis in the x dimenstion
    """
    graph = TrackGraph() & {'track_graph_name': track_graph_name}
    track_graph = ti.make_track_graph(
        graph.fetch1('node_positions'), graph.fetch1('edges'))
    max_ind = np.unravel_index(np.nanargmax(placefield),placefield.shape)
    max_location_2D = (xbins[max_ind[1]],ybins[max_ind[0]])
    max_location_1D = ti.get_linearized_position(np.array(max_location_2D).T.reshape((1,-1)),
                                                  track_graph,use_HMM = False,
                                                  edge_order=graph.fetch1('linear_edge_order'),
                                                  edge_map=graph.fetch1('edge_map'),
                                                  edge_spacing=graph.fetch1('linear_edge_spacing')).linear_position
    return float(max_location_1D)

def cell_by_arm(peak1ds_list, cell_list):
    """
    return cells and cell indices of cell_list that are in arm x
    """
    cell_list_by_arm = {}
    cell_list_ind_by_arm = {}

    linear_map,node_location=get_linearization_map()

    spatial_bins = []
    segments_ind = [0,3,5,7,9]
    segments_name = ['home','arm1','arm2','arm3','arm4']

    ind = 0
    for seg in segments_ind:
        p0p1= linear_map[seg]
        name = segments_name[ind]
        cell_ind = np.argwhere(np.logical_and(peak1ds_list >= p0p1[0], peak1ds_list <= p0p1[1])).ravel()
        cell_list_by_arm[name] = np.array(cell_list)[cell_ind]
        cell_list_ind_by_arm[name] = cell_ind
        ind = ind + 1

    return cell_list_by_arm, cell_list_ind_by_arm