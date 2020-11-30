import numpy as np
from operator import itemgetter 
  
# import matplotlib.pyplot as plt

#NWB helper functions for finding processing modules and data interfaces

def get_data_interface(nwbf, data_interface_name):
    """ Search for a specified data interface in an NWB file
    :param nwbf: NWB file object
    :param data_interface_name: string with name of data interface
    :return: data interface object or None
    """
    for module_name in nwbf.processing:
        module = nwbf.processing[module_name]
        if data_interface_name in module.data_interfaces:
            return module.get(data_interface_name) 
    return None

def estimate_sampling_rate(timestamps, multiplier):
    '''Estimate the sampling rate given a list of timestamps. Assumes that the most common temporal differences
       between timestamps approximate the sampling rate. Note that this can fail for very high sampling rates and
       irregular timestamps
    :param timestamps: 1D numpy array (list of timestamps)
    :return: estimated_rate: float
    '''

    #approach:
    # 1. use a box car smoother and a histogram to get the modal value
    # 2. identify adjacent samples as those that have a time difference < the multiplier * the modal value
    # 3. average the time differences between adjacent samples
    sample_diff = np.diff(timestamps[~np.isnan(timestamps)])
    if len(sample_diff) < 10:
        print(f'Error in estimate_sampling_rate: only {len(sample_diff)} timestamps are valid. Check the data.')
        return -1
    nsmooth = 10
    smoother = np.ones(nsmooth) / nsmooth
    smooth_diff = np.convolve(sample_diff, smoother, mode='same')
#    plt.figure()
#    plt.plot(timestamps)
#    plt.show()
    # we histogram with 100 bins out to 3 * mean, which should be fine for any reasonable number of samples
    hist, bins = np.histogram(smooth_diff, bins=100, range=[0, 3 * np.mean(smooth_diff)])
    mode = bins[np.where(hist == np.max(hist))]

    adjacent = sample_diff < mode[0] * multiplier
    return np.round(1.0 / np.mean(sample_diff[adjacent]))


def get_valid_intervals(timestamps, sampling_rate, gap_proportion, min_valid_len):
    """ Given a list of timestamps from sampled data, and the sampling rate of the data, find all gaps >
    gap_proporition * sampling rate and return a set of valid times excluding these gaps.
    :param timestamps: 1D numpy array (list of timestamps)
    :param gap_proportion: float (multiple of inferred sampling rate for gap detection)
    :param min_valid_len: float (length of smallest valid interval)
    :return: valid_times (2D array of start and stop times for valid data)"""

    eps = 0.0000001
    # get rid of NaN elements
    timestamps = timestamps[~np.isnan(timestamps)]

    gap = np.diff(timestamps) > 1.0 / sampling_rate * gap_proportion

    #all true entries of gap represent gaps. Get the times bounding these intervals.
    gapind = np.asarray(np.where(gap))
    # The end of each valid interval are the indices of the gaps and the final value
    valid_end = np.append(gapind, np.asarray(len(timestamps)-1))

    # the beginning of the gaps are the first element and gapind+1
    valid_start = np.insert(gapind + 1, 0, 0)

    valid_indices = np.vstack([valid_start, valid_end]).transpose()

    valid_times = timestamps[valid_indices]
    # adjust the times to deal with single valid samples
    valid_times[:,0] = valid_times[:,0] - eps
    valid_times[:,1] = valid_times[:,1] + eps

    valid_intervals = (valid_times[:,1] - valid_times[:,0]) > min_valid_len

    return valid_times[valid_intervals,:]

def get_electrode_indices(electrical_series, electrode_ids):
    """Given an NWB file or electrical series object , returns the indices of the specified electrode_ids. 

    :param electrical_series: NWB file handle or nwb electrical series object
    :type electrical_series: 
    :param electrode_ids: array or list of electrode_ids
    :type electrode_ids: numpy array or list
    :return: electrode_indices (numpy array of indices)
    """
    electrode_table_region = list(electrical_series.electrodes.to_dataframe().index)
    return [elect_idx for elect_idx, elect_id in enumerate(electrode_table_region) if elect_id in electrode_ids]

