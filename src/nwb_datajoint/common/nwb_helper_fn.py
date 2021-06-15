"""NWB helper functions for finding processing modules and data interfaces."""

import numpy as np
import pynwb
import warnings

__open_nwb_files = dict()  # dict mapping file path to an open NWBHDF5IO object in read mode and its NWBFile


def get_nwb_file(nwb_file_path):
    """Return an NWBFile object with the given file path in read mode.

    Parameters
    ----------
    nwb_file_path : str
        Path to the NWB file.

    Returns
    -------
    nwbfile : pynwb.NWBFile
        NWB file object for the given path opened in read mode.
    """
    _, nwbfile = __open_nwb_files.get(nwb_file_path, (None, None))
    if nwbfile is None:
        io = pynwb.NWBHDF5IO(path=nwb_file_path, mode='r', load_namespaces=True)  # keep file open
        nwbfile = io.read()
        __open_nwb_files[nwb_file_path] = (io, nwbfile)
    return nwbfile


def close_nwb_files():
    for io, _ in __open_nwb_files.values():
        io.close()
    __open_nwb_files.clear()


def get_data_interface(nwbfile, data_interface_name, data_interface_class=pynwb.core.NWBDataInterface):
    """Search for a specified data interface in the processing modules of an NWB file.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file object to search in.
    data_interface_name : str
        The name of the NWBDataInterface to search for.
    data_interface_class : type, optional
        The class (or superclass) of the NWBDataInterface to search for (default: pynwb.core.NWBDataInterface).

    Warns
    -----
    UserWarning
        If multiple data interfaces with the matching name are found.

    Returns
    -------
    data_interface : NWBDataInterface
        The data interface object with the given name, or None if not found.
    """
    ret = []
    for module in nwbfile.processing.values():
        match = module.data_interfaces.get(data_interface_name, None)
        if match is not None and isinstance(match, data_interface_class):
            ret.append(match)
    if len(ret) > 1:
        warnings.warn(f"Multiple data interfaces with name '{data_interface_name}' and class {data_interface_class} "
                      f"found in NWBFile with identifier {nwbfile.identifier}. Using the first one found.")
    if len(ret) >= 1:
        return ret[0]
    else:
        return None


def get_raw_eseries(nwbfile):
    """Return all ElectricalSeries in the acquisition group of an NWB file.

    ElectricalSeries found within LFP objects in the acquisition will also be returned.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file object to search in.

    Returns
    -------
    ret : list
        A list of all ElectricalSeries in the acquisition group of an NWB file
    """
    ret = []
    for nwb_object in nwbfile.acquisition.values():
        if isinstance(nwb_object, pynwb.ecephys.ElectricalSeries):
            ret.append(nwb_object)
        elif isinstance(nwb_object, pynwb.ecephys.LFP):
            ret.extend(nwb_object.electrical_series.values())
    return ret


def estimate_sampling_rate(timestamps, multiplier):
    """Estimate the sampling rate given a list of timestamps.

    Assumes that the most common temporal differences between timestamps approximate the sampling rate. Note that this
    can fail for very high sampling rates and irregular timestamps.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D numpy array of timestamp values.
    multiplier : float or int

    Returns
    -------
    estimated_rate : float
        The estimated sampling rate.
    """

    # approach:
    # 1. use a box car smoother and a histogram to get the modal value
    # 2. identify adjacent samples as those that have a time difference < the multiplier * the modal value
    # 3. average the time differences between adjacent samples
    sample_diff = np.diff(timestamps[~np.isnan(timestamps)])
    if len(sample_diff) < 10:
        raise ValueError(f'Only {len(sample_diff)} timestamps are valid. Check the data.')
    nsmooth = 10
    smoother = np.ones(nsmooth) / nsmooth
    smooth_diff = np.convolve(sample_diff, smoother, mode='same')

    # we histogram with 100 bins out to 3 * mean, which should be fine for any reasonable number of samples
    hist, bins = np.histogram(smooth_diff, bins=100, range=[0, 3 * np.mean(smooth_diff)])
    mode = bins[np.where(hist == np.max(hist))]

    adjacent = sample_diff < mode[0] * multiplier
    return np.round(1.0 / np.mean(sample_diff[adjacent]))


def get_valid_intervals(timestamps, sampling_rate, gap_proportion, min_valid_len):
    """Given a list of timestamps from sampled data, and the sampling rate of the data, find all gaps >
    gap_proportion * sampling rate and return a set of valid times excluding these gaps.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D numpy array of timestamp values.
    sampling_rate : float
        Sampling rate of the data.
    gap_proportion : float
        Multiple of inferred sampling rate for gap detection.
    min_valid_len : float
        Length of smallest valid interval.

    Returns
    -------
    valid_times : np.ndarray
        Array of start and stop times of shape (N, 2) for valid data.
    """

    eps = 0.0000001
    # get rid of NaN elements
    timestamps = timestamps[~np.isnan(timestamps)]

    gap = np.diff(timestamps) > 1.0 / sampling_rate * gap_proportion

    # all true entries of gap represent gaps. Get the times bounding these intervals.
    gapind = np.asarray(np.where(gap))
    # The end of each valid interval are the indices of the gaps and the final value
    valid_end = np.append(gapind, np.asarray(len(timestamps)-1))

    # the beginning of the gaps are the first element and gapind+1
    valid_start = np.insert(gapind + 1, 0, 0)

    valid_indices = np.vstack([valid_start, valid_end]).transpose()

    valid_times = timestamps[valid_indices]
    # adjust the times to deal with single valid samples
    valid_times[:, 0] = valid_times[:, 0] - eps
    valid_times[:, 1] = valid_times[:, 1] + eps

    valid_intervals = (valid_times[:, 1] - valid_times[:, 0]) > min_valid_len

    return valid_times[valid_intervals, :]


def get_electrode_indices(nwb_object, electrode_ids):
    """Given an NWB file or electrical series object, return the indices of the specified electrode_ids from the
    NWB file electrodes table.

    Parameters
    ----------
    nwb_object : pynwb.NWBFile or pynwb.ecephys.ElectricalSeries
        The NWB file object or NWB electrical series object.
    electrode_ids : np.ndarray or list
        Array or list of electrode IDs.

    Returns
    -------
    electrode_indices : np.ndarray
        Array of indices of the specified electrode IDs.
    """
    if isinstance(nwb_object, pynwb.ecephys.ElectricalSeries):
        # electrode_table_region = list(electrical_series.electrodes.to_dataframe().index)  # TODO verify
        electrode_table_indices = nwb_object.electrodes.data[:]  # dynamictableregion of row indices
    elif isinstance(nwb_object, pynwb.NWBFile):
        electrode_table_indices = nwb_object.electrodes.id[:]

    return [elect_idx for elect_idx, elect_id in enumerate(electrode_table_indices) if elect_id in electrode_ids]

def get_all_spatial_series(nwbf, verbose=False):
    """Given an nwb file object, gets the spatial series and Interval lists from the file and returns a dictionary by epoch
    :param nwbf: the nwb file object
    :type nwbf: file object
    :param verbose: flag to cause printing of sampling rate
    :type verbose: bool
    """
    position = get_data_interface(nwbf, 'position', pynwb.behavior.Position)
    if position is None:
        return None

    pos_data_dict = dict()
    for pos_epoch, spatial_series in enumerate(position.spatial_series.values()):
        pos_data_dict[pos_epoch] = dict()
        # get the valid intervals for the position data
        timestamps = np.asarray(spatial_series.timestamps)

        # estimate the sampling rate
        sampling_rate = estimate_sampling_rate(timestamps, 1.75)
        if sampling_rate < 0:
            raise ValueError(f'Error adding position data for position epoch {pos_epoch}')
        if verbose:
            print("Processing raw position data. Estimated sampling rate: {} Hz".format(sampling_rate))
        # add the valid intervals to the Interval list
        pos_data_dict[pos_epoch]['valid_times'] = get_valid_intervals(timestamps, sampling_rate, 2.5, 0)
        pos_data_dict[pos_epoch]['raw_position_object_id'] = spatial_series.object_id

    return pos_data_dict
    
            