"""NWB helper functions for finding processing modules and data interfaces."""

import os
import os.path
import warnings
from itertools import groupby
from pathlib import Path

import numpy as np
import pynwb
import yaml

# dict mapping file path to an open NWBHDF5IO object in read mode and its NWBFile
__open_nwb_files = dict()

# dict mapping NWB file path to config after it is loaded once
__configs = dict()

global invalid_electrode_index
invalid_electrode_index = 99999999


def get_nwb_file(nwb_file_path):
    """Return an NWBFile object with the given file path in read mode.

    If the file is not found locally, this will check if it has been shared
    with kachery and if so, download it and open it.


    Parameters
    ----------
    nwb_file_path : str
        Path to the NWB file or NWB file name. If it does not start with a "/",
        get path with Nwbfile.get_abs_path

    Returns
    -------
    nwbfile : pynwb.NWBFile
        NWB file object for the given path opened in read mode.
    """
    if not nwb_file_path.startswith("/"):
        from ..common import Nwbfile

        nwb_file_path = Nwbfile.get_abs_path(nwb_file_path)

    _, nwbfile = __open_nwb_files.get(nwb_file_path, (None, None))

    if nwbfile is None:
        # check to see if the file exists
        if not os.path.exists(nwb_file_path):
            print(
                "NWB file not found locally; checking kachery for "
                + f"{nwb_file_path}"
            )
            # first try the analysis files
            from ..sharing.sharing_kachery import AnalysisNwbfileKachery

            # the download functions assume just the filename, so we need to
            # get that from the path
            if not AnalysisNwbfileKachery.download_file(
                os.path.basename(nwb_file_path)
            ):
                return None
        # now open the file
        io = pynwb.NWBHDF5IO(
            path=nwb_file_path, mode="r", load_namespaces=True
        )  # keep file open
        nwbfile = io.read()
        __open_nwb_files[nwb_file_path] = (io, nwbfile)

    return nwbfile


def get_config(nwb_file_path):
    """Return a dictionary of config settings for the given NWB file.
    If the file does not exist, return an empty dict.

    Parameters
    ----------
    nwb_file_path : str
        Absolute path to the NWB file.

    Returns
    -------
    dict
        Dictionary of configuration settings loaded from the corresponding YAML file
    """
    if nwb_file_path in __configs:  # load from cache if exists
        return __configs[nwb_file_path]

    p = Path(nwb_file_path)
    # NOTE use p.stem[:-1] to remove the underscore that was added to the file
    config_path = p.parent / (p.stem[:-1] + "_spyglass_config.yaml")
    if not os.path.exists(config_path):
        print(f"No config found at file path {config_path}")
        return dict()
    with open(config_path, "r") as stream:
        d = yaml.safe_load(stream)

    # TODO write a JSON schema for the yaml file and validate the yaml file
    __configs[nwb_file_path] = d  # store in cache
    return d


def close_nwb_files():
    for io, _ in __open_nwb_files.values():
        io.close()
    __open_nwb_files.clear()


def get_data_interface(nwbfile, data_interface_name, data_interface_class=None):
    """
    Search for NWBDataInterface or DynamicTable in processing modules of an NWB.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file object to search in.
    data_interface_name : str
        The name of the NWBDataInterface or DynamicTable to search for.
    data_interface_class : type, optional
        The class (or superclass) to search for. This argument helps to prevent
        accessing an object with the same name but the incorrect type. Default:
        no restriction.

    Warns
    -----
    UserWarning
        If multiple NWBDataInterface and DynamicTable objects with the matching
        name are found.

    Returns
    -------
    data_interface : NWBDataInterface
        The data interface object with the given name, or None if not found.
    """
    ret = []
    for module in nwbfile.processing.values():
        match = module.data_interfaces.get(data_interface_name, None)
        if match is not None:
            if data_interface_class is not None and not isinstance(
                match, data_interface_class
            ):
                continue
            ret.append(match)
    if len(ret) > 1:
        warnings.warn(
            f"Multiple data interfaces with name '{data_interface_name}' "
            f"found in NWBFile with identifier {nwbfile.identifier}. "
            + "Using the first one found. "
            "Use the data_interface_class argument to restrict the search."
        )
    if len(ret) >= 1:
        return ret[0]

    return None


def get_raw_eseries(nwbfile):
    """Return all ElectricalSeries in the acquisition group of an NWB file.

    ElectricalSeries found within LFP objects in the acquisition will also be
    returned.

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


def estimate_sampling_rate(
    timestamps, multiplier=1.75, verbose=False, filename="file"
):
    """Estimate the sampling rate given a list of timestamps.

    Assumes that the most common temporal differences between timestamps
    approximate the sampling rate. Note that this can fail for very high
    sampling rates and irregular timestamps.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D numpy array of timestamp values.
    multiplier : float or int, optional
        Deft
    verbose : bool, optional
        Print sampling rate to stdout. Default, False
    filename : str, optional
        Filename to reference when printing or err. Default, "file"

    Returns
    -------
    estimated_rate : float
        The estimated sampling rate.

    Raises
    ------
    ValueError
        If estimated rate is less than 0.
    """

    # approach:
    # 1. use a box car smoother and a histogram to get the modal value
    # 2. identify adjacent samples as those that have a
    #    time difference < the multiplier * the modal value
    # 3. average the time differences between adjacent samples

    sample_diff = np.diff(timestamps[~np.isnan(timestamps)])

    if len(sample_diff) < 10:
        raise ValueError(
            f"Only {len(sample_diff)} timestamps are valid. Check the data."
        )

    smooth_diff = np.convolve(sample_diff, np.ones(10) / 10, mode="same")

    # we histogram with 100 bins out to 3 * mean, which should be fine for any
    # reasonable number of samples
    hist, bins = np.histogram(
        smooth_diff, bins=100, range=[0, 3 * np.mean(smooth_diff)]
    )
    mode = bins[np.where(hist == np.max(hist))][0]

    adjacent = sample_diff < mode * multiplier

    sampling_rate = np.round(1.0 / np.mean(sample_diff[adjacent]))

    if sampling_rate < 0:
        raise ValueError(f"Error calculating sampling rate. For {filename}")
    if verbose:
        print(f"Estimated sampling rate for {filename}: {sampling_rate} Hz")

    return sampling_rate


def get_valid_intervals(
    timestamps, sampling_rate, gap_proportion=2.5, min_valid_len=0
):
    """Finds the set of all valid intervals in a list of timestamps.
    Valid interval: (start time, stop time) during which there are
    no gaps (i.e. missing samples).

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D numpy array of timestamp values.
    sampling_rate : float
        Sampling rate of the data.
    gap_proportion : float, optional
        Threshold for detecting a gap; i.e. if the difference (in samples)
        between consecutive timestamps exceeds gap_proportion, it is considered
        a gap. Must be > 1. Default to 2.5
    min_valid_len : float, optional
        Length of smallest valid interval. Default to 0. If greater
        than interval duration, print warning and use half the total time.

    Returns
    -------
    valid_times : np.ndarray
        Array of start and stop times of shape (N, 2) for valid data.
    """

    eps = 0.0000001

    total_time = timestamps[-1] - timestamps[0]

    if total_time < min_valid_len:
        half_total_time = total_time / 2
        print(f"WARNING: Setting minimum valid interval to {half_total_time}")
        min_valid_len = half_total_time

    # get rid of NaN elements
    timestamps = timestamps[~np.isnan(timestamps)]
    # find gaps
    gap = np.diff(timestamps) > 1.0 / sampling_rate * gap_proportion

    # all true entries of gap represent gaps. Get the times bounding these intervals.
    gapind = np.asarray(np.where(gap))
    # The end of each valid interval are the indices of the gaps and the final value
    valid_end = np.append(gapind, np.asarray(len(timestamps) - 1))

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
    """Return indices of the specified electrode_ids given an NWB file.

    Also accepts electrical series object. If an ElectricalSeries is given,
    then the indices returned are relative to the selected rows in
    ElectricalSeries.electrodes. For example, if electricalseries.electrodes =
    [5], and row index 5 of nwbfile.electrodes has ID 10, then calling
    get_electrode_indices(electricalseries, 10) will return 0, the index of the
    matching electrode in electricalseries.electrodes.

    Indices for electrode_ids that are not in the electrical series are
    returned as np.nan

    If an NWBFile is given, then the row indices with the matching IDs in the
    file's electrodes table are returned.

    Parameters
    ----------
    nwb_object : pynwb.NWBFile or pynwb.ecephys.ElectricalSeries
        The NWB file object or NWB electrical series object.
    electrode_ids : np.ndarray or list
        Array or list of electrode IDs.

    Returns
    -------
    electrode_indices : list
        Array of indices of the specified electrode IDs.
    """
    if isinstance(nwb_object, pynwb.ecephys.ElectricalSeries):
        # electrodes is a DynamicTableRegion which may contain a subset of the
        # rows in NWBFile.electrodes match against only the subset of
        # electrodes referenced by this ElectricalSeries
        electrode_table_indices = nwb_object.electrodes.data[:]
        selected_elect_ids = [
            nwb_object.electrodes.table.id[x] for x in electrode_table_indices
        ]
    elif isinstance(nwb_object, pynwb.NWBFile):
        # electrodes is a DynamicTable that contains all electrodes
        selected_elect_ids = list(nwb_object.electrodes.id[:])
    else:
        raise ValueError(
            "nwb_object must be of type ElectricalSeries or NWBFile"
        )

    # for each electrode_id, find its index in selected_elect_ids and return
    # that if it's there and invalid_electrode_index if not.

    return [
        selected_elect_ids.index(elect_id)
        if elect_id in selected_elect_ids
        else invalid_electrode_index
        for elect_id in electrode_ids
    ]


def _get_epoch_groups(position: pynwb.behavior.Position):
    epoch_start_time = {}
    for pos_epoch, spatial_series in enumerate(
        position.spatial_series.values()
    ):
        epoch_start_time[pos_epoch] = spatial_series.timestamps[0]

    return {
        i: [j[0] for j in j]
        for i, j in groupby(
            sorted(epoch_start_time.items(), key=lambda x: x[1]), lambda x: x[1]
        )
    }


def _get_pos_dict(
    position: dict,
    epoch_groups: dict,
    session_id: str = None,
    verbose: bool = False,
    incl_times: bool = True,
):
    """Return dict with obj ids and valid intervals for each epoch.

    Parameters
    ----------
    position : hdmf.utils.LabeledDict
        pynwb.behavior.Position.spatial_series
    epoch_groups : dict
        Epoch start times as keys, spatial series indices as values
    session_id : str, optional
        Optional session ID for verbose print during sampling rate estimation
    verbose : bool, optional
        Default to False. Print estimated sampling rate
    incl_times : bool, optional
        Default to True. Include valid intervals. Requires additional
        computation not needed for RawPosition
    """
    # prev, this was just a list. now, we need to gen mult dict per epoch
    pos_data_dict = dict()
    all_spatial_series = list(position.values())

    for epoch, index_list in enumerate(epoch_groups.values()):
        pos_data_dict[epoch] = []
        for index in index_list:
            spatial_series = all_spatial_series[index]
            valid_times = None
            if incl_times:  # get the valid intervals for the position data
                timestamps = np.asarray(spatial_series.timestamps)
                sampling_rate = estimate_sampling_rate(
                    timestamps, verbose=verbose, filename=session_id
                )
                valid_times = get_valid_intervals(
                    timestamps=timestamps,
                    sampling_rate=sampling_rate,
                    min_valid_len=int(sampling_rate),
                )
            # add the valid intervals to the Interval list
            pos_data_dict[epoch].append(
                {
                    "valid_times": valid_times,
                    "raw_position_object_id": spatial_series.object_id,
                    "name": spatial_series.name,
                }
            )

    return pos_data_dict


def get_all_spatial_series(nwbf, verbose=False, incl_times=True) -> dict:
    """
    Given an NWB, get the spatial series and return a dictionary by epoch.

    If incl_times is True, then the valid intervals are included in the output.

    Parameters
    ----------
    nwbf : pynwb.NWBFile
        The source NWB file object.
    verbose : bool
        Flag representing whether to print the sampling rate.
    incl_times : bool
        Include valid times in the output. Default, True. Set to False for only
        spatial series object IDs.

    Returns
    -------
    pos_data_dict : dict
        Dict mapping indices to a dict with keys 'valid_times' and
        'raw_position_object_id'. Returns None if there is no position data in
        the file. The 'raw_position_object_id' is the object ID of the
        SpatialSeries object.
    """
    pos_interface = get_data_interface(
        nwbf, "position", pynwb.behavior.Position
    )

    if pos_interface is None:
        return None

    return _get_pos_dict(
        position=pos_interface.spatial_series,
        epoch_groups=_get_epoch_groups(pos_interface),
        session_id=nwbf.session_id,
        verbose=verbose,
        incl_times=incl_times,
    )


def get_nwb_copy_filename(nwb_file_name):
    """Get file name of copy of nwb file without the electrophys data"""

    filename, file_extension = os.path.splitext(nwb_file_name)

    if filename.endswith("_"):
        print(f"WARNING: File may already be a copy: {nwb_file_name}")

    return f"{filename}_{file_extension}"


def change_group_permissions(
    subject_ids, set_group_name, analysis_dir="/stelmo/nwb/analysis"
):
    # Change to directory with analysis nwb files
    os.chdir(analysis_dir)
    # Get nwb file directories with specified subject ids
    target_contents = [
        x
        for x in os.listdir(analysis_dir)
        if any([subject_id in x.split("_")[0] for subject_id in subject_ids])
    ]
    # Loop through nwb file directories and change group permissions
    for target_content in target_contents:
        print(
            f"For {target_content}, changing group to {set_group_name} "
            + "and giving read/write/execute permissions"
        )
        # Change group
        os.system(f"chgrp -R {set_group_name} {target_content}")
        # Give read, write, execute permissions to group
        os.system(f"chmod -R g+rwx {target_content}")
