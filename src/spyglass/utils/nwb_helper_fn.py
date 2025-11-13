"""NWB helper functions for finding processing modules and data interfaces."""

import os
import os.path
from itertools import groupby
from pathlib import Path
from typing import List

import numpy as np
import pynwb
import yaml

from spyglass.utils.logging import logger

# dict mapping file path to an open NWBHDF5IO object in read mode and its NWBFile
__open_nwb_files = dict()

# dict mapping NWB file path to config after it is loaded once
__configs = dict()

global invalid_electrode_index
invalid_electrode_index = 99999999


def _open_nwb_file(nwb_file_path, source="local"):
    """Open an NWB file, add to cache, return contents. Does not close file."""
    if source == "local":
        io = pynwb.NWBHDF5IO(path=nwb_file_path, mode="r", load_namespaces=True)
        nwbfile = io.read()
    elif source == "dandi":
        from ..common.common_dandi import DandiPath

        io, nwbfile = DandiPath().fetch_file_from_dandi(
            nwb_file_path=nwb_file_path
        )
    else:
        raise ValueError(f"Invalid open_nwb source: {source}")
    __open_nwb_files[nwb_file_path] = (io, nwbfile)
    return nwbfile


def get_nwb_file(nwb_file_path, query_expression=None):
    """Return an NWBFile object with the given file path in read mode.

    If the file is not found locally, this will check if it has been shared
    with kachery/dandi and if so, download it and open it. If not, and the
    query_expression has a `_make_file` method, it will call that method to
    recompute the file.

    Parameters
    ----------
    nwb_file_path : str
        Path to the NWB file or NWB file name. If it does not start with a "/",
        get path with Nwbfile.get_abs_path
    query_expression : dj.QueryExpression, optional
        The restricted table associated with the NWB file, used for recompute.

    Returns
    -------
    nwbfile : pynwb.NWBFile
        NWB file object for the given path opened in read mode.

    Raises
    ------
    FileNotFoundError
        If the NWB file is not found locally or in kachery/Dandi, and cannot be
        recomputed.
    """
    if not Path(nwb_file_path).is_absolute():
        from spyglass.common import Nwbfile

        nwb_file_path = Nwbfile.get_abs_path(nwb_file_path)

    _, nwbfile = __open_nwb_files.get(nwb_file_path, (None, None))

    if nwbfile is not None:
        return nwbfile

    if os.path.exists(nwb_file_path):
        return _open_nwb_file(nwb_file_path)

    logger.info(
        f"NWB file not found locally; checking kachery for {nwb_file_path}"
    )

    from ..sharing.sharing_kachery import AnalysisNwbfileKachery

    kachery_success = AnalysisNwbfileKachery.download_file(
        os.path.basename(nwb_file_path), permit_fail=True
    )
    if kachery_success:
        return _open_nwb_file(nwb_file_path)

    logger.info(
        "NWB file not found in kachery; checking Dandi for "
        + f"{nwb_file_path}"
    )

    # Dandi fallback SB 2024-04-03
    from ..common.common_dandi import DandiPath

    if DandiPath().has_file_path(file_path=nwb_file_path):
        return _open_nwb_file(nwb_file_path, source="dandi")

    if DandiPath().has_raw_path(file_path=nwb_file_path):
        raw = DandiPath().get_raw_path(file_path=nwb_file_path)["filename"]
        return _open_nwb_file(raw, source="dandi")

    if hasattr(query_expression, "_make_file"):
        # if the query_expression has a _make_file method, call it to
        # recompute the file
        basename = os.path.basename(nwb_file_path)
        logger.info(f"Recomputing {basename}")

        nwbfile = query_expression._make_file(recompute_file_name=basename)
        if nwbfile is not None:
            return nwbfile

    raise FileNotFoundError(
        "NWB file not found in kachery or Dandi: "
        + f"{os.path.basename(nwb_file_path)}."
    )


def file_from_dandi(filepath):
    """helper to determine if open file is streamed from Dandi"""
    if filepath not in __open_nwb_files:
        return False
    build_keys = __open_nwb_files[filepath][0]._HDF5IO__built.keys()
    for k in build_keys:
        if "HTTPFileSystem" in k:
            return True
    return False


def get_linked_nwbs(path: str) -> List[str]:
    """Return a paths linked in the given NWB file.

    Given a NWB file path, open & read the file to find any linked NWB objects.
    """
    with pynwb.NWBHDF5IO(path, "r") as io:
        # open the nwb file (opens externally linked files as well)
        _ = io.read()
        # get the linked files
        return [x for x in io._HDF5IO__built if x != path]


def get_config(nwb_file_path: str, calling_table: str = None) -> dict:
    """Return a dictionary of config settings for the given NWB file.
    If the file does not exist, return an empty dict.

    Parameters
    ----------
    nwb_file_path : str
        Absolute path to the NWB file.
    calling_table : str, optional
        The name of the table that is calling this function. Used for
        logging purposes.

    Returns
    -------
    dict
        Dictionary of configuration settings loaded from the YAML file
    """
    if nwb_file_path in __configs:  # load from cache if exists
        return __configs[nwb_file_path]

    obj_path = Path(nwb_file_path)
    # NOTE use stem[:-1] to remove the underscore that was added to the file
    config_path = obj_path.parent / (
        obj_path.stem[:-1] + "_spyglass_config.yaml"
    )
    if not os.path.exists(config_path):
        from spyglass.settings import base_dir  # noqa: F401

        rel_path = obj_path.relative_to(base_dir)
        table = f"{calling_table}: " if calling_table else ""
        logger.info(f"{table}No config found at {rel_path}")
        ret = dict()
        __configs[nwb_file_path] = ret  # cache to avoid repeated null lookups
        return ret
    with open(config_path, "r") as stream:
        config_dict = yaml.safe_load(stream)

    # TODO write a JSON schema for the yaml file and validate the yaml file
    __configs[nwb_file_path] = config_dict  # store in cache
    return config_dict


def close_nwb_files():
    """Close all open NWB files."""
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
    LoggerWarning
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
        logger.warning(
            f"Multiple data interfaces with name '{data_interface_name}' "
            f"found in NWBFile with identifier {nwbfile.identifier}. "
            + "Using the first one found. "
            "Use the data_interface_class argument to restrict the search."
        )
    if len(ret) >= 1:
        return ret[0]

    return None


def get_position_obj(nwbfile):
    """Return the Position object from the behavior processing module.
    Meant to find position spatial series that are not found by
    `get_data_interface(nwbfile, 'position', pynwb.behavior.Position)`.
    The code returns the first `pynwb.behavior.Position` object (technically
    there should only be one).

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file object.

    Returns
    -------
    pynwb.behavior.Position object
    """
    ret = []
    for obj in nwbfile.processing["behavior"].data_interfaces.values():
        if isinstance(obj, pynwb.behavior.Position):
            ret.append(obj)
    if len(ret) > 1:
        raise ValueError(f"Found more than one position object in {nwbfile}")
    return ret[0] if ret and len(ret) else None


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
        Log sampling rate to stdout. Default, False
    filename : str, optional
        Filename to reference when logging or err. Default, "file"

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
        np.log10(smooth_diff),
        bins=100,
        range=[-5, np.log10(3 * np.mean(smooth_diff))],
    )
    mode = 10 ** bins[np.where(hist == np.max(hist))][0]

    adjacent = sample_diff < mode * multiplier

    sampling_rate = np.round(1.0 / np.mean(sample_diff[adjacent]))

    if sampling_rate < 0:
        raise ValueError(f"Error calculating sampling rate. For {filename}")
    if verbose:
        logger.info(
            f"Estimated sampling rate for {filename}: {sampling_rate} Hz"
        )

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
        than interval duration, log warning and use half the total time.

    Returns
    -------
    valid_times : np.ndarray
        Array of start and stop times of shape (N, 2) for valid data.
    """

    eps = 0.0000001

    total_time = timestamps[-1] - timestamps[0]

    if total_time < min_valid_len:
        half_total_time = total_time / 2
        logger.warning(
            f"Setting minimum valid interval to {half_total_time:.4f}"
        )
        min_valid_len = half_total_time

    # get rid of NaN elements
    timestamps = timestamps[~np.isnan(timestamps)]
    # find gaps
    gap = np.diff(timestamps) > 1.0 / sampling_rate * gap_proportion

    # all true entries of gap represent gaps.
    # Get the times bounding these intervals.
    gapind = np.asarray(np.where(gap))
    # The end of each valid interval are the indices of gaps and the final value
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
        (
            selected_elect_ids.index(elect_id)
            if elect_id in selected_elect_ids
            else invalid_electrode_index
        )
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
        Optional session ID for verbose log during sampling rate estimation
    verbose : bool, optional
        Default to False. Log estimated sampling rate
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
        Flag representing whether to log the sampling rate.
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
    pos_interface = get_position_obj(nwbf)

    if pos_interface is None:
        return None

    pos_dict = _get_pos_dict(
        position=pos_interface.spatial_series,
        epoch_groups=_get_epoch_groups(pos_interface),
        session_id=nwbf.session_id,
        verbose=verbose,
        incl_times=incl_times,
    )
    if len(pos_dict) == 0:
        return None
    return pos_dict


def get_nwb_copy_filename(nwb_file_name):
    """Get file name of copy of nwb file without the electrophys data"""

    filename, file_extension = os.path.splitext(nwb_file_name)

    if filename.endswith("_"):
        logger.warning(f"File may already be a copy: {nwb_file_name}")

    return f"{filename}_{file_extension}"


def change_group_permissions(
    subject_ids, set_group_name, analysis_dir="/stelmo/nwb/analysis"
):
    """Change group permissions for specified subject ids in analysis dir."""
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("change_group_permissions")

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
        logger.info(
            f"For {target_content}, changing group to {set_group_name} "
            + "and giving read/write/execute permissions"
        )
        # Change group
        os.system(f"chgrp -R {set_group_name} {target_content}")
        # Give read, write, execute permissions to group
        os.system(f"chmod -R g+rwx {target_content}")
