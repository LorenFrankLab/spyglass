import warnings
from os import environ as os_environ
from typing import Dict, List

import datajoint as dj
import numpy as np
import scipy.stats as stats
import spikeinterface as si

from spyglass.common.common_ephys import Electrode
from spyglass.utils import logger


def get_group_by_shank(
    nwb_file_name: str,
    references: dict = None,
    omit_ref_electrode_group=False,
    omit_unitrode=True,
):
    """Divides electrodes into groups based on their shank position.

    * Electrodes from probes with 1 shank (e.g. tetrodes) are placed in a
      single group
    * Electrodes from probes with multiple shanks (e.g. polymer probes) are
      placed in one group per shank
    * Bad channels are omitted

    Parameters
    ----------
    nwb_file_name : str
        the name of the NWB file whose electrodes should be put into
        sorting groups
    references : dict, optional
        If passed, used to set references. Otherwise, references set using
        original reference electrodes from config. Keys: electrode groups.
        Values: reference electrode.
    omit_ref_electrode_group : bool
        Optional. If True, no sort group is defined for electrode group of
        reference.
    omit_unitrode : bool
        Optional. If True, no sort groups are defined for unitrodes.
    """
    # get the electrodes from this NWB file
    electrodes = (
        Electrode()
        & {"nwb_file_name": nwb_file_name}
        & {"bad_channel": "False"}
    ).fetch()

    e_groups = list(np.unique(electrodes["electrode_group_name"]))
    e_groups.sort(key=int)  # sort electrode groups numerically

    sort_group = 0
    sg_keys, sge_keys = list(), list()
    for e_group in e_groups:
        sg_key, sge_key = dict(), dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name

        # for each electrode group, get a list of the unique shank numbers
        shank_list = np.unique(
            electrodes["probe_shank"][
                electrodes["electrode_group_name"] == e_group
            ]
        )
        sge_key["electrode_group_name"] = e_group

        # get the indices of all electrodes in this group / shank and set their
        # sorting group
        for shank in shank_list:
            sg_key["sort_group_id"] = sge_key["sort_group_id"] = sort_group

            match_names_bool = np.logical_and(
                electrodes["electrode_group_name"] == e_group,
                electrodes["probe_shank"] == shank,
            )

            if references:  # Use 'references' if passed
                sort_ref_id = references.get(e_group, None)
                if not sort_ref_id:
                    raise Exception(
                        f"electrode group {e_group} not a key in "
                        + "references, so cannot set reference"
                    )
            else:  # otherwise use reference from config
                shank_elect_ref = electrodes["original_reference_electrode"][
                    match_names_bool
                ]
                if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                    sort_ref_id = shank_elect_ref[0]
                else:
                    ValueError(
                        f"Error in electrode group {e_group}: reference "
                        + "electrodes are not all the same"
                    )
            sg_key["sort_reference_electrode_id"] = sort_ref_id

            # Insert sort group and sort group electrodes
            match_elec = electrodes[electrodes["electrode_id"] == sort_ref_id]
            ref_elec_group = match_elec["electrode_group_name"]  # group ref

            n_ref_groups = len(ref_elec_group)
            if n_ref_groups == 1:  # unpack single reference
                ref_elec_group = ref_elec_group[0]
            elif int(sort_ref_id) > 0:  # multiple references
                raise Exception(
                    "Should have found exactly one electrode group for "
                    + f"reference electrode, but found {n_ref_groups}."
                )

            if omit_ref_electrode_group and (
                str(e_group) == str(ref_elec_group)
            ):
                logger.warning(
                    f"Omitting electrode group {e_group} from sort groups "
                    + "because contains reference."
                )
                continue
            shank_elect = electrodes["electrode_id"][match_names_bool]

            # omit unitrodes if indicated
            if omit_unitrode and len(shank_elect) == 1:
                logger.warning(
                    f"Omitting electrode group {e_group}, shank {shank} "
                    + "from sort groups because unitrode."
                )
                continue

            sg_keys.append(sg_key.copy())
            sge_keys.extend(
                [
                    {
                        **sge_key,
                        "electrode_id": elect,
                    }
                    for elect in shank_elect
                ]
            )
            sort_group += 1
    return sg_keys, sge_keys


def _init_artifact_worker(
    recording,
    zscore_thresh=None,
    amplitude_thresh_uV=None,
    proportion_above_thresh=1.0,
):
    recording = (
        si.load_extractor(recording)
        if isinstance(recording, dict)
        else recording
    )
    # create a local dict per worker
    return {
        "recording": recording,
        "zscore_thresh": zscore_thresh,
        "amplitude_thresh": amplitude_thresh_uV,
        "proportion_above_thresh": proportion_above_thresh,
    }


def _compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    zscore_thresh = worker_ctx["zscore_thresh"]
    amplitude_thresh = worker_ctx["amplitude_thresh"]
    proportion_above_thresh = worker_ctx["proportion_above_thresh"]
    # compute the number of electrodes that have to be above threshold
    nelect_above = np.ceil(
        proportion_above_thresh * len(recording.get_channel_ids())
    )

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # find artifact occurrences using one or both thresholds, across channels
    if (amplitude_thresh is not None) and (zscore_thresh is None):
        above_a = np.abs(traces) > amplitude_thresh
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_a, axis=1) >= nelect_above))
            + start_frame
        )
    elif (amplitude_thresh is None) and (zscore_thresh is not None):
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_z, axis=1) >= nelect_above))
            + start_frame
        )
    else:
        above_a = np.abs(traces) > amplitude_thresh
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(
                np.argwhere(
                    np.sum(np.logical_or(above_z, above_a), axis=1)
                    >= nelect_above
                )
            )
            + start_frame
        )

    return above_thresh


def _check_artifact_thresholds(
    amplitude_thresh, zscore_thresh, proportion_above_thresh
):
    """Alerts user to likely unintended parameters. Not an exhaustive verification.

    Parameters
    ----------
    zscore_thresh: float
    amplitude_thresh: float
        Measured in microvolts.
    proportion_above_thresh: float

    Return
    ------
    zscore_thresh: float
    amplitude_thresh: float
    proportion_above_thresh: float

    Raise
    ------
    ValueError: if signal thresholds are negative
    """
    # amplitude or zscore thresholds should be negative, as they are applied to
    # an absolute signal

    signal_thresholds = [
        t for t in [amplitude_thresh, zscore_thresh] if t is not None
    ]
    for t in signal_thresholds:
        if t < 0:
            raise ValueError(
                "Amplitude and Z-Score thresholds must be >= 0, or None"
            )

    # proportion_above_threshold should be in [0:1] inclusive
    if proportion_above_thresh < 0:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1."
            " Using proportion_above_thresh = 0.01 instead of "
            + f"{str(proportion_above_thresh)}"
        )
        proportion_above_thresh = 0.01
    elif proportion_above_thresh > 1:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1. "
            "Using proportion_above_thresh = 1 instead of "
            + f"{str(proportion_above_thresh)}"
        )
        proportion_above_thresh = 1
    return amplitude_thresh, zscore_thresh, proportion_above_thresh


def _get_recording_timestamps(recording):
    num_segments = recording.get_num_segments()

    if num_segments <= 1:
        return recording.get_times()

    frames_per_segment = [0] + [
        recording.get_num_frames(segment_index=i) for i in range(num_segments)
    ]

    cumsum_frames = np.cumsum(frames_per_segment)
    total_frames = np.sum(frames_per_segment)

    timestamps = np.zeros((total_frames,))
    for i in range(num_segments):
        start_index = cumsum_frames[i]
        end_index = cumsum_frames[i + 1]
        timestamps[start_index:end_index] = recording.get_times(segment_index=i)

    return timestamps


def _reformat_metrics(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    return [
        {
            "name": metric_name,
            "label": metric_name,
            "tooltip": metric_name,
            "data": {
                str(unit_id): metric_value
                for unit_id, metric_value in metric.items()
            },
        }
        for metric_name, metric in metrics.items()
    ]
