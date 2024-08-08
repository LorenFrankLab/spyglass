import warnings
from typing import Dict, List

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
    sg_keys = list()
    sge_keys = list()
    for e_group in e_groups:
        sg_key = dict()
        sge_key = dict()
        sg_key["nwb_file_name"] = sge_key["nwb_file_name"] = nwb_file_name
        # for each electrode group, get a list of the unique shank numbers
        shank_list = np.unique(
            electrodes["probe_shank"][
                electrodes["electrode_group_name"] == e_group
            ]
        )
        sge_key["electrode_group_name"] = e_group
        # get the indices of all electrodes in this group / shank and set their sorting group
        for shank in shank_list:
            sg_key["sort_group_id"] = sge_key["sort_group_id"] = sort_group
            # specify reference electrode. Use 'references' if passed, otherwise use reference from config
            if not references:
                shank_elect_ref = electrodes["original_reference_electrode"][
                    np.logical_and(
                        electrodes["electrode_group_name"] == e_group,
                        electrodes["probe_shank"] == shank,
                    )
                ]
                if np.max(shank_elect_ref) == np.min(shank_elect_ref):
                    sg_key["sort_reference_electrode_id"] = shank_elect_ref[0]
                else:
                    ValueError(
                        f"Error in electrode group {e_group}: reference "
                        + "electrodes are not all the same"
                    )
            else:
                if e_group not in references.keys():
                    raise Exception(
                        f"electrode group {e_group} not a key in "
                        + "references, so cannot set reference"
                    )
                else:
                    sg_key["sort_reference_electrode_id"] = references[e_group]
            # Insert sort group and sort group electrodes
            reference_electrode_group = electrodes[
                electrodes["electrode_id"]
                == sg_key["sort_reference_electrode_id"]
            ][
                "electrode_group_name"
            ]  # reference for this electrode group
            if len(reference_electrode_group) == 1:  # unpack single reference
                reference_electrode_group = reference_electrode_group[0]
            elif (int(sg_key["sort_reference_electrode_id"]) > 0) and (
                len(reference_electrode_group) != 1
            ):
                raise Exception(
                    "Should have found exactly one electrode group for "
                    + "reference electrode, but found "
                    + f"{len(reference_electrode_group)}."
                )
            if omit_ref_electrode_group and (
                str(e_group) == str(reference_electrode_group)
            ):
                logger.warn(
                    f"Omitting electrode group {e_group} from sort groups "
                    + "because contains reference."
                )
                continue
            shank_elect = electrodes["electrode_id"][
                np.logical_and(
                    electrodes["electrode_group_name"] == e_group,
                    electrodes["probe_shank"] == shank,
                )
            ]
            if (
                omit_unitrode and len(shank_elect) == 1
            ):  # omit unitrodes if indicated
                logger.warn(
                    f"Omitting electrode group {e_group}, shank {shank} "
                    + "from sort groups because unitrode."
                )
                continue
            sg_keys.append(sg_key)
            for elect in shank_elect:
                sge_key["electrode_id"] = elect
                sge_keys.append(sge_key.copy())
            sort_group += 1


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
        "amplitude_thresh_uV": amplitude_thresh_uV,
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

    # find the artifact occurrences using one or both thresholds, across channels
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
