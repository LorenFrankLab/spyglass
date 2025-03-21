import warnings
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import spikeinterface as si
from datajoint.expression import QueryExpression

from spyglass.common.common_ephys import Electrode
from spyglass.utils import logger


def validate_pairs(
    query: QueryExpression, pairs: Union[Tuple, List[Tuple]]
) -> List[Tuple]:
    """Validate unit pairs for a given query.

    Checks that the passed pairs are valid pairs in the query. May flip the
    order of the pairs if the reverse pair is found. Query is assumed to have
    columns unit1, unit2.
    """

    def validate_pair(
        query: QueryExpression, unit1: int, unit2: int
    ) -> Tuple[int, int]:
        """Ensure that unit1, unit2 is a valid pair in the table."""
        if query & f"unit1={unit2} AND unit2={unit1}":
            return unit1, unit2
        elif query & f"unit1={unit1} AND unit2={unit2}":
            logger.warning(f"Using reverse pair {unit1}, {unit2}")
            return unit2, unit1
        else:
            logger.warning(f"No entry found for pair {unit1}, {unit2}")
            return None

    if isinstance(pairs, tuple) and len(pairs) == 2:
        pairs = [pairs]

    valid_pairs = []
    for p in pairs:
        if valid_pair := validate_pair(query, *p):
            valid_pairs.append(valid_pair)

    if not valid_pairs:
        raise ValueError("No valid pairs found")
    return valid_pairs


def plot_burst_xcorrel(fig, pairs, ccgs_e, bins):
    """Plot cross-correlograms for a list of unit pairs."""
    col_num = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(
        2,
        int(np.ceil(len(pairs) / 2)),
        figsize=(col_num * 3, 4),
        squeeze=True,
    )

    for ind, p in enumerate(pairs):
        (u1, u2) = p
        axes[np.unravel_index(ind, axes.shape)].bar(
            bins[1:], ccgs_e[u1 - 1, u2 - 1, :]
        )
        axes[np.unravel_index(ind, axes.shape)].set_xlabel("ms")

    if len(pairs) < col_num * 2:  # remove the last unused axis
        axes[np.unravel_index(ind, axes.shape)].axis("off")

    plt.tight_layout()


def plot_burst_pair_peaks(pairs, peak_amps, peak_timestamps):
    """Plot peak amplitudes and timestamps for a list of unit pairs."""
    fig, axes = plt.subplots(len(pairs), 4, figsize=(12, 2 * len(pairs)))

    def get_kwargs(unit, data):
        return dict(
            alpha=0.5,
            weights=np.ones(len(data)) / len(data),
            label=str(unit),
        )

    for ind, (u1, u2) in enumerate(pairs):

        peak1 = peak_amps[u1]
        peak2 = peak_amps[u2]

        axes[ind, 0].set_ylabel("percent")
        for i in range(4):
            data1, data2 = peak1[:, i], peak2[:, i]
            axes[ind, i].hist(data1, **get_kwargs(u1, data1))
            axes[ind, i].hist(data2, **get_kwargs(u2, data2))
            axes[ind, i].set_title("channel " + str(i + 1))
            axes[ind, i].set_xlabel("uV")
            axes[ind, i].legend()

    plt.tight_layout()


def plot_burst_peak_over_time(
    peak_v: Dict[int, np.ndarray],
    peak_t: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int]],
    overlap: bool = True,
):
    """Plot peak amplitudes over time for a given key.

    Parameters
    ----------
    key : dict
        key of BurstPair
    to_investigate_pairs : list of tuples of int
        pairs of units to investigate
    overlap : bool, optional
        if True, plot units in pair on the same plot
    """

    def select_voltages(max_channel, voltages, sub_ind):
        if len(sub_ind) > len(voltages):
            sub_ind = sub_ind[: len(voltages)]
            logger.warning("Timestamp index out of bounds, truncating")
        return voltages[sub_ind, max_channel]

    def plot_burst_one_peak(
        voltages,
        timestamps,
        fig: plt.Figure = None,
        axes: plt.Axes = None,
        row_duration: int = 600,
        show_plot: bool = False,
    ):

        max_channel = np.argmax(-np.mean(voltages, 0))
        time_since = timestamps - timestamps[0]
        row_num = int(np.ceil(time_since[-1] / row_duration))

        if axes is None:
            fig, axes = plt.subplots(
                row_num,
                1,
                figsize=(20, 2 * row_num),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

        for ind in range(row_num):
            t0 = ind * row_duration
            t1 = t0 + row_duration
            sub_ind = np.logical_and(time_since >= t0, time_since <= t1)
            axes[ind, 0].scatter(
                time_since[sub_ind] - t0,
                select_voltages(max_channel, voltages, sub_ind),
            )

        if not show_plot:
            plt.close()

        return fig, axes

    for pair in pairs:
        kwargs = (
            dict(fig=None, axes=None, row_duration=100) if overlap else dict()
        )

        for u in pair:
            ret1, ret2 = plot_burst_one_peak(
                peak_v[u], peak_t[u], show_plot=overlap, **kwargs
            )
            if overlap:
                fig, axes = ret1, ret2
                kwargs = dict(fig=fig, axes=axes)
            else:
                if fig is None:
                    fig, axes = dict(), dict()
                fig[u], axes[u] = ret1, ret2

        axes[0, 0].set_title(f"pair:{pair}")


def plot_burst_by_sort_group(fig):
    for sg_id, f in fig.items():
        managed_fig, _ = plt.subplots(
            1, 4, figsize=(24, 4), sharex=False, sharey=False
        )
        canvas_manager = managed_fig.canvas.manager
        canvas_manager.canvas.figure = f
        managed_fig.set_canvas(canvas_manager.canvas)
        plt.suptitle(f"sort group {sg_id}", fontsize=20)


def plot_burst_metrics(
    sg_query: Union[List[Dict], QueryExpression],
) -> plt.Figure:
    """Parameters are 4 metrics to be plotted against each other.

    Parameters
    ----------
    sg_query : Union[List[Dict], QueryExpression]
        Query result or list of dictionaries with the following keys:
        wf_similarity : dict
            waveform similarities
        isi_violation : dict
            isi violation
        xcorrel_asymm : dict
            spike cross correlogram asymmetry

    Returns
    -------
    figure for plotting later
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for color_ind, row in enumerate(sg_query):
        color = dict(color=f"C{color_ind}")
        wf = row["wf_similarity"]
        ca = row["xcorrel_asymm"]
        ax.scatter(wf, ca, **color)
        ax.text(wf, ca, f"({row['unit1']},{row['unit2']})", **color)

    ax.set_xlabel("waveform similarity")
    ax.set_ylabel("cross-correlogram asymmetry")

    plt.close()

    return fig


def calculate_ca(bins: np.ndarray, correl: np.ndarray) -> float:
    """Calculate Correlogram Asymmetry (CA)

    defined as the contrast ratio of the area of the correlogram right and
    left of coincident activity (zero).
    http://www.psy.vanderbilt.edu/faculty/roeaw/edgeinduction/Fig-W6.htm

    Parameters
    ----------
    bins : np.ndarray
        array of bin edges
    correl : np.ndarray
        array of correlogram values
    """
    if not len(bins) == len(correl):
        raise ValueError("Mismatch in lengths for correl asymmetry")
    right = np.sum(correl[bins > 0])
    left = np.sum(correl[bins < 0])
    return 0 if (right + left) == 0 else (right - left) / (right + left)


def calculate_isi_violation(
    peak1: np.ndarray, peak2: np.ndarray, isi_threshold_s: float = 1.5
) -> float:
    """Calculate isi violation between two spike trains"""
    spike_train = np.sort(np.concatenate((peak1, peak2)))
    isis = np.diff(spike_train)
    num_spikes = len(spike_train)
    num_violations = np.sum(isis < (isi_threshold_s * 1e-3))
    return num_violations / num_spikes


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
