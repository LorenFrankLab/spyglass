from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from datajoint.expression import QueryExpression

from spyglass.utils import logger


def validate_pairs(
    query: QueryExpression, pairs: Union[Tuple, List[Tuple]]
) -> List[Tuple]:
    """Validate unit pairs for a given query.

    Checks that the passed pairs are valid pairs in the query. May flip the
    order of the pairs if the reverse pair is found. Query is assumed to have
    columns unit1, unit2.
    """

    if isinstance(pairs, tuple) and len(pairs) == 2:
        pairs = [pairs]

    # set[tuple[int, int]]
    query_pairs = set(zip(*query.fetch("unit1", "unit2")))

    def validate_pair(unit1: int, unit2: int) -> Tuple[int, int]:
        """Ensure that unit1, unit2 is a valid pair in the table."""
        if (unit1, unit2) in query_pairs:
            return unit1, unit2
        elif (unit2, unit1) in query_pairs:
            logger.warning(f"Using reverse pair {unit1}, {unit2}")
            return unit2, unit1
        else:
            logger.warning(f"No entry found for pair {unit1}, {unit2}")
            return None

    valid_pairs = []
    for p in pairs:
        if valid_pair := validate_pair(*p):
            valid_pairs.append(valid_pair)

    if not valid_pairs:
        raise ValueError("No valid pairs found")
    return valid_pairs


def plot_burst_xcorrel(
    pairs: List[Tuple[int, int]],
    ccgs_e: np.ndarray,
    bins: np.ndarray,
) -> plt.Figure:
    """Plot cross-correlograms for a list of unit pairs.

    Parameters
    ----------
    pairs : list of tuples of int
        pairs of units to investigate
    ccgs_e : np.array
        Correlograms with shape (num_units, num_units, num_bins)
        The diagonal of ccgs is the auto correlogram.
        ccgs[A, B, :] is the symetrie of ccgs[B, A, :]
        ccgs[A, B, :] have to be read as the histogram of
            spiketimesA - spiketimesB
    bins :  np.array
        The bin edges in ms

    Returns
    -------
    plt.Figure
    """
    col_num = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(2, col_num, figsize=(col_num * 3, 4), squeeze=True)

    for ind, p in enumerate(pairs):
        (u1, u2) = p
        axes[np.unravel_index(ind, axes.shape)].bar(
            bins[1:], ccgs_e[u1 - 1, u2 - 1, :]
        )
        axes[np.unravel_index(ind, axes.shape)].set_xlabel("ms")

    if len(pairs) < col_num * 2:  # remove the last unused axis
        axes[np.unravel_index(ind, axes.shape)].axis("off")

    plt.tight_layout()

    return fig


def plot_burst_pair_peaks(
    pairs: List[Tuple[int, int]], peak_amps: Dict[int, np.ndarray]
):
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
        for i in range(min(peak1.shape[1], peak2.shape[1])):
            data1, data2 = peak1[:, i], peak2[:, i]
            axes[ind, i].hist(data1, **get_kwargs(u1, data1))
            axes[ind, i].hist(data2, **get_kwargs(u2, data2))
            axes[ind, i].set_title("channel " + str(i + 1))
            axes[ind, i].set_xlabel("uV")
            axes[ind, i].legend()

    plt.tight_layout()

    return fig


def plot_burst_peak_over_time(
    peak_v: Dict[int, np.ndarray],
    peak_t: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int]],
    overlap: bool = True,
):
    """Plot peak amplitudes over time for a given key.

    Parameters
    ----------
    peak_v : dict of int to np.ndarray
        peak amplitudes for each unit
    peak_t : dict of int to np.ndarray
        peak timestamps for each unit
    pairs : list of tuples of int
        pairs of units to plot
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
    ) -> Tuple[plt.Figure, plt.Axes]:

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

    return fig


def plot_burst_by_sort_group(
    fig: Dict[int, plt.Figure],
) -> Dict[int, plt.Figure]:
    for sg_id, f in fig.items():
        managed_fig, _ = plt.subplots(
            1, 4, figsize=(24, 4), sharex=False, sharey=False
        )
        canvas_manager = managed_fig.canvas.manager
        canvas_manager.canvas.figure = f
        managed_fig.set_canvas(canvas_manager.canvas)
        plt.suptitle(f"sort group {sg_id}", fontsize=20)
    return fig


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
