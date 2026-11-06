"""DB-free plotting and analyzer-extraction helpers for ``CurationEvaluation``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Metrics shown by default in the QC histograms, in display order; only those
# present in the metrics table are used.
_DEFAULT_QC_METRICS = (
    "snr",
    "isi_violation",
    "presence_ratio",
    "amplitude_cutoff",
    "firing_rate",
)


def _flatten_axes(axes):
    """Return caller-supplied axes as a flat list."""
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    if isinstance(axes, (list, tuple)):
        return list(np.asarray(axes, dtype=object).ravel())
    return [axes]


def draw_metric_histogram(ax, values, *, title, ylabel="count"):
    """Draw one metric's histogram on ``ax``, dropping non-finite values.

    The dropped count is annotated in the corner.
    """
    finite = values[np.isfinite(values)]
    n_dropped = len(values) - len(finite)
    if len(finite):
        ax.hist(finite, bins=min(20, max(5, len(finite))))
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel(ylabel)
    if n_dropped:
        ax.text(
            0.98,
            0.95,
            f"{n_dropped} NaN dropped",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )


def _empty_qc_axes(axes):
    """Select an axis and render the empty-population state."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, ax = plt.subplots(figsize=(5, 3))
    elif isinstance(axes, dict):
        if not axes:
            raise ValueError(
                "plot_units_qc_figure zero-unit axes mapping must include "
                "'empty' or at least one axis."
            )
        ax = axes.get("empty") or next(iter(axes.values()))
    else:
        supplied = _flatten_axes(axes)
        if not supplied:
            raise ValueError(
                "plot_units_qc_figure zero-unit axes sequence must include "
                "at least one axis."
            )
        ax = supplied[0]
        for extra in supplied[1:]:
            extra.set_axis_off()
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        "No units to display (zero-unit sort).",
        ha="center",
        va="center",
    )
    return {"empty": ax}


def _qc_population_axes(metric_names, axes):
    """Allocate or validate histogram and scatter axes for a population."""
    import matplotlib.pyplot as plt

    n_hist = len(metric_names)
    n_cols = min(3, max(1, n_hist))
    n_hist_rows = math.ceil(n_hist / n_cols) if n_hist else 0
    if axes is None:
        # One extra full-width row for the depth scatter.
        n_rows = n_hist_rows + 1
        fig, axes_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False,
        )
        hist_axes = {
            metric: axes_grid[index // n_cols][index % n_cols]
            for index, metric in enumerate(metric_names)
        }
        # Hide unused histogram cells.
        for index in range(n_hist, n_hist_rows * n_cols):
            axes_grid[index // n_cols][index % n_cols].set_axis_off()

        # Depth scatter spans the bottom row.
        for col in range(n_cols):
            axes_grid[n_rows - 1][col].remove()
        scatter_ax = fig.add_subplot(n_rows, 1, n_rows)
    elif isinstance(axes, dict):
        missing = [metric for metric in metric_names if metric not in axes]
        if "scatter" not in axes:
            missing.append("scatter")
        if missing:
            raise ValueError(
                "plot_units_qc_figure axes mapping is missing required "
                f"key(s): {missing}"
            )
        hist_axes = {metric: axes[metric] for metric in metric_names}
        scatter_ax = axes["scatter"]
        fig = scatter_ax.figure
    else:
        supplied = _flatten_axes(axes)
        needed = n_hist + 1
        if len(supplied) < needed:
            raise ValueError(
                "plot_units_qc_figure needs at least "
                f"{needed} axes ({n_hist} histogram + 1 scatter); got "
                f"{len(supplied)}."
            )
        hist_axes = {
            metric: supplied[index] for index, metric in enumerate(metric_names)
        }
        scatter_ax = supplied[n_hist]
        for extra in supplied[needed:]:
            extra.set_axis_off()
        fig = scatter_ax.figure

    return fig, hist_axes, scatter_ax


def plot_units_qc_figure(
    metrics_df: pd.DataFrame,
    unit_locations: np.ndarray | None,
    unit_ids: list,
    *,
    metric_names: list[str] | None = None,
    color_metric: str = "snr",
    axes=None,
):
    """Render a population QC overview: metric histograms + a depth scatter.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Quality metrics, one row per unit (indexed by unit_id). Values may be
        ``None`` (sanitized non-finite); they are coerced to NaN and dropped
        per-metric from the histograms rather than distorting an axis.
    unit_locations : numpy.ndarray or None
        ``(n_units, 2 or 3)`` estimated unit locations aligned with
        ``unit_ids``; ``None`` for a zero-unit sort.
    unit_ids : list
        Unit ids aligned with ``unit_locations`` rows.
    metric_names : list of str, optional
        Metrics to histogram; defaults to the present subset of
        ``_DEFAULT_QC_METRICS`` (or all numeric columns).
    color_metric : str
        Metric used to color the depth scatter (depth is column 1, y, of
        ``unit_locations``).
    axes : dict, sequence, numpy.ndarray, or matplotlib.axes.Axes, optional
        Axes to draw into. For non-empty sorts, pass either a mapping with one
        axis per metric name plus ``"scatter"``, or a flat/numpy sequence where
        the histogram axes come first and the scatter axis follows. For zero-unit
        sorts, pass a single axis or ``{"empty": ax}``. If omitted, a figure is
        created.

    Returns
    -------
    dict[str, matplotlib.axes.Axes]
        Axes keyed by metric name plus ``"scatter"``. A zero-unit sort returns
        ``{"empty": ax}`` (never raises).
    """
    numeric = (
        metrics_df.apply(pd.to_numeric, errors="coerce")
        if len(metrics_df.columns)
        else metrics_df
    )
    if metric_names is None:
        present = [m for m in _DEFAULT_QC_METRICS if m in numeric.columns]
        metric_names = present or list(numeric.columns)

    has_units = unit_locations is not None and len(unit_ids) > 0

    if not has_units:
        return _empty_qc_axes(axes)

    fig, hist_axes, scatter_ax = _qc_population_axes(metric_names, axes)

    # Histograms: one small-multiple per metric, NaN dropped.
    for metric, ax in hist_axes.items():
        draw_metric_histogram(
            ax, numeric[metric].to_numpy(dtype=float), title=metric
        )

    # Depth scatter spanning the bottom row: each unit at its probe position,
    # colored by the chosen metric (units with a NaN color metric drawn gray).
    locations = np.asarray(unit_locations, dtype=float)
    x = locations[:, 0]
    depth = locations[:, min(1, locations.shape[1] - 1)]
    color_values = (
        numeric.reindex(unit_ids)[color_metric].to_numpy(dtype=float)
        if color_metric in numeric.columns
        else np.full(len(unit_ids), np.nan)
    )
    finite_mask = np.isfinite(color_values)
    if finite_mask.any():
        sc = scatter_ax.scatter(
            x[finite_mask],
            depth[finite_mask],
            c=color_values[finite_mask],
            cmap="viridis",
        )
        fig.colorbar(sc, ax=scatter_ax, label=color_metric)
    if (~finite_mask).any():
        scatter_ax.scatter(
            x[~finite_mask],
            depth[~finite_mask],
            c="lightgray",
            label=f"{color_metric} NaN",
        )
        scatter_ax.legend(loc="best", fontsize=8)
    scatter_ax.set_xlabel("probe x (um)")
    scatter_ax.set_ylabel("depth (um)")
    scatter_ax.set_title(f"unit depth colored by {color_metric}")

    # Caller-supplied axes own their layout.
    if axes is None:
        fig.tight_layout()
    return {**hist_axes, "scatter": scatter_ax}


def correlograms_from_analyzer(
    analyzer, *, window_ms: float = 100.0, bin_ms: float = 5.0
):
    """Return ``(ccgs, bins, unit_ids)`` computed fresh at the given window/bin.

    The stored ``correlograms`` extension uses SI's 50 / 1 ms default, so it
    is not reused or overwritten. Indexed by the analyzer's unit-id order.
    """
    from spikeinterface.postprocessing import compute_correlograms

    ccgs, bins = compute_correlograms(
        analyzer.sorting, window_ms=window_ms, bin_ms=bin_ms
    )
    return np.asarray(ccgs), np.asarray(bins), list(analyzer.unit_ids)


def _plot_correlogram_grid(ccgs, bins, panels, *, n_cols, panel_size):
    """Render a grid of correlogram bar plots.

    ``panels`` is a list of ``(row_index, col_index, title)``; each cell
    plots ``ccgs[row, col, :]``.
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    n_cols = min(n_cols, max(1, n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
    )
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = bins[1] - bins[0]
    for k, (row, col, title) in enumerate(panels):
        ax = axes[k // n_cols][k % n_cols]
        ax.bar(centers, ccgs[row, col, :], width=width)
        ax.set_title(title)
        ax.set_xlabel("ms")
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_axis_off()
    fig.tight_layout()
    return fig


def plot_autocorrelograms_figure(ccgs, bins, ids, unit_ids=None):
    """Render an autocorrelogram (ACG) grid, one panel per unit."""
    show = list(unit_ids) if unit_ids is not None else list(ids)
    index_of = {unit_id: i for i, unit_id in enumerate(ids)}
    panels = [(index_of[u], index_of[u], f"unit {u} ACG") for u in show]
    return _plot_correlogram_grid(
        ccgs, bins, panels, n_cols=4, panel_size=(3, 2)
    )


def plot_pair_correlograms_figure(ccgs, bins, ids, pairs):
    """Render cross-correlograms for unit pairs (indexed by unit id, not 1-based).

    Unit ids may be arbitrary, so pairs map through the analyzer's order.
    """
    index_of = {unit_id: i for i, unit_id in enumerate(ids)}
    panels = [(index_of[a], index_of[b], f"{a} x {b}") for a, b in pairs]
    return _plot_correlogram_grid(
        ccgs, bins, panels, n_cols=3, panel_size=(4, 3)
    )


def peak_amplitudes_from_analyzer(analyzer):
    """Return ``(peak_amps, peak_times)`` per unit from the waveforms ext.

    - ``peak_amps[unit_id]``: ``(n_spikes, n_channels)`` at the ``nbefore``
      alignment sample, not the array center (the window is asymmetric).
      Dense, so burst-pair overlays compare the same contact across units.
    - ``peak_times[unit_id]``: seconds (sample index / fs) of the same
      ``random_spikes`` subset the waveforms hold, not the full train.
    """
    waveforms = analyzer.get_extension("waveforms")
    nbefore = waveforms.nbefore
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    selected = sorting.to_spike_vector()[
        analyzer.get_extension("random_spikes").get_data()
    ]
    unit_index = {unit_id: i for i, unit_id in enumerate(analyzer.unit_ids)}
    peak_amps: dict = {}
    peak_times: dict = {}
    for unit_id in sorting.get_unit_ids():
        wf = waveforms.get_waveforms_one_unit(unit_id, force_dense=True)
        peak_amps[unit_id] = wf[:, nbefore, :]
        subset = selected[selected["unit_index"] == unit_index[unit_id]]
        peak_times[unit_id] = subset["sample_index"].astype(float) / fs
    return peak_amps, peak_times


def burst_pair_metrics_from_analyzer(
    analyzer,
    pairs=None,
    *,
    isi_threshold_ms: float = 2.0,
    window_ms: float = 100.0,
    bin_ms: float = 5.0,
    continuity_spans=None,
):
    """Per-pair burst-merge diagnostics computed on the fly (nothing stored).

    One dict per ordered pair (default: all; explicit pairs are validated):

    - ``wf_similarity``: SI's cosine ``template_similarity``; the auto-merge
      preset uses ``l1``, so this corroborates rather than reproduces it.
    - ``isi_violation``: violation fraction of the merged train (0.4 ms
      dedup), never across a ``continuity_spans`` join.
    - ``xcorrel_asymm``: directional cross-correlogram asymmetry.
    - ``unit_distance``: euclidean distance between ``unit_locations``.

    Pass the unwhitened display analyzer; whitening distorts similarity and
    locations.
    """
    from itertools import permutations

    from spyglass.spikesorting.utils_burst import calculate_ca
    from spyglass.spikesorting.v2._core.signal_math import (
        _MERGE_DEDUP_DELTA_MS,
        _segment_times_at,
    )
    from spyglass.spikesorting.v2._curation.metrics import (
        count_short_isis,
        isi_continuity_boundaries,
        isi_violation_fraction,
    )
    from spyglass.spikesorting.v2._storage.units_nwb import (
        _dedup_merged_spike_times_and_frames,
    )

    for name in ("templates", "template_similarity", "unit_locations"):
        if not analyzer.has_extension(name):
            analyzer.compute(name)

    unit_ids = list(analyzer.unit_ids)
    index_of = {unit_id: i for i, unit_id in enumerate(unit_ids)}
    if pairs is None:
        pairs = list(permutations(unit_ids, 2))
    else:
        pairs = validate_unit_pairs(unit_ids, pairs)

    similarity = np.asarray(
        analyzer.get_extension("template_similarity").get_data()
    )
    locations = np.asarray(analyzer.get_extension("unit_locations").get_data())
    ccgs, bins, _ = correlograms_from_analyzer(
        analyzer, window_ms=window_ms, bin_ms=bin_ms
    )
    # calculate_ca splits on sign; right edges would drop the innermost
    # negative bin and bias asymmetry positive.
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    spike_frames = {
        u: np.asarray(analyzer.sorting.get_unit_spike_train(u), dtype=np.int64)
        for u in unit_ids
    }
    spike_times = {
        u: _segment_times_at(analyzer.recording, frames)
        for u, frames in spike_frames.items()
    }
    boundaries = isi_continuity_boundaries(analyzer, continuity_spans)

    # Order-independent, so once per unordered pair.
    isi_violation_by_pair: dict[frozenset, float] = {}
    rows = []
    for u1, u2 in pairs:
        i1, i2 = index_of[u1], index_of[u2]
        pair_key = frozenset((u1, u2))
        if pair_key not in isi_violation_by_pair:
            merged_times, merged_frames = _dedup_merged_spike_times_and_frames(
                [spike_times[u1], spike_times[u2]],
                [spike_frames[u1], spike_frames[u2]],
                _MERGE_DEDUP_DELTA_MS / 1000.0,
            )
            violations = count_short_isis(
                merged_times,
                merged_frames,
                isi_threshold_ms * 1e-3,
                boundaries,
            )
            isi_violation_by_pair[pair_key] = float(
                isi_violation_fraction(violations, len(merged_times))
            )
        rows.append(
            {
                "unit1": u1,
                "unit2": u2,
                "wf_similarity": float(similarity[i1, i2]),
                "isi_violation": isi_violation_by_pair[pair_key],
                "xcorrel_asymm": calculate_ca(bin_centers, ccgs[i1, i2, :]),
                "unit_distance": float(
                    np.linalg.norm(locations[i1] - locations[i2])
                ),
            }
        )
    return rows


BURST_PAIR_METRIC_COLUMNS = (
    "wf_similarity",
    "isi_violation",
    "xcorrel_asymm",
    "unit_distance",
)


def burst_pair_metrics_frame(analyzer, pairs=None, **kwargs):
    """Burst-pair diagnostics as a ``(unit1, unit2)``-indexed DataFrame.

    Wraps :func:`burst_pair_metrics_from_analyzer`; ordered pairs stay
    distinct because ``xcorrel_asymm`` is directional.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_pairs, 4)``, MultiIndex ``(unit1, unit2)``, float columns
        ``wf_similarity``, ``isi_violation``, ``xcorrel_asymm``,
        ``unit_distance``.
    """
    import pandas as pd

    rows = burst_pair_metrics_from_analyzer(analyzer, pairs=pairs, **kwargs)
    frame = pd.DataFrame(
        rows, columns=("unit1", "unit2", *BURST_PAIR_METRIC_COLUMNS)
    )
    return frame.set_index(["unit1", "unit2"]).astype(float)


def validate_unit_pairs(unit_ids, pairs):
    """Return the pairs, raising if any member is not a real unit id."""
    unit_set = set(unit_ids)
    for a, b in pairs:
        if a not in unit_set or b not in unit_set:
            raise ValueError(
                f"Pair {(a, b)} references unit id(s) not in this sort "
                f"{sorted(unit_set)}."
            )
    return list(pairs)
