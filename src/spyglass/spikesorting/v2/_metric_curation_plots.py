"""DB-free plotting + analyzer-extraction helpers for analyzer curation.

The ``@schema`` ``AnalyzerCuration`` methods delegate here so the rendering /
SortingAnalyzer-reading logic stays importable and unit-testable without a
DataJoint connection. ``plot_units_qc_figure`` takes plain data (a metrics
DataFrame + unit locations) and returns a matplotlib figure; the burst helpers
read a SortingAnalyzer's correlograms / waveforms extensions and reuse the
shared ``utils_burst`` renderers.
"""

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


def plot_units_qc_figure(
    metrics_df: pd.DataFrame,
    unit_locations: np.ndarray | None,
    unit_ids: list,
    *,
    metric_names: list[str] | None = None,
    color_metric: str = "snr",
    depth_axis: int = 1,
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
        Metric used to color the depth scatter.
    depth_axis : int
        Column of ``unit_locations`` to treat as depth (default 1 = y).

    Returns
    -------
    matplotlib.figure.Figure
        The QC figure. A zero-unit sort returns an empty, labeled figure
        (never raises).
    """
    import matplotlib.pyplot as plt

    numeric = (
        metrics_df.apply(pd.to_numeric, errors="coerce")
        if len(metrics_df.columns)
        else metrics_df
    )
    if metric_names is None:
        present = [m for m in _DEFAULT_QC_METRICS if m in numeric.columns]
        metric_names = present or list(numeric.columns)

    n_hist = len(metric_names)
    has_units = unit_locations is not None and len(unit_ids) > 0

    if not has_units:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No units to display (zero-unit sort).",
            ha="center",
            va="center",
        )
        return fig

    n_cols = min(3, max(1, n_hist))
    n_hist_rows = math.ceil(n_hist / n_cols) if n_hist else 0
    # One extra full-width row for the depth scatter.
    n_rows = n_hist_rows + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    # Histograms: one small-multiple per metric, NaN dropped.
    for index, metric in enumerate(metric_names):
        ax = axes[index // n_cols][index % n_cols]
        values = numeric[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_dropped = len(values) - len(finite)
        if len(finite):
            ax.hist(finite, bins=min(20, max(5, len(finite))))
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
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
    # Hide unused histogram cells.
    for index in range(n_hist, n_hist_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_axis_off()

    # Depth scatter spanning the bottom row: each unit at its probe position,
    # colored by the chosen metric (units with a NaN color metric drawn gray).
    for col in range(n_cols):
        axes[n_rows - 1][col].remove()
    scatter_ax = fig.add_subplot(n_rows, 1, n_rows)
    locations = np.asarray(unit_locations, dtype=float)
    x = locations[:, 0]
    depth = locations[:, min(depth_axis, locations.shape[1] - 1)]
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

    fig.tight_layout()
    return fig


def correlograms_from_analyzer(
    analyzer, *, window_ms: float = 100.0, bin_ms: float = 5.0
):
    """Return ``(ccgs, bins, unit_ids)`` from the analyzer's correlograms ext.

    Computes the ``correlograms`` extension if absent (with the given window /
    bin), then returns the cross-correlogram cube, bin edges, and the analyzer
    unit-id order that indexes its first two axes.
    """
    if not analyzer.has_extension("correlograms"):
        analyzer.compute(
            "correlograms", window_ms=window_ms, bin_ms=bin_ms
        )
    extension = analyzer.get_extension("correlograms")
    ccgs, bins = extension.get_data()
    return np.asarray(ccgs), np.asarray(bins), list(analyzer.unit_ids)


def plot_autocorrelograms_figure(ccgs, bins, ids, unit_ids=None):
    """Render an autocorrelogram (ACG) grid, one panel per unit."""
    import matplotlib.pyplot as plt

    show = list(unit_ids) if unit_ids is not None else list(ids)
    index_of = {unit_id: i for i, unit_id in enumerate(ids)}
    n = len(show)
    n_cols = min(4, max(1, n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), squeeze=False
    )
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = bins[1] - bins[0]
    for k, unit_id in enumerate(show):
        ax = axes[k // n_cols][k % n_cols]
        i = index_of[unit_id]
        ax.bar(centers, ccgs[i, i, :], width=width)
        ax.set_title(f"unit {unit_id} ACG")
        ax.set_xlabel("ms")
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_axis_off()
    fig.tight_layout()
    return fig


def plot_pair_correlograms_figure(ccgs, bins, ids, pairs):
    """Render cross-correlograms for unit pairs (indexed by unit id, not 1-based).

    This is the v2 port of ``utils_burst.plot_burst_xcorrel`` which assumed
    1-based contiguous unit ids; v2 sorts can have arbitrary unit ids, so the
    pair is mapped through the analyzer's unit-id order.
    """
    import matplotlib.pyplot as plt

    index_of = {unit_id: i for i, unit_id in enumerate(ids)}
    n = len(pairs)
    n_cols = min(3, max(1, n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = bins[1] - bins[0]
    for k, (unit_a, unit_b) in enumerate(pairs):
        ax = axes[k // n_cols][k % n_cols]
        ax.bar(centers, ccgs[index_of[unit_a], index_of[unit_b], :], width=width)
        ax.set_title(f"{unit_a} x {unit_b}")
        ax.set_xlabel("ms")
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_axis_off()
    fig.tight_layout()
    return fig


def peak_amplitudes_from_analyzer(analyzer):
    """Return ``(peak_amps, peak_times)`` per unit from the waveforms ext.

    ``peak_amps[unit_id]`` is ``(n_spikes, n_channels)`` sampled at the
    waveform-window center; ``peak_times[unit_id]`` is the unit's spike times
    in seconds. Used by the burst-pair inspection plots.
    """
    waveforms = analyzer.get_extension("waveforms")
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    peak_amps: dict = {}
    peak_times: dict = {}
    for unit_id in sorting.get_unit_ids():
        wf = waveforms.get_waveforms_one_unit(unit_id)
        center = wf.shape[1] // 2
        peak_amps[unit_id] = wf[:, center, :]
        peak_times[unit_id] = (
            sorting.get_unit_spike_train(unit_id).astype(float) / fs
        )
    return peak_amps, peak_times


def validate_unit_pairs(unit_ids, pairs):
    """Return pairs whose members are real unit ids (orientation preserved)."""
    unit_set = set(unit_ids)
    validated = []
    for a, b in pairs:
        if a in unit_set and b in unit_set:
            validated.append((a, b))
        elif b in unit_set and a in unit_set:  # pragma: no cover - symmetric
            validated.append((b, a))
        else:
            raise ValueError(
                f"Pair {(a, b)} references unit id(s) not in this sort "
                f"{sorted(unit_set)}."
            )
    return validated
