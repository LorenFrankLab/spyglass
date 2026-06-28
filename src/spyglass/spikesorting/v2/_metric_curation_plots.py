"""DB-free plotting + analyzer-extraction helpers for analyzer curation.

The ``@schema`` ``CurationEvaluation`` methods delegate here so the rendering /
SortingAnalyzer-reading logic stays importable and unit-testable without a
DataJoint connection. ``plot_units_qc_figure`` takes plain data (a metrics
DataFrame + unit locations) and returns the matplotlib axes it drew into; the
burst helpers read a SortingAnalyzer's waveforms / template_similarity /
unit_locations extensions, compute correlograms on the fly at the requested
window/bin, and reuse the shared ``utils_burst`` renderers.
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


def _flatten_axes(axes):
    """Return caller-supplied axes as a flat list."""
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    if isinstance(axes, (list, tuple)):
        return list(np.asarray(axes, dtype=object).ravel())
    return [axes]


def draw_metric_histogram(ax, values, *, title, ylabel="count"):
    """Draw one metric's histogram on ``ax``, dropping non-finite values.

    The float-coerced ``values`` have their non-finite entries (the ``None``
    sanitized non-finite readback) dropped before histogramming, with the dropped
    count annotated in the corner. Shared by the population QC overview
    (``plot_units_qc_figure``) and the routed-metric plot
    (``plot_metrics_figure``) so a per-metric panel renders identically in both.
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


def plot_units_qc_figure(
    metrics_df: pd.DataFrame,
    unit_locations: np.ndarray | None,
    unit_ids: list,
    *,
    metric_names: list[str] | None = None,
    color_metric: str = "snr",
    depth_axis: int = 1,
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
        Metric used to color the depth scatter.
    depth_axis : int
        Column of ``unit_locations`` to treat as depth (default 1 = y).
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

    # Histograms: one small-multiple per metric, NaN dropped.
    for metric, ax in hist_axes.items():
        draw_metric_histogram(
            ax, numeric[metric].to_numpy(dtype=float), title=metric
        )

    # Depth scatter spanning the bottom row: each unit at its probe position,
    # colored by the chosen metric (units with a NaN color metric drawn gray).
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

    # Only tidy a figure we created; when the caller injects axes they own the
    # layout (tight_layout would fight a constrained_layout / mosaic host).
    if axes is None:
        fig.tight_layout()
    return {**hist_axes, "scatter": scatter_ax}


def correlograms_from_analyzer(
    analyzer, *, window_ms: float = 100.0, bin_ms: float = 5.0
):
    """Return ``(ccgs, bins, unit_ids)`` computed fresh at the given window/bin.

    Computes correlograms directly from the sorting with the REQUESTED
    ``window_ms`` / ``bin_ms`` rather than reusing the stored ``correlograms``
    extension -- curation computes that extension with SI's 50 / 1 ms default
    for the merge engine, so reusing it would silently ignore the window/bin
    asked for here. A fresh compute on the sorting does not overwrite the stored
    extension. Indexed by the analyzer's unit-id order.
    """
    from spikeinterface.postprocessing import compute_correlograms

    ccgs, bins = compute_correlograms(
        analyzer.sorting, window_ms=window_ms, bin_ms=bin_ms
    )
    return np.asarray(ccgs), np.asarray(bins), list(analyzer.unit_ids)


def _plot_correlogram_grid(ccgs, bins, panels, *, n_cols, panel_size):
    """Render a grid of correlogram bar plots.

    ``panels`` is a list of ``(row_index, col_index, title)`` -- ``ccgs[row,
    col, :]`` is the histogram for each cell. Shared by the autocorrelogram
    (i==j) and cross-correlogram (pair) views.
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

    This is the v2 port of ``utils_burst.plot_burst_xcorrel`` which assumed
    1-based contiguous unit ids; v2 sorts can have arbitrary unit ids, so the
    pair is mapped through the analyzer's unit-id order.
    """
    index_of = {unit_id: i for i, unit_id in enumerate(ids)}
    panels = [(index_of[a], index_of[b], f"{a} x {b}") for a, b in pairs]
    return _plot_correlogram_grid(
        ccgs, bins, panels, n_cols=3, panel_size=(4, 3)
    )


def peak_amplitudes_from_analyzer(analyzer):
    """Return ``(peak_amps, peak_times)`` per unit from the waveforms ext.

    ``peak_amps[unit_id]`` is ``(n_spikes, n_channels)`` sampled at the
    waveform PEAK -- the ``nbefore`` alignment sample (the trough), NOT the
    array center. The analyzer's asymmetric window (``ms_before=1.0`` !=
    ``ms_after=2.0``) places the peak off the geometric center, so the center
    sample reads ~0.5 ms into the repolarization tail rather than the peak.
    ``peak_times[unit_id]`` is the spike times (seconds) of THOSE SAME spikes.
    The ``waveforms`` extension holds only the ``random_spikes`` subset (capped
    at ``max_spikes_per_unit``), so the times are taken from that subset, not
    the full train -- otherwise the amplitude/time arrays mismatch for any unit
    with more spikes than the cap and ``plot_peak_over_time`` misaligns. The
    subset is read via the ``random_spikes`` extension, which is sorted by
    sample index in the same order as ``get_waveforms_one_unit``. Amplitudes are
    returned **dense** (all probe channels, global index) so the burst-pair
    plots -- which overlay two units channel-by-channel -- compare the SAME
    physical contact across units; a sparse analyzer gives each unit its own
    channel subset, so local-index overlay would mismatch contacts. Used by the
    burst-pair inspection plots.
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
):
    """Per-pair burst-merge diagnostics computed on the fly (nothing stored).

    Returns one dict per ordered unit pair with the four legs of the v1
    BurstPair merge test, read from the analyzer's extensions (computing
    ``templates`` / ``template_similarity`` / ``unit_locations`` on the fly if
    absent, and correlograms fresh at the requested window/bin):

    - ``wf_similarity`` -- SI's cosine ``template_similarity`` (not v1's flat
      Pearson, which mishandles per-unit sparsity). Note this is an INDEPENDENT
      corroboration, not a reproduction of the merge engine's metric: the
      ``similarity_correlograms`` auto-merge preset compares templates with
      ``l1`` (its ``similarity_method`` default), while this reports cosine.
    - ``isi_violation`` -- refractory-violation fraction of the merged train
      (``isi_threshold_ms`` default 2 ms, aligned with the single-unit metric).
    - ``xcorrel_asymm`` -- cross-correlogram left/right asymmetry (burst
      parent/child signature; directional, so computed per ordered pair).
    - ``unit_distance`` -- euclidean distance between ``unit_locations`` (the
      spatial leg v1 lacked; two cells can share a shape at different depths).

    ``pairs`` defaults to all ordered pairs; an explicit list is validated
    against the analyzer's unit ids.

    All four legs are template- or spike-time-derived, so callers pass the
    UNWHITENED display analyzer (whitening distorts ``template_similarity`` and
    the ``unit_locations`` positions): ``CurationEvaluation`` loads it via
    ``_analyzer_for``, the same display analyzer the merge engine runs on. The
    diagnostic is an independent corroboration of those suggestions (it differs
    in similarity metric, see above), not a reproduction of them.
    """
    from itertools import permutations

    from spyglass.spikesorting.utils_burst import (
        calculate_ca,
        calculate_isi_violation,
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
    fs = analyzer.sampling_frequency
    spike_times = {
        u: analyzer.sorting.get_unit_spike_train(u).astype(float) / fs
        for u in unit_ids
    }

    rows = []
    for u1, u2 in pairs:
        i1, i2 = index_of[u1], index_of[u2]
        rows.append(
            {
                "unit1": u1,
                "unit2": u2,
                "wf_similarity": float(similarity[i1, i2]),
                "isi_violation": calculate_isi_violation(
                    spike_times[u1],
                    spike_times[u2],
                    isi_threshold_ms=isi_threshold_ms,
                ),
                "xcorrel_asymm": calculate_ca(bins[1:], ccgs[i1, i2, :]),
                "unit_distance": float(
                    np.linalg.norm(locations[i1] - locations[i2])
                ),
            }
        )
    return rows


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
