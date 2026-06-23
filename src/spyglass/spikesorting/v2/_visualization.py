"""DB-free routing/registry helpers for the v2 visualization facade.

The ``@schema`` ``visualization`` facade delegates the dependency-free pieces
here so the registry, the per-widget required-extension policy, the
missing-extension error wording, and the Spyglass-owned metric plot stay
importable and unit-testable without a DataJoint connection or a real
SortingAnalyzer.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: ``pandas`` / ``numpy`` are dependency-light and
``matplotlib`` / SpikeInterface are imported lazily inside the functions that
need them. The facade (``visualization.py``) owns the key -> recording /
analyzer / metric-table resolution that does need the schema classes.

The single load-bearing routing invariant lives in the table contract, not
here: visualization/export helpers read the UNWHITENED display analyzer (real
waveforms / locations / templates) or the saved preprocessed recording, never
the whitened metric analyzer. This module only encodes which display-safe
extensions each SpikeInterface widget needs and how to phrase the read-only
error when they are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MissingDisplayExtensionError(RuntimeError):
    """A richer SI widget needs display-safe analyzer extensions not present.

    Raised by the read-only default path so a notebook user gets an actionable
    message (the exact ``add_extensions`` call, or the ``compute_missing=True``
    opt-in) instead of a deep SpikeInterface ``check_extensions`` failure.
    """


# Display-safe extensions each SpikeInterface display widget reads beyond the
# sort-time base set (``random_spikes`` / ``noise_levels`` / ``templates`` /
# ``waveforms``), pinned against SI 0.104.3 widget behavior. ``principal_components``
# is deliberately absent from every list: it is the whitened metric-only
# extension and must never be computed from a display-only visualization path.
DISPLAY_WIDGET_EXTENSIONS = {
    "plot_sorting_summary": (
        "correlograms",
        "spike_amplitudes",
        "unit_locations",
        "template_similarity",
    ),
    "plot_unit_summary": ("unit_locations",),
    "plot_waveforms": (),
    "plot_spikes_on_traces": ("unit_locations",),
    "plot_unit_locations": ("unit_locations",),
    "plot_potential_merges": ("spike_amplitudes", "correlograms"),
}

# SI-native metric widgets read these analyzer extensions directly. They are
# NOT the official Spyglass routed metric table -- the default missing-extension
# error points users to ``plot_metrics`` instead.
SI_METRIC_WIDGET_EXTENSIONS = {
    "plot_si_quality_metrics": ("quality_metrics",),
    "plot_si_template_metrics": ("template_metrics",),
}

# SI ``export_report`` unconditionally computes ``unit_locations`` when absent
# (mutating the on-disk display analyzer), so it is the one extension the
# read-only (``force_computation=False``) path must require present. The wider
# set is what ``force_computation=True`` precomputes -- all display-safe -- so a
# richer report renders without any SI-side mutation.
REPORT_REQUIRED_EXTENSIONS = ("unit_locations",)
REPORT_DISPLAY_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "unit_locations",
    "template_similarity",
    "template_metrics",
)


# One row per public facade helper: the discovery surface returned by
# ``available_visualizations()``. ``compute_missing`` records whether the helper
# accepts the explicit opt-in to compute display-safe extensions it is missing.
_REGISTRY: tuple[dict, ...] = (
    {
        "name": "plot_recording_traces",
        "key_type": "recording",
        "implementation": "spikeinterface.widgets.plot_traces",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
    {
        "name": "plot_recording_probe_map",
        "key_type": "recording",
        "implementation": "spikeinterface.widgets.plot_probe_map",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
    {
        "name": "plot_sorting_summary",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_sorting_summary",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_unit_summary",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_summary",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_waveforms",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_waveforms",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
    {
        "name": "plot_spikes_on_traces",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_spikes_on_traces",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_unit_locations",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_locations",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_metrics",
        "key_type": "analyzer_curation",
        "implementation": "spyglass AnalyzerCuration.get_metrics",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
    {
        "name": "plot_si_quality_metrics",
        "key_type": "analyzer_curation",
        "implementation": "spikeinterface.widgets.plot_quality_metrics",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_si_template_metrics",
        "key_type": "analyzer_curation",
        "implementation": "spikeinterface.widgets.plot_template_metrics",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "plot_potential_merges",
        "key_type": "analyzer_curation",
        "implementation": "spikeinterface.widgets.plot_potential_merges",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
    {
        "name": "export_si_report",
        "key_type": "sorting",
        "implementation": "spikeinterface.exporters.export_report",
        "backend_default": "matplotlib",
        "compute_missing": True,
    },
    {
        "name": "export_to_phy",
        "key_type": "sorting",
        "implementation": "spikeinterface.exporters.export_to_phy",
        "backend_default": "matplotlib",
        "compute_missing": False,
    },
)

_REGISTRY_COLUMNS = (
    "name",
    "key_type",
    "implementation",
    "backend_default",
    "compute_missing",
)


def available_visualizations() -> pd.DataFrame:
    """Return the discoverable catalog of v2 visualization/export helpers.

    Documentation-as-code: one row per ``ssviz`` facade function, listing the
    DataJoint key type it accepts (``recording`` / ``sorting`` /
    ``analyzer_curation``), the underlying implementation target (a
    SpikeInterface widget/exporter or the Spyglass-routed
    ``AnalyzerCuration.get_metrics`` table), the default plotting backend, and
    whether the helper can compute missing display-safe extensions via an
    explicit ``compute_missing=True`` opt-in.

    Returns
    -------
    pandas.DataFrame
        One row per helper, columns ``name``, ``key_type``, ``implementation``,
        ``backend_default``, ``compute_missing``.
    """
    return pd.DataFrame(list(_REGISTRY), columns=list(_REGISTRY_COLUMNS))


def missing_extensions(analyzer, required_extensions) -> list[str]:
    """Return the requested extensions not already present on ``analyzer``.

    Pure: operates on a SortingAnalyzer-like object exposing
    ``has_extension(name)``; no DB access, no compute. Preserves the input
    order so the error message and any subsequent compute list are stable.
    """
    return [
        name
        for name in required_extensions
        if not analyzer.has_extension(name)
    ]


def format_missing_extension_error(
    missing,
    *,
    recommend_metrics: bool = False,
) -> str:
    """Phrase the read-only missing-extension error.

    Names the absent display-safe extensions, the ``compute_missing=True``
    opt-in, and the explicit ``Sorting().add_extensions(...)`` call a user can
    run themselves. When ``recommend_metrics`` is set (the SI-native metric
    widgets), it also points back to the official Spyglass ``plot_metrics``
    overview, since the SI quality/template-metric widgets read analyzer
    extensions directly rather than the routed Spyglass metric table.
    """
    missing_list = list(missing)
    lines = [
        "This visualization needs display-safe analyzer extension(s) that are "
        f"not yet computed: {missing_list}.",
        "Re-run with compute_missing=True to compute only these display-safe "
        "extensions, or add them yourself first with "
        f"Sorting().add_extensions(sorting_key, {missing_list}).",
    ]
    if recommend_metrics:
        lines.append(
            "For the official Spyglass quality metrics (the routed metric "
            "table, including surfaced waveform-shape columns), use "
            "ssviz.plot_metrics(analyzer_curation_key) instead -- it reads "
            "AnalyzerCuration.get_metrics(), not the raw SI analyzer extension."
        )
    return " ".join(lines)


def plot_metrics_figure(metrics_df: pd.DataFrame, *, columns=None):
    """Render the Spyglass-routed quality-metric table as a histogram grid.

    A Spyglass-owned plot of exactly what ``AnalyzerCuration.get_metrics()``
    persists -- the configured SpikeInterface quality metrics plus the surfaced
    waveform-shape (template) columns (``trough_half_width`` by default) -- so
    the displayed values are the routed metric provenance, not a re-derivation
    off a raw analyzer extension. One histogram per numeric column; non-finite
    values (the ``None`` sanitized non-finite readback) are coerced to NaN and
    dropped per-column. A zero-unit / empty table returns a labeled empty
    figure rather than raising.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-unit metric table indexed by ``unit_id`` (the
        ``AnalyzerCuration.get_metrics`` return value).
    columns : list of str, optional
        Subset of columns to histogram; defaults to every numeric column in
        display order.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import math

    import matplotlib.pyplot as plt

    numeric = (
        metrics_df.apply(pd.to_numeric, errors="coerce")
        if len(metrics_df.columns)
        else metrics_df
    )
    if columns is None:
        columns = list(numeric.columns)
    else:
        columns = [c for c in columns if c in numeric.columns]

    if len(metrics_df) == 0 or not columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No quality metrics to display.",
            ha="center",
            va="center",
        )
        return fig

    n_cols = min(3, len(columns))
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    for index, column in enumerate(columns):
        ax = axes[index // n_cols][index % n_cols]
        values = numeric[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_dropped = len(values) - len(finite)
        if len(finite):
            ax.hist(finite, bins=min(20, max(5, len(finite))))
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("unit count")
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
    for index in range(len(columns), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_axis_off()
    fig.tight_layout()
    return fig
