"""DB-free routing/registry helpers for the v2 visualization facade.

The ``@schema`` ``visualization`` facade delegates the dependency-free pieces
here so the registry, the per-widget required-extension policy, the
missing-extension error wording, and the Spyglass-owned metric plot stay
importable and unit-testable without a DataJoint connection or a real
SortingAnalyzer.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: ``pandas`` is dependency-light and ``matplotlib`` /
SpikeInterface are imported lazily inside the functions that need them. The
facade (``visualization.py``) owns the key -> recording / analyzer /
metric-table resolution that does need the schema classes.

The single load-bearing routing invariant lives in the table contract, not
here: visualization/export helpers read the UNWHITENED display analyzer (real
waveforms / locations / templates) or the saved preprocessed recording, never
the whitened metric analyzer. This module only encodes which display-safe
extensions each SpikeInterface widget needs and how to phrase the read-only
error when they are absent.
"""

from __future__ import annotations

import pandas as pd

# Re-exported from the central exceptions module (the single home for every
# named v2 invariant); imported here so existing
# ``from ..._visualization import MissingDisplayExtensionError`` paths keep
# working.
from spyglass.spikesorting.v2.exceptions import (  # noqa: F401
    MissingDisplayExtensionError,
)

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
# set is exactly the display-safe extensions SI's ``export_report`` actually
# renders (``spike_amplitudes`` distributions, ``correlograms`` in the per-unit
# summaries, ``unit_locations`` maps), so ``force_computation=True`` precomputes
# only those -- a richer report renders with no SI-side mutation, and nothing
# unread is computed onto the display analyzer.
REPORT_REQUIRED_EXTENSIONS = ("unit_locations",)
REPORT_DISPLAY_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "unit_locations",
)


# One row per visualization/export helper: the discovery surface returned by
# ``available_visualizations()``. (The ``recording_key_for_sorting`` key resolver
# is a public facade convenience but not a visualization, so it is not catalogued
# here.) ``compute_missing`` records whether the helper accepts the explicit
# opt-in to compute display-safe extensions it is missing.
_REGISTRY: tuple[dict, ...] = (
    {
        "name": "plot_recording_traces",
        "key_type": "recording",
        "implementation": "spikeinterface.widgets.plot_traces",
        "backend_default": "matplotlib",
        "compute_missing": False,
        "description": "Voltage traces of the saved preprocessed recording",
    },
    {
        "name": "plot_recording_probe_map",
        "key_type": "recording",
        "implementation": "spikeinterface.widgets.plot_probe_map",
        "backend_default": "matplotlib",
        "compute_missing": False,
        "description": "Probe geometry / channel map of the saved recording",
    },
    {
        "name": "plot_sorting_summary",
        "key_type": "sorting",
        # SortingSummaryWidget has no matplotlib backend in SI 0.104.3; a
        # backend (spikeinterface_gui / sortingview / figpack) is required.
        "implementation": "spikeinterface.widgets.plot_sorting_summary",
        "backend_default": None,
        "compute_missing": True,
        "description": (
            "Multi-panel summary of the whole sort (GUI/web backend required, "
            "no matplotlib)"
        ),
    },
    {
        "name": "plot_unit_summary",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_summary",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": (
            "Single-unit summary (waveform, location, correlogram, amplitudes)"
        ),
    },
    {
        "name": "plot_waveforms",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_waveforms",
        "backend_default": "matplotlib",
        "compute_missing": False,
        "description": "Per-unit waveforms and templates",
    },
    {
        "name": "plot_spikes_on_traces",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_spikes_on_traces",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": "Detected spikes overlaid on the recording traces",
    },
    {
        "name": "plot_unit_locations",
        "key_type": "sorting",
        "implementation": "spikeinterface.widgets.plot_unit_locations",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": "Estimated unit positions on the probe",
    },
    {
        "name": "plot_metrics",
        "key_type": "curation_evaluation",
        "implementation": "spyglass CurationEvaluation.get_metrics",
        "backend_default": "matplotlib",
        "compute_missing": False,
        "description": "Histograms of the routed Spyglass quality-metric table",
    },
    {
        "name": "plot_si_quality_metrics",
        "key_type": "curation_evaluation",
        "implementation": "spikeinterface.widgets.plot_quality_metrics",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": (
            "Raw SI quality-metric view (display-analyzer extension, not routed)"
        ),
    },
    {
        "name": "plot_si_template_metrics",
        "key_type": "curation_evaluation",
        "implementation": "spikeinterface.widgets.plot_template_metrics",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": (
            "Raw SI template-metric view (display-analyzer extension, not routed)"
        ),
    },
    {
        "name": "plot_potential_merges",
        "key_type": "curation_evaluation",
        # PotentialMergesWidget supports only the interactive ipywidgets backend.
        "implementation": "spikeinterface.widgets.plot_potential_merges",
        "backend_default": "ipywidgets",
        "compute_missing": False,
        "description": (
            "Persisted merge-group suggestions (never recomputed; ipywidgets)"
        ),
    },
    {
        "name": "export_si_report",
        "key_type": "sorting",
        "implementation": "spikeinterface.exporters.export_report",
        "backend_default": None,  # an exporter: no plotting backend
        "compute_missing": True,
        "description": "Write a local SI report folder of figures / tables",
    },
    {
        "name": "export_to_phy",
        "key_type": "sorting",
        "implementation": "spikeinterface.exporters.export_to_phy",
        "backend_default": None,  # an exporter: no plotting backend
        "compute_missing": False,
        "description": "Export the sort to a local Phy folder",
    },
)

_REGISTRY_COLUMNS = (
    "name",
    "key_type",
    "implementation",
    "backend_default",
    "compute_missing",
    "description",
)


def available_visualizations() -> pd.DataFrame:
    """Return the discoverable catalog of v2 visualization/export helpers.

    Documentation-as-code: one row per ``ssviz`` visualization/export helper
    (the ``recording_key_for_sorting`` key resolver is a public convenience but
    not a visualization, so it is intentionally not catalogued here), listing a
    one-line ``description``, the DataJoint key type it accepts (``recording`` /
    ``sorting`` / ``curation_evaluation``), the underlying implementation target (a
    SpikeInterface widget/exporter or the Spyglass-routed
    ``CurationEvaluation.get_metrics`` table), the default plotting backend
    (``None`` for the exporters, which take no backend), and whether the helper
    can compute missing display-safe extensions via an explicit
    ``compute_missing=True`` opt-in.

    Returns
    -------
    pandas.DataFrame
        One row per helper, columns ``name``, ``key_type``, ``implementation``,
        ``backend_default``, ``compute_missing``, ``description``.
    """
    return pd.DataFrame(list(_REGISTRY), columns=list(_REGISTRY_COLUMNS))


def missing_extensions(analyzer, required_extensions) -> list[str]:
    """Return the requested extensions not already present on ``analyzer``.

    Pure: operates on a SortingAnalyzer-like object exposing
    ``has_extension(name)``; no DB access, no compute. Preserves the input
    order so the error message and any subsequent compute list are stable.
    """
    return [
        name for name in required_extensions if not analyzer.has_extension(name)
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
            "ssviz.plot_metrics(curation_evaluation_key) instead -- it reads "
            "CurationEvaluation.get_metrics(), not the raw SI analyzer extension."
        )
    return " ".join(lines)


def plot_metrics_figure(metrics_df: pd.DataFrame, *, columns=None):
    """Render the Spyglass-routed quality-metric table as a histogram grid.

    A Spyglass-owned plot of exactly what ``CurationEvaluation.get_metrics()``
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
        ``CurationEvaluation.get_metrics`` return value).
    columns : list of str, optional
        Subset of columns to histogram; defaults to every numeric column in
        display order.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import math

    import matplotlib.pyplot as plt

    from spyglass.spikesorting.v2._metric_curation_plots import (
        draw_metric_histogram,
    )

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
        draw_metric_histogram(
            ax,
            numeric[column].to_numpy(dtype=float),
            title=column,
            ylabel="unit count",
        )
    for index in range(len(columns), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_axis_off()
    fig.tight_layout()
    return fig
