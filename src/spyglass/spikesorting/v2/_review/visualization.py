"""Registry and extension policy for the v2 visualization facade.

The facade (``visualization.py``) resolves keys and guarantees helpers read
the unwhitened display analyzer, never the whitened metric analyzer.
"""

from __future__ import annotations

import pandas as pd

# Extensions each widget reads beyond the base set (SI 0.104.3). Never
# ``principal_components``, which belongs to the whitened metric analyzer.
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
    "plot_suggested_merges": ("spike_amplitudes", "correlograms"),
}

# Raw SI extensions, not the routed metric table (see ``plot_metrics``).
SI_METRIC_WIDGET_EXTENSIONS = {
    "plot_si_quality_metrics": ("quality_metrics",),
    "plot_si_template_metrics": ("template_metrics",),
}

# ``export_report`` computes ``unit_locations`` itself (mutating the analyzer),
# so read-only mode requires it. ``compute_missing=True`` adds what it renders.
REPORT_REQUIRED_EXTENSIONS = ("unit_locations",)
REPORT_DISPLAY_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "unit_locations",
)


# Returned by ``available_visualizations()``.
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
        "key_type": "curation",
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
        "key_type": "curation",
        "implementation": "spikeinterface.widgets.plot_unit_summary",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": (
            "Single-unit summary (waveform, location, correlogram, amplitudes)"
        ),
    },
    {
        "name": "plot_waveforms",
        "key_type": "curation",
        "implementation": "spikeinterface.widgets.plot_unit_waveforms",
        "backend_default": "matplotlib",
        "compute_missing": False,
        "description": "Per-unit waveforms and templates",
    },
    {
        "name": "plot_spikes_on_traces",
        "key_type": "curation",
        "implementation": "spikeinterface.widgets.plot_spikes_on_traces",
        "backend_default": "matplotlib",
        "compute_missing": True,
        "description": "Detected spikes overlaid on the recording traces",
    },
    {
        "name": "plot_unit_locations",
        "key_type": "curation",
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
        "name": "plot_suggested_merges",
        "key_type": "curation_evaluation",
        # PotentialMergesWidget supports only the interactive ipywidgets backend.
        "implementation": "spikeinterface.widgets.plot_potential_merges",
        "backend_default": "ipywidgets",
        "compute_missing": True,
        "description": (
            "Persisted merge-group suggestions (never recomputed; ipywidgets)"
        ),
    },
    {
        "name": "export_si_report",
        "key_type": "curation",
        "implementation": "spikeinterface.exporters.export_report",
        "backend_default": None,  # an exporter: no plotting backend
        "compute_missing": True,
        "description": "Write a local SI report folder of figures / tables",
    },
    {
        "name": "export_to_phy",
        "key_type": "curation",
        "implementation": "spikeinterface.exporters.export_to_phy",
        "backend_default": None,  # an exporter: no plotting backend
        "compute_missing": False,
        "description": "Export the exact curated units to a local Phy folder",
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

    Returns
    -------
    pandas.DataFrame
        One row per ``ssviz`` helper, columns ``name``, ``key_type``
        (``recording`` / ``curation`` / ``curation_evaluation``),
        ``implementation``, ``backend_default`` (``None`` for exporters),
        ``compute_missing`` (accepts ``compute_missing=True``),
        ``description``.
    """
    return pd.DataFrame(list(_REGISTRY), columns=list(_REGISTRY_COLUMNS))


def missing_extensions(analyzer, required_extensions) -> list[str]:
    """Return the requested extensions not already present on ``analyzer``.

    Preserves input order.
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

    ``recommend_metrics`` (SI-native metric widgets) also points to
    ``plot_metrics``.
    """
    missing_list = list(missing)
    lines = [
        "This visualization needs display-safe analyzer extension(s) that are "
        f"not yet computed: {missing_list}.",
        "Re-run with compute_missing=True to compute only these display-safe "
        "extensions on the analyzer this curation's plots read.",
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

    One histogram per numeric column, non-finite values dropped. An empty
    table returns a labeled empty figure.

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

    Raises
    ------
    ValueError
        If ``columns`` names a column the table does not have.
    """
    import math

    import matplotlib.pyplot as plt

    from spyglass.spikesorting.v2._curation.metric_plots import (
        draw_metric_histogram,
    )

    numeric = (
        metrics_df.apply(pd.to_numeric, errors="coerce")
        if len(metrics_df.columns)
        else metrics_df
    )
    if columns is None:
        columns = list(numeric.columns)
    elif len(numeric.columns):
        # A zero-unit table has no columns to check against.
        unknown = [c for c in columns if c not in numeric.columns]
        if unknown:
            raise ValueError(
                f"Unknown metric column(s) {unknown}. Pass output column "
                "names, not metric names (nn_advanced emits "
                "nn_noise_overlap/nn_isolation). Available columns: "
                f"{list(numeric.columns)}."
            )

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
