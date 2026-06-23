"""DB-free unit tests for the visualization registry / routing helpers.

Exercises ``_visualization`` with synthetic inputs under the Agg backend (no
DataJoint, no real SortingAnalyzer): the discovery catalog, the missing-display-
extension policy + error wording, and the Spyglass-owned metric histogram plot.
The key -> recording/analyzer resolution (which needs the schema classes) is
covered by the ``db_unit`` facade tests in ``test_visualization_facade.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spyglass.spikesorting.v2._visualization import (  # noqa: E402
    DISPLAY_WIDGET_EXTENSIONS,
    MissingDisplayExtensionError,
    SI_METRIC_WIDGET_EXTENSIONS,
    available_visualizations,
    format_missing_extension_error,
    missing_extensions,
    plot_metrics_figure,
)


class _FakeAnalyzer:
    """Minimal SortingAnalyzer stand-in exposing only ``has_extension``."""

    def __init__(self, present):
        self._present = set(present)

    def has_extension(self, name):
        return name in self._present


@pytest.mark.unit
def test_available_visualizations_lists_documented_helpers():
    """The catalog covers every public facade helper with its routing facts."""
    table = available_visualizations()
    assert set(table.columns) == {
        "name",
        "key_type",
        "implementation",
        "backend_default",
        "compute_missing",
    }
    names = set(table["name"])
    expected = {
        "plot_recording_traces",
        "plot_recording_probe_map",
        "plot_sorting_summary",
        "plot_unit_summary",
        "plot_waveforms",
        "plot_spikes_on_traces",
        "plot_unit_locations",
        "plot_metrics",
        "plot_si_quality_metrics",
        "plot_si_template_metrics",
        "plot_potential_merges",
        "export_si_report",
        "export_to_phy",
    }
    assert expected <= names
    # Every helper documents matplotlib as its default backend.
    assert set(table["backend_default"]) == {"matplotlib"}
    # Key types are constrained to the three resolvable kinds.
    assert set(table["key_type"]) <= {
        "recording",
        "sorting",
        "analyzer_curation",
    }


@pytest.mark.unit
def test_available_visualizations_routes_metrics_to_spyglass_not_si():
    """``plot_metrics`` is implemented by the routed Spyglass metric table.

    The SI-native quality/template-metric widgets are separately named and read
    analyzer extensions, so only those point at ``spikeinterface.widgets``.
    """
    table = available_visualizations().set_index("name")
    assert table.loc["plot_metrics", "implementation"] == (
        "spyglass AnalyzerCuration.get_metrics"
    )
    assert "spikeinterface" not in table.loc["plot_metrics", "implementation"]
    assert table.loc["plot_si_quality_metrics", "implementation"].startswith(
        "spikeinterface.widgets"
    )


@pytest.mark.unit
def test_widget_extension_maps_exclude_principal_components():
    """No display widget requires the whitened metric-only PC extension."""
    for required in DISPLAY_WIDGET_EXTENSIONS.values():
        assert "principal_components" not in required
    for required in SI_METRIC_WIDGET_EXTENSIONS.values():
        assert "principal_components" not in required


@pytest.mark.unit
def test_missing_extensions_returns_absent_in_order():
    """Only absent extensions are returned, preserving request order."""
    analyzer = _FakeAnalyzer(["templates", "waveforms", "unit_locations"])
    required = ("unit_locations", "correlograms", "spike_amplitudes")
    assert missing_extensions(analyzer, required) == [
        "correlograms",
        "spike_amplitudes",
    ]


@pytest.mark.unit
def test_missing_extensions_empty_when_all_present():
    """A fully provisioned analyzer reports nothing missing."""
    analyzer = _FakeAnalyzer(["unit_locations", "correlograms"])
    assert missing_extensions(analyzer, ("unit_locations", "correlograms")) == []


@pytest.mark.unit
def test_format_missing_extension_error_names_remediation():
    """The error names the extensions, the opt-in, and the add_extensions call."""
    message = format_missing_extension_error(["unit_locations", "correlograms"])
    assert "unit_locations" in message
    assert "correlograms" in message
    assert "compute_missing=True" in message
    assert "add_extensions" in message
    # Default wording does not steer toward the metric overview.
    assert "plot_metrics" not in message


@pytest.mark.unit
def test_format_missing_extension_error_recommends_metrics_for_si_widgets():
    """SI-native metric widgets point users back to the routed ``plot_metrics``."""
    message = format_missing_extension_error(
        ["quality_metrics"], recommend_metrics=True
    )
    assert "plot_metrics" in message
    assert "get_metrics" in message


@pytest.mark.unit
def test_missing_extension_error_is_runtime_error():
    """The typed error is catchable as a RuntimeError by generic handlers."""
    assert issubclass(MissingDisplayExtensionError, RuntimeError)


@pytest.mark.unit
def test_plot_metrics_figure_one_histogram_per_metric_with_nan_dropped():
    """One histogram per numeric column; non-finite values dropped per-column."""
    metrics = pd.DataFrame(
        {
            "snr": [3.0, 5.0, np.nan],
            "trough_half_width": [0.2, 0.25, 0.3],
        },
        index=pd.Index([0, 1, 2], name="unit_id"),
    )
    fig = plot_metrics_figure(metrics)
    titles = [ax.get_title() for ax in fig.axes]
    assert "snr" in titles
    # The surfaced waveform-shape column is plotted as-is, not re-derived.
    assert "trough_half_width" in titles
    snr_ax = next(ax for ax in fig.axes if ax.get_title() == "snr")
    assert any("NaN dropped" in txt.get_text() for txt in snr_ax.texts)
    plt.close(fig)


@pytest.mark.unit
def test_plot_metrics_figure_empty_table_is_labeled_not_raising():
    """A zero-unit metric table renders a labeled empty figure."""
    fig = plot_metrics_figure(pd.DataFrame())
    assert any(
        "No quality metrics" in txt.get_text() for txt in fig.axes[0].texts
    )
    plt.close(fig)


@pytest.mark.unit
def test_plot_metrics_figure_respects_explicit_columns():
    """An explicit column subset limits the histograms shown."""
    metrics = pd.DataFrame(
        {"snr": [1.0, 2.0], "firing_rate": [3.0, 4.0]},
        index=pd.Index([0, 1], name="unit_id"),
    )
    fig = plot_metrics_figure(metrics, columns=["snr"])
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["snr"]
    plt.close(fig)
