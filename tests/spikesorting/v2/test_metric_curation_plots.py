"""DB-free tests for the analyzer-curation QC plot helper.

Exercises ``plot_units_qc_figure`` with synthetic data under the Agg backend
(no DataJoint, no analyzer): one histogram axis per requested metric, a depth
scatter, NaN metrics dropped from histograms, and a zero-unit empty-but-labeled
figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spyglass.spikesorting.v2._metric_curation_plots import (  # noqa: E402
    plot_units_qc_figure,
)


def test_plot_units_qc_figure_renders_histograms_and_scatter():
    """One histogram per metric + a depth scatter; NaN dropped from a hist."""
    metrics = pd.DataFrame(
        {
            "snr": [3.0, 5.0, np.nan],  # one NaN -> dropped from snr hist
            "firing_rate": [1.0, 2.0, 3.0],
        },
        index=pd.Index([0, 1, 2], name="unit_id"),
    )
    locations = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    fig = plot_units_qc_figure(
        metrics, locations, [0, 1, 2], metric_names=["snr", "firing_rate"]
    )
    # 2 histogram axes + 1 scatter axis (the per-column bottom cells are
    # replaced by a single spanning scatter subplot).
    titles = [ax.get_title() for ax in fig.axes]
    assert any("snr" == t for t in titles)
    assert any("firing_rate" == t for t in titles)
    assert any("depth" in t for t in titles)
    # The snr histogram dropped exactly one NaN unit.
    snr_ax = next(ax for ax in fig.axes if ax.get_title() == "snr")
    assert any("NaN dropped" in txt.get_text() for txt in snr_ax.texts)


def test_plot_units_qc_figure_zero_unit_is_empty_but_labeled():
    """A zero-unit sort returns a labeled empty figure without raising."""
    fig = plot_units_qc_figure(pd.DataFrame(), None, [])
    assert fig is not None
    assert any("No units" in txt.get_text() for txt in fig.axes[0].texts)
