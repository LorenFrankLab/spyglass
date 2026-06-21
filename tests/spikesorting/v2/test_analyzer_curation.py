"""Tests for analyzer-driven quality-metric curation (AnalyzerCuration).

Two tiers:

- ``db_unit`` Lookup-validation tests (Docker MySQL, no populate): structured
  parameter tables reject bogus values at insert, rule rows are queryable, and
  direct master inserts are blocked.
- ``slow`` / ``integration`` end-to-end tests: populate ``AnalyzerCuration``
  on the shared MEArec smoke sort, round-trip the fetch helpers, exercise the
  zero-label (#1625) path, and materialize a child curation.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------- Lookup validation (db_unit) -------------------------------------


@pytest.mark.db_unit
def test_quality_metric_parameters_rejects_bogus_metric(dj_conn):
    """A bogus metric name fails at QualityMetricParameters insert."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    with pytest.raises(ValidationError):
        QualityMetricParameters().insert1(
            {
                "metric_params_name": "bogus_metric_row",
                "metric_names": ["snr", "not_a_metric"],
                "metric_kwargs": {},
            }
        )
    assert not (
        QualityMetricParameters & {"metric_params_name": "bogus_metric_row"}
    )


@pytest.mark.db_unit
def test_auto_curation_rules_insert1_blocked(dj_conn):
    """Direct AutoCurationRules.insert1 is unsupported (helper-only)."""
    from spyglass.spikesorting.v2.exceptions import (
        UnsupportedDirectInsertError,
    )
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    with pytest.raises(UnsupportedDirectInsertError):
        AutoCurationRules().insert1(
            {"auto_curation_rules_name": "x", "auto_merge_preset": "none"}
        )


@pytest.mark.db_unit
def test_auto_curation_rules_insert_rules_queryable(dj_conn):
    """insert_rules stores ordered rule rows queryable by index/metric/label."""
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    name = "test_rules_queryable"
    (AutoCurationRules.Rule & {"auto_curation_rules_name": name}).delete_quick()
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete_quick()
    AutoCurationRules.insert_rules(
        {"auto_curation_rules_name": name, "auto_merge_preset": "none"},
        [
            {
                "rule_index": 0,
                "rule_name": "snr_noise",
                "metric_name": "snr",
                "operator": "<",
                "threshold": 2.0,
                "label": "noise",
            },
            {
                "rule_index": 1,
                "rule_name": "isi_mua",
                "metric_name": "isi_violation",
                "operator": ">",
                "threshold": 0.1,
                "label": "mua",
            },
        ],
    )
    rules = AutoCurationRules.Rule & {"auto_curation_rules_name": name}
    assert len(rules) == 2
    assert (rules & {"metric_name": "snr"}).fetch1("label") == "noise"
    assert (rules & {"rule_index": 1}).fetch1("label") == "mua"
    (AutoCurationRules.Rule & {"auto_curation_rules_name": name}).delete_quick()
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete_quick()


@pytest.mark.db_unit
def test_auto_curation_rules_insert_rules_rejects_bad_operator(dj_conn):
    """insert_rules validates the whole payload (bad operator rejected)."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    with pytest.raises(ValidationError):
        AutoCurationRules.insert_rules(
            {"auto_curation_rules_name": "bad_op", "auto_merge_preset": "none"},
            [
                {
                    "rule_index": 0,
                    "rule_name": "x",
                    "metric_name": "snr",
                    "operator": "=<",
                    "threshold": 2.0,
                    "label": "noise",
                }
            ],
        )
    assert not (AutoCurationRules & {"auto_curation_rules_name": "bad_op"})


# ---------- end-to-end (slow / integration) ---------------------------------


@pytest.fixture
def analyzer_curation_defaults(dj_conn):
    """Ensure the default metric/auto-curation Lookup rows exist."""
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )

    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()


def _populate_analyzer_curation(curation_key, metric_params, rules_name):
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        AnalyzerCurationSelection,
    )

    sel = AnalyzerCurationSelection.insert_selection(
        {
            **curation_key,
            "metric_params_name": metric_params,
            "auto_curation_rules_name": rules_name,
        }
    )
    AnalyzerCuration.populate(sel, reserve_jobs=False)
    return sel


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_end_to_end(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """make() writes 3 tables; fetch helpers round-trip; materialize forks."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    sel = _populate_analyzer_curation(
        populated_sorting_with_curation, "minimal", "none"
    )
    assert AnalyzerCuration & sel

    metrics = AnalyzerCuration.get_metrics(sel)
    assert isinstance(metrics, pd.DataFrame)
    assert "snr" in metrics.columns
    assert "isi_violation" in metrics.columns  # Spyglass fraction added
    assert "firing_rate" in metrics.columns

    # 'none' rules + 'none' preset => no labels, no merges.
    assert AnalyzerCuration.get_labels(sel) == {}
    assert AnalyzerCuration.get_merge_groups(sel) == []

    # materialize forks a child CurationV2 with the right provenance.
    child = AnalyzerCuration().materialize_curation(sel)
    child_row = (CurationV2 & child).fetch1()
    assert child_row["curation_source"] == "analyzer_curation"
    assert child_row["parent_curation_id"] == (
        populated_sorting_with_curation["curation_id"]
    )
    # Auto-registered in the merge master.
    assert SpikeSortingOutput.CurationV2 & child


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_zero_labels_saves_clean(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """Clean units under the default nn rule produce no labels, no crash (#1625).

    franklab_default computes nn_advanced (PCA) so the nn_noise_overlap column
    exists; on the clean MEArec fixture no unit trips the > 0.1 noise/reject
    rule, so proposed_labels saves cleanly and get_labels is empty.
    """
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    sel = _populate_analyzer_curation(
        populated_sorting_with_curation,
        "franklab_default",
        "v1_default_nn_noise",
    )
    assert AnalyzerCuration & sel
    # No hdmf "cannot infer dtype of empty list" crash; labels read back empty.
    labels = AnalyzerCuration.get_labels(sel)
    assert isinstance(labels, dict)
    metrics = AnalyzerCuration.get_metrics(sel)
    assert "nn_noise_overlap" in metrics.columns


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_selection_warns_on_auto_source(
    populated_sorting_with_curation, analyzer_curation_defaults, caplog
):
    """insert_selection warns when the upstream curation is auto-curated."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        AnalyzerCurationSelection,
    )

    sel = _populate_analyzer_curation(
        populated_sorting_with_curation, "minimal", "none"
    )
    child = AnalyzerCuration().materialize_curation(sel)
    assert (CurationV2 & child).fetch1("curation_source") == "analyzer_curation"

    import logging

    with caplog.at_level(logging.WARNING):
        AnalyzerCurationSelection.insert_selection(
            {
                **child,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
    assert any("analyzer_curation" in r.message for r in caplog.records)


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_viz_renders(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """plot_units_qc + ported BurstPair viz render against a real sort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    sel = _populate_analyzer_curation(
        populated_sorting_with_curation, "minimal", "none"
    )
    ac = AnalyzerCuration()

    fig = ac.plot_units_qc(sel)
    assert fig is not None
    plt.close(fig)

    # Ported BurstPair views read the analyzer's correlograms/waveforms.
    unit_ids = list(
        ac.get_metrics(sel).index
    )
    acg = ac.plot_correlograms(sel, unit_ids=unit_ids[:2])
    assert acg is not None
    plt.close(acg)

    if len(unit_ids) >= 2:
        pair = [(unit_ids[0], unit_ids[1])]
        xcorr = ac.investigate_pair_xcorrel(sel, pair)
        assert xcorr is not None
        plt.close(xcorr)
        peaks = ac.investigate_pair_peaks(sel, pair)
        assert peaks is not None
        plt.close(peaks)
