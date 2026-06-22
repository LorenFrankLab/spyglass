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


@pytest.mark.db_unit
def test_compute_merge_groups_suggests_planted_oversplit(dj_conn):
    """Auto-merge proposes a group for a planted duplicate/oversplit pair.

    Builds a synthetic analyzer whose units 0 and 1 are one cell split in two
    (same waveform), then drives AnalyzerCuration's auto-merge wrapper. The
    ``x_contaminations`` preset reliably catches a temporal oversplit on
    synthetic data; the wrapper must return the pair as python ints, and the
    ``none`` preset must short-circuit to an empty list.
    """
    import numpy as np  # noqa: F401
    import spikeinterface.full as si

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    rec, gt = si.generate_ground_truth_recording(
        durations=[60.0], num_units=3, seed=0, sampling_frequency=30000.0
    )
    fs = rec.get_sampling_frequency()
    st0 = gt.get_unit_spike_train(gt.get_unit_ids()[0])
    half = len(st0) // 2
    sort = si.NumpySorting.from_unit_dict(
        {
            0: st0[:half],
            1: st0[half:],  # same waveform as unit 0 -> an oversplit
            2: gt.get_unit_spike_train(gt.get_unit_ids()[1]),
        },
        sampling_frequency=fs,
    )
    analyzer = si.create_sorting_analyzer(
        sort, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        [
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            "correlograms",
            "template_similarity",
            "unit_locations",
            "spike_amplitudes",
            "principal_components",
        ],
        extension_params={
            "random_spikes": {"seed": 0},
            "principal_components": {"n_components": 3},
        },
    )

    groups = AnalyzerCuration._compute_merge_groups(
        analyzer, "x_contaminations", {}
    )
    assert groups, "auto-merge should propose the planted oversplit"
    flat = [u for group in groups for u in group]
    assert 0 in flat and 1 in flat
    # Wrapper coerces SI's np.int64 unit ids to python ints.
    assert all(isinstance(u, int) for group in groups for u in group)
    # 'none' short-circuits without calling SI.
    assert AnalyzerCuration._compute_merge_groups(analyzer, "none", {}) == []


@pytest.mark.db_unit
def test_auto_curation_rules_none_default_passes_integrity(dj_conn):
    """The shipped 'none' inert default is exempt; a custom no-op is flagged."""
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    AutoCurationRules.insert_default()
    offenders = AutoCurationRules.check_rule_integrity()
    flagged = {o["auto_curation_rules_name"] for o in offenders}
    assert "none" not in flagged  # canonical inert sentinel is not malformed

    # A CUSTOM preset='none' + no-rules set genuinely does nothing -> flagged.
    name = "custom_noop_rules"
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete_quick()
    AutoCurationRules.insert_rules(
        {"auto_curation_rules_name": name, "auto_merge_preset": "none"}, []
    )
    try:
        offenders = AutoCurationRules.check_rule_integrity()
        custom = [
            o for o in offenders if o["auto_curation_rules_name"] == name
        ]
        assert custom and custom[0]["issue"] == "no_effect"
    finally:
        (AutoCurationRules & {"auto_curation_rules_name": name}).delete_quick()


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_selection_idempotent_content_addressed(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """insert_selection is content-addressed: repeat calls return one id."""
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )

    key = {
        **populated_sorting_with_curation,
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
    }
    sel1 = AnalyzerCurationSelection.insert_selection(key)
    sel2 = AnalyzerCurationSelection.insert_selection(key)  # idempotent refetch
    assert sel1 == sel2

    identity = {
        "sorting_id": key["sorting_id"],
        "curation_id": key["curation_id"],
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
    }
    expected = deterministic_id("analyzer_curation", identity)
    assert str(sel1["analyzer_curation_id"]) == str(expected)
    assert len(AnalyzerCurationSelection & sel1) == 1  # no duplicate row
    (AnalyzerCurationSelection & sel1).delete_quick()


@pytest.mark.db_unit
def test_analyzer_curation_selection_blocks_direct_insert(dj_conn):
    """A raw insert1 (no allow_direct_insert) is rejected by the guard.

    The PK is content-addressed, so insert_selection is the only sanctioned
    create path; a direct insert could otherwise land a non-canonical row that
    AnalyzerCuration.populate would compute from before insert_selection flags
    it. The guard raises before any FK check, so no valid upstream is needed.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )

    with pytest.raises(dj.errors.DataJointError, match="insert_selection"):
        AnalyzerCurationSelection().insert1(
            {
                "analyzer_curation_id": uuid.uuid4(),
                "sorting_id": uuid.uuid4(),
                "curation_id": 0,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
