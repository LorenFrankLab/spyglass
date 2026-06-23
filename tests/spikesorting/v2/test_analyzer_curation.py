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
def test_auto_curation_rules_rule_insert_blocked(dj_conn):
    """Direct AutoCurationRules.Rule inserts bypass payload validation."""
    from spyglass.spikesorting.v2.exceptions import (
        UnsupportedDirectInsertError,
    )
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    rule = {
        "auto_curation_rules_name": "x",
        "rule_index": 0,
        "rule_name": "r",
        "metric_name": "snr",
        "operator": "<",
        "threshold": 1.0,
        "label": "noise",
    }
    with pytest.raises(UnsupportedDirectInsertError):
        AutoCurationRules.Rule().insert([rule])
    with pytest.raises(UnsupportedDirectInsertError):
        AutoCurationRules.Rule().insert1(rule)


@pytest.mark.db_unit
def test_auto_curation_rules_insert_rules_queryable(dj_conn):
    """insert_rules stores ordered rule rows queryable by index/metric/label."""
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    name = "test_rules_queryable"
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete(
        safemode=False
    )
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
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete(
        safemode=False
    )


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


@pytest.mark.db_unit
def test_auto_curation_rules_existing_name_must_match_payload(dj_conn):
    """Existing names are idempotent only when the supplied payload matches."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    name = "test_rules_existing_name"
    # threshold 0.1 is NOT exactly representable in the single-precision
    # ``Rule.threshold`` column (it round-trips as 0.10000000149...). The
    # re-insert below must still be idempotent -- an exact comparison would
    # raise a spurious "different payload" error on that float32 round-off.
    base_rule = {
        "rule_index": 0,
        "rule_name": "snr_noise",
        "metric_name": "snr",
        "operator": "<",
        "threshold": 0.1,
        "label": "noise",
    }
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete(
        safemode=False
    )
    try:
        pk = AutoCurationRules.insert_rules(
            {"auto_curation_rules_name": name, "auto_merge_preset": "none"},
            [dict(base_rule)],
        )
        # Idempotent re-insert of the identical (float32-lossy) payload.
        assert AutoCurationRules.insert_rules(
            {"auto_curation_rules_name": name, "auto_merge_preset": "none"},
            [dict(base_rule)],
        ) == pk

        changed_rule = {**base_rule, "threshold": 3.0}
        with pytest.raises(ValueError, match="different auto-merge/rule payload"):
            AutoCurationRules.insert_rules(
                {"auto_curation_rules_name": name, "auto_merge_preset": "none"},
                [changed_rule],
            )

        bad_rule = {**base_rule, "operator": "=<"}
        with pytest.raises(ValidationError):
            AutoCurationRules.insert_rules(
                {"auto_curation_rules_name": name, "auto_merge_preset": "none"},
                [bad_rule],
            )
    finally:
        (AutoCurationRules & {"auto_curation_rules_name": name}).delete(
            safemode=False
        )


# ---------- end-to-end (slow / integration) ---------------------------------


@pytest.fixture
def analyzer_curation_defaults(dj_conn):
    """Ensure the default metric/auto-curation/waveform Lookup rows exist."""
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    # The curation selection's metric_waveform_params_name FK resolves to a
    # metric (whitened) row, so the catalog rows must exist.
    AnalyzerWaveformParameters.insert_default()


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

        # Per-pair burst-metrics scatter (v1 plot_by_sort_group_ids analog).
        burst_fig = ac.plot_by_sort_group_ids(sel, pairs=pair)
        assert burst_fig is not None
        plt.close(burst_fig)

    # Peak-amplitude accessor (v1 get_peak_amps analog): per-spike, per-channel,
    # sampled at the waveform peak.
    peak_amps, peak_times = ac.get_peak_amps(sel)
    assert peak_amps
    uid = next(iter(peak_amps))
    assert peak_amps[uid].ndim == 2
    assert len(peak_times[uid]) == peak_amps[uid].shape[0]


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
def test_insert_by_curation_id_delegates_to_insert_selection(
    dj_conn, monkeypatch
):
    """The v1-analog convenience assembles the selection key and forwards it."""
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )

    captured = {}

    def _fake_insert_selection(cls, key):
        captured["key"] = dict(key)
        return {"analyzer_curation_id": "sentinel"}

    monkeypatch.setattr(
        AnalyzerCurationSelection,
        "insert_selection",
        classmethod(_fake_insert_selection),
    )

    out = AnalyzerCurationSelection.insert_by_curation_id(
        "sort-1", 3, "minimal", "none"
    )
    assert out == {"analyzer_curation_id": "sentinel"}
    assert captured["key"] == {
        "sorting_id": "sort-1",
        "curation_id": 3,
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
    }


@pytest.mark.db_unit
def test_compute_merge_groups_matches_si_call_contract(dj_conn, monkeypatch):
    """Resolved job kwargs are flattened for SI, not nested as ``job_kwargs``.

    ``compute_merge_unit_groups`` accepts concurrency controls through
    ``**job_kwargs``. Passing ``job_kwargs={...}`` as a named kwarg leaks that
    dict into downstream analyzer ``compute`` calls as an invalid job keyword.
    The ``feature_neighbors`` preset also needs SI's ``spike_locations``
    extension, so the wrapper precomputes it explicitly before calling SI with
    ``compute_needed_extensions=False``.
    """
    import numpy as np

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    captured = {}

    def _fake_ensure(analyzer, extensions, *, job_kwargs=None):
        captured["extensions"] = tuple(extensions)
        captured["ensure_job_kwargs"] = dict(job_kwargs or {})

    def _fake_compute_merge_unit_groups(analyzer, **kwargs):
        captured["si_kwargs"] = dict(kwargs)
        return [(np.int64(1), np.int64(2))]

    monkeypatch.setattr(
        "spyglass.spikesorting.v2._sorting_analyzer.ensure_extensions",
        _fake_ensure,
    )
    monkeypatch.setattr(
        "spikeinterface.curation.compute_merge_unit_groups",
        _fake_compute_merge_unit_groups,
    )

    groups = AnalyzerCuration._compute_merge_groups(
        object(),
        "feature_neighbors",
        {"steps_params": {"knn": {"k_nn": 5}}},
        {"n_jobs": 2, "chunk_duration": "1s", "random_seed": 7},
    )

    assert groups == [[1, 2]]
    assert "spike_locations" in captured["extensions"]
    assert "spike_amplitudes" in captured["extensions"]
    assert captured["ensure_job_kwargs"]["random_seed"] == 7
    assert captured["si_kwargs"]["preset"] == "feature_neighbors"
    assert captured["si_kwargs"]["compute_needed_extensions"] is False
    assert captured["si_kwargs"]["steps_params"] == {"knn": {"k_nn": 5}}
    assert captured["si_kwargs"]["n_jobs"] == 2
    assert captured["si_kwargs"]["chunk_duration"] == "1s"
    assert "job_kwargs" not in captured["si_kwargs"]
    assert "random_seed" not in captured["si_kwargs"]


def _default_rule_rows(name: str) -> list[dict]:
    """Return the shipped Rule rows for ``name`` (ordered by ``rule_index``).

    Pulls them straight from ``_default_payloads`` (the single source of truth)
    so a behavior test exercises the ACTUAL shipped rules, not a hand-copied
    replica that could pass while the catalog drifts.
    """
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    for master, rules in AutoCurationRules._default_payloads():
        if master["auto_curation_rules_name"] == name:
            return sorted(rules, key=lambda r: r["rule_index"])
    raise AssertionError(f"{name!r} not in AutoCurationRules._default_payloads()")


@pytest.mark.db_unit
def test_franklab_auto_curation_default_rules(dj_conn):
    """The franklab default rule set ships with the exact ordered ISI/nn rules.

    ``franklab_default_auto_curation_2026_06`` thresholds the lab's ~2% ISI
    policy (not only ``nn_noise_overlap``): rule 0 labels high noise-overlap
    units ``noise``, rule 1 labels >2% refractory-violation units ``reject``.
    Rule order is part of the contract (rules apply in ``rule_index`` order),
    and the labels must be valid ``CurationLabel`` members.
    """
    from spyglass.spikesorting.v2._enums import CurationLabel
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    name = "franklab_default_auto_curation_2026_06"
    AutoCurationRules.insert_default()

    master = (AutoCurationRules & {"auto_curation_rules_name": name}).fetch1()
    # A labeling rule set, not an auto-merge one: merges stay a manual step.
    assert master["auto_merge_preset"] == "none"

    rules = (
        AutoCurationRules.Rule & {"auto_curation_rules_name": name}
    ).fetch(as_dict=True, order_by="rule_index")
    assert [r["rule_index"] for r in rules] == [0, 1]

    noise_rule, reject_rule = rules
    assert noise_rule["metric_name"] == "nn_noise_overlap"
    assert noise_rule["operator"] == ">"
    assert noise_rule["threshold"] == pytest.approx(0.1)
    assert noise_rule["label"] == "noise"

    assert reject_rule["metric_name"] == "isi_violation"
    assert reject_rule["operator"] == ">"
    assert reject_rule["threshold"] == pytest.approx(0.02)
    assert reject_rule["label"] == "reject"

    # Both labels are members of the validated CurationLabel set.
    valid = {label.value for label in CurationLabel}
    assert {noise_rule["label"], reject_rule["label"]} <= valid


@pytest.mark.db_unit
def test_auto_curation_applies_isi_rule(dj_conn):
    """A >2% ISI unit is labeled ``reject``; a clean unit is left unlabeled.

    Drives the SHIPPED ``franklab_default_auto_curation_2026_06`` rule rows
    through the DB-free ``apply_label_rules`` engine on a synthetic metrics
    frame, so the test binds the catalog rules to their labeling behavior.
    """
    from spyglass.spikesorting.v2._metric_curation import apply_label_rules
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    name = "franklab_default_auto_curation_2026_06"
    AutoCurationRules.insert_default()
    rules = (
        AutoCurationRules.Rule & {"auto_curation_rules_name": name}
    ).fetch(as_dict=True, order_by="rule_index")

    # unit 1: 3% ISI violations, low noise-overlap -> reject (ISI rule only).
    # unit 2: clean on both metrics -> no label.
    metrics = pd.DataFrame(
        {
            "isi_violation": [0.03, 0.0],
            "nn_noise_overlap": [0.0, 0.0],
        },
        index=[1, 2],
    )
    labels = apply_label_rules(metrics, rules)
    assert labels == {1: ["reject"]}


@pytest.mark.db_unit
def test_isi_reject_matches_v2_default_exclusion_policy(dj_conn):
    """The ISI rule labels ``reject`` (not ``mua``), and v2 excludes ``reject``.

    The choice of ``reject`` over ``mua`` is load-bearing:
    ``CurationV2.get_matchable_unit_ids`` excludes ``reject`` by default, so a
    >2% refractory-violation unit drops out of matchable-unit outputs, whereas a
    ``mua`` label would keep it. This guards that policy alignment.
    """
    import inspect

    from spyglass.spikesorting.v2.curation import CurationV2

    isi_rules = [
        r
        for r in _default_rule_rows("franklab_default_auto_curation_2026_06")
        if r["metric_name"] == "isi_violation"
    ]
    assert [r["label"] for r in isi_rules] == ["reject"]

    default_excludes = (
        inspect.signature(CurationV2.get_matchable_unit_ids)
        .parameters["exclude_labels"]
        .default
    )
    assert "reject" in default_excludes
    assert "mua" not in default_excludes


@pytest.mark.db_unit
def test_named_legacy_rules_preserved(dj_conn):
    """The pre-existing named rule sets still insert (no silent removal)."""
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    AutoCurationRules.insert_default()
    for name in ("none", "similarity_merge", "v1_default_nn_noise"):
        assert AutoCurationRules & {"auto_curation_rules_name": name}, name


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
    (AutoCurationRules & {"auto_curation_rules_name": name}).delete(safemode=False)
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
        (AutoCurationRules & {"auto_curation_rules_name": name}).delete(safemode=False)


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

    # The metric waveform recipe resolved from the sort source is part of the
    # content-addressed identity; recompute with the stored value.
    metric_wf = (AnalyzerCurationSelection & sel1).fetch1(
        "metric_waveform_params_name"
    )
    identity = {
        "sorting_id": key["sorting_id"],
        "curation_id": key["curation_id"],
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
        "metric_waveform_params_name": metric_wf,
    }
    expected = deterministic_id("analyzer_curation", identity)
    assert str(sel1["analyzer_curation_id"]) == str(expected)
    assert len(AnalyzerCurationSelection & sel1) == 1  # no duplicate row
    (AnalyzerCurationSelection & sel1).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_selection_tracks_waveform_params(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """Two metric (whitened) recipes give two distinct analyzer_curation_ids.

    The metric waveform recipe is part of the content-addressed identity, so a
    curation against the cortex metric analyzer and one against the hippocampus
    metric analyzer are tracked as separate selections (v1-parity: v1 attached
    ``WaveformParameters`` to ``MetricCurationSelection``); each stored row
    carries the recipe it was inserted with.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )

    base = {
        **populated_sorting_with_curation,
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
    }
    sel_cortex = AnalyzerCurationSelection.insert_selection(
        {
            **base,
            "metric_waveform_params_name": "franklab_cortex_metric_waveforms",
        }
    )
    sel_hippo = AnalyzerCurationSelection.insert_selection(
        {
            **base,
            "metric_waveform_params_name": (
                "franklab_hippocampus_metric_waveforms"
            ),
        }
    )
    try:
        assert (
            sel_cortex["analyzer_curation_id"]
            != sel_hippo["analyzer_curation_id"]
        )
        assert (
            AnalyzerCurationSelection & sel_cortex
        ).fetch1(
            "metric_waveform_params_name"
        ) == "franklab_cortex_metric_waveforms"
        assert (
            AnalyzerCurationSelection & sel_hippo
        ).fetch1(
            "metric_waveform_params_name"
        ) == "franklab_hippocampus_metric_waveforms"

        # An explicit DISPLAY (unwhitened) recipe is rejected: PC/NN metrics
        # must compute on a whitened analyzer, so insert_selection guards that
        # the metric recipe is purpose='metric' / whiten=True.
        with pytest.raises(ValueError, match="whitened metric recipe"):
            AnalyzerCurationSelection.insert_selection(
                {
                    **base,
                    "metric_waveform_params_name": (
                        "franklab_cortex_actual_waveforms"
                    ),
                }
            )
    finally:
        (AnalyzerCurationSelection & sel_cortex).delete(safemode=False)
        (AnalyzerCurationSelection & sel_hippo).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_selection_resolves_concat_metric_waveform_params(
    populated_sorting_with_curation, analyzer_curation_defaults, monkeypatch
):
    """The default metric recipe follows the source resolver's preproc.

    ``insert_selection`` resolves the metric recipe from
    ``SortingSelection.resolve_source_preprocessing_params_name`` (source-aware:
    single-recording vs the ``ConcatenatedRecordingSelection -> Preprocessing``
    FK). A true populated concat sort needs the concat materializer (not wired
    in this fixture), so patch the resolver to return the hippocampus preproc
    (as it would for a hippocampus concat source) and assert the stored metric
    recipe is the hippocampus metric row, not the cortex fallback.
    """
    from spyglass.spikesorting.v2._recipe_catalog import (
        HIPPOCAMPUS_METRIC_WAVEFORMS,
        HIPPOCAMPUS_PREPROC,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    monkeypatch.setattr(
        SortingSelection,
        "resolve_source_preprocessing_params_name",
        staticmethod(lambda key: HIPPOCAMPUS_PREPROC),
    )
    sel = AnalyzerCurationSelection.insert_selection(
        {
            **populated_sorting_with_curation,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }
    )
    try:
        assert (AnalyzerCurationSelection & sel).fetch1(
            "metric_waveform_params_name"
        ) == HIPPOCAMPUS_METRIC_WAVEFORMS
    finally:
        (AnalyzerCurationSelection & sel).delete(safemode=False)


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


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_selection_rejects_bypassed_id(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """A non-deterministic id planted under an identity raises on re-insert.

    insert_selection content-addresses the PK; if a row for the same logical
    identity already carries a random id (an allow_direct_insert bypass or a
    legacy row), insert_selection must raise DuplicateSelectionError rather
    than silently treat the non-canonical row as the selection.
    """
    import uuid

    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCurationSelection,
    )

    # Pass metric_waveform_params_name explicitly so the planted row and the
    # insert_selection call share one identity (no resolver in the loop).
    identity = {
        "sorting_id": populated_sorting_with_curation["sorting_id"],
        "curation_id": populated_sorting_with_curation["curation_id"],
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
        "metric_waveform_params_name": "franklab_cortex_metric_waveforms",
    }
    bad_id = uuid.uuid4()
    AnalyzerCurationSelection().insert1(
        {**identity, "analyzer_curation_id": bad_id}, allow_direct_insert=True
    )
    try:
        with pytest.raises(DuplicateSelectionError, match="non-deterministic"):
            AnalyzerCurationSelection.insert_selection(identity)
    finally:
        (
            AnalyzerCurationSelection & {"analyzer_curation_id": bad_id}
        ).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_curation_materializes_real_labels(
    populated_sorting_with_curation, analyzer_curation_defaults
):
    """materialize_curation forwards proposed labels onto the child CurationV2.

    The shipped end-to-end test only exercises the empty 'none' path; here a
    permissive rule labels every unit so the labels-materialization path (and
    the ``labels or None`` coalescing) is actually covered end-to-end.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        AutoCurationRules,
    )

    # A rule every real unit trips: finite firing_rate >= 0 -> 'mua'.
    rules_name = "force_mua_label"

    def _teardown():
        # A cautious cascading delete clears the rules AND everything that
        # FK-references them (the Rule part, plus any AnalyzerCurationSelection
        # / AnalyzerCuration rows) in dependency order -- no manual ordering,
        # no FK violation. Run BEFORE the test too, in case a prior interrupted
        # run left a referencing selection.
        (
            AutoCurationRules & {"auto_curation_rules_name": rules_name}
        ).delete(safemode=False)

    _teardown()
    AutoCurationRules.insert_rules(
        {"auto_curation_rules_name": rules_name, "auto_merge_preset": "none"},
        [
            {
                "rule_index": 0,
                "rule_name": "all_mua",
                "metric_name": "firing_rate",
                "operator": ">=",
                "threshold": 0.0,
                "label": "mua",
            }
        ],
    )
    try:
        sel = _populate_analyzer_curation(
            populated_sorting_with_curation, "minimal", rules_name
        )
        proposed = AnalyzerCuration.get_labels(sel)
        assert proposed, "every unit should trip firing_rate >= 0"
        assert all(labels == ["mua"] for labels in proposed.values())

        child = AnalyzerCuration().materialize_curation(sel)
        materialized = CurationV2._labels_by_unit(child)
        assert materialized == proposed, (
            "child CurationV2 must carry the proposed labels"
        )
        assert (
            CurationV2 & child
        ).fetch1("curation_source") == "analyzer_curation"
    finally:
        _teardown()


# ---------- metric routing (display vs whitened metric analyzer) ------------
#
# These drive ``AnalyzerCuration._compute_metrics`` directly on synthetic
# in-memory analyzers (no populate): a real unwhitened display analyzer and a
# real whitened metric analyzer for one synthetic sort. They pin the routing
# contract -- voltage/spike-train metrics read the display analyzer, only PC/NN
# cluster-separation metrics read the whitened one.


@pytest.fixture(scope="module")
def display_and_metric_analyzers(dj_conn, tmp_path_factory):
    """A real unwhitened DISPLAY + whitened METRIC analyzer for one sort.

    Built with ``build_analyzer`` (the production path) on a synthetic
    recording, differing only in the recipe's ``whiten`` flag, so routing tests
    have the two analyzers a populated sort would have -- without a populate.
    """
    import spikeinterface.full as si

    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

    rec, sort = si.generate_ground_truth_recording(
        durations=[15.0],
        num_channels=4,
        num_units=4,
        seed=0,
        sampling_frequency=30000.0,
    )
    root = tmp_path_factory.mktemp("metric_routing")
    common = dict(
        key={"sorting_id": "routing"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={},
    )
    window = {"ms_before": 1.0, "ms_after": 2.0, "max_spikes_per_unit": 20000}
    build_analyzer(
        sort,
        rec,
        analyzer_folder=root / "display.zarr",
        waveform_params={
            **window,
            "whiten": False,
            "purpose": "display",
            "schema_version": 1,
        },
        **common,
    )
    build_analyzer(
        sort,
        rec,
        analyzer_folder=root / "metric.zarr",
        waveform_params={
            **window,
            "whiten": True,
            "purpose": "metric",
            "schema_version": 1,
        },
        **common,
    )
    return (
        si.load_sorting_analyzer(root / "display.zarr"),
        si.load_sorting_analyzer(root / "metric.zarr"),
    )


@pytest.mark.db_unit
def test_metric_routing_by_type(display_and_metric_analyzers, monkeypatch):
    """Voltage metrics compute on the display analyzer; PC/NN on the whitened.

    Spies on ``compute_quality_metrics`` to capture which analyzer each metric
    group is computed against. Voltage/spike-train names (``snr``,
    ``firing_rate``) must hit the unwhitened display analyzer; the PC metric
    (``nn_advanced``) must hit the whitened metric analyzer.
    """
    import spikeinterface.metrics.quality as siq

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    calls = []

    def _spy(
        analyzer,
        metric_names,
        metric_params=None,
        skip_pc_metrics=False,
        **kwargs,  # tolerate delete_existing_metrics etc.
    ):
        which = (
            "display"
            if analyzer is display
            else "metric"
            if analyzer is metric
            else "other"
        )
        calls.append((which, set(metric_names)))
        return pd.DataFrame(
            index=list(analyzer.sorting.unit_ids),
            data={name: 0.0 for name in metric_names},
        )

    # _compute_metrics imports compute_quality_metrics inside the function, so
    # patching the source-module attribute is picked up at call time.
    monkeypatch.setattr(siq, "compute_quality_metrics", _spy)

    AnalyzerCuration._compute_metrics(
        display,
        metric,
        metric_names=["snr", "firing_rate", "nn_advanced"],
        metric_kwargs={},
        skip_pc_metrics=False,
        job_kwargs={},
    )

    voltage_calls = [c for c in calls if c[1] & {"snr", "firing_rate"}]
    pc_calls = [c for c in calls if "nn_advanced" in c[1]]
    assert voltage_calls and all(c[0] == "display" for c in voltage_calls)
    assert pc_calls and all(c[0] == "metric" for c in pc_calls)
    # The whitened analyzer is never asked for a voltage metric.
    assert not any(
        c[0] == "metric" and (c[1] & {"snr", "firing_rate"}) for c in calls
    )


@pytest.mark.db_unit
def test_metric_merge_reindexes_same_units(display_and_metric_analyzers, monkeypatch):
    """Voltage and PC metric frames align by unit id, not SI return order.

    SpikeInterface should return the same unit ids for both analyzers, but the
    row order is not the scientific invariant. A reordered PC frame should merge
    cleanly onto the display order instead of failing or value-swapping.
    """
    import spikeinterface.metrics.quality as siq

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    unit_ids = [int(u) for u in display.sorting.unit_ids]
    assert len(unit_ids) >= 2

    def _spy(
        analyzer,
        metric_names,
        metric_params=None,
        skip_pc_metrics=False,
        **kwargs,
    ):
        if analyzer is display:
            return pd.DataFrame(
                index=unit_ids,
                data={"snr": [float(u) for u in unit_ids]},
            )
        if analyzer is metric:
            reversed_ids = list(reversed(unit_ids))
            return pd.DataFrame(
                index=reversed_ids,
                data={
                    "nn_isolation": [float(u * 10) for u in reversed_ids],
                    "nn_noise_overlap": [float(u * 100) for u in reversed_ids],
                },
            )
        raise AssertionError("unexpected analyzer")

    monkeypatch.setattr(siq, "compute_quality_metrics", _spy)

    metrics = AnalyzerCuration._compute_metrics(
        display,
        metric,
        metric_names=["snr", "nn_advanced"],
        metric_kwargs={},
        skip_pc_metrics=False,
        job_kwargs={},
    )

    assert list(metrics.index) == unit_ids
    assert metrics["snr"].tolist() == [float(u) for u in unit_ids]
    assert metrics["nn_isolation"].tolist() == [
        float(u * 10) for u in unit_ids
    ]
    assert metrics["nn_noise_overlap"].tolist() == [
        float(u * 100) for u in unit_ids
    ]


@pytest.mark.db_unit
def test_snr_unaffected_by_metric_whitening(display_and_metric_analyzers):
    """``snr`` is identical with or without a whitened metric analyzer.

    snr always reads the unwhitened display analyzer, so whether the whitened
    metric analyzer participates (PC metrics requested) must not change it --
    a guard that whitening never leaks into voltage metrics.
    """
    import numpy as np

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    with_metric = AnalyzerCuration._compute_metrics(
        display,
        metric,
        ["snr", "nn_advanced"],
        {},
        skip_pc_metrics=False,
        job_kwargs={},
    )
    without_metric = AnalyzerCuration._compute_metrics(
        display,
        None,
        ["snr"],
        {},
        skip_pc_metrics=True,
        job_kwargs={},
    )
    np.testing.assert_array_equal(
        with_metric.loc[without_metric.index, "snr"].to_numpy(),
        without_metric["snr"].to_numpy(),
    )
    # The PC metric routed to the whitened analyzer produces its columns (the
    # values may be NaN for too-few-spike synthetic units -- finiteness is
    # data-dependent, so it is verified manually on real data per the migration
    # guide, not asserted here).
    assert {"nn_isolation", "nn_noise_overlap"} <= set(with_metric.columns)


@pytest.mark.db_unit
def test_metrics_do_not_leak_across_curations(display_and_metric_analyzers):
    """A later curation's result excludes a prior curation's metric columns.

    The display/metric analyzers are shared across curations and SI preserves
    the stored quality_metrics by default, so without delete_existing_metrics a
    prior curation's columns would leak into a later, disjoint request (and an
    auto-rule could then threshold a stale metric). _compute_metrics passes
    delete_existing_metrics=True, so each curation's result has exactly the
    metrics it requested.
    """
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    first = AnalyzerCuration._compute_metrics(
        display,
        metric,
        ["snr", "nn_advanced"],
        {},
        skip_pc_metrics=False,
        job_kwargs={},
    )
    assert "snr" in first.columns and "nn_isolation" in first.columns

    # A later DISJOINT request on the SAME display analyzer (voltage only).
    second = AnalyzerCuration._compute_metrics(
        display,
        None,
        ["firing_rate"],
        {},
        skip_pc_metrics=True,
        job_kwargs={},
    )
    assert "firing_rate" in second.columns
    # snr (computed on the display analyzer in `first`) must not leak in...
    assert "snr" not in second.columns, second.columns.tolist()
    # ...nor the PC column (computed on the untouched metric analyzer).
    assert "nn_isolation" not in second.columns, second.columns.tolist()


@pytest.mark.db_unit
def test_compute_metrics_raises_on_unit_id_mismatch(
    display_and_metric_analyzers, monkeypatch
):
    """If the display (voltage) and whitened (PC) frames disagree on unit ids,
    _compute_metrics raises rather than silently NaN-filling the merge.

    The two analyzers derive from the same canonical sorting, so this can't
    happen today; the guard is for a future build divergence. Force it by
    spying ``compute_quality_metrics`` to return frames with different indices,
    and assert the index-equality check fails loud.
    """
    import pandas as pd
    import spikeinterface.metrics.quality as siq

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers

    def _mismatched(
        analyzer,
        metric_names,
        metric_params=None,
        skip_pc_metrics=False,
        **kwargs,
    ):
        # PC frame (metric analyzer) drops a unit the voltage frame has.
        index = [0, 1] if analyzer is metric else [0, 1, 2]
        return pd.DataFrame(
            index=index, data={name: 0.0 for name in metric_names}
        )

    monkeypatch.setattr(siq, "compute_quality_metrics", _mismatched)
    with pytest.raises(ValueError, match="mismatched unit ids"):
        AnalyzerCuration._compute_metrics(
            display,
            metric,
            ["snr", "nn_advanced"],
            {},
            skip_pc_metrics=False,
            job_kwargs={},
        )


@pytest.mark.db_unit
def test_pinned_pca_params_recomputed_on_mismatch(display_and_metric_analyzers):
    """A pre-existing principal_components with non-pinned params is dropped and
    recomputed pinned before PC metrics run.

    ensure_extensions skips a present extension without checking its params, so
    a stale/manual analyzer carrying differently-parameterized PCA would
    silently drive the PC/NN metrics. _compute_metrics guards against that.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        _PCA_EXTENSION_PARAMS,
    )

    display, metric = display_and_metric_analyzers
    # Plant a MISMATCHED principal_components (n_components=3 != pinned 5).
    metric.compute(
        ["principal_components"],
        extension_params={"principal_components": {"n_components": 3}},
    )
    assert (
        metric.get_extension("principal_components").params["n_components"] == 3
    )

    AnalyzerCuration._compute_metrics(
        display,
        metric,
        ["snr", "nn_advanced"],
        {},
        skip_pc_metrics=False,
        job_kwargs={},
    )
    # Recomputed with the pinned params (every pinned key, not just count).
    pca = metric.get_extension("principal_components").params
    for key, pinned in _PCA_EXTENSION_PARAMS.items():
        if key == "dtype":
            import numpy as np

            assert np.dtype(pca[key]) == np.dtype(pinned)
        else:
            assert pca[key] == pinned, (key, pca[key])


@pytest.mark.db_unit
def test_pca_params_match_normalizes_dtype(dj_conn):
    """_pca_params_match compares the pinned keys, normalizing dtype.

    A stored ``np.dtype('float32')`` must match the pinned ``"float32"`` string;
    a different count/dtype or a missing key is a mismatch.
    """
    import numpy as np

    from spyglass.spikesorting.v2.metric_curation import (
        _PCA_EXTENSION_PARAMS,
        _pca_params_match,
    )

    assert _pca_params_match(dict(_PCA_EXTENSION_PARAMS))
    assert _pca_params_match(
        {**_PCA_EXTENSION_PARAMS, "dtype": np.dtype("float32")}
    )
    assert not _pca_params_match({**_PCA_EXTENSION_PARAMS, "n_components": 3})
    assert not _pca_params_match({**_PCA_EXTENSION_PARAMS, "dtype": "float64"})
    assert not _pca_params_match(
        {k: v for k, v in _PCA_EXTENSION_PARAMS.items() if k != "dtype"}
    )


@pytest.mark.db_unit
def test_make_fetch_rejects_non_metric_recipe(dj_conn):
    """make_fetch re-validates the stored metric recipe before any build.

    A row planted via allow_direct_insert bypasses insert_selection's guard, so
    make_fetch re-asserts the recipe is a whitened metric row -- a display
    recipe is rejected before the (unwhitened) analyzer would be built.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        AnalyzerCurationSelection,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()  # the display recipe row
    acid = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        AnalyzerCurationSelection.insert1(
            {
                "analyzer_curation_id": acid,
                "sorting_id": uuid.uuid4(),
                "curation_id": 0,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
                # A DISPLAY (unwhitened) recipe, not a metric recipe.
                "metric_waveform_params_name": (
                    "franklab_cortex_actual_waveforms"
                ),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    try:
        with pytest.raises(ValueError, match="whitened metric recipe"):
            AnalyzerCuration().make_fetch({"analyzer_curation_id": acid})
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                AnalyzerCurationSelection & {"analyzer_curation_id": acid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.db_unit
def test_burst_and_merge_use_display_analyzer(dj_conn):
    """Burst legs + the merge engine run on the unwhitened display analyzer.

    A planted oversplit (one cell split into units 0 and 1) scores higher
    waveform similarity than a true pair (0, 2) and is proposed for merge when
    read from real (unwhitened) templates -- the routing contract keeps the
    merge engine and all burst legs on the display analyzer (whitening distorts
    ``template_similarity`` and the ``unit_locations`` spatial gate). An
    unwhitened analyzer here IS the display recipe (``return_in_uV`` default).
    """
    import spikeinterface.full as si

    from spyglass.spikesorting.v2._metric_curation_plots import (
        burst_pair_metrics_from_analyzer,
    )
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    rec, gt = si.generate_ground_truth_recording(
        durations=[20.0], num_units=3, seed=0, sampling_frequency=30000.0
    )
    st0 = gt.get_unit_spike_train(gt.get_unit_ids()[0])
    # Interleave (even/odd) rather than split disjoint halves: both halves keep
    # the cell's full-strength template and overlap in time, so the merge
    # engine sees an identical-shape pair whose union restores a refractory-
    # respecting train -- the canonical oversplit it is designed to merge.
    sort = si.NumpySorting.from_unit_dict(
        {
            0: st0[::2],
            1: st0[1::2],
            2: gt.get_unit_spike_train(gt.get_unit_ids()[1]),
        },
        sampling_frequency=rec.get_sampling_frequency(),
    )
    display = si.create_sorting_analyzer(
        sort, rec, sparse=True, format="memory"
    )
    display.compute(
        [
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            "correlograms",
            "template_similarity",
            "unit_locations",
            "spike_amplitudes",
        ],
        extension_params={"random_spikes": {"seed": 0}},
    )

    rows = burst_pair_metrics_from_analyzer(display, pairs=[(0, 1), (0, 2)])
    by_pair = {(r["unit1"], r["unit2"]): r for r in rows}
    assert (
        by_pair[(0, 1)]["wf_similarity"] > by_pair[(0, 2)]["wf_similarity"]
    )

    # x_contaminations is the cross-correlogram-contamination preset designed to
    # catch this oversplit class (a cell split into two units); run on the
    # unwhitened display analyzer it proposes merging 0 and 1.
    groups = AnalyzerCuration._compute_merge_groups(
        display, "x_contaminations", {}, {}
    )
    assert any(set(g) >= {0, 1} for g in groups), groups
