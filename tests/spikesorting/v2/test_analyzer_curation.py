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

import pathlib

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
def test_franklab_default_metric_params_include_full_qc_set(dj_conn):
    """The Frank-lab default metric row computes the full Phase-3 QC surface."""
    from spyglass.spikesorting.v2.metric_curation import QualityMetricParameters

    QualityMetricParameters.insert_default()
    row = (
        QualityMetricParameters & {"metric_params_name": "franklab_default"}
    ).fetch1()

    assert row["metric_names"] == [
        "snr",
        "isi_violation",
        "firing_rate",
        "num_spikes",
        "presence_ratio",
        "amplitude_cutoff",
        "nn_advanced",
    ]
    assert bool(row["skip_pc_metrics"]) is False
    assert "nn_noise_overlap" not in row["metric_names"]
    assert "nn_advanced" in row["metric_kwargs"]
    assert row["metric_kwargs"]["nn_advanced"]["seed"] == 0


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
    child_restriction = CurationV2 & {
        "sorting_id": child["sorting_id"],
        "parent_curation_id": populated_sorting_with_curation["curation_id"],
        "curation_source": "analyzer_curation",
        "description": "auto-curation",
    }
    n_children = len(child_restriction)
    assert AnalyzerCuration().materialize_curation(sel) == child
    assert len(child_restriction) == n_children
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
    # Idempotent re-seed: the single-precision Rule.threshold column round-trips
    # 0.1 / 0.02 as 0.10000000149... so an exact-equality content guard would
    # spuriously raise "different payload" on a second insert_default. This must
    # no-op (the guard compares thresholds with a tolerance).
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
        assert AnalyzerCuration().materialize_curation(sel) == child
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
def test_drift_metric_dependency_computed_not_silently_skipped(dj_conn):
    """Requesting ``drift`` computes its ``spike_locations`` dep, not skips it.

    ``drift`` depends on the ``spike_locations`` extension, which is not in the
    default curation ensure-set. SI's ``compute_quality_metrics`` warns-and-skips
    a metric whose extension is absent, so without deriving the dependency the
    ``drift`` columns were silently missing from the persisted table (and any
    AutoCurationRule thresholding drift would never fire). The drift columns must
    now be present, and the dependency computed on the display analyzer.
    """
    import spikeinterface.full as si

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    rec, sort = si.generate_ground_truth_recording(
        durations=[30.0],
        num_channels=4,
        num_units=3,
        seed=0,
        sampling_frequency=30000.0,
    )
    analyzer = si.create_sorting_analyzer(
        sort, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        ["random_spikes", "noise_levels", "templates", "waveforms"]
    )
    assert not analyzer.has_extension("spike_locations")

    result = AnalyzerCuration._compute_metrics(
        analyzer, None, ["snr", "drift"], {}, True
    )

    # drift emits drift_ptp / drift_mad / drift_std; before deriving the
    # dependency none of these appeared (drift was silently skipped).
    assert any(c.startswith("drift") for c in result.columns)
    assert "snr" in result.columns
    # The dependency was computed on the (display) analyzer it was handed.
    assert analyzer.has_extension("spike_locations")


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


# ---------- waveform-shape (template) metric columns ------------------------
#
# Phase: expose SI template (waveform-shape) output COLUMNS in the per-unit
# metric table for downstream cell typing. The pipeline exposes the columns; it
# ships no cell-type thresholds. These guard: the default surfaced columns, the
# display-analyzer routing, the discoverability helper, and "expose, not
# threshold".


@pytest.mark.db_unit
def test_quality_metric_params_default_template_columns(dj_conn):
    """Every default metric row surfaces the single trough-local shape column.

    A default ``QualityMetricParameters`` row surfaces ``trough_half_width``
    (an SI OUTPUT column) for downstream cell typing -- and nothing else, so the
    post-trough columns that clip on the narrow hippocampus window stay opt-in.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    QualityMetricParameters.insert_default()
    for name in ("franklab_default", "neuropixels_default", "minimal"):
        cols = (
            QualityMetricParameters & {"metric_params_name": name}
        ).fetch1("template_metric_columns")
        assert list(cols) == ["trough_half_width"], name


@pytest.mark.db_unit
def test_available_template_metric_columns(dj_conn):
    """The discoverability classmethod lists SI's single-channel output COLUMNS.

    It includes ``trough_half_width`` (so a user sees the ``half_width`` metric
    is surfaced under that column name) but never the ``half_width`` metric
    name, and excludes multi-channel columns (e.g. ``spread``).
    """
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    cols = QualityMetricParameters.available_template_metric_columns()
    assert "trough_half_width" in cols
    assert "peak_to_trough_duration" in cols
    assert "half_width" not in cols  # a metric NAME, not a column
    assert "spread" not in cols  # multi-channel column, excluded


@pytest.mark.db_unit
def test_compute_metrics_surfaces_template_columns(
    display_and_metric_analyzers,
):
    """_compute_metrics joins the configured shape columns onto the metric frame.

    The default surfaces ``trough_half_width`` alongside ``firing_rate`` (a
    quality metric); the metric NAME ``half_width`` is never a column (columns
    are selected directly), and a non-default column (``recovery_slope``, opt-in)
    is NOT surfaced. Explicitly requesting a column (the opt-in path) surfaces
    it. Values are finite on the real (unwhitened) display analyzer.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.metric_curation import (
        DEFAULT_TEMPLATE_METRIC_COLUMNS,
    )
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, _ = display_and_metric_analyzers
    metrics = AnalyzerCuration._compute_metrics(
        display,
        None,
        ["snr", "firing_rate"],
        {},
        skip_pc_metrics=True,
        job_kwargs={},
        template_metric_columns=list(DEFAULT_TEMPLATE_METRIC_COLUMNS),
    )
    assert {"firing_rate", "trough_half_width"} <= set(metrics.columns)
    assert "half_width" not in metrics.columns  # a metric NAME, never a column
    # Non-default columns are not surfaced unless explicitly requested.
    assert "recovery_slope" not in metrics.columns
    assert "peak_to_trough_duration" not in metrics.columns
    assert np.isfinite(metrics["trough_half_width"].to_numpy()).all()

    # Opt-in path: an explicit request surfaces a non-default column too.
    opt_in = AnalyzerCuration._compute_metrics(
        display,
        None,
        ["snr"],
        {},
        skip_pc_metrics=True,
        job_kwargs={},
        template_metric_columns=["trough_half_width", "peak_to_trough_duration"],
    )
    assert "peak_to_trough_duration" in opt_in.columns


@pytest.mark.db_unit
def test_compute_metrics_surfaces_template_for_pc_only_row(
    display_and_metric_analyzers,
):
    """A PC-only metric row still surfaces its configured shape columns.

    A row requesting only PC/NN metrics (no voltage metric) skips the voltage
    branch that grows the display curation extensions, so without an explicit
    ensure the display ``template_metrics`` extension would be absent and the
    configured ``trough_half_width`` silently dropped. ``_compute_metrics`` grows
    ``template_metrics`` whenever shape columns are requested, independent of
    voltage metrics.
    """
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    # Force the precondition a PC-only sort starts from: no display
    # template_metrics yet (the shared fixture may carry it from a prior test).
    if display.has_extension("template_metrics"):
        display.delete_extension("template_metrics")

    metrics = AnalyzerCuration._compute_metrics(
        display,
        metric,
        ["nn_advanced"],
        {},
        skip_pc_metrics=False,
        job_kwargs={},
        template_metric_columns=["trough_half_width"],
    )
    assert "nn_noise_overlap" in metrics.columns  # the PC metric computed
    assert "trough_half_width" in metrics.columns  # surfaced despite no voltage
    assert display.has_extension("template_metrics")  # the fix grew it


@pytest.mark.db_unit
def test_template_metrics_read_from_display_analyzer(
    display_and_metric_analyzers, monkeypatch
):
    """Shape columns are read from the DISPLAY analyzer, never the whitened one.

    Whitening distorts waveform shape, so a width column on a whitened template
    is meaningless (same reason as ``snr``). Spies ``get_extension`` to record
    which analyzer the ``template_metrics`` read lands on: the display analyzer,
    never the metric (whitened) analyzer.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        DEFAULT_TEMPLATE_METRIC_COLUMNS,
    )
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    display, metric = display_and_metric_analyzers
    reads = []
    original_get_extension = type(display).get_extension

    def _spy(self, name, *args, **kwargs):
        if name == "template_metrics":
            reads.append(
                "display"
                if self is display
                else "metric"
                if self is metric
                else "other"
            )
        return original_get_extension(self, name, *args, **kwargs)

    monkeypatch.setattr(type(display), "get_extension", _spy)
    AnalyzerCuration._compute_metrics(
        display,
        metric,
        ["snr", "nn_advanced"],
        {},
        skip_pc_metrics=False,
        job_kwargs={},
        template_metric_columns=list(DEFAULT_TEMPLATE_METRIC_COLUMNS),
    )
    assert "display" in reads
    assert "metric" not in reads


@pytest.mark.db_unit
def test_no_shipped_rule_thresholds_template_column(dj_conn):
    """No shipped auto-curation rule thresholds a template-metric column.

    The load-bearing decision of this feature is "expose, don't classify": the
    Frank-lab default rule set thresholds ``nn_noise_overlap`` and
    ``isi_violation`` only. A rule that thresholded a region-specific shape
    column would silently mislabel every other region.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        _available_template_metric_columns,
    )
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    template_cols = set(_available_template_metric_columns())
    rule_metrics = {
        rule["metric_name"]
        for _, rules in AutoCurationRules._default_payloads()
        for rule in rules
    }
    leaked = rule_metrics & template_cols
    assert not leaked, leaked


@pytest.mark.db_unit
def test_surface_template_columns_edge_cases(dj_conn, caplog):
    """The defensive branches of the column-surfacing helper behave as documented.

    Drives ``_surface_template_columns`` directly with a controlled
    ``template_metrics`` frame so the no-ops (empty config / absent extension),
    the never-shadow-a-quality-column rule, and the upstream-drift warning (a
    validated column absent from the computed frame) are each exercised --
    branches the real-analyzer happy-path tests do not reach.
    """
    import logging

    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    class _Ext:
        def __init__(self, df):
            self._df = df

        def get_data(self):
            return self._df

    class _Analyzer:
        """Minimal stand-in exposing only the analyzer surface the helper uses."""

        def __init__(self, tm_df):
            self._tm_df = tm_df

        def has_extension(self, name):
            return name == "template_metrics" and self._tm_df is not None

        def get_extension(self, name):
            return _Ext(self._tm_df)

    metrics_df = pd.DataFrame(
        {"snr": [3.0, 4.0]}, index=pd.Index([0, 1], name="unit_id")
    )
    tm_df = pd.DataFrame(
        {"trough_half_width": [0.0004, 0.0003], "snr": [9.9, 9.9]},
        index=pd.Index([0, 1]),
    )
    surface = AnalyzerCuration._surface_template_columns

    # Empty config -> no shape columns added.
    assert list(surface(metrics_df, _Analyzer(tm_df), []).columns) == ["snr"]
    # Extension absent -> the helper defensively no-ops. (Its _compute_metrics
    # caller ensures template_metrics whenever columns are configured, so this
    # is the belt-and-suspenders branch, not a normal path -- the PC-only-row
    # surfacing is covered behaviorally below.)
    assert list(
        surface(metrics_df, _Analyzer(None), ["trough_half_width"]).columns
    ) == ["snr"]
    # Happy path: the requested column is joined by unit id.
    joined = surface(metrics_df, _Analyzer(tm_df), ["trough_half_width"])
    assert joined.loc[0, "trough_half_width"] == 0.0004
    # Never shadow a same-named quality-metric column: the template-frame 'snr'
    # must not overwrite the quality 'snr'.
    assert surface(metrics_df, _Analyzer(tm_df), ["snr"])["snr"].tolist() == [
        3.0,
        4.0,
    ]
    # A validated column absent from the COMPUTED frame -> loud warning, surface
    # the present ones only (upstream-SI-drift signal, not silent).
    with caplog.at_level(logging.WARNING):
        out = surface(
            metrics_df,
            _Analyzer(tm_df),
            ["trough_half_width", "not_computed_col"],
        )
    assert "trough_half_width" in out.columns
    assert "not_computed_col" not in out.columns
    assert any("not_computed_col" in record.message for record in caplog.records)


# ---------- waveform-shape on real biophysical templates (slow) -------------

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _load_recording_and_ground_truth(fixture_name):
    """Load a fixture's recording + ground-truth sorting + per-unit cell types.

    Reads the preprocessed-free recording from the NWB ``e-series`` and the
    planted ground-truth units (spike trains + ``cell_type`` in {E, I}) from the
    sidecar processing module, attaching a 2D probe from the contact geometry as
    the DB pipeline does (SI's >=64-channel path computes multi-channel template
    metrics that assert 2D channel locations). Skips if the fixture is absent.
    """
    import numpy as np
    import pynwb
    import spikeinterface.full as si
    from probeinterface import Probe
    from spikeinterface.extractors import read_nwb_recording

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    path = _FIXTURES_DIR / f"{fixture_name}.nwb"
    if not path.exists():
        pytest.skip(
            f"Generated MEArec fixture {path.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py` first."
        )
    rec = read_nwb_recording(
        file_path=str(path), electrical_series_path="acquisition/e-series"
    )
    fs = rec.get_sampling_frequency()
    xy = np.asarray(rec.get_channel_locations())[:, :2]
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=xy, shapes="circle", shape_params={"radius": 5}
    )
    probe.set_device_channel_indices(np.arange(rec.get_num_channels()))
    rec = rec.set_probe(probe)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        gt_table = get_ground_truth_units_table(io.read())
        assert gt_table is not None, fixture_name
        df = gt_table.to_dataframe()
        cell_types = [str(value).strip().upper() for value in df["cell_type"]]
        trains = {
            idx: np.round(
                np.asarray(df.iloc[idx]["spike_times"]) * fs
            ).astype(np.int64)
            for idx in range(len(df))
        }
    sorting = si.NumpySorting.from_unit_dict(trains, sampling_frequency=fs)
    return rec, sorting, cell_types


def _display_analyzer(rec, sorting, *, ms_before, ms_after):
    """In-memory unwhitened display analyzer with the template_metrics ext."""
    import spikeinterface.full as si

    analyzer = si.create_sorting_analyzer(
        sorting, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        [
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            "template_metrics",
        ],
        extension_params={
            "random_spikes": {"seed": 0},
            "waveforms": {"ms_before": ms_before, "ms_after": ms_after},
        },
    )
    return analyzer


@pytest.mark.slow
@pytest.mark.integration
def test_hippocampus_template_metrics_not_boundary_clipped():
    """The shipped default shape column is not boundary-clipped at 0.5/0.5 ms.

    The hippocampus display recipe is intentionally narrow (0.5 ms before / 0.5
    after) for dense, tight hippocampal spikes. The shipped default,
    ``trough_half_width``, is trough-local: on a representative hippocampal
    (tetrode) fixture its value is finite and the trough plus its two
    half-amplitude crossing points -- the extrema SI uses to compute it -- stay
    strictly interior to the window, so the surfaced default is reliable.

    The same probe documents WHY ``peak_to_trough_duration`` is opt-in, not a
    shipped default: it measures to the post-trough repolarization peak, which on
    this narrow window saturates at the edge-exclusion boundary instead of
    finding a true interior peak. Surfacing it as a default would hand
    downstream a clipped, non-discriminating column on hippocampal sorts.
    """
    import numpy as np
    from spikeinterface.metrics.template.metrics import (
        get_trough_and_peak_idx,
    )

    from spyglass.spikesorting.v2._params.metric_curation import (
        DEFAULT_TEMPLATE_METRIC_COLUMNS,
    )

    assert DEFAULT_TEMPLATE_METRIC_COLUMNS == ["trough_half_width"]

    rec, sorting, _ = _load_recording_and_ground_truth("mearec_tetrode_60s")
    fs = rec.get_sampling_frequency()
    analyzer = _display_analyzer(rec, sorting, ms_before=0.5, ms_after=0.5)
    tm_ext = analyzer.get_extension("template_metrics")
    tm_df = tm_ext.get_data()
    detect_kwargs = {
        key: tm_ext.params[key]
        for key in (
            "min_thresh_detect_peaks_troughs",
            "edge_exclusion_ms",
            "min_peak_trough_distance_ratio",
            "min_extremum_distance_samples",
        )
    }
    waveforms = analyzer.get_extension("waveforms")
    n_samples = waveforms.nbefore + waveforms.nafter
    edge = int(detect_kwargs["edge_exclusion_ms"] / 1000 * fs)
    right_edge = n_samples - edge
    templates = analyzer.get_extension("templates")

    for unit_id in analyzer.unit_ids:
        assert np.isfinite(tm_df.loc[unit_id, "trough_half_width"]), unit_id
        template = templates.get_unit_template(unit_id)  # (n_samples, n_chan)
        extremum_channel = int(np.argmax(np.ptp(template, axis=0)))
        info = get_trough_and_peak_idx(
            template[:, extremum_channel], fs, **detect_kwargs
        )
        # trough_half_width's extrema (the trough + its half-amplitude crossings)
        # are strictly interior -> the shipped default is not clipped.
        for key in (
            "trough_index",
            "trough_half_width_left",
            "trough_half_width_right",
        ):
            index = info[key]
            assert 0 < index < n_samples - 1, (unit_id, key, index, n_samples)
        # Rationale guard: the repolarization peak peak_to_trough_duration would
        # measure to is pinned to the window's edge-exclusion boundary on this
        # narrow recipe -- the documented reason that column is opt-in here.
        assert info["peak_after_index"] >= (
            right_edge - detect_kwargs["min_extremum_distance_samples"]
        ), (unit_id, info["peak_after_index"], right_edge)


@pytest.mark.slow
@pytest.mark.integration
def test_mearec_celltype_metrics_separable():
    """Ground-truth E vs I units separate on trough_half_width x firing_rate.

    A descriptive justification for EXPOSING (not thresholding) waveform-shape
    metrics: on the MEArec smoke fixture's biophysical templates, inhibitory
    interneurons fire faster and have narrower-or-equal spikes than excitatory
    cells, so the joint rate x width space separates the two groups. This is a
    margin on the two groups, NOT a baked pipeline threshold (cutoffs are
    region-specific and live downstream/user-side).
    """
    import numpy as np

    rec, sorting, cell_types = _load_recording_and_ground_truth(
        "mearec_polymer_smoke"
    )
    analyzer = _display_analyzer(rec, sorting, ms_before=1.0, ms_after=2.0)
    tm_df = analyzer.get_extension("template_metrics").get_data()
    duration = rec.get_total_duration()

    by_type = {"E": {"thw": [], "fr": []}, "I": {"thw": [], "fr": []}}
    for unit_id in analyzer.unit_ids:
        cell_type = cell_types[int(unit_id)]
        trough_half_width = float(tm_df.loc[unit_id, "trough_half_width"])
        firing_rate = (
            len(analyzer.sorting.get_unit_spike_train(unit_id)) / duration
        )
        assert np.isfinite(trough_half_width), unit_id
        by_type[cell_type]["thw"].append(trough_half_width)
        by_type[cell_type]["fr"].append(firing_rate)

    assert by_type["E"]["fr"] and by_type["I"]["fr"], cell_types
    # Interneuron signature: every I unit fires faster than every E unit, and
    # the I group's mean spike is narrower-or-equal -- the groups separate in
    # the joint (rate, width) space the surfaced columns expose.
    assert min(by_type["I"]["fr"]) > max(by_type["E"]["fr"])
    assert np.mean(by_type["I"]["thw"]) <= np.mean(by_type["E"]["thw"])
