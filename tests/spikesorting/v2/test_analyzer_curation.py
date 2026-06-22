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

    identity = {
        "sorting_id": key["sorting_id"],
        "curation_id": key["curation_id"],
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
    }
    expected = deterministic_id("analyzer_curation", identity)
    assert str(sel1["analyzer_curation_id"]) == str(expected)
    assert len(AnalyzerCurationSelection & sel1) == 1  # no duplicate row
    (AnalyzerCurationSelection & sel1).delete(safemode=False)


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

    identity = {
        "sorting_id": populated_sorting_with_curation["sorting_id"],
        "curation_id": populated_sorting_with_curation["curation_id"],
        "metric_params_name": "minimal",
        "auto_curation_rules_name": "none",
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
