"""DB-free unit tests for analyzer-curation transform helpers.

These exercise the pure logic in
``spyglass.spikesorting.v2._metric_curation`` -- label-rule application
(the three #1513 bug-class invariants), NaN sanitization for serialization
(#1556), and the Spyglass ``isi_violation`` fraction -- with no DataJoint
server and no SpikeInterface analyzer. Importing the service module never
opens a database connection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spyglass.spikesorting.v2._metric_curation import (
    apply_label_rules,
    apply_snr_peak_sign,
    isi_violation_fraction,
    rules_payloads_match,
    sanitize_for_json,
)


def _rule(rule_index, metric_name, operator, threshold, label, name=None):
    return {
        "rule_index": rule_index,
        "rule_name": name or f"{metric_name}_{label}",
        "metric_name": metric_name,
        "operator": operator,
        "threshold": threshold,
        "label": label,
    }


# ---------- #1513 invariant 1: loop completion (Bug A) ----------------------


def test_apply_label_rules_processes_every_rule():
    """Every rule runs; later rules are not dropped by an early return.

    Three rules each add a DISTINCT label to the same single unit; the unit
    must end with all three. A ``return`` inside the rule loop would silently
    drop rules 2 and 3.
    """
    metrics = pd.DataFrame({"snr": [0.5]}, index=[7])
    rules = [
        _rule(0, "snr", "<", 1.0, "noise"),
        _rule(1, "snr", "<", 1.0, "mua"),
        _rule(2, "snr", "<", 1.0, "reject"),
    ]
    labels = apply_label_rules(metrics, rules)
    assert labels[7] == ["noise", "mua", "reject"]


# ---------- #1513 invariant 2: per-unit list isolation (Bug B) --------------


def test_apply_label_rules_per_unit_lists_are_independent():
    """One unit's later label must not contaminate another unit's list.

    Two units are both flagged ``noise`` by rule 1; rule 2 flags only unit A
    with ``mua``. B must keep just ``["noise"]`` -- a shared list object would
    leak A's ``mua`` into B.
    """
    metrics = pd.DataFrame(
        {"snr": [0.5, 0.5], "isi_violation": [0.9, 0.0]},
        index=[1, 2],  # unit 1 = A, unit 2 = B
    )
    rules = [
        _rule(0, "snr", "<", 1.0, "noise"),
        _rule(1, "isi_violation", ">", 0.5, "mua"),
    ]
    labels = apply_label_rules(metrics, rules)
    assert labels[1] == ["noise", "mua"]
    assert labels[2] == ["noise"]


# ---------- #1513 invariant 3: per-rule membership dedupe (Bug C) -----------


def test_apply_label_rules_dedupes_repeated_label():
    """Two rules that both emit ``noise`` for one unit yield a single label."""
    metrics = pd.DataFrame({"snr": [0.2], "nn_noise_overlap": [0.9]}, index=[3])
    rules = [
        _rule(0, "snr", "<", 1.0, "noise"),
        _rule(1, "nn_noise_overlap", ">", 0.1, "noise"),
    ]
    labels = apply_label_rules(metrics, rules)
    assert labels[3] == ["noise"]


# ---------- unlabeled units / NaN comparison / errors -----------------------


def test_apply_label_rules_omits_unlabeled_units():
    """Units matching no rule get no key (drives the omit-empty-column path)."""
    metrics = pd.DataFrame({"snr": [5.0, 0.2]}, index=[10, 11])
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    labels = apply_label_rules(metrics, rules)
    assert 11 in labels and labels[11] == ["noise"]
    assert 10 not in labels  # high-SNR unit not flagged


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("operator", ["<", "<=", ">", ">=", "==", "!="])
def test_apply_label_rules_non_finite_compares_false(value, operator):
    """Non-finite metrics never satisfy thresholds (even ``NaN != x``)."""
    metrics = pd.DataFrame({"nn_noise_overlap": [value]}, index=[4])
    rules = [_rule(0, "nn_noise_overlap", operator, 0.1, "noise")]
    labels = apply_label_rules(metrics, rules)
    assert labels == {}


def test_apply_label_rules_missing_metric_raises():
    """A rule referencing an absent metric column fails loudly before write."""
    metrics = pd.DataFrame({"snr": [1.0]}, index=[0])
    rules = [_rule(0, "not_computed", ">", 0.1, "noise")]
    with pytest.raises(ValueError, match="not_computed"):
        apply_label_rules(metrics, rules)


def test_apply_label_rules_empty_rules_returns_empty():
    """No rules => no labels, regardless of metrics."""
    metrics = pd.DataFrame({"snr": [0.1, 0.2]}, index=[0, 1])
    assert apply_label_rules(metrics, []) == {}


def test_apply_label_rules_coerces_numpy_int_unit_ids():
    """Label keys are plain python int even from a numpy-int64 index.

    SI unit ids are often ``np.int64``; downstream NWB writers / ``get_labels``
    consumers expect python ints, so the ``int(unit_id)`` cast must hold.
    """
    metrics = pd.DataFrame(
        {"snr": [0.2]}, index=pd.Index(np.array([7], dtype=np.int64))
    )
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    labels = apply_label_rules(metrics, rules)
    assert labels == {7: ["noise"]}
    assert all(type(key) is int for key in labels)  # not np.int64


# ---------- #1556 NaN sanitization ------------------------------------------


def test_sanitize_for_json_replaces_non_finite_with_none():
    """NaN / +-inf become None in the copy; the source DataFrame is untouched."""
    df = pd.DataFrame(
        {"snr": [1.0, np.nan], "amp": [np.inf, -np.inf]}, index=[0, 1]
    )
    out = sanitize_for_json(df)
    assert out.loc[0, "snr"] == 1.0
    assert out.loc[1, "snr"] is None
    assert out.loc[0, "amp"] is None
    assert out.loc[1, "amp"] is None
    # Source keeps NaN/inf semantics for downstream filtering.
    assert np.isnan(df.loc[1, "snr"])
    assert np.isinf(df.loc[0, "amp"])


# ---------- Spyglass isi_violation fraction ---------------------------------


def test_isi_violation_fraction_count_over_n_minus_one():
    """Fraction is count / (n_spikes - 1), elementwise (v1 parity)."""
    counts = np.array([0, 3, 10])
    n_spikes = np.array([101, 301, 11])
    frac = isi_violation_fraction(counts, n_spikes)
    np.testing.assert_allclose(frac, [0.0, 3 / 300, 10 / 10])


@pytest.mark.parametrize("n", [0, 1])
def test_isi_violation_fraction_guards_low_spike(n):
    """0- and 1-spike units yield NaN, not the spurious finite 1.0 artifact."""
    frac = isi_violation_fraction(np.array([0]), np.array([n]))
    assert np.isnan(frac[0])


# ---------- rules-payload idempotency comparison ----------------------------


def _payload(threshold, *, operator=">", label="noise", preset="none"):
    """A minimal normalized AutoCurationRules payload for comparison."""
    return {
        "auto_curation_rules_name": "r",
        "auto_merge_preset": preset,
        "auto_merge_kwargs": {},
        "params_schema_version": 1,
        "job_kwargs": None,
        "rules": [
            {
                "rule_index": 0,
                "rule_name": "nn_noise",
                "metric_name": "nn_noise_overlap",
                "operator": operator,
                "threshold": threshold,
                "label": label,
            }
        ],
    }


def test_rules_payloads_match_tolerates_float32_round_trip():
    """A threshold's single-precision round-off must not break idempotency."""
    # 0.1 is not exactly representable in float32; this is the value the DB
    # column returns on fetch, and what broke the exact-equality comparison.
    stored = _payload(float(np.float32(0.1)))
    expected = _payload(0.1)
    assert stored["rules"][0]["threshold"] != expected["rules"][0]["threshold"]
    assert rules_payloads_match(expected, stored)


def test_rules_payloads_match_rejects_genuinely_different_threshold():
    """Thresholds differing by more than float round-off are NOT equal."""
    assert not rules_payloads_match(_payload(0.1), _payload(0.2))


@pytest.mark.parametrize(
    "other",
    [
        _payload(0.1, operator="<"),  # different operator
        _payload(0.1, label="reject"),  # different label
        _payload(0.1, preset="similarity_correlograms"),  # different preset
    ],
)
def test_rules_payloads_match_rejects_non_numeric_differences(other):
    """Non-numeric field changes are detected by exact comparison."""
    assert not rules_payloads_match(_payload(0.1), other)


def test_rules_payloads_match_rejects_differing_rule_count():
    """A payload with extra/fewer rules does not match."""
    one = _payload(0.1)
    two = _payload(0.1)
    two["rules"].append({**two["rules"][0], "rule_index": 1, "label": "reject"})
    assert not rules_payloads_match(one, two)


def test_rules_payloads_match_does_not_conflate_bool_and_int():
    """bool is an int subclass; True must not match 1 through the numeric path."""
    assert not rules_payloads_match({"k": True}, {"k": 1})
    assert rules_payloads_match({"k": True}, {"k": True})


# ---------- SIG-2: SNR peak_sign follows the sorter's polarity --------------


@pytest.mark.parametrize(
    "sorter_params,expected",
    [
        ({"peak_sign": "pos"}, "pos"),
        ({"peak_sign": "both"}, "both"),
        ({"detect_sign": 1}, "pos"),
        ({"detect_sign": -1}, "neg"),
        ({}, "neg"),  # no sign field -> SI default fallback
        (None, "neg"),
    ],
)
def test_apply_snr_peak_sign_injects_resolved_sign(sorter_params, expected):
    """When ``snr`` is requested, the sorter's resolved peak_sign is injected
    into ``metric_kwargs['snr']`` even if it carried no kwargs yet.

    SNR was hard-coded ``peak_sign='neg'``, so a positive/bidirectional sorter
    measured SNR on the most-negative channel instead of its true peak.
    """
    out = apply_snr_peak_sign(["snr", "isi_violation"], {}, sorter_params)
    assert out["snr"]["peak_sign"] == expected


def test_apply_snr_peak_sign_preserves_existing_snr_kwargs():
    """An existing snr kwarg survives; only peak_sign is overridden."""
    out = apply_snr_peak_sign(
        ["snr"], {"snr": {"peak_mode": "extremum"}}, {"peak_sign": "pos"}
    )
    assert out["snr"] == {"peak_mode": "extremum", "peak_sign": "pos"}


def test_apply_snr_peak_sign_noop_when_snr_not_requested():
    """No snr metric -> kwargs are returned untouched (no snr key added)."""
    kwargs = {"isi_violation": {"isi_threshold_ms": 2.0}}
    out = apply_snr_peak_sign(["isi_violation"], kwargs, {"peak_sign": "pos"})
    assert "snr" not in out
    assert out == kwargs


def test_apply_snr_peak_sign_does_not_mutate_input():
    """The override returns a new mapping; the caller's dict is unchanged."""
    kwargs = {"snr": {"peak_mode": "extremum"}}
    apply_snr_peak_sign(["snr"], kwargs, {"peak_sign": "pos"})
    assert kwargs == {"snr": {"peak_mode": "extremum"}}


def test_snr_peak_sign_follows_sorter_polarity():
    """End-to-end (hermetic): a positive-going planted sort computes SNR on its
    POSITIVE-peak channel, and a negative-default sort's SNR is unchanged.

    The planted unit has a +50 uV peak on channel 0 and a larger -100 uV
    deflection on channel 1, so peak_sign decides which channel SNR is measured
    on. Driving SI's ``compute_quality_metrics`` with the kwargs
    ``apply_snr_peak_sign`` produces for a ``peak_sign='pos'`` sorter must give
    the smaller (positive-peak) SNR and differ from the ``'neg'`` default --
    which stays the regression-pinned value.
    """
    from spikeinterface.metrics.quality import compute_quality_metrics

    from tests.spikesorting.v2.test_peak_sign_resolution import (
        _pos_neg_analyzer,
    )

    analyzer, _ = _pos_neg_analyzer()
    analyzer.compute(["noise_levels", "spike_amplitudes"])

    def _snr(sorter_params):
        kwargs = apply_snr_peak_sign(["snr"], {}, sorter_params)
        df = compute_quality_metrics(
            analyzer,
            metric_names=["snr"],
            metric_params={"snr": kwargs["snr"]},
            skip_pc_metrics=True,
            delete_existing_metrics=True,
        )
        return float(df["snr"].iloc[0])

    snr_pos = _snr({"peak_sign": "pos"})  # follows the +channel
    snr_neg = _snr({"peak_sign": "neg"})  # SI default, unchanged
    snr_default = _snr({})  # no sign -> neg fallback

    # The positive peak (+50) is smaller than the negative deflection (-100),
    # so measuring on the positive channel yields a smaller SNR.
    assert snr_pos < snr_neg
    # The negative-default path is the regression pin: same as explicit 'neg'.
    assert snr_default == pytest.approx(snr_neg)
