"""DB-free unit tests for analyzer-curation transform helpers.

These exercise the pure logic in
``spyglass.spikesorting.v2._metric_curation`` -- label-rule application
(the three #1513 bug-class invariants), NaN sanitization for serialization
(#1556), the Spyglass ``isi_violation`` fraction, and the classification of
expected-missing versus failed metric values -- with no DataJoint
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
    assert_rule_metrics_computed,
    expected_missing_units,
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
        "missing_policy": "error",
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
    rules[0]["missing_policy"] = "pass"
    labels = apply_label_rules(metrics, rules)
    assert labels == {}


def test_apply_label_rules_all_nan_defaults_to_actionable_error():
    """An all-missing rule input cannot silently disable a rule."""
    metrics = pd.DataFrame({"nn_noise_overlap": [np.nan, np.nan]}, index=[4, 5])
    rules = [_rule(0, "nn_noise_overlap", ">", 0.1, "noise")]
    with pytest.raises(
        ValueError, match=r"non-finite values for unit_id\(s\) \[4, 5\]"
    ):
        apply_label_rules(metrics, rules)


def test_apply_label_rules_partial_missing_error_fails_fast():
    """The default error policy rejects even one missing unit value."""
    metrics = pd.DataFrame({"snr": [np.nan, 0.5, 5.0]}, index=[1, 2, 3])
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    with pytest.raises(
        ValueError, match=r"non-finite values for unit_id\(s\) \[1\]"
    ):
        apply_label_rules(metrics, rules)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("fail", {4: ["noise"], 5: ["noise"]}),
        ("pass", {}),
    ],
)
def test_apply_label_rules_all_nan_explicit_policies(policy, expected):
    """Explicit missing-value policies make all-NaN behavior deliberate."""
    metrics = pd.DataFrame({"nn_noise_overlap": [np.nan, np.nan]}, index=[4, 5])
    rules = [_rule(0, "nn_noise_overlap", ">", 0.1, "noise")]
    rules[0]["missing_policy"] = policy
    assert apply_label_rules(metrics, rules) == expected


def test_apply_label_rules_partial_missing_fail_labels_only_missing_unit():
    """The fail policy labels missing units as failures alongside matches."""
    metrics = pd.DataFrame({"snr": [np.nan, 0.5, 5.0]}, index=[1, 2, 3])
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    rules[0]["missing_policy"] = "fail"
    assert apply_label_rules(metrics, rules) == {
        1: ["noise"],
        2: ["noise"],
    }


def test_is_finite_metric_value_warns_on_non_numeric(caplog):
    """A genuinely non-numeric metric cell (an SI shape/dtype drift) is filtered
    AND logged -- the swallow must be visible, unlike a legitimate NaN which is
    a silent low-spike skip."""
    from spyglass.spikesorting.v2._metric_curation import (
        _is_finite_metric_value,
    )

    with caplog.at_level("WARNING"):
        result = _is_finite_metric_value("not-a-number")
    assert result is False
    assert any(
        "metric" in record.getMessage().lower() for record in caplog.records
    ), "non-numeric metric value was filtered silently"


def test_is_finite_metric_value_nan_filtered_silently(caplog):
    """A legitimate NaN (low-spike unit) is filtered without any warning."""
    from spyglass.spikesorting.v2._metric_curation import (
        _is_finite_metric_value,
    )

    with caplog.at_level("WARNING"):
        result = _is_finite_metric_value(np.nan)
    assert result is False
    assert not caplog.records


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


def _payload(
    threshold,
    *,
    operator=">",
    label="noise",
    preset="none",
    missing_policy="error",
):
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
                "missing_policy": missing_policy,
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


def test_rules_payloads_match_includes_missing_policy():
    """Rule sets differing only in missing-value semantics are distinct."""
    assert not rules_payloads_match(
        _payload(0.1, missing_policy="error"),
        _payload(0.1, missing_policy="pass"),
    )


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


def test_apply_label_rules_pass_is_silent_for_partial_missing(caplog):
    """A legitimate low-spike NaN is skipped without log noise.

    ``min_spikes`` makes NaN an expected, per-unit outcome, so the shipped
    ``pass`` policy must not warn once per affected unit on every sort.
    """
    metrics = pd.DataFrame({"snr": [np.nan, 0.5, 5.0]}, index=[1, 2, 3])
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    rules[0]["missing_policy"] = "pass"
    with caplog.at_level("WARNING"):
        labels = apply_label_rules(metrics, rules)
    assert labels == {2: ["noise"]}
    assert not caplog.records, "partial low-spike NaN skip must stay silent"


def test_apply_label_rules_pass_warns_when_rule_is_wholly_inert(caplog):
    """An all-missing metric under ``pass`` disables the rule -- say so.

    This is the regression class where ``nn_noise_overlap`` was NaN for every
    unit and default auto-curation became silently inert. ``pass`` still must
    not raise, but a rule that labelled nothing because its metric was missing
    everywhere is a computation failure, not a low-spike skip.
    """
    metrics = pd.DataFrame({"nn_noise_overlap": [np.nan, np.nan]}, index=[4, 5])
    rules = [_rule(0, "nn_noise_overlap", ">", 0.1, "noise")]
    rules[0]["missing_policy"] = "pass"
    with caplog.at_level("WARNING"):
        assert apply_label_rules(metrics, rules) == {}
    assert any(
        "nn_noise_overlap" in record.getMessage() for record in caplog.records
    ), "a wholly inert rule was disabled silently"


def test_apply_label_rules_rejects_retired_ignore_policy():
    """``ignore`` was a byte-identical duplicate of ``pass`` and is retired."""
    metrics = pd.DataFrame({"snr": [np.nan]}, index=[1])
    rules = [_rule(0, "snr", "<", 1.0, "noise")]
    rules[0]["missing_policy"] = "ignore"
    with pytest.raises(ValueError, match="invalid missing_policy"):
        apply_label_rules(metrics, rules)


# ---------- metric eligibility: expected-missing vs failed metrics ----------

_FS = 30_000.0
# The shipped ``nn_advanced`` kwargs of the default QualityMetricParameters.
_SHIPPED_NN_KWARGS = {
    "n_components": 7,
    "n_neighbors": 5,
    "max_spikes": 20000,
    "min_spikes": 10,
    "seed": 0,
}


def _expected(rule_columns, n_spikes_by_unit, duration_s, metric_kwargs=None):
    return expected_missing_units(
        rule_columns,
        n_spikes_by_unit=n_spikes_by_unit,
        total_samples=int(duration_s * _FS),
        sampling_frequency=_FS,
        metric_kwargs=metric_kwargs or {},
    )


def test_expected_missing_nn_below_floor():
    """nn_* are expected-missing below ``min_spikes`` or below ``min_fr``."""
    nn_columns = {"nn_isolation", "nn_noise_overlap"}
    n_spikes = {3: 5, 17: 50, 58: 9, 61: 10, 80: 0}
    expected = _expected(
        nn_columns,
        n_spikes,
        duration_s=10.0,
        metric_kwargs={"nn_advanced": _SHIPPED_NN_KWARGS},
    )
    assert expected == {
        "nn_isolation": {3, 58, 80},
        "nn_noise_overlap": {3, 58, 80},
    }

    # A silent unit cannot be assessed even when min_spikes allows it.
    no_floor = _expected(
        nn_columns,
        n_spikes,
        duration_s=10.0,
        metric_kwargs={"nn_advanced": {**_SHIPPED_NN_KWARGS, "min_spikes": 0}},
    )
    assert no_floor == {"nn_isolation": {80}, "nn_noise_overlap": {80}}

    min_fr_kwargs = {"nn_advanced": {**_SHIPPED_NN_KWARGS, "min_fr": 1.0}}
    # 20 spikes over 100 s is 0.2 Hz (below min_fr); over 10 s it is 2 Hz.
    slow = _expected(nn_columns, {42: 20}, 100.0, min_fr_kwargs)
    fast = _expected(nn_columns, {42: 20}, 10.0, min_fr_kwargs)
    assert slow == {"nn_isolation": {42}, "nn_noise_overlap": {42}}
    assert fast == {"nn_isolation": set(), "nn_noise_overlap": set()}


def test_expected_missing_presence_ratio_short_recording():
    """A recording shorter than one presence bin leaves every unit NaN."""
    n_spikes = {3: 0, 17: 1, 42: 50, 58: 2}
    rule_columns = {"presence_ratio", "isi_violation"}
    short = expected_missing_units(
        rule_columns,
        n_spikes_by_unit=n_spikes,
        total_samples=int(60 * _FS) - 1,
        sampling_frequency=_FS,
        metric_kwargs={},
    )
    assert short == {
        "presence_ratio": {3, 17, 42, 58},
        # Only the <=1-spike units; the short recording does not leak here.
        "isi_violation": {3, 17},
    }

    # Exactly one bin of samples is long enough: only the silent unit is NaN.
    long = expected_missing_units(
        {"presence_ratio"},
        n_spikes_by_unit=n_spikes,
        total_samples=int(60 * _FS),
        sampling_frequency=_FS,
        metric_kwargs={},
    )
    assert long == {"presence_ratio": {3}}


def test_expected_missing_amplitude_cutoff_spike_floor():
    """amplitude_cutoff is NaN below ``num_histogram_bins * ratio`` spikes."""
    default = _expected({"amplitude_cutoff"}, {3: 499, 17: 500}, 60.0)
    assert default == {"amplitude_cutoff": {3}}

    custom = _expected(
        {"amplitude_cutoff"},
        {3: 19, 17: 20},
        60.0,
        {
            "amplitude_cutoff": {
                "num_histogram_bins": 10,
                "amplitudes_bins_min_ratio": 2,
            }
        },
    )
    assert custom == {"amplitude_cutoff": {3}}

    metrics = pd.DataFrame({"amplitude_cutoff": [np.nan, 0.01]}, index=[42, 3])
    expected = _expected({"amplitude_cutoff"}, {42: 600, 3: 700}, 60.0)
    with pytest.raises(ValueError, match=r"amplitude_cutoff.*\[42\]"):
        assert_rule_metrics_computed(metrics, {"amplitude_cutoff"}, expected)


def test_unreferenced_columns_never_inspected():
    """All-NaN columns outside the rule columns are not checked."""
    metrics = pd.DataFrame(
        {
            "snr": [4.0, 6.0],
            "my_custom_metric": [np.nan, np.nan],
            "trough_half_width": [np.nan, np.nan],
        },
        index=[3, 17],
    )
    expected = _expected({"snr"}, {3: 50, 17: 60}, 60.0)
    assert expected == {"snr": set()}
    assert assert_rule_metrics_computed(metrics, {"snr"}, expected) is None

    # The same frame fails once a rule does reference an all-NaN column.
    expected = _expected({"snr", "my_custom_metric"}, {3: 50, 17: 60}, 60.0)
    with pytest.raises(ValueError, match="my_custom_metric"):
        assert_rule_metrics_computed(
            metrics, {"snr", "my_custom_metric"}, expected
        )


def test_unregistered_rule_column_with_nan_fails_closed():
    """A NaN in a column with no eligibility rule cannot be classified."""
    expected = _expected({"trough_half_width"}, {3: 50, 17: 60}, 60.0)
    assert expected == {"trough_half_width": None}

    with_nan = pd.DataFrame({"trough_half_width": [0.2, np.nan]}, index=[3, 17])
    with pytest.raises(ValueError, match="no registered eligibility rule"):
        assert_rule_metrics_computed(with_nan, {"trough_half_width"}, expected)

    finite = pd.DataFrame({"trough_half_width": [0.2, 0.3]}, index=[3, 17])
    assert (
        assert_rule_metrics_computed(finite, {"trough_half_width"}, expected)
        is None
    )


def test_assert_metrics_computed_raises_on_unexpected_nan():
    """A NaN for a unit that meets the metric's preconditions is a failure."""
    n_spikes = {3: 1, 17: 50, 42: 80}
    expected = _expected({"isi_violation"}, n_spikes, 60.0)
    assert expected == {"isi_violation": {3}}

    legit = pd.DataFrame(
        {"isi_violation": [np.nan, 0.0, 0.01]}, index=[3, 17, 42]
    )
    assert (
        assert_rule_metrics_computed(legit, {"isi_violation"}, expected) is None
    )

    failed = legit.copy()
    failed.loc[17, "isi_violation"] = np.nan
    with pytest.raises(ValueError, match=r"'isi_violation'.*\[17\]"):
        assert_rule_metrics_computed(failed, {"isi_violation"}, expected)


def test_snr_inf_is_a_failure():
    """snr is never legitimately missing, and +/-inf counts as missing."""
    expected = _expected({"snr"}, {3: 50, 17: 60}, 60.0)
    assert expected == {"snr": set()}
    metrics = pd.DataFrame({"snr": [5.0, np.inf]}, index=[3, 17])
    with pytest.raises(ValueError, match=r"'snr'.*\[17\]"):
        assert_rule_metrics_computed(metrics, {"snr"}, expected)


# ---------- apply_label_rules consuming expected_missing --------------------


@pytest.mark.parametrize(
    ("policy", "labels_when_only_expected"),
    [("pass", {}), ("fail", {42: ["noise"]})],
)
def test_apply_label_rules_unexpected_nan_raises_regardless_of_policy(
    policy, labels_when_only_expected
):
    """A non-finite value outside the eligibility set is a computation failure.

    Unit 42's NaN is in the column's expected-missing set (e.g. below
    min_spikes); unit 99's is not, so it must raise under BOTH ``"pass"`` and
    ``"fail"``. With only the expected unit missing, the same NaN instead
    follows missing_policy as usual.
    """
    expected_missing = {"nn_noise_overlap": {42}}
    rules = [_rule(0, "nn_noise_overlap", ">", 0.1, "noise")]
    rules[0]["missing_policy"] = policy

    unexpected = pd.DataFrame(
        {"nn_noise_overlap": [np.nan, np.nan]}, index=[42, 99]
    )
    with pytest.raises(ValueError, match=r"unit_id\(s\) \[99\]"):
        apply_label_rules(unexpected, rules, expected_missing=expected_missing)

    only_expected = pd.DataFrame({"nn_noise_overlap": [np.nan]}, index=[42])
    assert (
        apply_label_rules(
            only_expected, rules, expected_missing=expected_missing
        )
        == labels_when_only_expected
    )


def test_apply_label_rules_unregistered_column_nan_fails_closed_with_expected_missing():
    """``expected_missing={col: None}`` fails closed on any non-finite value.

    No eligibility rule is registered for ``trough_half_width``, so a NaN
    cannot be classified -- even under the lenient ``"pass"`` policy.
    """
    expected_missing = {"trough_half_width": None}
    rules = [_rule(0, "trough_half_width", ">", 0.5, "wide")]
    rules[0]["missing_policy"] = "pass"

    with_nan = pd.DataFrame({"trough_half_width": [0.6, np.nan]}, index=[3, 17])
    with pytest.raises(ValueError, match="no registered eligibility rule"):
        apply_label_rules(with_nan, rules, expected_missing=expected_missing)

    finite = pd.DataFrame({"trough_half_width": [0.6, 0.4]}, index=[3, 17])
    assert apply_label_rules(
        finite, rules, expected_missing=expected_missing
    ) == {3: ["wide"]}


def test_apply_label_rules_fail_policy_all_missing_warns(caplog):
    """``"fail"`` is inert (every unit labelled for the same reason) too.

    The existing warning only covered ``"pass"``; ``"fail"`` silently applied
    its label to every unit with no indication the metric itself was broken.
    """
    metrics = pd.DataFrame({"nn_noise_overlap": [np.nan, np.nan]}, index=[4, 5])
    rules = [_rule(0, "nn_noise_overlap", ">", 0.1, "noise")]
    rules[0]["missing_policy"] = "fail"
    with caplog.at_level("WARNING"):
        labels = apply_label_rules(metrics, rules)
    assert labels == {4: ["noise"], 5: ["noise"]}
    assert any(
        "nn_noise_overlap" in record.getMessage() for record in caplog.records
    ), "an all-missing rule under 'fail' was inert without saying so"


def test_isi_violation_one_spike_unit_follows_missing_policy():
    """The real ``isi_violation_fraction`` NaN is expected-missing, not failed.

    A 1-spike unit's fraction is undefined (NaN); 2- and 50-spike units get a
    real (zero) value. The shipped isi rule (``isi_violation > 0.02 ->
    "reject"``) must treat the 1-spike unit's NaN per missing_policy, and must
    never raise or label the two ordinary units.
    """
    n_spikes_by_unit = {3: 1, 17: 2, 42: 50}
    unit_ids = list(n_spikes_by_unit)
    isi_violation = isi_violation_fraction(
        [0, 0, 0], [n_spikes_by_unit[u] for u in unit_ids]
    )
    assert np.isnan(isi_violation[0]) and list(isi_violation[1:]) == [0.0, 0.0]
    metrics = pd.DataFrame({"isi_violation": isi_violation}, index=unit_ids)
    expected_missing = _expected(
        {"isi_violation"}, n_spikes_by_unit, duration_s=60.0
    )
    assert expected_missing == {"isi_violation": {3}}

    rule = _rule(0, "isi_violation", ">", 0.02, "reject")

    rule["missing_policy"] = "pass"
    assert (
        apply_label_rules(metrics, [rule], expected_missing=expected_missing)
        == {}
    )

    rule["missing_policy"] = "fail"
    assert apply_label_rules(
        metrics, [rule], expected_missing=expected_missing
    ) == {3: ["reject"]}

    rule["missing_policy"] = "error"
    with pytest.raises(ValueError, match=r"unit_id\(s\) \[3\]"):
        apply_label_rules(metrics, [rule], expected_missing=expected_missing)
