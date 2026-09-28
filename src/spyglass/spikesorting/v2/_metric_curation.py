"""DB-free transform helpers for analyzer-driven quality-metric curation.

Pure logic shared by ``CurationEvaluation`` (the ``@schema`` table lives in
``metric_curation.py``): turning quality-metric columns into per-unit labels,
sanitizing non-finite metric values before serialization, reproducing
Spyglass's ``isi_violation`` fraction, and telling a legitimately unassessable
unit's NaN metric apart from a failed metric computation. Keeping these here -- importable with
only NumPy / pandas, no DataJoint connection and no SpikeInterface analyzer --
lets them be unit-tested without a database.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from spyglass.utils import logger

# Threshold-rule comparison operators. Mirrors the ``operator`` enum on
# ``AutoCurationRules.Rule`` and the ``RuleOperator`` Literal in
# ``_params/metric_curation.py``. Non-finite metric values are filtered before
# operator dispatch so low-spike / invalid metrics are NOT auto-flagged,
# including for ``!=`` where NumPy would otherwise treat NaN as unequal.
_COMPARISON_TO_FUNCTION = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _is_finite_metric_value(value) -> bool:
    """Return whether one metric scalar can participate in thresholding.

    A legitimate ``NaN`` (a low-spike unit's metric) is filtered silently. A
    genuinely non-numeric value -- which would mean SpikeInterface's metric
    output drifted to a non-scalar shape/dtype -- is also filtered, but logged
    at WARNING rather than swallowed. The NWB write path is stricter and rejects
    that drift before a persisted evaluation can reach this read-side filter.
    """
    if pd.isna(value):
        return False
    try:
        return bool(np.isfinite(value))
    except TypeError:
        logger.warning(
            "auto-curation: metric value %r (type %s) is not a finite number; "
            "the unit is skipped for this threshold rule. This usually means a "
            "quality-metric column drifted to a non-scalar dtype -- check the "
            "QualityMetricParameters output.",
            value,
            type(value).__name__,
        )
        return False


def apply_snr_peak_sign(
    metric_names, metric_kwargs: dict, sorter_params
) -> dict:
    """Inject the sorter's resolved ``peak_sign`` into the ``snr`` metric kwargs.

    SI's SNR is measured at each template's extremum for a configured
    ``peak_sign`` (default ``"neg"``). The v2 default metric rows hard-code
    ``peak_sign="neg"``, so a positive/bidirectional sorter (clusterless
    ``peak_sign="pos"/"both"``, MountainSort ``detect_sign=1/0``) would measure
    SNR on the most-negative channel instead of its true peak. Resolve the sign
    from the sorter params (the same ``resolve_peak_sign`` mapping that drives
    per-unit attribution) and override it whenever ``snr`` is requested -- even
    if it carried no kwargs yet, otherwise a ``metric_names=["snr"]`` with no
    ``metric_kwargs["snr"]`` would silently keep the hard-coded default. Any
    other existing ``snr`` kwargs are preserved.

    For negative-default sorts the resolved sign is ``"neg"``, so the kwargs
    are numerically unchanged; only positive/bidirectional sorters move.

    Returns a NEW mapping (the caller's dict is not mutated).

    Parameters
    ----------
    metric_names : iterable of str
        Requested quality-metric names.
    metric_kwargs : dict
        Per-metric kwargs mapping (``{metric_name: {...}}``).
    sorter_params : Mapping or None
        The validated ``SorterParameters.params`` blob for the sort.

    Returns
    -------
    dict
        ``metric_kwargs`` with ``snr.peak_sign`` set to the resolved sign when
        ``snr`` is requested, else the input unchanged.
    """
    if "snr" not in metric_names:
        return metric_kwargs
    # Lazy import keeps this module import DB-free (utils pulls in heavier deps).
    from spyglass.spikesorting.v2.utils import resolve_peak_sign

    resolved = resolve_peak_sign(sorter_params)
    return {
        **metric_kwargs,
        "snr": {**(metric_kwargs.get("snr") or {}), "peak_sign": resolved},
    }


def apply_label_rules(
    metrics_df: pd.DataFrame, rule_rows: list[dict]
) -> dict[int, list[str]]:
    """Apply ordered threshold rules to a metrics table, producing labels.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        One row per unit (indexed by ``unit_id``), one column per quality
        metric. NaN entries (legitimately produced for low-spike units) never
        satisfy a threshold.
    rule_rows : list of dict
        ``AutoCurationRules.Rule`` rows, each with ``rule_index``,
        ``metric_name``, ``operator``, ``threshold``, and ``label``. Rules are
        applied in ascending ``rule_index``. ``missing_policy`` defaults to
        ``"error"``; ``"fail"`` applies the rule label to a unit whose value
        is non-finite, while ``"pass"`` and ``"ignore"`` leave it unlabelled.

    Returns
    -------
    dict[int, list[str]]
        ``unit_id -> [label, ...]`` containing ONLY units that matched at
        least one rule. Units with no labels are absent (the caller omits the
        ragged label column entirely when this dict is empty, per the #1625
        empty-list-of-lists fix).

    Raises
    ------
    ValueError
        If a rule references a metric column absent from ``metrics_df``, or if
        any unit has a non-finite value and the rule's missing policy is
        ``"error"``.

    Notes
    -----
    Addresses the three #1513 bug classes: every rule is processed (the return
    is at function scope, never inside the rule loop); each unit's label list
    is an independent object (a fresh list per ``unit_id``); and a label is
    appended only when not already present (element-against-list dedupe).
    """
    # A plain dict with explicit fresh-list construction -- each unit_id gets
    # its own list object, so one unit's append cannot leak into another.
    labels: dict[int, list[str]] = {}

    for rule in sorted(rule_rows, key=lambda r: r["rule_index"]):
        metric_name = rule["metric_name"]
        if metric_name not in metrics_df.columns:
            raise ValueError(
                f"Auto-curation rule {rule.get('rule_name', metric_name)!r} "
                f"references metric column {metric_name!r}, which is not in the "
                f"computed quality metrics {sorted(metrics_df.columns)}. Add "
                "it to the QualityMetricParameters row's metric_names (or fix "
                "the rule) before populating."
            )
        compare = _COMPARISON_TO_FUNCTION[rule["operator"]]
        column = metrics_df[metric_name]
        label = rule["label"]
        missing_policy = rule.get("missing_policy", "error")
        if missing_policy not in {"error", "fail", "pass"}:
            raise ValueError(
                f"Auto-curation rule {rule.get('rule_name', metric_name)!r} "
                f"has invalid missing_policy {missing_policy!r}; expected "
                "'error', 'fail', or 'pass'."
            )
        finite_by_unit = {
            unit_id: _is_finite_metric_value(column.loc[unit_id])
            for unit_id in metrics_df.index
        }
        missing_unit_ids = [
            int(unit_id)
            for unit_id, is_finite in finite_by_unit.items()
            if not is_finite
        ]
        if missing_policy == "error" and missing_unit_ids:
            raise ValueError(
                f"Auto-curation rule {rule.get('rule_name', metric_name)!r} "
                f"references metric {metric_name!r}, which has non-finite "
                f"values for unit_id(s) {missing_unit_ids}. Fix the metric "
                "computation or choose an explicit missing_policy ('fail' "
                "or 'pass') for this rule."
            )
        if (
            missing_policy == "pass"
            and missing_unit_ids
            and len(missing_unit_ids) == len(metrics_df.index)
        ):
            # A per-unit NaN is an expected low-spike skip, but a metric that
            # is missing for EVERY unit means the rule labelled nothing at all
            # -- the silently-inert-rule regression. Say so without raising.
            logger.warning(
                f"Auto-curation rule {rule.get('rule_name', metric_name)!r} "
                f"was inert: metric {metric_name!r} is non-finite for all "
                f"{len(missing_unit_ids)} unit(s), so the rule applied no "
                f"{label!r} labels. Check the metric computation."
            )
        for unit_id in metrics_df.index:
            value = column.loc[unit_id]
            if not finite_by_unit[unit_id]:
                if missing_policy == "fail":
                    unit_labels = labels.setdefault(int(unit_id), [])
                    if label not in unit_labels:
                        unit_labels.append(label)
                continue
            if not compare(value, rule["threshold"]):
                continue
            unit_labels = labels.setdefault(int(unit_id), [])
            if label not in unit_labels:
                unit_labels.append(label)

    return labels


def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every non-finite value replaced by None.

    ``compute_quality_metrics`` legitimately returns NaN for low-spike units
    (and rarely +-inf). Those values break NWB / JSON serialization, so every
    serialized target coerces them to ``None`` first (#1556). The input
    DataFrame is not modified -- in-memory consumers that filter on NaN keep
    their semantics; only the serialized copy is sanitized.
    """
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    non_finite = sanitized.isna()
    # Cast to object first: ``pandas.where(..., other=None)`` treats ``None``
    # as "use the default NaN" and leaves NaN in place, so use an explicit
    # boolean-mask assignment, which writes real ``None`` into object columns.
    sanitized = sanitized.astype(object)
    sanitized[non_finite] = None
    return sanitized


def isi_violation_fraction(isi_violations_count, num_spikes) -> np.ndarray:
    """Reproduce Spyglass's bounded ISI-violation fraction.

    Spyglass defines ``isi_violation`` as ``count / (n_spikes - 1)`` -- the
    observed fraction of too-short inter-spike intervals (``v1/metric_utils``).
    This is NOT SpikeInterface's ``isi_violations_ratio`` (the unbounded
    Hill/UMS2000 contamination estimate), so v2 derives the fraction from
    SI's raw ``isi_violations_count`` column rather than reading SI's ratio.

    Parameters
    ----------
    isi_violations_count : array-like
        SpikeInterface's raw per-unit violation counts.
    num_spikes : array-like
        Per-unit spike counts.

    Returns
    -------
    numpy.ndarray
        The per-unit fraction. Units with <=1 spike yield NaN (caught by NaN
        sanitization), which avoids both the ``0/0`` 1-spike case and the
        spurious finite ``(-1)/(0-1)=1.0`` 0-spike artifact.
    """
    counts = np.asarray(isi_violations_count, dtype=float)
    n = np.asarray(num_spikes, dtype=float)
    denom = n - 1.0
    fraction = np.full(counts.shape, np.nan, dtype=float)
    valid = n > 1.0
    fraction[valid] = counts[valid] / denom[valid]
    return fraction


# ---------- metric eligibility: which NaNs SpikeInterface leaves on purpose --
#
# SpikeInterface swallows a failing metric to NaN (per metric in
# ``core/analyzer_extension_core.py:1281-1286``; per unit for ``nn_advanced``
# in ``metrics/quality/pca_metrics.py:180-181,193-194``), so a NaN alone does
# not say whether the unit was simply not assessable or the computation broke.
# Each predicate below reproduces the exact condition under which SI 0.104.3
# (and Spyglass's ``isi_violation`` fraction) returns NaN without any error.
# Paths are relative to the installed ``spikeinterface`` package.


def _nn_advanced_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``nn_isolation`` / ``nn_noise_overlap``: below the spike or rate floor.

    SI returns NaN when ``n_spikes < min_spikes`` or ``firing_rate < min_fr``
    (``metrics/quality/pca_metrics.py:619-630`` isolation, ``825-836`` noise
    overlap; Spyglass's patched noise overlap keeps both checks,
    ``spyglass/spikesorting/v2/_si_metric_patches.py:224-227``). The rate is
    ``n_spikes / (total_samples / fs)`` (``metrics/spiketrain/metrics.py:69-77``
    with ``metrics/utils.py:100-125``). A zero-spike unit fails every nn
    computation, so it is expected-missing even with ``min_spikes=0``.
    """
    duration_s = total_samples / fs
    below_floor = n_spikes < max(params["min_spikes"], 1)
    below_rate = n_spikes / duration_s < params["min_fr"]
    return below_floor | below_rate


def _presence_ratio_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``presence_ratio``: shorter than one bin (all units), else silent units.

    ``metrics/quality/misc_metrics.py:79-80,97-101`` (sample-based bin floor)
    and ``105-106`` (zero-spike unit).
    """
    if total_samples < int(params["bin_duration_s"] * fs):
        return np.ones(n_spikes.shape, dtype=bool)
    return n_spikes == 0


def _amplitude_cutoff_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``amplitude_cutoff``: too few amplitudes for the histogram.

    ``metrics/quality/misc_metrics.py:1670-1671`` is the only NaN path (the
    result below it is always finite, 1673-1686); the amplitudes are every
    spike of the unit (1005-1011).
    """
    return (
        n_spikes / params["num_histogram_bins"]
        < params["amplitudes_bins_min_ratio"]
    )


def _isi_violation_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """Spyglass ``isi_violation`` fraction: undefined for <= 1 spike.

    See ``isi_violation_fraction`` above.
    """
    return n_spikes <= 1


def _firing_rate_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``firing_rate``: NaN for a zero-spike unit.

    ``metrics/spiketrain/metrics.py:74-75``.
    """
    return n_spikes == 0


def _never_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``snr`` / ``num_spikes``: never legitimately missing.

    ``num_spikes`` is an integer count (``metrics/spiketrain/metrics.py:7-34``).
    ``snr`` is ``abs(amplitude) / noise`` (``metrics/quality/misc_metrics.py:
    205-209``); it is non-finite only when the extremum channel's noise level is
    zero, which is a degenerate recording, not an unassessable unit.
    """
    return np.zeros(n_spikes.shape, dtype=bool)


class _EligibilityRule(NamedTuple):
    """One registered column's NaN predicate.

    Attributes
    ----------
    si_metric : str
        SI metric name whose ``metric_kwargs`` entry (merged over SI's
        defaults) parameterises the predicate.
    is_missing : Callable
        ``(n_spikes, *, total_samples, fs, params) -> bool mask``;
        ``n_spikes`` has shape ``(n_units,)``.
    """

    si_metric: str
    is_missing: Callable[..., np.ndarray]


# Keyed by OUTPUT column name. Deliberately absent: SI's
# ``isi_violations_ratio`` / ``isi_violations_count``, template metrics (e.g.
# ``trough_half_width``), Spyglass's ``observed_*`` columns and custom metrics;
# a rule on any of them fails closed on NaN.
_METRIC_ELIGIBILITY: dict[str, _EligibilityRule] = {
    "nn_isolation": _EligibilityRule("nn_advanced", _nn_advanced_missing),
    "nn_noise_overlap": _EligibilityRule("nn_advanced", _nn_advanced_missing),
    "presence_ratio": _EligibilityRule(
        "presence_ratio", _presence_ratio_missing
    ),
    "amplitude_cutoff": _EligibilityRule(
        "amplitude_cutoff", _amplitude_cutoff_missing
    ),
    "isi_violation": _EligibilityRule("isi_violation", _isi_violation_missing),
    "firing_rate": _EligibilityRule("firing_rate", _firing_rate_missing),
    "snr": _EligibilityRule("snr", _never_missing),
    "num_spikes": _EligibilityRule("num_spikes", _never_missing),
}


def _si_metric_params(si_metric: str, metric_kwargs: Mapping) -> dict:
    """SI's defaults for one metric merged with the caller's kwargs.

    Mirrors the merge in ``core/analyzer_extension_core.py:1172-1178``. SI
    reads its defaults from the metric classes' ``metric_params`` (943) and
    that merge updates those class dicts in place, so a compute with custom
    kwargs changes the defaults for later computes in the same process.
    Reading them here at call time, rather than a fixed copy, keeps the
    classifier on the params SI actually applies. The defaults are copied,
    never mutated.
    """
    from spikeinterface.metrics.quality import (
        get_default_quality_metrics_params,
    )

    defaults = get_default_quality_metrics_params([si_metric])[si_metric]
    return {**defaults, **(metric_kwargs.get(si_metric) or {})}


def expected_missing_units(
    rule_columns: Iterable[str],
    *,
    n_spikes_by_unit: Mapping[Any, int],
    total_samples: int,
    sampling_frequency: float,
    metric_kwargs: Mapping[str, Mapping[str, Any]],
) -> dict[str, set | None]:
    """Units for which SpikeInterface legitimately leaves each column NaN.

    Parameters
    ----------
    rule_columns : iterable of str
        Metric output columns referenced by auto-curation rules.
    n_spikes_by_unit : Mapping
        ``unit_id -> spike count``, as returned by
        ``sorting.count_num_spikes_per_unit()``.
    total_samples : int
        Total samples across segments (``analyzer.get_total_samples()``). SI
        derives firing rates from ``total_samples / sampling_frequency``, not
        from the recording's time vector, so a recording whose timestamps
        have gaps must still pass its sample count here.
    sampling_frequency : float
        Sampling frequency in Hz.
    metric_kwargs : Mapping
        ``QualityMetricParameters.metric_kwargs`` (``{si_metric: {...}}``);
        each predicate merges its SI metric's entry over SI's defaults.

    Returns
    -------
    dict[str, set or None]
        ``column -> set of unit ids`` whose NaN is expected, for every
        column in ``rule_columns``. ``None`` marks a column with no
        registered eligibility rule, for which no NaN can be classified as
        expected.
    """
    unit_ids = list(n_spikes_by_unit)
    n_spikes = np.array([n_spikes_by_unit[u] for u in unit_ids], dtype=float)
    expected: dict[str, set | None] = {}
    for column in rule_columns:
        rule = _METRIC_ELIGIBILITY.get(column)
        if rule is None:
            expected[column] = None
            continue
        missing = rule.is_missing(
            n_spikes,
            total_samples=total_samples,
            fs=sampling_frequency,
            params=_si_metric_params(rule.si_metric, metric_kwargs),
        )
        expected[column] = {u for u, m in zip(unit_ids, missing) if m}
    return expected


def assert_rule_metrics_computed(
    metrics_df: pd.DataFrame,
    rule_columns: Iterable[str],
    expected_missing: Mapping[str, set | None],
) -> None:
    """Raise if a rule-referenced metric is non-finite where it should not be.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        One row per unit (indexed by ``unit_id``), one column per metric.
    rule_columns : iterable of str
        Metric columns referenced by auto-curation rules. No other column is
        inspected. A rule column absent from ``metrics_df`` is skipped here
        (``apply_label_rules`` rejects it).
    expected_missing : Mapping
        Output of ``expected_missing_units`` for the same ``rule_columns``.

    Raises
    ------
    ValueError
        If a registered column is non-finite (NaN or +/-inf) for a unit
        outside its expected-missing set, or an unregistered column is
        non-finite for any unit.
    """
    for column in sorted(rule_columns):
        if column not in metrics_df.columns:
            continue
        values = metrics_df[column]
        non_finite = [
            unit_id
            for unit_id in metrics_df.index
            if not _is_finite_metric_value(values.loc[unit_id])
        ]
        expected = expected_missing[column]
        if expected is None:
            if non_finite:
                raise ValueError(
                    f"Metric column {column!r} is non-finite for unit_id(s) "
                    f"{non_finite}, and it has no registered eligibility "
                    "rule, so a NaN cannot be classified as an expected "
                    "unassessable unit rather than a computation failure. "
                    "Register an eligibility rule for this metric, or "
                    "threshold a metric that has one."
                )
            continue
        failed = [unit_id for unit_id in non_finite if unit_id not in expected]
        if failed:
            raise ValueError(
                f"Metric column {column!r} is non-finite for unit_id(s) "
                f"{failed}, which meet the metric's preconditions: this is a "
                "metric computation failure, not an unassessable unit. Check "
                "the quality-metric computation (SpikeInterface replaces a "
                "failing metric with NaN and only warns)."
            )


def rules_payloads_match(
    expected: dict,
    stored: dict,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-12,
) -> bool:
    """Return whether two normalized ``AutoCurationRules`` payloads are equal.

    Compares the ``{master, rules}`` payloads built by
    ``AutoCurationRules._payload_for_compare`` for the idempotency guard.
    Numbers are compared with ``math.isclose`` rather than ``==`` because the
    ``Rule.threshold`` column is single precision: a threshold such as ``0.1``
    round-trips from the database as ``0.10000000149...``, which is not
    bit-equal to the freshly validated Python float. An exact comparison would
    raise a spurious "different payload" error when re-inserting identical
    rules (e.g. a second ``insert_default()`` run), breaking idempotency. The
    default ``rel_tol`` comfortably exceeds float32 round-off (~6e-8) while
    still distinguishing thresholds that differ by more than one part per
    million.
    """
    return _values_match(expected, stored, rel_tol=rel_tol, abs_tol=abs_tol)


def _values_match(a, b, *, rel_tol: float, abs_tol: float) -> bool:
    """Recursively compare JSON-native values; floats within tolerance match."""
    # bool is an int subclass; compare by exact type+value so True and 1 do
    # not conflate through the numeric branch below.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _values_match(a[key], b[key], rel_tol=rel_tol, abs_tol=abs_tol)
            for key in a
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _values_match(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
            for x, y in zip(a, b)
        )
    return a == b
