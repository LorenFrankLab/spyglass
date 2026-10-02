"""DB-free compute and transform helpers for analyzer-driven metric curation.

Pure logic shared by ``CurationEvaluation`` (the ``@schema`` table lives in
``metric_curation.py``): turning quality-metric columns into per-unit labels,
sanitizing non-finite metric values before serialization, reproducing
Spyglass's ``isi_violation`` fraction, and telling a legitimately unassessable
unit's NaN metric apart from a failed metric computation. It also holds the
evaluation's compute steps over SpikeInterface analyzers: ``compute_metrics``
(voltage metrics on the display analyzer, PC/NN metrics on the whitened metric
analyzer), ``compute_merge_groups`` and ``evaluate_analyzers``, which labels
the units and enforces the curation's unit namespace. SpikeInterface is
imported inside those functions, so the module imports with only NumPy /
pandas and no DataJoint connection, and its helpers can be unit-tested
without a database.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from spyglass.spikesorting.v2._sorting_analyzer import (
    STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
)
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
    metrics_df: pd.DataFrame,
    rule_rows: list[dict],
    *,
    expected_missing: dict[str, set | None] | None = None,
) -> dict[int, list[str]]:
    """Apply ordered threshold rules to a metrics table, producing labels.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        One row per unit (indexed by ``unit_id``), one column per quality
        metric. A non-finite entry never satisfies a threshold; whether it is
        treated as an expected, legitimately-unassessable value or as a
        metric computation failure is decided by ``expected_missing`` (or, if
        that is not given, by the rule's ``missing_policy`` alone, as before).
    rule_rows : list of dict
        ``AutoCurationRules.Rule`` rows, each with ``rule_index``,
        ``metric_name``, ``operator``, ``threshold``, and ``label``. Rules are
        applied in ascending ``rule_index``. ``threshold`` must be finite.
        ``missing_policy`` defaults to ``"error"``; ``"fail"`` applies the
        rule label to a unit whose value is non-finite, while ``"pass"``
        leaves it unlabelled.
    expected_missing : dict[str, set or None] or None, optional
        Output of ``expected_missing_units`` for the columns these rules
        reference. When given, a rule's ``missing_policy`` only governs units
        whose non-finite value is in the column's expected-missing set. A
        non-finite value for a unit NOT in that set, or any non-finite value
        in a column with no registered eligibility rule (the column's entry
        is ``None`` or it is absent from this mapping), is instead treated as
        a metric computation failure and raises regardless of
        ``missing_policy``. When ``expected_missing`` is ``None`` (the
        default), every non-finite value follows ``missing_policy`` as if it
        were expected.

    Returns
    -------
    dict[int, list[str]]
        ``unit_id -> [label, ...]`` containing ONLY units that matched at
        least one rule. Units with no labels are absent (the caller omits the
        ragged label column entirely when this dict is empty, because an
        empty list-of-lists column cannot be written; see #1625).

    Raises
    ------
    ValueError
        If a rule references a metric column absent from ``metrics_df``; if a
        rule's ``threshold`` is non-finite; if any unit has a non-finite
        value and the rule's missing policy is ``"error"``; or if
        ``expected_missing`` classifies a unit's non-finite value as a metric
        computation failure (regardless of ``missing_policy``).

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
        rule_name = rule.get("rule_name", metric_name)
        if metric_name not in metrics_df.columns:
            raise ValueError(
                f"Auto-curation rule {rule_name!r} "
                f"references metric column {metric_name!r}, which is not in the "
                f"computed quality metrics {sorted(metrics_df.columns)}. Add "
                "it to the QualityMetricParameters row's metric_names (or fix "
                "the rule) before populating."
            )
        compare = _COMPARISON_TO_FUNCTION[rule["operator"]]
        column = metrics_df[metric_name]
        label = rule["label"]
        threshold = rule["threshold"]
        if not math.isfinite(threshold):
            raise ValueError(
                f"Auto-curation rule {rule_name!r} has a non-finite "
                f"threshold ({threshold!r}); a rule's threshold must be a "
                "finite number."
            )
        missing_policy = rule.get("missing_policy", "error")
        if missing_policy not in {"error", "fail", "pass"}:
            raise ValueError(
                f"Auto-curation rule {rule_name!r} "
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

        if expected_missing is None:
            # No classifier was given: every non-finite value is handled by
            # missing_policy alone.
            policy_missing_ids = missing_unit_ids
        else:
            expected_set = expected_missing.get(metric_name)
            if expected_set is None:
                # No registered eligibility rule for this column (or it was
                # never classified): fail closed on any non-finite value.
                if missing_unit_ids:
                    raise ValueError(
                        f"Auto-curation rule {rule_name!r}: "
                        + _unregistered_column_message(
                            metric_name, missing_unit_ids
                        )
                    )
                policy_missing_ids = []
            else:
                unexpected_ids = [
                    unit_id
                    for unit_id in missing_unit_ids
                    if unit_id not in expected_set
                ]
                if unexpected_ids:
                    raise ValueError(
                        f"Auto-curation rule {rule_name!r}: "
                        + _computation_failure_message(
                            metric_name, unexpected_ids
                        )
                    )
                policy_missing_ids = [
                    unit_id
                    for unit_id in missing_unit_ids
                    if unit_id in expected_set
                ]

        if missing_policy == "error" and policy_missing_ids:
            raise ValueError(
                f"Auto-curation rule {rule_name!r} "
                f"references metric {metric_name!r}, which has non-finite "
                f"values for unit_id(s) {policy_missing_ids}. Fix the metric "
                "computation or choose an explicit missing_policy ('fail' "
                "or 'pass') for this rule."
            )
        if (
            missing_policy in {"pass", "fail"}
            and policy_missing_ids
            and len(policy_missing_ids) == len(metrics_df.index)
        ):
            # A per-unit NaN is an expected low-spike skip, but a metric that
            # is missing for EVERY unit means the rule made no real
            # comparison at all, so the rule is silently inert. Say so
            # without raising: "pass" applied no labels; "fail" applied its
            # label to everyone for that reason, not because any unit
            # crossed the threshold.
            if missing_policy == "pass":
                logger.warning(
                    f"Auto-curation rule {rule_name!r} "
                    f"was inert: metric {metric_name!r} is non-finite for "
                    f"all {len(policy_missing_ids)} unit(s), so the rule "
                    f"applied no {label!r} labels. Check the metric "
                    "computation."
                )
            else:
                logger.warning(
                    f"Auto-curation rule {rule_name!r} "
                    f"applied its {label!r} label to all "
                    f"{len(policy_missing_ids)} unit(s) because metric "
                    f"{metric_name!r} is non-finite for every one of them, "
                    "not because any unit crossed the threshold. Check the "
                    "metric computation."
                )
        for unit_id in metrics_df.index:
            value = column.loc[unit_id]
            if not finite_by_unit[unit_id]:
                if missing_policy == "fail":
                    unit_labels = labels.setdefault(int(unit_id), [])
                    if label not in unit_labels:
                        unit_labels.append(label)
                continue
            if not compare(value, threshold):
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
    that merge updates those class dicts in place, so a direct compute with
    custom kwargs changes the defaults for later computes in the same
    process. ``CurationEvaluation`` computes inside
    ``isolated_si_metric_defaults``, which leaves them unchanged; reading
    them here at call time, rather than a fixed copy, keeps the classifier
    on the params SI applies either way. The defaults are copied, never
    mutated, under ``SI_METRIC_STATE_LOCK`` so no isolated compute on
    another thread swaps them mid-read.
    """
    from spikeinterface.metrics.quality import (
        get_default_quality_metrics_params,
    )

    from spyglass.spikesorting.v2._si_metric_patches import (
        SI_METRIC_STATE_LOCK,
    )

    with SI_METRIC_STATE_LOCK:
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


def _unregistered_column_message(column: str, unit_ids: list) -> str:
    """Message: a column with no eligibility rule is non-finite somewhere."""
    return (
        f"Metric column {column!r} is non-finite for unit_id(s) "
        f"{unit_ids}, and it has no registered eligibility "
        "rule, so a NaN cannot be classified as an expected "
        "unassessable unit rather than a computation failure. "
        "Register an eligibility rule for this metric, or "
        "threshold a metric that has one."
    )


def _computation_failure_message(column: str, unit_ids: list) -> str:
    """Message: a column is non-finite for a unit that should compute fine."""
    return (
        f"Metric column {column!r} is non-finite for unit_id(s) "
        f"{unit_ids}, which meet the metric's preconditions: this is a "
        "metric computation failure, not an unassessable unit. Check "
        "the quality-metric computation (SpikeInterface replaces a "
        "failing metric with NaN and only warns)."
    )


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
                    _unregistered_column_message(column, non_finite)
                )
            continue
        failed = [unit_id for unit_id in non_finite if unit_id not in expected]
        if failed:
            raise ValueError(_computation_failure_message(column, failed))


# ---------- SpikeInterface metric-computation errors -------------------------
#
# SI runs each metric in its own ``try``; on any exception it warns
# ``f"Error computing metric {metric_name}: {e}"`` and fills every column of
# that metric with NaN (``core/analyzer_extension_core.py:1270-1286``). The
# same calculator computes the template metrics. ``metric_name`` is SI's
# metric name (``nn_advanced``, ``half_width``), not an output column.
_SI_METRIC_ERROR = re.compile(
    r"Error computing metric (?P<metric>\S+?): (?P<error>.*)", re.DOTALL
)


def si_quality_metric_output_columns(si_metric: str) -> tuple[str, ...]:
    """Ordered output columns SpikeInterface fills for one quality metric.

    Read from SI's own metric metadata: the quality metric class's
    ``metric_columns`` (``core/analyzer_extension_core.py:838``), found by
    name through ``BaseMetricExtension.get_metric_by_name`` (1052-1070), in
    SI's order. Spyglass's ``isi_violation`` fraction is derived from SI's
    ``isi_violation`` metric, so it leads that metric's columns. A name that
    is not an SI quality metric maps to no columns.
    """
    from spikeinterface.metrics.quality import ComputeQualityMetrics

    if si_metric not in ComputeQualityMetrics.get_available_metric_names():
        return ()
    metric = ComputeQualityMetrics.get_metric_by_name(si_metric)
    columns = tuple(metric.metric_columns)
    if si_metric == "isi_violation":
        columns = ("isi_violation", *columns)
    return columns


def si_metric_output_columns(si_metric: str) -> frozenset[str]:
    """Output columns SpikeInterface fills for one SI metric name.

    The quality metric's columns (:func:`si_quality_metric_output_columns`,
    which includes Spyglass's ``isi_violation`` fraction) plus the template
    metric class's ``metric_columns`` when the name is a template metric. An
    unknown name maps to no columns.
    """
    from spikeinterface.metrics.template import ComputeTemplateMetrics

    columns = set(si_quality_metric_output_columns(si_metric))
    if si_metric in ComputeTemplateMetrics.get_available_metric_names():
        metric = ComputeTemplateMetrics.get_metric_by_name(si_metric)
        columns.update(metric.metric_columns)
    return frozenset(columns)


def _si_metric_error_message(
    si_metric: str, error: str, rule_columns: list[str]
) -> str:
    """Message: SI failed a metric whose column a rule thresholds."""
    return (
        f"SpikeInterface failed to compute metric {si_metric!r} ({error}), "
        f"so rule-referenced column(s) {rule_columns} are NaN for every "
        "unit and every auto-curation rule on them would make no real "
        "comparison. SpikeInterface replaces a failing metric with NaN and "
        "only warns; fix the metric computation or remove the rule."
    )


def report_si_metric_errors(
    caught: Iterable[warnings.WarningMessage],
    rule_columns: Collection[str],
) -> None:
    """Act on warnings recorded around a SpikeInterface metric compute.

    Parameters
    ----------
    caught : iterable of warnings.WarningMessage
        Warnings recorded with ``warnings.catch_warnings(record=True)``.
    rule_columns : collection of str
        Metric columns referenced by auto-curation rules.

    Raises
    ------
    ValueError
        If any "Error computing metric" warning names an SI metric one of
        whose output columns (``si_metric_output_columns``) is in
        ``rule_columns``; the message names the metric, those columns and
        SI's error text.

    Notes
    -----
    Every other warning is re-emitted unchanged, so it passes through the
    caller's warning filters as if it had never been recorded. A metric
    error on columns no rule references is logged at WARNING and its
    columns stay NaN. All warnings are handled before anything is raised.
    """
    rule_columns = frozenset(rule_columns)
    failures = []
    for record in caught:
        match = _SI_METRIC_ERROR.fullmatch(str(record.message))
        if match is None:
            warnings.warn_explicit(
                record.message,
                record.category,
                record.filename,
                record.lineno,
                source=record.source,
            )
            continue
        si_metric, error = match.group("metric"), match.group("error")
        referenced = sorted(si_metric_output_columns(si_metric) & rule_columns)
        if referenced:
            failures.append(
                _si_metric_error_message(si_metric, error, referenced)
            )
        else:
            logger.warning(
                "SpikeInterface failed to compute metric %r (%s); its "
                "columns are NaN for every unit.",
                si_metric,
                error,
            )
    if failures:
        raise ValueError("\n".join(failures))


@contextmanager
def escalate_si_metric_errors(rule_columns: Collection[str]):
    """Raise on SpikeInterface metric errors that affect rule columns.

    Records every warning raised in the block (``simplefilter("always")``,
    so neither a caller's ``ignore`` filter nor the once-per-location
    registry can hide one) and passes them to ``report_si_metric_errors``
    when the block exits. If the block itself raises, that error
    propagates unchanged: metric errors are then only logged and other
    warnings are still re-emitted.

    ``warnings.catch_warnings`` swaps process-wide state, so the block
    holds ``SI_METRIC_STATE_LOCK`` (``_si_metric_patches``) from before the
    capture starts until its warnings are reported: captures on different
    threads run one after another and each records only its own thread's
    warnings. Warnings raised meanwhile by other (non-Spyglass) threads are
    still recorded here.

    Parameters
    ----------
    rule_columns : collection of str
        Metric columns referenced by auto-curation rules; empty escalates
        nothing.
    """
    from spyglass.spikesorting.v2._si_metric_patches import (
        SI_METRIC_STATE_LOCK,
    )

    caught: list[warnings.WarningMessage] = []
    with SI_METRIC_STATE_LOCK:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                yield
        except BaseException:
            report_si_metric_errors(caught, frozenset())
            raise
        report_si_metric_errors(caught, rule_columns)


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
    Numbers are compared with ``math.isclose`` rather than ``==`` because rows
    stored before the ``Rule.threshold`` column became ``double`` keep their
    single-precision values: a threshold such as ``0.1`` stored while the
    column was still ``float`` reads back as ``0.10000000149...`` after the
    column is widened, which is not bit-equal to the freshly validated Python
    float. An exact comparison would raise a spurious "different payload"
    error when re-inserting identical rules against such a row (e.g. running
    ``insert_default()`` after the documented column upgrade), breaking
    idempotency. The default ``rel_tol`` comfortably exceeds float32
    round-off (~6e-8) while still distinguishing thresholds that differ by
    more than one part per million.
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


# Extensions CurationEvaluation adds to the sort-time analyzer before metrics and
# auto-merge. The sort-time base set (random_spikes, noise_levels, templates,
# waveforms) is already present; these derive from it. ``principal_components``
# is added separately, only when PCA metrics are requested.
_CURATION_EXTENSIONS = STANDARD_DISPLAY_ANALYZER_EXTENSIONS

# Pinned principal_components params for the whitened METRIC analyzer. The
# recording is already SPATIALLY whitened (sip.whiten decorrelates channels);
# SI's PCA then computes components on those decorrelated waveforms and, with
# whiten=True, normalizes the component variances -- the standard input space
# for the PC/NN cluster-separation metrics. These are SI 0.104's defaults
# (including dtype), pinned explicitly so a future SI default change cannot
# silently alter the whitened-space PCA (and therefore the PC/NN metric values).
_PCA_EXTENSION_PARAMS = {
    "n_components": 5,
    "mode": "by_channel_local",
    "whiten": True,
    "dtype": "float32",
}


def _pca_params_match(existing: dict) -> bool:
    """True if a stored ``principal_components`` matches the pinned params.

    Compared key-by-key over ``_PCA_EXTENSION_PARAMS`` only (SI may store extra
    keys); ``dtype`` is normalized through ``np.dtype`` so the stored
    ``dtype('float32')`` matches the pinned ``"float32"`` string.
    """
    import numpy as np

    for key, pinned in _PCA_EXTENSION_PARAMS.items():
        value = existing.get(key)
        if key == "dtype":
            if value is None or np.dtype(value) != np.dtype(pinned):
                return False
        elif value != pinned:
            return False
    return True


_AUTO_MERGE_EXTRA_EXTENSIONS = {
    # SI's ``feature_neighbors`` preset includes the ``knn`` step, whose
    # required extensions are templates, spike_locations, and spike_amplitudes.
    # The sort-time/base curation extension set already covers templates and
    # spike_amplitudes; add spike_locations only for that preset.
    "feature_neighbors": ("spike_locations",),
}


def _requested_pc_metrics(metric_names) -> list[str]:
    """PC/NN metrics among ``metric_names`` (route to the whitened analyzer).

    PCA-based metrics (``d_prime``, ``mahalanobis``, ``nearest_neighbor``,
    ``nn_advanced``, ``silhouette`` in SI 0.104) need the ``principal_components``
    extension and measure cluster separation, so they compute in the
    decorrelated (whitened) space -- the whitened METRIC analyzer; everything
    else stays on the unwhitened DISPLAY analyzer. The PCA set is the same one
    the insert-time validator uses (``_available_pca_metric_names``), so routing
    and validation cannot disagree about which metrics are PCA-based.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        _available_pca_metric_names,
    )

    pca = set(_available_pca_metric_names())
    return [name for name in metric_names if name in pca]


def evaluate_analyzers(
    table,
    display_analyzer,
    metric_analyzer,
    *,
    metric_names,
    metric_kwargs,
    skip_pc_metrics,
    metric_job_kwargs,
    template_metric_columns,
    auto_merge_preset,
    auto_merge_kwargs,
    rule_rows,
    expected_unit_ids,
    observation_metrics=None,
    statistics_spans=None,
):
    """Compute metrics / labels / merge suggestions and enforce namespace.

    Reuses ``table._compute_metrics`` (the ``CurationEvaluation``
    staticmethod, called on the table so a patched one takes effect) /
    :func:`compute_merge_groups` and ``apply_label_rules`` over the
    curation's analyzers, then enforces
    the unit-namespace invariant BEFORE labels/merges are returned (and
    before the NWB write): the metric index must equal the curation's unit
    set, and every suggested merge member must be a unit in that set. This
    catches a stale temp analyzer, accidental raw-sort analyzer reuse, or a
    preview row that slipped past selection. ``statistics_spans`` (the
    sort's persisted spans; ``None`` = the whole recording) are forwarded
    to ``_compute_metrics``.

    Rule-referenced metrics must be computed wherever SpikeInterface can
    compute them: a metric SpikeInterface failed (``_compute_metrics``'s
    ``rule_columns``), or a non-finite value for a unit that meets the
    metric's preconditions (``expected_missing_units``), raises
    ``ValueError`` instead of leaving the rule silently inert. A NaN for a
    unit SpikeInterface cannot assess (e.g. below ``nn_advanced``'s
    ``min_spikes``) follows the rule's ``missing_policy``.

    The whole evaluation holds ``SI_METRIC_STATE_LOCK``
    (``_si_metric_patches``), taken after any ``analyzer_cache_lock``
    the caller holds, so evaluations on different threads of one process
    run one after another.
    """
    from spyglass.spikesorting.v2._si_metric_patches import (
        SI_METRIC_STATE_LOCK,
    )

    # SpikeInterface's metric defaults (which the classifier reads) and
    # Spyglass's capture of SpikeInterface warnings are process-wide:
    # hold the lock from the computes through the classification so no
    # evaluation on another thread changes them in between. The merge
    # suggestions stay inside too: SI's auto-merge can compute
    # quality_metrics itself (spikeinterface/curation/auto_merge.py:256).
    with SI_METRIC_STATE_LOCK:
        rule_columns = frozenset(row["metric_name"] for row in rule_rows)
        metrics_df = table._compute_metrics(
            display_analyzer,
            metric_analyzer,
            metric_names,
            metric_kwargs,
            skip_pc_metrics,
            metric_job_kwargs,
            template_metric_columns=template_metric_columns,
            statistics_spans=statistics_spans,
            rule_columns=rule_columns,
        )
        assert_unit_namespace(metrics_df, expected_unit_ids)
        if observation_metrics is not None:
            metrics_df = metrics_df.join(observation_metrics)
        n_spikes_by_unit = spike_counts(display_analyzer, metric_analyzer)
        total_samples = display_analyzer.get_total_samples()
        if (
            metric_analyzer is not None
            and metric_analyzer.get_total_samples() != total_samples
        ):
            raise ValueError(
                "The display and metric analyzers disagree on the "
                f"recording's total samples (display={total_samples}, "
                f"metric={metric_analyzer.get_total_samples()}); both "
                "must be built from the same traces, since the "
                "eligibility classifier rates every metric's spikes over "
                "one duration."
            )
        expected_missing = expected_missing_units(
            rule_columns,
            n_spikes_by_unit=n_spikes_by_unit,
            # SI rates spikes over total samples / fs, not the time-vector
            # span (spikeinterface/metrics/utils.py:100-126).
            total_samples=total_samples,
            sampling_frequency=display_analyzer.sampling_frequency,
            metric_kwargs=metric_kwargs or {},
        )
        assert_rule_metrics_computed(metrics_df, rule_columns, expected_missing)
        labels_by_unit = apply_label_rules(
            metrics_df, rule_rows, expected_missing=expected_missing
        )
        merge_groups = compute_merge_groups(
            display_analyzer,
            auto_merge_preset,
            auto_merge_kwargs,
            metric_job_kwargs,
        )
        assert_merge_membership(merge_groups, expected_unit_ids)
        return metrics_df, labels_by_unit, merge_groups


def spike_counts(display_analyzer, metric_analyzer) -> dict[int, int]:
    """Per-unit spike counts, identical on both analyzers.

    The voltage metrics count spikes on the display analyzer and the
    PC/NN metrics on the whitened metric analyzer (when there is one).
    Both are built from the same sorting, so their full per-unit counts
    must agree; the eligibility classifier uses one set for both.
    """
    counts = {
        int(unit_id): int(n)
        for unit_id, n in (
            display_analyzer.sorting.count_num_spikes_per_unit().items()
        )
    }
    if metric_analyzer is not None:
        metric_counts = {
            int(unit_id): int(n)
            for unit_id, n in (
                metric_analyzer.sorting.count_num_spikes_per_unit().items()
            )
        }
        if metric_counts != counts:
            raise ValueError(
                "The display and metric analyzers disagree on per-unit "
                f"spike counts (display={counts}, metric={metric_counts}); "
                "both must be built from the same sorting."
            )
    return counts


def assert_unit_namespace(metrics_df, expected_unit_ids) -> None:
    """Raise unless the metric index equals the curation's unit set."""
    computed = {int(u) for u in metrics_df.index}
    expected = {int(u) for u in expected_unit_ids}
    if computed != expected:
        raise ValueError(
            "CurationEvaluation namespace invariant violated: computed "
            f"metric unit ids {sorted(computed)} != the curation's unit set "
            f"{sorted(expected)}. The metrics must be scored over the "
            "evaluated curation's own units (e.g. the merged unit set), not "
            "the raw-sort analyzer; this indicates a stale temp analyzer or "
            "accidental raw-analyzer reuse."
        )


def assert_merge_membership(merge_groups, expected_unit_ids) -> None:
    """Raise unless every suggested merge member is a curation unit."""
    expected = {int(u) for u in expected_unit_ids}
    for group in merge_groups:
        members = {int(u) for u in group}
        if not members <= expected:
            raise ValueError(
                "CurationEvaluation merge-suggestion invariant violated: "
                f"suggested merge {sorted(members)} contains units outside "
                f"the curation's unit set {sorted(expected)}."
            )


def compute_metrics(
    display_analyzer,
    metric_analyzer,
    metric_names,
    metric_kwargs,
    skip_pc_metrics,
    job_kwargs=None,
    template_metric_columns=None,
    statistics_spans=None,
    rule_columns=frozenset(),
):
    """Compute quality metrics, routing PC/NN metrics to the whitened one.

    Voltage / spike-train metrics (``snr``, ``amplitude_*``,
    ``firing_rate``, ``num_spikes``, ``presence_ratio``, ``isi_violation``)
    compute on the unwhitened DISPLAY analyzer -- whitening normalizes
    per-channel variance, so SNR / amplitude on whitened traces would be
    meaningless. PC / cluster-separation metrics (the SI PCA-metric set)
    compute on the whitened METRIC analyzer, where decorrelated separation
    is meaningful. The two frames are merged by unit id. Spyglass's
    ``isi_violation`` fraction is added from the (recipe-independent) spike
    times. ``job_kwargs`` drive the heavy extension computation.

    ``template_metric_columns`` (SI output COLUMN names) are surfaced from
    the DISPLAY analyzer's ``template_metrics`` extension -- waveform SHAPE
    must come from real, unwhitened templates, exactly as ``snr`` does. The
    columns are selected directly (config already holds column names, so no
    name->column mapping) and joined onto the result by unit id; they are
    exposed for downstream cell typing, never thresholded here.

    ``statistics_spans`` are the sort's artifact-free, join-free frame
    spans (``Sorting.get_statistics_spans``); ``nn_noise_overlap`` draws
    its noise cluster only from inside them. They are set (via
    ``noise_cluster_spans``) around both metric computes, which run in
    this process. ``sd_ratio``'s noise standard deviation is likewise
    estimated from the span samples, and its correction for the unit's
    own template variance counts only the spikes and samples inside
    them. ``None`` (or one span covering the recording) keeps
    SpikeInterface's whole-recording estimates.

    ``rule_columns`` are the metric columns auto-curation rules
    threshold. SpikeInterface turns a metric that raises into a warning
    and an all-NaN column; around every compute that can run quality or
    template metrics, such a failure raises ``ValueError`` if it hits a
    rule column and is logged otherwise (``escalate_si_metric_errors``).
    The default, empty, escalates nothing.
    """
    import numpy as np
    import pandas as pd
    from spikeinterface.metrics.quality import compute_quality_metrics

    from spyglass.spikesorting.v2._si_metric_patches import (
        isolated_si_metric_defaults,
        noise_cluster_spans,
    )
    from spyglass.spikesorting.v2._params.metric_curation import (
        required_extensions_for_metrics,
    )
    from spyglass.spikesorting.v2._sorting_analyzer import (
        ensure_extensions,
    )

    metric_kwargs = metric_kwargs or {}
    pc_names = _requested_pc_metrics(metric_names)
    pc_set = set(pc_names)
    voltage_names = [m for m in metric_names if m not in pc_set]

    frames = []
    # Voltage / spike-train metrics -> unwhitened display analyzer.
    if voltage_names:
        # Compute each requested voltage metric's display-safe extension
        # dependencies (read from SI's registry, not hardcoded) beyond the
        # default curation set -- otherwise SI silently skips a metric whose
        # extension is absent (e.g. ``drift`` needs ``spike_locations``),
        # leaving the column missing and any rule thresholding it never
        # firing. ``principal_components`` is excluded defensively: it is
        # metric-analyzer-only and voltage metrics never depend on it.
        base_present = {
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            *_CURATION_EXTENSIONS,
        }
        extra_extensions = [
            ext
            for ext in required_extensions_for_metrics(
                voltage_names, base_present
            )
            if ext != "principal_components"
        ]
        # _CURATION_EXTENSIONS includes template_metrics.
        with escalate_si_metric_errors(rule_columns):
            ensure_extensions(
                display_analyzer,
                list(_CURATION_EXTENSIONS) + extra_extensions,
                job_kwargs=job_kwargs,
            )
        if "sd_ratio" in voltage_names:
            # SI's sd_ratio divides by get_noise_levels(method="std") on
            # this analyzer's recording, which returns a cached
            # noise_level_std_* property when present. Cache the std of
            # the span samples there (same seed and budget as the
            # analyzer's MAD), so masked zeros do not bias it low.
            # Covering spans cache nothing and leave SI's estimator.
            # SI's correction for the unit's own template variance must
            # then count spikes and samples over the same spans; the
            # patched metric reads them from noise_cluster_spans below
            # (SI runs each metric in this thread).
            from spyglass.spikesorting.v2._si_metric_patches import (
                patch_sd_ratio_statistics_spans,
            )
            from spyglass.spikesorting.v2._sorting_dispatch import (
                cache_span_noise_levels,
            )

            cache_span_noise_levels(
                display_analyzer.recording,
                statistics_spans,
                return_in_uV=display_analyzer.return_in_uV,
                seed=(job_kwargs or {}).get("random_seed", 0),
                method="std",
            )
            patch_sd_ratio_statistics_spans()
        # SI would otherwise keep this row's kwargs as its defaults for
        # every later compute in the process.
        with (
            noise_cluster_spans(statistics_spans),
            isolated_si_metric_defaults(),
            escalate_si_metric_errors(rule_columns),
        ):
            voltage_df = compute_quality_metrics(
                display_analyzer,
                metric_names=voltage_names,
                metric_params={
                    k: v for k, v in metric_kwargs.items() if k in voltage_names
                }
                or None,
                skip_pc_metrics=True,
                # The analyzer is shared across curations; SI preserves
                # the stored quality_metrics by default, so a prior
                # curation's columns would leak into this result (and an
                # auto-rule could threshold a stale metric). Compute only
                # THIS row's metrics.
                delete_existing_metrics=True,
            )
        voltage_df.index = voltage_df.index.astype(int)
        frames.append(voltage_df)

    # PC / cluster-separation metrics -> whitened metric analyzer.
    if pc_names and not skip_pc_metrics:
        if metric_analyzer is None:
            raise ValueError(
                "PC/NN metrics were requested but no whitened metric "
                "analyzer was provided -- make_compute must build it when "
                "PC metrics are requested."
            )
        # nn_noise_overlap needs a 'median' template AND SI 0.104.3's metric
        # is broken for sparse analyzers (it derives the peak channel from a
        # dense median but indexes the sparse noise cluster -> IndexError,
        # swallowed as NaN). Ensure the median operator, then install the
        # sparse fix. The PC compute below runs n_jobs=1 so the fix (a
        # main-process monkeypatch) is the code that actually runs.
        from spyglass.spikesorting.v2._si_metric_patches import (
            patch_nn_noise_overlap_sparsity,
        )

        ensure_extensions(metric_analyzer, ["templates"], job_kwargs=job_kwargs)
        templates_operators = list(
            metric_analyzer.get_extension("templates").params.get("operators")
            or []
        )
        if "median" not in templates_operators:
            metric_analyzer.compute(
                "templates", operators=[*templates_operators, "median"]
            )
        patch_nn_noise_overlap_sparsity()
        # Enforce the pinned PCA params even if a stale/manual analyzer
        # already carries principal_components computed with different ones
        # (ensure_extensions skips a present extension without checking its
        # params): drop a mismatched extension so it recomputes pinned.
        if metric_analyzer.has_extension("principal_components"):
            existing_pca = dict(
                metric_analyzer.get_extension("principal_components").params
            )
            if not _pca_params_match(existing_pca):
                logger.warning(
                    "CurationEvaluation: principal_components on the metric "
                    f"analyzer has params {existing_pca} != pinned "
                    f"{_PCA_EXTENSION_PARAMS}; deleting and recomputing "
                    "with the pinned params so PC/NN metrics are consistent."
                )
                metric_analyzer.delete_extension("principal_components")
        ensure_extensions(
            metric_analyzer,
            ["principal_components"],
            job_kwargs=job_kwargs,
            extension_params={"principal_components": _PCA_EXTENSION_PARAMS},
        )
        with (
            noise_cluster_spans(statistics_spans),
            isolated_si_metric_defaults(),
            escalate_si_metric_errors(rule_columns),
        ):
            pc_df = compute_quality_metrics(
                metric_analyzer,
                metric_names=pc_names,
                metric_params={
                    k: v for k, v in metric_kwargs.items() if k in pc_names
                }
                or None,
                skip_pc_metrics=False,
                # As above: compute only this row's PC metrics, never
                # inherit a prior curation's stored quality_metrics on this
                # analyzer.
                delete_existing_metrics=True,
                # nn_noise_overlap's sparse fix and its span-restricted
                # noise cluster (patch_nn_noise_overlap_sparsity /
                # noise_cluster_spans) are a main-process monkeypatch and a
                # ContextVar; SI parallelises nn_advanced with spawned
                # workers that re-import SI and would see neither, so pin
                # the PC/NN metric compute to the main process.
                n_jobs=1,
            )
        pc_df.index = pc_df.index.astype(int)
        frames.append(pc_df)

    if not frames:
        raise ValueError(
            "_compute_metrics: no metrics to compute -- metric_names "
            f"{sorted(metric_names)} contains only PC/NN metrics but "
            "skip_pc_metrics=True. Set skip_pc_metrics=False to compute "
            "them, or include a voltage-based metric."
        )
    if len(frames) > 1:
        # The display and metric analyzers derive from the SAME canonical
        # sorting, so the voltage and PC frames must share a unit-id index.
        # Concat defaults to an OUTER join that would silently NaN-fill a
        # mismatched unit (and a label rule would then never fire on the
        # NaN); assert the set invariant loudly. Ordering differences are
        # harmless, so reindex PC metrics to the display order before
        # concatenating.
        if set(frames[0].index) != set(frames[1].index):
            raise ValueError(
                "voltage and PC metric frames have mismatched unit ids "
                f"(voltage={sorted(frames[0].index)}, "
                f"pc={sorted(frames[1].index)}); both derive from the same "
                "canonical sorting, so this indicates an analyzer build "
                "divergence and the metrics cannot be safely merged."
            )
        frames[1] = frames[1].reindex(frames[0].index)
        metrics_df = pd.concat(frames, axis=1)
    else:
        metrics_df = frames[0]

    if (
        "isi_violation" in metric_names
        and "isi_violations_count" in metrics_df.columns
    ):
        counts = metrics_df["isi_violations_count"].to_numpy()
        n_by_unit = display_analyzer.sorting.count_num_spikes_per_unit()
        n_spikes = np.array(
            [n_by_unit[int(u)] for u in metrics_df.index], dtype=float
        )
        metrics_df["isi_violation"] = isi_violation_fraction(counts, n_spikes)

    # Surfacing the configured shape columns must not depend on a voltage
    # metric having grown the display extensions: a PC-only row (e.g.
    # metric_names=["nn_advanced"], skip_pc_metrics=False) skips the voltage
    # branch above, so ensure template_metrics on the display analyzer
    # whenever shape columns are requested -- otherwise they would be
    # silently dropped rather than surfaced.
    if template_metric_columns and not display_analyzer.has_extension(
        "template_metrics"
    ):
        with escalate_si_metric_errors(rule_columns):
            ensure_extensions(
                display_analyzer,
                ["template_metrics"],
                job_kwargs=job_kwargs,
            )

    metrics_df = surface_template_columns(
        metrics_df, display_analyzer, template_metric_columns
    )
    return metrics_df


def surface_template_columns(
    metrics_df, display_analyzer, template_metric_columns
):
    """Join configured waveform-shape columns onto the metric frame.

    Reads the already-computed ``template_metrics`` extension from the
    DISPLAY (unwhitened) analyzer per the display-vs-metric routing
    contract, selects the configured output COLUMNS directly (no
    name->column mapping), and joins them by unit id so a unit missing from
    either frame yields ``NaN`` rather than a misaligned row. Surfaces, does
    not threshold. A no-op when no columns are configured or the extension
    is absent (e.g. a PC-only row that never grew the display extensions).
    """
    if not template_metric_columns:
        return metrics_df
    if not display_analyzer.has_extension("template_metrics"):
        return metrics_df
    tm_df = display_analyzer.get_extension("template_metrics").get_data()
    # Select the configured columns directly (config holds column names, so
    # no name->column mapping), never shadowing a same-named quality-metric
    # column with a template one.
    present = [
        c
        for c in template_metric_columns
        if c in tm_df.columns and c not in metrics_df.columns
    ]
    missing = [c for c in template_metric_columns if c not in tm_df.columns]
    if missing:
        # Validation guarantees the configured columns are real
        # single-channel output columns, so this only fires if SI's default
        # template_metrics compute omits a validated column -- an
        # upstream-version drift signal, surfaced loudly, not silent.
        logger.warning(
            "template_metric_columns %s absent from computed "
            "template_metrics columns %s; surfacing %s only.",
            missing,
            list(tm_df.columns),
            present,
        )
    # Copy the selected columns before retyping the index so the analyzer's
    # cached template_metrics frame is never mutated in place.
    selected = tm_df[present].copy()
    selected.index = selected.index.astype(int)
    return metrics_df.join(selected)


def compute_merge_groups(
    analyzer, auto_merge_preset, auto_merge_kwargs, job_kwargs=None
):
    """Return proposed merge groups for a preset (``[]`` for 'none')."""
    if auto_merge_preset == "none":
        return []
    from spikeinterface.curation import compute_merge_unit_groups

    from spyglass.spikesorting.v2._sorting_analyzer import (
        ensure_extensions,
    )

    extensions = list(_CURATION_EXTENSIONS)
    extensions.extend(_AUTO_MERGE_EXTRA_EXTENSIONS.get(auto_merge_preset, ()))
    ensure_extensions(analyzer, extensions, job_kwargs=job_kwargs)
    merge_job_kwargs = {
        key: value
        for key, value in (job_kwargs or {}).items()
        if key != "random_seed"
    }
    compute_kwargs = dict(auto_merge_kwargs or {})
    compute_kwargs.update(merge_job_kwargs)
    groups = compute_merge_unit_groups(
        analyzer,
        preset=auto_merge_preset,
        compute_needed_extensions=False,
        **compute_kwargs,
    )
    return [[int(u) for u in group] for group in groups]
