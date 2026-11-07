"""Metric computation and labeling for ``CurationEvaluation``.

- ``compute_metrics``: voltage metrics on the display analyzer, PC/NN
  metrics on the whitened metric analyzer.
- ``apply_label_rules``: threshold rules to per-unit labels.
- ``expected_missing_units``: separates an unassessable unit's NaN from a
  failed computation.
- ``evaluate_analyzers``: metrics, labels, and merge groups with namespace
  checks.
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

from spyglass.spikesorting._numerical import (
    integer_scalar,
    integer_vector,
)

from spyglass.spikesorting.v2._sorting.analyzer import (
    BASE_ANALYZER_EXTENSIONS,
    STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
)
from spyglass.utils import logger

# Mirrors ``AutoCurationRules.Rule.operator`` and ``RuleOperator``. Non-finite
# values are filtered before dispatch, so NaN never matches (even ``!=``).
_COMPARISON_TO_FUNCTION = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _is_finite_metric_value(value) -> bool:
    """Return whether one metric scalar can be thresholded.

    NaN is filtered silently; a non-numeric value (SI output drift) is
    filtered with a warning.
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


_POLARITY_METRICS = ("snr", "sd_ratio", "nn_advanced")


def apply_metric_peak_sign(
    metric_names, metric_kwargs: dict, peak_sign
) -> dict:
    """Set the sort's ``peak_sign`` on every requested polarity metric.

    SI reads these metrics at the template extremum for ``peak_sign``
    (default ``"neg"``), so a positive or bidirectional sort would measure the
    wrong channel. Other kwargs are kept; the input is not mutated.

    Parameters
    ----------
    metric_names : iterable of str
        Requested quality-metric names.
    metric_kwargs : dict
        Per-metric kwargs mapping (``{metric_name: {...}}``).
    peak_sign : {"neg", "pos", "both"}
        The sort's resolved polarity.

    Returns
    -------
    dict
        ``metric_kwargs`` with ``peak_sign`` set on each requested polarity
        metric.
    """
    out = dict(metric_kwargs)
    for name in _POLARITY_METRICS:
        if name in metric_names:
            out[name] = {
                **(metric_kwargs.get(name) or {}),
                "peak_sign": peak_sign,
            }
    return out


def apply_metric_seed(metric_names, metric_kwargs: dict, seed) -> dict:
    """Seed the requested random metrics a row leaves unseeded.

    ``nn_advanced`` and ``sd_ratio`` draw random samples, so unseeded
    re-evaluations can change labels. A row's own seed wins. Returns a new
    mapping.

    Parameters
    ----------
    metric_names : iterable of str
        Requested quality-metric names.
    metric_kwargs : dict
        Per-metric kwargs mapping (``{metric_name: {...}}``).
    seed : int
        The display analyzer's noise seed.

    Returns
    -------
    dict
        ``metric_kwargs`` with every requested random metric seeded.
    """
    out = dict(metric_kwargs)
    if "nn_advanced" in metric_names:
        nn_kwargs = dict(metric_kwargs.get("nn_advanced") or {})
        if nn_kwargs.get("seed") is None:
            nn_kwargs["seed"] = seed
        out["nn_advanced"] = nn_kwargs
    if "sd_ratio" in metric_names:
        sd_kwargs = dict(metric_kwargs.get("sd_ratio") or {})
        slices = dict(sd_kwargs.get("random_slices_kwargs") or {})
        if slices.get("seed") is None:
            slices["seed"] = seed
        out["sd_ratio"] = {**sd_kwargs, "random_slices_kwargs": slices}
    return out


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
        metric. A non-finite entry never satisfies a threshold.
    rule_rows : list of dict
        ``AutoCurationRules.Rule`` rows, each with ``rule_index``,
        ``metric_name``, ``operator``, ``threshold``, and ``label``. Rules are
        applied in ascending ``rule_index``. ``threshold`` must be finite.
        ``missing_policy`` defaults to ``"error"``; ``"fail"`` applies the
        rule label to a unit whose value is non-finite, while ``"pass"``
        leaves it unlabelled.
    expected_missing : dict[str, set or None] or None, optional
        Output of ``expected_missing_units`` for the rule columns. When
        given, ``missing_policy`` governs only expected-missing units; any
        other non-finite value, or one in a column with no eligibility rule
        (``None`` or absent), raises. When ``None``, every non-finite value
        follows ``missing_policy``.

    Returns
    -------
    dict[int, list[str]]
        ``unit_id -> [label, ...]`` for units matching at least one rule.

    Raises
    ------
    ValueError
        If a rule references a metric column absent from ``metrics_df``; if a
        rule's ``threshold`` is non-finite; if any unit has a non-finite
        value and the rule's missing policy is ``"error"``; or if
        ``expected_missing`` classifies a unit's non-finite value as a metric
        computation failure (regardless of ``missing_policy``).
    """
    _metric_unit_ids(metrics_df.index)
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
            policy_missing_ids = missing_unit_ids
        else:
            # An unclassified (absent) column fails closed.
            policy_missing_ids = _expected_missing_unit_ids(
                metric_name,
                missing_unit_ids,
                expected_missing.get(metric_name),
                prefix=f"Auto-curation rule {rule_name!r}: ",
            )

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
            # NaN for every unit means the rule compared nothing; warn.
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

    SI returns NaN (rarely inf) for low-spike units, which breaks NWB/JSON
    serialization. The input is not modified.
    """
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    non_finite = sanitized.isna()
    # ``where(..., other=None)`` keeps NaN; mask assignment on object dtype
    # writes a real None.
    sanitized = sanitized.astype(object)
    sanitized[non_finite] = None
    return sanitized


def isi_violation_fraction(isi_violations_count, num_spikes) -> np.ndarray:
    """Reproduce Spyglass's bounded ISI-violation fraction.

    ``count / (n_spikes - 1)`` (``v1/metric_utils``), not SI's unbounded
    ``isi_violations_ratio``.

    Parameters
    ----------
    isi_violations_count : array-like
        Per-unit violation counts on the retained spike trains.
    num_spikes : array-like
        Per-unit spike counts.

    Returns
    -------
    numpy.ndarray
        The per-unit fraction; NaN for units with <= 1 spike.
    """
    counts = np.asarray(isi_violations_count, dtype=float)
    n = np.asarray(num_spikes, dtype=float)
    denom = n - 1.0
    fraction = np.full(counts.shape, np.nan, dtype=float)
    valid = n > 1.0
    fraction[valid] = counts[valid] / denom[valid]
    return fraction


def isi_continuity_boundaries(analyzer, continuity_spans):
    """Join frames between ``continuity_spans``, or ``None`` without spans.

    Concat-backed sources use a synthetic clock; their continuity spans mark
    joins between unrelated chunks. Artifact exclusions do not split spans.
    """
    if continuity_spans is None:
        return None
    spans = np.asarray(continuity_spans)
    if spans.ndim != 2 or spans.shape[1] != 2 or not len(spans):
        raise ValueError("ISI continuity_spans must contain frame pairs.")
    spans = integer_vector(
        spans.reshape(-1), name="ISI continuity_spans", nonnegative=True
    ).reshape(-1, 2)
    if (
        analyzer.sorting.get_num_segments() != 1
        or spans[0, 0] != 0
        or spans[-1, 1] != analyzer.recording.get_num_samples()
        or np.any(spans[:, 1] <= spans[:, 0])
        or np.any(spans[1:, 0] != spans[:-1, 1])
    ):
        raise ValueError(
            "ISI continuity_spans must partition the single-segment recording."
        )
    return spans[:-1, 1]


def count_short_isis(times, frames, threshold_s, boundaries) -> int:
    """Intervals shorter than ``threshold_s`` in one time-sorted train.

    ``times`` are source-clock seconds of the spikes at ``frames``; a pair
    straddling a continuity join (``boundaries``) is not an interval.
    """
    violations = np.diff(times) < threshold_s
    if boundaries is not None:
        spans_for_spikes = np.searchsorted(boundaries, frames, side="right")
        violations &= spans_for_spikes[1:] == spans_for_spikes[:-1]
    return int(np.count_nonzero(violations))


def _source_clock_isi_counts(analyzer, unit_ids, threshold_s, continuity_spans):
    """Count short ISIs from spike timestamps without loading the full clock."""
    from spyglass.spikesorting.v2._core.signal_math import _segment_times_at

    sorting, recording = analyzer.sorting, analyzer.recording
    boundaries = isi_continuity_boundaries(analyzer, continuity_spans)
    counts = np.zeros(len(unit_ids), dtype=float)
    for index, unit_id in enumerate(unit_ids):
        for segment_index in range(sorting.get_num_segments()):
            frames = sorting.get_unit_spike_train(
                unit_id=unit_id, segment_index=segment_index
            )
            if len(frames) < 2:
                continue
            times = _segment_times_at(
                recording, frames, segment_index=segment_index
            )
            counts[index] += count_short_isis(
                times, frames, threshold_s, boundaries
            )
    return counts


# ---------- metric eligibility: which NaNs SpikeInterface leaves on purpose --
# SI also swallows failures to NaN, so each predicate reproduces when SI 0.104.3
# returns NaN without error. Paths are relative to ``spikeinterface``.


def _nn_advanced_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """``nn_isolation`` / ``nn_noise_overlap``: below the spike or rate floor.

    ``metrics/quality/pca_metrics.py:619-630,825-836``; Spyglass's patched
    noise overlap keeps both checks. A zero-spike unit is always missing.
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

    ``metrics/quality/misc_metrics.py:1670-1671`` is the only NaN path.
    """
    return (
        n_spikes / params["num_histogram_bins"]
        < params["amplitudes_bins_min_ratio"]
    )


def _isi_violation_missing(
    n_spikes: np.ndarray, *, total_samples: int, fs: float, params: dict
) -> np.ndarray:
    """Spyglass ``isi_violation`` fraction: undefined for <= 1 spike."""
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

    Non-finite ``snr`` means zero noise on the extremum channel, a degenerate
    recording.
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


# Keyed by output column. Unlisted columns (``isi_violations_*``, template,
# ``observed_*``, custom) fail closed on NaN.
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

    Mirrors ``core/analyzer_extension_core.py:1172-1178``. Defaults are read
    at call time because SI mutates them in place, and copied under
    ``SI_METRIC_STATE_LOCK``.
    """
    from spikeinterface.metrics.quality import (
        get_default_quality_metrics_params,
    )

    from spyglass.spikesorting.v2._core.si_metric_patches import (
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
        rates spikes over samples, not timestamps, even with gaps.
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


def _expected_missing_unit_ids(
    column: str, non_finite_ids: list, expected: set | None, *, prefix=""
) -> list:
    """Return the non-finite unit ids that are expected to be missing.

    Raises ``ValueError`` (prefixed by ``prefix``) if ``expected`` is ``None``
    (no eligibility rule) and any unit is non-finite, or if a non-finite unit
    is not in ``expected``.
    """
    if expected is None:
        if non_finite_ids:
            raise ValueError(
                prefix + _unregistered_column_message(column, non_finite_ids)
            )
        return []
    failed = [unit_id for unit_id in non_finite_ids if unit_id not in expected]
    if failed:
        raise ValueError(prefix + _computation_failure_message(column, failed))
    return [unit_id for unit_id in non_finite_ids if unit_id in expected]


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
        _expected_missing_unit_ids(column, non_finite, expected_missing[column])


# ---------- SpikeInterface metric-computation errors -------------------------
# SI warns "Error computing metric {name}: {e}" and NaN-fills that metric
# (``core/analyzer_extension_core.py:1270-1286``); ``name`` is not a column.
_SI_METRIC_ERROR = re.compile(
    r"Error computing metric (?P<metric>\S+?): (?P<error>.*)", re.DOTALL
)


def si_quality_metric_output_columns(si_metric: str) -> tuple[str, ...]:
    """Ordered output columns SpikeInterface fills for one quality metric.

    Read from the metric class's ``metric_columns``. Spyglass's
    ``isi_violation`` fraction leads that metric's columns. Unknown names map
    to ``()``.
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
    Other warnings are re-emitted unchanged. A metric error on columns no
    rule references is logged. All warnings are handled before raising.
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

    Records every warning in the block (``simplefilter("always")``) and
    passes them to ``report_si_metric_errors`` on exit. If the block raises,
    metric errors are only logged. Holds ``SI_METRIC_STATE_LOCK`` because
    ``catch_warnings`` swaps process-wide state; warnings from non-Spyglass
    threads are still recorded.

    Parameters
    ----------
    rule_columns : collection of str
        Metric columns referenced by auto-curation rules; empty escalates
        nothing.
    """
    from spyglass.spikesorting.v2._core.si_metric_patches import (
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


def rules_payloads_match(expected: dict, stored: dict) -> bool:
    """Return whether two normalized parameter payloads are equal.

    Numeric columns are ``double``, so stored floats compare exactly.
    """
    return _values_match(expected, stored)


def _values_match(a, b) -> bool:
    """Recursively compare JSON-native values exactly; bools stay distinct."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _values_match(a[key], b[key]) for key in a
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _values_match(x, y) for x, y in zip(a, b)
        )
    return a == b


# Added before metrics and auto-merge; ``principal_components`` only for PCA
# metrics.
_CURATION_EXTENSIONS = STANDARD_DISPLAY_ANALYZER_EXTENSIONS

# SI 0.104 defaults, pinned so an SI default change cannot alter PC/NN metrics.
_PCA_EXTENSION_PARAMS = {
    "n_components": 5,
    "mode": "by_channel_local",
    "whiten": True,
    "dtype": "float32",
}


def _pca_params_match(existing: dict) -> bool:
    """True if a stored ``principal_components`` matches the pinned params.

    Compares only the pinned keys; ``dtype`` is normalized via ``np.dtype``.
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
    # The ``knn`` step also needs spike_locations.
    "feature_neighbors": ("spike_locations",),
}

# Presets whose ``snr`` step reads the stored ``quality_metrics["snr"]``
# (``curation/auto_merge.py:255-261``).
_SNR_MERGE_PRESETS = frozenset({"feature_neighbors"})


def assert_merge_preset_metrics(auto_merge_preset, metric_names) -> None:
    """Raise if the auto-merge preset filters on SNR the evaluation lacks.

    Without ``snr`` the preset fails with a bare ``KeyError``; requiring it
    also makes the step use this evaluation's sorter-signed SNR.

    Raises
    ------
    ValueError
        If ``auto_merge_preset`` has an ``snr`` step and ``snr`` is not
        requested.
    """
    if auto_merge_preset in _SNR_MERGE_PRESETS and "snr" not in metric_names:
        raise ValueError(
            f"auto_merge_preset {auto_merge_preset!r} filters merge candidates "
            "by SNR, but the metric params do not compute 'snr'. Add 'snr' to "
            "the QualityMetricParameters metric_names, or choose another "
            "auto-merge preset."
        )


def _requested_pc_metrics(metric_names) -> list[str]:
    """PC/NN metrics among ``metric_names`` (route to the whitened analyzer).

    Uses the insert-time validator's set (``_available_pca_metric_names``) so
    routing and validation agree.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        _available_pca_metric_names,
    )

    pca = set(_available_pca_metric_names())
    return [name for name in metric_names if name in pca]


def evaluate_analyzers(
    display_analyzer,
    metric_analyzer,
    *,
    metric_inputs,
    expected_unit_ids,
    observation_metrics=None,
    statistics_spans,
    continuity_spans,
):
    """Compute metrics, labels, and merge suggestions; enforce the namespace.

    - The metric index must equal ``expected_unit_ids`` and every merge
      member must be in it, catching a stale or raw-sort analyzer.
    - A rule metric SI failed, or NaN for a unit meeting the metric's
      preconditions, raises ``ValueError``; other NaNs follow
      ``missing_policy``.
    - Holds ``SI_METRIC_STATE_LOCK`` (after any ``analyzer_cache_lock``), so
      evaluations on different threads run one after another.

    ``metric_inputs`` is an ``EvaluationMetricInputs``. ``statistics_spans``
    (``None``: whole recording) goes to :func:`compute_metrics`.
    """
    from spyglass.spikesorting.v2._core.si_metric_patches import (
        SI_METRIC_STATE_LOCK,
    )

    # SI metric defaults and warning capture are process-wide; the lock also
    # covers auto-merge, which can compute quality_metrics itself.
    from spyglass.utils.progress import report_progress

    report_progress("Computing quality metrics and proposed labels…")
    metric_kwargs = metric_inputs.metric_kwargs
    metric_job_kwargs = metric_inputs.metric_job_kwargs
    rule_rows = metric_inputs.rule_rows
    with SI_METRIC_STATE_LOCK:
        rule_columns = frozenset(row["metric_name"] for row in rule_rows)
        metrics_df = compute_metrics(
            display_analyzer,
            metric_analyzer,
            metric_inputs.metric_names,
            metric_kwargs,
            metric_inputs.skip_pc_metrics,
            metric_job_kwargs,
            template_metric_columns=metric_inputs.template_metric_columns,
            statistics_spans=statistics_spans,
            rule_columns=rule_columns,
            continuity_spans=continuity_spans,
            search_threads=metric_inputs.search_threads,
            peak_sign=metric_inputs.peak_sign,
            noise_seed=metric_inputs.noise_seed,
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
            metric_inputs.auto_merge_preset,
            metric_inputs.auto_merge_kwargs,
            metric_job_kwargs,
            peak_sign=metric_inputs.peak_sign,
        )
        assert_merge_membership(merge_groups, expected_unit_ids)
        return metrics_df, labels_by_unit, merge_groups


def spike_counts(display_analyzer, metric_analyzer) -> dict[int, int]:
    """Per-unit spike counts, checked equal on both analyzers.

    The eligibility classifier uses one set for voltage and PC/NN metrics.
    """
    counts = {
        integer_scalar(unit_id, name="unit_id"): integer_scalar(
            n, name="n_spikes", nonnegative=True
        )
        for unit_id, n in (
            display_analyzer.sorting.count_num_spikes_per_unit().items()
        )
    }
    if metric_analyzer is not None:
        metric_counts = {
            integer_scalar(unit_id, name="unit_id"): integer_scalar(
                n, name="n_spikes", nonnegative=True
            )
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


def _metric_unit_ids(index):
    identifiers = integer_vector(index.to_numpy(), name="metric unit_id")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(
            "CurationEvaluation metric unit_id index must be unique."
        )
    return identifiers


def assert_unit_namespace(metrics_df, expected_unit_ids) -> None:
    """Raise unless the metric index equals the curation's unit set."""
    computed_ids = _metric_unit_ids(metrics_df.index)
    expected_ids = integer_vector(
        list(expected_unit_ids), name="expected unit_id"
    )
    if len(np.unique(expected_ids)) != len(expected_ids):
        raise ValueError(
            "CurationEvaluation expected unit_id namespace must be unique."
        )
    computed = set(computed_ids.tolist())
    expected = set(expected_ids.tolist())
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
    expected = set(
        integer_vector(
            list(expected_unit_ids), name="expected unit_id"
        ).tolist()
    )
    for group in merge_groups:
        member_ids = integer_vector(list(group), name="merge member unit_id")
        if len(np.unique(member_ids)) != len(member_ids):
            raise ValueError(
                "CurationEvaluation merge member unit_id values must be unique within a group."
            )
        members = set(member_ids.tolist())
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
    rule_columns=frozenset(),
    search_threads=None,
    *,
    statistics_spans,
    continuity_spans,
    peak_sign,
    noise_seed,
):
    """Compute quality metrics, routing PC/NN metrics to the whitened one.

    - Voltage and spike-train metrics use the unwhitened display analyzer,
      since whitening normalizes per-channel variance. PC/NN metrics use the
      whitened metric analyzer. Frames join by unit id.
    - ``isi_violation`` counts short intervals on original spike timestamps;
      ``continuity_spans`` excludes pairs across concat joins.
    - ``template_metric_columns`` (SI column names) come from the display
      analyzer's ``template_metrics`` for cell typing, never thresholded.
    - ``statistics_spans`` (``Sorting.get_statistics_spans``) restrict
      ``nn_noise_overlap``'s noise cluster and ``sd_ratio``'s noise
      estimate; ``None`` uses the whole recording.
    - An SI metric failure on a ``rule_columns`` column raises
      ``ValueError``; others are logged.
    - ``peak_sign`` is the sort's polarity; ``noise_seed`` seeds
      ``sd_ratio``'s span noise; ``search_threads`` (``None``:
      ``resolve_search_threads()``) sets nn neighbor-search threads.
    """
    import numpy as np
    import pandas as pd
    from spikeinterface.metrics.quality import (
        compute_quality_metrics,
        get_default_quality_metrics_params,
    )

    from spyglass.spikesorting.v2._core.si_metric_patches import (
        isolated_si_metric_defaults,
        noise_cluster_spans,
    )
    from spyglass.spikesorting.v2._params.metric_curation import (
        required_extensions_for_metrics,
    )
    from spyglass.spikesorting.v2._sorting.analyzer import ensure_extensions

    metric_kwargs = metric_kwargs or {}
    pc_names = _requested_pc_metrics(metric_names)
    pc_set = set(pc_names)
    voltage_names = [m for m in metric_names if m not in pc_set]

    frames = []
    # Voltage / spike-train metrics -> unwhitened display analyzer.
    if voltage_names:
        # SI silently skips a metric whose extension is absent (``drift``
        # needs ``spike_locations``). PCA belongs to the metric analyzer.
        base_present = {*BASE_ANALYZER_EXTENSIONS, *_CURATION_EXTENSIONS}
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
                peak_sign=peak_sign,
                job_kwargs=job_kwargs,
            )
        if "sd_ratio" in voltage_names:
            # Cache span-only noise std (masked zeros would bias it low) and
            # patch sd_ratio's template-variance correction to the same spans.
            from spyglass.spikesorting.v2._core.si_metric_patches import (
                patch_sd_ratio_statistics_spans,
            )
            from spyglass.spikesorting.v2._sorting.dispatch import (
                cache_span_noise_levels,
            )

            cache_span_noise_levels(
                display_analyzer.recording,
                statistics_spans,
                return_in_uV=display_analyzer.return_in_uV,
                seed=noise_seed,
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
            if "isi_violation" in voltage_names:
                # Read under the lock so a concurrent compute cannot change
                # the refractory window.
                defaults = get_default_quality_metrics_params()["isi_violation"]
                isi_kwargs = metric_kwargs.get("isi_violation") or {}
                isi_threshold_ms = isi_kwargs.get(
                    "isi_threshold_ms", defaults["isi_threshold_ms"]
                )
            voltage_df = compute_quality_metrics(
                display_analyzer,
                metric_names=voltage_names,
                metric_params={
                    k: v for k, v in metric_kwargs.items() if k in voltage_names
                }
                or None,
                skip_pc_metrics=True,
                # The analyzer is shared across curations; drop stored
                # metrics so stale columns cannot leak in.
                delete_existing_metrics=True,
            )
        voltage_df.index = pd.Index(
            _metric_unit_ids(voltage_df.index), name=voltage_df.index.name
        )
        frames.append(voltage_df)

    # PC / cluster-separation metrics -> whitened metric analyzer.
    if pc_names and not skip_pc_metrics:
        if metric_analyzer is None:
            raise ValueError(
                "PC/NN metrics were requested but no whitened metric "
                "analyzer was provided -- make_compute must build it when "
                "PC metrics are requested."
            )
        # nn_noise_overlap needs a 'median' template, and SI 0.104.3 breaks
        # on sparse analyzers; install the sparse fix and k-d tree search.
        from spyglass.spikesorting.v2._core.job_config import (
            resolve_search_threads,
        )
        from spyglass.spikesorting.v2._core.si_metric_patches import (
            isolation_workers,
            patch_isolation_kdtree,
            patch_nn_noise_overlap_sparsity,
        )

        ensure_extensions(
            metric_analyzer,
            ["templates"],
            peak_sign=peak_sign,
            job_kwargs=job_kwargs,
        )
        templates_operators = list(
            metric_analyzer.get_extension("templates").params.get("operators")
            or []
        )
        if "median" not in templates_operators:
            metric_analyzer.compute(
                "templates", operators=[*templates_operators, "median"]
            )
        patch_nn_noise_overlap_sparsity()
        patch_isolation_kdtree()
        # ensure_extensions skips a present extension without checking its
        # params, so drop a mismatched PCA first.
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
            peak_sign=peak_sign,
            job_kwargs=job_kwargs,
            extension_params={"principal_components": _PCA_EXTENSION_PARAMS},
        )
        with (
            noise_cluster_spans(statistics_spans),
            isolation_workers(
                resolve_search_threads()
                if search_threads is None
                else search_threads
            ),
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
                delete_existing_metrics=True,
                # Spawned SI workers would not see the main-process patches
                # or the noise_cluster_spans ContextVar.
                n_jobs=1,
            )
        pc_df.index = pd.Index(
            _metric_unit_ids(pc_df.index), name=pc_df.index.name
        )
        frames.append(pc_df)

    if not frames:
        raise ValueError(
            "compute_metrics: no metrics to compute -- metric_names "
            f"{sorted(metric_names)} contains only PC/NN metrics but "
            "skip_pc_metrics=True. Set skip_pc_metrics=False to compute "
            "them, or include a voltage-based metric."
        )
    if len(frames) > 1:
        # An outer concat would NaN-fill a mismatched unit, so require equal
        # unit sets, then align order.
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
        counts = metrics_df["isi_violations_count"].to_numpy(
            dtype=float, copy=True
        )
        # Keep SI failures NaN for the rule-error checks.
        computed = np.isfinite(counts)
        if computed.any():
            counts[computed] = _source_clock_isi_counts(
                display_analyzer,
                metrics_df.index[computed],
                float(isi_threshold_ms) / 1000.0,
                continuity_spans,
            )
        n_by_unit = display_analyzer.sorting.count_num_spikes_per_unit()
        n_spikes = np.array(
            [n_by_unit[int(u)] for u in metrics_df.index], dtype=float
        )
        metrics_df["isi_violation"] = isi_violation_fraction(counts, n_spikes)

    # A PC-only row skips the voltage branch, which would drop shape columns.
    if template_metric_columns and not display_analyzer.has_extension(
        "template_metrics"
    ):
        with escalate_si_metric_errors(rule_columns):
            ensure_extensions(
                display_analyzer,
                ["template_metrics"],
                peak_sign=peak_sign,
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

    Reads ``template_metrics`` from the unwhitened display analyzer and
    joins by unit id. No-op when no columns are configured or the extension
    is absent.
    """
    if not template_metric_columns:
        return metrics_df
    if not display_analyzer.has_extension("template_metrics"):
        return metrics_df
    tm_df = display_analyzer.get_extension("template_metrics").get_data()
    # Never shadow a same-named quality-metric column.
    present = [
        c
        for c in template_metric_columns
        if c in tm_df.columns and c not in metrics_df.columns
    ]
    missing = [c for c in template_metric_columns if c not in tm_df.columns]
    if missing:
        # Validation guarantees these columns, so a miss means SI drift.
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
    selected.index = pd.Index(
        _metric_unit_ids(selected.index), name=selected.index.name
    )
    return metrics_df.join(selected)


def compute_merge_groups(
    analyzer,
    auto_merge_preset,
    auto_merge_kwargs,
    job_kwargs=None,
    *,
    peak_sign,
):
    """Return proposed merge groups for a preset (``[]`` for 'none').

    ``peak_sign`` is the sort's polarity for the amplitude and location
    extensions the preset needs.
    """
    if auto_merge_preset == "none":
        return []
    from spikeinterface.curation import compute_merge_unit_groups

    from spyglass.spikesorting.v2._sorting.analyzer import ensure_extensions
    from spyglass.spikesorting.v2._sorting.dispatch import without_random_seed

    extensions = list(_CURATION_EXTENSIONS)
    extensions.extend(_AUTO_MERGE_EXTRA_EXTENSIONS.get(auto_merge_preset, ()))
    ensure_extensions(
        analyzer, extensions, peak_sign=peak_sign, job_kwargs=job_kwargs
    )
    compute_kwargs = dict(auto_merge_kwargs or {})
    compute_kwargs.update(without_random_seed(job_kwargs))
    groups = compute_merge_unit_groups(
        analyzer,
        preset=auto_merge_preset,
        compute_needed_extensions=False,
        **compute_kwargs,
    )
    return [
        integer_vector(list(group), name="merge member unit_id").tolist()
        for group in groups
    ]
