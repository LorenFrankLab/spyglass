"""DB-free transform helpers for analyzer-driven quality-metric curation.

Pure logic shared by ``CurationEvaluation`` (the ``@schema`` table lives in
``metric_curation.py``): turning quality-metric columns into per-unit labels,
sanitizing non-finite metric values before serialization, and reproducing
Spyglass's ``isi_violation`` fraction. Keeping these here -- importable with
only NumPy / pandas, no DataJoint connection and no SpikeInterface analyzer --
lets them be unit-tested without a database.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

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
    """Return whether one metric scalar can participate in thresholding."""
    if pd.isna(value):
        return False
    try:
        return bool(np.isfinite(value))
    except TypeError:
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
        applied in ascending ``rule_index``.

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
        If a rule references a metric column absent from ``metrics_df`` -- a
        clear error before serialization rather than a silent miss.

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
        for unit_id in metrics_df.index:
            value = column.loc[unit_id]
            if not _is_finite_metric_value(value):
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


def isi_violation_fraction(
    isi_violations_count, num_spikes
) -> np.ndarray:
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


def rules_payloads_match(
    expected: dict, stored: dict, *, rel_tol: float = 1e-6, abs_tol: float = 1e-12
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
