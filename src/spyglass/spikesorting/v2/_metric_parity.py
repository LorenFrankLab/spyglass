"""DB-free comparison of v2 quality metrics against a v1 baseline.

The v1↔v2 parity test loads a v1 ``MetricCuration`` baseline (captured under
the legacy SI 0.99 runtime) and checks the v2 ``AnalyzerCuration`` metrics
against it per documented tolerances. The comparison itself is pure (two
DataFrames in, a structured report out) so the tolerance logic is unit-testable
without a database or a real-data baseline.

Tolerances:

- ``num_spikes``: exact integer match (deterministic given the sorted spikes).
- ``isi_violation``: exact within float tolerance -- v2 replicates Spyglass's
  ``count / (n_spikes - 1)`` fraction, NOT SI's ``isi_violations_ratio``.
- ``firing_rate``: exact within float rounding (spike count / duration).
- ``snr``: distributional, NOT per-unit -- v1 (mean-based) and SI 0.104
  (median-based) use different SNR definitions, so the check asserts the median
  v2/v1 ratio is within ``snr_median_ratio_tol`` and no unit is an
  order-of-magnitude outlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParityReport:
    """Result of comparing v2 metrics to a v1 baseline."""

    matched: bool
    failures: list[str] = field(default_factory=list)
    snr_median_ratio: float | None = None
    n_common_units: int = 0


def _finite_pairs(v1_series, v2_series, unit_ids):
    pairs = []
    for unit_id in unit_ids:
        a = float(v1_series.loc[unit_id])
        b = float(v2_series.loc[unit_id])
        if np.isfinite(a) and np.isfinite(b):
            pairs.append((unit_id, a, b))
    return pairs


def compare_to_v1_baseline(
    v2_metrics,
    v1_metrics,
    *,
    exact_rtol: float = 1e-3,
    exact_atol: float = 1e-6,
    snr_median_ratio_tol: float = 0.30,
    snr_outlier_factor: float = 10.0,
) -> ParityReport:
    """Compare v2 ``AnalyzerCuration`` metrics to a v1 baseline.

    Parameters
    ----------
    v2_metrics, v1_metrics : pandas.DataFrame
        Indexed by ``unit_id``; columns include ``snr``, ``isi_violation``,
        ``firing_rate``, ``num_spikes``.
    exact_rtol, exact_atol : float
        Tolerances for the "exact" metrics (isi_violation, firing_rate).
    snr_median_ratio_tol : float
        Allowed deviation of the median v2/v1 SNR ratio from 1.0.
    snr_outlier_factor : float
        A unit whose v2/v1 SNR ratio exceeds this factor (or is below its
        reciprocal) is an unexplained order-of-magnitude outlier.

    Returns
    -------
    ParityReport
        ``matched`` is False if any check fails; ``failures`` carries a
        per-metric / per-unit diff report.
    """
    common = sorted(set(v2_metrics.index) & set(v1_metrics.index))
    failures: list[str] = []

    if not common:
        return ParityReport(
            matched=False,
            failures=["no common unit ids between v1 baseline and v2 metrics"],
        )

    # num_spikes: exact integer.
    if "num_spikes" in v1_metrics and "num_spikes" in v2_metrics:
        for unit_id in common:
            a = int(v1_metrics.loc[unit_id, "num_spikes"])
            b = int(v2_metrics.loc[unit_id, "num_spikes"])
            if a != b:
                failures.append(f"num_spikes[{unit_id}]: v1={a} v2={b}")

    # isi_violation + firing_rate: exact within float tolerance.
    for column in ("isi_violation", "firing_rate"):
        if column not in v1_metrics or column not in v2_metrics:
            continue
        for unit_id, a, b in _finite_pairs(
            v1_metrics[column], v2_metrics[column], common
        ):
            if not np.isclose(a, b, rtol=exact_rtol, atol=exact_atol):
                failures.append(f"{column}[{unit_id}]: v1={a:.6g} v2={b:.6g}")

    # snr: distributional ratio check.
    snr_median_ratio = None
    if "snr" in v1_metrics and "snr" in v2_metrics:
        ratios = []
        for unit_id, a, b in _finite_pairs(
            v1_metrics["snr"], v2_metrics["snr"], common
        ):
            if a == 0:
                continue
            ratio = b / a
            ratios.append(ratio)
            if ratio > snr_outlier_factor or ratio < 1.0 / snr_outlier_factor:
                failures.append(
                    f"snr[{unit_id}]: order-of-magnitude ratio v2/v1={ratio:.3g}"
                )
        if ratios:
            snr_median_ratio = float(np.median(ratios))
            if abs(snr_median_ratio - 1.0) > snr_median_ratio_tol:
                failures.append(
                    f"snr median v2/v1 ratio {snr_median_ratio:.3g} exceeds "
                    f"+-{snr_median_ratio_tol:.0%}"
                )

    return ParityReport(
        matched=not failures,
        failures=failures,
        snr_median_ratio=snr_median_ratio,
        n_common_units=len(common),
    )
