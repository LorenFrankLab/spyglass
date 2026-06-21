"""v1 MetricCuration <-> v2 AnalyzerCuration parity.

The pure tolerance comparator (``compare_to_v1_baseline``) is unit-tested with
synthetic data so the parity LOGIC has real coverage. The integration test
loads the v1 ``MetricCuration`` baseline pickle captured under the legacy SI
0.99 runtime and compares the v2 metrics against it; it skips with an explicit
message when that real-data baseline is absent (no
``SPIKESORTING_V2_REAL_NWB_PATH`` capture has been run).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
import pytest

from spyglass.spikesorting.v2._metric_parity import compare_to_v1_baseline

_BASELINE = (
    Path(__file__).resolve().parent / "baselines" / "baseline_metric_curation.pkl"
)


# ---------- DB-free comparator logic ----------------------------------------


def _frame(snr, isi, fr, n):
    idx = pd.Index(range(len(snr)), name="unit_id")
    return pd.DataFrame(
        {"snr": snr, "isi_violation": isi, "firing_rate": fr, "num_spikes": n},
        index=idx,
    )


def test_parity_comparator_passes_and_fails():
    v1 = _frame([4.0, 6.0, 8.0], [0.001, 0.0, 0.01], [1.0, 2.0, 3.0], [100, 200, 300])
    v2_ok = _frame(
        [4.4, 6.6, 8.8], [0.001, 0.0, 0.01], [1.0, 2.0, 3.0], [100, 200, 300]
    )
    ok = compare_to_v1_baseline(v2_ok, v1)
    assert ok.matched, ok.failures
    assert abs(ok.snr_median_ratio - 1.1) < 1e-6

    # isi_violation off, num_spikes off, and an SNR order-of-magnitude outlier.
    v2_bad = _frame(
        [4.0, 6.0, 800.0], [0.5, 0.0, 0.01], [1.0, 2.0, 3.0], [101, 200, 300]
    )
    bad = compare_to_v1_baseline(v2_bad, v1)
    assert not bad.matched
    assert any("isi_violation" in f for f in bad.failures)
    assert any("num_spikes" in f for f in bad.failures)
    assert any("snr" in f for f in bad.failures)


def test_parity_comparator_no_common_units():
    a = _frame([4.0], [0.0], [1.0], [100])
    b = _frame([4.0], [0.0], [1.0], [100])
    b.index = pd.Index([99], name="unit_id")
    report = compare_to_v1_baseline(b, a)
    assert not report.matched
    assert any("no common unit" in f for f in report.failures)


# ---------- integration (gated on a captured real-data baseline) ------------


@pytest.mark.slow
@pytest.mark.integration
def test_v2_analyzer_curation_vs_v1():
    """Compare v2 AnalyzerCuration metrics to the captured v1 baseline.

    Skips unless the v1 ``MetricCuration`` baseline pickle exists -- produced
    by ``tests/spikesorting/v2/baseline_capture.py`` under the SI 0.99 runtime
    with ``SPIKESORTING_V2_REAL_NWB_PATH`` set.
    """
    if not _BASELINE.exists():
        pytest.skip(
            "v1 MetricCuration baseline not captured. Run baseline_capture.py "
            "under the SI 0.99 runtime with SPIKESORTING_V2_REAL_NWB_PATH set "
            f"to produce {_BASELINE.name}."
        )
    if not os.environ.get("SPIKESORTING_V2_REAL_NWB_PATH"):
        pytest.skip(
            "SPIKESORTING_V2_REAL_NWB_PATH unset; cannot ingest the matching "
            "real sort for the v1<->v2 parity comparison."
        )

    with open(_BASELINE, "rb") as handle:
        baseline = pickle.load(handle)
    v1_metrics = (
        baseline["metrics"]
        if isinstance(baseline, dict) and "metrics" in baseline
        else baseline
    )
    v1_metrics = (
        v1_metrics
        if isinstance(v1_metrics, pd.DataFrame)
        else pd.DataFrame(v1_metrics)
    )

    # Populate the v2 AnalyzerCuration on the matching real sort and fetch its
    # metrics. The capture must record the sort identity (an "analyzer_curation"
    # selection key) alongside the metrics for the replay to find the sort.
    if not (isinstance(baseline, dict) and baseline.get("analyzer_curation_key")):
        pytest.skip(
            "Baseline pickle lacks the 'analyzer_curation_key' replay metadata "
            "(baseline_capture.py must be extended to capture v1 MetricCuration "
            "metrics plus the matching v2 selection key)."
        )
    from spyglass.spikesorting.v2.metric_curation import AnalyzerCuration

    sel = baseline["analyzer_curation_key"]
    AnalyzerCuration.populate(sel, reserve_jobs=False)
    v2_metrics = AnalyzerCuration.get_metrics(sel)
    report = compare_to_v1_baseline(v2_metrics, v1_metrics)
    assert report.matched, "v1<->v2 parity failed:\n" + "\n".join(report.failures)
