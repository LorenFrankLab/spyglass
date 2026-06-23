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


def test_parity_comparator_flags_missing_and_extra_units():
    """A run that drops/adds units fails even if the overlap matches exactly."""
    v1 = _frame([4.0, 6.0], [0.0, 0.0], [1.0, 2.0], [100, 200])
    # v2 keeps unit 0 (identical), drops unit 1, and adds a new unit 2.
    v2 = _frame([4.0, 9.0], [0.0, 0.0], [1.0, 9.0], [100, 900])
    v2.index = pd.Index([0, 2], name="unit_id")
    report = compare_to_v1_baseline(v2, v1)
    assert not report.matched
    assert any("missing from v2: [1]" in f for f in report.failures)
    assert any("not in v1 baseline: [2]" in f for f in report.failures)


def test_parity_comparator_no_common_units():
    a = _frame([4.0], [0.0], [1.0], [100])
    b = _frame([4.0], [0.0], [1.0], [100])
    b.index = pd.Index([99], name="unit_id")
    report = compare_to_v1_baseline(b, a)
    assert not report.matched
    assert any("no common unit" in f for f in report.failures)


def test_parity_comparator_handles_sanitized_none():
    """A sanitized None (non-finite metric) is skipped, not a crash.

    ``get_metrics`` returns None for non-finite metric values, which is exactly
    what feeds the integration comparison; ``float(None)`` would otherwise raise
    and abort the whole parity check.
    """
    v1 = _frame([4.0, 6.0], [0.001, 0.002], [1.0, 2.0], [100, 200])
    v2 = _frame([4.0, 6.0], [0.001, 0.002], [1.0, 2.0], [100, 200])
    # Unit 1's v2 isi_violation came back non-finite -> sanitized to None.
    v2 = v2.astype(object)
    v2.loc[1, "isi_violation"] = None
    report = compare_to_v1_baseline(v2, v1)  # must not raise
    assert report.matched, report.failures
    # Unit 0's isi pair is still compared; unit 1's None pair is dropped.
    assert report.n_metrics_compared > 0


def test_parity_comparator_fails_on_missing_expected_column():
    """A metric column absent from one side fails, never a silent match."""
    v1 = _frame([4.0], [0.0], [1.0], [100])
    v2 = _frame([4.0], [0.0], [1.0], [100]).drop(columns=["snr"])
    report = compare_to_v1_baseline(v2, v1)
    assert not report.matched
    assert any("snr" in f and "missing" in f for f in report.failures)


def test_parity_comparator_fails_when_metric_all_non_finite():
    """A present column with no finite values to compare is a failure."""
    v1 = _frame([4.0, 6.0], [0.0, 0.0], [1.0, 2.0], [100, 200])
    v2 = _frame([4.0, 6.0], [0.0, 0.0], [1.0, 2.0], [100, 200]).astype(object)
    v2["snr"] = None  # entire snr column sanitized away
    report = compare_to_v1_baseline(v2, v1)
    assert not report.matched
    assert any("snr present but no finite" in f for f in report.failures)


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
    if not (isinstance(baseline, dict) and "sort_identity" in baseline):
        pytest.skip(
            "Baseline pickle predates the metrics+sort_identity format; "
            "re-run baseline_capture.py to regenerate it."
        )
    v1_metrics = baseline["metrics"]
    if not isinstance(v1_metrics, pd.DataFrame):
        v1_metrics = pd.DataFrame(v1_metrics)
    identity = baseline["sort_identity"]

    # Replay under SI 0.104: re-ingest the real NWB, run the v2 pipeline on the
    # captured sort group, then populate AnalyzerCuration and compare. Note the
    # v1 (SI 0.99) and v2 (SI 0.104) sorts must use a comparable sorter for the
    # exact num_spikes/isi_violation parity to hold; if they diverge across SI
    # versions the comparator's tolerances are the calibration point (per the
    # plan's "tighten/relax only with recorded baseline evidence").
    from pathlib import Path

    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_run import run_v2_pipeline
    from spyglass.spikesorting.v2.metric_curation import (
        AnalyzerCuration,
        AnalyzerCurationSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )

    nwb_file_name = copy_and_insert_nwb(
        Path(os.environ["SPIKESORTING_V2_REAL_NWB_PATH"])
    )
    configure_v2_run_inputs(
        nwb_file_name,
        identity["team_name"],
        interval_list_name=identity["interval_list_name"],
    )
    initialize_v2_defaults()
    summary = run_v2_pipeline(
        nwb_file_name,
        identity["sort_group_id"],
        identity["interval_list_name"],
        identity["team_name"],
    )
    sel = AnalyzerCurationSelection.insert_selection(
        {
            "sorting_id": summary["sorting_id"],
            "curation_id": summary["curation_id"],
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "none",
        }
    )
    AnalyzerCuration.populate(sel, reserve_jobs=False)
    v2_metrics = AnalyzerCuration.get_metrics(sel)
    report = compare_to_v1_baseline(v2_metrics, v1_metrics)
    assert report.matched, "v1<->v2 parity failed:\n" + "\n".join(report.failures)
