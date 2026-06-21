"""Ground-truth accuracy and v1-parity tests on real MEArec fixtures for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb


# ---------- 60s MEArec ground-truth correctness gate ----------------------


@pytest.mark.slow
def test_mountainsort5_ground_truth_polymer_60s(polymer_60s_session):
    """MS5 on the 60s polymer fixture finds the planted units.

    Primary correctness gate: per-unit accuracy >= 0.7 for at least
    half of planted units, computed via
    ``spikeinterface.comparison.compare_sorter_to_ground_truth``
    against the sidecar ground-truth units table written by
    ``mearec_to_spyglass_nwb`` and read via
    ``get_ground_truth_units_table``.
    """
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = polymer_60s_session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )

    if not (SortGroupV2 & polymer_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Sort each of the 4 shanks; aggregate the per-shank SI sortings
    # by shifting unit ids so MS5's overlapping local ids do not
    # collide across shanks.
    sort_group_ids = sorted(
        int(g)
        for g in (SortGroupV2 & polymer_60s_session).fetch("sort_group_id")
    )
    sortings_by_shank = []
    for sg_id in sort_group_ids:
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": (
                    "franklab_30khz_ms5_2026_06"
                ),
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    # Combine the per-shank sortings into one BaseSorting for the
    # ground-truth comparison. SI's ``aggregate_units`` produces a
    # single sorting with shank-disjoint unit ids -- but the result's
    # unit_ids dtype is ``uint64``, which SI 0.104's
    # ``compare_sorter_to_ground_truth`` Hungarian matcher rejects
    # (it only accepts dtype kinds 'i' or 'U'). Rename the units to
    # contiguous signed-int ids to round-trip through the matcher.
    import numpy as np
    from spikeinterface import aggregate_units

    aggregated = aggregate_units(sortings_by_shank)
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    # Ground truth: the planted MEArec units stored in the sidecar
    # ``ProcessingModule("ground_truth")["units"]`` table. We don't
    # use ``ImportedSpikeSorting`` (its merge-table dispatch raises
    # NotImplementedError for v0/v1/imported sources); we read the
    # NWB directly via the sidecar helper and build a NumpySorting.
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        # MEArec writes a fixed-rate ElectricalSeries (no timestamps
        # array). Read ``rate`` directly; fall back to the timestamps
        # diff only if rate is missing.
        if es.rate is not None:
            fs = float(es.rate)
        else:
            fs = float(1.0 / np.diff(es.timestamps[:2])[0])
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {}
        for idx, unit_id in enumerate(gt_units_table.id[:]):
            gt_units[int(unit_id)] = np.asarray(
                gt_units_table["spike_times"][idx]
            )
    gt_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[
            {
                int(uid): (times * fs).astype(np.int64)
                for uid, times in gt_units.items()
            }
        ],
        sampling_frequency=fs,
    )

    comparison = compare_sorter_to_ground_truth(
        gt_sorting=gt_sorting,
        tested_sorting=tested_sorting,
        gt_name="mearec_polymer_60s_planted",
        tested_name="v2_mountainsort5",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    # Per-unit performance is a DataFrame indexed by GT unit_id.
    # SI 0.104's ``get_performance(method='by_unit')`` exposes:
    #   accuracy  = tp / (tp + fn + fp)
    #   recall    = tp / (tp + fn)
    #   precision = tp / (tp + fp)
    #   false_discovery_rate
    #   miss_rate
    # Each is keyed by GT unit_id (Hungarian-matched to a detected unit).
    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    # ---- Diagnostic summary so a failure (or a soft regression) surfaces
    # the distribution rather than a single pass/fail bit. Always logged.
    summary = (
        f"\n[MS5 vs 60s polymer GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f} "
        f"min={recall.min():.3f} max={recall.max():.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f} "
        f"min={precision.min():.3f} max={precision.max():.3f}\n"
        f"  per-unit accuracy (sorted): "
        f"{[round(float(a), 3) for a in sorted(accuracies, reverse=True)]}"
    )
    print(summary)

    # Guard against a vacuous pass: ``n_planted // 2 == 0 >= 0`` would
    # silently approve a fixture that produced zero ground-truth units
    # (e.g., MEArec generation half-failed). The polymer 60s fixture is
    # built with 24 planted units; require well over half so a half-run
    # generation surfaces.
    assert n_planted >= 12, (
        f"60s polymer fixture has only {n_planted} planted units; expected "
        f"around 24. Regenerate the fixture before trusting this gate."
        f"{summary}"
    )

    # Detection-count floor: at least half the planted units must have
    # accuracy >= 0.7. This is the primary correctness gate.
    n_well_detected = int((accuracies >= 0.7).sum())
    threshold_half = n_planted // 2
    assert n_well_detected >= threshold_half, (
        f"MS5 on the 60s polymer fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"validation goal requires >= {threshold_half}."
        f"{summary}"
    )

    # Precision floor: the well-detected units must also have
    # precision >= 0.5. Otherwise a sorter that emits many spurious
    # spikes alongside each true unit (e.g., a buggy duplicate-spike
    # bug) could clear the accuracy gate by coincidence.
    well_detected_mask = accuracies >= 0.7
    well_detected_precision = precision[well_detected_mask]
    assert (well_detected_precision >= 0.5).all(), (
        f"Well-detected GT units have low precision: min="
        f"{well_detected_precision.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7.{summary}"
    )

    # Recall floor on well-detected units. accuracy is symmetric in fn
    # and fp; a unit can have accuracy=0.7 with recall=0.7/precision=1.0
    # (missing real spikes) or recall=1.0/precision=0.7 (spurious extra
    # spikes). The precision floor above bounds the second mode; this
    # bounds the first so a well-detected unit must actually find most
    # of its planted spikes.
    well_detected_recall = recall[well_detected_mask]
    assert (well_detected_recall >= 0.5).all(), (
        f"Well-detected GT units have low recall: min="
        f"{well_detected_recall.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7 (a unit can't be 'well detected' "
        f"if it's missing more than half its planted spikes).{summary}"
    )

    # Distribution-shape floor: well-detected units should have HIGH
    # accuracy, not barely-passing accuracy. A pipeline regression that
    # drives every accuracy from ~0.95 down to ~0.71 should fail this
    # test even though the count gate above still passes.
    well_detected_accuracy = accuracies[well_detected_mask]
    assert well_detected_accuracy.mean() >= 0.85, (
        f"Mean accuracy of well-detected units is "
        f"{well_detected_accuracy.mean():.3f}; expected >= 0.85 "
        f"(MS5 typically produces ~0.95 on this fixture). A regression "
        f"that drove accuracies down while keeping enough above 0.7 to "
        f"pass the count gate would surface here.{summary}"
    )


# =========================================================================
# Real-data correctness gates (scaffolded; skip when fixtures absent).
# These activate when the user generates the fixture or sets the gating
# env var; the scaffolding documents the contract and guarantees the test
# surface exists for future CI gating.
# =========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_mountainsort5_ground_truth_neuropixels_60s(neuropixels_60s_session):
    """MS5 on the 60s Neuropixels fixture finds the planted units.

    Dense-probe informational correctness coverage. Mirrors
    ``test_mountainsort5_ground_truth_polymer_60s`` but with the
    informational dense-probe correctness coverage. Primary count
    gate is the plan-specified ``accuracy >= 0.7`` for at least
    three-quarters of planted units, plus precision / recall /
    mean-accuracy floors on the well-detected subset (matching
    the polymer gate's secondary floors). The threshold may only
    be loosened with committed calibration evidence in
    ``fixtures_manifest.json`` showing the MS5 distribution this
    dense-probe fixture actually produces; until then a failure
    here is a signal that MS5 settings need tuning, NOT that the
    threshold should be relaxed silently. The upstream
    ``neuropixels_60s_session`` fixture skips cleanly when the
    MEArec NWB is not on disk, so this test runs only when the
    fixture has been generated.
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = neuropixels_60s_session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )

    if not (SortGroupV2 & neuropixels_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g)
        for g in (SortGroupV2 & neuropixels_60s_session).fetch("sort_group_id")
    )

    sortings_by_shank = []
    for sg_id in sort_group_ids:
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": (
                    "franklab_30khz_ms5_2026_06"
                ),
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    aggregated = aggregate_units(sortings_by_shank)
    # uint64 unit_ids reject the Hungarian matcher's accepted dtypes;
    # rename to contiguous signed ints. Same trick
    # ``test_mountainsort5_ground_truth_polymer_60s`` uses.
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        if es.rate is not None:
            fs = float(es.rate)
        else:
            diff_s = float(np.diff(es.timestamps[:2])[0])
            assert diff_s > 0, (
                f"NWB e-series timestamps near t=0 are not "
                f"monotonic (diff={diff_s}); cannot derive fs."
            )
            fs = 1.0 / diff_s
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }
    gt_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[
            {
                int(uid): (times * fs).astype(np.int64)
                for uid, times in gt_units.items()
            }
        ],
        sampling_frequency=fs,
    )

    comparison = compare_sorter_to_ground_truth(
        gt_sorting=gt_sorting,
        tested_sorting=tested_sorting,
        gt_name="mearec_neuropixels_60s_planted",
        tested_name="v2_mountainsort5",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    summary = (
        f"\n[MS5 vs 60s Neuropixels GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f}"
    )
    print(summary)

    # Non-vacuous-pass guard: the fixture must actually have
    # planted units. A half-failed MEArec generation would
    # silently produce zero GT units.
    assert n_planted >= 1, (
        f"Neuropixels 60s fixture has only {n_planted} planted units; "
        f"regenerate the fixture before trusting this gate.{summary}"
    )

    # Plan-required informational threshold: at least three
    # quarters of planted units detected at accuracy >= 0.7. The
    # threshold may only be loosened with calibration evidence
    # committed to ``fixtures_manifest.json`` showing the actual
    # MS5 distribution this fixture produces; a failure here is a
    # signal that MS5 settings need tuning, not a license to relax
    # the gate.
    import math

    n_well_detected = int((accuracies >= 0.7).sum())
    threshold = max(1, math.ceil(n_planted * 0.75))
    assert n_well_detected >= threshold, (
        f"MS5 on the 60s Neuropixels fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"plan threshold requires >= {threshold} (3/4 of planted, "
        f"ceil-rounded so non-multiples of 4 don't quietly pass below 75 %)."
        f"{summary}"
    )

    # Secondary floors on the well-detected subset. Mirror the
    # polymer gate so a sorter that explodes the false-positive
    # rate, misses most planted spikes, or produces barely-passing
    # accuracies cannot clear the count gate by coincidence.
    well_detected_mask = accuracies >= 0.7
    well_detected_precision = precision[well_detected_mask]
    well_detected_recall = recall[well_detected_mask]
    well_detected_accuracy = accuracies[well_detected_mask]
    assert (well_detected_precision >= 0.5).all(), (
        f"Well-detected GT units have low precision: min="
        f"{well_detected_precision.min():.3f}; expected >= 0.5 for "
        f"every unit with accuracy >= 0.7.{summary}"
    )
    assert (well_detected_recall >= 0.5).all(), (
        f"Well-detected GT units have low recall: min="
        f"{well_detected_recall.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7.{summary}"
    )
    assert well_detected_accuracy.mean() >= 0.8, (
        f"Mean accuracy of well-detected units is "
        f"{well_detected_accuracy.mean():.3f}; expected >= 0.8 (the "
        f"informational threshold the docstring claims).{summary}"
    )


_PARITY_FIXTURE_CASES = [
    ("mearec_polymer_smoke", 0),
    ("mearec_polymer_smoke", 1),
    ("mearec_polymer_smoke", 2),
    ("mearec_polymer_smoke", 3),
    ("mearec_polymer_128ch_60s", 0),
    ("mearec_polymer_128ch_60s", 1),
    ("mearec_polymer_128ch_60s", 2),
    ("mearec_polymer_128ch_60s", 3),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_stem,sort_group_id",
    _PARITY_FIXTURE_CASES,
    ids=[f"{stem}-shank{sg}" for stem, sg in _PARITY_FIXTURE_CASES],
)
def test_v2_real_data_v1_parity(fixture_stem, sort_group_id, dj_conn):
    """v1 ↔ v2 ``clusterless_thresholder`` parity on the polymer matrix.

    **NOT a per-PR CI gate.** This is a manual / nightly baseline-verification
    matrix: all parametrized cases SKIP unless ``SPIKESORTING_V2_BASELINE_ROOT``
    (or the legacy single-case ``SPIKESORTING_V2_BASELINE_DIR``) points at
    operator-captured v1 baselines. The ``pytest-v2`` workflow does not set
    either, so reviewers should not count these cases as per-PR coverage --
    they activate only under the capture workflow described in "Env vars" below.

    **Referencing scope.** This v1↔v2 spike-parity contract holds only for
    *order-invariant* referencing -- the parity fixtures use the inherited
    default, which is ``reference_mode="none"`` here, and ``specific`` (and a
    global *average* reference) are also linear and commute with the filter.
    It does NOT hold for ``reference_mode="global_median"``: v2 bandpass-
    filters BEFORE referencing while v1 referenced first, and the per-sample
    median is non-linear, so global-median output intentionally diverges from
    v1. Do not add a global-median case to this matrix expecting a match; that
    divergence is guarded instead by
    ``test_recording_make_global_median_reference`` (Pin 3) and documented in
    the migration guide. (``baseline_capture.py --reference-mode global_median``
    can capture a v1 reference-first global-median baseline for manual
    divergence inspection.)

    The deterministic threshold sorter SHOULD detect the same peaks
    under SI 0.104 as it did under SI 0.99. The test enforces a
    ±1.5-sample asymmetric per-spike contract: every v1 spike must
    match a v2 spike within 1.5 samples (``unmatched_v1 > 0`` FAILs);
    v2 may have bounded extras (50 % + 5 detections budget; see the
    in-test comment near ``extra_spike_ratio = 0.50`` for the SI
    PR #4341 calibration evidence) attributed to documented cross-
    SI-version ``detect_peaks`` drift. Both ``unmatched_v1`` and
    ``unmatched_v2`` are reported per case.

    **Invariant fingerprint contract:** before any spike-time
    comparison runs, the v2 side reconstructs its own state and
    asserts each fingerprint (``nwb_sha256``, ``sort_group_electrode_ids``,
    ``bad_channel_by_electrode_id``, ``canonical_preproc_params``,
    ``canonical_artifact_params``, ``artifact_valid_times``,
    ``canonical_sorter_params``) matches the v1 baseline. A
    fingerprint mismatch FAILs with a path-tagged diff so an
    input-layer skew never gets conflated with an output-layer
    regression. See ``.claude/docs/plans/spikesorting-v2/parity-extensions.md``
    § "Invariant fingerprinting".

    Sidecar history: the smoke parity row (``smoke_clusterless_5uv``)
    must NOT carry ``noise_levels`` -- that lets SI compute its own
    per-channel MAD and the threshold is interpreted as a MAD
    multiplier, matching how ``baseline_capture._ensure_smoke_sorter_param_row``
    inserts the v1 row. If ``noise_levels=[1.0]`` is forwarded
    (the production default for the 100 uV ``default_clusterless``
    row), the smoke threshold becomes raw uV and produces ~1,400x
    more detections than v1. The schema v3 + tolerance-aware
    matcher landed in followup #11 to close that divergence; the
    canonical-sorter fingerprint check guarantees it stays closed.

    Env vars:
      ``SPIKESORTING_V2_BASELINE_ROOT`` -> directory tree
          ``$ROOT/<fixture_stem>/clusterless/shank<N>/``
          containing ``baseline_v1_spike_times.pkl`` and
          ``baseline_v1_recording_meta.json``. The matrix-verification
          entrypoint -- when set, missing baselines FAIL (a broken
          capture must not skip silently) unless the
          ``(fixture_stem, sorter, sort_group_id)`` triple is in
          ``EXPECTED_DEGENERATE_CASES`` (then SKIP).

      ``SPIKESORTING_V2_BASELINE_DIR`` -> legacy single-dir shim
          for ``(mearec_polymer_smoke, 0)`` only; other
          parametrizations skip with a clear message.
    """
    import json
    import os
    import pickle
    from pathlib import Path as _Path

    import numpy as np

    from tests.spikesorting.v2._smoke_constants import EXPECTED_DEGENERATE_CASES

    # The NWB lives in the v2 checkout's fixtures directory; the
    # capture-side ran from the v1 worktree but wrote spike times +
    # meta under SPIKESORTING_V2_BASELINE_ROOT, so the v2 test reads
    # from the v2 fixture directly and verifies sha256 against the
    # baseline meta below.
    fixture_path = _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
    if not fixture_path.exists():
        pytest.skip(
            f"NWB fixture {fixture_path} missing; generate via "
            "tests/spikesorting/v2/fixtures/generate_mearec.py first."
        )

    sorter_label = "clusterless"
    baseline_root_env = os.environ.get("SPIKESORTING_V2_BASELINE_ROOT")
    baseline_dir_legacy_env = os.environ.get("SPIKESORTING_V2_BASELINE_DIR")
    legacy_case = (fixture_stem, sort_group_id) == ("mearec_polymer_smoke", 0)

    if baseline_root_env:
        baseline_dir = (
            _Path(baseline_root_env)
            / fixture_stem
            / sorter_label
            / f"shank{sort_group_id}"
        )
    elif baseline_dir_legacy_env and legacy_case:
        baseline_dir = _Path(baseline_dir_legacy_env)
    else:
        pytest.skip(
            "SPIKESORTING_V2_BASELINE_ROOT unset -- set it to a "
            "directory tree of `<root>/<fixture_stem>/clusterless/"
            "shank<N>/{baseline_v1_spike_times.pkl,baseline_v1_recording_meta.json}` "
            "to enable the v1↔v2 matrix verification. Captures are "
            "produced by tests/spikesorting/v2/scripts/"
            "capture_polymer_clusterless.sh."
        )

    triple = (fixture_stem, "clusterless_thresholder", sort_group_id)
    if triple in EXPECTED_DEGENERATE_CASES:
        # Under active matrix verification (ROOT set), require an
        # operator-written DEGENERATE_MARKER file in the per-case
        # baseline dir so a broken capture (no run, no marker)
        # can't masquerade as an intentional skip. The marker must
        # exist alongside the missing baseline files; absence is a
        # FAIL not a SKIP.
        if baseline_root_env:
            marker = baseline_dir / "DEGENERATE_MARKER"
            if not marker.exists():
                pytest.fail(
                    f"triple {triple} is in EXPECTED_DEGENERATE_CASES "
                    "but no DEGENERATE_MARKER artifact at "
                    f"{marker}. Operator must `touch` the marker after "
                    "evidence-based capture-side triage so broken "
                    "captures don't look like intentional skips. "
                    f"Documented reason: "
                    f"{EXPECTED_DEGENERATE_CASES[triple]}"
                )
        pytest.skip(
            f"SKIP-expected-degenerate: {EXPECTED_DEGENERATE_CASES[triple]}"
        )

    spikes_pkl = baseline_dir / "baseline_v1_spike_times.pkl"
    meta_json = baseline_dir / "baseline_v1_recording_meta.json"
    if not (spikes_pkl.exists() and meta_json.exists()):
        msg = (
            f"v1 baseline artifacts not found at {baseline_dir}/. "
            "Generate via `tests/spikesorting/v2/scripts/"
            "capture_polymer_clusterless.sh` under the v1 conda env."
        )
        if baseline_root_env:
            # SPIKESORTING_V2_BASELINE_ROOT is set: the operator opted
            # into matrix verification, so a missing baseline is a
            # capture-side regression, not a skip. To intentionally
            # skip a known-degenerate case, add the triple to
            # ``EXPECTED_DEGENERATE_CASES`` with evidence.
            pytest.fail(msg)
        pytest.skip(msg)

    nwb_path = fixture_path
    regen_hint = (
        f"Regenerate via baseline_capture.py --nwb-file {nwb_path} "
        f"--sort-group-id {sort_group_id} --output-dir {baseline_dir}."
    )
    try:
        with open(spikes_pkl, "rb") as fh:
            v1_spike_times = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise RuntimeError(
            f"v1 baseline pickle at {spikes_pkl} is corrupt or "
            f"class-incompatible ({type(exc).__name__}: {exc}). "
            f"{regen_hint}"
        ) from exc
    try:
        meta = json.loads(meta_json.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"v1 baseline meta at {meta_json} is malformed "
            f"({type(exc).__name__}: {exc}). {regen_hint}"
        ) from exc

    required_meta_keys = {
        "sorter",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        "sampling_frequency_hz",
    }
    missing_meta_keys = required_meta_keys - set(meta)
    if missing_meta_keys:
        raise RuntimeError(
            f"v1 baseline meta at {meta_json} is missing required keys "
            f"{sorted(missing_meta_keys)}; baseline schema has drifted "
            f"or the file is truncated. {regen_hint}"
        )

    if meta["sorter"] != "clusterless_thresholder":
        pytest.skip(
            f"v1 baseline captured for sorter={meta['sorter']!r}; "
            "this parity gate only implements the "
            "clusterless_thresholder ±1-sample tolerance. "
            "Regenerate the baseline with "
            "--sorter clusterless_thresholder, or extend this "
            "test for mountainsort/kilosort tolerance bands."
        )

    # Run v2 on the same (sort_group_id, interval_list_name, team).
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = copy_and_insert_nwb(nwb_path)
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    # Mirror baseline_capture's smoke-row insertion: if the v1
    # baseline used a non-default clusterless threshold (e.g.
    # ``smoke_clusterless_5uv`` for the MEArec smoke fixture),
    # pre-insert the corresponding row in v2 so the FK check in
    # ``SortingSelection.insert_selection`` resolves. Without
    # this guard the test fails opaquely with a missing-row
    # ValueError instead of producing the spike-count signal the
    # parity comparison is supposed to expose.
    # Map v1's row-name conventions to v2's equivalents. v1 ships
    # ``preproc_param_name="default"`` / ``sorter_param_name=
    # "default_clusterless"``; v2 ships ``preprocessing_params_name=
    # "default"`` / ``sorter_params_name="default"``. The
    # smoke row (``smoke_clusterless_5uv``) is consistently named
    # across both pipelines. Custom names that the lab user picks
    # are honored verbatim (fall-through in the mapping helpers).
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
        v2_preproc_name_for_v1,
        v2_sorter_name_for_v1,
    )

    sorter_params_name = v2_sorter_name_for_v1(
        meta.get("sorter_param_name", "default_clusterless")
    )
    if sorter_params_name == SMOKE_CLUSTERLESS_PARAM_NAME:
        # Use delete-then-insert (not skip_duplicates=True) because a
        # stale row left over from an earlier test run could silently
        # shadow this insert and flip the threshold's semantic meaning
        # (the historical noise_levels=[1.0] regression injected raw-uV
        # semantics; a delete-then-insert guarantees the row matches
        # the schema version this test was written for).
        (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
            }
        ).delete(safemode=False)
        SorterParameters().insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                # Smoke params shared with baseline_capture via
                # _smoke_constants; ``noise_levels`` is omitted so
                # SI computes per-channel MAD on both sides.
                "params": dict(SMOKE_CLUSTERLESS_PARAMS),
                "params_schema_version": 4,
                "job_kwargs": None,
            },
            skip_duplicates=False,
        )
    LabTeam.insert1(
        {"team_name": meta["team_name"], "team_description": "v1 parity"},
        skip_duplicates=True,
    )
    if not (
        SortGroupV2
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
        }
    ):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    preproc_name_v2 = v2_preproc_name_for_v1(
        meta.get("preproc_param_name", "default")
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
            "interval_list_name": meta["interval_list_name"],
            "preprocessing_params_name": preproc_name_v2,
            "team_name": meta["team_name"],
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    # Honor the baseline's artifact-detection params. If v1 ran
    # with ``"default"`` (which detects + masks above-threshold
    # peaks) and v2 ran with ``"none"`` (skip artifact scanning),
    # v2 would be sorting unmasked data and the per-spike drift
    # check below would compare different pipelines. v2 ships the
    # same ``"default"`` artifact-params name with the same
    # semantics.
    artifact_name_meta = meta.get("artifact_param_name", "default")
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": artifact_name_meta,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1↔v2 invariant fingerprint check ---
    # Reconstruct the same fingerprints from v2 tables and assert
    # each matches what v1 baseline_capture wrote. Done BEFORE the
    # sort so an input-layer skew (different electrodes, different
    # bad-channel mask, schema-incompatible params, divergent
    # artifact valid_times) is never silently absorbed into the
    # downstream spike-time comparison. See
    # ``.claude/docs/plans/spikesorting-v2/parity-extensions.md`` §
    # "Invariant fingerprinting".
    import hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )
    from tests.spikesorting.v2._parity_canonical import (
        _normalize,
        assert_canonical_dict_equal,
        canonical_artifact,
        canonical_preproc,
        canonical_sorter,
    )

    _fingerprint_fields = (
        "nwb_sha256",
        "sort_group_electrode_ids",
        "bad_channel_by_electrode_id",
        "canonical_preproc_params",
        "canonical_artifact_params",
        "artifact_valid_times",
        "canonical_sorter_params",
    )
    missing_fp = sorted(set(_fingerprint_fields) - set(meta))
    if missing_fp:
        msg = (
            f"v1 baseline meta at {meta_json} lacks fingerprint "
            f"field(s) {missing_fp}; baseline was captured before "
            f"the invariant-fingerprint schema landed. {regen_hint}"
        )
        # Under SPIKESORTING_V2_BASELINE_ROOT (active matrix
        # verification), a stale baseline is an invalid input -- not
        # an acceptable skip. Match the missing-baseline-files
        # policy: FAIL when ROOT is set, SKIP under the legacy
        # BASELINE_DIR shim only.
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    v2_valid_times_raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_detection_interval_list_name(
                art_pk["artifact_detection_id"]
            ),
        }
    ).fetch1("valid_times")
    # Canonicalize to ``(n_intervals, 2)`` and round to 1 ms -- see
    # baseline_capture._compute_invariant_fingerprints for why
    # (1-sample-at-end convention difference between v1 and v2).
    v2_valid_times = np.round(
        np.ascontiguousarray(v2_valid_times_raw, dtype="<f8").reshape(-1, 2),
        decimals=3,
    )
    v2_fingerprints = {
        "nwb_sha256": hashlib.sha256(nwb_path.read_bytes()).hexdigest(),
        "sort_group_electrode_ids": sorted(
            int(eid)
            for eid in (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": int(meta["sort_group_id"]),
                }
            ).fetch("electrode_id")
        ),
        "bad_channel_by_electrode_id": {
            str(int(row["electrode_id"])): bool(row["bad_channel"] == "True")
            for row in (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
                "KEY", "electrode_id", "bad_channel", as_dict=True
            )
        },
        "canonical_preproc_params": canonical_preproc(
            (
                PreprocessingParameters
                & {"preprocessing_params_name": preproc_name_v2}
            ).fetch1("params")
        ),
        "canonical_artifact_params": canonical_artifact(
            (
                ArtifactDetectionParameters
                & {"artifact_detection_params_name": artifact_name_meta}
            ).fetch1("params")
        ),
        # Stored as the canonical (shape-aware) array rather than a
        # sha256 hash: bit-level float drift between the v1 (SI 0.99)
        # and v2 (SI 0.104) ``IntervalList.valid_times`` round-trips
        # would otherwise FAIL even when the two intervals are
        # semantically identical. ``assert_canonical_dict_equal``
        # compares element-wise with ``math.isclose(rel_tol=1e-9)``.
        "artifact_valid_times": _normalize(v2_valid_times),
        "canonical_sorter_params": canonical_sorter(
            "clusterless_thresholder",
            (
                SorterParameters
                & {
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": sorter_params_name,
                }
            ).fetch1("params"),
        ),
    }
    v1_fingerprints = {field: meta[field] for field in _fingerprint_fields}
    assert_canonical_dict_equal(
        v1_fingerprints, v2_fingerprints, path="fingerprints"
    )

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")

    # Pull v2 spike times in seconds via the merge dispatcher --
    # same surface a v1 consumer would hit; if v1 and v2 use the
    # same units list the per-unit arrays line up by index.
    v2_per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    v2_unit_ids = (Sorting.Unit & sort_pk).fetch("unit_id", order_by="unit_id")
    v2_spike_times = {
        int(uid): np.asarray(arr) for uid, arr in zip(v2_unit_ids, v2_per_unit)
    }

    fs = float(meta["sampling_frequency_hz"])
    one_sample_s = 1.0 / fs

    v1_keys = sorted(int(k) for k in v1_spike_times.keys())
    v2_keys = sorted(v2_spike_times.keys())
    assert v1_keys == v2_keys, (
        f"v1 baseline has units {v1_keys}, v2 produced {v2_keys}; "
        "clusterless_thresholder is deterministic so the unit set "
        "should match exactly. A mismatch indicates a real "
        "regression in detection / labeling."
    )
    # Non-vacuous-pass guard: if both unit sets are empty (e.g.
    # the baseline was captured on a recording with no above-
    # threshold peaks), the loop below iterates zero times and
    # the test passes without verifying any spike times.
    assert len(v1_keys) >= 1, (
        "v1 baseline has zero units; the per-unit drift check "
        "below would pass vacuously. Regenerate "
        "baseline_v1_spike_times.pkl on a recording with above-"
        "threshold activity."
    )

    # Per-unit asymmetric per-spike comparison.
    #
    # Clusterless contract (parity-extensions.md § "Documented v2
    # divergences"): every v1 spike must match a v2 spike within
    # ±1.5 samples. Budgets account for the SI 0.99 → 0.104
    # ``locally_exclusive`` numba kernel rewrite ([SI PR #4341](
    # https://github.com/SpikeInterface/spikeinterface/pull/4341)
    # "more accurate for corner cases: ratio not raw amplitude;
    # neighbour peak-to-peak not trace-value"), which the SI author
    # classifies as v1-wrong. Empirically v2 detects ~25–30% more
    # near-threshold peaks on contacts adjacent to v1 peaks (757/3392
    # of v2 peaks on shank 0 of the 60 s polymer, 100% within
    # ±30 samples and 99.5% within ``radius_um=100`` of a v1 peak).
    # plus a small residual unmatched_v1 (≤ 6 / 2652 = 0.2% after
    # decoupling from the [PR #3359](
    # https://github.com/SpikeInterface/spikeinterface/pull/3359)
    # ``noise_levels`` ``seed=None`` change). The budgets below
    # cover both verified mechanisms.
    threshold_samples = 1.5
    # v2 extras allowed up to 50% + 5 (PR #4341 v1-wrong adjacent
    # peaks). A 1,400x explosion (the historical noise_levels=[1.0]
    # regression) is still well outside this budget and fails loud.
    extra_spike_ratio = 0.50
    tol_s = threshold_samples * one_sample_s
    diagnostic_rows: list[tuple[int, int, int, int, int, int]] = []
    failures: list[str] = []
    for uid in v1_keys:
        v1_arr = np.sort(np.asarray(v1_spike_times[uid]))
        v2_arr = np.sort(v2_spike_times[uid])
        assert v1_arr.size >= 1, (
            f"unit {uid}: v1 baseline has zero spikes -- per-unit "
            "drift check is meaningless. Regenerate baseline."
        )
        # Defend against ``v2_arr.size == 0`` (a regression that
        # drops every detection); ``np.searchsorted`` would yield
        # ``idx_left = idx_right = 0`` and the indexed read would
        # raise ``IndexError`` with a stack trace instead of the
        # diagnostic message this assertion is supposed to surface.
        if v2_arr.size == 0:
            diagnostic_rows.append(
                (uid, int(v1_arr.size), 0, 0, int(v1_arr.size), 0)
            )
            failures.append(
                f"unit {uid}: v1 has {v1_arr.size} spikes, v2 "
                "detected zero; nearest-neighbor match impossible. "
                "Regression in v2 sort path."
            )
            continue

        # v1 -> v2 nearest neighbor: each v1 spike's distance to the
        # closest v2 spike. Two candidates per v1 spike (searchsorted
        # gives the insertion index; left/right neighbors), take
        # the minimum.
        idx = np.searchsorted(v2_arr, v1_arr)
        idx_left = np.clip(idx - 1, 0, v2_arr.size - 1)
        idx_right = np.clip(idx, 0, v2_arr.size - 1)
        v1_to_v2_diff = np.minimum(
            np.abs(v1_arr - v2_arr[idx_left]),
            np.abs(v1_arr - v2_arr[idx_right]),
        )
        v1_matched = v1_to_v2_diff <= tol_s
        n_matched = int(v1_matched.sum())
        unmatched_v1 = int(v1_arr.size - n_matched)

        # v2 -> v1 nearest neighbor: each v2 spike's distance to the
        # closest v1 spike. ``unmatched_v2`` is a triage signal (the
        # 50%+5 budget on v2_arr.size is the assertion that catches a
        # blow-up; per-spike unmatched_v2 surfaces a quieter drift).
        idx2 = np.searchsorted(v1_arr, v2_arr)
        idx2_left = np.clip(idx2 - 1, 0, v1_arr.size - 1)
        idx2_right = np.clip(idx2, 0, v1_arr.size - 1)
        v2_to_v1_diff = np.minimum(
            np.abs(v2_arr - v1_arr[idx2_left]),
            np.abs(v2_arr - v1_arr[idx2_right]),
        )
        unmatched_v2 = int((v2_to_v1_diff > tol_s).sum())

        diagnostic_rows.append(
            (
                uid,
                int(v1_arr.size),
                int(v2_arr.size),
                n_matched,
                unmatched_v1,
                unmatched_v2,
            )
        )

        # unmatched_v1 budget: max(2, 1% of v1 peaks). Covers the
        # PR #3359 noise_levels stochastic drift residual (6/2652 =
        # 0.2% observed on shank 0 after the control experiment
        # decoupled algorithm- vs noise-level mechanisms).
        unmatched_v1_budget = max(2, int(0.01 * v1_arr.size))
        if unmatched_v1 > unmatched_v1_budget:
            failures.append(
                f"unit {uid}: {unmatched_v1}/{v1_arr.size} v1 spikes "
                f"lack a v2 match within {threshold_samples} samples "
                f"(budget {unmatched_v1_budget}; tol {tol_s:.6f}s). "
                f"max unmatched drift "
                f"{float(v1_to_v2_diff[~v1_matched].max() * fs):.2f} samples."
            )

        # Cap v2's excess detections; a 1,400x explosion (the pre-fix
        # state when ``noise_levels=[1.0]`` was injected into the
        # smoke row) fails this assertion loud and clear.
        max_v2 = int(v1_arr.size * (1.0 + extra_spike_ratio)) + 5
        if v2_arr.size > max_v2:
            failures.append(
                f"unit {uid}: v2 detected {v2_arr.size} spikes vs v1's "
                f"{v1_arr.size}; excess > {extra_spike_ratio:.0%} budget "
                f"(allowed {max_v2}). Likely a sorter-parameter semantic "
                "mismatch (e.g. forwarding ``noise_levels=[1.0]`` so the "
                "threshold reads in raw uV instead of MAD multiples)."
            )

    # Diagnostic block: one line per unit, then a summary line.
    # Printed unconditionally so a passing run still records the
    # per-unit unmatched counts for the next reviewer.
    print(f"\n[parity] {fixture_stem} shank{sort_group_id} clusterless")
    print(
        f"  {'unit':>5} {'v1_n':>6} {'v2_n':>6} {'matched':>8} "
        f"{'unmatched_v1':>13} {'unmatched_v2':>13}"
    )
    for row in diagnostic_rows:
        uid, v1n, v2n, nm, um1, um2 = row
        print(f"  {uid:>5} {v1n:>6} {v2n:>6} {nm:>8} {um1:>13} {um2:>13}")

    if failures:
        pytest.fail("v1↔v2 spike-time divergence:\n  " + "\n  ".join(failures))


_MS4_PARITY_CASES = [("mearec_polymer_128ch_60s", sg) for sg in (0, 1, 2, 3)]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_stem,sort_group_id",
    _MS4_PARITY_CASES,
    ids=[f"{stem}-shank{sg}" for stem, sg in _MS4_PARITY_CASES],
)
def test_v2_real_data_v1_parity_mountainsort4(
    fixture_stem, sort_group_id, dj_conn
):
    """v1 ↔ v2 MountainSort4 parity on the polymer 60s fixture.

    MS4 is stochastic (no seed control -- ``MountainSort4Sorter.default_params()``
    has no ``seed`` field in either SI 0.99.1 or 0.104.3, confirmed
    empirically). The C++ MS4 1.0.7 binary is byte-identical across
    envs; any v1↔v2 divergence comes from SI's wrapping (preproc,
    chunking, parameter forwarding) layered on top.

    Contract: aggregate bands (NOT per-spike, unlike clusterless),
    sourced from
    :data:`tests.spikesorting.v2._smoke_constants.MS4_BANDS`. After
    Phase B11 calibration on shanks 0 and 2 (two repeats per side)
    these are:

    * ``|v2_n_units - v1_n_units| ≤ max(10% of v1_n_units, 2)``
    * ``|v2_median_fr - v1_median_fr| / v1_median_fr ≤ 10%`` (rel)

    See :data:`MS4_VARIANCE_TABLE` for the calibration measurements
    (v1v1 drift = 0 on both shanks → MS4 deterministic on v1; v2v2
    drift ≤ 1 unit / 0.74 Hz ≤ 3% of median_fr).

    Env vars / SKIP semantics: same as :func:`test_v2_real_data_v1_parity`
    plus a SKIP-env-unavailable when MS4 is not installed in v2's
    sorter set (overridable to FAIL via
    ``SPIKESORTING_V2_REQUIRE_MS4=1``).
    """
    import json
    import os
    import pickle
    from pathlib import Path as _Path

    import numpy as np

    from tests.spikesorting.v2._smoke_constants import (
        EXPECTED_DEGENERATE_CASES,
        MS4_60S_POLYMER_PARAM_NAME,
        MS4_60S_POLYMER_PARAMS,
        MS4_BANDS,
    )

    # MS4 install gate (v2 side; v1 install is checked at capture time).
    import spikeinterface.sorters as ss

    if "mountainsort4" not in ss.installed_sorters():
        msg = (
            "mountainsort4 not in spikeinterface.sorters.installed_sorters(); "
            "skipping. Set SPIKESORTING_V2_REQUIRE_MS4=1 to make this a "
            "hard fail."
        )
        if os.environ.get("SPIKESORTING_V2_REQUIRE_MS4") == "1":
            pytest.fail(msg)
        pytest.skip(msg)

    fixture_path = _Path(__file__).parent / "fixtures" / f"{fixture_stem}.nwb"
    if not fixture_path.exists():
        pytest.skip(
            f"NWB fixture {fixture_path} missing; generate via "
            "tests/spikesorting/v2/fixtures/generate_mearec.py first."
        )

    sorter_label = "ms4"
    baseline_root_env = os.environ.get("SPIKESORTING_V2_BASELINE_ROOT")
    if baseline_root_env:
        baseline_dir = (
            _Path(baseline_root_env)
            / fixture_stem
            / sorter_label
            / f"shank{sort_group_id}"
        )
    else:
        pytest.skip(
            "SPIKESORTING_V2_BASELINE_ROOT unset -- set it to a "
            "directory tree of `<root>/<fixture_stem>/ms4/shank<N>/"
            "{baseline_v1_spike_times.pkl,baseline_v1_recording_meta.json}` "
            "to enable the v1↔v2 MS4 matrix verification. Captures are "
            "produced by tests/spikesorting/v2/scripts/capture_polymer_ms4.sh."
        )

    triple = (fixture_stem, "mountainsort4", sort_group_id)
    if triple in EXPECTED_DEGENERATE_CASES:
        # See clusterless test for marker-artifact rationale.
        if baseline_root_env:
            marker = baseline_dir / "DEGENERATE_MARKER"
            if not marker.exists():
                pytest.fail(
                    f"triple {triple} is in EXPECTED_DEGENERATE_CASES "
                    "but no DEGENERATE_MARKER artifact at "
                    f"{marker}. Operator must `touch` the marker after "
                    "evidence-based capture-side triage. Documented "
                    f"reason: {EXPECTED_DEGENERATE_CASES[triple]}"
                )
        pytest.skip(
            f"SKIP-expected-degenerate: {EXPECTED_DEGENERATE_CASES[triple]}"
        )

    spikes_pkl = baseline_dir / "baseline_v1_spike_times.pkl"
    meta_json = baseline_dir / "baseline_v1_recording_meta.json"
    if not (spikes_pkl.exists() and meta_json.exists()):
        msg = (
            f"v1 MS4 baseline artifacts not found at {baseline_dir}/. "
            "Generate via `tests/spikesorting/v2/scripts/"
            "capture_polymer_ms4.sh` under the v1 conda env."
        )
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    with open(spikes_pkl, "rb") as fh:
        v1_spike_times = pickle.load(fh)
    meta = json.loads(meta_json.read_text())

    if meta.get("sorter") != "mountainsort4":
        pytest.skip(
            f"v1 baseline at {baseline_dir} was captured for "
            f"sorter={meta.get('sorter')!r}; MS4 parity test refuses "
            "to apply MS4-band contract against a non-MS4 baseline."
        )

    # Set up v2 pipeline -- mirrors the clusterless test until the
    # SorterParameters insert and the comparator.
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = copy_and_insert_nwb(fixture_path)
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()

    # v2-side insert of the polymer-60s MS4 row. Owned end-to-end by
    # this test pair (parity_canonical also strips schema_version, so
    # the canonical fingerprint matches v1's row even though v2's
    # Pydantic schema stamps it).
    sorter_params_name = MS4_60S_POLYMER_PARAM_NAME
    (
        SorterParameters
        & {"sorter": "mountainsort4", "sorter_params_name": sorter_params_name}
    ).delete(safemode=False)
    SorterParameters().insert1(
        {
            "sorter": "mountainsort4",
            "sorter_params_name": sorter_params_name,
            "params": dict(MS4_60S_POLYMER_PARAMS),
            "params_schema_version": 1,
            "job_kwargs": None,
        },
        skip_duplicates=False,
    )

    LabTeam.insert1(
        {"team_name": meta["team_name"], "team_description": "v1 MS4 parity"},
        skip_duplicates=True,
    )
    if not (
        SortGroupV2
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
        }
    ):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    from tests.spikesorting.v2._smoke_constants import (
        v2_preproc_name_for_v1,
    )

    preproc_name_v2 = v2_preproc_name_for_v1(
        meta.get("preproc_param_name", "default")
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(meta["sort_group_id"]),
            "interval_list_name": meta["interval_list_name"],
            "preprocessing_params_name": preproc_name_v2,
            "team_name": meta["team_name"],
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    artifact_name_meta = meta.get("artifact_param_name", "default")
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": artifact_name_meta,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1↔v2 invariant fingerprint check (same shape as clusterless) ---
    import hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )
    from tests.spikesorting.v2._parity_canonical import (
        _normalize,
        assert_canonical_dict_equal,
        canonical_artifact,
        canonical_preproc,
        canonical_sorter,
    )

    _fingerprint_fields = (
        "nwb_sha256",
        "sort_group_electrode_ids",
        "bad_channel_by_electrode_id",
        "canonical_preproc_params",
        "canonical_artifact_params",
        "artifact_valid_times",
        "canonical_sorter_params",
    )
    missing_fp = sorted(set(_fingerprint_fields) - set(meta))
    if missing_fp:
        msg = (
            f"v1 MS4 baseline meta at {meta_json} lacks fingerprint "
            f"field(s) {missing_fp}; baseline was captured before the "
            "invariant-fingerprint schema landed."
        )
        if baseline_root_env:
            pytest.fail(msg)
        pytest.skip(msg)

    v2_valid_times_raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_detection_interval_list_name(
                art_pk["artifact_detection_id"]
            ),
        }
    ).fetch1("valid_times")
    v2_valid_times = np.round(
        np.ascontiguousarray(v2_valid_times_raw, dtype="<f8").reshape(-1, 2),
        decimals=3,
    )
    v2_fingerprints = {
        "nwb_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "sort_group_electrode_ids": sorted(
            int(eid)
            for eid in (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": int(meta["sort_group_id"]),
                }
            ).fetch("electrode_id")
        ),
        "bad_channel_by_electrode_id": {
            str(int(row["electrode_id"])): bool(row["bad_channel"] == "True")
            for row in (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
                "KEY", "electrode_id", "bad_channel", as_dict=True
            )
        },
        "canonical_preproc_params": canonical_preproc(
            (
                PreprocessingParameters
                & {"preprocessing_params_name": preproc_name_v2}
            ).fetch1("params")
        ),
        "canonical_artifact_params": canonical_artifact(
            (
                ArtifactDetectionParameters
                & {"artifact_detection_params_name": artifact_name_meta}
            ).fetch1("params")
        ),
        "artifact_valid_times": _normalize(v2_valid_times),
        "canonical_sorter_params": canonical_sorter(
            "mountainsort4",
            (
                SorterParameters
                & {
                    "sorter": "mountainsort4",
                    "sorter_params_name": sorter_params_name,
                }
            ).fetch1("params"),
        ),
    }
    v1_fingerprints = {field: meta[field] for field in _fingerprint_fields}
    assert_canonical_dict_equal(
        v1_fingerprints, v2_fingerprints, path="fingerprints"
    )

    # Now run the MS4 sort and compute aggregate metrics.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort4",
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")

    v2_per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    v2_spike_times = {i: np.asarray(arr) for i, arr in enumerate(v2_per_unit)}
    v1_spike_times_arrs = {
        int(uid): np.asarray(times) for uid, times in v1_spike_times.items()
    }

    # Use ``artifact_valid_times`` duration (the interval the SORTER
    # actually saw) as the firing-rate denominator. The pre-fix code
    # used ``approx_last_spike_s`` from the v1 meta which is the
    # last v1 spike time -- making both v1 and v2 firing rates
    # NORMALIZED BY v1's OUTPUT (not the fixed input window). The
    # fingerprint check above has already asserted v1's and v2's
    # ``artifact_valid_times`` match, so using v1's stored array is
    # equivalent to v2's reconstructed one.
    v1_avt = meta["artifact_valid_times"]
    v1_intervals = np.asarray(v1_avt["_array_data"]).reshape(
        v1_avt["_array_meta"]["shape"]
    )
    duration_s = float(np.sum(v1_intervals[:, 1] - v1_intervals[:, 0]))
    if duration_s <= 0:
        pytest.fail(
            "artifact_valid_times duration <= 0; non-comparable. "
            f"intervals={v1_intervals.tolist()}"
        )

    def _aggregate(spike_dict, dur):
        n_units = len(spike_dict)
        if n_units == 0 or dur <= 0:
            return n_units, 0.0
        rates = sorted(len(t) / dur for t in spike_dict.values())
        mid = n_units // 2
        median_fr = (
            rates[mid]
            if n_units % 2 == 1
            else (rates[mid - 1] + rates[mid]) / 2
        )
        return n_units, median_fr

    v1_n, v1_fr = _aggregate(v1_spike_times_arrs, duration_s)
    v2_n, v2_fr = _aggregate(v2_spike_times, duration_s)

    print(
        f"\n[parity-ms4] {fixture_stem} shank{sort_group_id}\n"
        f"  v1: n_units={v1_n}, median_fr={v1_fr:.3f} Hz\n"
        f"  v2: n_units={v2_n}, median_fr={v2_fr:.3f} Hz"
    )

    # MS4_BANDS = MS4_CALIBRATED post-Phase B11 (n_units ± 10% or
    # ± 2, median_fr ± 10% rel). See MS4_VARIANCE_TABLE for the
    # calibration measurements.
    n_band = max(
        v1_n * MS4_BANDS["n_units_rel_band"],
        MS4_BANDS["n_units_abs_band"],
    )
    fr_band_abs = abs(v1_fr * MS4_BANDS["median_fr_rel_band"])
    failures = []
    if abs(v2_n - v1_n) > n_band:
        failures.append(
            f"n_units divergence: |v2_n({v2_n}) - v1_n({v1_n})|={abs(v2_n - v1_n)} "
            f"> band={n_band} (rel={MS4_BANDS['n_units_rel_band']:.0%} "
            f"abs={MS4_BANDS['n_units_abs_band']})."
        )
    if v1_fr > 0 and abs(v2_fr - v1_fr) > fr_band_abs:
        failures.append(
            f"median_fr divergence: |v2_fr({v2_fr:.3f}) - v1_fr({v1_fr:.3f})|"
            f"={abs(v2_fr - v1_fr):.3f} Hz > rel-band="
            f"{MS4_BANDS['median_fr_rel_band']:.0%} ({fr_band_abs:.3f} Hz)."
        )
    if failures:
        pytest.fail(
            "v1↔v2 MS4 aggregate-band divergence:\n  " + "\n  ".join(failures)
        )


# MS4 GT is polymer-only. A tetrode case was tried and removed: MS4 with
# default Frank-lab tetrode params cannot resolve the 5 planted units on a
# 4-channel synthetic MEArec tetrode regardless of detect_threshold. Verified
# with pure SpikeInterface (no Spyglass code in the path) by sweeping
# detect_threshold ∈ {3, 4, 5, 6}: every threshold recovers at most 1/5 GT
# units at accuracy ≥ 0.5 (GT unit 4 is 96 % recovered but split across 4 MS4
# output units; GT units 0, 2, 3 missed entirely). This is a sorter
# limitation on low-channel-count probes, not a v2 bug — v2 byte-parity with
# v1 on the polymer MS4 path (test_v2_real_data_v1_parity_mountainsort4) is
# the canonical v2-correctness gate for MS4.
_MS4_GT_CASES = [
    ("polymer_60s", "franklab_30khz_ms4_2026_06"),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "session_label,ms4_param_name",
    _MS4_GT_CASES,
    ids=[case[0] for case in _MS4_GT_CASES],
)
def test_mountainsort4_ground_truth(
    session_label, ms4_param_name, request, dj_conn
):
    """MS4 recovers planted units on the polymer MEArec fixture.

    Correctness gate independent of v1 parity. Uses SI's
    ``compare_sorter_to_ground_truth`` against the sidecar GT units
    table written by ``mearec_to_spyglass_nwb``.

    Parametrization:
      * ``polymer_60s`` -- 128-channel polymer probe (4 shanks),
        ``franklab_30khz_ms4_2026_06`` params (MS4 runs ``filter=False``;
        the 300-6000 Hz band comes from the ``default`` preproc row).

    Tetrode coverage was attempted and removed (see comment above
    ``_MS4_GT_CASES``): MS4 fundamentally cannot resolve a 4-channel
    synthetic tetrode regardless of detect_threshold, so the case was
    measuring a sorter limitation rather than v2 correctness.

    Threshold: at least 1/2 of planted units detected at accuracy
    >= 0.5 (looser than the MS5 polymer gate's 1/2 >= 0.7 because
    MS4 is a less-accurate sorter).
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    session = request.getfixturevalue(f"{session_label}_session")
    nwb_file_name = session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    if not (LabTeam & {"team_name": "v2_test_team"}):
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
        )

    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g) for g in (SortGroupV2 & session).fetch("sort_group_id")
    )

    sortings_by_shank = []
    for sg_id in sort_group_ids:
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort4",
                "sorter_params_name": ms4_param_name,
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    aggregated = aggregate_units(sortings_by_shank)
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        fs = (
            float(es.rate)
            if es.rate is not None
            else 1.0 / float(np.diff(es.timestamps[:2])[0])
        )
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert gt_units_table is not None, (
            f"Fixture {nwb_file_name!r} has no sidecar ground-truth "
            "units; regenerate via generate_mearec.py."
        )
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }
    gt_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[
            {
                int(uid): (times * fs).astype(np.int64)
                for uid, times in gt_units.items()
            }
        ],
        sampling_frequency=fs,
    )

    comparison = compare_sorter_to_ground_truth(
        gt_sorting=gt_sorting,
        tested_sorting=tested_sorting,
        gt_name=f"{session_label}_planted",
        tested_name="v2_mountainsort4",
        delta_time=0.4,
        match_score=0.5,
        exhaustive_gt=True,
    )

    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    summary = (
        f"\n[MS4 vs {session_label} GT] n_planted={n_planted}, "
        f"n_detected={len(tested_sorting.unit_ids)}\n"
        f"  accuracy  mean={accuracies.mean():.3f} "
        f"median={float(np.median(accuracies)):.3f} "
        f"min={accuracies.min():.3f} max={accuracies.max():.3f}\n"
        f"  recall    mean={recall.mean():.3f} "
        f"median={float(np.median(recall)):.3f}\n"
        f"  precision mean={precision.mean():.3f} "
        f"median={float(np.median(precision)):.3f}"
    )
    print(summary)

    assert n_planted >= 1, (
        f"{session_label} fixture has only {n_planted} planted units; "
        f"regenerate before trusting this gate.{summary}"
    )

    # Looser MS4 threshold (half of planted at acc >= 0.5) than the
    # MS5 polymer gate (1/2 >= 0.7) because MS4 is a less-accurate
    # sorter. Tightening this threshold without committed calibration
    # evidence is regression-prone.
    n_well_detected = int((accuracies >= 0.5).sum())
    threshold = max(1, n_planted // 2)
    assert n_well_detected >= threshold, (
        f"MS4 on {session_label} detected {n_well_detected} of "
        f"{n_planted} planted units at accuracy >= 0.5; "
        f"threshold requires >= {threshold} (1/2 of planted).{summary}"
    )


_CLUSTERLESS_GT_CASES = [
    ("polymer_60s", "smoke_clusterless_5uv"),
    ("tetrode_60s", "smoke_clusterless_5uv"),
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "session_label,sorter_params_name",
    _CLUSTERLESS_GT_CASES,
    ids=[case[0] for case in _CLUSTERLESS_GT_CASES],
)
def test_clusterless_thresholder_ground_truth(
    session_label, sorter_params_name, request, dj_conn
):
    """Clusterless detector recovers planted-unit spikes (shank-aware recall).

    Correctness gate for the clusterless path. Unlike MS4/MS5, the
    clusterless detector emits unsorted peaks (not units), so SI's
    ``compare_sorter_to_ground_truth`` doesn't apply. We compute a
    shank-aware recall metric: per planted unit, compute the
    ±0.4-ms time-recall against each shank's v2 peak stream
    independently and take the max. The match constraint is "v2
    peaks from a SINGLE shank must explain the planted spikes" --
    cross-shank peaks can no longer satisfy a planted spike. Best-
    shank-per-unit (rather than nearest-shank-by-soma-position) is
    robust to cells placed between shanks or off the probe edge:
    such cells fire most strongly on whichever shank actually picks
    them up, which is an electrical question, not a Euclidean one.

    Shank-aware matching closes the false-recall window that
    time-only-across-all-shanks opens: with ~17 K peaks across
    4 shanks and 60 s, uniform-cross-channel time-only matching at
    a 0.4 ms tolerance could inflate recall by ~20 percentage points
    from unrelated peaks. Best-shank routing eliminates that
    inflation while keeping the rigorous time tolerance.

    ``delta_time = 0.4 ms`` matches SI's
    ``compare_sorter_to_ground_truth`` default and the sibling MS4/
    MS5 GT gates; the natural physical offset between a planted
    intracellular fire moment and the detected extracellular trough
    (filter group delay + propagation) is empirically ~1-3 samples
    at 32 kHz, so a sub-sample tolerance would reject valid
    detections.

    Threshold: mean recall ≥ 0.80 across planted units. Clusterless
    detect_threshold = 5 (a MAD multiplier, not 5σ or 5 µV -- with
    ``noise_levels`` omitted SI scales the threshold by per-channel
    MAD) should catch all well-isolated spikes; higher recall bars
    risk flake from near-threshold borderline spikes (PR #4341
    algorithm change).

    Parametrization:
      * ``polymer_60s`` -- 4 shanks, planted units distributed across them
      * ``tetrode_60s`` -- 1 shank, 4 channels (trivially per-shank)
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    session = request.getfixturevalue(f"{session_label}_session")
    nwb_file_name = session["nwb_file_name"]

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    if not (LabTeam & {"team_name": "v2_test_team"}):
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
        )

    # Pre-insert the smoke clusterless row (detect_threshold=5, a MAD
    # multiplier -- not µV; doesn't ship as a v2 default; needs
    # delete-then-insert for the same noise_levels-semantics safety
    # reason as the v1↔v2 parity test).
    if sorter_params_name == SMOKE_CLUSTERLESS_PARAM_NAME:
        (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
            }
        ).delete(safemode=False)
        SorterParameters().insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                "params": dict(SMOKE_CLUSTERLESS_PARAMS),
                "params_schema_version": 4,
                "job_kwargs": None,
            },
            skip_duplicates=False,
        )

    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(g) for g in (SortGroupV2 & session).fetch("sort_group_id")
    )

    # Per-shank: populate clusterless sorting and keep v2 peak times
    # separate per shank so the recall matcher below can constrain
    # each planted unit's matches to a single shank's v2 peaks.
    v2_times_s_by_shank: dict[int, np.ndarray] = {}
    sampling_frequency = None
    for sg_id in sort_group_ids:
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "clusterless_thresholder",
                "sorter_params_name": sorter_params_name,
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        curation_pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
        merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1(
            "merge_id"
        )
        per_unit = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
        # Clusterless collapses all peaks into one synthetic unit per
        # shank. Concatenate (in case > 1 entry from get_spike_times)
        # and keep this shank's spike-times tagged by sort_group_id.
        arrays = [np.asarray(a) for a in per_unit if np.asarray(a).size]
        if arrays:
            v2_times_s_by_shank[int(sg_id)] = np.sort(
                np.concatenate(arrays).astype(np.float64)
            )
        else:
            v2_times_s_by_shank[int(sg_id)] = np.empty(0, dtype=np.float64)
        if sampling_frequency is None:
            sampling_frequency = float(
                Sorting().get_sorting(sort_pk).get_sampling_frequency()
            )

    if not v2_times_s_by_shank or all(
        a.size == 0 for a in v2_times_s_by_shank.values()
    ):
        pytest.fail(f"v2 clusterless produced no peaks on {session_label}.")
    # Aggregate copy for the false-positive sanity bound (counts only).
    v2_times_s = np.sort(
        np.concatenate([a for a in v2_times_s_by_shank.values() if a.size])
    ).astype(np.float64)
    assert sampling_frequency is not None and sampling_frequency > 0

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        gt_units_table = get_ground_truth_units_table(gt_nwb)
        assert (
            gt_units_table is not None
        ), f"{nwb_file_name!r} has no sidecar GT units; regenerate."
        gt_units = {
            int(uid): np.asarray(gt_units_table["spike_times"][idx])
            for idx, uid in enumerate(gt_units_table.id[:])
        }

    sorted_sgs = sorted(v2_times_s_by_shank.keys())

    # Shank-aware time-recall: for each planted unit, compute the
    # ±0.4-ms time-recall against EACH shank's v2 peak stream
    # separately, then take the max. The match constraint is "v2
    # peaks from a single shank must explain the planted spikes" --
    # cross-shank peaks no longer satisfy a planted spike. Compared
    # to assigning each unit to its nearest shank by soma position,
    # the max-over-shanks form is robust to cells placed between
    # shanks or off the probe edge in the shank-spanning axis: such
    # cells fire most strongly on whichever shank actually picks
    # them up (which is an electrical question, not a Euclidean-
    # nearest-shank question). Empirically on the polymer 60s
    # fixture, soma-position routing mis-routed 18/24 cells (z
    # extent [-570, +583] vs shanks at z=0/350/774/1124); empirical
    # best-shank routing gives every unit its true assignment.
    #
    # ``delta_time = 0.4 ms`` matches SI's
    # ``compare_sorter_to_ground_truth`` default (also used by the
    # sibling MS4/MS5 GT gates). At 32 kHz that is 12.8 samples; the
    # empirical (detected_peak - planted_spike) distribution on the
    # polymer 60s fixture has IQR ~1-2 samples (post-PR-#4341), so
    # 0.4 ms accepts every physically valid detection without
    # admitting unrelated noise peaks.
    tol_s = 0.4 / 1000.0

    def _recall_against(peaks_s: np.ndarray, gt_sorted: np.ndarray) -> float:
        if peaks_s.size == 0:
            return 0.0
        idx = np.searchsorted(peaks_s, gt_sorted)
        idx_left = np.clip(idx - 1, 0, peaks_s.size - 1)
        idx_right = np.clip(idx, 0, peaks_s.size - 1)
        nearest_diff = np.minimum(
            np.abs(gt_sorted - peaks_s[idx_left]),
            np.abs(gt_sorted - peaks_s[idx_right]),
        )
        return float((nearest_diff <= tol_s).mean())

    per_unit_recall: dict[int, float] = {}
    gt_unit_to_shank: dict[int, int] = {}
    for uid, gt_times in gt_units.items():
        if gt_times.size == 0:
            continue
        gt_sorted = np.sort(gt_times.astype(np.float64))
        per_shank = {
            sg: _recall_against(v2_times_s_by_shank[sg], gt_sorted)
            for sg in sorted_sgs
        }
        best_sg = max(per_shank, key=lambda sg: per_shank[sg])
        per_unit_recall[uid] = per_shank[best_sg]
        gt_unit_to_shank[uid] = best_sg

    n_planted = len(per_unit_recall)
    # Non-vacuity check FIRST: with n_planted == 0 the recall_values
    # array is empty and ``recall_values.mean()`` would raise inside
    # the summary string, masking the intended "no planted GT units"
    # error. Fail with a clear message before any aggregate stats.
    assert n_planted >= 1, (
        f"{session_label} clusterless GT: no planted units recovered "
        "from sidecar (per_unit_recall is empty -- either the fixture "
        "has zero GT units or every unit has zero spike times)."
    )
    recall_values = np.array(list(per_unit_recall.values()))
    shank_distribution = {
        sg: sum(1 for s in gt_unit_to_shank.values() if s == sg)
        for sg in sorted_sgs
    }
    summary = (
        f"\n[Clusterless vs {session_label} GT] "
        f"n_planted={n_planted}, "
        f"n_v2_peaks_total={v2_times_s.size}, "
        f"units_per_shank={shank_distribution}\n"
        f"  shank-aware time-recall (±0.4 ms, best-shank-per-unit): "
        f"mean={recall_values.mean():.3f} "
        f"median={float(np.median(recall_values)):.3f} "
        f"min={recall_values.min():.3f} max={recall_values.max():.3f}"
    )
    print(summary)

    # Threshold: mean recall >= 0.80 across planted units. Loosely
    # justifiable: well-isolated spikes should always be detected
    # by a detect_threshold of 5 (a MAD multiplier, not 5σ); missing
    # some is expected for near-threshold planted units (especially on
    # the sparse-coverage tetrode).
    assert recall_values.mean() >= 0.80, (
        f"Clusterless mean time-recall {recall_values.mean():.3f} < 0.80 "
        f"on {session_label}; v2 detector missing planted spikes.{summary}"
    )

    # Per-fixture calibrated false-positive bound: clusterless emits
    # unsorted peaks (not units), so SI's precision metric doesn't
    # apply, but a regression that emits substantially more peaks than
    # the planted population must not pass this gate by virtue of high
    # recall alone. Bounds are set with ~2x headroom over empirical
    # observation so SI minor-version drift does not cause flake, but
    # tight enough to flag real false-positive regressions (the
    # previous loose 10x bound would only catch the historical 1,400x
    # ``noise_levels=[1.0]`` explosion, missing smaller regressions).
    #
    # Observed v2_peaks/planted_spike ratios (full GT gate runs):
    #   * polymer_60s  -- ~1.5x (4 shanks aggregated)
    #   * tetrode_60s  -- ~1.4x (single shank, 5 planted)
    # Bound is ``3 x observed`` so legitimate background MEArec
    # activity + adjacent-channel duplicates still pass.
    extras_ratio_max = {
        "polymer_60s": 3.0,
        "tetrode_60s": 3.0,
    }.get(session_label)
    if extras_ratio_max is None:
        # New fixture without committed calibration evidence; fall
        # back to the smoke-guard 10x bound and surface the gap.
        extras_ratio_max = 10.0
        print(
            f"WARNING: no per-fixture extras-ratio calibration for "
            f"{session_label!r}; falling back to 10x. Add a "
            f"committed observed ratio + 2-3x bound to extras_ratio_max."
        )
    total_planted_spikes = int(sum(t.size for t in gt_units.values() if t.size))
    if total_planted_spikes > 0:
        extras_ratio = v2_times_s.size / total_planted_spikes
        assert extras_ratio <= extras_ratio_max, (
            f"Clusterless on {session_label} emitted "
            f"{v2_times_s.size} peaks vs {total_planted_spikes} "
            f"planted spikes (ratio={extras_ratio:.2f}, > "
            f"{extras_ratio_max:.1f}x calibrated bound). "
            f"Recall is met but the detector is firing well above "
            f"the planted population -- possible noise_levels / "
            f"threshold regression.{summary}"
        )
