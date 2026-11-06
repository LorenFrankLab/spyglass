"""Ground-truth accuracy and v1-parity tests on MEArec fixtures."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
from tests.spikesorting.v2.single_session._helpers import (
    fixture_nwb_path,
    validate_v1_baseline_case,
)

# ---------- 60s MEArec ground-truth correctness gate ----------------------


@pytest.mark.slow
def test_mountainsort5_ground_truth_polymer_60s(polymer_60s_session):
    """MS5 on the 60s polymer fixture finds the planted units.

    Per-unit accuracy >= 0.7 for at least half of planted units, via
    ``compare_sorter_to_ground_truth`` against the sidecar GT units table.
    """
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )

    if not (SortGroupV2 & polymer_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Sort each shank; per-shank unit ids are made disjoint below.
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
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    # ``aggregate_units`` yields uint64 ids, which SI 0.104's Hungarian
    # matcher rejects (dtype kind 'i' or 'U' only); rename to signed ints.
    import numpy as np
    from spikeinterface import aggregate_units

    aggregated = aggregate_units(sortings_by_shank)
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    # Ground truth from the sidecar ``ground_truth`` units table;
    # ``ImportedSpikeSorting`` merge dispatch raises NotImplementedError.
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import Nwbfile
    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        gt_nwb = io.read()
        es = gt_nwb.acquisition["e-series"]
        # MEArec writes a fixed-rate series without timestamps.
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

    # Indexed by GT unit_id. accuracy = tp / (tp + fn + fp),
    # recall = tp / (tp + fn), precision = tp / (tp + fp).
    perf = comparison.get_performance(method="by_unit", output="pandas")
    accuracies = perf["accuracy"].values
    recall = perf["recall"].values
    precision = perf["precision"].values
    n_planted = len(accuracies)

    # Always logged so a failure shows the distribution.
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

    # The fixture plants 24 units; a half-failed generation must not pass.
    assert n_planted >= 12, (
        f"60s polymer fixture has only {n_planted} planted units; expected "
        f"around 24. Regenerate the fixture before trusting this gate."
        f"{summary}"
    )

    n_well_detected = int((accuracies >= 0.7).sum())
    threshold_half = n_planted // 2
    assert n_well_detected >= threshold_half, (
        f"MS5 on the 60s polymer fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"validation goal requires >= {threshold_half}."
        f"{summary}"
    )

    # Precision floor: spurious extra spikes must not pass the accuracy gate.
    well_detected_mask = accuracies >= 0.7
    well_detected_precision = precision[well_detected_mask]
    assert (well_detected_precision >= 0.5).all(), (
        f"Well-detected GT units have low precision: min="
        f"{well_detected_precision.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7.{summary}"
    )

    # Recall floor: accuracy is symmetric in fn and fp, so bound misses too.
    well_detected_recall = recall[well_detected_mask]
    assert (well_detected_recall >= 0.5).all(), (
        f"Well-detected GT units have low recall: min="
        f"{well_detected_recall.min():.3f}; expected >= 0.5 for every "
        f"unit with accuracy >= 0.7 (a unit can't be 'well detected' "
        f"if it's missing more than half its planted spikes).{summary}"
    )

    # Mean-accuracy floor: catches every unit dropping to barely passing.
    well_detected_accuracy = accuracies[well_detected_mask]
    assert well_detected_accuracy.mean() >= 0.85, (
        f"Mean accuracy of well-detected units is "
        f"{well_detected_accuracy.mean():.3f}; expected >= 0.85 "
        f"(MS5 typically produces ~0.95 on this fixture). A regression "
        f"that drove accuracies down while keeping enough above 0.7 to "
        f"pass the count gate would surface here.{summary}"
    )


# =========================================================================
# Real-data correctness gates; skip when fixtures are absent.
# =========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_mountainsort5_ground_truth_neuropixels_60s(neuropixels_60s_session):
    """MS5 on the 60s Neuropixels fixture finds the planted units.

    Accuracy >= 0.7 for at least three quarters of planted units, plus
    the polymer gate's precision / recall / mean-accuracy floors. Loosen
    only with calibration evidence in ``fixtures_manifest.json``. Skips
    when the fixture is not on disk.
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
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
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )
        Sorting.populate(sort_pk, reserve_jobs=False)
        sortings_by_shank.append(Sorting().get_sorting(sort_pk))

    aggregated = aggregate_units(sortings_by_shank)
    # The Hungarian matcher rejects uint64 unit ids.
    tested_sorting = aggregated.rename_units(
        np.arange(len(aggregated.unit_ids), dtype=np.int64)
    )

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
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

    assert n_planted >= 1, (
        f"Neuropixels 60s fixture has only {n_planted} planted units; "
        f"regenerate the fixture before trusting this gate.{summary}"
    )

    import math

    n_well_detected = int((accuracies >= 0.7).sum())
    threshold = max(1, math.ceil(n_planted * 0.75))
    assert n_well_detected >= threshold, (
        f"MS5 on the 60s Neuropixels fixture detected {n_well_detected} "
        f"of {n_planted} planted units at accuracy >= 0.7; "
        f"threshold requires >= {threshold} (3/4 of planted, "
        f"ceil-rounded so non-multiples of 4 don't quietly pass below 75 %)."
        f"{summary}"
    )

    # Secondary floors mirror the polymer gate.
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

    Manual / nightly only: every case skips unless a baseline env var
    below points at captured v1 baselines.

    Holds only for linear referencing. ``global_median`` diverges from v1
    by design (v2 filters before referencing); see
    ``test_recording_make_global_median_reference``.

    Every v1 spike must match a v2 spike within 1.5 samples; v2 may have
    bounded extras from SI ``detect_peaks`` drift (see
    ``extra_spike_ratio``).

    Input fingerprints (``nwb_sha256``, ``sort_group_electrode_ids``,
    ``bad_channel_by_electrode_id``, ``canonical_preproc_params``,
    ``canonical_artifact_params``, ``artifact_valid_times``,
    ``canonical_sorter_params``) must match the v1 baseline before spike
    times are compared, so input skew is not mistaken for output drift.

    The smoke row (``smoke_clusterless_5uv``) omits ``noise_levels`` so
    the threshold is a MAD multiplier on both sides; ``noise_levels=[1.0]``
    makes it raw uV (~1,400x more detections).

    Env vars:
      ``SPIKESORTING_V2_BASELINE_ROOT`` -> directory tree
          ``$ROOT/<fixture_stem>/clusterless/shank<N>/``
          containing ``baseline_v1_spike_times.pkl`` and
          ``baseline_v1_recording_meta.json``. When set, missing
          baselines FAIL unless the ``(fixture_stem, sorter,
          sort_group_id)`` triple is in ``EXPECTED_DEGENERATE_CASES``.

      ``SPIKESORTING_V2_BASELINE_DIR`` -> single-dir form for
          ``(mearec_polymer_smoke, 0)`` only; other cases skip.
    """
    import json
    import os
    import pickle
    from pathlib import Path as _Path

    import numpy as np

    from tests.spikesorting.v2._smoke_constants import EXPECTED_DEGENERATE_CASES

    # Read from the v2 fixtures dir; sha256 is checked against the meta.
    fixture_path = fixture_nwb_path(fixture_stem)
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
        # With ROOT set, a missing baseline skips only if the case dir has
        # an operator-written DEGENERATE_MARKER; otherwise FAIL.
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
            # To skip a degenerate case, add it to EXPECTED_DEGENERATE_CASES.
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

    validate_v1_baseline_case(
        meta, sorter="clusterless_thresholder", sort_group_id=sort_group_id
    )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    # Pre-insert the smoke sorter row as baseline_capture does, and map
    # v1 param names to v2 (custom names pass through).
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
        # Delete-then-insert: a stale row with noise_levels set would make
        # the threshold raw uV.
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
                # ``noise_levels`` omitted: SI uses per-channel MAD.
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
    # Use the baseline's artifact params so both sides sort the same data.
    artifact_name_meta = meta.get("artifact_param_name", "default")
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": artifact_name_meta,
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1/v2 input fingerprint check, before the sort ---
    import hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2._artifacts.naming import (
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
        # Stale baseline: FAIL when ROOT is set, SKIP under BASELINE_DIR.
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
    # ``(n_intervals, 2)`` rounded to 1 ms: v1 and v2 differ by one end
    # sample (see baseline_capture._compute_invariant_fingerprints).
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
        # An array, not a hash, so float drift between SI versions is
        # compared with isclose.
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

    # Via the merge dispatcher, the surface a v1 consumer uses.
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
    # Empty unit sets would make the loop below vacuous.
    assert len(v1_keys) >= 1, (
        "v1 baseline has zero units; the per-unit drift check "
        "below would pass vacuously. Regenerate "
        "baseline_v1_spike_times.pkl on a recording with above-"
        "threshold activity."
    )

    # Budgets cover two SI 0.99 -> 0.104 changes: the ``locally_exclusive``
    # rewrite (SpikeInterface PR 4341) adds ~25-30% near-threshold peaks next
    # to v1 peaks, and the ``noise_levels`` seed change (SpikeInterface PR
    # 3359) leaves <= 0.2% unmatched v1 spikes.
    threshold_samples = 1.5
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
        # An empty v2 array would make the searchsorted read IndexError.
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

        # v1 -> v2 nearest-neighbor distance.
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

        # v2 -> v1 distance; ``unmatched_v2`` is reported, not asserted.
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

        unmatched_v1_budget = max(2, int(0.01 * v1_arr.size))
        if unmatched_v1 > unmatched_v1_budget:
            failures.append(
                f"unit {uid}: {unmatched_v1}/{v1_arr.size} v1 spikes "
                f"lack a v2 match within {threshold_samples} samples "
                f"(budget {unmatched_v1_budget}; tol {tol_s:.6f}s). "
                f"max unmatched drift "
                f"{float(v1_to_v2_diff[~v1_matched].max() * fs):.2f} samples."
            )

        max_v2 = int(v1_arr.size * (1.0 + extra_spike_ratio)) + 5
        if v2_arr.size > max_v2:
            failures.append(
                f"unit {uid}: v2 detected {v2_arr.size} spikes vs v1's "
                f"{v1_arr.size}; excess > {extra_spike_ratio:.0%} budget "
                f"(allowed {max_v2}). Likely a sorter-parameter semantic "
                "mismatch (e.g. forwarding ``noise_levels=[1.0]`` so the "
                "threshold reads in raw uV instead of MAD multiples)."
            )

    # Printed even on pass to record per-unit unmatched counts.
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

    MS4 has no seed parameter, so the contract is aggregate bands from
    :data:`tests.spikesorting.v2._smoke_constants.MS4_BANDS`, not per-spike:

    * ``|v2_n_units - v1_n_units| ≤ max(10% of v1_n_units, 2)``
    * ``|v2_median_fr - v1_median_fr| / v1_median_fr ≤ 10%`` (rel)

    Calibration is in :data:`MS4_VARIANCE_TABLE`.

    Env vars / SKIP semantics: same as :func:`test_v2_real_data_v1_parity`.
    Native MS4 is required by the v2 extra; a missing installation fails.
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

    # v2-side MS4 install check; v1 is checked at capture.
    import spikeinterface.sorters as ss

    assert "mountainsort4" in ss.installed_sorters()

    fixture_path = fixture_nwb_path(fixture_stem)
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

    validate_v1_baseline_case(
        meta, sorter="mountainsort4", sort_group_id=sort_group_id
    )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()

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
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": artifact_name_meta,
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    # --- v1↔v2 invariant fingerprint check (same shape as clusterless) ---
    import hashlib

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v2._artifacts.naming import (
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
            f"field(s) {missing_fp}; re-capture the baseline with the "
            "current baseline_capture to record the fingerprint fields."
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

    # Firing-rate denominator is the sorted interval, not v1's last spike
    # time; the fingerprint check already matched both sides' intervals.
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


# Polymer only: MS4 cannot resolve the synthetic 4-channel tetrode units.
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

    At least half of planted units at accuracy >= 0.5 (looser than MS5's
    0.7; MS4 is less accurate).
    """
    import numpy as np
    import pynwb
    import spikeinterface as si
    from spikeinterface import aggregate_units
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
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
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
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

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
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

    Clusterless emits peaks, not units, so per planted unit the test takes
    the best single-shank recall within 0.4 ms (SI's GT default). Mean
    recall must be >= 0.80.

    Parametrization:
      * ``polymer_60s``: 4 shanks, planted units spread across them
      * ``tetrode_60s``: 1 shank, 4 channels
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
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
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    if not (LabTeam & {"team_name": "v2_test_team"}):
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
        )

    # Delete-then-insert the smoke row, as in the parity test.
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

    # Peak times kept per shank for single-shank matching.
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
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
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
        # One synthetic unit per shank; concatenate in case of more.
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
    v2_times_s = np.sort(
        np.concatenate([a for a in v2_times_s_by_shank.values() if a.size])
    ).astype(np.float64)
    assert sampling_frequency is not None and sampling_frequency > 0

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
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

    # Best shank, not nearest-by-soma: soma routing mis-assigned 18/24
    # polymer cells. All-shank matching would inflate recall ~20 points.
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
    # Checked first: an empty mean would raise inside the summary string.
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

    # Near-threshold units may be missed, especially on the tetrode.
    assert recall_values.mean() >= 0.80, (
        f"Clusterless mean time-recall {recall_values.mean():.3f} < 0.80 "
        f"on {session_label}; v2 detector missing planted spikes.{summary}"
    )

    # False-positive bound, ~2x over observed peaks/planted ratios
    # (polymer ~1.5x, tetrode ~1.4x), so high recall alone cannot pass.
    extras_ratio_max = {
        "polymer_60s": 3.0,
        "tetrode_60s": 3.0,
    }.get(session_label)
    if extras_ratio_max is None:
        # Uncalibrated fixture: loose 10x bound.
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


@pytest.mark.unit
def test_parity_fixture_locator_uses_shared_fixture_package():
    """Moving the test module cannot turn available fixture inputs into skips."""
    from pathlib import Path

    import tests.spikesorting.v2.fixtures as fixture_package

    roots = [Path(path).resolve() for path in fixture_package.__path__]
    assert len(roots) == 1
    assert (
        fixture_nwb_path("mearec_polymer_smoke")
        == roots[0] / "mearec_polymer_smoke.nwb"
    )
    assert (roots[0] / "fixtures_manifest.json").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("sorter", ["clusterless_thresholder", "mountainsort4"])
@pytest.mark.parametrize(
    "bad_field, bad_value",
    [("sorter", "other"), ("sort_group_id", 0), ("sort_group_id", True)],
)
def test_v1_baseline_case_rejects_wrong_sorter_or_shank(
    sorter, bad_field, bad_value
):
    meta = {"sorter": sorter, "sort_group_id": 1}
    validate_v1_baseline_case(meta, sorter=sorter, sort_group_id=1)
    meta[bad_field] = bad_value
    with pytest.raises(
        ValueError, match="v1 baseline must describe.*sort_group_id=1"
    ):
        validate_v1_baseline_case(meta, sorter=sorter, sort_group_id=1)
