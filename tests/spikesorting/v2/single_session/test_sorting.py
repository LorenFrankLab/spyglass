"""Sorting.make, get_sorting, analyzer, and sorter tests."""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._core.recipe_catalog import (
    CORTEX_DISPLAY_WAVEFORMS,
)
from tests.spikesorting.v2._ingest_helpers import _clean_session_v2
from tests.spikesorting.v2._sorter_stub import plant_sorter
from tests.spikesorting.v2.single_session._helpers import _build_synthetic_rec

# The 'default' preprocessing recipe maps to the cortex display recipe,
# which keys the analyzer cache folder.
_DISPLAY = CORTEX_DISPLAY_WAVEFORMS

# Cap 500 forces ``random_spikes`` to subsample (>500 spikes/unit), so the
# seed is observable.
_CAP500_DISPLAY = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 500,
    "whiten": False,
    "sparsity": {},
    "purpose": "display",
    "schema_version": 1,
}

# The shipped cortex display recipe; build_analyzer requires an explicit one.
_DISPLAY_PARAMS = {**_CAP500_DISPLAY, "max_spikes_per_unit": 20000}


@pytest.mark.slow
def test_sorting_populates_with_mountainsort5(populated_recording):
    """``Sorting.make`` runs MountainSort5 on the smoke fixture and writes
    a fresh Units NWB + SortingAnalyzer folder. ``Sorting.Unit`` is
    populated with the peak electrode + amplitude per unit."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    ArtifactDetectionParameters.insert_default()

    # No-op artifact detection so the sort uses the full recording.
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )

    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    assert row["n_units"] > 0
    from pathlib import Path

    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path

    # Folder keyed by (sorting_id, display recipe), not a column.
    analyzer_folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    assert isinstance(analyzer_folder, Path)
    assert analyzer_folder.exists()

    n_unit_rows = len(Sorting.Unit & sort_pk)
    assert n_unit_rows == row["n_units"]

    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    sg_id = int(
        (RecordingSelection & populated_recording).fetch1("sort_group_id")
    )
    unit_electrodes = (Sorting.Unit & sort_pk).fetch("electrode_id")
    sort_group_electrodes = set(
        (
            SortGroupV2.SortGroupElectrode
            & populated_recording
            & {"sort_group_id": sg_id}
        ).fetch("electrode_id")
    )
    assert set(unit_electrodes).issubset(sort_group_electrodes)

    # make() takes abs() of peak amplitudes.
    amplitudes = (Sorting.Unit & sort_pk).fetch("peak_amplitude_uv")
    assert (amplitudes > 0).all()


@pytest.mark.slow
def test_sorting_get_sorting_round_trips(populated_sorting):
    """``Sorting.get_sorting`` returns a v1-style frame-relative sorting.

    - unit_ids match the row's ``n_units``;
    - ``return_times=False`` yields recording frame indices whose
      ``timestamps[frame]`` equals the stored absolute spike times;
    - ``return_times=True`` yields frame-relative seconds, not wall-clock.
    """
    import numpy as np

    from spyglass.spikesorting.v2.recording import (
        Recording,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    row = (Sorting & populated_sorting).fetch1()
    si_sorting = Sorting().get_sorting(populated_sorting)
    assert len(si_sorting.unit_ids) == row["n_units"]

    # ``recording_id`` is a non-PK FK on ``SortingSelection.RecordingSource``,
    # so ``fetch1("KEY")`` would omit it.
    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    rec = Recording().get_recording({"recording_id": recording_id})
    timestamps = rec.get_times()
    n_samples = len(timestamps)
    fs = rec.get_sampling_frequency()

    # as_dataframe carries the stored absolute spike times.
    df = Sorting().get_sorting(populated_sorting, as_dataframe=True)

    for uid in si_sorting.unit_ids:
        frames = si_sorting.get_unit_spike_train(
            unit_id=uid, return_times=False
        )
        if len(frames) == 0:
            continue
        frames = frames.astype(int)
        assert frames.min() >= 0
        assert frames.max() < n_samples
        abs_times = np.sort(np.asarray(df.loc[int(uid), "spike_times"]))
        recovered = np.sort(timestamps[frames])
        # Absolute tolerance: rtol on large times would allow ~ms slack.
        assert np.allclose(recovered, abs_times, rtol=0.0, atol=1.0 / fs)
        # Frame-relative (t_start=0), as in v1.
        rel = si_sorting.get_unit_spike_train(unit_id=uid, return_times=True)
        assert np.allclose(
            np.sort(rel), np.sort(frames / fs), rtol=0.0, atol=1.0 / fs
        )


@pytest.mark.slow
def test_sorting_make_fetch_resolves_artifact_obs_intervals(populated_sorting):
    """``make_fetch`` derives obs_intervals from the ArtifactDetectionSource part.

    ``make_fetch`` must attach the resolved ``artifact_detection_id`` to
    ``sel_row``; otherwise artifact-backed sorts skip masking and write
    ``obs_intervals=None``.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    assert artifact_detection_id is not None, (
        "populated_sorting must be artifact-backed for this guard to be "
        "meaningful (expected one ArtifactDetectionSource part row)."
    )

    fetched = Sorting().make_fetch(populated_sorting)
    assert fetched.sel_row.get("artifact_detection_id") == artifact_detection_id
    assert fetched.obs_intervals is not None, (
        "make_fetch returned obs_intervals=None for an artifact-backed "
        "sort; the ArtifactDetectionSource artifact_detection_id was not resolved, so "
        "artifact masking is silently skipped."
    )


@pytest.mark.slow
def test_masked_sort_snr_matches_unmasked_within_tolerance(
    populated_sorting, curation_evaluation_defaults
):
    """Masking ~10% of the recording leaves per-unit SNR within 5%.

    MAD over the zero-filled mask would inflate SNR by ~25% here; the
    statistics-span noise estimate matches the unmasked noise.
    """
    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    unmasked_pk = dict(populated_sorting)
    recording_id = SortingSelection.resolve_source(unmasked_pk).key[
        "recording_id"
    ]
    nwb_file_name, interval_list_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name", "interval_list_name")
    t0 = float(
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")[0][0]
    )
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_id,
            "artifact_detection_params_name": "none",
            # Each kept side stays above the "none" recipe's 1 s min length.
            "manual_excluded_times": [[t0 + 1.8, t0 + 2.2]],
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sel = (SortingSelection & unmasked_pk).fetch1()
    masked_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sel["sorter"],
            "sorter_params_name": sel["sorter_params_name"],
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & masked_pk).super_delete(warn=False)
    Sorting.populate(masked_pk, reserve_jobs=False)

    n_samples = (
        Recording()
        .get_recording({"recording_id": recording_id})
        .get_num_samples()
    )
    spans = Sorting().get_statistics_spans(masked_pk)
    masked_fraction = 1 - sum(b - a for a, b in spans) / n_samples
    assert 0.08 < masked_fraction < 0.12, masked_fraction

    def _snr(sort_pk):
        clear_curations_for(sort_pk)
        root = CurationV2.insert_curation(sorting_key=sort_pk)
        eval_pk = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(eval_pk, reserve_jobs=False)
        return CurationEvaluation.get_metrics(eval_pk)["snr"]

    try:
        unmasked_snr = _snr(unmasked_pk)
        masked_snr = _snr(masked_pk)
        unmasked_sorting = Sorting().get_sorting(unmasked_pk)
        masked_sorting = Sorting().get_sorting(masked_pk)
        fs = unmasked_sorting.get_sampling_frequency()
        tolerance = int(round(0.0005 * fs))
        compared = 0
        for masked_unit in masked_sorting.unit_ids:
            train = masked_sorting.get_unit_spike_train(masked_unit)
            best, best_agreement = None, 0.0
            for unit in unmasked_sorting.unit_ids:
                ref = unmasked_sorting.get_unit_spike_train(unit)
                idx = np.clip(np.searchsorted(ref, train), 1, len(ref) - 1)
                nearest = np.minimum(
                    np.abs(ref[idx] - train), np.abs(ref[idx - 1] - train)
                )
                agreement = float(np.mean(nearest <= tolerance))
                if agreement > best_agreement:
                    best, best_agreement = unit, agreement
            if best_agreement < 0.8:
                continue
            compared += 1
            ratio = float(masked_snr.loc[int(masked_unit)]) / float(
                unmasked_snr.loc[int(best)]
            )
            assert abs(ratio - 1) < 0.05, (masked_unit, best, ratio)
        assert compared >= 1, "no masked unit matched an unmasked unit"
    finally:
        clear_curations_for(unmasked_pk)
        clear_curations_for(masked_pk)
        (Sorting & masked_pk).super_delete(warn=False)


@pytest.mark.slow
def test_sorting_get_analyzer_loads_folder(populated_sorting):
    """``Sorting.get_analyzer`` loads the SortingAnalyzer from the folder."""
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer = Sorting().get_analyzer(populated_sorting)
    assert isinstance(analyzer, si.SortingAnalyzer)
    assert analyzer.has_extension("templates")


@pytest.mark.slow
def test_clusterless_thresholder_end_to_end(polymer_smoke_session):
    """Clusterless-thresholder finds peaks and round-trips through the
    pipeline.

    The default 100 uV row finds no peaks on the smoke fixture, so a lower
    threshold row is inserted.
    """
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

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # Shared with baseline capture and the parity test.
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    custom_params_name = SMOKE_CLUSTERLESS_PARAM_NAME
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": custom_params_name,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 4,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
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
            "sorter_params_name": custom_params_name,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    assert row["n_units"] == 1
    assert len(Sorting.Unit & sort_pk) == 1
    unit_row = (Sorting.Unit & sort_pk).fetch1()
    assert unit_row["unit_id"] == 0

    # ~150-200 planted spikes; ``> 0`` would accept a single noise spike.
    assert unit_row["n_spikes"] >= 20, (
        f"Clusterless found only {unit_row['n_spikes']} peaks on the "
        "smoke fixture; expected at least 20 given the planted firing "
        "rates and the detect_threshold of 5 (a MAD multiplier, not "
        "µV). A buggy detector that returns a single false positive "
        "would pass `n_spikes > 0`."
    )
    # No amplitude-vs-threshold check: with threshold_unit="mad" the
    # threshold is 5 x each channel's noise MAD, not microvolts.
    assert unit_row["peak_amplitude_uv"] > 0, (
        f"Clusterless reported non-positive peak_amplitude_uv="
        f"{unit_row['peak_amplitude_uv']}; a detector that returns "
        "zero or negative amplitudes is broken (sanity floor)."
    )
    import math

    assert math.isfinite(unit_row["peak_amplitude_uv"])


# ---------- Sorting.make rollback file cleanup ---------------------------


@pytest.mark.slow
def test_sorting_make_rollback_cleans_units_nwb(
    polymer_smoke_session, monkeypatch
):
    """A ``Sorting.make`` failure after ``AnalysisNwbfile.add()`` unlinks
    the staged units NWB.
    """
    import pathlib

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
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
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    # Upstream cascades can leave a cache from an earlier committed result.
    # A failed new attempt must neither replace nor remove that folder.
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_folder_storage_fingerprint,
        analyzer_path,
    )

    analyzer_folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    cache_before = (
        analyzer_folder_storage_fingerprint(analyzer_folder)
        if analyzer_folder.exists()
        else None
    )

    from spyglass.settings import analysis_dir as ad

    analysis_dir = pathlib.Path(ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    # Raise after the file is written. A plain function binds ``self``,
    # unlike the original staticmethod.
    unit_part_calls = []

    def _broken_unit_part(self, unit_rows):
        unit_part_calls.append(unit_rows)
        raise RuntimeError("simulated unit-part failure")

    monkeypatch.setattr(Sorting, "_populate_unit_part", _broken_unit_part)

    # suppress_errors swallows the exception; check the after-state.
    Sorting.populate(sort_pk, reserve_jobs=False, suppress_errors=True)
    assert len(unit_part_calls) == 1, "injected insert failure was not reached"
    assert unit_part_calls[0], "the rollback fixture must produce unit rows"
    assert len(Sorting & sort_pk) == 0, (
        "Sorting row should not be present after rollback; "
        "the transaction was supposed to roll back."
    )

    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"Sorting.make rollback left orphan analysis files: {new_files}. "
        "A failed populate must unlink the staged file when the "
        "transaction rolls back."
    )

    # A peer's canonical cache survives the failed attempt.
    cache_after = (
        analyzer_folder_storage_fingerprint(analyzer_folder)
        if analyzer_folder.exists()
        else None
    )
    assert cache_after == cache_before
    assert not list(
        analyzer_folder.parent.glob(f".{sort_pk['sorting_id']}__*.analyzer")
    )
    (SortingSelection & sort_pk).super_delete(warn=False)


@pytest.mark.slow
def test_changed_second_fetch_leaves_no_staged_sort_outputs(
    polymer_smoke_session, monkeypatch
):
    """A sort DataJoint refuses after compute leaves no NWB, build or lock.

    ``make_compute`` stages the units NWB and a private analyzer build
    (holding its ownership lock) outside the insert transaction. When the
    in-transaction ``make_fetch`` differs, DataJoint raises before
    ``make_insert`` runs; the NWB, the build folder and its lock file must
    all go, and no canonical analyzer may be published.
    """
    import numpy as np
    import spikeinterface as si
    from datajoint.errors import DataJointError

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_path,
        remove_analyzer_cache,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._tripart_helpers import (
        assert_no_staged_analysis_files,
        change_second_fetch,
        record_created_analysis_files,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
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
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)
    # Deterministic sorting_id; ``super_delete`` skips cache cleanup.
    remove_analyzer_cache(sort_pk["sorting_id"], missing_ok=True)
    canonical = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    assert not canonical.exists()

    def _plant(sorter, sorter_params, recording, sorting_id, **kwargs):
        # A known two-unit sorting stands in for the sorter run.
        frames = np.arange(1_000, 40_000, 1_500, dtype=np.int64)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[frames],
            labels_list=[np.arange(len(frames)) % 2],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _plant)
    created = record_created_analysis_files(monkeypatch)
    change_second_fetch(monkeypatch, Sorting)
    try:
        with pytest.raises(DataJointError, match="Referential integrity"):
            Sorting.populate(sort_pk, reserve_jobs=False)
        assert not (Sorting & sort_pk)
        assert_no_staged_analysis_files(created)
        assert not canonical.exists(), "a refused sort published its analyzer"
        leftovers = sorted(
            path.name
            for path in canonical.parent.glob(f".{sort_pk['sorting_id']}__*")
        )
        assert not leftovers, f"analyzer staging left behind: {leftovers}"
    finally:
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)


def test_run_si_sorter_restores_global_job_kwargs(dj_conn, monkeypatch):
    """``run_si_sorter`` leaves SI's global job kwargs byte-identical to
    their pre-sort state.

    ``set_global_job_kwargs`` updates rather than replaces, so an added key
    such as ``chunk_size`` would leak into later populates.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._sorting.dispatch import run_si_sorter

    si.reset_global_job_kwargs()
    before = dict(si.get_global_job_kwargs())
    assert "chunk_size" not in before  # the key we expect could leak

    # A real sorting: ``NumpySorting.from_sorting`` needs ``to_spike_vector``.
    stub_sorting = si.generate_sorting(
        num_units=2, sampling_frequency=30000.0, durations=[0.1]
    )
    monkeypatch.setattr(sis, "run_sorter", lambda **kw: stub_sorting)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    out = run_si_sorter(
        sorter="mountainsort5",
        sorter_params={"whiten": False},
        recording=rec,
        sorting_id="r3_test",
        job_kwargs={"n_jobs": 1, "chunk_size": 1000},
    )
    # The return is the materialized in-memory sorting, not the stub itself.
    assert isinstance(out, si.NumpySorting)
    assert out.get_num_units() == 2

    after = dict(si.get_global_job_kwargs())
    assert after == before, (
        f"global job kwargs leaked across the sort: before={before}, "
        f"after={after}"
    )
    assert "chunk_size" not in after
    si.reset_global_job_kwargs()


def test_clusterless_detect_peaks_strips_random_seed(dj_conn, monkeypatch):
    """``run_clusterless_thresholder`` strips ``random_seed`` from the
    job kwargs before calling ``detect_peaks``.

    SI's ``fix_job_kwargs`` rejects the Spyglass-only ``random_seed``.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["job_kwargs"] = job_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    run_clusterless_thresholder(
        sorter_params={
            "detect_threshold": 5.0,
            "threshold_unit": "uv",
            "noise_levels": [1.0],
        },
        recording=rec,
        job_kwargs={"random_seed": 7, "n_jobs": 1},
    )

    jk = captured["job_kwargs"]
    assert jk is not None, "detect_peaks was not called"
    assert (
        "random_seed" not in jk
    ), f"random_seed leaked into detect_peaks job kwargs: {jk}"
    assert jk.get("n_jobs") == 1  # other kwargs preserved


@pytest.mark.slow
def test_clusterless_detect_peaks_strips_threshold_unit(dj_conn, monkeypatch):
    """``run_clusterless_thresholder`` strips ``threshold_unit`` before
    calling ``detect_peaks``.

    ``threshold_unit`` is Spyglass-only; SI would raise on the unknown
    keyword. Real detector knobs survive.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["method_kwargs"] = method_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    run_clusterless_thresholder(
        sorter_params={
            "detect_threshold": 5.0,
            "threshold_unit": "uv",
            "noise_levels": [1.0],
        },
        recording=rec,
        job_kwargs={"n_jobs": 1},
    )

    mk = captured["method_kwargs"]
    assert mk is not None, "detect_peaks was not called"
    assert (
        "threshold_unit" not in mk
    ), f"threshold_unit leaked into detect_peaks method kwargs: {mk}"
    assert mk.get("detect_threshold") == 5.0


def _stub_recording_with_2d_probe():
    """Two-channel recording with a planar probe for the tests below.

    Passes ``build_analyzer``'s pre-build checks: distinct, already-2D
    contact positions.
    """
    import numpy as np
    import spikeinterface as si
    from probeinterface import generate_linear_probe

    recording = si.NumpyRecording(
        np.zeros((100, 2), dtype="float32"), sampling_frequency=30_000.0
    )
    probe = generate_linear_probe(num_elec=2, ypitch=20)
    probe.set_device_channel_indices(np.arange(2))
    return recording.set_probe(probe)


def test_build_analyzer_strips_random_seed(dj_conn, monkeypatch, tmp_path):
    """``build_analyzer`` strips ``random_seed`` before
    ``SortingAnalyzer.compute``.

    SI's ``compute`` rejects the Spyglass-only ``random_seed``.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def has_extension(self, name):
            return False

    def _fake_create(**kwargs):
        kwargs["folder"].mkdir(exist_ok=True)  # SI creates the store
        return _FakeAnalyzer()

    monkeypatch.setattr(si, "create_sorting_analyzer", _fake_create)

    class _FakeSorting:
        def get_num_units(self):
            return 2

    build_analyzer(
        _FakeSorting(),
        _stub_recording_with_2d_probe(),
        {"sorting_id": "test-sorting-id"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"random_seed": 7, "n_jobs": 1},
        analyzer_folder=tmp_path / "analyzer",
        waveform_params=_DISPLAY_PARAMS,
    )

    jk = captured["kwargs"]
    assert (
        "random_seed" not in jk
    ), f"random_seed leaked into analyzer.compute kwargs: {jk}"
    assert jk.get("n_jobs") == 1  # other job kwargs preserved


def test_build_analyzer_compute_args(dj_conn, monkeypatch, tmp_path):
    """``build_analyzer`` requests the right extensions + analyzer kwargs.

    ``sparse=True`` and ``return_in_uV=True`` set the persisted peak
    channel and amplitude. Extensions are checked as a required subset.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["compute_args"] = args
            captured["compute_kwargs"] = kwargs

        def has_extension(self, name):
            return False

    def _fake_create(**kwargs):
        captured["create_kwargs"] = kwargs
        kwargs["folder"].mkdir(exist_ok=True)  # SI creates the store
        return _FakeAnalyzer()

    monkeypatch.setattr(si, "create_sorting_analyzer", _fake_create)

    class _FakeSorting:
        def get_num_units(self):
            return 2

    build_analyzer(
        _FakeSorting(),
        _stub_recording_with_2d_probe(),
        {"sorting_id": "test-sorting-id"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"random_seed": 3, "n_jobs": 1},
        analyzer_folder=tmp_path / "analyzer",
        waveform_params=_DISPLAY_PARAMS,
    )

    ck = captured["create_kwargs"]
    assert ck.get("sparse") is True, f"expected sparse=True, got {ck}"
    assert (
        ck.get("return_in_uV") is True
    ), f"expected return_in_uV=True, got {ck}"
    assert ck.get("format") == "binary_folder"

    # Required subset of the ``compute`` extensions (positional first arg).
    ca = captured["compute_args"]
    assert ca, "analyzer.compute was called with no positional extension list"
    required_extensions = {
        "random_spikes",
        "noise_levels",
        "templates",
        "waveforms",
    }
    assert required_extensions <= set(ca[0]), (
        f"missing required extension(s) {required_extensions - set(ca[0])}; "
        f"got {ca[0]}"
    )

    # Specific values, not dict equality, so added SI params pass.
    ext_params = captured["compute_kwargs"]["extension_params"]
    assert (
        ext_params["random_spikes"]["seed"] == 3
    ), "random_spikes seed must honor the job_kwargs random_seed override"
    assert ext_params["random_spikes"]["max_spikes_per_unit"] == 20000
    assert ext_params["waveforms"]["ms_before"] == 1.0
    assert ext_params["waveforms"]["ms_after"] == 2.0
    assert "random_seed" not in captured["compute_kwargs"]
    # Empty ``sparsity`` -> explicit radius / 100 um; polarity from the sorter.
    assert ck.get("method") == "radius"
    assert ck.get("radius_um") == 100.0
    assert ck.get("peak_sign") == "neg"
    assert ck.get("num_spikes_for_sparsity") == 100

    # best_channels forwards its settings; dense passes no sparsity kwargs.
    for sparsity, expected in (
        (
            {"method": "best_channels", "num_channels": 6},
            {
                "sparse": True,
                "method": "best_channels",
                "num_channels": 6,
                # Polarity comes from the sorter (none set: neg).
                "peak_sign": "neg",
            },
        ),
        ({"method": "dense"}, {"sparse": False}),
    ):
        captured.clear()
        build_analyzer(
            _FakeSorting(),
            _stub_recording_with_2d_probe(),
            {"sorting_id": "test-sorting-id"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={"random_seed": 3, "n_jobs": 1},
            analyzer_folder=tmp_path / "analyzer",
            waveform_params={**_DISPLAY_PARAMS, "sparsity": sparsity},
        )
        ck = captured["create_kwargs"]
        for key, value in expected.items():
            assert ck.get(key) == value, (sparsity, key, ck)
        if sparsity["method"] == "dense":
            assert "radius_um" not in ck and "method" not in ck
        else:
            assert "radius_um" not in ck


@pytest.mark.slow
def test_analyzer_rebuild_is_seeded_reproducible(
    dj_conn, monkeypatch, tmp_path
):
    """``build_analyzer`` seeds ``random_spikes`` so rebuilds are stable.

    SI 0.104's ``random_spikes`` defaults to ``seed=None``. Every unit here
    fires >500 spikes so the 500-spike cap subsamples and the seed matters.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import (
        generate_ground_truth_recording,
        template_tools,
    )

    from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

    recording, sorting = generate_ground_truth_recording(
        durations=[40.0],
        num_channels=8,
        num_units=3,
        generate_sorting_kwargs={
            "firing_rates": 20,
            "refractory_period_ms": 4.0,
        },
        seed=0,
    )
    totals = {
        int(u): len(sorting.get_unit_spike_train(unit_id=u))
        for u in sorting.unit_ids
    }
    assert min(totals.values()) > 500, (
        "fixture must exceed max_spikes_per_unit=500 so random_spikes "
        f"actually subsamples; got per-unit totals {totals}"
    )

    def _build_and_read(folder):
        build_analyzer(
            sorting,
            recording,
            {"sorting_id": "repro-test"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={"n_jobs": 1},
            analyzer_folder=folder,
            waveform_params=_CAP500_DISPLAY,
        )
        analyzer = si.load_sorting_analyzer(folder)
        selected = np.asarray(
            analyzer.get_extension("random_spikes").get_data()
        )
        peak_channels = template_tools.get_template_extremum_channel(
            analyzer, outputs="id"
        )
        peak_amplitudes = template_tools.get_template_extremum_amplitude(
            analyzer
        )
        return selected, peak_channels, peak_amplitudes

    sel1, chans1, amps1 = _build_and_read(tmp_path / "build_a.analyzer")
    sel2, chans2, amps2 = _build_and_read(tmp_path / "build_b.analyzer")

    # Subsampling fired, so the seed is observable.
    assert len(sel1) == 500 * len(totals)
    assert len(sel1) < sum(totals.values())

    np.testing.assert_array_equal(
        sel1, sel2, err_msg="random_spikes selection differs across builds"
    )
    assert chans1 == chans2
    assert amps1.keys() == amps2.keys()
    for unit_id in amps1:
        assert amps1[unit_id] == amps2[unit_id], (
            f"peak amplitude for unit {unit_id} is not reproducible across "
            "seeded analyzer rebuilds"
        )


@pytest.mark.slow
def test_analyzer_random_seed_override_is_honored(
    dj_conn, monkeypatch, tmp_path
):
    """``build_analyzer`` honors the per-row ``random_seed`` override.

    The ``random_spikes`` subsample reads ``job_kwargs["random_seed"]``,
    the same knob whitening uses: seed 7 is reproducible and differs from
    seed 0.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import generate_ground_truth_recording

    from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

    recording, sorting = generate_ground_truth_recording(
        durations=[40.0],
        num_channels=8,
        num_units=3,
        generate_sorting_kwargs={
            "firing_rates": 20,
            "refractory_period_ms": 4.0,
        },
        seed=0,
    )
    totals = sum(
        len(sorting.get_unit_spike_train(unit_id=u)) for u in sorting.unit_ids
    )

    def _selection_for_seed(folder, random_seed):
        build_analyzer(
            sorting,
            recording,
            {"sorting_id": "seed-override-test"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={"n_jobs": 1, "random_seed": random_seed},
            analyzer_folder=folder,
            waveform_params=_CAP500_DISPLAY,
        )
        analyzer = si.load_sorting_analyzer(folder)
        return np.asarray(analyzer.get_extension("random_spikes").get_data())

    seed7_a = _selection_for_seed(tmp_path / "seed7_a.analyzer", 7)
    seed7_b = _selection_for_seed(tmp_path / "seed7_b.analyzer", 7)
    seed0 = _selection_for_seed(tmp_path / "seed0.analyzer", 0)

    assert len(seed7_a) < totals
    np.testing.assert_array_equal(
        seed7_a,
        seed7_b,
        err_msg="random_seed=7 is not reproducible across builds",
    )
    assert not np.array_equal(seed7_a, seed0), (
        "random_seed override did not change the analyzer subsample -- the "
        "extension seed is ignoring job_kwargs['random_seed']."
    )


# Concat-source sorting is covered in ``test_session_group_concat.py``.


@pytest.mark.slow
def test_sorting_selection_artifact_id_none_is_distinct_identity(
    populated_sorting,
):
    """No-artifact and artifact-backed selections are distinct identities.

    A no-artifact request with the same recording, sorter and params gets
    a new ``sorting_id``, not the artifact-backed one.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    artifact_backed_pk = populated_sorting
    # Derived from the fixture row to track its sorter/params.
    backed_row = (SortingSelection & artifact_backed_pk).fetch1()
    rec_row = (SortingSelection.RecordingSource & artifact_backed_pk).fetch1()
    recording_id = rec_row["recording_id"]
    sorter = backed_row["sorter"]
    sorter_params_name = backed_row["sorter_params_name"]
    assert (
        SortingSelection.resolve_artifact_detection(artifact_backed_pk)
        is not None
    ), (
        "populated_sorting must yield an artifact-backed row (an "
        "ArtifactDetectionSource part row) for this test to be meaningful."
    )

    no_artifact_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": None,
        }
    )
    assert no_artifact_pk["sorting_id"] != artifact_backed_pk["sorting_id"], (
        "insert_selection with artifact_detection_id=None aliased onto the "
        "artifact-backed row; no-artifact must be a distinct identity, "
        "not match any artifact_detection_id."
    )
    assert SortingSelection.resolve_artifact_detection(no_artifact_pk) is None
    assert len(SortingSelection.ArtifactDetectionSource & no_artifact_pk) == 0
    # Idempotent.
    repeat_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": None,
        }
    )
    assert repeat_pk["sorting_id"] == no_artifact_pk["sorting_id"], (
        "Repeat insert_selection(artifact_detection_id=None) created a duplicate "
        "row instead of returning the existing no-artifact row."
    )


@pytest.mark.slow
def test_sorting_selection_artifact_detection_source_part_shape(
    populated_recording,
):
    """Artifact state lives on the ArtifactDetectionSource part, not a master FK.

    No-artifact selections have zero ``ArtifactDetectionSource`` rows,
    artifact-backed ones exactly one, and ``resolve_source`` still returns
    one recording source.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    sorter_kwargs = {
        "recording_id": populated_recording["recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }

    no_art_pk = SortingSelection.insert_selection(dict(sorter_kwargs))
    assert len(SortingSelection.ArtifactDetectionSource & no_art_pk) == 0
    assert SortingSelection.resolve_artifact_detection(no_art_pk) is None

    art_sort_pk = SortingSelection.insert_selection(
        {
            **sorter_kwargs,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    assert len(SortingSelection.ArtifactDetectionSource & art_sort_pk) == 1
    assert (
        SortingSelection.resolve_artifact_detection(art_sort_pk)
        == art_pk["artifact_detection_id"]
    )

    resolved = SortingSelection.resolve_source(art_sort_pk)
    assert resolved.kind == "recording"
    assert resolved.key == {"recording_id": populated_recording["recording_id"]}


# ---------- write_sorting_units_nwb zero-unit guard (sorting layer) ------


@pytest.mark.slow
def test_write_units_nwb_handles_zero_unit_sorter(populated_recording):
    """``write_sorting_units_nwb`` initializes an empty Units NWB
    when the sorter produces zero unit ids.

    Sorting-layer analog of
    ``test_curation_v2_stages_empty_units_nwb_on_zero_kept_units``.
    """
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2._storage.units_nwb import (
        write_sorting_units_nwb,
    )

    recording = Recording().get_recording(
        {"recording_id": populated_recording["recording_id"]}
    )
    empty_sorting = si.NumpySorting.from_unit_dict(
        units_dict_list=[{}],
        sampling_frequency=recording.get_sampling_frequency(),
    )

    from spyglass.spikesorting.v2.recording import RecordingSelection

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")

    from tests.spikesorting.v2._provenance_helpers import sorting_provenance

    analysis_file_name, units_object_id = write_sorting_units_nwb(
        sorting=empty_sorting,
        recording=recording,
        nwb_file_name=nwb_file_name,
        source_provenance=sorting_provenance(
            recording_id=str(populated_recording["recording_id"]),
            statistics_spans=[[0, recording.get_num_frames()]],
        ),
    )
    try:
        # An empty Units table, not ``nwbf.units = None``.
        assert isinstance(units_object_id, str)
        assert len(units_object_id) > 0

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            assert nwbf.units is not None
            assert (
                len(nwbf.units.id[:]) == 0
            ), f"Expected empty Units table; got {len(nwbf.units.id[:])} rows."
    finally:
        # Unregistered staged file (the caller registers it).
        import pathlib

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        if pathlib.Path(abs_path).exists():
            pathlib.Path(abs_path).unlink()


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_nwb_writes_obs_intervals_and_curation_label_placeholder(
    populated_sorting,
):
    """Pre-curation NWB carries per-unit ``obs_intervals`` and a
    ``curation_label="uncurated"`` scalar placeholder.

    Matches v1's pre-curation NWB; a list label would break v1-style
    equality checks.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    row = (Sorting & populated_sorting).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert len(df) > 0, "populated_sorting yielded zero units"
    assert (
        "obs_intervals" in df.columns
    ), "Units NWB missing per-unit obs_intervals column."
    # Non-empty ``(n_intervals, 2)`` per unit.
    for uid, obs in df["obs_intervals"].items():
        assert (
            obs is not None and len(obs) >= 1
        ), f"Unit {uid} has empty obs_intervals."
    assert (
        "curation_label" in df.columns
    ), "Units NWB missing curation_label placeholder column."
    for uid, lbl in df["curation_label"].items():
        assert lbl == "uncurated", (
            f"Unit {uid} has curation_label={lbl!r}; expected scalar "
            "string ``'uncurated'``."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_delete_removes_analyzer_folder(populated_sorting):
    """``Sorting.delete()`` cleans up the analyzer folder on disk.

    The folder (5-50 GB) is not tracked by DataJoint.
    """
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path

    folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
    # Missing folder fails rather than letting the check pass vacuously.
    assert folder.exists(), (
        f"precondition: analyzer_folder {folder} should exist after the "
        "populated_sorting fixture; the populate path did not write it."
    )

    (Sorting & populated_sorting).delete(safemode=False)
    assert not folder.exists(), (
        f"analyzer_folder {folder} still exists after "
        "Sorting.delete(); cleanup regression."
    )
