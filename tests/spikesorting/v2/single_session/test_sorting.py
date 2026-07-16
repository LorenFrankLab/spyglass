"""Sorting.make, get_sorting, analyzer, and sorter tests for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._recipe_catalog import CORTEX_DISPLAY_WAVEFORMS
from tests.spikesorting.v2._ingest_helpers import _clean_session_v2
from tests.spikesorting.v2.single_session._helpers import _build_synthetic_rec

# These sorts use the 'default' preprocessing recipe, which is not in the
# region map -> the wide cortex display recipe; the analyzer cache folder is
# keyed by that name.
_DISPLAY = CORTEX_DISPLAY_WAVEFORMS

# A cap-500 display recipe for the seeded-subsample tests: the synthetic
# fixtures have >500 spikes/unit, so cap=500 forces ``random_spikes`` to
# actually subsample (the schema-default 20000 would select every spike and
# the seed would be a no-op, making those reproducibility tests vacuous).
_CAP500_DISPLAY = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 500,
    "whiten": False,
    "purpose": "display",
    "schema_version": 1,
}

# The shipped cortex display recipe (1.0/2.0 ms, 20000 spikes) -- build_analyzer
# requires a resolved recipe (it never picks a default), so the fake-analyzer
# build tests that capture compute kwargs pass this explicitly.
_DISPLAY_PARAMS = {**_CAP500_DISPLAY, "max_spikes_per_unit": 20000}


@pytest.mark.slow
def test_sorting_populates_with_mountainsort5(populated_recording):
    """``Sorting.make`` runs MountainSort5 on the smoke fixture and writes
    a fresh Units NWB + SortingAnalyzer folder. ``Sorting.Unit`` is
    populated with the peak electrode + amplitude per unit."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetection,
        ArtifactDetectionSelection,
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

    # Ensure a no-op artifact detection is in place so the sort uses
    # the full recording.
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": ("franklab_30khz_ms5_2026_06"),
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )

    # Clear any prior populate.
    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    assert row["n_units"] > 0
    from pathlib import Path

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    # The analyzer folder is no longer a column; resolve it from
    # (sorting_id, display recipe name).
    analyzer_folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    assert isinstance(analyzer_folder, Path)
    assert analyzer_folder.exists()

    n_unit_rows = len(Sorting.Unit & sort_pk)
    assert n_unit_rows == row["n_units"]

    # Every Sorting.Unit row's peak electrode_id must belong to the
    # sort group of the upstream Recording.
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

    # Peak amplitudes are positive (we abs() them in make).
    amplitudes = (Sorting.Unit & sort_pk).fetch("peak_amplitude_uv")
    assert (amplitudes > 0).all()


@pytest.mark.slow
def test_sorting_get_sorting_round_trips(populated_sorting):
    """``Sorting.get_sorting`` returns a v1-style frame-relative sorting.

    ``get_sorting`` returns a ``NumpySorting`` (segment frames,
    ``t_start=0``), matching v1's ``NumpySorting.from_unit_dict`` shape:

    - unit_ids match the row's ``n_units``;
    - ``return_times=False`` yields the original recording FRAME indices
      in ``[0, n_samples)`` -- recovered from the stored ABSOLUTE spike
      times via ``np.searchsorted``, so ``timestamps[frame]`` recovers the
      stored time (the ``as_dataframe=True`` ``spike_times`` column);
    - ``return_times=True`` yields frame-relative seconds (``frame / fs``),
      NOT absolute wall-clock. Absolute times live in the units NWB and
      the DataFrame path; the SI object is frame-relative like v1's.

    Depends on ``populated_sorting`` (not ``populated_recording``)
    so the Sorting row this test inspects is guaranteed to exist
    regardless of test ordering or selection (``-k``, ``--lf``).
    """
    import numpy as np

    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    row = (Sorting & populated_sorting).fetch1()
    si_sorting = Sorting().get_sorting(populated_sorting)
    assert len(si_sorting.unit_ids) == row["n_units"]

    # Recording's wall-clock window. Look up the recording via the
    # source-part: ``recording_id`` is a non-PK FK on
    # ``SortingSelection.RecordingSource`` (the PK is sorting_id),
    # so fetch the recording_id explicitly rather than via
    # ``fetch1("KEY")`` which only returns PK fields.
    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    rec = Recording().get_recording({"recording_id": recording_id})
    timestamps = rec.get_times()
    n_samples = len(timestamps)
    fs = rec.get_sampling_frequency()

    # The as_dataframe path carries the stored ABSOLUTE spike times
    # (read straight from the units NWB), indexed by unit_id.
    df = Sorting().get_sorting(populated_sorting, as_dataframe=True)

    for uid in si_sorting.unit_ids:
        frames = si_sorting.get_unit_spike_train(
            unit_id=uid, return_times=False
        )
        if len(frames) == 0:
            continue
        frames = frames.astype(int)
        # Frames are valid recording sample indices.
        assert frames.min() >= 0
        assert frames.max() < n_samples
        # searchsorted round-trip: looking the frames up in the
        # recording timestamps recovers the stored ABSOLUTE spike times.
        abs_times = np.sort(np.asarray(df.loc[int(uid), "spike_times"]))
        recovered = np.sort(timestamps[frames])
        # Absolute (not relative) tolerance: spike times are seconds that
        # can be large, so the default ``rtol=1e-5`` would silently allow
        # ~ms slack on a late spike. Pin to one sample period, absolute.
        assert np.allclose(recovered, abs_times, rtol=0.0, atol=1.0 / fs)
        # return_times=True is frame-relative (t_start=0), v1 shape --
        # NOT absolute wall-clock. Pins that get_sorting returns a
        # NumpySorting, not an absolute-t_start NwbSortingExtractor.
        rel = si_sorting.get_unit_spike_train(unit_id=uid, return_times=True)
        assert np.allclose(
            np.sort(rel), np.sort(frames / fs), rtol=0.0, atol=1.0 / fs
        )


@pytest.mark.slow
def test_sorting_make_fetch_resolves_artifact_obs_intervals(populated_sorting):
    """``make_fetch`` derives obs_intervals from the ArtifactDetectionSource part.

    Regression guard for the artifact-source schema: the artifact pass
    lives on the zero-or-one ``ArtifactDetectionSource`` part, not a nullable
    ``artifact_detection_id`` FK on the ``SortingSelection`` master. ``make_fetch`` /
    ``make_compute`` / ``_rebuild_analyzer_folder`` gate artifact masking
    on ``sel_row["artifact_detection_id"]``; after the column was dropped, that key
    is absent on the raw ``fetch1()`` row, so ``make_fetch`` must resolve
    it via ``SortingSelection.resolve_artifact_detection(key)`` and stash it. If it
    does not, every artifact-backed sort silently skips masking and
    writes ``obs_intervals=None`` (full-session envelope) -- a silent
    scientific-correctness regression this test exists to catch.

    ``populated_sorting`` is artifact-backed (it inserts an
    ``ArtifactDetectionSelection`` + ``ArtifactDetection`` and threads the
    ``artifact_detection_id`` into the sorting selection), so ``resolve_artifact_detection``
    must be non-None and ``obs_intervals`` must be populated.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # The sort is artifact-backed: exactly one ArtifactDetectionSource part row.
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    assert artifact_detection_id is not None, (
        "populated_sorting must be artifact-backed for this guard to be "
        "meaningful (expected one ArtifactDetectionSource part row)."
    )

    fetched = Sorting().make_fetch(populated_sorting)
    # make_fetch must have re-attached the resolved artifact_detection_id onto
    # sel_row (the dropped master column).
    assert fetched.sel_row.get("artifact_detection_id") == artifact_detection_id
    # ...and therefore derived the artifact-removed observation window,
    # not the None full-session fallback.
    assert fetched.obs_intervals is not None, (
        "make_fetch returned obs_intervals=None for an artifact-backed "
        "sort; the ArtifactDetectionSource artifact_detection_id was not resolved, so "
        "artifact masking is silently skipped."
    )


@pytest.mark.slow
def test_sorting_get_analyzer_loads_folder(populated_sorting):
    """``Sorting.get_analyzer`` loads the SortingAnalyzer from the folder.

    Depends on ``populated_sorting`` so the analyzer folder is
    guaranteed to exist, eliminating the silent
    ``IndexError`` on an empty ``sortings[0]`` lookup that the
    prior ``populated_recording`` dependency exposed under
    isolated / reordered runs.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer = Sorting().get_analyzer(populated_sorting)
    assert isinstance(analyzer, si.SortingAnalyzer)
    assert analyzer.has_extension("templates")


@pytest.mark.slow
def test_prune_orphaned_selections_finds_and_cleans(populated_recording):
    """``prune_orphaned_selections`` finds master rows with no source
    part and removes them when called with ``dry_run=False``.

    Source-part atomicity is enforced at insert time, so orphans only
    arise from upstream maintenance (e.g. a cascade delete from
    Recording that removes the source-part row but leaves the master).
    The helper backs the orphan-master cleanup path.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()

    # Inject orphans directly via dj.Manual.insert1 (bypassing
    # insert_selection -- this is what an upstream cascade-delete leaves
    # behind in production).
    orphan_artifact = uuid.uuid4()
    ArtifactDetectionSelection().insert1(
        {
            "artifact_detection_id": orphan_artifact,
            "artifact_detection_params_name": "default",
        },
        allow_direct_insert=True,
    )
    orphan_sorting = uuid.uuid4()
    # The SortingSelection master has no artifact_detection_id FK (artifact state
    # lives on the ArtifactDetectionSource part); a master row with no source part
    # is exactly the orphan this exercises.
    SortingSelection().insert1(
        {
            "sorting_id": orphan_sorting,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        },
        allow_direct_insert=True,
    )

    dry = ArtifactDetectionSelection.prune_orphaned_selections(dry_run=True)
    assert {"artifact_detection_id": orphan_artifact} in dry
    # Non-dry-run removes the orphan and returns it for review.
    deleted = ArtifactDetectionSelection.prune_orphaned_selections(
        dry_run=False
    )
    assert {"artifact_detection_id": orphan_artifact} in deleted
    assert not (
        ArtifactDetectionSelection & {"artifact_detection_id": orphan_artifact}
    )

    dry = SortingSelection.prune_orphaned_selections(dry_run=True)
    assert {"sorting_id": orphan_sorting} in dry
    deleted = SortingSelection.prune_orphaned_selections(dry_run=False)
    assert {"sorting_id": orphan_sorting} in deleted
    assert not (SortingSelection & {"sorting_id": orphan_sorting})


@pytest.mark.slow
def test_clusterless_thresholder_end_to_end(polymer_smoke_session):
    """Clusterless-thresholder finds peaks and round-trips through the
    pipeline.

    The default ``clusterless_thresholder`` Lookup row uses a 100 uV
    amplitude threshold that finds zero peaks on the 4-second smoke
    fixture's amplitudes. This test inserts a custom ``SorterParameters``
    row with a tuned (lower) threshold and exercises
    Recording -> Artifact -> Sorting through that row, bypassing the
    preset bundle but reusing every Selection / make() body.
    """
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

    # Insert the smoke-fixture sort row from the shared constants
    # so any future param tweak lands in one place
    # (tests/spikesorting/v2/_smoke_constants.py) and the v1
    # baseline-capture + v2 pipeline + parity test all stay in
    # lockstep.
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
            "sorter_params_name": custom_params_name,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    # Clusterless puts every peak into a single unit_id=0.
    assert row["n_units"] == 1
    assert len(Sorting.Unit & sort_pk) == 1
    unit_row = (Sorting.Unit & sort_pk).fetch1()
    assert unit_row["unit_id"] == 0

    # Signal-quality floor: the smoke fixture is 4 s with ~6 planted
    # units firing at ~5-10 Hz, so the planted spike count is on the
    # order of ~150-200 across the sort group. ``n_spikes > 0`` would
    # accept a broken detector that returned a single noise spike, so
    # require a count consistent with finding most planted peaks.
    assert unit_row["n_spikes"] >= 20, (
        f"Clusterless found only {unit_row['n_spikes']} peaks on the "
        "smoke fixture; expected at least 20 given the planted firing "
        "rates and the detect_threshold of 5 (a MAD multiplier, not "
        "µV -- see the note below). A buggy detector that returns a "
        "single false-positive would pass `n_spikes > 0`."
    )
    # Peak amplitude sanity: must be a finite positive number.
    # NOTE: we cannot assert ``peak_amplitude_uv >= detect_threshold``
    # because ``detect_threshold`` is interpreted by SI's
    # ``detect_peaks`` in the recording's native units (raw counts,
    # since the v2 preprocessing pipeline -- bandpass +
    # common_reference -- does NOT gain-scale traces to uV before
    # detection; gain conversion happens only at NWB-write time via
    # ``ElectricalSeries.conversion``). The "detect_threshold stays
    # in microvolts" docstring inherits a v1-era assumption that
    # only holds if the recording was pre-scaled to uV. The unit-confusion
    # is pre-existing and out of scope here; document it so a future
    # maintainer doesn't reinstate the over-specified assertion. The
    # template peak (post-gain-applied via channel_gains in
    # _build_analyzer) being ~0.6 uV is consistent with a 5-count
    # detection threshold on a ~0.2 uV/count probe.
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
    """If ``Sorting.make`` fails between ``AnalysisNwbfile.add()`` and
    a successful ``Sorting.insert1``, the staged units NWB on disk is
    cleaned up.

    Patches ``Sorting._populate_unit_part`` to raise so the
    transaction rolls back AFTER the file is written and registered.
    The rollback path in ``Sorting.make``'s ``except`` block must
    unlink the staged NWB so the file system doesn't accumulate
    orphans on each retry.

    Self-sufficient setup so it doesn't depend on the
    populated_recording module fixture's row still being live
    (other tests in the module may have wiped it).
    """
    import pathlib

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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

    # Self-sufficient Recording chain setup.
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
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    # Ensure no leftover Sorting row.
    (Sorting & sort_pk).super_delete(warn=False)

    # Snapshot the analysis-file directory contents before the
    # broken populate runs so we can detect any orphan file that
    # appears AFTER the rollback. The except block in Sorting.make
    # is responsible for unlinking the staged file before re-raising.
    from spyglass.settings import analysis_dir as ad

    analysis_dir = pathlib.Path(ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    # Patch the Unit-part populate to raise AFTER the file is
    # written and AnalysisNwbfile.add has run inside the transaction.
    # (_populate_unit_part is a staticmethod taking only unit_rows; replacing it
    # with a plain function binds self on instance access.)
    def _broken_unit_part(self, unit_rows):
        raise RuntimeError("simulated unit-part failure")

    monkeypatch.setattr(Sorting, "_populate_unit_part", _broken_unit_part)

    # populate swallows the exception into DataJoint's error
    # machinery via suppress_errors; assert by checking the
    # after-state instead.
    Sorting.populate(sort_pk, reserve_jobs=False, suppress_errors=True)
    assert len(Sorting & sort_pk) == 0, (
        "Sorting row should not be present after rollback; "
        "the transaction was supposed to roll back."
    )

    # No new orphan units NWB should have appeared in the analysis
    # directory after the rolled-back populate.
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"Sorting.make rollback left orphan analysis files: {new_files}. "
        "The except-block in Sorting.make must unlink the staged file "
        "when the transaction rolls back."
    )

    # The analyzer folder published by ``_build_analyzer`` is deliberately LEFT
    # on disk by the rollback (NOT eagerly removed): it is shared by sorting_id
    # and regeneratable, and a concurrent in-flight worker (published but not yet
    # committed) could own it, so failure cleanup never rmtrees it. With no
    # committed Sorting row it is a disk-side orphan reclaimed by
    # ``find_orphaned_analyzer_folders`` (``get_analyzer`` self-heals a missing
    # one). The attempt-owned units NWB IS cleaned (asserted above).
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    analyzer_folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    try:
        assert analyzer_folder.exists(), (
            f"rollback removed the analyzer folder {analyzer_folder}; the "
            "shared, regeneratable analyzer must be left for orphan-sweep, not "
            "eagerly deleted (a concurrent in-flight worker could own it)."
        )
    finally:
        # The orphan is real now; clean it (and the selection) so it does not
        # litter the analyzer root for later orphan-sweep tests in this session.
        if analyzer_folder.exists():
            import shutil

            shutil.rmtree(analyzer_folder, ignore_errors=True)
        (SortingSelection & sort_pk).super_delete(warn=False)


def test_run_si_sorter_restores_global_job_kwargs(dj_conn, monkeypatch):
    """``_run_si_sorter`` leaves SI's global job kwargs byte-identical to
    their pre-sort state.

    ``set_global_job_kwargs`` UPDATES the global rather than replacing it,
    so a job kwarg the sort installs that is absent from the default
    global set (``chunk_size``/``total_memory``/``chunk_memory``) would
    otherwise leak into every later populate. The actual sort is stubbed
    so only the save/restore is exercised.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    si.reset_global_job_kwargs()
    before = dict(si.get_global_job_kwargs())
    assert "chunk_size" not in before  # the key we expect could leak

    # run_si_sorter materializes the sorter output into an in-memory
    # NumpySorting (severing the temp-dir file backing before the scratch dir is
    # cleaned), so the stub must return a real SI sorting -- a bare sentinel has
    # no ``to_spike_vector`` for ``NumpySorting.from_sorting`` to materialize.
    stub_sorting = si.generate_sorting(
        num_units=2, sampling_frequency=30000.0, durations=[0.1]
    )
    monkeypatch.setattr(sis, "run_sorter", lambda **kw: stub_sorting)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    out = Sorting._run_si_sorter(
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
    """``_run_clusterless_thresholder`` strips ``random_seed`` from the
    job kwargs before calling ``detect_peaks``.

    ``random_seed`` is a Spyglass-side knob (already threaded into
    ``random_slices_kwargs``); it is not a valid SI job kwarg, so SI's
    ``fix_job_kwargs`` raises ``AssertionError`` if it reaches
    ``detect_peaks``. ``detect_peaks`` is stubbed to capture the job
    kwargs it receives.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["job_kwargs"] = job_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    Sorting._run_clusterless_thresholder(
        sorter_params={"detect_threshold": 5.0, "noise_levels": [1.0]},
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
    """``_run_clusterless_thresholder`` strips ``threshold_unit`` before
    calling ``detect_peaks``.

    ``threshold_unit`` is a Spyglass-side knob (it selects how
    ``noise_levels`` is derived); it is NOT a ``detect_peaks`` method
    kwarg, so leaving it in the params dict would reach SI and raise an
    unexpected-keyword error at sort time. ``detect_peaks`` is stubbed to
    capture the method kwargs it receives; assert ``threshold_unit`` is
    absent while the real detector knobs survive.
    """
    import numpy as np
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    def _fake_detect_peaks(
        recording, method=None, method_kwargs=None, job_kwargs=None
    ):
        captured["method_kwargs"] = method_kwargs
        return np.array([(10,)], dtype=[("sample_index", "<i8")])

    monkeypatch.setattr(pd_mod, "detect_peaks", _fake_detect_peaks)

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    Sorting._run_clusterless_thresholder(
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
    # The real detector knobs survive the strip.
    assert mk.get("detect_threshold") == 5.0


def _stub_recording_with_2d_probe():
    """Recording stub for the ``_build_analyzer`` unit tests below.

    ``_build_analyzer`` projects the probe to 2D (via ``recording.get_probe()``)
    before building the analyzer; these tests stub the analyzer factory, so the
    recording only needs to report an already-planar probe (``ndim == 2``) so
    the projection step is skipped.
    """

    class _Probe:
        ndim = 2

    class _Recording:
        def get_probe(self):
            return _Probe()

    return _Recording()


def test_build_analyzer_strips_random_seed(dj_conn, monkeypatch, tmp_path):
    """``_build_analyzer`` strips ``random_seed`` before
    ``SortingAnalyzer.compute``.

    ``random_seed`` is a Spyglass-side knob (consumed by the sorter and
    the whitening pin); SI's ``compute`` rejects it with
    ``AssertionError: please remove {'random_seed'}``. The analyzer
    factory + ``compute`` are stubbed to capture the kwargs without a
    real sort.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        si, "create_sorting_analyzer", lambda **k: _FakeAnalyzer()
    )

    class _FakeSorting:
        def get_num_units(self):
            return 2

    Sorting._build_analyzer(
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
    """``_build_analyzer`` requests the right extensions + analyzer kwargs.

    ``test_build_analyzer_strips_random_seed`` captures only the
    ``compute`` job-kwargs; the analyzer factory kwargs and the extension
    set it computes are unasserted. A regression dropping ``sparse=True``
    (dense templates -> wrong peak-channel attribution) or
    ``return_in_uV=True`` (peak amplitudes in raw counts, not µV) would
    pass that test. This pins both the ``create_sorting_analyzer`` kwargs
    and the required ``analyzer.compute`` extensions + per-extension
    params (the persisted ``peak_amplitude_uv`` / peak channel depend on
    them). The extension check is a subset, not exact equality, so adding a
    future extension is not a regression; the downstream correctness of the
    values they feed is covered by the real-analyzer outcome tests
    (``test_unit_conventions``, ``test_analyzer_rebuild_is_seeded_reproducible``).
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    class _FakeAnalyzer:
        def compute(self, *args, **kwargs):
            captured["compute_args"] = args
            captured["compute_kwargs"] = kwargs

    def _fake_create(**kwargs):
        captured["create_kwargs"] = kwargs
        return _FakeAnalyzer()

    monkeypatch.setattr(si, "create_sorting_analyzer", _fake_create)

    class _FakeSorting:
        def get_num_units(self):
            return 2

    Sorting._build_analyzer(
        _FakeSorting(),
        _stub_recording_with_2d_probe(),
        {"sorting_id": "test-sorting-id"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"random_seed": 3, "n_jobs": 1},
        analyzer_folder=tmp_path / "analyzer",
        waveform_params=_DISPLAY_PARAMS,
    )

    # ``create_sorting_analyzer`` kwargs: sparse + µV-return are
    # correctness-critical for the persisted peak amplitude / channel.
    ck = captured["create_kwargs"]
    assert ck.get("sparse") is True, f"expected sparse=True, got {ck}"
    assert (
        ck.get("return_in_uV") is True
    ), f"expected return_in_uV=True, got {ck}"
    assert ck.get("format") == "zarr"

    # ``compute`` extension set (positional first arg). Assert the REQUIRED
    # extensions are present rather than pinning the exact set: a removal
    # still fails (it would break the persisted peak channel / amplitude),
    # but adding a future extension (e.g. metrics) is not a regression.
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

    # Per-extension params: the seeded random-spikes subsample (honoring
    # the per-row random_seed) and the waveform window. Assert the specific
    # scientific values, not exact-dict equality, so an added SI param does
    # not break the guard.
    ext_params = captured["compute_kwargs"]["extension_params"]
    assert (
        ext_params["random_spikes"]["seed"] == 3
    ), "random_spikes seed must honor the job_kwargs random_seed override"
    # The explicit display params fixture is the wide cortex window with the
    # lab's 20000-spike subsample.
    assert ext_params["random_spikes"]["max_spikes_per_unit"] == 20000
    assert ext_params["waveforms"]["ms_before"] == 1.0
    assert ext_params["waveforms"]["ms_after"] == 2.0
    # The Spyglass-only random_seed knob is still stripped from the
    # forwarded job kwargs (SI.compute would reject it).
    assert "random_seed" not in captured["compute_kwargs"]


@pytest.mark.slow
def test_analyzer_rebuild_is_seeded_reproducible(
    dj_conn, monkeypatch, tmp_path
):
    """``_build_analyzer`` seeds ``random_spikes`` so rebuilds are stable.

    The ``random_spikes`` extension uniformly subsamples each unit's
    spikes down to ``max_spikes_per_unit=500`` before computing
    templates; the SI 0.104 default is ``seed=None`` (verified against
    ``ComputeRandomSpikes._set_params``), so without a pinned seed two
    builds of the same sort pick different subsets and the persisted
    peak amplitude / peak channel drift. ``_build_analyzer`` now passes
    ``seed=0``.

    CRITICAL: the seed only changes anything for units with MORE than
    500 spikes -- at or below 500 every spike is selected and the build
    is deterministic regardless of the seed. The MEArec smoke fixture is
    4 s (~tens of spikes/unit), so it would pass this test even with the
    seed reverted (false confidence). This test therefore uses a
    synthetic 40 s, 20 Hz recording whose every unit fires >500 spikes,
    and asserts subsampling actually fired. It drives the real
    ``Sorting._build_analyzer`` (not SI directly) so reverting the seed
    line makes it fail.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import (
        generate_ground_truth_recording,
        template_tools,
    )

    from spyglass.spikesorting.v2.sorting import Sorting

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
        # Pass the explicit cache folder and the cap-500 display recipe so the
        # build subsamples (the seed is observable only when subsampling fires).
        Sorting._build_analyzer(
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

    sel1, chans1, amps1 = _build_and_read(tmp_path / "build_a.zarr")
    sel2, chans2, amps2 = _build_and_read(tmp_path / "build_b.zarr")

    # Subsampling actually fired: 500 selected per unit, strictly fewer
    # than the total available -- otherwise the seed is a no-op and the
    # test proves nothing.
    assert len(sel1) == 500 * len(totals)
    assert len(sel1) < sum(totals.values())

    # The two seeded builds are bit-identical in the quantities the
    # pipeline persists (peak channel + peak amplitude per unit) and in
    # the underlying random_spikes selection itself.
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
    """``_build_analyzer`` honors the per-row ``random_seed`` override.

    ``random_seed`` in ``job_kwargs`` is the established per-row knob that
    the whitening / clusterless-noise pins read
    (``(job_kwargs or {}).get("random_seed", 0)``). The analyzer's
    ``random_spikes`` subsample must read the SAME knob, not a hardcoded
    0 -- otherwise a user changing the seed gets a different sort but the
    same analyzer subsample. Asserts: two builds with ``random_seed=7``
    agree with each other, and differ from the default (seed 0) build --
    proving the override flows through to the extension.
    """
    import numpy as np

    import spikeinterface as si
    from spikeinterface.core import generate_ground_truth_recording

    from spyglass.spikesorting.v2.sorting import Sorting

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
        Sorting._build_analyzer(
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

    seed7_a = _selection_for_seed(tmp_path / "seed7_a.zarr", 7)
    seed7_b = _selection_for_seed(tmp_path / "seed7_b.zarr", 7)
    seed0 = _selection_for_seed(tmp_path / "seed0.zarr", 0)

    # Subsampling fired (otherwise the seed is a no-op and the override
    # can't be observed).
    assert len(seed7_a) < totals
    # The override is deterministic for a fixed seed...
    np.testing.assert_array_equal(
        seed7_a,
        seed7_b,
        err_msg="random_seed=7 is not reproducible across builds",
    )
    # ...and a different seed selects a different subset, proving the
    # override is read (not the hardcoded 0).
    assert not np.array_equal(seed7_a, seed0), (
        "random_seed override did not change the analyzer subsample -- the "
        "extension seed is ignoring job_kwargs['random_seed']."
    )


# Concatenated-recording sorting is now wired end-to-end; the concat-source
# insert_selection + Sorting.populate + split round-trip is covered by the
# chronic smoke in ``tests/spikesorting/v2/test_session_group_concat.py``.


@pytest.mark.slow
def test_sorting_selection_artifact_id_none_is_distinct_identity(
    populated_sorting,
):
    """No-artifact and artifact-backed selections are distinct identities.

    Regression guard: an earlier ``insert_selection`` lookup omitted
    ``artifact_detection_id`` from the master restriction whenever it was None,
    which effectively meant "match any artifact_detection_id." That caused a
    no-artifact ``insert_selection`` call to alias onto a pre-existing
    artifact-backed row for the same
    ``(recording_id, sorter, sorter_params_name)`` triple. Under the
    ``ArtifactDetectionSource`` part-table design, "no artifact pass" is "no
    ``ArtifactDetectionSource`` row" and must stay a distinct identity from an
    artifact-backed selection -- not a wildcard that aliases onto it.

    The ``populated_sorting`` fixture creates an artifact-backed row;
    we then request a no-artifact row with the same recording/sorter/
    params and assert the helper creates a NEW ``sorting_id`` rather
    than returning the artifact-backed one.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    artifact_backed_pk = populated_sorting
    # Derive recording_id + sorter + params from the fixture row so this
    # test does not silently drift if ``populated_sorting`` changes its
    # sorter/params choice.
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
    # The fresh row has no ArtifactDetectionSource part row.
    assert SortingSelection.resolve_artifact_detection(no_artifact_pk) is None
    assert len(SortingSelection.ArtifactDetectionSource & no_artifact_pk) == 0
    # And it's idempotent: a second no-artifact call returns the same row.
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

    A no-artifact-detection selection has zero ArtifactDetectionSource rows; an
    artifact-backed selection has exactly one whose artifact_detection_id
    ``resolve_artifact_detection`` returns. The ArtifactDetectionSource part does NOT leak
    into ``resolve_source`` -- an artifact-backed sort still resolves to
    exactly one recording source.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sorter_kwargs = {
        "recording_id": populated_recording["recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }

    # No-artifact-detection selection: zero ArtifactDetectionSource rows; resolve_artifact_detection None.
    no_art_pk = SortingSelection.insert_selection(dict(sorter_kwargs))
    assert len(SortingSelection.ArtifactDetectionSource & no_art_pk) == 0
    assert SortingSelection.resolve_artifact_detection(no_art_pk) is None

    # Artifact-backed selection: exactly one ArtifactDetectionSource row.
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

    # ArtifactDetectionSource did NOT leak into the recording-source resolution.
    resolved = SortingSelection.resolve_source(art_sort_pk)
    assert resolved.kind == "recording"
    assert resolved.key == {"recording_id": populated_recording["recording_id"]}


# ---------- _write_units_nwb zero-unit guard (sorting layer) -------------


@pytest.mark.slow
def test_write_units_nwb_handles_zero_unit_sorter(populated_recording):
    """``Sorting._write_units_nwb`` initializes an empty Units NWB
    when the sorter produces zero unit ids.

    The guard at ``sorting.py:1690-1698`` is the sorting-layer analog
    of the curation-layer guard tested by
    ``test_curation_v2_stages_empty_units_nwb_on_zero_kept_units``.
    Without this test, a regression in the zero-unit handling at
    the Sorting layer (e.g., a refactor that adds ``add_unit_column``
    before the guard, exactly the bug recently fixed in
    CurationV2) would crash on real datasets where the sorter
    finds no units.
    """
    import numpy as np
    import pynwb
    import spikeinterface as si

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

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

    analysis_file_name, units_object_id = Sorting._write_units_nwb(
        sorting=empty_sorting,
        recording=recording,
        nwb_file_name=nwb_file_name,
    )
    try:
        # Object id is defined (the guard initialized an empty
        # Units table rather than leaving ``nwbf.units = None``).
        assert isinstance(units_object_id, str)
        assert len(units_object_id) > 0

        # The staged NWB on disk is readable and has an empty Units
        # table.
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
        # Tidy up the staged file. _write_units_nwb does not register
        # the file (the caller does so inside a transaction); we
        # unlink the bare file since no DJ row was registered.
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

    v1's pre-curation NWB (``v1/sorting.py:583-598``) wrote both;
    an earlier v2 implementation wrote only ``spike_times`` +
    ``id``, and a subsequent fix inflated ``curation_label`` to a
    ragged list which broke v1-style equality checks. v2 now
    matches v1 exactly: ``obs_intervals`` per-unit + scalar
    ``"uncurated"`` string in ``curation_label``.
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
    # obs_intervals should be a non-empty (n_intervals, 2) array
    # per unit. Each row should have a valid window.
    for uid, obs in df["obs_intervals"].items():
        assert (
            obs is not None and len(obs) >= 1
        ), f"Unit {uid} has empty obs_intervals."
    assert (
        "curation_label" in df.columns
    ), "Units NWB missing curation_label placeholder column."
    # Scalar shape: every unit carries the string ``"uncurated"``
    # -- NOT a list (a list shape would break v1-style equality
    # checks).
    for uid, lbl in df["curation_label"].items():
        assert lbl == "uncurated", (
            f"Unit {uid} has curation_label={lbl!r}; expected scalar "
            "string ``'uncurated'``."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_delete_removes_analyzer_folder(populated_sorting):
    """``Sorting.delete()`` cleans up the analyzer folder on disk.

    v2 introduced ``analyzer_folder`` as a 5-50 GB scratch path
    not tracked by DataJoint. Without ``Sorting``'s delete
    override, the folder leaks every time a Sorting row is
    dropped. Test populates a sort, asserts the folder exists,
    deletes the row with ``safemode=False``, then asserts the
    folder is gone.
    """
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
    # ``_build_analyzer`` runs at populate time, so the folder is a
    # precondition of this test. Treat absence as a FAILURE rather than a
    # vacuous skip: if it is missing, the populate path is broken and the
    # delete-cleanup assertion below would pass without exercising the
    # rmtree it is meant to verify.
    assert folder.exists(), (
        f"precondition: analyzer_folder {folder} should exist after the "
        "populated_sorting fixture; the populate path did not write it."
    )

    # Use cautious_delete with safemode=False so the test runs
    # non-interactively. Sorting.delete() override is the unit
    # under test; it runs ``super().delete(safemode=False)`` then
    # rmtree.
    (Sorting & populated_sorting).delete(safemode=False)
    assert not folder.exists(), (
        f"analyzer_folder {folder} still exists after "
        "Sorting.delete(); cleanup regression."
    )
