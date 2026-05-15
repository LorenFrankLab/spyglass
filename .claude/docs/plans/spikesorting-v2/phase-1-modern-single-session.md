# Phase 1 — Modern single-session sorting end-to-end

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#preprocessingparameters--recordingselection--recording)

The MVP. Builds the complete single-session sort pipeline: preprocessing → artifact detection → sorting → initial curation → merge-table registration. Uses SortingAnalyzer (SI 0.104), Pydantic-validated parameters, and `insert_selection()` helpers that return a single dict. Includes a minimal `run_v2_pipeline()` orchestrator covering this phase's stages (Phase 5 extends with metrics, concat, FigPack, and a separate UnitMatch convenience helper). Parity-tests against the v1 baseline captured in Phase 0.

**PREREQUISITES — must be merged before Phase 1 lands** (see [Phase 0c — SpikeInterface 0.104 prerequisite](phase-0c-si-0104-prerequisite.md)):

1. v1's `extract_waveforms` / `load_waveforms` calls ported to `create_sorting_analyzer` / `load_sorting_analyzer`.
2. v1 test suite green under SI 0.104.
3. `pyproject.toml` SI pin bumped to `>=0.104,<0.105`; `mountainsort5>=0.5` added; MS4 runtime status resolved; optional `spikesorting-v2-matching` extra includes `UnitMatchPy>=3.3,<4` and `mat73`.
4. Phase 0b MEArec fixtures generated; real-data v1 baseline captured when `SPIKESORTING_V2_REAL_NWB_PATH` is available, or explicitly documented as skipped for that environment. Any real-data capture uses the isolated integration database unless `SPYGLASS_ALLOW_PRODUCTION_SMOKE=1` is explicitly set for read-only production metadata lookup.

The first task of this phase (after prerequisites land) is to verify the new SI baseline by running the existing v1 test suite under 0.104 once more and capturing any newly-discovered regressions to fold into Phase 1 implementation notes.

Phase 1 is large. The implementer may land it as one PR or as the following recommended slices; each slice's schemas must be in their final zero-migration shape regardless of how the work is chunked:

- **1a — params + schema shells**: `_params/artifact_detection.py`, `_params/sorter.py`, `_params/motion_correction.py`, final table definitions for Phase 1 tables, plus forward-compatible Phase 3 table definitions with gated `make()` bodies.
- **1b — recording + artifact path**: `SortGroupV2`, `PreprocessingParameters`, `RecordingSelection`, `Recording`, `ArtifactDetection*`, shared-artifact groups, tests.
- **1c — sorting + analyzer path**: `SorterParameters`, `SortingSelection`, `Sorting`, analyzer-folder lifecycle, `Sorting.Unit`, unit-brain-region accessors, XOR bypass guards.
- **1d — curation + merge dispatch + minimal pipeline**: `CurationV2`, `SpikeSortingOutput.CurationV2`, merge-table accessors, minimal `run_v2_pipeline()`, docs, end-to-end validation.

## Executor Checklist

- Re-run the SI 0.104 v1 baseline from Phase 0c before coding.
- Use the isolated `uv` environment and isolated DataJoint integration database from Phase 0; do not run Phase 1 populate/recompute tests against production.
- Implement Phase 1 `_params/`, `recording.py`, `artifact.py`, `sorting.py`, `curation.py`, and the minimal `pipeline.py` in the slice order above.
- Declare forward-compatible Phase 3 tables exactly as designed, with `ConcatenatedRecording.make()` still gated by `NotImplementedError`.
- Add `SpikeSortingOutput.CurationV2` registration and v1-compatible merge-table accessors.
- Preserve the nullable-XOR, NWB-resident cache, `insert_selection()` return, and unit-brain-region contracts from `shared-contracts.md`.
- Run the Phase 1 validation goals plus `code_graph.py describe/path` for every new table.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/recording.py](../../../../src/spyglass/spikesorting/v1/recording.py) — entire file; v2's `recording.py` replaces this with modern SI APIs and the Pydantic params pattern.
- [src/spyglass/spikesorting/v1/sorting.py](../../../../src/spyglass/spikesorting/v1/sorting.py) — entire file.
- [src/spyglass/spikesorting/v1/artifact.py](../../../../src/spyglass/spikesorting/v1/artifact.py) — artifact detection logic.
- [src/spyglass/spikesorting/v1/curation.py](../../../../src/spyglass/spikesorting/v1/curation.py) — `CurationV1` patterns (lineage, label conventions).
- [src/spyglass/spikesorting/spikesorting_merge.py:34-166](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L34-L166) — `SpikeSortingOutput` merge master; v2 adds a new part here.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet) — API rename table.
- [.claude/docs/plans/spikesorting-v2/appendix.md § MountainSort 5 install + params](appendix.md#mountainsort-5-install--params) — default params.

**Contracts referenced:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety), [Code Artifact Naming](shared-contracts.md#code-artifact-naming), [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout), [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention), [SpikeSortingOutput Part-Table Convention for v2](shared-contracts.md#spikesortingoutput-part-table-convention-for-v2), [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution), [Curation Label Enum](shared-contracts.md#curation-label-enum), [`insert_selection()` Return-Value Normalization](shared-contracts.md#insert_selection-return-value-normalization).

**Designs referenced:** [SortGroupV2](designs.md#sortgroupv2), [PreprocessingParameters + RecordingSelection + Recording](designs.md#preprocessingparameters--recordingselection--recording), [ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection), [SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting), [CurationV2](designs.md#curationv2).

## Tasks

- **Implement `_params/` Pydantic models** for every Lookup table this phase introduces:
  - `_params/artifact_detection.py` — `ArtifactDetectionParamsSchema` with the same fields as v1's defaults plus `detect: bool`.
  - `_params/sorter.py` — one Pydantic model per supported sorter (`MountainSort4Schema`, `MountainSort5Schema`, `Kilosort4Schema`, `SpykingCircus2Schema`, `Tridesclous2Schema`, `ClusterlessThresholderSchema`) + `_get_sorter_schema(sorter_name) -> type[BaseModel]` dispatch helper. **MS4 schema fields** mirror v1's defaults at [src/spyglass/spikesorting/v1/sorting.py](../../../../src/spyglass/spikesorting/v1/sorting.py) (`mountain_default` block) without the runtime `tempdir` field-mutation hack. **MS5 schema** uses the defaults from [appendix.md § MountainSort 5 install + params](appendix.md#mountainsort-5-install--params). Add `GenericSorterParamsSchema` (`extra="allow"`) for SpikeInterface sorters that are available in the installed environment but not one of the dedicated v2-supported sorters; this preserves v1's "try any installed SI sorter" escape hatch without auto-inserting defaults for every installed sorter.

- **Implement `recording.py`** — full content per [designs.md § PreprocessingParameters + RecordingSelection + Recording](designs.md#preprocessingparameters--recordingselection--recording). Specific tasks:
  - `SortGroupV2` Manual table per [designs.md § SortGroupV2](designs.md#sortgroupv2). Ship TWO classmethod constructors:
  - `set_group_by_shank(...)` — Frank-lab pattern (one group per shank).
  - `set_group_by_electrode_table_column(column, groups, ...)` — generalized pattern adapted from [Spyglass PR #1438](https://github.com/LorenFrankLab/spyglass/pull/1438) (still DRAFT upstream as of this plan; v2 ships the design). Lets labs whose grouping is keyed off non-shank metadata (e.g., Berke Lab's `intan_channel_number`) configure sort groups without modifying v2 internals.
  - Both use the existing-entry handling pattern in [designs.md § SortGroupV2](designs.md#sortgroupv2): additive insert by default; `delete_existing_entries=True, confirm=True` only after reviewing `DeletionPreview`; no silent overwrite.
  - `sort_reference_electrode_id` is a parameter on both methods (default -1; per-call configurable so labs that want different reference behavior aren't blocked).
  - Validation covers overlap rejection, column grouping with an `intan_channel_number`-style column, and invalid-column errors that list valid columns.
  - `PreprocessingParameters` Lookup, Pydantic-validated, three contents rows: `("default_franklab", ...)`, `("default_neuropixels", {"freq_min": 300, "freq_max": 6000, ...})`, and `("no_filter", {"bandpass_filter": {"freq_min": 1, "freq_max": 14999, "filter_order": 1}, "whitening": None, ...})`. `bandpass_filter` is mandatory in the Pydantic schema, so the "no-op" preset uses a wide-open band rather than `None`; whitening is disabled.
  - `RecordingSelection` Manual with `insert_selection(key) -> dict` per the [insert_selection contract](shared-contracts.md#insert_selection-return-value-normalization).
  - `Recording` Computed with `make()` and `get_recording(key)` that auto-recomputes on missing NWB artifact. The preprocessed recording is materialized **NWB-resident** inside an `AnalysisNwbfile` (`electrical_series_path` + `object_id` on the row). HDF5 is the Phase 1 default because the current `AnalysisNwbfile.build()` path writes via `NWBHDF5IO`. No binary sidecar. Any future Zarr or binary-cache optimization must land as a separate lifecycle/scoping PR without changing this schema. See [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format).

- **Implement `artifact.py`** — `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` per [designs.md § ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection). Note the design has a real `ArtifactDetectionSelection` Manual table (UUID `artifact_id` PK; FKs Recording + ArtifactDetectionParameters). `ArtifactDetection` is Computed and keys off the Selection — required so DataJoint populate-restriction semantics work (`ArtifactDetection.populate({"recording_id": X})` resolves via the Selection table's join, not against a UUID-only PK). Artifact intervals live on `ArtifactDetection.Interval` part table (NOT `IntervalList`). Helper `ArtifactDetection.get_artifact_removed_intervals(key) -> np.ndarray` returns the valid-time intervals for use by `Sorting.make()`.
  - **Cross-sort-group artifact detection** — addresses [GitHub issue #928](https://github.com/LorenFrankLab/spyglass/issues/928). Add a `SharedArtifactGroup` Manual table with PK `shared_artifact_group_name` and secondary FK `Session`, plus a `Member` part FK'ing `Recording`. `ArtifactDetectionSelection` is a single table with **nullable XOR FKs** to `Recording` and `SharedArtifactGroup`: exactly one of `recording_id` / `shared_artifact_group_name` must be non-NULL. **XOR enforcement follows [shared-contracts.md § Nullable XOR Foreign-Key Pattern](shared-contracts.md#nullable-xor-foreign-key-pattern)**: helper validation in `insert_selection()`, re-check at the start of `ArtifactDetection.make()`, plus one small parametrized integrity test shared with `SortingSelection`. `SharedArtifactGroup.insert_group()` validates all member recordings belong to the group's session. The single-recording path is the v1 default; the shared-group path is opt-in and runs once over the union of channels, producing a single `ArtifactDetection.Interval` set that applies to all member recordings (behavioral chewing / licking artifacts visible on every probe). Schema details at [designs.md § ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection).

- **Recording timestamp-range validation in `Recording.make()`** (addresses the silent-truncation class of bugs surfaced by [#1133](https://github.com/LorenFrankLab/spyglass/issues/1133) and [#1585](https://github.com/LorenFrankLab/spyglass/issues/1585)). After writing the preprocessed `ElectricalSeries` to the `AnalysisNwbfile`, assert: `(recording.get_times()[0], recording.get_times()[-1])` covers the requested `IntervalList.valid_times` range within `1.0 / sampling_frequency` tolerance. If the saved range is shorter than requested by more than one sample, raise `RecordingTruncatedError` with a clear message naming the missing seconds — does NOT silently produce a short recording that downstream sorting blindly consumes. Phase 1 test: `test_recording_truncation_caught` synthesizes a recording whose NWB timestamps are truncated and asserts the error is raised at `Recording.populate`, not downstream.

- **Declare the concat-scaffolding tables in Phase 1** (zero-migration policy: every FK target must exist in the phase that introduces the FK). This includes:
  - `SessionGroup` (Manual) + `SessionGroup.Member` (Part) — full schema per [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording). The master PK is `(session_group_owner, session_group_name)`, where `session_group_owner` is a projected `LabTeam.team_name`; this prevents collisions between teams that both create a user-facing group named "day1". Phase 1 inserts no `make()` for these — they're Manual; users can insert rows but the concat workflow is gated downstream.
  - `MotionCorrectionParameters` (Lookup) — full schema + `contents` rows.
  - `ConcatenatedRecordingSelection` (Manual, UUID PK) — full schema.
  - `ConcatenatedRecording` (Computed) — DECLARED with full `definition`, but `make()` raises `NotImplementedError("ConcatenatedRecording.make() is implemented in Phase 3")`. The schema is final from Phase 1 so `SortingSelection` can FK it; the populate is deferred.
  - **`_params/motion_correction.py` ships in Phase 1**. Per the Pydantic Parameter Schema Convention, every Lookup `insert1` validates its `params` blob — so `MotionCorrectionParameters.insert_default()` in Phase 1 needs `MotionCorrectionParamsSchema` available at module-import time. The schema is small (Literal preset enum + optional `preset_kwargs` dict). The schema rejects SI `correct_motion` kwargs that would change the return type or write untracked side artifacts (`output_motion`, `output_motion_info`, `folder`). Phase 3 makes no changes to the schema, only fills in the consumer (`ConcatenatedRecording.make()`).

  Validation goal #9 covers the Phase 1 `.populate()` `NotImplementedError` gate.

- **Implement `sorting.py`** — `SorterParameters`, `SortingSelection`, `Sorting` per [designs.md § SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting). Specific points:
  - **`SortingSelection` schema is FINAL in Phase 1** per the zero-migration policy. Columns: `sorting_id` PK; `-> [nullable] Recording` (adds `recording_id`); `-> [nullable] ConcatenatedRecording` (adds `concat_recording_id`); `-> SorterParameters`; `-> [nullable] ArtifactDetection` (real FK, NOT a loose `artifact_id=NULL: uuid` — see designs.md and the round-5 #1437 fix). **XOR enforcement follows [shared-contracts.md § Nullable XOR Foreign-Key Pattern](shared-contracts.md#nullable-xor-foreign-key-pattern)**: `_validate_xor` is called inside `insert_selection()`, re-run at the start of `Sorting.make()` to catch direct `dj.Manual.insert1()` bypasses, and covered by one small parametrized integrity test. Phase 1's `insert_selection` rejects `concat_recording_id` with `NotImplementedError("Concat path requires Phase 3")`; the validator gate is what changes in Phase 3, not the schema.
  - `Sorting.make()` uses `sis.run_sorter()` then immediately `sic.remove_excess_spikes()` (still in SI 0.104).
  - After sort, builds a `SortingAnalyzer(format="binary_folder", sparse=True)` per the shared contract.
  - Computes `random_spikes`, `noise_levels`, `templates`, `waveforms` at sort time. Other extensions deferred to `AnalyzerCuration` (Phase 2).
  - Writes the units NWB via fresh/whitelisted `AnalysisNwbfile` construction. Stores `analysis_file_name`, `object_id`, `analyzer_folder` on the row. **Do not use v1's raw-NWB-copy-then-mutate-Units pattern.** If the current `AnalysisNwbfile().build(...)` path would copy a parent `/units` table, add the minimal whitelist builder mode needed for this write. The invariant is that the analysis NWB written by v2 contains only the v2 sorting Units table, not raw/imported Units copied from the parent NWB (fixes [GitHub issue #1437](https://github.com/LorenFrankLab/spyglass/issues/1437)).
  - `Sorting.get_analyzer(key)` recomputes if folder missing.
  - **Populate `Sorting.Unit` part table** at the end of `make()`. For each unit: peak channel and peak amplitude from SI's documented template helpers (`get_template_extremum_channel(..., outputs="id")` and `get_template_extremum_amplitude(...)`, using the computed `templates` extension); insert the full `Electrode` FK for that peak channel by restricting through the sort group's `SortGroupV2.SortGroupElectrode` rows (not by `electrode_id` alone). Brain region is reached through `Sorting.Unit * Electrode * BrainRegion`; it is non-null in the Spyglass schema. Installs without annotated regions should use a real `BrainRegion` row named `"Unknown"` upstream rather than storing NULL. Per-unit `peak_amplitude_uV` recorded. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - `Sorting.get_unit_brain_regions(key, *, allow_anchor_member=False) -> pd.DataFrame` method as a constant-time `Sorting.Unit * Electrode * BrainRegion` join. **Concat-sort guard (binding)**: when the upstream `SortingSelection` has `concat_recording_id` set, raise `ConcatBrainRegionAmbiguousError` by default; callers opt into anchor-only resolution with `allow_anchor_member=True` (returned rows carry `region_resolution='anchor_member'`). Single-session sorts return `region_resolution='single_session'`. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - Default `SorterParameters` rows include **MountainSort 4** alongside MS5: `("mountainsort4", "franklab_tetrode_hippocampus_30kHz_ms4")`, `("mountainsort5", "franklab_tetrode_hippocampus_30kHz_ms5")`, `("kilosort4", "franklab_neuropixels_default")`, `("clusterless_thresholder", "default")`, `("spykingcircus2", "default")`, `("tridesclous2", "default")`. MS4 stays in v2 only if Phase 0c has proven the runtime package installs and `spikeinterface.sorters.installed_sorters()` includes `mountainsort4` on supported envs; otherwise Phase 1 must treat MS4 as a documented blocker rather than a runnable default. `clusterless_thresholder` is a Spyglass special-case path built on SI peak detection (`detect_peaks`), not a name returned by `spikeinterface.sorters.available_sorters()`.
  - Do **not** extend default contents with every `sis.available_sorters()` row as v1 does. For non-default installed sorters, users insert an explicit `SorterParameters` row validated by `GenericSorterParamsSchema`; the row must fail clearly if `sorter` is not in `sis.available_sorters()`.

- **Implement `curation.py`** — `CurationV2` Manual per [designs.md § CurationV2](designs.md#curationv2). Specific:
  - `insert_curation(sorting_key, labels, parent_curation_id=-1, merge_groups=None, apply_merges=False, description="")` requires an explicit labels dict, validates labels against `CurationLabel` enum, and returns a single dict. Use `{}` for no labels; `None` is invalid. **DB-transaction + file-cleanup guarantee**: the `CurationV2` master insert, `CurationV2.Unit` part inserts, `CurationV2.UnitLabel` part inserts, and `SpikeSortingOutput.CurationV2` merge-part registration run inside one `cls.connection.transaction` block. The curated-units NWB is staged separately and must be deleted on any later failure because DataJoint cannot roll back filesystem side effects. See [designs.md § CurationV2](designs.md#curationv2) for the canonical code shape.
  - **Populates `CurationV2.Unit` part table** as part of `insert_curation()` by reading the upstream `Sorting.Unit` rows, applying merge_groups (a merged unit inherits the peak channel and region of the contributing unit with the highest amplitude), and writing labels per unit. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - **Populates `CurationV2.UnitLabel` part table** with one row per `(unit_id, curation_label)`. A unit may have multiple labels; unlabeled units have no `UnitLabel` rows. The NWB units table still gets a `curation_label` indexed column so v1-style consumers see empty lists for unlabeled units.
  - Auto-registers into `SpikeSortingOutput.CurationV2` after insert.
  - `get_sorting(key, as_dataframe=False)` and `get_merged_sorting(key)` methods analogous to v1's.
  - `get_unit_brain_regions(key, *, include_labels=None, allow_anchor_member=False) -> pd.DataFrame` — constant-time `CurationV2.Unit * Electrode * BrainRegion` join; if `include_labels` is provided, restricts through `CurationV2.UnitLabel` and returns units with any requested label. **Concat-sort guard (binding)**: same behavior as `Sorting.get_unit_brain_regions` — raise `ConcatBrainRegionAmbiguousError` by default for concat-backed curations, accept anchor-only output with `allow_anchor_member=True` and `region_resolution='anchor_member'` column.
  - `get_matchable_unit_ids(key, exclude_labels={"reject", "noise", "artifact"}) -> np.ndarray` — returns curated units with no excluded labels. Unlabeled units and units labeled only `accept` / `mua` are included; a unit with any excluded label is excluded even if it also has another label.
  - `get_sort_group_info(key) -> dj.Table` — returns ALL electrodes in the sort group joined to `Electrode * BrainRegion`, NOT `fetch(limit=1)`. This is the fix for the v1 multi-region under-reporting bug. Returns a DataJoint relation (not a DataFrame) so callers can chain restrictions.

- **Implement a MINIMAL `run_v2_pipeline()` orchestrator** in `src/spyglass/spikesorting/v2/pipeline.py`. Phase 1 ships a usable single-call API so the MVP is actually usable without writing the orchestration boilerplate; Phase 5 extends it with metrics + concat + FigPack and adds `run_v2_unit_match()` for the sort-then-match workflow. Phase 1 scope:
  - Signature: `run_v2_pipeline(nwb_file_name, sort_group_id, interval_list_name, team_name, preset="franklab_tetrode_mountainsort5") -> dict`.
  - Internally chains: `RecordingSelection.insert_selection` → `Recording.populate` → `ArtifactDetectionSelection.insert_selection` → `ArtifactDetection.populate` → `SortingSelection.insert_selection` → `Sorting.populate` → `CurationV2.insert_curation` → auto-registration in `SpikeSortingOutput.CurationV2`.
  - NO metrics / auto-curation hookup (Phase 2 adds).
  - NO concat support (Phase 3 extends).
  - NO matcher / FigPack (Phases 4–5).
  - Phase 1 ships **3 presets**: `franklab_tetrode_mountainsort4`, `franklab_tetrode_mountainsort5`, `clusterless_thresholder_default`. The preset is a Pydantic-validated bundle of Lookup-row names; the orchestrator looks them up at first call.
  - Returns a manifest dict listing every `(stage, key)` tuple inserted/populated plus the final `merge_id`.
  - Idempotent: re-running with the same inputs finds the same existing rows and returns the same manifest, no duplicates.

- **Modify `spikesorting_merge.py`** — add new part [per shared-contracts.md § SpikeSortingOutput Part-Table Convention for v2](shared-contracts.md#spikesortingoutput-part-table-convention-for-v2). Specifically:
  - Add `class CurationV2(SpyglassMixinPart)` part to `SpikeSortingOutput`.
  - Extend `get_restricted_merge_ids` to handle `sources=['v2']` (parallel to existing v1 branch at [src/spyglass/spikesorting/spikesorting_merge.py:111](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L111)). Implement the Phase 1 subset of [shared-contracts.md § SpikeSortingOutput Part-Table Convention for v2](shared-contracts.md#spikesortingoutput-part-table-convention-for-v2): restrictions by `nwb_file_name`, `team_name`, `sort_group_id`, `interval_list_name`, `preproc_params_name`, `recording_id`, `artifact_id`, `sorter`, `sorter_params_name`, `sorting_id`, and `curation_id` must all resolve through v2 Selection tables to `SpikeSortingOutput.CurationV2` rows.
  - **Register `CurationV2` in `source_class_dict`** at [src/spyglass/spikesorting/spikesorting_merge.py:26-30](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L26-L30). Without this, `SpikeSortingOutput.get_recording()`, `get_sorting()`, `get_sort_group_info()` all KeyError on v2-sourced merge_ids. Per [shared-contracts.md § SpikeSortingOutput.source_class_dict Registration for v2](shared-contracts.md#spikesortingoutputsource_class_dict-registration-for-v2).
  - **Add `SpikeSortingOutput.get_unit_brain_regions(merge_key)`** as a new dispatch method on the merge master (current merge surface has `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_firing_rate` but no per-unit brain-region accessor — see [spikesorting_merge.py:168-214](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L168-L214)). Implementation: resolves the merge_key's source via `source_class_dict[part_camel_name]`, then delegates to that source's `get_unit_brain_regions(key)` if defined. `CurationV2` defines it (Phase 1 task above); `CurationV1` and `CuratedSpikeSorting` (v0) do not, so the dispatch raises `AttributeError` with a clear message for v0/v1 merge_ids. The plan does NOT backfill v0/v1.
  - Verify `get_spike_times()` works without modification by confirming `CurationV2` exposes `object_id` (per [shared-contracts.md NWB Column-Name Convention](shared-contracts.md#nwb-column-name-convention-for-spikesortingoutput-routing)); add an explicit test (see Validation goals).
  - Leave `ImportedSpikeSorting` unchanged. External/ground-truth NWB Units continue to register through the existing `SpikeSortingOutput.ImportedSpikeSorting` part; Phase 1 only adds `CurationV2`.

- **Implement an `insert_default()` classmethod** on each new Lookup table that bulk-inserts the contents rows with `skip_duplicates=True` on the Lookup-level (allowed) — mirrors v1 pattern at [src/spyglass/spikesorting/v1/recording.py:127](../../../../src/spyglass/spikesorting/v1/recording.py#L127).

- **Write integration tests** in behavior-named modules such as `tests/spikesorting/v2/test_single_session_pipeline.py`. See Validation goals for the full table. Do not encode plan phases in test module or test function names.

- **Sort-correctness validation uses MEArec ground-truth fixtures, not minirec.** Per the revised fixture strategy in Phase 0, minirec is reduced to a plumbing-only regression guard (test rows `test_v2_minirec_plumbing` below). Real sort-correctness lives in `test_v2_mearec_polymer_ground_truth` and `test_v2_mearec_neuropixels_ground_truth` (Phase 1 validation goals below) which compute precision/recall against planted MEArec units. The v1-comparison parity test moves to `test_v2_real_data_v1_parity` — runs only when `SPIKESORTING_V2_REAL_NWB_PATH` is set (a real dataset with real spikes), with tolerances `±1 sample` for `clusterless_thresholder` and `±30-50%` for stochastic sorters. **No minirec-based parity test is shipped.**

- **Documentation update** (Phase 1 ships user-visible changes, so docs go with it):
  - New `docs/src/Features/SpikeSortingV2.md` — overview, single-session walkthrough, link to the new notebook (Phase 5 will write the full notebook; Phase 1 ships a minimal end-to-end Python script as the example).
  - Add CHANGELOG entry noting the new v2 single-session surface (`Recording`, `Sorting`, `CurationV2`, `SpikeSortingOutput.CurationV2`) and the SI 0.104 requirement.
  - Add v2 module to `docs/src/api/spikesorting.md` (mkdocs API page).

## Deliberately not in this phase

- **No metric-based curation.** `AnalyzerCuration` is Phase 2.
- **No session groups / concatenation.** Phase 3.
- **No cross-session matching.** Phase 4.
- **No FULL orchestrator with metrics / concat / FigPack and no UnitMatch helper.** Phase 1's minimal `run_v2_pipeline()` covers only recording → artifact → sorting → initial curation → merge. Phase 5 extends it with the additional sorting stages and adds `run_v2_unit_match()`.
- **No FigPack / FigURL curation table for v2.** Phase 5 ships FigPack; if a user needs UI curation in Phase 1, they edit a `CurationV2` row in Python.
- **No removal of v1 source.** v1 stays in tree.
- **No recompute table implementation in Phase 1.** Phase 1 exposes `Recording.get_recording()` and `Sorting.get_analyzer()` missing-artifact rebuild helpers. Phase 2 adds the explicit `RecordingArtifactRecompute*` / `SortingAnalyzerRecompute*` verification and safe-deletion tables, so this is a sequencing boundary, not a deferral out of the v2 plan.
- **No automated import of v1 curations into v2.** Different question; if needed, handle in a separate `legacy_import.py` module later.

## Validation goals

Behaviors the Phase 1 validation goals must cover. Implementer chooses test names and splits; each goal must have at least one assertion exercising it.

1. **SortGroupV2 set-by-shank inspect-before-destroy**: fresh insert works; second call without flags raises (no silent overwrite); `delete_existing_entries=True, confirm=False` returns a `DeletionPreview` and raises; `confirm=True` runs cautious_delete + reinsert. Same surface on `set_group_by_electrode_table_column` (column-based grouping); bogus column lists valid columns in the error.
2. **Pydantic params validation at insert**: `PreprocessingParameters`, `ArtifactDetectionParameters`, `SorterParameters` (per-sorter dispatch including MS4 `detect_sign` and `GenericSorterParamsSchema`) all reject bogus values; valid params round-trip.
3. **`insert_selection()` returns single dict**: every Selection helper returns a PK-only dict — never a list — on both fresh and repeat call (the v1 footgun).
4. **`Recording` NWB-resident round-trip** (slow): row has `analysis_file_name`, `electrical_series_path`, `object_id`, `cache_hash`; on missing artifact, `get_recording(key)` rebuilds in place without deleting the DataJoint row; rebuilt hash matches stored.
5. **Recording timestamp coverage check**: a synthetically truncated NWB raises `RecordingTruncatedError` at populate, not downstream.
6. **`ArtifactDetection.Interval` part table** (not `IntervalList`): populate fills the part rows; `get_artifact_removed_intervals` returns the complement intervals.
7. **`Sorting` + analyzer lifecycle** (slow): populate produces the analyzer folder; `get_analyzer(key)` recomputes if folder missing; `sic.remove_excess_spikes()` handles boundary spikes without duration errors; `Sorting.Unit` has one row per unit with peak `electrode_id` from the sort group and `peak_amplitude_uV > 0`.
8. **XOR enforcement (two-layer + integrity test)**: `insert_selection` rejects both-NULL and both-set on `SortingSelection` and `ArtifactDetectionSelection`; bypass-inserted rows raise at `make()` start; one parametrized integrity test asserts no XOR violations exist in the production DB.
9. **Forward-compat schema (zero-migration)**: `SortingSelection.heading.attributes` includes nullable `concat_recording_id` from Phase 1; `concat_recording_id`-non-NULL `insert_selection` raises `NotImplementedError` (Phase 3 lifts this without schema change); `ConcatenatedRecording.populate()` raises `NotImplementedError` in Phase 1; `CurationV2.object_id` (not `units_object_id`) is the column name per the merge convention.
10. **CurationV2 integration**: `insert_curation(labels={})` succeeds (zero-unit invariant); labels validated against `CurationLabel` enum; auto-registers in `SpikeSortingOutput.CurationV2`; transaction is atomic — forced part-insert failure leaves no DB rows AND no orphan staged analysis file. Brain-region accessors return per-unit regions (single-session = `region_resolution='single_session'`; concat-backed raises `ConcatBrainRegionAmbiguousError` unless `allow_anchor_member=True`); `get_sort_group_info` returns ALL electrodes (regression vs v1 `fetch(limit=1)`); `get_matchable_unit_ids` excludes any unit with `reject`/`noise`/`artifact`.

**Merge-dispatch goals** (verify `SpikeSortingOutput` works on v2 merge_ids, slow): `source_class_dict["CurationV2"]` resolves; `get_restricted_merge_ids(sources=['v2'])` returns v2 rows; the v2 restriction surface (nwb_file_name, team_name, sort_group_id, interval_list_name, preproc/artifact/sorter/sorting/curation IDs) all resolve; `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, `get_firing_rate`, `get_unit_brain_regions` all work end-to-end; `SortedSpikesGroup.get_spike_times` returns sane arrays for a v2-sourced merge_id.

**Sort-correctness goals** (slow, integration):

- `mearec_polymer_128ch_60s.nwb` with `clusterless_thresholder`: per-unit `accuracy ≥ 0.8` for at least two-thirds of planted units (primary correctness gate; Phase 0 fixture targets ~24 planted units).
- `mearec_neuropixels_60s.nwb` with MS5: per-unit `accuracy ≥ 0.7` for ≥15 of 20 planted units.
- `mearec_polymer_128ch_60s.nwb` with planted brain regions: `Sorting.get_unit_brain_regions(key)` matches the planted soma → peak-channel → group mapping (directly tests unit→region tracing).
- `minirec` plumbing-only: pipeline chain produces a merge_id; **no correctness claim**.
- `SPIKESORTING_V2_REAL_NWB_PATH` env-var gated: v1↔v2 parity per tolerances (`clusterless_thresholder` ±1 sample; MS4/MS5 ±50% unit count + ±30% median FR). Skipped if env var unset. The test uses the isolated database/write directories; `SPYGLASS_ALLOW_PRODUCTION_SMOKE=1` permits read-only production metadata lookup only.
- `run_v2_pipeline(..., preset="clusterless_thresholder_default")` returns a complete manifest; second call returns identical manifest with no duplicate inserts; unknown preset raises `ValueError`.

## Commands to run

If landing slices, run the relevant subset of `tests/spikesorting/v2/test_single_session_pipeline.py` and `code_graph.py describe` for each table touched in that slice. Before considering Phase 1 complete, run the full gate:

```bash
source .venv-spikesorting-v2/bin/activate
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v1/ -q
pytest tests/spikesorting/v2/test_single_session_pipeline.py -q
pytest tests/spikesorting/v2/test_integrity.py -q
pytest tests/decoding tests/spikesorting/v1/test_merge.py -q

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe RecordingSelection --file spyglass/spikesorting/v2/recording.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe Recording --file spyglass/spikesorting/v2/recording.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe ArtifactDetectionSelection --file spyglass/spikesorting/v2/artifact.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe ConcatenatedRecording --file spyglass/spikesorting/v2/session_group.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe SortingSelection --file spyglass/spikesorting/v2/sorting.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe Sorting --file spyglass/spikesorting/v2/sorting.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe CurationV2 --file spyglass/spikesorting/v2/curation.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up Recording --file spyglass/spikesorting/v2/recording.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up Sorting --file spyglass/spikesorting/v2/sorting.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --down CurationV2 --file spyglass/spikesorting/v2/curation.py --json

git diff --check -- src/spyglass/spikesorting/v2 src/spyglass/spikesorting/spikesorting_merge.py tests/spikesorting/v2 docs/src/Features CHANGELOG.md
```

## Fixtures

- **`minirec`** — existing v1 fixture; reused. No changes needed.
- **`baseline_v1_*.{nwb,pkl,json}`** — checked in by Phase 0 baseline capture run.
- **`synthetic_recording_with_2_artifacts`** (new in `tests/spikesorting/v2/conftest.py`) — a 5-second synthetic SI recording with two injected artifact pulses at known timestamps; used by `test_artifact_get_removed_intervals_returns_complement`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation goals are covered; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Shared setup is in fixtures, not copy-pasted across tests.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- The ground-truth tests (`test_v2_mearec_polymer_ground_truth`, `test_v2_mearec_neuropixels_ground_truth`) genuinely call `spikeinterface.comparison.compare_sorter_to_ground_truth` against the planted Units table and assert real per-unit accuracy thresholds — not mocked tautologies. The `test_v2_real_data_v1_parity` test (env-var gated) loads the v1-baseline pickle and asserts tolerance against the real-data sort; skipped with explicit message if the env var is unset. **No minirec-based parity test ships.**
- `SpikeSortingOutput.CurationV2` part addition does NOT break existing v0/v1 merge queries — confirm by running the existing v1 test suite and downstream consumer tests (`tests/decoding`, `tests/ripple`).
- `set_group_by_shank()` overwrite-guard is honored (regression vs v1 silent overwrite).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG. Run with paths relative to `--src src`, review the JSON `warnings` block, and treat any unaccounted heuristic resolution as a blocker.
- Documentation tasks (CHANGELOG, `docs/src/Features/SpikeSortingV2.md`, API stub) are landed.
