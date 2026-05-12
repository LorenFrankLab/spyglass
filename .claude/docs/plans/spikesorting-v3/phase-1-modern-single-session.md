# Phase 1 — Modern single-session sorting end-to-end

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#preprocessingparameters--recordingselection--recording)

The MVP. Builds the complete single-session sort pipeline: preprocessing → artifact detection → sorting → initial curation → merge-table registration. Uses SortingAnalyzer (SI 0.104), Pydantic-validated parameters, and `insert_selection()` helpers that return a single dict. Includes a minimal `run_v3_pipeline()` orchestrator covering this phase's stages (Phase 5 extends with metrics, concat, UnitMatch, FigPack). Parity-tests against the v1 baseline captured in Phase 0.

**PREREQUISITES — must be merged before Phase 1 lands** (see Phase 0's "SI 0.104 upgrade gating" tasks):

1. v1's `extract_waveforms` / `load_waveforms` calls ported to `create_sorting_analyzer` / `load_sorting_analyzer`.
2. v1 test suite green under SI 0.104.
3. `pyproject.toml` SI pin bumped to `>=0.104,<0.105`; `mountainsort5>=0.5` added; optional `spikesorting-v3-matching` extra added.

The first task of this phase (after prerequisites land) is to verify the new SI baseline by running the existing v1 test suite under 0.104 once more and capturing any newly-discovered regressions to fold into Phase 1 implementation notes.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/recording.py](src/spyglass/spikesorting/v1/recording.py) — entire file; v3's `recording.py` replaces this with modern SI APIs and the Pydantic params pattern.
- [src/spyglass/spikesorting/v1/sorting.py](src/spyglass/spikesorting/v1/sorting.py) — entire file.
- [src/spyglass/spikesorting/v1/artifact.py](src/spyglass/spikesorting/v1/artifact.py) — artifact detection logic.
- [src/spyglass/spikesorting/v1/curation.py](src/spyglass/spikesorting/v1/curation.py) — `CurationV1` patterns (lineage, label conventions).
- [src/spyglass/spikesorting/spikesorting_merge.py:34-150](src/spyglass/spikesorting/spikesorting_merge.py#L34-L150) — `SpikeSortingOutput` merge master; v3 adds a new part here.
- [src/spyglass/common/common_nwbfile.py:431](src/spyglass/common/common_nwbfile.py#L431) — `AnalysisNwbfile.build()` context manager API.
- [.claude/docs/plans/spikesorting-v3/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet) — API rename table.
- [.claude/docs/plans/spikesorting-v3/appendix.md § SortingAnalyzer extension dependencies](appendix.md#sortinganalyzer-extension-dependencies) — what to compute at sort time.
- [.claude/docs/plans/spikesorting-v3/appendix.md § MountainSort 5 install + params](appendix.md#mountainsort-5-install--params) — default params.

**Contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — `Sorting.make()` writes analyzer per this convention; `Sorting.get_analyzer()` reads.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — every Lookup table validates on insert.
- [SpikeSortingOutput Part-Table Convention for v3](shared-contracts.md#spikesortingoutput-part-table-convention-for-v3) — adds `CurationV3` part.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — all compute stages use `_resolved_job_kwargs()`.
- [Curation Label Enum](shared-contracts.md#curation-label-enum) — `CurationV3.insert_curation()` validates labels.
- [`insert_selection()` Return-Value Normalization](shared-contracts.md#insert_selection-return-value-normalization) — every helper returns a single dict.

**Designs referenced:** [SortGroupV3](designs.md#sortgroupv3), [PreprocessingParameters + RecordingSelection + Recording](designs.md#preprocessingparameters--recordingselection--recording), [ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection), [SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting), [CurationV3](designs.md#curationv3).

## Tasks

- **Implement `_params/` Pydantic models** for every Lookup table this phase introduces:
  - `_params/artifact_detection.py` — `ArtifactDetectionParamsSchema` with the same fields as v1's defaults plus `detect: bool`.
  - `_params/sorter.py` — one Pydantic model per supported sorter (`MountainSort4Schema`, `MountainSort5Schema`, `Kilosort4Schema`, `SpykingCircus2Schema`, `Tridesclous2Schema`, `ClusterlessThresholderSchema`) + `_get_sorter_schema(sorter_name) -> type[BaseModel]` dispatch helper. **MS4 schema fields** mirror v1's defaults at [src/spyglass/spikesorting/v1/sorting.py](src/spyglass/spikesorting/v1/sorting.py) (`mountain_default` block) without the runtime `tempdir` field-mutation hack. **MS5 schema** uses the defaults from [appendix.md § MountainSort 5 install + params](appendix.md#mountainsort-5-install--params).

- **Implement `recording.py`** — full content per [designs.md § PreprocessingParameters + RecordingSelection + Recording](designs.md#preprocessingparameters--recordingselection--recording). Specific tasks:
  - `SortGroupV3` Manual table per [designs.md § SortGroupV3](designs.md#sortgroupv3). Ship TWO classmethod constructors:
    - `set_group_by_shank(...)` — Frank-lab pattern (one group per shank).
    - `set_group_by_electrode_table_column(column, groups, ...)` — generalized pattern adapted from [Spyglass PR #1438](https://github.com/LorenFrankLab/spyglass/pull/1438) (still DRAFT upstream as of this plan; v3 ships the design). Lets labs whose grouping is keyed off non-shank metadata (e.g., Berke Lab's `intan_channel_number`) configure sort groups without modifying v3 internals.
    - Both share the existing-entry collision-handling pattern: either `delete_existing_entries=True` + `confirm=True` (after reviewing a `DeletionPreview` of cascading impact) OR caller-provided `sort_group_ids` that don't overlap existing IDs. **No silent overwrite, no unconfirmed cascade-delete.** Implementation follows the spyglass-skill inspect-before-destroy discipline: `delete_existing_entries=True` without `confirm=True` returns a `DeletionPreview` (rows to be deleted + downstream-cascade row counts + reclaimable disk + cross-team-owned rows) and raises with "Pass `confirm=True` after reviewing the preview". The standalone `SortGroupV3.preview_existing_entries(nwb_file_name)` exposes the same preview without any destructive intent.
    - `sort_reference_electrode_id` is a parameter on both methods (default -1; per-call configurable so labs that want different reference behavior aren't blocked).
    - Phase 1 validation slice adds: `test_set_group_by_shank_rejects_overlap` (overlapping `sort_group_ids` raises), `test_set_group_by_electrode_table_column_intan` (synthesizes an NWB with an `intan_channel_number` column and asserts the new classmethod groups correctly), `test_set_group_invalid_column_lists_valid` (asserting the error message lists valid columns).
  - `PreprocessingParameters` Lookup, Pydantic-validated, three contents rows: `("default_franklab", ...)`, `("default_neuropixels", {"freq_min": 300, "freq_max": 6000, ...})`, `("no_preproc", {"bandpass_filter": None, ...})` — wait, no, `bandpass_filter` is mandatory. Adjust: `"no_filter"` preset has wide-open band, no whitening.
  - `RecordingSelection` Manual with `insert_selection(key) -> dict` per the [insert_selection contract](shared-contracts.md#insert_selection-return-value-normalization).
  - `Recording` Computed with `make()`, `get_recording(key)` that auto-recomputes on missing cache.

- **Implement `artifact.py`** — `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` per [designs.md § ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection). Note the design has a real `ArtifactDetectionSelection` Manual table (UUID `artifact_id` PK; FKs Recording + ArtifactDetectionParameters). `ArtifactDetection` is Computed and keys off the Selection — required so DataJoint populate-restriction semantics work (`ArtifactDetection.populate({"recording_id": X})` resolves via the Selection table's join, not against a UUID-only PK). Artifact intervals live on `ArtifactDetection.Interval` part table (NOT `IntervalList`). Helper `ArtifactDetection.get_artifact_removed_intervals(key) -> np.ndarray` returns the valid-time intervals for use by `Sorting.make()`.
  - **Cross-sort-group artifact detection** — `ArtifactDetectionSelection` accepts EITHER a single `Recording` (single-group artifact detection, the v1 default) OR a list of `Recording` rows from the same session (cross-group shared-artifact detection, addresses [GitHub issue #928](https://github.com/LorenFrankLab/spyglass/issues/928)). Implementation: add a `SharedArtifactGroup` Manual table with a `Member` part FK'ing `Recording`, and a second `ArtifactDetectionSelection` variant FK'ing `SharedArtifactGroup`. The shared-artifact detection runs once over the union of channels and produces a single `ArtifactDetection.Interval` set that applies to all member recordings (behavioral chewing / licking artifacts visible on every probe). Default path is unchanged; the shared-group path is opt-in.

- **Recording timestamp-range validation in `Recording.make()`** (addresses the silent-truncation class of bugs surfaced by [#1133](https://github.com/LorenFrankLab/spyglass/issues/1133) and [#1585](https://github.com/LorenFrankLab/spyglass/issues/1585)). After materializing the binary cache, assert: `(recording.get_times()[0], recording.get_times()[-1])` covers the requested `IntervalList.valid_times` range within `1.0 / sampling_frequency` tolerance. If the saved range is shorter than requested by more than one sample, raise `RecordingTruncatedError` with a clear message naming the missing seconds — does NOT silently produce a short recording that downstream sorting blindly consumes. Phase 1 test: `test_recording_truncation_caught` synthesizes a recording whose NWB timestamps are truncated and asserts the error is raised at `Recording.populate`, not downstream.

- **Declare the concat-scaffolding tables in Phase 1** (zero-migration policy: every FK target must exist in the phase that introduces the FK). This includes:
  - `SessionGroup` (Manual) + `SessionGroup.Member` (Part) — full schema per [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording). Phase 1 inserts no `make()` for these — they're Manual; users can insert rows but the concat workflow is gated downstream.
  - `MotionCorrectionParameters` (Lookup) — full schema + `contents` rows.
  - `ConcatenatedRecordingSelection` (Manual, UUID PK) — full schema.
  - `ConcatenatedRecording` (Computed) — DECLARED with full `definition`, but `make()` raises `NotImplementedError("ConcatenatedRecording.make() is implemented in Phase 3")`. The schema is final from Phase 1 so `SortingSelection` can FK it; the populate is deferred.
  - **`_params/motion_correction.py` ships in Phase 1** (NOT Phase 3, despite earlier drafts). Per the Pydantic Parameter Schema Convention, every Lookup `insert1` validates its `params` blob — so `MotionCorrectionParameters.insert_default()` in Phase 1 needs `MotionCorrectionParamsSchema` available at module-import time. The schema is small (Literal preset enum + optional `preset_kwargs` dict). Phase 3 makes no changes to the schema, only fills in the consumer (`ConcatenatedRecording.make()`).

  Test added to validation slice: `test_concatenated_recording_make_raises_in_phase_1` asserts `.populate()` raises `NotImplementedError` until Phase 3 ships.

- **Implement `sorting.py`** — `SorterParameters`, `SortingSelection`, `Sorting` per [designs.md § SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting). Specific points:
  - **`SortingSelection` schema is FINAL in Phase 1** per the zero-migration policy. Columns: `sorting_id` PK; `-> [nullable] Recording` (adds `recording_id`); `-> [nullable] ConcatenatedRecording` (adds `concat_recording_id`); `-> SorterParameters`; `-> [nullable] ArtifactDetection` (real FK, NOT a loose `artifact_id=NULL: uuid` — see designs.md and the round-5 #1437 fix). XOR enforced in `insert_selection()`: exactly one of `recording_id` / `concat_recording_id` must be non-NULL. Phase 1's `insert_selection` rejects `concat_recording_id` with `NotImplementedError("Concat path requires Phase 3")`; the validator gate is what changes in Phase 3, not the schema.
  - `Sorting.make()` uses `sis.run_sorter()` then immediately `sic.remove_excess_spikes()` (still in SI 0.104).
  - After sort, builds a `SortingAnalyzer(format="binary_folder", sparse=True)` per the shared contract.
  - Computes `random_spikes`, `noise_levels`, `templates`, `waveforms` at sort time. Other extensions deferred to `AnalyzerCuration` (Phase 2).
  - Writes the units NWB via `AnalysisNwbfile().build(...)`. Stores `analysis_file_name`, `object_id`, `analyzer_folder` on the row. **Use the whitelist-construction pattern, NOT the raw-NWB-copy pattern** that v1 uses. v1's `_write_sorting_to_nwb` copies the raw NWB and then tries to add a `curation_label` column to the existing units table, which fails when the raw NWB already had a Units table of a different length (see [GitHub issue #1437](https://github.com/LorenFrankLab/spyglass/issues/1437)). v3's writer constructs a fresh analysis NWB and copies only the fields it needs (electrodes, devices, session metadata); if the raw NWB has a Units table, it is NOT copied. The `nwbfile._remove_child(nwbfile.units)` workaround from rly's comment on #1437 is unnecessary because v3 never copies it in the first place.
  - `Sorting.get_analyzer(key)` recomputes if folder missing.
  - **Populate `Sorting.Unit` part table** at the end of `make()`. For each unit: peak channel from `analyzer.get_extension("templates").get_unit_template(unit_id)` (channel with max abs amplitude); insert the full `Electrode` FK for that peak channel by restricting through the sort group's `SortGroupV3.SortGroupElectrode` rows (not by `electrode_id` alone). Brain region is reached through `Sorting.Unit * Electrode * BrainRegion`; it is non-null in the Spyglass schema. Installs without annotated regions use the synthetic `BrainRegion` row named `"Unknown"` upstream rather than storing NULL. Per-unit `peak_amplitude_uV` recorded. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - `Sorting.get_unit_brain_regions(key) -> pd.DataFrame` method as a constant-time `Sorting.Unit * Electrode * BrainRegion` join.
  - Default `SorterParameters` rows include **MountainSort 4** alongside MS5: `("mountainsort4", "franklab_tetrode_hippocampus_30kHz_ms4")`, `("mountainsort5", "franklab_tetrode_hippocampus_30kHz_ms5")`, `("kilosort4", "franklab_neuropixels_default")`, `("clusterless_thresholder", "default")`, `("spykingcircus2", "default")`, `("tridesclous2", "default")`. MS4 stays in v3 per resolved decision #1.

- **Implement `curation.py`** — `CurationV3` Manual per [designs.md § CurationV3](designs.md#curationv3). Specific:
  - `insert_curation(sorting_key, labels, parent_curation_id=-1, merge_groups=None, apply_merges=False, description="")` requires an explicit labels dict, validates labels against `CurationLabel` enum, and returns a single dict. Use `{}` for no labels; `None` is invalid.
  - **Populates `CurationV3.Unit` part table** as part of `insert_curation()` by reading the upstream `Sorting.Unit` rows, applying merge_groups (a merged unit inherits the peak channel and region of the contributing unit with the highest amplitude), and writing labels per unit. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - **Populates `CurationV3.UnitLabel` part table** with one row per `(unit_id, curation_label)`. A unit may have multiple labels; unlabeled units have no `UnitLabel` rows. The NWB units table still gets a `curation_label` indexed column so v1-style consumers see empty lists for unlabeled units.
  - Auto-registers into `SpikeSortingOutput.CurationV3` after insert.
  - `get_sorting(key, as_dataframe=False)` and `get_merged_sorting(key)` methods analogous to v1's.
  - `get_unit_brain_regions(key, include_labels=None) -> pd.DataFrame` — constant-time `CurationV3.Unit * Electrode * BrainRegion` join; if `include_labels` is provided, restricts through `CurationV3.UnitLabel` and returns units with any requested label.
  - `get_matchable_unit_ids(key, exclude_labels={"reject", "noise", "artifact"}) -> np.ndarray` — returns curated units with no excluded labels. Unlabeled units and units labeled only `accept` / `mua` are included; a unit with any excluded label is excluded even if it also has another label.
  - `get_sort_group_info(key) -> dj.Table` — returns ALL electrodes in the sort group joined to `Electrode * BrainRegion`, NOT `fetch(limit=1)`. This is the fix for the v1 multi-region under-reporting bug. Returns a DataJoint relation (not a DataFrame) so callers can chain restrictions.

- **Implement a MINIMAL `run_v3_pipeline()` orchestrator** in `src/spyglass/spikesorting/v3/pipeline.py`. Phase 1 ships a usable single-call API so the MVP is actually usable without writing the orchestration boilerplate; Phase 5 extends it with metrics + concat + UnitMatch + FigPack. Phase 1 scope:
  - Signature: `run_v3_pipeline(nwb_file_name, sort_group_id, interval_list_name, team_name, preset="franklab_tetrode_mountainsort5") -> dict`.
  - Internally chains: `RecordingSelection.insert_selection` → `Recording.populate` → `ArtifactDetectionSelection.insert_selection` → `ArtifactDetection.populate` → `SortingSelection.insert_selection` → `Sorting.populate` → `CurationV3.insert_curation` → auto-registration in `SpikeSortingOutput.CurationV3`.
  - NO metrics / auto-curation hookup (Phase 2 adds).
  - NO concat support (Phase 3 extends).
  - NO matcher / FigPack (Phases 4–5).
  - Phase 1 ships **3 presets**: `franklab_tetrode_mountainsort4`, `franklab_tetrode_mountainsort5`, `clusterless_thresholder_default`. The preset is a Pydantic-validated bundle of Lookup-row names; the orchestrator looks them up at first call.
  - Returns a manifest dict listing every `(stage, key)` tuple inserted/populated plus the final `merge_id`.
  - Idempotent: re-running with the same inputs finds the same existing rows and returns the same manifest, no duplicates.

- **Modify `spikesorting_merge.py`** — add new part [per shared-contracts.md § SpikeSortingOutput Part-Table Convention for v3](shared-contracts.md#spikesortingoutput-part-table-convention-for-v3). Specifically:
  - Add `class CurationV3(SpyglassMixinPart)` part to `SpikeSortingOutput`.
  - Extend `get_restricted_merge_ids` to handle `sources=['v3']` (parallel to existing v1 branch at [src/spyglass/spikesorting/spikesorting_merge.py:111](src/spyglass/spikesorting/spikesorting_merge.py#L111)).
  - **Register `CurationV3` in `source_class_dict`** at [src/spyglass/spikesorting/spikesorting_merge.py:26-30](src/spyglass/spikesorting/spikesorting_merge.py#L26-L30). Without this, `SpikeSortingOutput.get_recording()`, `get_sorting()`, `get_sort_group_info()` all KeyError on v3-sourced merge_ids. Per [shared-contracts.md § SpikeSortingOutput.source_class_dict Registration for v3](shared-contracts.md#spikesortingoutputsource_class_dict-registration-for-v3).
  - **Add `SpikeSortingOutput.get_unit_brain_regions(merge_key)`** as a new dispatch method on the merge master (current merge surface has `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_firing_rate` but no per-unit brain-region accessor — see [spikesorting_merge.py:168-220](src/spyglass/spikesorting/spikesorting_merge.py#L168-L220)). Implementation: resolves the merge_key's source via `source_class_dict[part_camel_name]`, then delegates to that source's `get_unit_brain_regions(key)` if defined. `CurationV3` defines it (Phase 1 task above); `CurationV1` and `CuratedSpikeSorting` (v0) do not, so the dispatch raises `AttributeError` with a clear message for v0/v1 merge_ids. The plan does NOT backfill v0/v1.
  - Verify `get_spike_times()` works without modification by confirming `CurationV3` exposes `object_id` (per [shared-contracts.md NWB Column-Name Convention](shared-contracts.md#nwb-column-name-convention-for-spikesortingoutput-routing)); add an explicit test (see Validation slice).

- **Implement an `insert_default()` classmethod** on each new Lookup table that bulk-inserts the contents rows with `skip_duplicates=True` on the Lookup-level (allowed) — mirrors v1 pattern at [src/spyglass/spikesorting/v1/recording.py:127](src/spyglass/spikesorting/v1/recording.py#L127).

- **Run the Phase 0 baseline capture script** against the `minirec` fixture (this is a one-shot prep step; not a code task) and check the artifacts into `tests/spikesorting/v3/baselines/` (small files only; lock down the file sizes before committing).

- **Write integration tests** in `tests/spikesorting/v3/test_phase1_pipeline.py`. See Validation slice for the full table.

- **Sort-correctness validation uses MEArec ground-truth fixtures, not minirec.** Per the revised fixture strategy in Phase 0, minirec is reduced to a plumbing-only regression guard (test rows `test_v3_minirec_plumbing` below). Real sort-correctness lives in `test_v3_mearec_polymer_ground_truth` and `test_v3_mearec_neuropixels_ground_truth` (Phase 1 validation slice below) which compute precision/recall against planted MEArec units. The v1-comparison parity test moves to `test_v3_real_data_v1_parity` — runs only when `SPIKESORTING_V3_REAL_NWB_PATH` is set (a real dataset with real spikes), with tolerances `±1 sample` for `clusterless_thresholder` and `±30-50%` for stochastic sorters. **No minirec-based parity test is shipped.**

- **Documentation update** (Phase 1 ships user-visible changes, so docs go with it):
  - New `docs/src/Pipelines/SpikeSorting/v3.md` — overview, single-session walkthrough, link to the new notebook (Phase 5 will write the full notebook; Phase 1 ships a minimal end-to-end Python script as the example).
  - Update CHANGELOG.md "Unreleased" section: "v3 spike sorting pipeline single-session path lands. New `spyglass.spikesorting.v3` module with `RecordingSelection`, `SortingSelection`, `CurationV3`, plus `SpikeSortingOutput.CurationV3` part. SortingAnalyzer-based; SI 0.104 required."
  - Add v3 module to `docs/src/api/spikesorting.md` (mkdocs API page).

## Deliberately not in this phase

- **No metric-based curation.** `AnalyzerCuration` is Phase 2.
- **No session groups / concatenation.** Phase 3.
- **No cross-session matching.** Phase 4.
- **No FULL orchestrator with metrics / concat / UnitMatch / FigPack.** Phase 1's minimal `run_v3_pipeline()` covers only recording → artifact → sorting → initial curation → merge. Phase 5 extends it with the additional stages.
- **No FigPack / FigURL curation table for v3.** Phase 5 ships FigPack; if a user needs UI curation in Phase 1, they edit a `CurationV3` row in Python.
- **No removal of v1 source.** v1 stays in tree.
- **No recompute table for the binary cache.** Stash this as a Phase 6 follow-up if cache management becomes painful.
- **No automated import of v1 curations into v3.** Different question; if needed, handle in a separate `legacy_import.py` module later.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sort_group_v3_set_by_shank` | `SortGroupV3.set_group_by_shank()` on fresh session inserts expected rows; calling again raises by default; calling with `delete_existing_entries=True, confirm=False` returns/raises with a `DeletionPreview`; only `delete_existing_entries=True, confirm=True` performs the cautious-delete + reinsert path in an isolated fixture. |
| `test_preprocessing_params_validation_at_insert` | `PreprocessingParameters.insert1({"params": {"bandpass_filter": {"freq_min": -1}}})` raises `pydantic.ValidationError`. Valid params inserted as a v1-style dict round-trip cleanly. |
| `test_recording_selection_returns_single_dict` | `RecordingSelection.insert_selection(key)` returns dict on fresh insert; calling again with same key returns the SAME dict, not a list (the v1 footgun). |
| `test_recording_make_writes_binary_cache` (slow) | After `Recording.populate(key)`, `Path(row.binary_cache_path).exists()`; binary file size > 0; hash recorded on row matches recomputed hash. |
| `test_recording_get_recording_recomputes_on_missing_cache` (slow) | Delete the binary cache file. Call `Recording.get_recording(key)` — asserts the cache is repopulated and content hash matches original. |
| `test_artifact_detection_writes_intervals_to_part_table` | After `ArtifactDetection.populate(key)`, `ArtifactDetection.Interval & key` is non-empty (or empty if `detect=False`); does NOT insert into `IntervalList` (regression vs v1). |
| `test_artifact_get_removed_intervals_returns_complement` | Synthesized recording with 2 known artifacts; `get_artifact_removed_intervals(key)` returns 3 valid-time intervals (start-to-art1, art1-to-art2, art2-to-end). |
| `test_sorter_params_dispatched_pydantic_schemas` | `SorterParameters.insert1({"sorter": "mountainsort5", "params": {"scheme": "5"}})` raises (invalid scheme); valid `scheme="2"` inserts cleanly. |
| `test_sorting_make_produces_analyzer_folder` (slow) | After `Sorting.populate(key)`, analyzer folder exists on disk; `load_sorting_analyzer(row.analyzer_folder).get_extension("waveforms")` returns non-None. |
| `test_sorting_get_analyzer_recomputes_on_missing_folder` (slow) | Delete analyzer folder. `Sorting.get_analyzer(key)` recomputes and returns a loadable analyzer. |
| `test_curation_v3_auto_registers_in_merge` | After `CurationV3.insert_curation(...)`, `SpikeSortingOutput.CurationV3 & key` has one row. |
| `test_curation_v3_label_enum_enforced` | `CurationV3.insert_curation(labels={1: ["typo_label"]})` raises `ValueError`. `labels={1: ["accept", "mua"]}` succeeds and inserts two `CurationV3.UnitLabel` rows. |
| `test_spike_sorting_output_routes_v3_in_get_restricted_merge_ids` | `SpikeSortingOutput.get_restricted_merge_ids(key, sources=['v3'])` returns the v3-sourced merge_ids only; `sources=['v1', 'v3']` returns both. |
| `test_spike_sorting_output_source_class_dict_has_curation_v3` | `from spyglass.spikesorting.spikesorting_merge import source_class_dict; assert source_class_dict["CurationV3"]` is the v3 CurationV3 class. The dict is module-level (see [spikesorting_merge.py:26](src/spyglass/spikesorting/spikesorting_merge.py#L26)), NOT a class attribute on `SpikeSortingOutput`. |
| `test_spike_sorting_output_get_recording_works_for_v3` (slow) | After inserting a v3 sort into the merge, `SpikeSortingOutput.get_recording(merge_key)` returns a BaseRecording (does NOT KeyError on dispatch). |
| `test_spike_sorting_output_get_sorting_works_for_v3` (slow) | Same, for `get_sorting`. |
| `test_spike_sorting_output_get_sort_group_info_works_for_v3` | Same, for `get_sort_group_info`. |
| `test_spike_sorting_output_get_spike_times_works_for_v3` (slow) | `SpikeSortingOutput.get_spike_times(merge_key)` returns spike-time arrays for the v3 sort. Validates the `object_id` column-name convention end-to-end. |
| `test_curation_v3_object_id_column_name` | `"object_id" in CurationV3.heading.attributes` and `"units_object_id" not in CurationV3.heading.attributes` (regression test for the NWB column-name convention). |
| `test_sorting_selection_concat_fk_declared_in_phase_1` | `"concat_recording_id" in SortingSelection.heading.attributes` and it is nullable (regression test for forward-compat schema; Phase 3 must NOT have to alter this). |
| `test_sorting_selection_xor_enforced` | `SortingSelection.insert_selection({recording_id: r, concat_recording_id: c, ...})` (both set) raises ValueError; `{recording_id: None, concat_recording_id: None, ...}` (neither set) also raises. Exactly-one-non-null is valid. |
| `test_sorting_selection_rejects_concatenated_in_phase_1` | `SortingSelection.insert_selection({concat_recording_id: c, recording_id: None, ...})` raises `NotImplementedError`; helpful message points at Phase 3. |
| `test_sorter_params_mountainsort4_default_row_present` | After `SorterParameters().insert_default()`, `SorterParameters & {"sorter": "mountainsort4"}` has at least one row. |
| `test_sorter_params_mountainsort4_validation` (slow) | Insert a custom MS4 params row with `detect_sign=0` (invalid in v1 too); raises. Valid `detect_sign=-1` inserts. |
| `test_sorting_mountainsort4_run` (slow, integration) | Run v3 with MS4 against `minirec`; assert `Sorting & key` row produced; `n_units > 0`. |
| `test_sorting_unit_part_populated` (slow) | After `Sorting.populate(key)`, `Sorting.Unit & key` has one row per unit; `peak_amplitude_uV > 0` on every row; `electrode_id` is one of the sort group's electrodes. |
| `test_sorting_unit_brain_region_via_electrode` (slow) | Synthesize a sort group where some channels point at a synthetic `BrainRegion(region_name="Unknown")` row; assert `Sorting.get_unit_brain_regions(key)` returns `"Unknown"` for those units and the planted region for others. (Brain region is always non-null per the Spyglass schema — represented as "Unknown" rather than NULL.) |
| `test_sorting_get_unit_brain_regions_returns_dataframe` (slow) | Returned DataFrame has columns `unit_id`, `electrode_id`, `region_name`, `peak_amplitude_uV`, `n_spikes`. |
| `test_curation_v3_unit_part_populated` | `CurationV3.insert_curation(sort_key, labels={}, ...)` populates `CurationV3.Unit` with one row per surviving unit (after merges applied); merged unit inherits peak channel from highest-amplitude contributor. |
| `test_curation_v3_unit_filter_by_label` | `CurationV3.get_unit_brain_regions(key, include_labels=["accept"])` restricts via `CurationV3.UnitLabel` and returns units carrying the `accept` label, including multi-labeled units. |
| `test_curation_v3_matchable_unit_ids_excludes_any_bad_label` | Units labeled `reject`, `noise`, or `artifact` are excluded even if also labeled `accept`; unlabeled, `accept`, and `mua` units are included. |
| `test_curation_v3_get_sort_group_info_returns_all_electrodes` | Test fixture with a multi-region sort group (synthetic); `get_sort_group_info(key).fetch("region_name")` returns ALL regions, NOT just one (regression fix vs v1's `fetch(limit=1)`). |
| `test_spike_sorting_output_get_unit_brain_regions_for_v3` (slow) | `SpikeSortingOutput.get_unit_brain_regions(merge_key)` works on v3 merge_ids and returns the same DataFrame as `CurationV3.get_unit_brain_regions(curation_key)`. |
| `test_sorted_spikes_group_works_with_v3` (slow) | Insert a `SortedSpikesGroup` row using a v3-sourced merge_id; assert `SortedSpikesGroup.get_spike_times(key)` returns sane numpy arrays. |
| `test_v3_minirec_plumbing` (slow) | Run v3 with `clusterless_thresholder` against `minirec`. Asserts the *pipeline plumbing* works end-to-end (no exception; merge_id minted; `SortGroupV3`, `Recording`, `Sorting`, `CurationV3` rows produced) — **does NOT assert anything about sort correctness** (`minirec` is too short to contain real spikes). This is the regression guard for "did v3 ingestion / populate chain break". |
| `test_v3_mearec_polymer_ground_truth` (slow, integration) | Run v3 with `clusterless_thresholder` against `mearec_polymer_60s.nwb` (the Phase-0-generated ground-truth fixture). Use `spikeinterface.comparison.compare_sorter_to_ground_truth(gt_sorting, v3_sorting)` against the planted Units table. Assert per-unit `accuracy ≥ 0.8` for at least 4 of 6 planted units. **This is the real correctness test.** |
| `test_v3_mearec_neuropixels_ground_truth` (slow, integration) | Same as above against `mearec_neuropixels_60s.nwb` with MountainSort 5. Assert per-unit `accuracy ≥ 0.7` for at least 15 of 20 planted units (MS5 stochasticity wider than clusterless thresholder). |
| `test_v3_mearec_brain_region_correct` (slow) | On `mearec_polymer_60s.nwb` with planted `brain_region_map = {0: "CA1", 1: "CA3"}`, run v3 sort and assert `Sorting.get_unit_brain_regions(key)` returns regions that match the planted soma → peak-channel → group mapping. Directly tests the unit→region tracing invariant. |
| `test_v3_real_data_v1_parity` (slow, integration, requires env var) | If `SPIKESORTING_V3_REAL_NWB_PATH` is set, run v3 + v1 against that NWB and compare per-unit spike times. Tolerance: `clusterless_thresholder` exact within ±1 sample; MS4/MS5 within ±50% unit count + ±30% median FR. Skipped with explicit message if env var unset. **This replaces the prior `test_v3_clusterless_parity` against minirec** (which has no real spikes). |
| `test_run_v3_pipeline_minimal_minirec` (slow, integration) | Single call to `run_v3_pipeline(..., preset="clusterless_thresholder_default")` returns a manifest with stages: recording, artifact, sorting, initial_curation; final `merge_id` is a valid `SpikeSortingOutput.CurationV3` row. |
| `test_run_v3_pipeline_idempotent` (slow) | Two consecutive calls with identical args return identical manifests; no new rows are inserted on the second call. |
| `test_run_v3_pipeline_rejects_unknown_preset` | `run_v3_pipeline(..., preset="not_a_preset")` raises ValueError with the list of registered presets. |

## Fixtures

- **`minirec`** — existing v1 fixture; reused. No changes needed.
- **`baseline_v1_*.{nwb,pkl,json}`** — checked in by Phase 0 baseline capture run.
- **`synthetic_recording_with_2_artifacts`** (new in `tests/spikesorting/v3/conftest.py`) — a 5-second synthetic SI recording with two injected artifact pulses at known timestamps; used by `test_artifact_get_removed_intervals_returns_complement`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Shared setup is in fixtures, not copy-pasted across tests.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- The ground-truth tests (`test_v3_mearec_polymer_ground_truth`, `test_v3_mearec_neuropixels_ground_truth`) genuinely call `spikeinterface.comparison.compare_sorter_to_ground_truth` against the planted Units table and assert real per-unit accuracy thresholds — not mocked tautologies. The `test_v3_real_data_v1_parity` test (env-var gated) loads the v1-baseline pickle and asserts tolerance against the real-data sort; skipped with explicit message if the env var is unset. **No minirec-based parity test ships.**
- `SpikeSortingOutput.CurationV3` part addition does NOT break existing v0/v1 merge queries — confirm by running the existing v1 test suite and downstream consumer tests (`tests/decoding`, `tests/ripple`).
- `set_group_by_shank()` overwrite-guard is honored (regression vs v1 silent overwrite).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG. Run with paths relative to `--src src`, review the JSON `warnings` block, and treat any unaccounted heuristic resolution as a blocker.
- Documentation tasks (CHANGELOG, `docs/src/Pipelines/SpikeSorting/v3.md`, API stub) are landed.
