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
  - `SortGroupV3` Manual table per [designs.md § SortGroupV3](designs.md#sortgroupv3); `set_group_by_shank()` raises on overwrite without `force=True`.
  - `PreprocessingParameters` Lookup, Pydantic-validated, three contents rows: `("default_franklab", ...)`, `("default_neuropixels", {"freq_min": 300, "freq_max": 6000, ...})`, `("no_preproc", {"bandpass_filter": None, ...})` — wait, no, `bandpass_filter` is mandatory. Adjust: `"no_filter"` preset has wide-open band, no whitening.
  - `RecordingSelection` Manual with `insert_selection(key) -> dict` per the [insert_selection contract](shared-contracts.md#insert_selection-return-value-normalization).
  - `Recording` Computed with `make()`, `get_recording(key)` that auto-recomputes on missing cache.

- **Implement `artifact.py`** — `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` per [designs.md § ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection). Note the design has a real `ArtifactDetectionSelection` Manual table (UUID `artifact_id` PK; FKs Recording + ArtifactDetectionParameters). `ArtifactDetection` is Computed and keys off the Selection — required so DataJoint populate-restriction semantics work (`ArtifactDetection.populate({"recording_id": X})` resolves via the Selection table's join, not against a UUID-only PK). Artifact intervals live on `ArtifactDetection.Interval` part table (NOT `IntervalList`). Helper `ArtifactDetection.get_artifact_removed_intervals(key) -> np.ndarray` returns the valid-time intervals for use by `Sorting.make()`.

- **Declare the concat-scaffolding tables in Phase 1** (zero-migration policy: every FK target must exist in the phase that introduces the FK). This includes:
  - `SessionGroup` (Manual) + `SessionGroup.Member` (Part) — full schema per [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording). Phase 1 inserts no `make()` for these — they're Manual; users can insert rows but the concat workflow is gated downstream.
  - `MotionCorrectionParameters` (Lookup) — full schema + `contents` rows.
  - `ConcatenatedRecordingSelection` (Manual, UUID PK) — full schema.
  - `ConcatenatedRecording` (Computed) — DECLARED with full `definition`, but `make()` raises `NotImplementedError("ConcatenatedRecording.make() is implemented in Phase 3")`. The schema is final from Phase 1 so `SortingSelection` can FK it; the populate is deferred.

  Test added to validation slice: `test_concatenated_recording_make_raises_in_phase_1` asserts `.populate()` raises `NotImplementedError` until Phase 3 ships.

- **Implement `sorting.py`** — `SorterParameters`, `SortingSelection`, `Sorting` per [designs.md § SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting). Specific points:
  - **`SortingSelection` schema is FINAL in Phase 1** per the zero-migration policy. Columns: `sorting_id` PK; `-> [nullable] Recording` (adds `recording_id`); `-> [nullable] ConcatenatedRecording` (adds `concat_recording_id`); `-> SorterParameters`; `artifact_id=NULL: uuid`. XOR enforced in `insert_selection()`: exactly one of the two recording FKs must be non-NULL. Phase 1's `insert_selection` rejects `concat_recording_id` with `NotImplementedError("Concat path requires Phase 3")` even though the FK is structurally valid; the validator gate is what changes in Phase 3, not the schema.
  - `Sorting.make()` uses `sis.run_sorter()` then immediately `sic.remove_excess_spikes()` (still in SI 0.104).
  - After sort, builds a `SortingAnalyzer(format="binary_folder", sparse=True)` per the shared contract.
  - Computes `random_spikes`, `noise_levels`, `templates`, `waveforms` at sort time. Other extensions deferred to `AnalyzerCuration` (Phase 2).
  - Writes the units NWB via `AnalysisNwbfile().build(...)`. Stores `analysis_file_name`, `object_id`, `analyzer_folder` on the row.
  - `Sorting.get_analyzer(key)` recomputes if folder missing.
  - **Populate `Sorting.Unit` part table** at the end of `make()`. For each unit: peak channel from `analyzer.get_extension("templates").get_unit_template(unit_id)` (channel with max abs amplitude); brain region via `(Electrode * BrainRegion & {"electrode_id": peak_ch})`. Brain region is NULL if no row exists. Per-unit `peak_amplitude_uV` recorded. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - `Sorting.get_unit_brain_regions(key) -> pd.DataFrame` method as a constant-time `Sorting.Unit * Electrode * BrainRegion` join.
  - Default `SorterParameters` rows include **MountainSort 4** alongside MS5: `("mountainsort4", "franklab_tetrode_hippocampus_30kHz_ms4")`, `("mountainsort5", "franklab_tetrode_hippocampus_30kHz_ms5")`, `("kilosort4", "franklab_neuropixels_default")`, `("clusterless_thresholder", "default")`, `("spykingcircus2", "default")`, `("tridesclous2", "default")`. MS4 stays in v3 per resolved decision #1.

- **Implement `curation.py`** — `CurationV3` Manual per [designs.md § CurationV3](designs.md#curationv3). Specific:
  - `insert_curation(sorting_key, parent_curation_id=-1, labels=None, merge_groups=None, apply_merges=False, description="")` validates labels against `CurationLabel` enum, returns single dict.
  - **Populates `CurationV3.Unit` part table** as part of `insert_curation()` by reading the upstream `Sorting.Unit` rows, applying merge_groups (a merged unit inherits the peak channel and region of the contributing unit with the highest amplitude), and writing labels per unit. See [shared-contracts.md § Unit-Level Brain Region Tracing](shared-contracts.md#unit-level-brain-region-tracing).
  - Auto-registers into `SpikeSortingOutput.CurationV3` after insert.
  - `get_sorting(key, as_dataframe=False)` and `get_merged_sorting(key)` methods analogous to v1's.
  - `get_unit_brain_regions(key, include_labels=None) -> pd.DataFrame` — constant-time `CurationV3.Unit * Electrode * BrainRegion` join; if `include_labels` is provided, filters by `curation_label IN include_labels`.
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
  - Verify `get_spike_times()` works without modification by confirming `CurationV3` exposes `object_id` (per [shared-contracts.md NWB Column-Name Convention](shared-contracts.md#nwb-column-name-convention-for-spikesortingoutput-routing)); add an explicit test (see Validation slice).

- **Implement an `insert_default()` classmethod** on each new Lookup table that bulk-inserts the contents rows with `skip_duplicates=True` on the Lookup-level (allowed) — mirrors v1 pattern at [src/spyglass/spikesorting/v1/recording.py:127](src/spyglass/spikesorting/v1/recording.py#L127).

- **Run the Phase 0 baseline capture script** against the `minirec` fixture (this is a one-shot prep step; not a code task) and check the artifacts into `tests/spikesorting/v3/baselines/` (small files only; lock down the file sizes before committing).

- **Write integration tests** in `tests/spikesorting/v3/test_phase1_pipeline.py`. See Validation slice for the full table.

- **Parity test against v1 baseline.** New test `test_v3_clusterless_parity` (marked `@pytest.mark.slow`): runs v3 with `clusterless_thresholder` on the same `minirec` slice as the Phase 0 baseline, fetches spike times via `SpikeSortingOutput`, asserts **exact** integer-sample equality unit-by-unit against the baseline pickle. Deterministic sorter → tolerance is zero. If parity fails, the test reports per-unit deltas to aid debugging.

- **Smoke test against v1 MountainSort baseline.** New test `test_v3_mountainsort5_smoke` (marked `@pytest.mark.slow`): runs v3 with MountainSort 5 against `minirec` and asserts: `(a)` `n_units` is within ±50% of v1's `n_units` (MS5 vs MS4, stochastic); `(b)` median firing rate across units is within 30% of v1's; `(c)` total spike count is within 30%. Quantitative tolerances chosen to catch order-of-magnitude bugs without being noise-sensitive.

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
| `test_sort_group_v3_set_by_shank` | `SortGroupV3.set_group_by_shank()` on fresh session inserts expected rows; calling again raises without `force=True`; with `force=True` deletes downstream cascade and reinserts. |
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
| `test_curation_v3_label_enum_enforced` | `CurationV3.insert_curation(labels={1: ["typo_label"]})` raises `ValueError`. `labels={1: ["accept"]}` succeeds. |
| `test_spike_sorting_output_routes_v3_in_get_restricted_merge_ids` | `SpikeSortingOutput.get_restricted_merge_ids(key, sources=['v3'])` returns the v3-sourced merge_ids only; `sources=['v1', 'v3']` returns both. |
| `test_spike_sorting_output_source_class_dict_has_curation_v3` | `SpikeSortingOutput.source_class_dict["CurationV3"]` is the v3 CurationV3 class (regression test for the dispatch dict). |
| `test_spike_sorting_output_get_recording_works_for_v3` (slow) | After inserting a v3 sort into the merge, `SpikeSortingOutput.get_recording(merge_key)` returns a BaseRecording (does NOT KeyError on dispatch). |
| `test_spike_sorting_output_get_sorting_works_for_v3` (slow) | Same, for `get_sorting`. |
| `test_spike_sorting_output_get_sort_group_info_works_for_v3` | Same, for `get_sort_group_info`. |
| `test_spike_sorting_output_get_spike_times_works_for_v3` (slow) | `SpikeSortingOutput.get_spike_times(merge_key)` returns spike-time arrays for the v3 sort. Validates the `object_id` column-name convention end-to-end. |
| `test_curation_v3_object_id_column_name` | `"object_id" in CurationV3.heading.attributes` and `"units_object_id" not in CurationV3.heading.attributes` (regression test for the NWB column-name convention). |
| `test_sorting_selection_recording_source_present_in_phase_1` | `"recording_source" in SortingSelection.heading.attributes` (regression test for forward-compat schema decision; Phase 3 must NOT have to alter this). |
| `test_sorting_selection_rejects_concatenated_in_phase_1` | `SortingSelection.insert_selection({"recording_source": "concatenated", ...})` raises `NotImplementedError`; helpful message points at Phase 3. |
| `test_sorter_params_mountainsort4_default_row_present` | After `SorterParameters().insert_default()`, `SorterParameters & {"sorter": "mountainsort4"}` has at least one row. |
| `test_sorter_params_mountainsort4_validation` (slow) | Insert a custom MS4 params row with `detect_sign=0` (invalid in v1 too); raises. Valid `detect_sign=-1` inserts. |
| `test_sorting_mountainsort4_run` (slow, integration) | Run v3 with MS4 against `minirec`; assert `Sorting & key` row produced; `n_units > 0`. |
| `test_sorting_unit_part_populated` (slow) | After `Sorting.populate(key)`, `Sorting.Unit & key` has one row per unit; `peak_amplitude_uV > 0` on every row; `electrode_id` is one of the sort group's electrodes. |
| `test_sorting_unit_brain_region_nullable` (slow) | Synthesize a sort group where some channels have no `BrainRegion` row; assert `Sorting.Unit.brain_region` is NULL for those units, populated for others. |
| `test_sorting_get_unit_brain_regions_returns_dataframe` (slow) | Returned DataFrame has columns `unit_id`, `electrode_id`, `region_name`, `peak_amplitude_uV`, `n_spikes`. |
| `test_curation_v3_unit_part_populated` | `CurationV3.insert_curation(sort_key, ...)` populates `CurationV3.Unit` with one row per surviving unit (after merges applied); merged unit inherits peak channel from highest-amplitude contributor. |
| `test_curation_v3_unit_filter_by_label` | `CurationV3.get_unit_brain_regions(key, include_labels=["accept"])` returns only rows with `curation_label == "accept"`. |
| `test_curation_v3_get_sort_group_info_returns_all_electrodes` | Test fixture with a multi-region sort group (synthetic); `get_sort_group_info(key).fetch("region_name")` returns ALL regions, NOT just one (regression fix vs v1's `fetch(limit=1)`). |
| `test_spike_sorting_output_get_unit_brain_regions_for_v3` (slow) | `SpikeSortingOutput.get_unit_brain_regions(merge_key)` works on v3 merge_ids and returns the same DataFrame as `CurationV3.get_unit_brain_regions(curation_key)`. |
| `test_sorted_spikes_group_works_with_v3` (slow) | Insert a `SortedSpikesGroup` row using a v3-sourced merge_id; assert `SortedSpikesGroup.get_spike_times(key)` returns sane numpy arrays. |
| `test_v3_clusterless_parity` (slow, integration) | Run v3 with `clusterless_thresholder` on `minirec`; fetched spike times exactly match Phase 0 baseline pickle. Deterministic — zero tolerance. |
| `test_v3_mountainsort5_smoke` (slow, integration) | Run v3 with MountainSort 5 on `minirec`; `n_units` ±50%, median FR ±30%, total spikes ±30% vs v1 baseline. |
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
- The parity test (`test_v3_clusterless_parity`) is real (loads the baseline pickle, asserts equality), not a mocked tautology.
- `SpikeSortingOutput.CurationV3` part addition does NOT break existing v0/v1 merge queries — confirm by running the existing v1 test suite and downstream consumer tests (`tests/decoding`, `tests/ripple`).
- `set_group_by_shank()` overwrite-guard is honored (regression vs v1 silent overwrite).
- Documentation tasks (CHANGELOG, `docs/src/Pipelines/SpikeSorting/v3.md`, API stub) are landed.
