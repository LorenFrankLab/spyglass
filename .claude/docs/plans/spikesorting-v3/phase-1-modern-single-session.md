# Phase 1 — Modern single-session sorting end-to-end

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#preprocessingparameters--recordingselection--recording)

The MVP. Builds the complete single-session sort pipeline: preprocessing → artifact detection → sorting → initial curation → merge-table registration. Uses SortingAnalyzer (SI 0.104), Pydantic-validated parameters, and `insert_selection()` helpers that return a single dict. Parity-tests against the v1 baseline captured in Phase 0.

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
  - `_params/sorter.py` — one Pydantic model per supported sorter (`MountainSort5Schema`, `Kilosort4Schema`, `SpykingCircus2Schema`, `Tridesclous2Schema`, `ClusterlessThresholderSchema`) + `_get_sorter_schema(sorter_name) -> type[BaseModel]` dispatch helper. Field defaults from [appendix.md § MountainSort 5 install + params](appendix.md#mountainsort-5-install--params).

- **Implement `recording.py`** — full content per [designs.md § PreprocessingParameters + RecordingSelection + Recording](designs.md#preprocessingparameters--recordingselection--recording). Specific tasks:
  - `SortGroupV3` Manual table per [designs.md § SortGroupV3](designs.md#sortgroupv3); `set_group_by_shank()` raises on overwrite without `force=True`.
  - `PreprocessingParameters` Lookup, Pydantic-validated, three contents rows: `("default_franklab", ...)`, `("default_neuropixels", {"freq_min": 300, "freq_max": 6000, ...})`, `("no_preproc", {"bandpass_filter": None, ...})` — wait, no, `bandpass_filter` is mandatory. Adjust: `"no_filter"` preset has wide-open band, no whitening.
  - `RecordingSelection` Manual with `insert_selection(key) -> dict` per the [insert_selection contract](shared-contracts.md#insert_selection-return-value-normalization).
  - `Recording` Computed with `make()`, `get_recording(key)` that auto-recomputes on missing cache.

- **Implement `artifact.py`** — `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` per [designs.md § ArtifactDetectionParameters + ArtifactDetection](designs.md#artifactdetectionparameters--artifactdetection). Notably: artifact intervals live on `ArtifactDetection.Interval` part table (NOT `IntervalList`). Helper `ArtifactDetection.get_artifact_removed_intervals(key) -> np.ndarray` returns the valid-time intervals for use by `Sorting.make()`.

- **Implement `sorting.py`** — `SorterParameters`, `SortingSelection`, `Sorting` per [designs.md § SorterParameters + SortingSelection + Sorting](designs.md#sorterparameters--sortingselection--sorting). Specific points:
  - `Sorting.make()` uses `sis.run_sorter()` then immediately `sic.remove_excess_spikes()` (still in SI 0.104).
  - After sort, builds a `SortingAnalyzer(format="binary_folder", sparse=True)` per the shared contract.
  - Computes `random_spikes`, `noise_levels`, `templates`, `waveforms` at sort time. Other extensions deferred to `AnalyzerCuration` (Phase 2).
  - Writes the units NWB via `AnalysisNwbfile().build(...)`. Stores both `analysis_file_name` + `analyzer_folder` on the row.
  - `Sorting.get_analyzer(key)` recomputes if folder missing (mirrors v1's `SpikeSortingRecording.get_recording` pattern at [src/spyglass/spikesorting/v1/recording.py:475-712](src/spyglass/spikesorting/v1/recording.py#L475-L712)).
  - Default `SorterParameters` rows: `("mountainsort5", "franklab_tetrode_30kHz")`, `("kilosort4", "franklab_neuropixels_default")`, `("clusterless_thresholder", "default")`, `("spykingcircus2", "default")`, `("tridesclous2", "default")`. MS4 wrapper NOT added (see Open Question #1 in overview).

- **Implement `curation.py`** — `CurationV3` Manual per [designs.md § CurationV3](designs.md#curationv3). Specific:
  - `insert_curation(sorting_key, parent_curation_id=-1, labels=None, merge_groups=None, apply_merges=False, description="")` validates labels against `CurationLabel` enum, returns single dict.
  - Auto-registers into `SpikeSortingOutput.CurationV3` after insert.
  - `get_sorting(key, as_dataframe=False)` and `get_merged_sorting(key)` methods analogous to v1's.

- **Modify `spikesorting_merge.py`** — add new part [per shared-contracts.md § SpikeSortingOutput Part-Table Convention for v3](shared-contracts.md#spikesortingoutput-part-table-convention-for-v3). Specifically: add `class CurationV3(SpyglassMixinPart)` part to `SpikeSortingOutput`. Extend `get_restricted_merge_ids` to handle `sources=['v3']` (parallel to existing v1 branch at [src/spyglass/spikesorting/spikesorting_merge.py:111](src/spyglass/spikesorting/spikesorting_merge.py#L111)).

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
- **No `run_v3_pipeline()` orchestrator.** Phase 5.
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
| `test_sorted_spikes_group_works_with_v3` (slow) | Insert a `SortedSpikesGroup` row using a v3-sourced merge_id; assert `SortedSpikesGroup.get_spike_times(key)` returns sane numpy arrays. |
| `test_v3_clusterless_parity` (slow, integration) | Run v3 with `clusterless_thresholder` on `minirec`; fetched spike times exactly match Phase 0 baseline pickle. Deterministic — zero tolerance. |
| `test_v3_mountainsort5_smoke` (slow, integration) | Run v3 with MountainSort 5 on `minirec`; `n_units` ±50%, median FR ±30%, total spikes ±30% vs v1 baseline. |

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
