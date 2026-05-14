# Spike Sorting v2 — Research Notes & Hypothesis Tree

Working document accumulating evidence and design rationale for the v2 design.
Date: 2026-05-11. Author: Eric Denovellis + Claude.

## Confidence Legend

- 🟢 **High** — directly verified from source or vendor docs.
- 🟡 **Medium** — strong inference from multiple sources but unverified in our exact context.
- 🔴 **Low** — open question or speculation.

---

## 1. Goals (from user prompt)

1. Support SpikeInterface ≥0.104 (currently pinned `>=0.99.1,<0.100`).
2. Cross-session sorting via concatenation or UnitMatch / DeepUnitMatch.
3. Better UX — fewer hand-edited dicts, less notebook-driven table manipulation.
4. Handle chronic recordings (30 kHz × many channels × days).
5. Verify against v1 where reasonable (modulo stochasticity of MountainSort).

## 2. Current State (v1) — High-Confidence Facts

🟢 V1 lives in `src/spyglass/spikesorting/v1/`. Tables:

| File | Tables |
|------|--------|
| `recording.py` | `SortGroup`, `SpikeSortingPreprocessingParameters`, `SpikeSortingRecordingSelection`, `SpikeSortingRecording` |
| `sorting.py` | `SpikeSorterParameters`, `SpikeSortingSelection`, `SpikeSorting` |
| `artifact.py` | `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` |
| `curation.py` | `CurationV1` |
| `metric_curation.py` | `WaveformParameters`, `MetricParameters`, `MetricCurationParameters`, `MetricCurationSelection`, `MetricCuration` |
| `figurl_curation.py` | `FigURLCurationSelection`, `FigURLCuration` |
| `burst_curation.py` | `BurstPairParams`, `BurstPairSelection`, `BurstPair` |
| `recompute.py` | `RecordingRecomputeVersions`, `RecordingRecomputeSelection`, `RecordingRecompute` |

🟢 Output merge table: `SpikeSortingOutput` at `src/spyglass/spikesorting/spikesorting_merge.py` with parts `CurationV1`, `ImportedSpikeSorting`, `CuratedSpikeSorting` (v0).

🟢 Downstream consumer: `SortedSpikesGroup` at `src/spyglass/spikesorting/analysis/v1/group.py` keys off `SpikeSortingOutput.merge_id`.

### V1 SpikeInterface API usage (`spikeinterface>=0.99.1,<0.100`)

🟢 V1 uses the **WaveformExtractor** API (`si.extract_waveforms`, `si.load_waveforms`) — removed in SI 0.101. This is the #1 forcing function for v2.
🟢 V1 uses `si.preprocessing.bandpass_filter`, `common_reference`, `whiten` — all still available in 0.104.
🟢 V1 uses `sis.run_sorter`, `sq.compute_*` quality metrics — all still available with renames.
🟡 V1 mutates `sorter_params` in place at `sorting.py:402-408` (`sorter_params.pop(...)`) — fragility.

### V1 storage

🟢 Preprocessed recording → AnalysisNwbfile (NWB-wrapped via `SpikeInterfaceRecordingDataChunkIterator`).
🟢 Sorting → AnalysisNwbfile `/units` table.
🟢 Waveforms → SI WaveformExtractor folder under `temp_dir/{metric_curation_id}/`.
🟢 Metrics → columns on AnalysisNwbfile units table.
🟢 Artifact intervals → `IntervalList` with `interval_list_name = str(artifact_id)`.

---

## 3. UX Pain Points (from notebook audit)

🟢 Confirmed from `notebooks/10_Spike_SortingV1.ipynb` (63 cells, 35 code):

1. **Sequential dependency chain**: 5+ `insert_selection()`+`populate()` pairs, manual UUID propagation between steps.
2. **`insert_selection` return-value polymorphism**: returns `dict` on fresh, `list[dict]` on rerun — splatting `**rec_key` fails on rerun.
3. **Sorter-specific dict construction inline**: cell 27–34 has if/else branching on sorter name.
4. **Hard-coded FigURL `gh://` URI strings**: brittle to repo restructuring.
5. **Manual loop to insert into `SpikeSortingOutput`**: easy to forget; no downstream consumer until you do it.
6. **No multi-session pattern**: `set_group_by_shank()` is per-session-file; user must loop manually.
7. **Parameter sets discovered by string name**: typos fail at populate, not insert.
8. **Curation versioning** uses auto-increment + parent_curation_id; linear chain, no DAG.
9. **MetricCuration → CurationV1 bridge** (`insert_metric_curation`) is non-obvious — must remember to call it explicitly.
10. **Tetrode geometry hard-coded** at `recording.py:630-643` — silent failure on other probes.

---

## 4. SpikeInterface 0.99 → 0.104 — Critical Migration Points

🟢 **WaveformExtractor REMOVED in 0.101.** `SortingAnalyzer` replaces it:

```python
from spikeinterface import create_sorting_analyzer
analyzer = create_sorting_analyzer(sorting, recording, sparse=True, format="binary_folder", folder=path)
analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])
analyzer.compute(["principal_components", "spike_amplitudes", "correlograms",
                  "template_metrics", "unit_locations"])
```

🟢 **PreprocessingPipeline** declarative API in 0.103+ — perfect for our Lookup table:

```python
preprocessing_dict = {
    "bandpass_filter":   {"freq_min": 300, "freq_max": 6000},
    "common_reference":  {"reference": "global", "operator": "median"},
    "whiten":            {"dtype": "float32"},
}
pipeline = PreprocessingPipeline(preprocessing_dict)
rec_processed = pipeline.apply(recording)
```

🟢 **`set_global_job_kwargs(n_jobs=N, chunk_duration="1s")`** is the canonical pattern for parallel writes.

🟢 **`concatenate_recordings([rec1, rec2])` → mono-segment virtual recording**; sorter sees one long timeline. Required for concat-and-sort.

🟢 **`correct_motion(...)` presets**: `dredge_fast`, `medicine`, `kilosort_like`, `rigid_fast`, `nonrigid_accurate`. DREDge (0.101+) handles cross-day drift best.

🟢 **Quality metric renames in 0.104**:
- `peak_to_valley` → `peak_to_trough_duration`
- `peak_trough_ratio` → `peak_to_trough_ratio` (now absolute-valued)
- `snr` switched from mean→median (numeric thresholds shift)

🟢 **`return_scaled` → `return_in_uV`** (audit all `extract_waveforms` callsites).

🟢 **`spikeinterface.curation` modern primitives**:
- `apply_curation(analyzer, curation_dict)` — apply JSON curation
- `apply_merges_to_sorting(sorting, merge_groups)`
- `compute_merge_unit_groups(analyzer, preset=...)` — auto-merge candidates
- `remove_redundant_units(...)`, `remove_duplicated_spikes(...)`

🟢 **Sorters**: Kilosort 4, MountainSort 5 (replaces MS4), SpykingCircus2, Tridesclous2 are current-gen. MS4 deprecated. SC2/TDC2 are pure-Python (no MATLAB, no containers).

🟡 **Zarr<3.0 remains a SpikeInterface dependency** as of 0.104 (`zarr>=2.18,<3`, with `numcodecs<0.16.0` for Zarr v2 support). v2 should not add a separate Zarr pin unless the SI upgrade exposes a concrete resolver/runtime issue.

---

## 5. Cross-Session Strategies

| | Concatenate-and-sort | Sort-then-UnitMatch |
|---|---|---|
| Timescale | Same-day with breaks | Days–weeks |
| Drift handling | `correct_motion` on concat | UnitMatch rigid-shift estimate |
| Output identity | Sorter assigns same unit ID across span | Match probability + FDR per pair |
| Re-running on new session | Re-sort full concat | Incremental match |
| Tetrode compatibility | OK (drift correction ≈ no-op) | 🔴 Open question — published UnitMatch validation is Neuropixels-heavy; v2 gates on polymer and records tetrode AUC as informational |
| Maturity | KS4/MS5 + DREDge | UnitMatchPy 3.3.1 active |

🟢 **UnitMatchPy** (https://pypi.org/project/UnitMatchPy/) is the maintained Python port. Includes `UMPy_spike_interface_demo.ipynb`.
🟢 **DeepUnitMatch** lives in the same UnitMatchPy repo (`DeepUnitMatch` subpackage) — pretrained model for inference, drop-in via same data interface.
🔴 **Tetrode UnitMatch validation needs empirical check** before production use on Frank lab data.

🟢 **DECISION RULE**:
- v2 supports BOTH paths, but they stay separate: concat-and-sort uses `ConcatenatedRecording` + `Sorting`; sort-then-match uses the matcher plugin API.
- MVP is sort-then-UnitMatch (more general, incremental, introspectable).
- Concat-and-sort is built on top: `ConcatenatedRecording` virtual table → existing sort path. It does not emit UnitMatch rows or identity-mapping matches in this plan.

---

## 6. Chronic Recording / Large Data

🟢 **Memory model**: SI is lazy by default. 30 kHz × 64 ch × 24 h ≈ 138 GB never holds in memory. Workflow:
1. Lazy `read_*` extractor (metadata only)
2. Lazy preprocessing chain (no I/O)
3. `recording.save(format="binary", chunk_duration="2s", n_jobs=8)` materializes preprocessed to NVMe ONCE
4. Sorter reads materialized binary
5. `SortingAnalyzer(format="binary_folder", sparse=True)` for postprocessing

🟢 **Sparse waveforms by default in 0.101+** (`create_sorting_analyzer(..., sparse=True)`) — 5-10× storage savings on dense probes.

🟢 **`SharedmemRecording` + `SharedMemoryTemplates`** avoid duplicating arrays across worker processes.

🟢 **`dump_to_dict()` / `dump_to_json()` / `dump_to_pickle()`** — every SI object serializable to recipe, lazily reconstructed in workers.

🟡 **Kilosort 4 `max_cluster_subset=25_000`** default — 65% sorting-time reduction on overnight data per KS4 paper.

---

## 7. Spyglass Conventions (from skill)

🟢 **Schema naming**: v2 lives in shared `spikesorting` schema (lab-shared module in `SHARED_MODULES`).

🟢 **Tier discipline** for each new pipeline step:
- Lookup (Parameters, contents-baked)
- Manual (Selection, user-inserted, FK to Params)
- Computed (with `make()`, FK to Selection)
- Merge ONLY for multi-source convergence.

🟢 **Use existing `SpikeSortingOutput`** as the downstream entry point. Add new part `SpikeSortingOutput.CurationV2` rather than creating a new merge table. `SortedSpikesGroup` keeps working.

🟢 **AnalysisNwbfile `.build()` context manager** for all NWB writes:
```python
with AnalysisNwbfile().build(nwb_file_name) as builder:
    obj_id = builder.add_nwb_object(my_array, table_name="result")
    analysis_file_name = builder.analysis_file_name
self.insert1({**key, "analysis_file_name": analysis_file_name, "result_object_id": obj_id})
```

🟢 **Group table pattern** (e.g., `SortedSpikesGroup`) is the right shape for our cross-session grouping needs.

🟢 **SpyglassMixin must be first in MRO**.

🟢 **`IntervalList.insert1(..., skip_duplicates=True)` is BANNED in custom `make()`** — bypasses orphan protection.

🟢 **`set_group_by_shank()` issue (#11)** — overwrites existing sort groups silently, cascades downstream. v2 should be additive or warn loudly.

---

## 8. Architecture Hypothesis Tree

### H1: Reuse `SpikeSortingOutput` merge table → 🟢 ADOPT
- **Pro**: `SortedSpikesGroup`, decoding, MUA, ripple — all downstream pipelines keep working without changes.
- **Pro**: Users with v1 sorts and v2 sorts in the same database can mix.
- **Con**: Couples us to the existing merge schema (`merge_id`-only PK).
- **Verdict**: ADOPT. Add `SpikeSortingOutput.CurationV2` part table.

### H2: SortingAnalyzer-first storage → 🟢 ADOPT
- Single source of truth for waveforms, templates, metrics, locations.
- Persisted as `binary_folder` for the v2 SortingAnalyzer plan. Zarr remains an SI-supported format and may be selected by the Phase 0 benchmark as the `AnalysisNwbfile` backend for Recording artifacts, but it is not a SortingAnalyzer storage dependency.
- v2 `Sorting` table writes `SortingAnalyzer` folder + lightweight units NWB; downstream tables read from analyzer extensions.

### H3: Parameters as Pydantic-validated schemas → 🟢 ADOPT
- Lookup tables get `params: blob` typed via Pydantic models.
- `insert_selection` validates params on insert (`PydanticModel.model_validate(...)`).
- Eliminates "typo at populate" failure mode.
- Backward compatible: blob in DB stays a dict.

### H4: Pipeline orchestration via convenience helpers → 🟢 ADOPT
- `run_v2_pipeline()` is the sort-and-curate entry point: takes either single-session inputs or `concat_session_group_name`, returns final `merge_id`.
- `run_v2_unit_match()` is the separate sort-then-match entry point: takes `session_group_name` plus explicit per-member `curation_choices`, returns `unitmatch_id`.
- Internally: helpers insert selection rows, populate each stage, and return manifest dictionaries.
- Idempotent: re-run finds existing rows, doesn't duplicate.
- Notebook becomes ~5 cells instead of 35.

### H5: Cross-session as plugin matcher → 🟢 ADOPT
- `MatcherParameters` Lookup starts with `matcher='unitmatch'`. `deepunitmatch` remains a future plugin. `concat_identity` is deferred because concat-backed sorting has one curation for the concatenated recording, while Phase 4 intentionally models one pinned curation per `SessionGroup.Member`.
- `UnitMatch` Computed dispatches to matcher backend.
- Per-session sortings remain valid; matching is additive.
- Concat path: `ConcatenatedRecording` builds virtual recording → existing `Sorting` table. It does not emit identity-mapping matches in this plan; concat-backed sorting and sort-then-match remain separate workflows.

### H6: Session group table for chronic recordings → 🟢 ADOPT
- `SessionGroup` (Manual, master) + `SessionGroup.Member` (Part).
- Member declares `(nwb_file_name, interval_list_name, sort_group_id)`.
- Reusable across pipeline branches: same group used for concat-and-sort AND for sort-then-match.
- Matches `SortedSpikesGroup` shape (skill recommended pattern).

### H7: Replace `MetricCuration` + `BurstPair` with `AnalyzerCuration` → 🟡 PROPOSE
- Single Computed table walks `SortingAnalyzer` extensions, applies auto-curation rules, optional auto-merge.
- Consolidates `MetricCuration` + `BurstPair` (which currently both compute waveform similarity separately).
- **Risk**: BurstPair has bespoke visualizations (`plot_by_sort_group_ids`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, `plot_peak_over_time`) — must preserve.
- **Verdict**: ADOPT but keep visualization helpers. The two tables compute similar things twice today.

### H8: FigPack instead of (or alongside) FigURL → 🟡 PROPOSE
- FigPack is the intended FigURL successor path, but its spike-sorting extension API and edited-curation state round trip must be verified at Phase 5 implementation time.
- Maturity is the open question; need to confirm install/auth/upload path works for our lab.
- **Verdict**: ADOPT as new path. FigURL remains available for v1 data only; v2 does not extend FigURL.

### H9: Don't break v0/v1 → 🟢 ADOPT
- v0 and v1 remain in tree, populate-runnable for existing data.
- v2 is additive. Existing v1 curations stay queryable through `SpikeSortingOutput.CurationV1`; no v1-to-v2 conversion helper is part of this plan.

### H10: Validation against v1 → 🟢 ADOPT
- `minirec` is only a plumbing fixture; it is too short to contain real spikes and must not be used as a sort-correctness or v1-parity oracle.
- Sort correctness uses MEArec ground-truth fixtures and `spikeinterface.comparison.compare_sorter_to_ground_truth`.
- v1 parity runs only when `SPIKESORTING_V2_REAL_NWB_PATH` is set. `clusterless_thresholder` is the deterministic tight-equivalence path; stochastic sorters use bounded qualitative tolerances.

---

## 9. Self-Critique of Architecture

**Risk 1**: Adding a fourth part table to `SpikeSortingOutput` may bloat downstream merge resolution.
- Mitigation: `SortedSpikesGroup` already handles N parts; one more is fine.

**Risk 2**: Plugin matcher API may leak SpikeInterface object details into DataJoint blobs.
- Mitigation: serialize matcher inputs/outputs to AnalysisNwbfile via `.build()`; DB stores only the analysis_file_name.

**Risk 3**: Convenience helpers hide reproducibility — what got inserted?
- Mitigation: `run_v2_pipeline()` and `run_v2_unit_match()` return manifest dicts with every `(selection_table, key)` they touched. Notebooks can print them.

**Risk 4**: Concat-and-sort across days breaks for sorters that assume contiguous time. Mitigation:
- Same-day is the default path.
- `SessionGroup.create_group(..., allow_multi_day=True)` is required for multi-day groups, and downstream concatenation requires an explicit non-`auto` motion-correction preset. The error points users to sort-then-match as the recommended cross-day workflow.

**Risk 5**: 5 phases is a lot. Phasing failure cascades.
- Mitigation: Phase 1 (Modern Single-Session) is independently shippable and replaces v1's biggest pain (WaveformExtractor → SortingAnalyzer migration). Phases 2-5 are optional additions.

**Risk 6**: SortingAnalyzer folder management at scale — chronic recording analyzer can be 50 GB+.
- Mitigation: Phase 1 stores analyzer paths and hashes rather than table blobs; Phase 2 adds `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*` verification tables so storage reclamation is explicit and gated on successful round-trip checks.

**Risk 7**: UnitMatch may not transfer cleanly from Neuropixels-published examples to Frank-lab polymer/tetrode use cases.
- Mitigation: Phase 4 gates on a MEArec polymer-probe fixture, because polymer is the lab-relevant standard for this workflow. Neuropixels and tetrode AUCs are recorded as informational checks; if tetrode features collapse, tetrode users are routed to concat-and-sort or future validation work rather than promised UnitMatch support.

---

## 10. Open Questions for User

1. **Sorter prioritization**: MS5 is the v1 successor for MountainSort users, KS4 is the field standard for Neuropixels. Should v2 ship with both or commit to MS5+KS4 only (deprecate MS4)?
2. **Schema location**: live in `spyglass.spikesorting.v2` (parallel to v0/v1) ✅ confirmed by directory convention. Stays in shared `spikesorting` schema.
3. **Migration policy**: resolved. Keep v0/v1 alive indefinitely in this plan; document v1-vs-v2 path selection, not a sunset.
4. **DeepUnitMatch in MVP?** Recommend "no" — Phase 4.1 enhancement after UnitMatch baseline.
5. **FigPack vs FigURL**: resolved. v2 defaults to FigPack after a feasibility check; no silent FigURL fallback.
6. **Concat-and-sort across days**: resolved. Schema supports multi-day via `allow_multi_day=True`, but same-day remains the default and sort-then-match is the recommended cross-day workflow.

---

## 11. Phasing Decision

5 independently shippable phases, plus Phase 0 scaffolding:

- **Phase 0**: Project scaffold, deps, baseline-capture infra. ~1 PR.
- **Phase 1**: Modern single-session sorting (SortingAnalyzer + SI 0.104). MVP — replaces v1 happy path. ~2-3 PRs.
- **Phase 2**: AnalyzerCuration (metrics + auto-merge + burst-pair consolidated) plus Recording/Sorting recompute verification for storage reclamation. 1-2 PRs.
- **Phase 3**: SessionGroup + ConcatenatedRecording (same-day chronic). 1-2 PRs.
- **Phase 4**: UnitMatch cross-session matching. 1-2 PRs (1 for matcher plugin scaffold + UnitMatch, 1 for tetrode validation).
- **Phase 5**: UX overhaul — `run_v2_pipeline()`, `run_v2_unit_match()`, FigPack, parameter Pydantic validation, notebook rewrite. 1-2 PRs.

Total estimated: 7-10 PRs over the v2 lifetime.
