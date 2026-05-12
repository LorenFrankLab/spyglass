# Spike Sorting v3 — Research Notes & Hypothesis Tree

Working document accumulating evidence and design rationale for the v3 design.
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

🟢 V1 uses the **WaveformExtractor** API (`si.extract_waveforms`, `si.load_waveforms`) — removed in SI 0.101. This is the #1 forcing function for v3.
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

🟡 **Zarr<3.0 still required** as of 0.104 (issue #4014 tracks v3 migration).

---

## 5. Cross-Session Strategies

| | Concatenate-and-sort | Sort-then-UnitMatch |
|---|---|---|
| Timescale | Same-day with breaks | Days–weeks |
| Drift handling | `correct_motion` on concat | UnitMatch rigid-shift estimate |
| Output identity | Sorter assigns same unit ID across span | Match probability + FDR per pair |
| Re-running on new session | Re-sort full concat | Incremental match |
| Tetrode compatibility | OK (drift correction ≈ no-op) | 🔴 Open question — UnitMatch validated on Neuropixels only |
| Maturity | KS4/MS5 + DREDge | UnitMatchPy 3.3.1 active |

🟢 **UnitMatchPy** (https://pypi.org/project/UnitMatchPy/) is the maintained Python port. Includes `UMPy_spike_interface_demo.ipynb`.
🟢 **DeepUnitMatch** lives in the same UnitMatchPy repo (`DeepUnitMatch` subpackage) — pretrained model for inference, drop-in via same data interface.
🔴 **Tetrode UnitMatch validation needs empirical check** before production use on Frank lab data.

🟢 **DECISION RULE**:
- v3 supports BOTH paths via plugin matcher API.
- MVP is sort-then-UnitMatch (more general, incremental, introspectable).
- Concat-and-sort is built on top: `ConcatenatedRecording` virtual table → existing sort path; the "match" is then identity-mapping from sort output.

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

🟢 **Schema naming**: v3 lives in shared `spikesorting` schema (lab-shared module in `SHARED_MODULES`).

🟢 **Tier discipline** for each new pipeline step:
- Lookup (Parameters, contents-baked)
- Manual (Selection, user-inserted, FK to Params)
- Computed (with `make()`, FK to Selection)
- Merge ONLY for multi-source convergence.

🟢 **Use existing `SpikeSortingOutput`** as the downstream entry point. Add new part `SpikeSortingOutput.CurationV3` rather than creating a new merge table. `SortedSpikesGroup` keeps working.

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

🟢 **`set_group_by_shank()` issue (#11)** — overwrites existing sort groups silently, cascades downstream. v3 should be additive or warn loudly.

---

## 8. Architecture Hypothesis Tree

### H1: Reuse `SpikeSortingOutput` merge table → 🟢 ADOPT
- **Pro**: `SortedSpikesGroup`, decoding, MUA, ripple — all downstream pipelines keep working without changes.
- **Pro**: Users with v1 sorts and v3 sorts in the same database can mix.
- **Con**: Couples us to the existing merge schema (`merge_id`-only PK).
- **Verdict**: ADOPT. Add `SpikeSortingOutput.CurationV3` part table.

### H2: SortingAnalyzer-first storage → 🟢 ADOPT
- Single source of truth for waveforms, templates, metrics, locations.
- Persisted as `binary_folder` (fast local) or Zarr (archival/cloud).
- v3 `Sorting` table writes `SortingAnalyzer` folder + lightweight units NWB; downstream tables read from analyzer extensions.

### H3: Parameters as Pydantic-validated schemas → 🟢 ADOPT
- Lookup tables get `params: blob` typed via Pydantic models.
- `insert_selection` validates params on insert (`PydanticModel.model_validate(...)`).
- Eliminates "typo at populate" failure mode.
- Backward compatible: blob in DB stays a dict.

### H4: Pipeline orchestration via `run_v3_pipeline()` convenience function → 🟢 ADOPT
- Single entry point: takes raw NWB + parameter names, returns final `merge_id`.
- Internally: inserts selection rows, populates each stage, registers with `SpikeSortingOutput`.
- Idempotent: re-run finds existing rows, doesn't duplicate.
- Notebook becomes ~5 cells instead of 35.

### H5: Cross-session as plugin matcher → 🟢 ADOPT
- `MatcherParameters` Lookup has `matcher: enum('unitmatch', 'deepunitmatch', 'concat_identity')`.
- `UnitMatch` Computed dispatches to matcher backend.
- Per-session sortings remain valid; matching is additive.
- Concat path: `ConcatenatedRecording` builds virtual recording → existing `Sorting` table → emits identity-mapping matches.

### H6: Session group table for chronic recordings → 🟢 ADOPT
- `SessionGroup` (Manual, master) + `SessionGroup.Member` (Part).
- Member declares `(nwb_file_name, interval_list_name, sort_group_id)`.
- Reusable across pipeline branches: same group used for concat-and-sort AND for sort-then-match.
- Matches `SortedSpikesGroup` shape (skill recommended pattern).

### H7: Replace `MetricCuration` + `BurstPair` with `AnalyzerCuration` → 🟡 PROPOSE
- Single Computed table walks `SortingAnalyzer` extensions, applies auto-curation rules, optional auto-merge.
- Consolidates `MetricCuration` + `BurstPair` (which currently both compute waveform similarity separately).
- **Risk**: BurstPair has bespoke visualizations (`plot_by_sort_group_ids` etc.) — must preserve.
- **Verdict**: ADOPT but keep visualization helpers. The two tables compute similar things twice today.

### H8: FigPack instead of (or alongside) FigURL → 🟡 PROPOSE
- FigPack is positioned by SI 0.104 as the FigURL successor.
- Maturity is the open question; need to confirm install/auth path works for our lab.
- **Verdict**: ADOPT as new path, keep FigURL parity for one release.

### H9: Don't break v0/v1 → 🟢 ADOPT
- v0 and v1 remain in tree, populate-runnable for existing data.
- v3 is additive. Migration tool: `v3_from_v1_curation(curation_v1_key)` to register an existing v1 curation as a `v3` row for downstream uniformity. NOT a re-sort — just a structural conversion.

### H10: Validation against v1 → 🟢 ADOPT
- Smoke-test on `minirec` (existing v1 test fixture, 9–10 s) — run v3 with MS5, then with v3-wrapped MS4-compatible params, compare unit counts and rough spike-time distributions to v1 baseline.
- MountainSort is stochastic so exact spike-time match is not expected; tolerance is "same order of magnitude unit count, similar firing-rate distribution".
- 🔴 **Need a non-stochastic sorter for tight equivalence**: clusterless thresholder (`detect_peaks`) is deterministic given seed → use that as the gold-standard reproducibility check.

---

## 9. Self-Critique of Architecture

**Risk 1**: Adding a fourth part table to `SpikeSortingOutput` may bloat downstream merge resolution.
- Mitigation: `SortedSpikesGroup` already handles N parts; one more is fine.

**Risk 2**: Plugin matcher API may leak SpikeInterface object details into DataJoint blobs.
- Mitigation: serialize matcher inputs/outputs to AnalysisNwbfile via `.build()`; DB stores only the analysis_file_name.

**Risk 3**: `run_v3_pipeline()` convenience function hides reproducibility — what got inserted?
- Mitigation: function returns a manifest dict with every `(selection_table, key)` it touched. Notebook can print it.

**Risk 4**: Concat-and-sort across days breaks for KS4 (sorter assumes contiguous time). Mitigation:
- Default DREDge preset before concat; gate concat to single-day groups in MVP.
- Add `concat_safe: bool` flag on SessionGroup that requires explicit user override for multi-day.

**Risk 5**: 5 phases is a lot. Phasing failure cascades.
- Mitigation: Phase 1 (Modern Single-Session) is independently shippable and replaces v1's biggest pain (WaveformExtractor → SortingAnalyzer migration). Phases 2-5 are optional additions.

**Risk 6**: SortingAnalyzer folder management at scale — chronic recording analyzer can be 50 GB+.
- Mitigation: Same `RecordingRecompute` pattern from v1 — recompute table for analyzer cache; delete + recompute for storage reclamation.

**Risk 7**: UnitMatch may not transfer cleanly from Neuropixels-published examples to Frank-lab polymer/tetrode use cases.
- Mitigation: Phase 4 gates on a MEArec polymer-probe fixture, because polymer is the lab-relevant standard for this workflow. Neuropixels and tetrode AUCs are recorded as informational checks; if tetrode features collapse, tetrode users are routed to concat-and-sort or future validation work rather than promised UnitMatch support.

---

## 10. Open Questions for User

1. **Sorter prioritization**: MS5 is the v1 successor for MountainSort users, KS4 is the field standard for Neuropixels. Should v3 ship with both or commit to MS5+KS4 only (deprecate MS4)?
2. **Schema location**: live in `spyglass.spikesorting.v3` (parallel to v0/v1) ✅ confirmed by directory convention. Stays in shared `spikesorting` schema.
3. **Migration policy**: keep v1 alive indefinitely? Sunset after N releases?
4. **DeepUnitMatch in MVP?** Recommend "no" — Phase 4.1 enhancement after UnitMatch baseline.
5. **FigPack vs FigURL**: should v3 default to FigPack or keep FigURL primary until FigPack proves?
6. **Concat-and-sort across days**: support in MVP or punt? Recommendation: same-day only in Phase 3; multi-day with DREDge precorrection is Phase 6 future work.

---

## 11. Phasing Decision

5 independently shippable phases, plus Phase 0 scaffolding:

- **Phase 0**: Project scaffold, deps, baseline-capture infra. ~1 PR.
- **Phase 1**: Modern single-session sorting (SortingAnalyzer + SI 0.104). MVP — replaces v1 happy path. ~2-3 PRs.
- **Phase 2**: AnalyzerCuration (metrics + auto-merge + burst-pair consolidated). 1 PR.
- **Phase 3**: SessionGroup + ConcatenatedRecording (same-day chronic). 1-2 PRs.
- **Phase 4**: UnitMatch cross-session matching. 1-2 PRs (1 for matcher plugin scaffold + UnitMatch, 1 for tetrode validation).
- **Phase 5**: UX overhaul — `run_v3_pipeline()`, FigPack, parameter Pydantic validation, notebook rewrite. 1-2 PRs.

Total estimated: 7-10 PRs over the v3 lifetime.
