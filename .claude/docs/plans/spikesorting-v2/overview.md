# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

What v2 touches in the existing tree and what is preserved.

**New module tree** (does not exist yet):

- `src/spyglass/spikesorting/v2/__init__.py` — public surface, mirrors `v1/__init__.py` structure.
- `src/spyglass/spikesorting/v2/recording.py` — `SortGroupV2`, `PreprocessingParameters`, `RecordingSelection`, `Recording`.
- `src/spyglass/spikesorting/v2/sorting.py` — `SorterParameters`, `ArtifactDetectionParameters`, `ArtifactDetection`, `SortingSelection`, `Sorting`.
- `src/spyglass/spikesorting/v2/curation.py` — `CurationV2` (manual labels/merges, registers into `SpikeSortingOutput`).
- `src/spyglass/spikesorting/v2/metric_curation.py` — `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCurationSelection`, `AnalyzerCuration`.
- `src/spyglass/spikesorting/v2/session_group.py` — `SessionGroup`, `SessionGroup.Member`, `ConcatenatedRecording`.
- `src/spyglass/spikesorting/v2/unit_matching.py` — `MatcherParameters`, `UnitMatchSelection`, `UnitMatch` + `UnitMatch.Pair`, `TrackedUnit` + `TrackedUnit.Member`.
- `src/spyglass/spikesorting/v2/figpack_curation.py` — `FigPackCurationSelection`, `FigPackCuration`.
- `src/spyglass/spikesorting/v2/pipeline.py` — `run_v2_pipeline()` sorting orchestrator and `run_v2_unit_match()` cross-session helper.
- `src/spyglass/spikesorting/v2/_params/` — Pydantic models for every Parameters table.

**Existing files MODIFIED**:

- [src/spyglass/spikesorting/spikesorting_merge.py:34-166](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L34-L166) — `SpikeSortingOutput`: add new part `class CurationV2(SpyglassMixinPart)` FK'ing `spyglass.spikesorting.v2.curation.CurationV2`. Update `_get_restricted_merge_ids_v1` analog `_get_restricted_merge_ids_v2` and route through `get_restricted_merge_ids(sources=[...])` so `'v2'` is a valid source string. Modification happens in Phase 1.
- [pyproject.toml](../../../../pyproject.toml) — **Phase 0 does not add direct `pydantic` or `zarr` pins.** Pydantic is required by v2 code, but Phase 0 imports that code only in the SI 0.104 `pytest-v2` job; the default SI 0.99/v1 CI job excludes `tests/spikesorting/v2/` until the prerequisite SI bump lands. SpikeInterface 0.104 already depends on `pydantic` and `zarr>=2.18,<3`; Zarr is a SpikeInterface runtime dependency, not a Spyglass storage default in this plan. Phase 1 uses the existing NWB-HDF5 `AnalysisNwbfile` path; any Zarr storage experiment is a separate follow-up. **The `spikeinterface` pin bump (>=0.104,<0.105) is Phase 0c, a separate prerequisite PR**, NOT done in Phase 0a/0b. The bump happens immediately before Phase 1 lands, alongside the v1 port that makes the bump non-breaking. `mountainsort5>=0.5` and the optional `spikesorting-v2-matching = ["UnitMatchPy>=3.3,<4"]` extra ship in Phase 0c if resolver checks pass. Phase 5 adds the optional `spikesorting-v2-curation` extra only after verifying the current `figpack` + `figpack-spike-sorting` API.

**Existing files PRESERVED unchanged**:

- [src/spyglass/spikesorting/v0/](../../../../src/spyglass/spikesorting/v0/) — entire directory left alone. v0 sorts and `CuratedSpikeSorting` part of `SpikeSortingOutput` remain queryable.
- [src/spyglass/spikesorting/v1/](../../../../src/spyglass/spikesorting/v1/) — entire directory left alone. v1 remains the production path until v2 reaches parity; Phase 5 documents how users choose v1 vs v2, not a sunset trigger.
- [src/spyglass/spikesorting/analysis/v1/group.py](../../../../src/spyglass/spikesorting/analysis/v1/group.py) — `SortedSpikesGroup` keys off `SpikeSortingOutput.merge_id`; adding `CurationV2` as a new part adds rows that this table will already see via the merge.
- [src/spyglass/spikesorting/imported.py](../../../../src/spyglass/spikesorting/imported.py) — untouched. Existing `ImportedSpikeSorting` remains the canonical import path for external/ground-truth Units and continues to register through `SpikeSortingOutput.ImportedSpikeSorting`; v2 does not duplicate it.
- [src/spyglass/decoding/](../../../../src/spyglass/decoding/), [src/spyglass/ripple/](../../../../src/spyglass/ripple/), [src/spyglass/mua/](../../../../src/spyglass/mua/) — all consume `SpikeSortingOutput` or `SortedSpikesGroup`, unaffected by v2 addition.

## Scope and dependency policy

### Goals

- **Modern SpikeInterface API**: All v2 paths use `SortingAnalyzer` (SI ≥0.101), `PreprocessingPipeline` (SI ≥0.103), and current `spikeinterface.curation` primitives.
- **Cross-session unit tracking**: Pluggable matcher backend with UnitMatchPy as the first plugin. Schema (`TrackedUnit`) treats a "biological unit observed across N sessions" as a first-class queryable entity. DeepUnitMatch swappable via the same plugin slot (future).
- **Chronic / large recording support**: Session groups, virtual concatenation (same-day default; multi-day opt-in behind `allow_multi_day=True` with an explicit motion-correction preset — see resolved decision #4 below), lazy streaming, sparse SortingAnalyzers, motion correction (rigid_fast default; DREDge available as an opt-in preset). 30 kHz × 64 ch × 24 h recordings produce one canonical NWB-resident recording artifact per recording (see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format)), not per analysis stage. For cross-day analyses the recommended path is Phase 4 sort-then-match (UnitMatch), not multi-day concat.
- **First-class unit → brain region traceability**: every curated unit row carries (at minimum) the peak channel and the brain region(s) for that channel, persisted at sort time and resolvable via a single query on `CurationV2` or `SpikeSortingOutput`. v1's `CurationV1.get_sort_group_info` samples *one* electrode per sort group; that's wrong for multi-region polymer probes. v2 walks every electrode in the sort group, persists per-unit peak-channel and region, and exposes `Sorting.get_unit_brain_regions(key) -> pd.DataFrame` and the same accessor on `CurationV2`. Brain region tracing must work end-to-end through `SpikeSortingOutput.get_sort_group_info` and through `TrackedUnit` (per-session brain regions for matched units).
- **Reduced UX friction**: `run_v2_pipeline(nwb_file_name, sort_group_id, interval_list_name, preset=...)` runs the standard single-session flow, while `run_v2_pipeline(concat_session_group_owner=..., concat_session_group_name=..., preset=...)` runs the concatenated sort-and-curate flow; both return the final `merge_id` plus a manifest. `run_v2_unit_match(session_group_owner, session_group_name, curation_choices=...)` provides the separate cross-session sort-then-match helper without overloading the concat path. Pydantic models validate parameter dicts at insert time. Notebook walkthrough drops from 35 cells to ~8.
- **Drop-in for existing downstream consumers**: New `SpikeSortingOutput.CurationV2` part. Decoding, ripple, MUA, `SortedSpikesGroup` work unchanged.
- **Explicit v1 parity boundaries**: [feature-parity.md](feature-parity.md) is the contract for which v1 workflows are preserved, which are replaced by successor tables, and which departures are intentional.
- **Team-based sorting preserved**. v2 retains v1's `LabTeam` FK structure: `RecordingSelection`, `SessionGroup.Member`, and `run_v2_pipeline(team_name=...)` all carry team ownership. `cautious_delete` continues to check team membership; per-team data partitioning, shared-team sorting, and team-based access control all work the same way they did in v1. Two teams running an identical sort get distinct `recording_id` UUIDs and distinct `AnalysisNwbfile` rows — preventing the v0 filesystem-overwrite bug ([#133](https://github.com/LorenFrankLab/spyglass/issues/133)).
- **Zero-migration policy** (binding): all v2 tables must be designed in their final shape in the phase that introduces them. **No `alter()` calls** are permitted across phases. New tables added in later phases are fine; modifying any existing v2 table's primary key, foreign keys, or column types is not. This includes anticipated needs: Phase 1's `SortingSelection` already supports both single and concatenated input paths; Phase 1's `Sorting.Unit` already supports per-unit brain region; Phase 1's `CurationV2` already supports the columns that subsequent phases consume. The plan documents which schema decisions are forward-compatibility decisions vs. clean-slate decisions.
- **v0/v1 coexistence**: Both legacy pipelines remain functional throughout the v2 lifetime. **No v1 sunset is planned in this work**.

### Non-Goals

- **Re-sorting existing v1 data.** v2 adds the new pipeline; users may run it on new sessions but the plan never invalidates or migrates existing v1 sorts.
- **A new merge table.** v2 plugs into the existing `SpikeSortingOutput`; adding `MergeV2Output` would force downstream code to choose between merges. Not in scope.
- **A v2-specific imported-sorting table.** Imported external Units remain in the existing `ImportedSpikeSorting` table and merge part. v2 compares against or coexists with imported Units; it does not absorb them into `CurationV2` lineage.
- **Removing v0 or v1 source.** Both stay in `src/spyglass/spikesorting/{v0,v1}/`. Documentation will mark them legacy but they remain populate-runnable. **No v1 sunset is planned here.**
- **DeepUnitMatch.** Pluggable via the Phase 4 matcher protocol but not implemented; future work.
- **Custom UI / web app.** All curation UI uses upstream FigPack; we do not build a Spyglass-specific viewer.
- **Cross-region / cross-probe matching.** UnitMatch assumes a single probe across sessions in a session group. Multi-probe handling is out of scope.

### Dependency policy

| Dependency | From | To | When | Reason |
| --- | --- | --- | --- | --- |
| `spikeinterface` | `>=0.99.1,<0.100` | `>=0.104,<0.105` | **Phase 0c prerequisite PR before Phase 1** | SortingAnalyzer, PreprocessingPipeline, modern curation primitives. Bumping this breaks v1's `extract_waveforms` calls — Phase 0c ports v1 to `create_sorting_analyzer` so the bump is non-breaking. |
| `mountainsort5` | (absent) | `>=0.5` | Phase 0c prerequisite PR | Additional sorter; v2 keeps MS4 too. |
| `unitmatchpy` | (absent) | `>=3.3,<4` | Phase 4 (optional extra `spikesorting-v2-matching`) | Cross-session matching. Phase 4a must verify resolver compatibility because current UnitMatchPy metadata requires `numpy<2.0`. |
| `figpack` | (absent) | version pinned after feasibility check | Phase 5 (optional extra `spikesorting-v2-curation`) | Core FigPack package for v2 curation UI. |
| `figpack-spike-sorting` | (absent) | version pinned after feasibility check | Phase 5 (same optional extra) | Spike-sorting extension package, imported as `figpack_spike_sorting`; Phase 5 must verify current API and curation-state round trip before schema finalization. |
| `MEArec` | (absent) | `>=1.9` | Phase 0 (optional extra `spikesorting-v2-validation`) | Ground-truth fixture generation for v2 validation. |
| `neuroconv[mearec]` | (absent) | (latest) | Phase 0 (same extra) | MEArec → NWB conversion via `MEArecRecordingInterface`. |
| `mountainsort4` | present | unchanged | n/a | v1 + v2 both ship MS4 wrapper. |

SpikeInterface 0.104's own dependency set provides `pydantic`, `zarr>=2.18,<3`, and `numcodecs<0.16.0`. Phase 0's v2-only tests run in the SI 0.104 development job so the Pydantic models are importable before the global pin bump. Spyglass should not duplicate those pins unless the SI upgrade PR exposes a resolver/runtime issue that needs an explicit temporary constraint.

All v2 additions are made via the existing `pyproject.toml`. `mountainsort5` is a required dependency after the prerequisite PR; optional deps such as `unitmatchpy`, validation fixtures, and FigPack gate at import time with a clear error pointing to the install command. Phase 4a must resolve-test the `spikesorting-v2-matching` extra before it lands because UnitMatchPy currently pins `numpy<2.0`; do not add the extra if it forces an incompatible NumPy downgrade in the SI 0.104 environment. Phase 5 must verify both `figpack` and `figpack-spike-sorting` before adding `spikesorting-v2-curation`. Storage benchmarks are optional follow-up work and do not gate Phase 1.

## Metrics

How we'll know v2 is working:

- **Ground-truth tests (Phase 1)**: Run v2 against MEArec-generated polymer-probe and Neuropixels fixtures with known planted units; compute precision/recall via `spikeinterface.comparison.compare_sorter_to_ground_truth`. Polymer is the primary lab-relevant fixture (per Chung et al. 2019, the Frank-lab standard). minirec serves only as a plumbing regression guard (test_v2_minirec_plumbing) — it is too short to contain real spikes. v1 parity moves to the env-var real-data path (`SPIKESORTING_V2_REAL_NWB_PATH`); no minirec parity test ships.
- **Memory and runtime (Phase 3)**: Single-channel-group, 30 kHz, 1 h recording must complete `Sorting.populate()` in <10 min and peak RSS <8 GB on a 16-core machine with default job kwargs. Smoke-test on 5 min slice first.
- **Notebook cells (Phase 5)**: The end-to-end notebook for a single sort drops from 35 code cells to ≤10. Counted manually before merge.
- **Downstream non-regression (Phase 1)**: A `SortedSpikesGroup` built from a v2 `CurationV2` row produces the same `get_spike_times` shape and units as one built from an equivalent v1 row.
- **UnitMatch polymer validation gate (Phase 4)**: On the MEArec-generated 2-session polymer fixture (Phase 4b task; planted shared templates across the 4-shank polymer probe), `test_v2_unitmatch_polymer_mearec_ground_truth` asserts AUC > 0.85 of match probability against ground-truth template correspondence. Polymer is the gate because it is the Frank-lab standard probe. Supplementary Neuropixels, tetrode, or real-data validation belongs in follow-up PRs.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| MountainSort output stochasticity makes parity testing noisy. | Phase 0 captures v1 baseline with `clusterless_thresholder` (deterministic, seeded) as the gold-standard parity test. MS4-vs-MS5 comparisons are bounded by qualitative metrics (unit count order, firing-rate distribution shape) and explicitly noted. |
| SortingAnalyzer folders are large for chronic recordings (50+ GB possible). | Phase 1 stores analyzer folder paths in DataJoint, **not** content. Phase 2 ports the v1 `RecordingRecompute` pattern into v2-specific `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*` tables for verified storage reclamation. Sparse=True default cuts size 5-10×. |
| UnitMatch's published validation is Neuropixels-heavy and may not transfer to Frank-lab polymer/tetrode data. | Phase 4 gates shipping on a MEArec polymer-probe fixture, because polymer is the lab-relevant standard for this workflow. Neuropixels and tetrode AUCs are recorded as informational checks; tetrode users are not promised production matching unless the informational result supports it. |
| Adding a 4th part to `SpikeSortingOutput` may surface latent merge bugs. | Phase 1 includes integration tests against `SortedSpikesGroup`, decoding, ripple, MUA with a v2-sourced merge_id. |
| `set_group_by_shank()` from v1 silently overwrites sort groups — v2 must not perpetuate this. | `SortGroupV2.set_group_by_shank()` raises if rows already exist for the session; destructive replacement requires `delete_existing_entries=True, confirm=True` after reviewing a `DeletionPreview`. Phase 1 task. |
| Pydantic schema migration risk: existing parameter blobs may not validate. | All Pydantic models are introduced fresh in v2 Lookup tables; v1 Lookup blobs remain unvalidated. No migration of v1 params. |
| Concatenation across days (out of scope but tempting). | `SessionGroup.Member` carries a `recording_date` field; `ConcatenatedRecording.populate()` raises if members span multiple dates without an explicit `allow_multi_day=True` parameter row. |
| Pipeline helpers may hide reproducibility. | `run_v2_pipeline()` and `run_v2_unit_match()` return manifest dicts of every `(table, key)` they touched. Notebooks print them. Phase 5 task. |

## Rollout Strategy

v2 is **strictly additive** for the lifetime of this plan. No feature flags, no parallel-running shims, no deprecation phase needed inside this plan.

- v1 stays as the documented production path until v2 reaches parity (defined as: Phases 1–5 merged, parity tests green on lab production data for one cycle).
- After parity is achieved, the documentation root (`docs/src/Features/`) gets a banner pointing users to v2 for new work; v1 docs remain.
- **v1 source code removal is explicitly out of scope** for this plan. Phase 5 documents how users choose v1 vs v2; it does not define a deletion or sunset trigger.
- Each phase ships as a stand-alone PR; users upgrading midway can use whatever phases are merged at the time. Phase 1 alone is usable in production.

## Resolved Design Decisions

These were open questions; user has decided. Documented here so subsequent phase implementers don't re-litigate.

1. **MountainSort 4 stays in v2.** Add `mountainsort4` to `SorterParameters` defaults alongside `mountainsort5`, `kilosort4`, `spykingcircus2`, `tridesclous2`, `clusterless_thresholder`. MS4 wrapper still exists in SpikeInterface 0.104; the v1 sorter_params dicts can be ported with minimal changes (notably removing the `tempdir` field-mutation hack). Pydantic schema for MS4 ships in Phase 1's `_params/sorter.py`.
2. **FigPack is the v2 curation UI.** Implemented in Phase 5 as the primary curation interface for v2 sorts. FigURL is not extended to v2; FigURL stays available for v1 data only. **Hard dependency on FigPack feasibility** — Phase 5 begins with a FigPack feasibility check (see Phase 5 first tasks); if FigPack is genuinely unusable at implementation time, the plan stops there and surfaces the blocker to the project owner rather than silently shipping a degraded fallback.
3. **DeepUnitMatch deferred.** Phase 4's `MatcherProtocol` slot accepts it as a future plugin; not implemented in this plan.
4. **Multi-day concatenation is supported by the schema, but is opt-in and not the recommended cross-day path.** `SessionGroup.create_group(..., allow_multi_day=True)` is required to insert members spanning ≥2 dates; the default rejects multi-day inputs with a clear pointer to Phase 4 sort-then-match. `ConcatenatedRecording` does NOT auto-dispatch DREDge for multi-day groups — the caller picks an explicit motion-correction preset. **Sort-then-match (Phase 4 UnitMatch) is the recommended cross-day workflow, and Phase 4's gating validation is the Frank-lab polymer-probe fixture.** Neuropixels, tetrode, and real-data UnitMatch checks are useful follow-up validation, not Phase 4 gates. Multi-day tetrode sorting is not a current Frank-lab use case, so neither multi-day tetrode concat nor tetrode UnitMatch is treated as a gating workflow. Both remain schema-supported (zero migration).
5. **Zero schema migration.** All v2 tables are designed in their final shape in Phase 1 (or in the phase that adds them). No `alter()` calls across phases. Concretely: Phase 1's `SortingSelection` declares two nullable typed FKs (`recording_id` → `Recording`, `concat_recording_id` → `ConcatenatedRecording`) with XOR enforced in `insert_selection()`; Phase 1's `Sorting.Unit`, `CurationV2.Unit`, and `CurationV2.UnitLabel` part tables are present from day one; Phase 1's `CurationV2` columns are final. The "two paths for `SortingSelection`" formerly in Open Question #8 is collapsed to one design.

## Remaining Open Questions

These are smaller, mostly orthogonal decisions where the plan's default is documented but the user can override.

1. **Pydantic model versioning?** *Current best-answer*: each Lookup table includes a `params_schema_version: int` secondary attribute. When a Pydantic model changes shape, increment the schema version; the validator dispatches by version. Documented in [shared-contracts.md](shared-contracts.md).
2. **Sorter container support (Docker / Singularity) in v2?** *Current best-answer*: defer. v2 ships without container wrappers; users who need MATLAB sorters (Kilosort 2/2.5/3, IronClust) can use the SI `docker_image=True` kwarg by inserting a custom `SorterParameters` row. Re-evaluate after Phase 5.
3. **Curation label enum enforcement?** *Current best-answer*: v2 uses a `CurationLabel` Pydantic Enum (`accept`, `mua`, `noise`, `artifact`, `reject`). Inserts validate against the enum at the API surface but raw `dj.Manual` inserts that bypass the helper remain free-form (DataJoint doesn't support enums on blob columns).
4. **Parity baseline SI version mismatch.** *Current best-answer*: v1 parity is no longer performed against minirec (minirec has no real spikes). v1 parity now runs only when `SPIKESORTING_V2_REAL_NWB_PATH` is set — `test_v2_real_data_v1_parity` runs v1 + v2 against the user-provided NWB and compares per-unit spike times. Tolerances: `clusterless_thresholder` exact within ±1 sample; stochastic sorters within ±50% unit count and ±30% median firing rate. If v1 cannot run under SI 0.104 (because the prerequisite port to `create_sorting_analyzer` is not yet complete), the test skips with an explicit message — it does not silently relax to a meaningless tolerance.

## Issue Traceability — v1 GitHub issues v2 closes

Mining of [LorenFrankLab/spyglass GitHub issues](https://github.com/LorenFrankLab/spyglass/issues?q=is%3Aissue+label%3A%22spike+sorting%22) surfaced concrete v1 bugs and friction points. The following are explicitly addressed by v2 design decisions:

| v1 Issue | Problem | v2 fix |
|---|---|---|
| [#1394](https://github.com/LorenFrankLab/spyglass/issues/1394) | `get_sort_group_info` shows only one electrode → wrong region on multi-region polymer probes | `Sorting.Unit` + `CurationV2.Unit` persist per-unit peak channel; `get_sort_group_info` walks all electrodes |
| [#1532](https://github.com/LorenFrankLab/spyglass/issues/1532), [#1154](https://github.com/LorenFrankLab/spyglass/issues/1154) | `CurationV1` fails on zero-unit sortings (`KeyError: 'spike_times'`) | Empty/NaN/Boundary Invariant 1 in shared-contracts; covered across Phase 1 `test_v2_empty_sorting_phase1`, Phase 2 `test_analyzer_curation_zero_unit_sorting`, and Phase 5 `test_figpack_zero_unit_sorting` |
| [#1556](https://github.com/LorenFrankLab/spyglass/issues/1556) | `FigURLCuration` fails on NaN-bearing metrics (low-spike units) | Empty/NaN/Boundary Invariant 2; `_sanitize_for_json` in `AnalyzerCuration.make()`; Phase 2 `test_metric_nan_round_trip` |
| [#1558](https://github.com/LorenFrankLab/spyglass/issues/1558) | Spike at recording boundary → "exceeds duration" error | Empty/NaN/Boundary Invariant 3; explicit boundary-tolerance test in Phase 1 |
| [#1437](https://github.com/LorenFrankLab/spyglass/issues/1437) | `SpikeSorting.populate` fails when raw NWB has existing Units table | Phase 1 `Sorting.make()` uses whitelist construction (does NOT copy raw NWB) |
| [#1133](https://github.com/LorenFrankLab/spyglass/issues/1133), [#1585](https://github.com/LorenFrankLab/spyglass/issues/1585) | Recording timestamps wrong / silent truncation | Phase 1 `Recording.make()` asserts saved time range covers requested interval; `RecordingTruncatedError` |
| [#928](https://github.com/LorenFrankLab/spyglass/issues/928) | Artifact detection per sort group misses behavioral artifacts visible across groups | Phase 1 `SharedArtifactGroup` opt-in path |
| [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513) | v0 `AutomaticCuration.get_labels` label-list aliasing across units | Phase 2 label-list isolation invariant + `test_label_list_isolation` |
| [#939](https://github.com/LorenFrankLab/spyglass/issues/939) | `CurationV1` doesn't track metric source | Phase 1 `CurationV2.metrics_source` enum column |
| [#528](https://github.com/LorenFrankLab/spyglass/issues/528) | No cross-day sorting | Phase 3 `SessionGroup` + `ConcatenatedRecording`; Phase 4 UnitMatch sort-then-match for cross-day |
| [#1436](https://github.com/LorenFrankLab/spyglass/issues/1436) | Need SI 0.103+ compatibility | Phase 0 prerequisite SI 0.104 upgrade; Phase 1 SortingAnalyzer |
| [#1286](https://github.com/LorenFrankLab/spyglass/issues/1286) | UX redesign request | Phase 1 minimal `run_v2_pipeline()` + Phase 5 full presets, `run_v2_unit_match()`, and FigPack |
| [#1530](https://github.com/LorenFrankLab/spyglass/issues/1530), [#1512](https://github.com/LorenFrankLab/spyglass/issues/1512), [#1504](https://github.com/LorenFrankLab/spyglass/issues/1504), [#1215](https://github.com/LorenFrankLab/spyglass/issues/1215) | `FigURLCuration` brittleness (opaque KeyError, sortingview API drift) | Phase 5 replaces FigURL with FigPack for v2 curations |
| [#133](https://github.com/LorenFrankLab/spyglass/issues/133) | When two teams run an otherwise-identical sort, v0's `SpikeSortingRecording.make()` deletes an existing recording folder before writing its own — silent filesystem destruction of another team's data ([v0/spikesorting_recording.py:431](../../../../src/spyglass/spikesorting/v0/spikesorting_recording.py#L431)) | v2's preprocessed recording lives inside its own UUID-keyed `AnalysisNwbfile` row, and `RecordingSelection.insert_selection` includes `team_name` in its existence check — so two teams running the same sort get distinct `recording_id`s and distinct `analysis_file_name`s. No deletion of another team's files is possible. (v1 already fixed the worst of this; v2 inherits the fix and removes the separate-filesystem-cache attack surface entirely by keeping the canonical artifact NWB-resident.) |
| [PR #1438](https://github.com/LorenFrankLab/spyglass/pull/1438) (DRAFT) | v1 `SortGroup` only supports `set_group_by_shank`; Berke Lab and other labs need to group by arbitrary electrode-table columns (e.g., `intan_channel_number`) | v2 `SortGroupV2.set_group_by_electrode_table_column(column, groups, ...)` ports the PR's design directly. Both shank and column-based constructors share PR #1438's existing-entry collision-handling pattern (delete-and-replace OR non-overlapping `sort_group_ids`), replacing both v1's silent-overwrite footgun and the earlier-draft `force=True` design. |

## Explicitly NOT fixed by v2 (separate v1-side patches required)

These bugs surfaced in the issue sweep but are NOT in v2's surface area. The plan should not absorb them; the implementer notes them here so an executor isn't tempted to scope-creep into them:

- [#1581](https://github.com/LorenFrankLab/spyglass/issues/1581) — `ImportedSpikeSorting.fetch_nwb` `pd.concat` index collision. Fix in `imported.py` (not in `spikesorting/v2/`): use `ignore_index=True` and add `nwb_file_name` + `unit_id` as MultiIndex levels.
- [#1273](https://github.com/LorenFrankLab/spyglass/issues/1273) — `UnitWaveformFeatures` assumes contiguous unit IDs. Fix in `decoding/v1/waveform_features.py`; one-line indexing change. v2's `Sorting.Unit` design preserves non-contiguous unit IDs, but the v1 `UnitWaveformFeatures` table is not rewritten.
- [#638](https://github.com/LorenFrankLab/spyglass/issues/638), [#1282](https://github.com/LorenFrankLab/spyglass/issues/1282) — Opaque `1217 IntegrityError` on cascade-delete. Database-config / upstream DataJoint issue affecting all merge-bearing tables. v2 cannot fix from the schema side.
- [#1159](https://github.com/LorenFrankLab/spyglass/issues/1159) — `get_spiking_v1_merge_ids` invalid restriction. Bug in `spikesorting_merge.py`'s v1 helper; v2's `get_restricted_merge_ids(sources=['v2'])` path is independent.

The implementer of any v2 phase should resist scope-creeping into these. They are tracked separately.

## Estimated Effort

LOC sanity check, not a time estimate:

- Phase 0: ~300 LOC (mostly scaffolding files + one baseline-capture script).
- Phase 1: ~1500 LOC across `recording.py`, `sorting.py`, `curation.py` + part-table addition to `spikesorting_merge.py` + tests.
- Phase 2: ~900 LOC for `metric_curation.py` + the curation rules engine + tests.
- Phase 3: ~700 LOC for `session_group.py` (including `ConcatenatedRecording` make()) + tests.
- Phase 4: ~1100 LOC for matcher plugin scaffold + UnitMatch implementation + `TrackedUnit` + tests.
- Phase 5: ~800 LOC for `pipeline.py` helpers + Pydantic models + FigPack curation + notebook rewrite.

Total v2 footprint: roughly 5300 LOC across 6 PRs. Compared to v1 at ~6000 LOC in `src/spyglass/spikesorting/v1/` (counted via `wc -l`), v2 is similar in size despite richer capability because the per-stage scaffolding is more consolidated.
