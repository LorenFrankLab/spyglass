# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

What v3 touches in the existing tree and what is preserved.

**New module tree** (does not exist yet):

- `src/spyglass/spikesorting/v3/__init__.py` — public surface, mirrors `v1/__init__.py` structure.
- `src/spyglass/spikesorting/v3/recording.py` — `SortGroupV3`, `PreprocessingParameters`, `RecordingSelection`, `Recording`.
- `src/spyglass/spikesorting/v3/sorting.py` — `SorterParameters`, `ArtifactDetectionParameters`, `ArtifactDetection`, `SortingSelection`, `Sorting`.
- `src/spyglass/spikesorting/v3/curation.py` — `CurationV3` (manual labels/merges, registers into `SpikeSortingOutput`).
- `src/spyglass/spikesorting/v3/metric_curation.py` — `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCurationSelection`, `AnalyzerCuration`.
- `src/spyglass/spikesorting/v3/session_group.py` — `SessionGroup`, `SessionGroup.Member`, `ConcatenatedRecording`.
- `src/spyglass/spikesorting/v3/unit_matching.py` — `MatcherParameters`, `UnitMatchSelection`, `UnitMatch` + `UnitMatch.Pair`, `TrackedUnit` + `TrackedUnit.Member`.
- `src/spyglass/spikesorting/v3/figpack_curation.py` — `FigPackCurationSelection`, `FigPackCuration`.
- `src/spyglass/spikesorting/v3/pipeline.py` — `run_v3_pipeline()` orchestrator.
- `src/spyglass/spikesorting/v3/_params/` — Pydantic models for every Parameters table.

**Existing files MODIFIED**:

- [src/spyglass/spikesorting/spikesorting_merge.py:34-150](src/spyglass/spikesorting/spikesorting_merge.py#L34-L150) — `SpikeSortingOutput`: add new part `class CurationV3(SpyglassMixinPart)` FK'ing `spyglass.spikesorting.v3.curation.CurationV3`. Update `_get_restricted_merge_ids_v1` analog `_get_restricted_merge_ids_v3` and route through `get_restricted_merge_ids(sources=[...])` so `'v3'` is a valid source string. Modification happens in Phase 1.
- [pyproject.toml](pyproject.toml) — **Phase 0** adds only `pydantic>=2.0` and `zarr<3.0` to `dependencies`. **The `spikeinterface` pin bump (>=0.104,<0.105) is a separate prerequisite work item** (see Phase 0's "SI 0.104 upgrade gating" tasks), NOT done in Phase 0. The bump happens immediately before Phase 1 lands, alongside the v1 port that makes the bump non-breaking. `mountainsort5>=0.5` and the optional `spikesorting-v3-matching = ["unitmatchpy>=3.3"]` extra ship in the same prerequisite PR.

**Existing files PRESERVED unchanged**:

- [src/spyglass/spikesorting/v0/](src/spyglass/spikesorting/v0/) — entire directory left alone. v0 sorts and `CuratedSpikeSorting` part of `SpikeSortingOutput` remain queryable.
- [src/spyglass/spikesorting/v1/](src/spyglass/spikesorting/v1/) — entire directory left alone. v1 remains the production path until v3 reaches parity; Phase 5 documents how users choose v1 vs v3, not a sunset trigger.
- [src/spyglass/spikesorting/analysis/v1/group.py](src/spyglass/spikesorting/analysis/v1/group.py) — `SortedSpikesGroup` keys off `SpikeSortingOutput.merge_id`; adding `CurationV3` as a new part adds rows that this table will already see via the merge.
- [src/spyglass/spikesorting/imported.py](src/spyglass/spikesorting/imported.py) — untouched. Existing `ImportedSpikeSorting` remains the canonical import path for external/ground-truth Units and continues to register through `SpikeSortingOutput.ImportedSpikeSorting`; v3 does not duplicate it.
- [src/spyglass/decoding/](src/spyglass/decoding/), [src/spyglass/ripple/](src/spyglass/ripple/), [src/spyglass/mua/](src/spyglass/mua/) — all consume `SpikeSortingOutput` or `SortedSpikesGroup`, unaffected by v3 addition.

## Scope and dependency policy

### Goals

- **Modern SpikeInterface API**: All v3 paths use `SortingAnalyzer` (SI ≥0.101), `PreprocessingPipeline` (SI ≥0.103), and current `spikeinterface.curation` primitives.
- **Cross-session unit tracking**: Pluggable matcher backend with UnitMatchPy as the first plugin. Schema (`TrackedUnit`) treats a "biological unit observed across N sessions" as a first-class queryable entity. DeepUnitMatch swappable via the same plugin slot (future).
- **Chronic / large recording support**: Session groups, virtual concatenation (same-day default; multi-day opt-in behind `allow_multi_day=True` with an explicit motion-correction preset — see resolved decision #4 below), lazy streaming, sparse SortingAnalyzers, motion correction (rigid_fast default; DREDge available as an opt-in preset). 30 kHz × 64 ch × 24 h recordings produce one materialized binary cache per recording, not per analysis stage. For cross-day analyses the recommended path is Phase 4 sort-then-match (UnitMatch), not multi-day concat.
- **First-class unit → brain region traceability**: every curated unit row carries (at minimum) the peak channel and the brain region(s) for that channel, persisted at sort time and resolvable via a single query on `CurationV3` or `SpikeSortingOutput`. v1's `CurationV1.get_sort_group_info` samples *one* electrode per sort group; that's wrong for multi-region polymer probes. v3 walks every electrode in the sort group, persists per-unit peak-channel and region, and exposes `Sorting.get_unit_brain_regions(key) -> pd.DataFrame` and the same accessor on `CurationV3`. Brain region tracing must work end-to-end through `SpikeSortingOutput.get_sort_group_info` and through `TrackedUnit` (per-session brain regions for matched units).
- **Reduced UX friction**: `run_v3_pipeline(nwb_file, sort_group_id, interval, preset)` runs the standard sort-and-curate flow with one user call, returning the final `merge_id` plus a manifest. Pydantic models validate parameter dicts at insert time. Notebook walkthrough drops from 35 cells to ~8.
- **Drop-in for existing downstream consumers**: New `SpikeSortingOutput.CurationV3` part. Decoding, ripple, MUA, `SortedSpikesGroup` work unchanged.
- **Explicit v1 parity boundaries**: [feature-parity.md](feature-parity.md) is the contract for which v1 workflows are preserved, which are replaced by successor tables, and which departures are intentional.
- **Team-based sorting preserved**. v3 retains v1's `LabTeam` FK structure: `RecordingSelection`, `SessionGroup.Member`, and `run_v3_pipeline(team_name=...)` all carry team ownership. `cautious_delete` continues to check team membership; per-team data partitioning, shared-team sorting, and team-based access control all work the same way they did in v1. Two teams running an identical sort get distinct `recording_id` UUIDs and distinct binary cache paths — preventing the v0 filesystem-overwrite bug ([#133](https://github.com/LorenFrankLab/spyglass/issues/133)).
- **Zero-migration policy** (binding): all v3 tables must be designed in their final shape in the phase that introduces them. **No `alter()` calls** are permitted across phases. New tables added in later phases are fine; modifying any existing v3 table's primary key, foreign keys, or column types is not. This includes anticipated needs: Phase 1's `SortingSelection` already supports both single and concatenated input paths; Phase 1's `Sorting.Unit` already supports per-unit brain region; Phase 1's `CurationV3` already supports the columns that subsequent phases consume. The plan documents which schema decisions are forward-compatibility decisions vs. clean-slate decisions.
- **v0/v1 coexistence**: Both legacy pipelines remain functional throughout the v3 lifetime. **No v1 sunset is planned in this work**.

### Non-Goals

- **Re-sorting existing v1 data.** v3 adds the new pipeline; users may run it on new sessions but the plan never invalidates or migrates existing v1 sorts.
- **A new merge table.** v3 plugs into the existing `SpikeSortingOutput`; adding `MergeV3Output` would force downstream code to choose between merges. Not in scope.
- **A v3-specific imported-sorting table.** Imported external Units remain in the existing `ImportedSpikeSorting` table and merge part. v3 compares against or coexists with imported Units; it does not absorb them into `CurationV3` lineage.
- **Removing v0 or v1 source.** Both stay in `src/spyglass/spikesorting/{v0,v1}/`. Documentation will mark them legacy but they remain populate-runnable. **No v1 sunset is planned here.**
- **DeepUnitMatch.** Pluggable via the Phase 4 matcher protocol but not implemented; future work.
- **Custom UI / web app.** All curation UI uses upstream FigPack; we do not build a Spyglass-specific viewer.
- **Cross-region / cross-probe matching.** UnitMatch assumes a single probe across sessions in a session group. Multi-probe handling is out of scope.

### Dependency policy

| Dependency | From | To | When | Reason |
| --- | --- | --- | --- | --- |
| `pydantic` | (not pinned) | `>=2.0` | Phase 0 | Parameter schema validation. |
| `zarr` | (transitive) | `<3.0` | Phase 0 | SI #4014 not yet landed; v3 archival uses Zarr v2. |
| `spikeinterface` | `>=0.99.1,<0.100` | `>=0.104,<0.105` | **Prerequisite PR before Phase 1** | SortingAnalyzer, PreprocessingPipeline, modern curation primitives. Bumping this breaks v1's `extract_waveforms` calls — the same prerequisite PR ports v1 to `create_sorting_analyzer` so the bump is non-breaking. |
| `mountainsort5` | (absent) | `>=0.5` | Same prerequisite PR | Additional sorter; v3 keeps MS4 too. |
| `unitmatchpy` | (absent) | `>=3.3` | Phase 4 (optional extra `spikesorting-v3-matching`) | Cross-session matching. |
| `MEArec` | (absent) | `>=1.9` | Phase 0 (optional extra `spikesorting-v3-validation`) | Ground-truth fixture generation for v3 validation. |
| `neuroconv[mearec]` | (absent) | (latest) | Phase 0 (same extra) | MEArec → NWB conversion via `MEArecRecordingInterface`. |
| `mountainsort4` | present | unchanged | n/a | v1 + v3 both ship MS4 wrapper. |

All v3 additions are made via the existing `pyproject.toml`. Optional deps (`unitmatchpy`, `mountainsort5` when not installed) gate at import time with a clear error pointing to the install command.

## Metrics

How we'll know v3 is working:

- **Ground-truth tests (Phase 1)**: Run v3 against MEArec-generated polymer-probe and Neuropixels fixtures with known planted units; compute precision/recall via `spikeinterface.comparison.compare_sorter_to_ground_truth`. Polymer is the primary lab-relevant fixture (per Chung et al. 2019, the Frank-lab standard). minirec serves only as a plumbing regression guard (test_v3_minirec_plumbing) — it is too short to contain real spikes. v1 parity moves to the env-var real-data path (`SPIKESORTING_V3_REAL_NWB_PATH`); no minirec parity test ships.
- **Memory and runtime (Phase 3)**: Single-channel-group, 30 kHz, 1 h recording must complete `Sorting.populate()` in <10 min and peak RSS <8 GB on a 16-core machine with default job kwargs. Smoke-test on 5 min slice first.
- **Notebook cells (Phase 5)**: The end-to-end notebook for a single sort drops from 35 code cells to ≤10. Counted manually before merge.
- **Downstream non-regression (Phase 1)**: A `SortedSpikesGroup` built from a v3 `CurationV3` row produces the same `get_spike_times` shape and units as one built from an equivalent v1 row.
- **UnitMatch polymer validation gate (Phase 4)**: On the MEArec-generated 2-session polymer fixture (Phase 4b task; planted shared templates across the 4-shank polymer probe), `test_v3_unitmatch_polymer_mearec_ground_truth` asserts AUC > 0.85 of match probability against ground-truth template correspondence. Polymer is the gate because it is the Frank-lab standard probe; Neuropixels and tetrode validations run alongside but are informational, not gating.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| MountainSort output stochasticity makes parity testing noisy. | Phase 0 captures v1 baseline with `clusterless_thresholder` (deterministic, seeded) as the gold-standard parity test. MS4-vs-MS5 comparisons are bounded by qualitative metrics (unit count order, firing-rate distribution shape) and explicitly noted. |
| SortingAnalyzer folders are large for chronic recordings (50+ GB possible). | Phase 1 stores analyzer folder paths in DataJoint, **not** content. Phase 2 ports the v1 `RecordingRecompute` pattern into v3-specific `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*` tables for verified storage reclamation. Sparse=True default cuts size 5-10×. |
| UnitMatch's published validation is Neuropixels-heavy and may not transfer to Frank-lab polymer/tetrode data. | Phase 4 gates shipping on a MEArec polymer-probe fixture, because polymer is the lab-relevant standard for this workflow. Neuropixels and tetrode AUCs are recorded as informational checks; tetrode users are not promised production matching unless the informational result supports it. |
| Adding a 4th part to `SpikeSortingOutput` may surface latent merge bugs. | Phase 1 includes integration tests against `SortedSpikesGroup`, decoding, ripple, MUA with a v3-sourced merge_id. |
| `set_group_by_shank()` from v1 silently overwrites sort groups — v3 must not perpetuate this. | `SortGroupV3.set_group_by_shank()` raises if rows already exist for the session; destructive replacement requires `delete_existing_entries=True, confirm=True` after reviewing a `DeletionPreview`. Phase 1 task. |
| Pydantic schema migration risk: existing parameter blobs may not validate. | All Pydantic models are introduced fresh in v3 Lookup tables; v1 Lookup blobs remain unvalidated. No migration of v1 params. |
| Concatenation across days (out of scope but tempting). | `SessionGroup.Member` carries a `recording_date` field; `ConcatenatedRecording.populate()` raises if members span multiple dates without an explicit `allow_multi_day=True` parameter row. |
| Pipeline orchestrator `run_v3_pipeline()` may hide reproducibility. | The function returns a manifest dict of every `(table, key)` it inserted. Notebook prints it. Phase 5 task. |

## Rollout Strategy

v3 is **strictly additive** for the lifetime of this plan. No feature flags, no parallel-running shims, no deprecation phase needed inside this plan.

- v1 stays as the documented production path until v3 reaches parity (defined as: Phases 1–5 merged, parity tests green on lab production data for one cycle).
- After parity is achieved, the documentation root (`docs/src/Pipelines/SpikeSorting/`) gets a banner pointing users to v3 for new work; v1 docs remain.
- **v1 source code removal is explicitly out of scope** for this plan. Phase 5 documents how users choose v1 vs v3; it does not define a deletion or sunset trigger.
- Each phase ships as a stand-alone PR; users upgrading midway can use whatever phases are merged at the time. Phase 1 alone is usable in production.

## Resolved Design Decisions

These were open questions; user has decided. Documented here so subsequent phase implementers don't re-litigate.

1. **MountainSort 4 stays in v3.** Add `mountainsort4` to `SorterParameters` defaults alongside `mountainsort5`, `kilosort4`, `spykingcircus2`, `tridesclous2`, `clusterless_thresholder`. MS4 wrapper still exists in SpikeInterface 0.104; the v1 sorter_params dicts can be ported with minimal changes (notably removing the `tempdir` field-mutation hack). Pydantic schema for MS4 ships in Phase 1's `_params/sorter.py`.
2. **FigPack is the v3 curation UI.** Implemented in Phase 5 as the primary curation interface for v3 sorts. FigURL is not extended to v3; FigURL stays available for v1 data only. **Hard dependency on FigPack feasibility** — Phase 5 begins with a FigPack feasibility check (see Phase 5 first tasks); if FigPack is genuinely unusable at implementation time, the plan stops there and surfaces the blocker to the project owner rather than silently shipping a degraded fallback.
3. **DeepUnitMatch deferred.** Phase 4's `MatcherProtocol` slot accepts it as a future plugin; not implemented in this plan.
4. **Multi-day concatenation is supported by the schema, but is opt-in and not the recommended cross-day path.** `SessionGroup.create_group(..., allow_multi_day=True)` is required to insert members spanning ≥2 dates; the default rejects multi-day inputs with a clear pointer to Phase 4 sort-then-match. `ConcatenatedRecording` does NOT auto-dispatch DREDge for multi-day groups — the caller picks an explicit motion-correction preset. **Sort-then-match (Phase 4 UnitMatch) is the validated path for cross-day analyses on Neuropixels; multi-day tetrode sorting is not a current Frank-lab use case**, so neither multi-day tetrode concat nor tetrode UnitMatch is treated as a gating workflow. Both remain schema-supported (zero migration), but the validation focus and the recommended-path docs are Neuropixels-centric.
5. **Zero schema migration.** All v3 tables are designed in their final shape in Phase 1 (or in the phase that adds them). No `alter()` calls across phases. Concretely: Phase 1's `SortingSelection` declares two nullable typed FKs (`recording_id` → `Recording`, `concat_recording_id` → `ConcatenatedRecording`) with XOR enforced in `insert_selection()`; Phase 1's `Sorting.Unit`, `CurationV3.Unit`, and `CurationV3.UnitLabel` part tables are present from day one; Phase 1's `CurationV3` columns are final. The "two paths for `SortingSelection`" formerly in Open Question #8 is collapsed to one design.

## Remaining Open Questions

These are smaller, mostly orthogonal decisions where the plan's default is documented but the user can override.

1. **Pydantic model versioning?** *Current best-answer*: each Lookup table includes a `params_schema_version: int` secondary attribute. When a Pydantic model changes shape, increment the schema version; the validator dispatches by version. Documented in [shared-contracts.md](shared-contracts.md).
2. **Sorter container support (Docker / Singularity) in v3?** *Current best-answer*: defer. v3 ships without container wrappers; users who need MATLAB sorters (Kilosort 2/2.5/3, IronClust) can use the SI `docker_image=True` kwarg by inserting a custom `SorterParameters` row. Re-evaluate after Phase 5.
3. **Curation label enum enforcement?** *Current best-answer*: v3 uses a `CurationLabel` Pydantic Enum (`accept`, `mua`, `noise`, `artifact`, `reject`). Inserts validate against the enum at the API surface but raw `dj.Manual` inserts that bypass the helper remain free-form (DataJoint doesn't support enums on blob columns).
4. **Parity baseline SI version mismatch.** *Current best-answer*: v1 parity is no longer performed against minirec (minirec has no real spikes). v1 parity now runs only when `SPIKESORTING_V3_REAL_NWB_PATH` is set — `test_v3_real_data_v1_parity` runs v1 + v3 against the user-provided NWB and compares per-unit spike times. Tolerances: `clusterless_thresholder` exact within ±1 sample; stochastic sorters within ±50% unit count and ±30% median firing rate. If v1 cannot run under SI 0.104 (because the prerequisite port to `create_sorting_analyzer` is not yet complete), the test skips with an explicit message — it does not silently relax to a meaningless tolerance.

## Issue Traceability — v1 GitHub issues v3 closes

Mining of [LorenFrankLab/spyglass GitHub issues](https://github.com/LorenFrankLab/spyglass/issues?q=is%3Aissue+label%3A%22spike+sorting%22) surfaced concrete v1 bugs and friction points. The following are explicitly addressed by v3 design decisions:

| v1 Issue | Problem | v3 fix |
|---|---|---|
| [#1394](https://github.com/LorenFrankLab/spyglass/issues/1394) | `get_sort_group_info` shows only one electrode → wrong region on multi-region polymer probes | `Sorting.Unit` + `CurationV3.Unit` persist per-unit peak channel; `get_sort_group_info` walks all electrodes |
| [#1532](https://github.com/LorenFrankLab/spyglass/issues/1532), [#1154](https://github.com/LorenFrankLab/spyglass/issues/1154) | `CurationV1` fails on zero-unit sortings (`KeyError: 'spike_times'`) | Empty/NaN/Boundary Invariant 1 in shared-contracts; covered across Phase 1 `test_v3_empty_sorting_phase1`, Phase 2 `test_analyzer_curation_zero_unit_sorting`, and Phase 5 `test_figpack_zero_unit_sorting` |
| [#1556](https://github.com/LorenFrankLab/spyglass/issues/1556) | `FigURLCuration` fails on NaN-bearing metrics (low-spike units) | Empty/NaN/Boundary Invariant 2; `_sanitize_for_json` in `AnalyzerCuration.make()`; Phase 2 `test_metric_nan_round_trip` |
| [#1558](https://github.com/LorenFrankLab/spyglass/issues/1558) | Spike at recording boundary → "exceeds duration" error | Empty/NaN/Boundary Invariant 3; explicit boundary-tolerance test in Phase 1 |
| [#1437](https://github.com/LorenFrankLab/spyglass/issues/1437) | `SpikeSorting.populate` fails when raw NWB has existing Units table | Phase 1 `Sorting.make()` uses whitelist construction (does NOT copy raw NWB) |
| [#1133](https://github.com/LorenFrankLab/spyglass/issues/1133), [#1585](https://github.com/LorenFrankLab/spyglass/issues/1585) | Recording timestamps wrong / silent truncation | Phase 1 `Recording.make()` asserts saved time range covers requested interval; `RecordingTruncatedError` |
| [#928](https://github.com/LorenFrankLab/spyglass/issues/928) | Artifact detection per sort group misses behavioral artifacts visible across groups | Phase 1 `SharedArtifactGroup` opt-in path |
| [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513) | v0 `AutomaticCuration.get_labels` label-list aliasing across units | Phase 2 label-list isolation invariant + `test_label_list_isolation` |
| [#939](https://github.com/LorenFrankLab/spyglass/issues/939) | `CurationV1` doesn't track metric source | Phase 1 `CurationV3.metrics_source` enum column |
| [#528](https://github.com/LorenFrankLab/spyglass/issues/528) | No cross-day sorting | Phase 3 `SessionGroup` + `ConcatenatedRecording`; Phase 4 UnitMatch sort-then-match for cross-day |
| [#1436](https://github.com/LorenFrankLab/spyglass/issues/1436) | Need SI 0.103+ compatibility | Phase 0 prerequisite SI 0.104 upgrade; Phase 1 SortingAnalyzer |
| [#1286](https://github.com/LorenFrankLab/spyglass/issues/1286) | UX redesign request | Phase 1 minimal `run_v3_pipeline()` + Phase 5 full presets/FigPack |
| [#1530](https://github.com/LorenFrankLab/spyglass/issues/1530), [#1512](https://github.com/LorenFrankLab/spyglass/issues/1512), [#1504](https://github.com/LorenFrankLab/spyglass/issues/1504), [#1215](https://github.com/LorenFrankLab/spyglass/issues/1215) | `FigURLCuration` brittleness (opaque KeyError, sortingview API drift) | Phase 5 replaces FigURL with FigPack for v3 curations |
| [#133](https://github.com/LorenFrankLab/spyglass/issues/133) | When two teams run an otherwise-identical sort, v0's `SpikeSortingRecording.make()` `rmtree`s the first team's recording folder before writing its own — silent filesystem destruction of another team's data ([v0/spikesorting_recording.py:319-321](src/spyglass/spikesorting/v0/spikesorting_recording.py#L319-L321)) | v3's binary cache path is keyed on `recording_id` (UUID), and `RecordingSelection.insert_selection` includes `team_name` in its existence check — so two teams running the same sort get distinct `recording_id`s and distinct cache paths. No `rmtree`-of-another-team's-files is possible. (v1 already fixed the worst of this; v3 inherits the fix and makes the UUID-keyed path explicit.) |
| [PR #1438](https://github.com/LorenFrankLab/spyglass/pull/1438) (DRAFT) | v1 `SortGroup` only supports `set_group_by_shank`; Berke Lab and other labs need to group by arbitrary electrode-table columns (e.g., `intan_channel_number`) | v3 `SortGroupV3.set_group_by_electrode_table_column(column, groups, ...)` ports the PR's design directly. Both shank and column-based constructors share PR #1438's existing-entry collision-handling pattern (delete-and-replace OR non-overlapping `sort_group_ids`), replacing both v1's silent-overwrite footgun and the earlier-draft `force=True` design. |

## Explicitly NOT fixed by v3 (separate v1-side patches required)

These bugs surfaced in the issue sweep but are NOT in v3's surface area. The plan should not absorb them; the implementer notes them here so an executor isn't tempted to scope-creep into them:

- [#1581](https://github.com/LorenFrankLab/spyglass/issues/1581) — `ImportedSpikeSorting.fetch_nwb` `pd.concat` index collision. Fix in `imported.py` (not in `spikesorting/v3/`): use `ignore_index=True` and add `nwb_file_name` + `unit_id` as MultiIndex levels.
- [#1273](https://github.com/LorenFrankLab/spyglass/issues/1273) — `UnitWaveformFeatures` assumes contiguous unit IDs. Fix in `decoding/v1/waveform_features.py`; one-line indexing change. v3's `Sorting.Unit` design preserves non-contiguous unit IDs, but the v1 `UnitWaveformFeatures` table is not rewritten.
- [#638](https://github.com/LorenFrankLab/spyglass/issues/638), [#1282](https://github.com/LorenFrankLab/spyglass/issues/1282) — Opaque `1217 IntegrityError` on cascade-delete. Database-config / upstream DataJoint issue affecting all merge-bearing tables. v3 cannot fix from the schema side.
- [#1159](https://github.com/LorenFrankLab/spyglass/issues/1159) — `get_spiking_v1_merge_ids` invalid restriction. Bug in `spikesorting_merge.py`'s v1 helper; v3's `get_restricted_merge_ids(sources=['v3'])` path is independent.

The implementer of any v3 phase should resist scope-creeping into these. They are tracked separately.

## Estimated Effort

LOC sanity check, not a time estimate:

- Phase 0: ~300 LOC (mostly scaffolding files + one baseline-capture script).
- Phase 1: ~1500 LOC across `recording.py`, `sorting.py`, `curation.py` + part-table addition to `spikesorting_merge.py` + tests.
- Phase 2: ~900 LOC for `metric_curation.py` + the curation rules engine + tests.
- Phase 3: ~700 LOC for `session_group.py` (including `ConcatenatedRecording` make()) + tests.
- Phase 4: ~1100 LOC for matcher plugin scaffold + UnitMatch implementation + `TrackedUnit` + tests.
- Phase 5: ~800 LOC for `pipeline.py` orchestrator + Pydantic models + FigPack curation + notebook rewrite.

Total v3 footprint: roughly 5300 LOC across 6 PRs. Compared to v1 at ~6000 LOC in `src/spyglass/spikesorting/v1/` (counted via `wc -l`), v3 is similar in size despite richer capability because the per-stage scaffolding is more consolidated.
