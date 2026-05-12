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

- [src/spyglass/spikesorting/spikesorting_merge.py:34-150](src/spyglass/spikesorting/spikesorting_merge.py#L34-L150) — `SpikeSortingOutput`: add new part `class CurationV3(SpyglassMixinPart)` FK'ing `spyglass.spikesorting.v3.curation.CurationV3`. Update `_get_restricted_merge_ids_v1` analog `_get_restricted_merge_ids_v3` and route through `get_restricted_merge_ids(sources=[...])` so `'v3'` is a valid source string.
- [pyproject.toml:62](pyproject.toml#L62) — `spikeinterface>=0.99.1,<0.100` → `spikeinterface>=0.104,<0.105`. Add `pydantic>=2.0`, `mountainsort5`, `unitmatchpy>=3.3`. Pin `zarr<3.0` (until SI #4014 lands).

**Existing files PRESERVED unchanged**:

- [src/spyglass/spikesorting/v0/](src/spyglass/spikesorting/v0/) — entire directory left alone. v0 sorts and `CuratedSpikeSorting` part of `SpikeSortingOutput` remain queryable.
- [src/spyglass/spikesorting/v1/](src/spyglass/spikesorting/v1/) — entire directory left alone. v1 remains the production path until v3 reaches parity (Phase 5 documents sunset trigger).
- [src/spyglass/spikesorting/analysis/v1/group.py](src/spyglass/spikesorting/analysis/v1/group.py) — `SortedSpikesGroup` keys off `SpikeSortingOutput.merge_id`; adding `CurationV3` as a new part adds rows that this table will already see via the merge.
- [src/spyglass/spikesorting/imported.py](src/spyglass/spikesorting/imported.py) — untouched.
- [src/spyglass/decoding/](src/spyglass/decoding/), [src/spyglass/ripple/](src/spyglass/ripple/), [src/spyglass/mua/](src/spyglass/mua/) — all consume `SpikeSortingOutput` or `SortedSpikesGroup`, unaffected by v3 addition.

## Scope and dependency policy

### Goals

- **Modern SpikeInterface API**: All v3 paths use `SortingAnalyzer` (SI ≥0.101), `PreprocessingPipeline` (SI ≥0.103), and current `spikeinterface.curation` primitives.
- **Cross-session unit tracking**: Pluggable matcher backend with UnitMatchPy as the first plugin. Schema (`TrackedUnit`) treats a "biological unit observed across N sessions" as a first-class queryable entity. DeepUnitMatch swappable via the same plugin slot (future).
- **Chronic / large recording support, including multi-day**: Session groups, virtual concatenation (same-day AND multi-day with DREDge pre-correction), lazy streaming, sparse SortingAnalyzers, motion correction. 30 kHz × 64 ch × 24 h recordings produce one materialized binary cache per recording, not per analysis stage.
- **First-class unit → brain region traceability**: every curated unit row carries (at minimum) the peak channel and the brain region(s) for that channel, persisted at sort time and resolvable via a single query on `CurationV3` or `SpikeSortingOutput`. v1's `CurationV1.get_sort_group_info` samples *one* electrode per sort group; that's wrong for multi-region polymer probes. v3 walks every electrode in the sort group, persists per-unit peak-channel and region, and exposes `Sorting.get_unit_brain_regions(key) -> pd.DataFrame` and the same accessor on `CurationV3`. Brain region tracing must work end-to-end through `SpikeSortingOutput.get_sort_group_info` and through `TrackedUnit` (per-session brain regions for matched units).
- **Reduced UX friction**: `run_v3_pipeline(nwb_file, sort_group_id, interval, preset)` runs the standard sort-and-curate flow with one user call, returning the final `merge_id` plus a manifest. Pydantic models validate parameter dicts at insert time. Notebook walkthrough drops from 35 cells to ~8.
- **Drop-in for existing downstream consumers**: New `SpikeSortingOutput.CurationV3` part. Decoding, ripple, MUA, `SortedSpikesGroup` work unchanged.
- **Zero-migration policy** (binding): all v3 tables must be designed in their final shape in the phase that introduces them. **No `alter()` calls** are permitted across phases. New tables added in later phases are fine; modifying any existing v3 table's primary key, foreign keys, or column types is not. This includes anticipated needs: Phase 1's `SortingSelection` already supports both single and concatenated input paths; Phase 1's `Sorting.Unit` already supports per-unit brain region; Phase 1's `CurationV3` already supports the columns that subsequent phases consume. The plan documents which schema decisions are forward-compatibility decisions vs. clean-slate decisions.
- **v0/v1 coexistence**: Both legacy pipelines remain functional throughout the v3 lifetime. **No v1 sunset is planned in this work**.

### Non-Goals

- **Re-sorting existing v1 data.** v3 adds the new pipeline; users may run it on new sessions but the plan never invalidates or migrates existing v1 sorts.
- **A new merge table.** v3 plugs into the existing `SpikeSortingOutput`; adding `MergeV3Output` would force downstream code to choose between merges. Not in scope.
- **Removing v0 or v1 source.** Both stay in `src/spyglass/spikesorting/{v0,v1}/`. Documentation will mark them legacy but they remain populate-runnable. **No v1 sunset is planned here.**
- **DeepUnitMatch.** Pluggable via the Phase 4 matcher protocol but not implemented; future work.
- **Custom UI / web app.** All curation UI uses upstream FigPack; we do not build a Spyglass-specific viewer.
- **Cross-region / cross-probe matching.** UnitMatch assumes a single probe across sessions in a session group. Multi-probe handling is out of scope.

### Dependency policy

| Dependency | From | To | Reason |
| --- | --- | --- | --- |
| `spikeinterface` | `>=0.99.1,<0.100` | `>=0.104,<0.105` | SortingAnalyzer, PreprocessingPipeline, modern curation primitives. |
| `pydantic` | (not pinned) | `>=2.0` | Parameter schema validation. |
| `mountainsort5` | (absent) | `>=0.5` | Replaces MS4 for new sorts. MS4 wrapper stays accessible via v1. |
| `unitmatchpy` | (absent) | `>=3.3` | Phase 4. Optional dependency. |
| `zarr` | (transitive) | `<3.0` | SI #4014 not yet landed; v3 archival uses Zarr v2. |
| `mountainsort4` | present | unchanged | v1 still uses it; remove only when v1 is sunset. |

All v3 additions are made via the existing `pyproject.toml`. Optional deps (`unitmatchpy`, `mountainsort5` when not installed) gate at import time with a clear error pointing to the install command.

## Metrics

How we'll know v3 is working:

- **Parity tests (Phase 1)**: Run v3 on the existing `minirec` fixture with `clusterless_thresholder` (deterministic sorter — fixed seed). Compare spike times unit-by-unit against a v1 baseline pickled in Phase 0. **Tolerance**: exact integer-sample equality for `clusterless_thresholder`; for MountainSort 5, assert same order of magnitude of unit count and median firing rate within 50% of v1 baseline.
- **Memory and runtime (Phase 3)**: Single-channel-group, 30 kHz, 1 h recording must complete `Sorting.populate()` in <10 min and peak RSS <8 GB on a 16-core machine with default job kwargs. Smoke-test on 5 min slice first.
- **Notebook cells (Phase 5)**: The end-to-end notebook for a single sort drops from 35 code cells to ≤10. Counted manually before merge.
- **Downstream non-regression (Phase 1)**: A `SortedSpikesGroup` built from a v3 `CurationV3` row produces the same `get_spike_times` shape and units as one built from an equivalent v1 row.
- **UnitMatch tetrode validation gate (Phase 4)**: On a 2-session tetrode dataset (provided by user during Phase 4), UnitMatchPy match probabilities must clearly separate self-matches (within-day cross-validation positives) from random-pair negatives. If they don't, Phase 4 documents tetrode as unsupported and v3 falls back to concatenation for multi-session tetrode work.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| MountainSort output stochasticity makes parity testing noisy. | Phase 0 captures v1 baseline with `clusterless_thresholder` (deterministic, seeded) as the gold-standard parity test. MS4-vs-MS5 comparisons are bounded by qualitative metrics (unit count order, firing-rate distribution shape) and explicitly noted. |
| SortingAnalyzer folders are large for chronic recordings (50+ GB possible). | Phase 1 stores analyzer folder paths in DataJoint, **not** content. Cleanup follows the `RecordingRecompute` pattern from v1 (deferred to a Phase 6 follow-up if needed). Sparse=True default cuts size 5-10×. |
| UnitMatch validated only on Neuropixels — may fail on Frank-lab tetrodes (4 channels). | Phase 4 starts with a tetrode validation task. If features collapse, Phase 4 ships the matcher plugin scaffold but UnitMatch is marked Neuropixels-only and tetrode users are routed to concatenation. |
| Adding a 4th part to `SpikeSortingOutput` may surface latent merge bugs. | Phase 1 includes integration tests against `SortedSpikesGroup`, decoding, ripple, MUA with a v3-sourced merge_id. |
| `set_group_by_shank()` from v1 silently overwrites sort groups — v3 must not perpetuate this. | `SortGroupV3.set_group_by_shank()` raises if rows already exist for the session; provides `set_group_by_shank(force=True)` as the explicit override. Phase 1 task. |
| Pydantic schema migration risk: existing parameter blobs may not validate. | All Pydantic models are introduced fresh in v3 Lookup tables; v1 Lookup blobs remain unvalidated. No migration of v1 params. |
| Concatenation across days (out of scope but tempting). | `SessionGroup.Member` carries a `recording_date` field; `ConcatenatedRecording.populate()` raises if members span multiple dates without an explicit `allow_multi_day=True` parameter row. |
| Pipeline orchestrator `run_v3_pipeline()` may hide reproducibility. | The function returns a manifest dict of every `(table, key)` it inserted. Notebook prints it. Phase 5 task. |

## Rollout Strategy

v3 is **strictly additive** for the lifetime of this plan. No feature flags, no parallel-running shims, no deprecation phase needed inside this plan.

- v1 stays as the documented production path until v3 reaches parity (defined as: Phases 1–5 merged, parity tests green on lab production data for one cycle).
- After parity is achieved, the documentation root (`docs/src/Pipelines/SpikeSorting/`) gets a banner pointing users to v3 for new work; v1 docs remain.
- **v1 source code removal is explicitly out of scope** for this plan. Phase 5 documents the sunset trigger (e.g., "after 6 months without active v1 populate calls in production") but the deletion is a future planning effort, not part of this work.
- Each phase ships as a stand-alone PR; users upgrading midway can use whatever phases are merged at the time. Phase 1 alone is usable in production.

## Resolved Design Decisions

These were open questions; user has decided. Documented here so subsequent phase implementers don't re-litigate.

1. **MountainSort 4 stays in v3.** Add `mountainsort4` to `SorterParameters` defaults alongside `mountainsort5`, `kilosort4`, `spykingcircus2`, `tridesclous2`, `clusterless_thresholder`. MS4 wrapper still exists in SpikeInterface 0.104; the v1 sorter_params dicts can be ported with minimal changes (notably removing the `tempdir` field-mutation hack). Pydantic schema for MS4 ships in Phase 1's `_params/sorter.py`.
2. **FigPack is the v3 curation UI.** Implemented in Phase 5 as the primary curation interface for v3 sorts. FigURL is not extended to v3; FigURL stays available for v1 data only. **Hard dependency on FigPack feasibility** — Phase 5 begins with a FigPack feasibility check (see Phase 5 first tasks); if FigPack is genuinely unusable at implementation time, the plan stops there and surfaces the blocker to the project owner rather than silently shipping a degraded fallback.
3. **DeepUnitMatch deferred.** Phase 4's `MatcherProtocol` slot accepts it as a future plugin; not implemented in this plan.
4. **Multi-day concatenation is supported by the schema, but is opt-in and not the recommended cross-day path.** `SessionGroup.create_group(..., allow_multi_day=True)` is required to insert members spanning ≥2 dates; the default rejects multi-day inputs with a clear pointer to Phase 4 sort-then-match. `ConcatenatedRecording` does NOT auto-dispatch DREDge for multi-day groups — the caller picks an explicit motion-correction preset. **Sort-then-match (Phase 4 UnitMatch) is the validated path for cross-day analyses**; multi-day concat is documented as experimental and behind a Phase 3 validation gate. Per review feedback: concat across days is empirically unreliable for large drifts, so it remains a supported-but-not-default path. The schema cost of supporting it is the same (no migration), so it stays in scope.
5. **Zero schema migration.** All v3 tables are designed in their final shape in Phase 1 (or in the phase that adds them). No `alter()` calls across phases. Concretely: Phase 1's `SortingSelection` has `recording_source` from day one; Phase 1's `Sorting.Unit` part table for brain-region tracing is present from day one; Phase 1's `CurationV3` columns are final. The "two paths for `SortingSelection`" formerly in Open Question #8 is collapsed to one design.

## Remaining Open Questions

These are smaller, mostly orthogonal decisions where the plan's default is documented but the user can override.

1. **Pydantic model versioning?** *Current best-answer*: each Lookup table includes a `params_schema_version: int` secondary attribute. When a Pydantic model changes shape, increment the schema version; the validator dispatches by version. Documented in [shared-contracts.md](shared-contracts.md).
2. **Sorter container support (Docker / Singularity) in v3?** *Current best-answer*: defer. v3 ships without container wrappers; users who need MATLAB sorters (Kilosort 2/2.5/3, IronClust) can use the SI `docker_image=True` kwarg by inserting a custom `SorterParameters` row. Re-evaluate after Phase 5.
3. **Curation label enum enforcement?** *Current best-answer*: v3 uses a `CurationLabel` Pydantic Enum (`accept`, `mua`, `noise`, `artifact`, `reject`). Inserts validate against the enum at the API surface but raw `dj.Manual` inserts that bypass the helper remain free-form (DataJoint doesn't support enums on blob columns).
4. **Parity baseline SI version mismatch.** *Current best-answer*: The Phase 0 baseline is captured against `clusterless_thresholder` running under SI 0.99 (the version v1 requires). Phase 1's parity test cannot expect bitwise-identical output against SI 0.104 because preprocessing internals changed between versions. Plan correction: Phase 0 baseline is captured at **two SI versions** — under 0.99 for record (the "v1 reference baseline") and again under 0.104 by running v1's preprocessing logic against the new SI APIs (a one-off compatibility check, documented in Phase 0). Phase 1's `test_v3_clusterless_parity` asserts spike-time equality against the **0.104-baseline** with tolerance "±1 sample per spike" (floating-point boundary effects acceptable; gross algorithmic divergence not). If the 0.104 baseline can't be captured because v1 can't run under 0.104, the test relaxes to ±5 samples per spike and the test name documents this.

## Estimated Effort

LOC sanity check, not a time estimate:

- Phase 0: ~300 LOC (mostly scaffolding files + one baseline-capture script).
- Phase 1: ~1500 LOC across `recording.py`, `sorting.py`, `curation.py` + part-table addition to `spikesorting_merge.py` + tests.
- Phase 2: ~900 LOC for `metric_curation.py` + the curation rules engine + tests.
- Phase 3: ~700 LOC for `session_group.py` (including `ConcatenatedRecording` make()) + tests.
- Phase 4: ~1100 LOC for matcher plugin scaffold + UnitMatch implementation + `TrackedUnit` + tests.
- Phase 5: ~800 LOC for `pipeline.py` orchestrator + Pydantic models + FigPack curation + notebook rewrite.

Total v3 footprint: roughly 5300 LOC across 6 PRs. Compared to v1 at ~6000 LOC in `src/spyglass/spikesorting/v1/` (counted via `wc -l`), v3 is similar in size despite richer capability because the per-stage scaffolding is more consolidated.
