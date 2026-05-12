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
- **Cross-session unit tracking**: Pluggable matcher backend with UnitMatchPy as the first plugin. Schema (`TrackedUnit`) treats a "biological unit observed across N sessions" as a first-class queryable entity. DeepUnitMatch swappable via the same plugin slot.
- **Chronic / large recording support**: Session groups, virtual concatenation, lazy streaming, sparse SortingAnalyzers, motion correction (DREDge). 30 kHz × 64 ch × 24 h recordings produce one materialized binary cache per recording, not per analysis stage.
- **Reduced UX friction**: `run_v3_pipeline(nwb_file, sort_group_id, interval, preset)` runs the standard sort-and-curate flow with one user call, returning the final `merge_id`. Pydantic models validate parameter dicts at insert time, not populate time. Notebook walkthrough drops from 35 cells to ~8.
- **Drop-in for existing downstream consumers**: New `SpikeSortingOutput.CurationV3` part. Decoding, ripple, MUA, `SortedSpikesGroup` work unchanged.
- **v0/v1 coexistence**: Both legacy pipelines remain functional throughout the v3 lifetime. Phase 5 sets a sunset trigger for v1 but does not remove it.

### Non-Goals

- **Re-sorting existing v1 data.** v3 adds the new pipeline; users may run it on new sessions but the plan never invalidates or migrates existing v1 sorts.
- **A new merge table.** v3 plugs into the existing `SpikeSortingOutput`; adding `MergeV3Output` would force downstream code to choose between merges. Not in scope.
- **Removing v0 or v1 source.** Both stay in `src/spyglass/spikesorting/{v0,v1}/`. Documentation will mark them legacy but they remain populate-runnable.
- **Multi-day concatenation in MVP.** Phase 3 supports single-day session groups only (sessions on the same recording mount with breaks). Multi-day with DREDge pre-correction is documented as future work.
- **Replacing FigURL immediately.** Phase 5 adds FigPack as a second curation path; FigURL remains usable for v3 curations during a one-release deprecation window.
- **Custom UI / web app.** All curation UI uses upstream FigPack / FigURL; we do not build a Spyglass-specific viewer.
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

## Open Questions

1. **MountainSort 4 wrapper in v3 — keep or drop?** *Current best-answer*: drop. MS4 stays available through v1 for legacy reproducibility; v3 ships with MS5 + KS4 + SpykingCircus2 + Tridesclous2 + `clusterless_thresholder`. Confirm with user before Phase 1 merges.
2. **FigPack vs FigURL default in v3?** *Current best-answer*: FigURL primary in Phase 1; FigPack added in Phase 5 as alternate path with optional config flag. Switch default to FigPack only after one release of side-by-side usage.
3. **DeepUnitMatch in MVP?** *Current best-answer*: no. The `MatcherProtocol` plugin slot in Phase 4 is designed to accept it later; Phase 4 ships only the UnitMatch plugin. DeepUnitMatch is a Phase 4.1 enhancement.
4. **Concatenation across days?** *Current best-answer*: not in MVP. `SessionGroup` carries a `recording_date` field and `ConcatenatedRecording` refuses cross-date members by default. Multi-day with DREDge pre-correction is future work.
5. **Pydantic model versioning?** *Current best-answer*: each Lookup table includes a `params_schema_version: int` secondary attribute. When a Pydantic model changes shape, increment the schema version; the validator dispatches by version. Documented in [shared-contracts.md](shared-contracts.md).
6. **Sorter container support (Docker / Singularity) in v3?** *Current best-answer*: defer. v3 ships without container wrappers; users who need MATLAB sorters can use the SI `docker_image=True` kwarg by inserting a custom `SorterParameters` row. Re-evaluate after Phase 5.
7. **Curation label enum enforcement?** *Current best-answer*: v3 uses a `CurationLabel` Pydantic Enum (`accept`, `mua`, `noise`, `artifact`, `reject`). Inserts validate against the enum at the API surface but raw `dj.Manual` inserts that bypass the helper remain free-form (DataJoint doesn't support enums on blob columns).

8. **`SortingSelection` schema migration in Phase 3 — when is the v1 schema considered "shipped externally"?** *Current best-answer*: Phase 1's `SortingSelection` is treated as internal-only until Phase 3 lands. Phase 1's CHANGELOG entry must include the phrase "schema may change before Phase 3 — do not depend on `SortingSelection` columns from external code". If a lab adopts v3 between Phase 1 and Phase 3 and inserts production rows into `SortingSelection`, Phase 3 ships a `dj_run_migration.py` helper that backfills the new `recording_source='single'` column on existing rows and converts `artifact_id` from non-nullable PK-component to nullable secondary. The migration script must: (a) be idempotent (re-runnable), (b) verify row counts before/after, (c) be runnable in a dry-run mode that prints SQL without executing. If migration risk is unacceptable, the fallback is to introduce a parallel `SortingSelectionV2` table in Phase 3 and deprecate the original, but the data still has to be migrated; the parallel-table approach just adds a deprecation window. Confirm with user before Phase 3 starts.

9. **Parity baseline SI version mismatch.** *Current best-answer*: The Phase 0 baseline is captured against `clusterless_thresholder` running under SI 0.99 (the version v1 requires). Phase 1's parity test cannot expect bitwise-identical output against SI 0.104 because preprocessing internals changed between versions. Plan correction: Phase 0 baseline is captured at **two SI versions** — under 0.99 for record (the "v1 reference baseline") and again under 0.104 by running v1's preprocessing logic against the new SI APIs (a one-off compatibility check, documented in Phase 0). Phase 1's `test_v3_clusterless_parity` asserts spike-time equality against the **0.104-baseline** with tolerance "±1 sample per spike" (floating-point boundary effects acceptable; gross algorithmic divergence not). If the 0.104 baseline can't be captured because v1 can't run under 0.104, the test relaxes to ±5 samples per spike and the test name documents this.

## Estimated Effort

LOC sanity check, not a time estimate:

- Phase 0: ~300 LOC (mostly scaffolding files + one baseline-capture script).
- Phase 1: ~1500 LOC across `recording.py`, `sorting.py`, `curation.py` + part-table addition to `spikesorting_merge.py` + tests.
- Phase 2: ~900 LOC for `metric_curation.py` + the curation rules engine + tests.
- Phase 3: ~700 LOC for `session_group.py` (including `ConcatenatedRecording` make()) + tests.
- Phase 4: ~1100 LOC for matcher plugin scaffold + UnitMatch implementation + `TrackedUnit` + tests.
- Phase 5: ~800 LOC for `pipeline.py` orchestrator + Pydantic models + FigPack curation + notebook rewrite.

Total v3 footprint: roughly 5300 LOC across 6 PRs. Compared to v1 at ~6000 LOC in `src/spyglass/spikesorting/v1/` (counted via `wc -l`), v3 is similar in size despite richer capability because the per-stage scaffolding is more consolidated.
