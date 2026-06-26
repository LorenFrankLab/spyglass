# Overview — Scope, sequencing, integration, risks

[← back to PLAN.md](PLAN.md)

Every phase remediates verified findings from
[../../reviews/spikesorting-v2/TRIAGE.md](../../reviews/spikesorting-v2/TRIAGE.md).
Root-issue IDs (R#) and finding IDs (e.g. TIME-1, REL-1) reference that document;
the owner-agreed disposition for each is in its "Decisions" section.

## Sequencing and independence

- **phase-1 (correctness) ships first** — these are silent-wrong-science bugs and
  do not depend on any other phase.
- **phase-1b (curation evaluation) ships immediately after phase-1** — phase-1's
  guard prevents raw-sort metrics from being attached to merged unit ids; phase-1b
  adds the intended replacement: metrics and suggestions computed over the exact
  committed `CurationV2` unit set, including merged units.
- **phase-1c (curation composition/acceptance) follows phase-1b** — this is the
  broader editing model: child curations compose from the parent curation state,
  labels inherit predictably, and evaluation outputs can be explicitly accepted
  into committed child curations. It is split out so final metrics can ship first.
- **phase-0 (decomposition) is largely independent of phases 1–4** and may proceed in
  parallel; its only hard constraint is that it **merges before the Phase 5 UX
  overhaul** begins (so Phase 5 extends the decomposed `CurationV2`, not the
  god-module). Whichever of phase-0 / phases 1–4 lands second rebases — they touch
  overlapping files (`curation.py`, `metric_curation.py`), so expect mechanical
  conflict resolution, not logical conflict. **One real dependency:** phase-4a's
  REL-4 task (concat-curation query routing) edits the routing helper phase-0
  extracts, so land REL-4 **after** phase-0.
- **phases 2, 3a, 3b, 4a, 4b** build on the current structure and are mutually
  independent except: phase-3b (NWB provenance) reads the columns/values phase-3a
  **and phase-2** add, so **phase-2 and phase-3a both precede phase-3b** (3b writes
  phase-3a's provenance fields and phase-2's resolved motion preset into the NWB).
- **phase-4c (concat lifecycle/integrity, Round 3)** overlaps phase-4a's concat-compat
  work in `_concat_recording.py` — whichever lands second rebases. Its cheap
  verify-`content_hash`-on-read half (task 2) is not deferrable; its concat-recompute
  half (task 3) may be deferred to first concat-data retention.
- **phase-6 (scientific-validation/CI gates, Round 3)** is best done **last** — it adds
  CI gates that should protect the *corrected* behavior from phases 1–4c, so run it
  after them. Its task 1 (publish + require the gating fixtures) is the shared blocker
  for the rest of the phase.
- The companion **Phase 5 adjustments** (below) are edits to the *existing*
  `../spikesorting-v2/phase-5-ux-overhaul.md`, applied as part of this work so the
  Phase 5 executor inherits the decided docs/orchestrator items (plus the Round-3 doc
  items MIG-4/5/6, the stale `13_`/`dev_walkthrough` notebooks, ALSC-8, CLUST-5, CNEP-7,
  AVTM-5).

## Current codebase integration points

> **Line numbers are point-in-time** (captured at plan-authoring HEAD) and may have
> drifted by a few lines — a code-review pass found fixture refs off by ~1. Treat
> them as "confirm by reading," not literal; the symbol/function names are the
> reliable anchor.

Touched (all paths under `src/spyglass/spikesorting/`):

- `v2/curation.py:1415-1668` — `resolve_restriction`: pure key-classification half (1511-1595) extracted; DataJoint join assembly preserved on the table. **phase-0.**
- `v2/curation.py:1102-1187` — `summarize_curation`: pure formatter core (`build_curation_summary`) extracted; the rest of the accessor cluster (`get_recording`/`get_sort_metadata` are irreducible DB routers, `get_sorting`/`get_merged_sorting` already delegate compute to `_units_nwb`) **stays on the table**. **phase-0.**
- `v2/_recording_restriction.py:274-280` — `_consolidate_regular_intervals` float→frame snapping. **phase-1.**
- `v2/_recording_preprocessing.py:188-232` + `v2/_nwb_metadata_helpers.py:59-66` — offset zeroing after referencing on the no-filter path. **phase-1.**
- `v2/metric_curation.py:255,273,1101-1115` — SNR `peak_sign` resolved from sorter polarity. **phase-1.**
- `v2/metric_curation.py:916,933,1399-1417` — AnalyzerCuration analyzer namespace + materialize. **phase-1.**
- `v2/metric_curation.py` + `v2/curation.py` + `v2/_sorting_analyzer.py` — canonical `CurationEvaluation` over committed curation states and final merged-unit metrics. **phase-1b.**
- `v2/curation.py` + `v2/_curation_transforms.py` + `v2/_curation_plan.py` + `v2/_units_nwb.py` + `v2/metric_curation.py` — parent-namespace child curation composition, label inheritance, and `CurationEvaluation` acceptance helpers. **phase-1c.**
- `v2/_sorting_dispatch.py:572-619` — sorter output materialized before temp-dir cleanup. **phase-1.**
- `spikesorting_merge.py:430-478,540-574` + `v2/unit_matching.py:286-297,587-625` — preview-curation guard. **phase-1.**
- `v2/utils.py:171-254,257-309` — `SelectionMasterInsertGuard` gains `update1`; new master guard for `CurationV2`/`SessionGroup`. **phase-2.**
- `v2/_params/preprocessing.py:7,106-121` + `v2/_concat_recording.py:22,93-146` + `v2/recording.py:2200` — runtime-alias versioning. **phase-2.**
- `v2/unit_matching.py:438-457,587-625,864-907` — persist + consume the frozen matchable universe. **phase-2.**
- `v2/_lookup_validation.py:83-93,264-282` + `v2/__init__.py:18-61` — schema-init version backfill + stale-default audit. **phase-2.**
- `v2/utils.py:607-632`, `v2/sorting.py:1127-1135`, `v2/unit_matching.py:103-110,675-681`, `v2/_unitmatch_backend.py:107-116` — provenance columns + effective-seed capture. **phase-3a.**
- `v2/_recording_nwb.py`, `v2/_units_nwb.py`, `v2/_unitmatch_nwb.py`, `v2/_metric_curation_nwb.py`, `v2/session_group.py` writers — NWB provenance. **phase-3b.**
- `v2/_analyzer_cache.py:92-135` + `v2/_sorting_analyzer.py`/`sorting.py`/`recompute.py` mutation sites — analyzer-cache lock + atomic publish. **phase-4a.**
- `v2/_concat_recording.py:217-287` + `v2/_unitmatch_backend.py:184` — concat compatibility + UnitMatch 2D geometry. **phase-4a.**
- `v2/sorting.py:1268-1272,1502-1511` + `v2/artifact.py:940-946` — bypass revalidation. **phase-4a.**
- `v2/unit_matching.py:780`, `v2/recompute.py:1115` — temp routing. **phase-4a.**
- `v2/sorting.py:328-331,2429-2449,504-558`, `v2/_sorting_dispatch.py:75-88,504-515` — dispatch mismatches. **phase-4b.**
- `pyproject.toml:58,69`, `environments/environment_spikesorting_v2.yml:56` — dependency pins. **phase-4b.**
- `utils/mixins/analysis.py:205-227,300-302,316-318,341-348`, `v2/_sorting_dispatch.py:486`, `common_nwbfile.py:107`, v2 writers — permissions/path/conda. **phase-4b.**
- `v2/metric_curation.py:73`, `v2/recompute.py:67` — `_assert_v2_db_safe()`. **phase-4b.**
- `spikesorting_merge.py:36-44,127-134` — log/surface the eager v2 probe failure while preserving the broad fallback (do NOT narrow the except). **phase-4b.**
- `v2/recompute.py:1216-1217`, `v2/sorting.py:2274-2278`, `v2/_recording_nwb.py:258-275`, `common/common_usage.py:547-550` — footguns + Export.File leak. **phase-4b.**

Left alone (explicit non-targets): all Phase 1–4 *table primary keys* (no PK changes — see R9-PK below); the recording content-fingerprint identity (R6 full recipe — decided won't-do); v0/v1 spike-sorting modules; the `cautious_delete`/`Session.Experimenter` permission model (unchanged, matches v1).

## Scope and dependency policy

### Goals

- Eliminate every verified silent-wrong-science path (phase-1).
- Add final metric evaluation over committed curated unit sets, including merged
  units (phase-1b).
- Make child curation edits compose from the parent state rather than raw
  `Sorting.Unit` rows, and provide explicit acceptance helpers for evaluation
  outputs (phase-1c).
- Make identity, reproducibility, and provenance correct in the schema **before any real data is retained** (phases 2, 3a, 3b).
- Harden operational and security surfaces to a trusted-lab-deployment standard (phases 4a, 4b).
- Leave `CurationV2` decomposed so the Phase 5 orchestrator/FigPack work extends clean service boundaries (phase-0).

### Non-Goals

- **No RBAC / team access enforcement** (R9 decided: `team_name` is a provenance tag; enforcement, if ever wanted, is a future additive PR — the ownership columns are already stored). Only the TEAM-1 overwrite footgun is fixed (phase-4b scope note).
- **No plugin frameworks** (R33 decided: no matcher/sorter/viz `register_*` API). phase-3a only adds matcher *backend-version provenance* and collapses the misleading registry honesty; it does not build a plugin contract.
- **No full construction-recipe-as-identity** for recordings (R6 decided: keep the post-write `content_hash`; only version the runtime aliases).
- **No PK changes** to the session-global keys (`SortGroupV2`/`SharedArtifactGroup`/`ExportSelection`) — see Open Question R9-PK.

### Dependency policy

phase-4b pins `numpy>=2,<3` (currently bare `pyproject.toml:58`) and reconciles the `environment_spikesorting_v2.yml` SpikeInterface range (`>=0.104,<0.105`) with the `pyproject.toml` hard pin (`==0.104.3`). No new runtime dependencies are introduced; `filelock` (already used for the recording lock) is reused in phase-4a.

## Metrics

- phase-1: each bug has a repro test that **fails before the fix and passes after**; the existing parity/attribution suites stay green.
- phase-1b: final metrics over a committed merged curation are recomputed in the
  curation's unit namespace; preview/draft curations are rejected at evaluation
  boundaries.
- phase-1c: child curations compose from parent unit ids rather than raw ids;
  labels inherit/replace according to explicit policy; evaluation acceptance
  helpers create committed children.
- phase-2: a new test asserts `update1` is rejected on every selection master and on `CurationV2`/`SessionGroup` (mirrors the existing param-Lookup `update1` tests); a content change under a fixed param name forces a detectable outcome.
- phase-3a/3b: round-trip tests assert the effective seed + producing versions are persisted on the computed row and re-emitted in the artifact NWB.
- phase-0: the full v2 test suite passes unchanged (byte-identical behavior); the new service module is covered by the import-boundary contract test.
- Full gate: the heavy v2 suite (`pytest tests/spikesorting/v2`) green at every phase boundary.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| phase-0 extraction subtly changes accessor behavior | Extraction keeps public classmethods as thin wrappers; behavior pinned by the existing accessor tests (listed in phase-0) which must pass unchanged. Anti-theater: only genuinely pure logic is moved; DB routers stay on the table. |
| phase-3a adds identity-bearing columns → could shift deterministic ids | Provenance columns are **secondary attributes**, never primary-key / identity inputs (per shared-contracts). Effective-seed *capture* changes stored data but not existing ids; a parity test confirms ids are unchanged for a fixed input. |
| phase-1 SNR-polarity fix changes metric values for existing test sorts | Default sorts are negative-going, so the resolved `peak_sign` stays `"neg"` and values are unchanged; the new behavior only affects positive/bidirectional sorters. A regression test pins the negative-default value. |
| phase-1b temporary analyzers for merged curations are slower than cache reuse | Root/label-only curations use the cached raw-sort analyzer fast path; merged curations build temp analyzers over the committed curation unit set and clean them immediately. A persistent curation-scoped cache can be added later with explicit identity/recompute/orphan policy. |
| phase-1c child-composition changes curation semantics | This matches the user workflow ("edit this curation") and fixes the raw-id resurrection problem. Tests cover label inheritance, parent-namespace merge groups, raw provenance expansion, and no preview rows as canonical outputs. |
| phase-2 frozen-universe persistence adds a `UnitMatch.MatchableUnit` part table | Pre-production, schema freeze lifted — additive part table, no migration. Covered by the existing TrackedUnit tests + a new relabel-divergence test. |
| phase-0 vs phases 1–4 file overlap | Sequence phase-1 first (urgent); rebase whichever of phase-0 / 2–4 lands second. Conflicts are mechanical (different methods in the same file). |

## Rollout Strategy

Replace-in-place. Pre-production, no users, schema freeze lifted, so schema
additions (phases 2, 3a) need no migration or deprecation window. Each phase is one
PR, merged behind the standard heavy-suite gate. Recommended merge order:
phase-1 → phase-1b → phase-1c → phase-2 → phase-3a → phase-3b → phase-4a → phase-4b → phase-4c → phase-6,
with phase-0 merged at any point before the Phase 5 UX work starts (and before
phase-4a's REL-4 task). phase-4c (concat lifecycle) can also land any time after
phase-4a; phase-6 (CI gates) is best **last**, so it protects the corrected behavior.

## Phase 5 adjustments (companion edit to `../spikesorting-v2/phase-5-ux-overhaul.md`)

Applied as part of this work, not a separate plan — these items were decided as
Phase-5-scoped in the triage:

- **R2 / DOCS-1 / API-2** — the canonical notebook and `SpikeSortingV2.md` must carry the **final curated** `merge_id`, not the root (`parent_curation_id=-1`) curation that `run_v2_pipeline` returns. Add to the Phase 5 docs/notebook task.
- **R20** — `run_v2_pipeline` orchestrator/preflight polish: add the required `"raw data valid times"` interval check to preflight (ERR-1), surface structured stage/cause in batch-failure results (ERR-7), distinguish selection-vs-computed in preflight expected ids (OBS-5).
- **R23 / MAINT-9** — fix the self-contradicting "not yet available / placeholder" status text for shipped UnitMatch/concat surfaces; storage page into nav; the `mkdocs.yaml` typo, broken `Export.md` example, and unset `version_string` fallback.
- **R33-EXT-2** — preset registration is a Phase 5 deliverable; provide a clean preset surface (not a generic plugin API). **R33-matcher** — the matcher docs must say "UnitMatchPy is the matcher," matching the registry-honesty change in phase-3a.
- **R7-composition UI** — phase-1c decides the backend semantics (child edits
  compose from the parent state). Phase 5 should build FigPack/notebook UI around
  that committed-state model rather than re-deciding the storage contract.
- **MAINT-10** — notebook structure (beginner path vs. advanced reference) is a Phase 5 design call.

## Open Questions

1. **R9-PK** — should `SortGroupV2`/`SharedArtifactGroup`/`ExportSelection` gain `owner` in their *primary key* to allow per-team namespacing? This is the only non-additive change (it shifts `recording_id` identity downstream), so it is free now and a migration later. **Current best answer: no** (keeps v2 == v1; access *enforcement* remains a clean future additive PR). Revisit only if per-team namespacing becomes likely. Not in any phase below; flagged here so the executor does not silently add it.
2. **R27 semantics** (phase-1 / phase-1b / phase-1c) — decided: phase-1 guards
   raw-sort `AnalyzerCuration` (no silent re-base); phase-1b adds the proper path
   `CurationEvaluation`, which evaluates the committed `CurationV2` unit set
   directly (including merged units). Because v2 is pre-production with no
   backwards-compatibility requirement, `CurationEvaluation` then *replaces*
   `AnalyzerCuration`: phase-1c migrates the remaining callers and DELETES
   `AnalyzerCuration` outright (no deprecation window). It is kept only through
   phase-1b because the auto-curate→materialize-child flow needs it until 1c's
   acceptance helpers land. Downstream phases that used to mention
   `AnalyzerCuration` are now interpreted as targeting `CurationEvaluation` and
   its selection/result NWB writer after phase-1c.

## Estimated Effort

- phase-0: ~+400 LOC service modules / −250 LOC from `curation.py` bodies (net small; mostly moved), + tests.
- phase-1: ~6 surgical fixes, ~+250 LOC incl. repro tests.
- phase-1b: ~+300-500 LOC depending on how much can reuse the existing
  metric/analyzer helpers; adds curation-evaluation tables, temp analyzer build,
  final-metric tests, fast-path coverage, and docs.
- phase-1c: ~+300-500 LOC for parent-state curation composition, optional
  parent-operation provenance, label inheritance, evaluation acceptance helpers,
  and tests.
- phase-2: ~+300 LOC (one guard mixin, alias-version fields, the `UnitMatch.MatchableUnit` part table, schema-init audit) + tests.
- phase-3a: ~+250 LOC (provenance columns across 3–4 tables + capture logic) + tests.
- phase-3b: ~+300 LOC (writer metadata across 5 writers) + tests.
- phase-4a: ~+250 LOC (lock wiring + 3 compat checks + temp `dir=`) + tests.
- phase-4b: ~+200 LOC (many small, localized fixes) + tests.
- phase-4c: ~+400 LOC (member snapshot + concat verify-on-read/rebuild; +recompute tables if not deferred) + tests.
- phase-6: mostly CI/fixtures + new science assertions — small LOC, but gated on publishing the two-session fixtures.
