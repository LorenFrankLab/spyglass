# Spike Sorting V2 — Combined Review Triage

Date: 2026-06-25
Branch: `spikesorting-v2` (verified against HEAD `b9cacd97`)
Status of pipeline: **pre-production, no current users.** Phases 0–4 complete;
**Phase 5 (UX overhaul: `run_v2_pipeline` orchestrator, FigPack curation UI,
Pydantic presets, notebook/docs) is next.**

## What this is

A consolidated, **claim-verified** triage of the **26 review documents** in this
directory (~207 individual findings), verified in **two rounds** as the reviews
arrived. Every finding was re-checked against the current code by independent
verification agents (read-only; one agent per thematic cluster), then key
correctness/identity claims were spot-checked directly. Some reviews were written
*before* fix commits landed, so a fraction are already stale.

- **Round 1** (16 reviews, ~130 findings): CONC, DESTR, OBS, REPRO, SCHEMA, TEST,
  DEP, CONFIG, IMPORT, DOWN, API, DOCS, ERR, NWB, PERF, TEAM.
- **Round 2** (10 reviews, ~77 findings): CLIFE (curation-label-lifecycle), REL
  (relational-query-integrity), EXT (extension-customization), SIG
  (signal-units-preprocessing), MOTION (motion-drift-spatial), TIME
  (timebase-sample-index), DISP (sorter-dispatch-execution), SEC
  (security-external-boundary), SER (serialization-payload), MAINT
  (maintainability-module-boundary). Root issues **R27–R37** below are new from
  this round; many Round-2 findings are duplicates that map onto existing R#.

### Verification method

- Each finding opened at its cited code path (or the current equivalent if a
  path moved) and classified: **CONFIRMED** (true now), **PARTIAL** (some
  sub-claims hold), **STALE-FIXED** (already resolved by a recent commit),
  **FALSE**, or **UNVERIFIABLE**.
- Cross-review duplicates were clustered into distinct *root issues* (the
  decision unit below). Finding IDs use a per-review prefix.

### Verdict tally (~207 findings)

| Verdict | Count | Notes |
|---|---|---|
| CONFIRMED | ~183 | true against current code |
| STALE-FIXED | 9 | already resolved — close out (see §Done); all from Round 1 |
| PARTIAL | ~13 | partly true / narrowed by a recent fix |
| FALSE | 0 | — |
| UNVERIFIABLE | 0 | — |

Round 2 found **0 stale-fixed and 0 false** — its 10 reviews target areas the
recent fix commits did not touch, so all 77 findings stand (2 PARTIAL: SEC-5,
SER-7). Already-fixed by post-review commits (385e7d31, 63e0ff5a, 12286892,
d08c319b, db8d271e, e8e06553, 18f05527): **CONC-3, OBS-1, REPRO-1, REPRO-2,
TEST-1, TEST-2, SCHEMA-5, DOCS-4, API-1** (plus the recording-breadcrumb half of
OBS-7).

## Triage philosophy (pre-production, no users, Phase 5 next)

Because there are **no users and no retained real data yet**, the usual
"don't break existing users" pressure is absent. That *inverts* two priorities:

1. **Identity / provenance / schema decisions are cheapest to fix NOW**, before
   real sorted data accumulates and a change becomes a migration. The schema
   freeze is lifted, so adding columns is free today and expensive later. These
   rank **higher** than they would in a shipped product.
2. **Operational hardening (concurrency races, admin footguns, observability)
   ranks lower** for now — single-user testing won't hit most of it — *unless*
   it can silently corrupt an output or bake a wrong value into a stored row.
3. **Phase 5 is a UX/docs/orchestrator pass**, so a large share of the
   docs/API/error-taxonomy findings should be *folded into Phase 5* rather than
   done as separate work.

Priority tiers used below:

- **MUST** — fix before retaining real v2 data / before Phase 5 ships. Silent
  wrong-science, wrong user guidance, or reproducibility-baked-into-rows.
- **SHOULD** — do during Phase 5; cheap safety/correctness or naturally in Phase 5 scope.
- **NICE** — post-Phase-5 robustness/observability/perf/portability.
- **DECIDE** — needs a product/design decision before any code (policy, scope).
- **DONE** — stale-fixed; verify and close.

---

## Decision table — root issues (deduplicated)

Each row is one underlying problem; the many overlapping findings collapse here.
"Effort" is a rough estimate (S < ½ day, M ~1–2 days, L > 2 days / design).

| # | Root issue | Findings | Verified | Tier | Effort | Rationale / Phase-5 note |
|---|---|---|---|---|---|---|
| R1 | **Reproducibility provenance not baked into output rows** — effective `random_seed` comes from SI globals/`dj.config` and is neither captured nor rejected from identity; producing versions (sorter, UnitMatchPy, key libs) not stored; UnitMatch waveform-bundle params (`ms_before/after`, `max_spikes_per_unit`, `seed`) absent from `MatcherParameters` identity; analyzer `noise_levels` unseeded/unmarked | REPRO-4, REPRO-5, REPRO-8, TEST-4, TEST-5, TEST-9, DEP-1, DEP-2, CONFIG-1, CONFIG-10 | CONFIRMED | **MUST** | M–L | Core promise of content-addressed v2. Columns/identity are cheapest to add pre-data. Do the *minimal high-value subset*, not a full runtime manifest. |
| R2 | **Docs/notebook steer users to the root (uncurated) `merge_id`** instead of the final curated output (`run_summary["merge_id"]` is the `parent_curation_id=-1` root) | API-2, DOCS-1 | CONFIRMED | **MUST** | S | Users following docs decode uncurated spikes. Cheap; fold into Phase 5 docs/notebook. Verified at `_pipeline_run.py:403-423`. |
| R3 | **Preview / unmerged curations are consumable as final outputs** — generic merge accessors (`get_spike_times`/`get_spike_indicator`/`get_firing_rate`) and `UnitMatch` don't check `CurationV2.has_unapplied_proposed_merges`; only the decode edge guards | DOWN-1, DOWN-2 | CONFIRMED | **MUST** | M | Silent wrong science. Decide enforcement at the shared output layer; Phase 5 touches these accessors. |
| R4 | **Sorter output can be freed before it is consumed** — `_run_si_sorter` returns `run_sorter(...)` then `sorter_temp_dir.cleanup()` runs in `finally` before the caller reads the (file-backed) lazy sorting | PERF-3 | CONFIRMED | **MUST (verify first)** | S–M | Verify the default MS4 path returns a file-backed sorting; if so, materialize (`NumpySorting`) or defer cleanup. Potential corruption/crash. `_sorting_dispatch.py:574/599`. |
| R5 | **Analyzer canonical-folder mutation is unserialized and non-atomic** — one per-sort lock exists but is held only by `AnalyzerCuration`; every rebuild/build/delete/orphan-sweep writes/removes the canonical `*.zarr` directly (`overwrite=True`/`rmtree`), no temp-then-rename. A duplicate-loser's cleanup can `rmtree` the winner's committed folder | CONC-1, CONC-2, DESTR-3, OBS-2 | CONFIRMED | **SHOULD** | M | The *recording* side already got this exact treatment (lock + atomic publish, commit d08c319b); mirror it on the analyzer side. Self-heal rebuild softens data-loss to wasted rebuild + concurrent-read risk, so SHOULD not MUST. |
| R6 | **Recording construction identity is row-name based, not recipe-based** — `content_hash` is a post-write digest, not part of identity; same row name + changed upstream (intervals, reference/sort-group, preprocessing runtime order, motion/drift preset alias) silently yields new content under the same `recording_id` | REPRO-3, TEST-3, REPRO-7, SCHEMA-6, SCHEMA-8 | PARTIAL/CONFIRMED | **DECIDE** | L | The full-construction-fingerprint open questions were already **decided won't-do** (commit b9cacd97). Respect that. Cheaper residual worth doing: version the runtime-semantics aliases (preprocessing order, `"auto"`→`rigid_fast`, dredge preset) so a semantics change forces new identity — pull that subset into R1/SHOULD. |
| R7 | **Curation lineage / tracked-unit universe not frozen** — child curations are full snapshots rebuilt from raw `Sorting.Unit` (parent is validation-only); `TrackedUnit` re-derives its node universe from *current* labels; analyzer-curation materializations keep no FK to the producing `AnalyzerCuration` | REPRO-6, TEST-6, TEST-10 | CONFIRMED | **DECIDE/NICE** | L | Design call: freeze label-set snapshots vs. accept re-derivation. Pre-data is the time to decide; implementation can wait. |
| R8 | **Schema-init is not an upgrade workflow** — omitted outer `params_schema_version` stored as current even when the blob disagrees; same-name stale shipped defaults survive seeding silently; interim row names block seeding with no rename/audit path | SCHEMA-1, SCHEMA-2, SCHEMA-3 | CONFIRMED | **SHOULD** | M | You are actively editing defaults pre-release; a mislabeled version or surviving stale default mis-tags computed provenance. SCHEMA-1 (version backfill) is the sharpest. |
| R9 | **Multi-user ownership is a namespace, not an enforced boundary** — no v2 path checks `database.user → LabTeamMember` on insert/curate/delete/recompute/group-creation; ownerless global keys (`SortGroupV2` session-global overwrite cascade, `SharedArtifactGroup`, `ExportSelection`) enable cross-team overwrite/collision; delete still uses the legacy Session-experimenter model | TEAM-1…TEAM-9 | CONFIRMED | **DECIDE** | L | One policy question repeated 9 ways: *is `team_name` a tag or an access boundary?* No live breach (no users). Decide the policy; at minimum guard the `SortGroupV2` session-global overwrite cascade (TEAM-1) before multi-team use. |
| R10 | **NWB artifacts store results, not provenance** — recording/concat/units/analyzer/UnitMatch NWBs omit source object-ids, sorter/params, member boundaries, merge lineage, versions; not self-describing for off-DataJoint exchange | NWB-1…NWB-8, NWB-10 | CONFIRMED | **NICE/DECIDE** | L | DataJoint is the source of truth today, so defer — *unless* near-term DANDI/portable export is on the roadmap, in which case NWB-1 (recording) + NWB-2 (concat boundaries) are the load-bearing first slice. |
| R11 | **Explicit per-frame timestamps re-introduce O(n_samples) cost** — every artifact writes explicit `timestamps` and reads them back eagerly through the pinned SI extractor, bypassing the affine `rate`/`starting_time` representation the lazy-timestamp helpers were built for | PERF-1, PERF-2, TEST-8 | CONFIRMED | **NICE** | M | Scales with recording size; Phase 5 will run real data. Measure baseline before optimizing (per repo norms). |
| R12 | **Scratch/temp not routed to configured Spyglass temp + no RAM/disk budget** — UnitMatch bundle and analyzer recompute use bare system `/tmp`; recompute hashing and dense UnitMatch/analyzer builds materialize multi-GB arrays with no preflight estimate | PERF-5, PERF-6, PERF-8, CONFIG-6 | CONFIRMED | **SHOULD** | S | Routing `dir=spyglass_temp_dir` is a 1-line fix in 2–3 places and prevents filling `/tmp`. Budget estimator is NICE. |
| R13 | **`job_kwargs`/`n_jobs` resolved then discarded** on the recording-write and analyzer/quality-metric paths — user parallelism silently no-ops | PERF-4, PERF-7 | CONFIRMED | **NICE** | M | Surprising but not incorrect. Document the limitation now; wire through later. |
| R14 | **`_assert_v2_db_safe()` not called directly in `metric_curation.py` and `recompute.py`** (transitive coverage only); guard is import-time-only and undocumented | CONFIG-4, IMPORT-6, CONFIG-5 | CONFIRMED | **SHOULD** | S | Two-line fix restores the per-module contract. Document the env override in Phase 5. |
| R15 | **`SpikeSortingOutput` eagerly probes v2 at shared-merge-table import** with a broad `except` — one transient import failure removes the v2 part for the whole process | IMPORT-2 | CONFIRMED | **SHOULD** | S–M | v0/v1/decoding imports pay the v2 probe; a flaky failure silently disables v2. Narrow the except / make lazy. |
| R16 | **Dependency contracts diverge** — `pyproject` pins `spikeinterface==0.104.3` but env yml is `>=0.104,<0.105`, docs say "0.104+", boundary test is a floor; base `numpy` is unpinned despite NumPy-2 assumptions; live SI registries drive metric validation | DEP-3, DEP-4, DEP-5, DEP-9 | CONFIRMED/PARTIAL | **SHOULD** | S | Reconcile the four contracts and pin `numpy>=2,<3`. Cheap; prevents a silently-wrong resolve. |
| R17 | **Import-light / DB-free boundary weaker than advertised** — v2 schema modules import `spyglass.common` root (runs prepopulate), `utils` barrel drags DataJoint+SI into pure helpers, pandas/pydantic required-but-unpinned, `common_nwbfile` pulls the NWB stack to declare an FK | IMPORT-1, IMPORT-3, IMPORT-4, IMPORT-5 | CONFIRMED | **NICE** | M | Latency/coupling, not correctness. Tighten import-boundary tests opportunistically. |
| R18 | **Destructive-admin footguns** — orphan analyzer sweep can `rmtree` arbitrary dirs under a misconfigured root (no name-pattern filter); negative `days_since_creation` bypasses the age gate; dead in-place writer branch can unlink the canonical file on failure; `force_stale_env` not durably audited; opaque deletion return (bare path list, ambiguous `[]`) | DESTR-2, DESTR-5, DESTR-6, DESTR-7, DESTR-8, OBS-6 | CONFIRMED | **SHOULD** | S–M | DESTR-6 (validate `>=0`) and DESTR-8 (name-pattern filter) are trivial and prevent catastrophic deletion → do these. Audit/structured-report → NICE. |
| R19 | **Side-effect cleanup tied to child `delete()` overrides; no orphan cleaner for artifact `IntervalList` rows** — parent/selection/`super_delete` cascades bypass analyzer + interval cleanup; partial deletes are non-resumable | DESTR-1, DESTR-4, OBS-3 | CONFIRMED | **NICE** | M | Leak, not corruption; analyzer orphans already have a finder. Add a dry-run interval-orphan finder. |
| R20 | **`run_v2_pipeline` orchestrator polish** — preflight misses the fixed `"raw data valid times"` interval (late populate failure); `SessionGroup.create_group` leaks `KeyError`/raw FK errors; failed receipts lose completed-stage timing; batch failures drop structured stage/cause fields; preflight expected-ids don't distinguish selection vs computed | ERR-1, ERR-2, OBS-4, OBS-5, ERR-7 | CONFIRMED | **SHOULD** | M | Phase 5 rewrites the orchestrator — fold these in. ERR-1 is cheap and high-value (add the raw-valid-times check + defensive guard). |
| R21 | **Error taxonomy depth** — direct/bypass workflows lack the stage wrapper and leak raw SI/HDF5/DataJoint errors; malformed public input dicts leak bare `KeyError`/`ValidationError`; no common `SpikeSortingV2Error` base for batch callers | ERR-3, ERR-4, ERR-5, ERR-6, API-6 | CONFIRMED | **NICE** | M | Add a `SpikeSortingV2Error` base (cheap) in Phase 5; wrap direct paths incrementally. ERR-6 (`preflight=False`) is by-design. |
| R22 | **Downstream accessor contracts** — all-unlabeled curations break `include_labels` filtering; `get_pairs()` loses schema for zero pairs; `get_unit_brain_regions()` drops disambiguators; duplicated `delete_quick` leaves stale `Export.File` rows; public docs misstate DataFrame shape | DOWN-3, DOWN-4, DOWN-5, DOWN-6, DOWN-8 | CONFIRMED | **SHOULD/NICE** | S–M | DOWN-6 is a clear cheap bug (fix the duplicated line + delete `self.File`). Others are small contract hardening. |
| R23 | **Docs/build paper cuts** — `mkdocs.yml`/`mkdosc.yaml` typo, broken generic `Export.md` example, unset `version_string` fallback, storage page orphaned from nav, stale "not yet available"/placeholder status for shipped UnitMatch/concat, preflight advertised but skipped in snippet, install snippets checkout-only | DOCS-2/3/5/6/7/8, API-3/5/7/8, DEP-9, IMPORT-7, TEAM-9 | CONFIRMED | **SHOULD** | S–M | Mostly trivial; all squarely in Phase 5 docs scope. The stale "placeholder" status text is the most misleading (contradicts shipped features). |
| R24 | **Selection-helper inconsistency** — `ConcatenatedRecordingSelection` ignores a supplied PK (no `assert_supplied_id_matches`); `QualityMetricParameters` has an undocumented implicit duplicate-content policy; uneven extra-field/signature shapes | API-4, SCHEMA-10, API-9, API-10 | CONFIRMED | **NICE** | S | API-4/SCHEMA-10 is a cheap consistency fix (the helper already exists). |
| R25 | **Recompute diagnostics lack input fingerprints** — version tables store output hashes but not upstream parameter-row fingerprints/schema versions, so drift can't be traced to a migration cause; v2 schema drift (new `content_hash`/concat fields) not snapshotted/gated | SCHEMA-4, SCHEMA-7 | CONFIRMED | **NICE** | M | Diagnostic completeness; do alongside R8 if convenient. |
| R26 | **Misc runtime/config gaps** — preflight gate (`ml_ms4alg`) not reused by direct execution; container images tag-pinned not digest-pinned; optional KS4/UnitMatch scientific CI lanes skip-gated; `conda env export` is a hard dep on the NWB write path; analyzer cache not namespaced by DB host | DEP-6, DEP-7, DEP-8, CONFIG-8, CONFIG-3, CONFIG-9, CONFIG-2 | CONFIRMED | **NICE** | S–M | CONFIG-8 (wrap `conda env export` in try/except) is cheap and prevents write failures in conda-less envs → consider SHOULD. |
| R27 | **AnalyzerCuration computes over the raw-sort namespace, not the selected curation** — `AnalyzerCuration.make_compute` loads the analyzer for `{sorting_id}` and computes metrics/labels/merges over raw `Sorting.Unit`, ignoring the selected child curation's merged/labeled unit set; only a soft warning when chaining analyzer curations | CLIFE-1 | CONFIRMED | **MUST (confirm semantics)** | M | Breaks the documented auto→merge→auto iterative-curation loop: re-running auto-curation on a merged curation silently scores the *un-merged* units. Wrong science for an advertised workflow. Confirm intended semantics with owner, then guard or re-base on the selected curation. `metric_curation.py:~916/933/1399`. |
| R28 | **Master-row identity is mutable / directly writable** — `SelectionMasterInsertGuard` guards `insert` only (no `update1`), and `ImmutableParamsLookup.update1` is not mixed into selection masters, so a deterministic selection ID can be retargeted by `update1`; `CurationV2` and `SessionGroup`/`Member` masters are plain `dj.Manual` with no guard (direct writes bypass parts/merge registration; member edits retarget materialized concat/UnitMatch provenance); `UnitMatch.Pair` FK is to all `CurationV2.Unit`, not the matchable subset; source-part exclusivity is read-time only, not audited; post-insert `UnitLabel` edits desync DB vs the already-written NWB; `allow_param_mutation=True` `update1` bypasses insert validation | REL-1, REL-2, REL-3, REL-5, REL-6, CLIFE-5, CLIFE-3, DISP-6 | CONFIRMED | **SHOULD** | M | Symmetric gap to the identity-hardening work already done on param Lookups. Cheapest while no data exists: add a default-deny `update1` to `SelectionMasterInsertGuard`, an insert/update guard + `insert_curation`-only contract on `CurationV2`/`SessionGroup`, and a source-part integrity audit. |
| R29 | **Silent signal / numerical bugs** — (a) `no_filter` + referencing persists a re-referenced signal with the original DC offset reattached (default filter-then-reference is safe); (b) SNR metric hard-codes `peak_sign="neg"` while unit attribution honors sorter polarity, so positive/bidirectional detectors (incl. clusterless `pos`/`both`) get opposite-signed SNR; (c) rate-based interval restriction loses boundary samples via unsnapped float `ceil`/`floor` (reproduced at 30 kHz) | SIG-1, SIG-2, TIME-1 | CONFIRMED | **MUST** | S–M | All three silently produce wrong scientific values and persist them into content-addressed rows. TIME-1 (boundary-sample loss on interval restriction) and SIG-2 (SNR polarity, broad blast radius) are the sharpest; SIG-1 is narrow to `no_filter`. Each fix is small. Numerically verified by the reviewer. |
| R30 | **Concat spatial/rate compatibility is under-checked** — `assert_concat_compatible` compares only channel-id order + numeric coords, not electrode identity/probe/shank/brain-region, so members in different physical electrode spaces can be concatenated then read in the anchor frame; no sampling-frequency compatibility check; UnitMatch receives raw `get_channel_locations()` with no 3D→2D projection or `shape[1]==2` guard (the analyzer's projection helper is not shared; the analyzer itself only warns) | MOTION-1, TIME-2, MOTION-2, SIG-4 | CONFIRMED | **SHOULD** | M | Can silently combine incompatible recordings / feed a 3D geometry into a 2D matcher contract. Add identity+region+fs checks to `assert_concat_compatible` and a 2D-projection/guard on the UnitMatch geometry path. |
| R31 | **Compute paths trust insert-time guarantees a direct write can violate** — concat+artifact selection is rejected at the normal insert boundary but a bypass-inserted row sorts *unmasked* while reporting artifact metadata; shared-artifact-group compute trusts insert-time same-session/fs checks and computes valid times on one member's timeline; `{preprocessing_params_name}` restriction silently omits concat-backed curations (and the valid `{concat_recording_id, preprocessing_params_name}` pair is wrongly rejected) | TIME-3, TIME-4, REL-4 | CONFIRMED | **SHOULD/NICE** | M | Re-validate at the compute boundary (don't trust insert-time only) and fix the concat-curation query routing. Lower likelihood under normal API use; pairs with R28. |
| R32 | **Sorter-dispatch execution mismatches** — clusterless rows accept/fingerprint `execution_params` and falsely require a container in preflight but run local Python; `insert_default_legacy_si_sorters` creates local MATLAB rows the dispatch then rejects; any truthy `whiten` is silently rewritten to pinned external whitening with no sorter allowlist (surprises a generic sorter); `delete_container_files=False` is a no-op under Spyglass temp cleanup | DISP-1, DISP-3, DISP-4, DISP-2 | CONFIRMED | **SHOULD** | S–M | Cheap correctness/consistency fixes in the execution plumbing: reject non-local `execution_params` for `_NON_SI_SORTERS`, skip/curate MATLAB rows in the legacy seeder, allowlist the whiten interception, document the temp-cleanup behavior. |
| R33 | **Extension surfaces advertised/readable but actually closed** — matcher plugins still hard-depend on UnitMatchPy bundle extraction (no `prepare_session` contract); pipeline presets are a private static dict with no `register_pipeline_preset`; sort-time waveform recipe is a private static map not a per-sort knob; matcher registry `replace=True` can re-route existing rows and stores no backend provenance; custom sorters are a generic SI escape hatch; visualization discovery is a closed static registry; custom auto-curation rule labels pass insertion then fail at materialize | EXT-1, EXT-2, EXT-3, EXT-4, EXT-5, EXT-6, EXT-7 | CONFIRMED | **DECIDE/Phase-5** | M–L | One theme: "advertised-but-closed contract." EXT-2 (presets) is in Phase-5 scope; EXT-4 matcher-backend provenance ties to R1; EXT-6 is a cheap late-failure fix (validate rule labels at insert). Mostly honest-docs + small registration APIs; decide how much extensibility to commit to. |
| R34 | **Filesystem permissions & path confinement (security)** — v2 NWB writers never pass `restrict_permission=True` (artifacts are `0o666`) and sorter scratch is `chmod 0o777` even on local runs; raw/analysis filenames are never basename-validated or root-confined (path traversal via crafted filename); shared file helpers interpolate filenames into `LIKE`/`=` DataJoint restrictions; `SorterParameters` rows are a code-execution boundary with no image/registry allowlist or documented trust model | SEC-2, SEC-1, SEC-6, SEC-3 | CONFIRMED | **SHOULD / DECIDE** | S–M | Trusted-lab threat model: most are latent, but **SEC-2 is the real one** on a multi-user cluster (other OS users can tamper with registered NWBs / live scratch) — `restrict_permission=True` + dropping the gratuitous local `0o777` is cheap → SHOULD. SEC-1/3/6 are latent hardening + a *documented* "writers are trusted operators" policy → DECIDE. |
| R35 | **Validation escape-hatches & payload-shape stability** — JSON-native/finite is enforced only as a fingerprint side effect, not a first-class insert invariant (bypassed by `allow_duplicate_params=True`; `_resolved_job_kwargs` blind-updates with no finite check; `threshold` is a bare float); `QualityMetricParameters`/`AutoCurationRules` skip the duplicate-content guard the docs imply; sort-time `curation_label` is a scalar while the docstring says a list; artifact-interval return shape flips array↔dict by source kind | SER-2, SER-3, SER-5, SER-6 | CONFIRMED | **NICE** | M | Hardening + contract consistency; SER-5 also has a code/docstring contradiction worth a quick fix. SER-3 overlaps R24. |
| R36 | **Cheap test / single-source-of-truth wins that ease Phase 5** — tri-part NamedTuple contract tests cover only 3 of ~7 carriers (extend to Artifact/Concat/UnitMatch/recompute/DriftEstimate); 4 parallel catalogs of stage/lookup knowledge with one live drift (notebook says 3 defaults, 8 are seeded) — add consistency tests + fix the doc; slow integration setup is hand-rolled in 4 modules (add a `populate_single_session_chain` helper); insert/insert1/insert_default boilerplate not yet hoisted into the existing `ImmutableParamsLookup` mixin | MAINT-3, MAINT-4, MAINT-5, MAINT-8 | CONFIRMED | **SHOULD** | S–M | All cheap, low-risk, and reduce drift before Phase 5 touches these surfaces. MAINT-8 == R24. Note: the file-splitting half of MAINT-1/2 is **already-rejected** per the user's decision; keep only the "table owns schema+txn, service owns logic" rule for new code. |
| R37 | **Stale scaffolding / contributor hazards** — the old phase-0 fixture-generation plan still describes the superseded ground-truth-in-`nwbfile.units` contract (current tests require `ProcessingModule["ground_truth"]` + empty `nwbfile.units`); clusterless smoke row named `5uv` but behavior is MAD; facade/module status text lags shipped surface (= R23) | EXT-8, SIG-7, MAINT-9 | CONFIRMED | **NICE/Phase-5** | S | EXT-8 can poison regenerated fixtures — mark the plan superseded. SIG-7 is naming hygiene. MAINT-9 folds into R23. |

---

## Tier rollup

### MUST — before retaining real v2 data / before Phase 5 ships
- **R29** Fix the silent signal/numerical bugs: rate-interval boundary-sample loss (TIME-1), SNR-polarity vs sorter detect-sign (SIG-2), `no_filter`+reference DC-offset reattach (SIG-1). *(New in Round 2 — highest-confidence wrong-science.)*
- **R1** Bake the minimal reproducibility provenance into rows (effective seed; reject/flag ambient science-affecting `job_kwargs`; store sorter + UnitMatchPy + key-lib versions; surface UnitMatch bundle params into identity; mark analyzer `noise_levels` provenance).
- **R27** AnalyzerCuration over the raw-sort namespace breaks the auto→merge→auto loop (CLIFE-1) — confirm intended semantics, then guard/re-base. *(New in Round 2.)*
- **R2** Fix docs/notebook to carry the **final curated** `merge_id`, not the root.
- **R3** Guard preview/unmerged curations at the shared output layer (and in `UnitMatch`).
- **R4** Fix sorter-output-freed-before-consumption (verify file-backed MS4 path first).

### SHOULD — during Phase 5 (cheap safety or in Phase-5 scope)
R5 (analyzer cache lock/atomic publish — mirror the recording fix) · R8 (schema-init version backfill + stale-default audit) · R12 (route scratch to configured temp) · R14 (direct `_assert_v2_db_safe` on the two modules) · R15 (narrow the eager v2 merge-table probe) · R16 (reconcile dependency pins; pin numpy) · R18 (DESTR-6 negative-days validation; DESTR-8 orphan-sweep name filter; DESTR-2 remove dead in-place branch) · R20 (orchestrator/preflight polish — esp. ERR-1) · R22 (DOWN-6 Export.File leak) · R23 (docs/build cuts + stale status text) · R26-partial (CONFIG-8 conda-export guard) · **R28** (deny `update1` on selection masters; guard `CurationV2`/`SessionGroup` direct writes) · **R30** (concat electrode-space + fs compatibility checks; UnitMatch 2D-geometry guard) · **R31** (re-validate concat+artifact and shared-artifact at the compute boundary) · **R32** (sorter-dispatch execution mismatches) · **R34**-SEC-2 (`restrict_permission=True`; drop local `0o777`) · **R36** (cheap test/single-source-of-truth wins before Phase 5).

### NICE — post-Phase-5
R10 (NWB provenance) · R11 (explicit-timestamp perf) · R13 (job_kwargs wiring) · R17 (import boundary) · R19 (interval-orphan finder) · R21 (error taxonomy base) · R24 (selection-helper consistency) · R25 (recompute input fingerprints) · **R35** (validation escape-hatches + payload-shape) · **R37** (stale scaffolding; EXT-8 mark plan superseded) · remainder of R18/R26/R34.

### DECIDE — product/design decision needed first
- **R9** Is `team_name` a tag or an enforced access boundary? (drives 9 findings)
- **R6** Accept row-name recording identity (already leaning won't-do) vs. version the runtime-semantics aliases (recommend doing the alias-version subset).
- **R7** Freeze curation/tracked-unit label snapshots vs. accept re-derivation.
- **R10** Is portable off-DataJoint NWB export on the near-term roadmap? (gates NWB work)
- **R33** How extensible should v2 actually be? (matcher/preset/sorter/viz plugin surfaces — currently advertised but closed; EXT-2 presets are Phase-5 scope)
- **R34**-policy Document the trusted-operator model for `SorterParameters`/filename inputs (SEC-1/3/6), or add allowlists/confinement.

### DONE — stale-fixed, close out
R-na: **REPRO-1/TEST-1** (raw source pinned to `Raw.raw_object_id`, 385e7d31) · **REPRO-2/TEST-2** (`channel_name` maps by electrode row, 63e0ff5a) · **CONC-3/OBS-1** + OBS-7-recording (recording rebuild per-recording lock + atomic publish + content-drift fail-closed, d08c319b/db8d271e/18f05527) · **SCHEMA-5/DOCS-4/API-1** (`cache_hash`→`content_hash` in docs + flipped reclamation tests, e8e06553).

---

## Notes on Phase 5 interaction

Phase 5 *is* the natural home for a large share of this list. When the executor
opens Phase 5 they should pull in: **R2, R3, R20, R23** (orchestrator + docs are
literally Phase-5 deliverables), **R33-EXT-2** (preset registration is a Phase-5
preset deliverable), and ideally **R1, R14, R16, R36** (schema/identity/dependency
hardening + cheap test/single-source-of-truth wins while touching presets and the
param layer). The Phase-5 plan explicitly forbids altering Phase 1–4 *table
definitions* — but the schema freeze is lifted pre-release and R1/R8/R28 require
new columns or guards, so confirm with the owner whether those changes are
in-scope for Phase 5 or a dedicated pre-Phase-5 hardening PR.

**Recommendation — two PRs before Phase 5 proper:**
1. **Correctness PR (MUST):** R29 signal/numerical bugs (TIME-1, SIG-2, SIG-1),
   R27 AnalyzerCuration namespace, R4 sorter-output-freed, R3 preview-as-final
   guard. These are silent-wrong-science and don't depend on Phase 5.
2. **Identity/provenance + cheap-hardening PR:** R1 reproducibility provenance,
   R28 master-row immutability, R5 analyzer cache lock, R30 concat compatibility,
   R32 dispatch mismatches, R34-SEC-2 permissions, R16 dependency pins, R36 tests.

Then Phase 5 absorbs the UX/docs/orchestrator cluster (R2, R20, R23, R33-presets).

## Appendix — full per-finding verdicts

Compact index for traceability. Sev = severity as claimed in the source review.

### datajoint-concurrency-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| CONC-1 | High | CONFIRMED | R5 | Duplicate-loser cleanup `rmtree`s the winner's canonical analyzer folder |
| CONC-2 | High | CONFIRMED | R5 | Per-sort analyzer lock held only by `AnalyzerCuration`; all other paths unguarded |
| CONC-3 | High | STALE-FIXED | DONE | Recording rebuild/recompute now share a per-recording lock + atomic publish |
| CONC-4 | Medium | CONFIRMED | R1-adj | `curation_id = max+1` check-then-insert race; expensive staging before insert |
| CONC-5 | Medium | CONFIRMED | — | `TrackedUnit.make` monolithic (graph derive inside txn) — known/accepted design |
| CONC-6 | Low | CONFIRMED | — | `AutoCurationRules.insert_rules` not race-idempotent |

### destructive-admin-operations-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| DESTR-1 | High | CONFIRMED | R19 | Cascaded/`super_delete` bypasses analyzer + interval cleanup |
| DESTR-2 | High | CONFIRMED* | R18 | In-place `write_nwb_artifact` unlinks canonical on failure (prod now passes None) |
| DESTR-3 | Med-high | CONFIRMED | R5 | Analyzer delete/rebuild not locked/atomic (= CONC-2) |
| DESTR-4 | Med-high | CONFIRMED | R19 | Artifact interval cleanup not resumable after partial delete |
| DESTR-5 | Medium | CONFIRMED | R18 | `force_stale_env=True` is a transient warning, not durable audit |
| DESTR-6 | Medium | CONFIRMED | R18 | Negative `days_since_creation` bypasses the age gate (no `>=0` check) |
| DESTR-7 | Medium | CONFIRMED | R18 | Bare path-list return; disk-space ignores env/age gate |
| DESTR-8 | Medium | CONFIRMED | R18 | Orphan sweep `rmtree`s any dir under root (no name-pattern filter) |
| DESTR-9 | Med-low | CONFIRMED | R18/R23 | Destructive sweep has no confirm/cancel test or runbook |
| DESTR-10 | Med-low | CONFIRMED | R23 | Sort-group preview docs name wrong call (raises, not returns preview) |
| DESTR-11 | Low-med | PARTIAL | R19 | `remove_matched` `delete_quick` narrowed (excludes matched=1) but `matched=0` child still blocks |

### operational-observability-recovery-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| OBS-1 | High | STALE-FIXED | DONE | Recording reclamation fetch-back now works (content-drift fail-closed) |
| OBS-2 | Med-high | CONFIRMED | R5 | Analyzer rebuild/delete bypass per-sort lock, write canonical directly |
| OBS-3 | Medium | CONFIRMED | R19 | Interval cleanup non-resumable (= DESTR-4) |
| OBS-4 | Medium | CONFIRMED | R20 | Failed receipts lose completed-stage timing/stage rows |
| OBS-5 | Medium | CONFIRMED | R20 | Preflight expected-ids don't distinguish selection vs computed |
| OBS-6 | Medium | CONFIRMED | R18 | Recompute deletion no-ops ambiguous (`[]` overloaded) |
| OBS-7 | Low-med | PARTIAL | R20 | Breadcrumbs thin; recording cache-miss log already added |

### scientific-reproducibility-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| REPRO-1 | High | STALE-FIXED | DONE | Raw source pinned to `Raw.raw_object_id` |
| REPRO-2 | High | STALE-FIXED | DONE | `channel_name` maps by electrode-table row |
| REPRO-3 | High | PARTIAL | R6 | `content_hash` is post-write, not recipe identity (open Qs decided won't-do) |
| REPRO-4 | High | CONFIRMED | R1 | Global/`dj.config` seed + versions not captured/rejected in identity |
| REPRO-5 | High | CONFIRMED | R1 | UnitMatch matcher defaults + bundle inputs omitted from identity |
| REPRO-6 | Med-high | CONFIRMED | R7 | Curation lineage + tracked-unit universe not frozen |
| REPRO-7 | Medium | CONFIRMED | R6 | Runtime aliases (preprocessing order, motion preset) need explicit versions |
| REPRO-8 | Medium | CONFIRMED | R1 | Analyzer `noise_levels` unseeded; analyzer-curation source not pinned |

### schema-evolution-migration-safety-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| SCHEMA-1 | High | CONFIRMED | R8 | Omitted outer schema version stored as current despite blob |
| SCHEMA-2 | High | CONFIRMED | R8 | Same-name stale shipped defaults survive seeding silently |
| SCHEMA-3 | Med-high | CONFIRMED | R8 | Interim row names block seeding; no upgrade/rename path |
| SCHEMA-4 | Med-high | CONFIRMED | R25 | v2 drift (new `content_hash`/concat fields) not snapshotted/gated |
| SCHEMA-5 | Medium | STALE-FIXED | DONE | Docs migrated `cache_hash`→`content_hash` |
| SCHEMA-6 | Medium | CONFIRMED | R6 | Preprocessing runtime order changed without a durable version boundary |
| SCHEMA-7 | Medium | CONFIRMED | R25 | Recompute inventories lack upstream param fingerprints |
| SCHEMA-8 | Medium | CONFIRMED | R6 | `"auto"`/dredge presets not independently versioned |
| SCHEMA-9 | Med-low | CONFIRMED | R24 | Duplicate/alias policy weaker for metric/auto-curation tables |
| SCHEMA-10 | Low | CONFIRMED | R24 | Concat selection ignores supplied deterministic ID (= API-4) |

### test-coverage-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| TEST-1 | High | STALE-FIXED | DONE | Raw-source-pin test now exists (two-ElectricalSeries fixture) |
| TEST-2 | High | STALE-FIXED | DONE | `channel_name` non-contiguous/shuffled-id tests now exist |
| TEST-3 | High | PARTIAL | R6 | Fingerprint/rebuild tests exist; interval-mutation→identity contract not covered |
| TEST-4 | High | CONFIRMED | R1 | No test pins effective seed / rejects ambient config / version provenance |
| TEST-5 | High | CONFIRMED | R1 | UnitMatch bundle inputs not identity/provenance tested |
| TEST-6 | Med-high | CONFIRMED | R7 | `TrackedUnit` universe-drift not tested at DB boundary |
| TEST-7 | Medium | CONFIRMED | R10 | Concat frozen-member + portable-NWB provenance untested |
| TEST-8 | Medium | CONFIRMED | R11 | Nonzero rate-based `starting_time` not covered on raw-NWB path |
| TEST-9 | Medium | CONFIRMED | R1 | `noise_levels`/extension-provenance manifest tests incomplete |
| TEST-10 | Medium | CONFIRMED | R7 | Curation parent-semantics + analyzer-curation source identity untested |

### dependency-runtime-upgrade-safety-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| DEP-1 | High | CONFIRMED | R1 | No per-output producer-runtime manifest beyond narrow SI/NWB |
| DEP-2 | High | CONFIRMED | R1 | UnitMatch defaults + package version not in stored identity |
| DEP-3 | Med-high | CONFIRMED | R16 | Docs/env/test don't enforce the `pyproject` SI pin |
| DEP-4 | Med-high | CONFIRMED | R16 | NumPy-2 baseline is a comment, not a dependency contract |
| DEP-5 | Medium | PARTIAL | R16 | Live SI registries drive metric validation (snapshot partially landed) |
| DEP-6 | Medium | CONFIRMED | R26 | Direct sorter execution doesn't reuse the preflight import gate |
| DEP-7 | Medium | CONFIRMED | R26 | Container recipes tag-pinned, not digest-pinned |
| DEP-8 | Medium | CONFIRMED | R26 | Optional KS4/UnitMatch scientific lanes skip-gated in CI |
| DEP-9 | Med-low | CONFIRMED | R23 | Stale "0.104+"/placeholder runtime docs |

### configuration-runtime-isolation-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| CONFIG-1 | High | CONFIRMED | R1 | Effective `job_kwargs` from SI globals/`dj.config`; only per-row layer in identity |
| CONFIG-2 | High | CONFIRMED | R26 | Directory config cached into module/class globals (shared Spyglass) |
| CONFIG-3 | High | CONFIRMED | R26 | Analyzer cache path not namespaced by DB/runtime context |
| CONFIG-4 | Med-high | CONFIRMED | R14 | v2 DB guard incomplete (missing on 2 modules) + import-time only |
| CONFIG-5 | Medium | CONFIRMED | R14 | DB guard undocumented; surprise import-time `RuntimeError` |
| CONFIG-6 | Medium | CONFIRMED | R12 | UnitMatch/recompute temp dirs bypass configured Spyglass temp |
| CONFIG-7 | Medium | CONFIRMED | R23/R26 | Storage roots underdocumented + not preflighted |
| CONFIG-8 | Medium | CONFIRMED | R26 | NWB write hard-depends on `conda env export` |
| CONFIG-9 | Medium | PARTIAL | R23 | Container execution config under-discovered (docs less bare than claimed) |
| CONFIG-10 | Med-low | CONFIRMED | R1 | Matcher registration is process-global mutable state; version not stored |

### import-time-lazy-boundaries-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| IMPORT-1 | High | CONFIRMED | R17 | v2 schema modules import `spyglass.common` root (runs prepopulate) |
| IMPORT-2 | High | CONFIRMED | R15 | `SpikeSortingOutput` eagerly probes v2; one failure disables v2 part |
| IMPORT-3 | Med-high | CONFIRMED | R17 | `v2.utils` heavy barrel drags DataJoint+SI into pure helpers |
| IMPORT-4 | Medium | CONFIRMED | R17 | `common_nwbfile` pulls NWB/SI stack at import to declare FK |
| IMPORT-5 | Medium | CONFIRMED | R17 | pandas/pydantic required-but-unpinned; boundary tests don't pin dep surface |
| IMPORT-6 | Med-low | CONFIRMED | R14 | `metric_curation`/`recompute` rely on transitive DB-guard (= CONFIG-4) |
| IMPORT-7 | Med-low | CONFIRMED | R23 | Docs/notebooks describe stale UnitMatch import behavior |

### downstream-consumer-contracts-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| DOWN-1 | High | CONFIRMED | R3 | Preview curations consumable as final via generic accessors |
| DOWN-2 | High | CONFIRMED | R3 | UnitMatch accepts preview curations / matches unmerged units |
| DOWN-3 | Med-high | CONFIRMED | R22 | All-unlabeled curations break `include_labels` filtering |
| DOWN-4 | Medium | CONFIRMED | R22 | `get_pairs()` loses schema for zero-pair runs |
| DOWN-5 | Medium | CONFIRMED | R22 | `get_unit_brain_regions()` drops curation/region disambiguators |
| DOWN-6 | Medium | CONFIRMED | R22 | Duplicated `delete_quick` leaves stale `Export.File` rows (clear bug) |
| DOWN-7 | Med-low | CONFIRMED | R10 | UnitMatch NWB pairs not self-describing |
| DOWN-8 | Low | CONFIRMED | R23 | Public docs misstate v2 DataFrame shape (index vs column) |

### user-api-ergonomics-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| API-1 | High | STALE-FIXED | DONE | Recording-cache recovery docs/tests now consistent |
| API-2 | High | CONFIRMED | R2 | Docs steer to root `merge_id`, not final curated output |
| API-3 | Med-high | CONFIRMED | R23 | Availability/status docs contradict shipped surfaces |
| API-4 | Med-high | CONFIRMED | R24 | `ConcatenatedRecordingSelection.insert_selection` ignores supplied PK |
| API-5 | Med-high | CONFIRMED | R23 | Setup prerequisites far from the runnable install path |
| API-6 | Medium | CONFIRMED | R21 | Advanced insert helpers skip `_ensure_lookup_row_exists` pre-checks |
| API-7 | Medium | CONFIRMED | R23 | Quickstart advertises preflight; main snippet skips it |
| API-8 | Medium | CONFIRMED | R23 | Storage-management page orphaned from docs nav (= DOCS-2) |
| API-9 | Low-med | CONFIRMED | R24 | `QualityMetricParameters` implicit duplicate-content policy |
| API-10 | Low | CONFIRMED | R24 | Selection-helper shape/extra-field handling uneven |

### docs-build-link-integrity-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| DOCS-1 | High | CONFIRMED | R2 | Markdown downstream examples point at root curation merge id (= API-2) |
| DOCS-2 | Med-high | CONFIRMED | R23 | Storage page orphaned from nav/feature index (= API-8) |
| DOCS-3 | Med-high | CONFIRMED | R23 | Status/migration prose calls shipped UnitMatch/concat roadmap/placeholder |
| DOCS-4 | Medium | STALE-FIXED | DONE | Stale `cache_hash` terminology removed from user docs |
| DOCS-5 | Medium | CONFIRMED | R23 | Clean-checkout MkDocs instructions broken (`mkdosc.yaml` typo) |
| DOCS-6 | Medium | CONFIRMED | R23 | Generic `Export.md` example copy-paste broken (`NameError`/bad kwarg) |
| DOCS-7 | Med-low | CONFIRMED | R23 | UnitMatch install snippet checkout-only, omits base v2 extra |
| DOCS-8 | Low | CONFIRMED | R23 | Build version fallback uses unset `version_string` |

### error-taxonomy-actionability-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| ERR-1 | High | CONFIRMED | R20 | Preflight misses required `"raw data valid times"` interval |
| ERR-2 | High | CONFIRMED* | R20 | `SessionGroup.create_group()` leaks `KeyError`/raw FK errors (partially hardened) |
| ERR-3 | Medium | CONFIRMED | R21 | Direct workflows lack the pipeline stage wrapper |
| ERR-4 | Medium | CONFIRMED | R21 | Malformed public input dicts leak raw Python errors |
| ERR-5 | Medium | CONFIRMED | R21 | Many named exceptions, no common public base |
| ERR-6 | Med-low | CONFIRMED | R21 | `preflight=False` second error world (by design) |
| ERR-7 | Med-low | CONFIRMED | R20 | Batch failure results drop structured stage/cause fields |

### nwb-portability-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| NWB-1 | High | CONFIRMED | R10 | Recording NWBs lack raw-source + construction provenance |
| NWB-2 | High | CONFIRMED | R10 | Concat NWBs can't be split/mapped back without DataJoint |
| NWB-3 | High | CONFIRMED | R10 | Motion-corrected concat artifacts have lossy motion provenance |
| NWB-4 | High | CONFIRMED | R10 | Sorting Units NWBs lack source/sorter/artifact/runtime context |
| NWB-5 | High | CONFIRMED | R10 | Unit-level peak/electrode/region metadata DB-only |
| NWB-6 | High | CONFIRMED | R10 | Curated Units NWBs don't export merge lineage |
| NWB-7 | Medium | CONFIRMED | R10 | AnalyzerCuration NWBs result-only |
| NWB-8 | Medium | CONFIRMED | R10 | UnitMatch NWB pairs not self-describing |
| NWB-9 | Medium | PARTIAL | R10 | `content_hash` on row, verified only on rebuild (renamed from `cache_hash`) |
| NWB-10 | Low-med | CONFIRMED | R10 | Processed recording written under `acquisition`, not a processing module |

### performance-memory-scaling-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| PERF-1 | High | CONFIRMED | R11 | Processed NWB timestamps force O(n_samples) storage/readback |
| PERF-2 | High | CONFIRMED | R11 | Explicit/irregular raw timestamp restriction takes eager full-vector path |
| PERF-3 | High | CONFIRMED | R4 | Sorter output temp dir cleaned before returned (file-backed) sorting consumed |
| PERF-4 | Med-high | CONFIRMED | R13 | Recording write resolves `job_kwargs` then discards them |
| PERF-5 | Med-high | CONFIRMED | R12 | Recompute hashing materializes multi-GB arrays + copies |
| PERF-6 | Medium | CONFIRMED | R12 | Analyzer recompute writes large temp analyzers to system `/tmp` |
| PERF-7 | Medium | CONFIRMED | R12/R13 | Analyzer curation doubles scratch + holds per-sort lock across heavy compute |
| PERF-8 | Medium | CONFIRMED | R12 | UnitMatch dense bundle extraction no RAM/disk budget, system `/tmp` |
| PERF-9 | Medium | CONFIRMED | R11 | Concat motion + split back-mapping scale poorly (per-pair mask loop) |

### multi-user-team-ownership-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| TEAM-1 | High | CONFIRMED | R9 | Sort-group overwrite session-global; cascades through every team's downstream |
| TEAM-2 | High | CONFIRMED | R9 | Delete checks session experimenters, not v2 owner teams |
| TEAM-3 | High | CONFIRMED | R9 | `team_name` is a namespace, not an access boundary |
| TEAM-4 | Med-high | CONFIRMED | R9 | Recompute deletion admin-facing but no admin/team gate |
| TEAM-5 | Medium | CONFIRMED | R9 | Mixed-team session groups representable but not authorized |
| TEAM-6 | Medium | CONFIRMED | R9 | `SharedArtifactGroup` globally named and ownerless |
| TEAM-7 | Medium | CONFIRMED | R9 | Paper export global by `paper_id`, no owner |
| TEAM-8 | Med-low | CONFIRMED | R9 | `LabTeam` creation not gated by admin controls |
| TEAM-9 | Low | CONFIRMED | R9/R23 | Docs blur availability vs ownership semantics |

\* CONFIRMED with nuance — see the source review and the cluster note for the residual surface.

---

## Appendix (Round 2) — full per-finding verdicts

10 reviews added after the first triage. All 77 findings CONFIRMED except SEC-5
and SER-7 (PARTIAL); none stale-fixed or false. "Root" maps each to a root issue
(R1–R37) or an existing Round-1 finding ID.

### curation-label-lifecycle-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| CLIFE-1 | High | CONFIRMED | R27 | AnalyzerCuration computes over raw-sort units, ignores the selected curation's merged/labeled set |
| CLIFE-2 | High | CONFIRMED | R7 | Child curations are raw-sort snapshots; parent labels/merges not inherited |
| CLIFE-3 | High | CONFIRMED | R28 | Post-insert `UnitLabel` edits desync DB labels from the already-written Units NWB |
| CLIFE-4 | Med-high | CONFIRMED | R22 | All-unlabeled curations omit the label column → `include_labels` filtering skipped (= DOWN-3) |
| CLIFE-5 | Medium | CONFIRMED | R28 | UnitMatch can persist FK-valid but non-matchable units (Pair FK to all `CurationV2.Unit`) |
| CLIFE-6 | Medium | CONFIRMED | R7 | TrackedUnit docs say "curated" but the universe is matchable-curated (excludes reject/noise) |
| CLIFE-7 | Med-low | CONFIRMED | CONC-4 | `curation_id = max+1` race; stages NWB before detecting the conflict |
| CLIFE-8 | Low | CONFIRMED | R22 | Export/split label semantics need clearer user docs |

### relational-query-integrity-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| REL-1 | High | CONFIRMED | R28 | Deterministic selection IDs retargetable via `update1` (guard covers `insert` only) |
| REL-2 | High | CONFIRMED | R28 | `SessionGroup.Member` is live mutable provenance for materialized concat/UnitMatch (no guard) |
| REL-3 | High | CONFIRMED | R28 | `CurationV2` direct writes bypass parts + merge registration (master has no guard) |
| REL-4 | Med-high | CONFIRMED | R31 | Concat-backed curations underselected by a `preprocessing_params_name` restriction |
| REL-5 | Medium | CONFIRMED | R28 | Source-part exclusivity is read-time only, not audited or find-existing safe |
| REL-6 | Medium | CONFIRMED | R28 | `UnitMatch.Pair` FK doesn't constrain rows to the selected curation universe |
| REL-7 | Med-low | CONFIRMED | R7 | `TrackedUnit` re-derives the matchable universe from current labels (= REPRO-6) |

### extension-customization-contracts-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| EXT-1 | High | CONFIRMED | R33 | Matcher plugins not backend-agnostic — hard UnitMatchPy bundle dependency in the table path |
| EXT-2 | High | CONFIRMED | R33 | Custom pipeline presets are a private static dict, no `register_pipeline_preset` (Phase-5 scope) |
| EXT-3 | Med-high | CONFIRMED | R33 | Custom waveform rows validated, but sort-time recipe selection is not first-class |
| EXT-4 | Medium | CONFIRMED | R33 | Matcher registry `replace=True` can re-route existing rows; no backend provenance stored |
| EXT-5 | Medium | CONFIRMED | R33 | Custom sorter support is a generic SI escape hatch, not a sorter plugin API |
| EXT-6 | Medium | CONFIRMED | R33 | Custom auto-curation rule labels pass insertion then fail late at materialize |
| EXT-7 | Low-med | CONFIRMED | R33 | Visualization discovery is a closed static registry |
| EXT-8 | Low-med | CONFIRMED | R37 | Old phase-0 fixture-generation plan conflicts with the current ground-truth fixture contract |

### signal-units-preprocessing-semantics-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| SIG-1 | Med-high | CONFIRMED | R29 | `no_filter` + reference reattaches the original DC offset (re-referencing cancelled it) |
| SIG-2 | Med-high | CONFIRMED | R29 | SNR hard-codes `peak_sign="neg"` while attribution honors sorter polarity |
| SIG-3 | Medium | CONFIRMED | R10 | Units NWB omits per-unit signal metadata the API/docs advertise (= NWB-5) |
| SIG-4 | Medium | CONFIRMED | R30 | Non-planar 3D geometry silently projected to 2D (warn only) |
| SIG-5 | Medium | CONFIRMED | R6 | Preprocessing order changed without a schema/identity bump (= SCHEMA-6/REPRO-7) |
| SIG-6 | Med-low | CONFIRMED | (test) | Peak-amplitude integration test not discriminating for off-center extrema |
| SIG-7 | Low | CONFIRMED | R37 | Clusterless smoke row named `5uv` but behavior is MAD |

### motion-drift-spatial-registration-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| MOTION-1 | High | CONFIRMED | R30 | Concat compatibility equates different physical electrode spaces (ids+coords match, electrodes differ) |
| MOTION-2 | High/med | CONFIRMED | R30 | UnitMatch can receive 3D channel positions despite a 2D matcher contract |
| MOTION-3 | Med-high | CONFIRMED | R6 | Concat split reads live `SessionGroup.Member` after materialization (= NWB-2 family) |
| MOTION-4 | Medium | CONFIRMED | R10 | Applied concat motion discards the displacement field + coordinate-frame context (= NWB-3) |
| MOTION-5 | Medium | CONFIRMED | R6 | UnitMatch chronological drift order is not part of selection identity |
| MOTION-6 | Medium | CONFIRMED | R6 | `DriftEstimate` keyed only by `Recording` despite an algorithm-dependent estimate (= SCHEMA-8) |
| MOTION-7 | Med-low | CONFIRMED | R30 | Spatial semantics under-documented in UnitMatch / unit-location APIs |

### timebase-sample-index-alignment-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| TIME-1 | High | CONFIRMED | R29 | Rate-based interval restriction loses boundary samples (unsnapped float `ceil`/`floor`) |
| TIME-2 | Medium | CONFIRMED | R30 | Concat sampling-frequency compatibility is implicit (not checked) |
| TIME-3 | Medium | CONFIRMED | R31 | Bypassed concat+artifact rows sort unmasked while claiming artifact metadata |
| TIME-4 | Med-low | CONFIRMED | R31 | Shared artifact detection trusts insert-time time-axis checks at compute |
| TIME-5 | Low | CONFIRMED | R11 | Concat duration vs endpoint obs-interval conventions differ by one sample |
| TIME-6 | Low | CONFIRMED | (edge) | Manual/legacy `valid_times` can hide a one-frame seam artifact |

### sorter-dispatch-execution-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| DISP-1 | Med-high | CONFIRMED | R32 | Clusterless rows claim container execution preflight enforces but runtime ignores |
| DISP-2 | Medium | CONFIRMED | R32 | `delete_container_files=False` ineffective under Spyglass temp cleanup (= R12) |
| DISP-3 | Medium | CONFIRMED | R32 | `insert_default_legacy_si_sorters` creates local MATLAB rows the dispatch rejects |
| DISP-4 | Med-low | CONFIRMED | R32 | External whitening interception broader than the curated contract (no allowlist) |
| DISP-5 | Med-low | CONFIRMED | R6 | Sort output has no DB-visible content fingerprint (by design; content-addressed by inputs) |
| DISP-6 | Med-low | CONFIRMED | R28 | `allow_param_mutation=True` `update1` bypasses insert validation |
| DISP-7 | Low-med | CONFIRMED | R23 | Custom container row creation tested but not copyably documented |
| DISP-8 | Low | CONFIRMED | R26 | KS4 algorithm-default drift unprotected unless KS4 installed (= DEP-8) |
| DISP-9 | Low | CONFIRMED | R23 | Two docs strings disagree with execution semantics (MS5 tetrode-vs-probe; external whiten) |

### security-external-boundary-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| SEC-1 | High | CONFIRMED | R34 | NWB filenames can escape configured raw/analysis roots (no basename/confinement check) |
| SEC-2 | High (shared) | CONFIRMED | R34 | Artifacts written `0o666`; sorter scratch `0o777` even on local runs |
| SEC-3 | High (if not admin-only) | CONFIRMED | R34 | Sorter/container param rows are a code-execution boundary with no allowlist/policy |
| SEC-4 | Med-high | CONFIRMED | R26 | "Pinned" container docs are tag-based, not digest-immutable (= DEP-7) |
| SEC-5 | Medium | PARTIAL | R18 | Cache path/deletion helpers rely on caller discipline (not reachable via normal API) |
| SEC-6 | Medium | CONFIRMED | R34 | Shared file helpers use formatted DataJoint `LIKE`/`=` string restrictions |
| SEC-7 | Medium | CONFIRMED | R23 | Operator docs don't surface the DB/deletion safety boundaries early enough |
| SEC-8 | Low | CONFIRMED | R18 | One recompute cleanup path masks `rmtree` error detail |

### serialization-payload-compatibility-review.md
| ID | Sev | Verdict | Root | One-line |
|---|---|---|---|---|
| SER-1 | High | CONFIRMED | R8 | `schema_version` is a loose tag, not a supported-version contract (= SCHEMA-1) |
| SER-2 | Med-high | CONFIRMED | R35 | JSON-native/finite is a fingerprint side effect, not a first-class insert invariant |
| SER-3 | Medium | CONFIRMED | R24 | Duplicate-content behavior differs from the documented contract (metric/rule tables exempt) |
| SER-4 | Medium | CONFIRMED | R22 | Empty UnitMatch pairs lose the DataFrame schema (= DOWN-4) |
| SER-5 | Med-low | CONFIRMED | R35 | NWB curation-label payload shapes undocumented; scalar value vs `["uncurated"]` docstring wrong |
| SER-6 | Med-low | CONFIRMED | R35 | Artifact interval return shape flips array↔dict by source path |
| SER-7 | Low | PARTIAL | R23 | Public docs understate exported payload schema / overstate setup coverage (not line-verified) |

### maintainability-module-boundary-review.md
| ID | Sev | Verdict | Root | One-line (+actionability) |
|---|---|---|---|---|
| MAINT-1 | High | CONFIRMED | R24 | Source-routing centralized but monolithic — LOW-VALUE; extraction adds indirection to a stable surface |
| MAINT-2 | High | CONFIRMED | — | Public table classes mix responsibilities — ALREADY-REJECTED (file-split campaign over); keep the rule for new code |
| MAINT-3 | Medium | CONFIRMED | R36 | Tri-part contract tests cover 3 of ~7 carriers — ACTIONABLE (extend tests; NamedTuple choice itself is settled) |
| MAINT-4 | Medium | CONFIRMED | R36 | 4 parallel stage/lookup catalogs, one live drift (notebook 3 vs 8) — ACTIONABLE (consistency tests + doc fix) |
| MAINT-5 | Medium | CONFIRMED | R36 | Slow integration setup hand-rolled in 4 test modules — ACTIONABLE (`populate_single_session_chain` helper) |
| MAINT-6 | Medium | CONFIRMED | — | Fixture-fetch detection source-text coupled — LOW-VALUE/defer (machinery works today) |
| MAINT-7 | Med-low | CONFIRMED | — | Private/public boundaries blurred by compat wrappers — LOW-VALUE (docs hygiene, no bug) |
| MAINT-8 | Low | CONFIRMED | R24 | Lookup insert patterns duplicated — ACTIONABLE (hoist into existing `ImmutableParamsLookup`) |
| MAINT-9 | Low | CONFIRMED | R23 | Docs/facade status text lags shipped surface (self-contradicting UnitMatch status) — PHASE-5 |
| MAINT-10 | Low | CONFIRMED | R23 | Canonical notebook is both beginner path + advanced reference — PHASE-5 design call |
