# Spike Sorting v2 — Architecture Evaluation (research notes)

**Author:** Claude (architecture review session)
**Date:** 2026-06-10
**Status:** COMPLETE. Deep-dive verified on the 4 boundary concerns; REVIEW-REPORT.md triaged
and folded in here (then deleted as superseded).
**Scope:** Architecture (layering, cohesion, coupling, contracts) of `src/spyglass/spikesorting/v2/`.
Phases 0–1b landed; 2–5 forward-declared in final-shape schema with gated bodies.
**Method:** Hypothesis-driven; parallel evidence agents + direct reads + direct grep/code
verification of every load-bearing claim. Confidence levels explicit.

---

## TL;DR verdict

The v2 single-session core is a **genuine, well-layered modernization** — clean separation
(schema / SI-runtime helpers / Pydantic params / orchestration / exceptions), tri-part parallel
populate that keeps SI work out of DB transactions, a fully-wired chunked NWB write path, real
per-unit brain-region tracing, and a sound fail-loud test/fixture architecture. The ~2× LOC vs v1
is mostly real capability + lifecycle defense, **not bloat or diluted cohesion**.

**After deep verification, the boundary risks are smaller than first thought:**
- **C1 export-safety — RETRACTED (non-issue).** Export captures v2 files via the AnalysisNwbfile
  FK-graph cascade, not accessor `fetch_nwb`; pinned by a passing test.
- **C2 merge-restriction coupling — real, but maintainability-only** (correctness now tested/fixed).
  Worth refactoring *before* Phase 3 concat grows the surface.
- **C3 preview-vs-apply merge-id divergence — confirmed live; small, fixable.** The one genuine
  correctness/UX seam in the landed core.
- **C4 provenance integrity — downgraded.** Inherited from v1, v2 is stricter, nothing dereferences
  it, self-FK is policy-forbidden. Document the contract; optional cheap `insert1` guard.

**Meta:** REVIEW-REPORT.md (2026-06-02) was substantially stale — nearly every HIGH finding and the
whole export-completeness/legacy-CI gap is now FIXED with dedicated tests. Triage table in §5.

---

## 0b. Implementation status (changes landed this session)

Five items were implemented on the `spikesorting-v2` branch and verified against the
Colima-backed v2 test DB (101 targeted tests green, black-clean at line-length 80):

| Item | Change | Files | Verification |
|---|---|---|---|
| **C3** (correctness) | Canonicalize fresh merged-unit id assignment to ascending **min-contributor** order on the applied path so it matches the lazy `get_merged_sorting` preview — guarantees preview==apply ids for reordered group input. TDD: new failing test → 1-line fix (`sorted(normalized_groups, key=min)`) → green; superseded `..._user_input_order` test updated to `..._canonical_min_order`. | `v2/curation.py` (`_build_curated_unit_rows` + comments/docstrings); `test_single_session_pipeline.py` | 7 green incl. e2e `test_lazy_vs_applied_merge_frames_equal`, `..._insert_with_merge_groups_apply_merges`, `..._keeps_cross_gap_pair` |
| **C2** (coupling) | Extracted the v2 source-part join-walk into `CurationV2.resolve_restriction` (single owner of v2's join topology); `SpikeSortingOutput._get_restricted_merge_ids_v2` is now a thin wrapper that only maps resolved curations → merge_ids + warns on fan-out. Removed now-unused `import uuid` from the merge module. | `v2/curation.py` (+`resolve_restriction`); `spikesorting_merge.py` | 13 green (`test_merge_id_artifact_resolution`, `test_preview_merge_warning`, resolves-through-chain, restrict-by-artifact) |
| **get_analyzer** | Decided the zero-unit raise-vs-degrade asymmetry is **correct** (a zero-unit sorting exists/empty; a zero-unit analyzer cannot be built by SI). Kept the typed raise; documented it as an intentional contract and added a non-exception precheck affordance (`n_units > 0`). | `v2/sorting.py` (`get_analyzer` docstring) | doc-only; covered by existing zero-unit suite |
| **C6** (cohesion) | Moved the 5 pure signal/frame/interval-math functions (327 LOC) out of `utils.py` into a focused `_signal_math.py`, **re-exported** from `utils` so all ~11 importers (incl. 7 test modules) keep working unchanged. | new `v2/_signal_math.py`; `v2/utils.py` (re-export) | 11 green (`test_consolidate_intervals`, `test_merge_dedup`) + import smoke + 70-test scaffold/params guard |
| **doc nuance** | Corrected `_nwb_iterators.py` module docstring overclaim ("collapses that indirection" → narrows the *public call surface*; the `_TimestampsExtractor` shim is retained internally). | `v2/_nwb_iterators.py` | doc-only |

**Deferred (with rationale), not done this session:**
- **C4** (provenance guards): leave the lineage shape as-is (inherited from v1, v2 already stricter,
  nothing downstream dereferences it, self-FK is zero-migration-forbidden). Optional cheap
  `CurationV2.insert1` guard mirroring `UnitLabel` is a future nicety, not done.
- **C6 remainder**: the `utils.py` parameter-Lookup validation cluster split and relocating
  `Sorting.find_orphaned_analyzer_folders` were **not** done — the former is more entangled
  (pydantic/dj) for less gain than the signal-math win already landed; the latter is a `@classmethod`
  on a table whose relocation is an API change (caller/test churn) for a maintenance helper. The
  signal-math split establishes the re-export pattern if the param cluster is split later.
- **C5** (SI version tripwire), **A8/A9/D7** (carry-forwards): unchanged this session (A8 is being
  fixed elsewhere per the owner).

---

## 1. Hypothesis tree — RESOLVED with confidence

| Hypothesis | Prior → Posterior | Outcome |
|---|---|---|
| **H1 — Clean modernization** | 30% → **78%** | Best supported, and *strengthened* by the deep-dive: the export design (FK cascade, not per-fetch logging) is elegant, and the merge `artifact_detection_id` resolution is correct + tested. Layering, tri-part populate, Pydantic params, brain-region tracing all real. |
| **H2 — God-files / diluted cohesion** | 35% → **25%** | Mostly refuted. Size is justified; only maintainability smells survive (C6). |
| **H3 — Plan/impl divergence & seams** | 25% → **12%** | Refuted. Stubs fenced; code is *ahead* of the planning docs (the review report lagged the code by ~5 days). |
| **H4 — Leaky coupling** | 35% → **40%** | Narrowed. SI coupling well-localized; export is NOT a leak (FK cascade). The only genuine leaks are C2 (merge-restriction resolver carries v2 schema knowledge) and C3 (curation merge-id ordering) — localized, not pervasive. |

Net: **a clean, idiomatic core with two localized boundary seams (C2 maintainability, C3 a small
correctness fix).** Not a structurally risky architecture.

---

## 2. Confirmed strengths (architecturally notable)
1. Clean layer separation; SI runtime isolated in named `@staticmethod` helpers, not inlined in `make()`.
2. Tri-part `make_fetch/compute/insert` + `_parallel_make` real on all Computed tables — heavy SI
   work runs outside the DB transaction with cleanup-on-failure (fixes v1 transaction-blocking).
3. Chunked NWB write path fully wired — no trace-array materialization / OOM on chronic data.
4. Pydantic param layer: per-Lookup schemas, documented `forbid`/`allow` policy, `schema_version`
   history, scientific params separated from `job_kwargs` (3-tier precedence). Bulk `insert` now also
   validates (was a gap in the review report; now fixed).
5. Per-unit brain-region persistence at sort time (real #1394 fix; walks every electrode).
6. Zero-migration forward-declaration executed cleanly; unbuilt phases fenced at 3 layers; shim
   modules imported by nobody. No half-activated abstractions.
7. `insert_curation`: idempotent, atomic, NWB-stage-then-DB-add-in-txn with rollback cleanup; lazy
   merge import breaks the curation↔merge cycle; label validation on *all* `UnitLabel` insert paths.
8. **Export design is sound** (verified): every NWB-producing table carries `-> AnalysisNwbfile`;
   `Export.populate_paper` harvests files by upward FK cascade — robust to accessor read style.
9. Exception design: every named invariant → dedicated class with a "next action" message.
10. Test/fixture architecture: behavior-organized, MEArec ground-truth + Box fetch + sha256
    fail-loud gating, parity/baseline/audit triad, no `test_mode` behavior branches in shipped code.

---

## 3. Boundary concerns — DEEP-DIVE RESULTS

### C1 [RETRACTED — non-issue] "v2 has no `fetch_nwb` → export gap"
**Verdict: not a gap, not a regression.** Verified mechanism:
- v1 reads NWB the *same* raw way — `get_sorting`/`get_recording`/`get_merged_sorting` use
  `AnalysisNwbfile.get_abs_path` + pynwb, never `fetch_nwb` (v1/curation.py:197,245; v1/recording.py:412;
  v1/sorting.py:499). v2 faithfully inherited this.
- **Export captures spike-sorting files via the AnalysisNwbfile FK-graph cascade**, not per-accessor
  logging. `Export.make` unions logged `ExportSelection.File` with `restr_graph.file_paths`
  (common_usage.py:556-560); `cascade_files` (dj_graph.py:1169-1224) harvests `analysis_file_name`
  from *every cascaded table that is an `AnalysisNwbfile` child* (dj_graph.py:699-711). All three v2
  tables (`Recording`, `Sorting`, `CurationV2`) carry the `-> AnalysisNwbfile` FK, so the upward
  cascade from any logged downstream leaf captures their files regardless of read style.
- **Pinned by a passing test:** `tests/spikesorting/v2/test_export_safety.py` asserts a two-sided
  invariant — the recording cache is ABSENT from selection-stage `ExportSelection.File` but PRESENT
  in final `Export.File` (captured by the cascade). Zero-unit export path also tested.
- Only nuance: capture happens at *populate* stage (cascade), not *selection* stage (fetch logging) —
  a non-difference in the export output, identical to v1.
- *Action: none.* (If selection-stage parity were ever wanted, route accessors through `fetch_nwb` —
  but that's a new behavior for v1+v2, not a v2 fix.)

### C2 [Med — real, maintainability/evolvability] merge-restriction resolver carries v2 schema knowledge
**Verified live and real, but NOT a correctness bug** (the acute `artifact_detection_id`-drop bug A1 is fixed).
`SpikeSortingOutput._get_restricted_merge_ids_v2` (spikesorting_merge.py:137-299) hard-codes:
- the rec/sort/curation key partitioning (`rec_keys`/`sort_keys`/`curation_keys`, :177-191),
- that `artifact_detection_id` lives on the optional `SortingSelection.ArtifactDetectionSource` part and must be
  resolved by part-intersection/anti-join (:245-268) — explicitly working around DataJoint's
  silent dict-restriction drop (the comment at :245-254 documents the footgun),
- the `artifact_detection_{uuid}` interval-name convention (:206-230),
- the multi-curation fan-out semantics (:282-299, warn-only).

This is ~160 lines of **v2-internal schema topology living in the merge master**. Correctness is now
tested (`test_merge_id_artifact_resolution.py`), but the coupling will *grow* when Phase 3 adds
`ConcatenatedRecordingSource` to this same logic, and every reader of the merge restriction depends
on the merge module re-implementing v2 part semantics correctly.
- **Residual sharp edge:** the multi-curation fan-out is warn-only — a caller that ignores the
  warning and feeds all returned merge_ids into `SortedSpikesGroup`/decoding double-counts units.
- *Fix direction (refactor, do before Phase 3 concat):* move the join-walk into a v2-owned resolver
  (e.g. `v2.utils.resolve_merge_restriction(key)` or a `CurationV2` classmethod) and have
  `SpikeSortingOutput` delegate, so v2 owns its own source-part topology and concat extends it in one
  place. Consider promoting the multi-curation fan-out from warn to an opt-in flag.

### C3 [Med — confirmed live; the one genuine correctness/UX seam] preview-vs-apply merge-id divergence
**Verified in current code.** Two paths assign fresh merged-unit ids in *different orders*:
- **Applied** (`apply_merge=True`, `_build_curated_unit_rows`, curation.py:791-799): iterates merge
  groups in **user-provided order**, assigning `max(unit_ids)+1, +2, …` (v1 parity, :730-739).
- **Lazy preview** (`get_merged_sorting` on an `apply_merge=False` curation, curation.py:1330-1377):
  `get_merge_groups` fetches `order_by=("unit_id",…)` so groups come in **kept-uid-ascending** order
  (kept-uid = `min(group)` for a preview, :798), then assigns `max+1, +2, …` in that order.

When user-provided group order ≠ ascending-min order, the two paths assign the **same fresh id to
different content groups**. Worked example: units 0–5, `merge_groups=[[4,5],[0,1]]`. Applied →
id6={4,5}, id7={0,1}. Lazy preview → id6={0,1}, id7={4,5}. Original passthrough units keep their ids
on both paths, so the blast radius is *only the fresh merged ids*. It bites the **preview-then-apply**
workflow (label "merged unit 6" in a preview, apply, "unit 6" is now different content) and any
cross-path comparison keyed on merged unit_id. Documented honestly in-code (:728-739) but unresolved.

- **Recommended fix:** canonicalize fresh-id assignment to **one content-derived order on both
  paths** — sort merge groups by `min(contributors)` (≡ the lazy path's order) before numbering, in
  the applied path's `_assign_merge_keys` (curation.py:791-799). Merged ids stay `max+1, max+2, …`
  (v1-shaped); the only change is *which* group gets `max+1` when the user passes them out of
  min-order — which is not a scientifically meaningful parity property (merged ids are arbitrary
  labels, not spike content). Add a test asserting `get_merged_sorting(preview)` and the
  `apply_merge=True` curation yield identical {id → spike-train} maps for a reordered-groups input.
  Document that merged-unit integer ids are content-canonical, not input-order-stable.
- *Cost/caveat:* check for any existing test pinning the user-order applied assignment; update its
  rationale (the v1-parity goal it cites is about content/count, not merged-id ordering).

### C4 [Low — downgraded; inherited + policy-frozen] provenance integrity enforced on helper path only
**Verified inherited and low-risk.**
- v1 is **identical**: `CurationV1.parent_curation_id=-1: int`, plain int, no self-FK, validated only
  in `insert_curation` (v1/curation.py:36,86-93). v0 too. So this is inherited, not introduced.
- **v2 is already stricter than v1**: `insert_curation` adds a parent-existence check
  (curation.py:365-378) that v1 lacks.
- **Nothing downstream dereferences the pointer.** Every read of `parent_curation_id` across v0/v1/v2
  is a `== -1` / `!= -1` root test (e.g. pipeline.py:255); no code fetches a non-`-1` parent's row.
  `MergeGroup.contributor_unit_id` is referenced nowhere outside curation.py. ⇒ a dangling pointer is
  cosmetic, never a wrong scientific result.
- **A self-FK is forbidden** by the zero-migration policy (changing an existing column to an FK is an
  ALTER), unprecedented in the codebase (no self-referential FK exists), brittle across the composite
  `(sorting_id, curation_id)` PK, and the *same* nullable-FK design was explicitly rejected on
  `SortingSelection` (sorting.py:465-475, "a nullable FK conflates 'no artifact' with 'match
  anything'").
- One real low-severity gap: deleting a parent silently orphans children (no cascade, because no FK).
- **Recommendation:** (d) document the contract [primary] ≈ (b) optional cheap `insert1`/`insert`
  override mirroring the proven `UnitLabel` pattern (curation.py:175-191) to re-run the parent-exists
  check on direct inserts (~15 LOC, no migration). Skip the self-FK (a) and the `contributor_unit_id`
  FK (c). Optionally add an additive `prune_orphaned_curations` helper (mirrors
  `prune_orphaned_selections`) to surface lineage orphans on demand.

### C5 [Low-Med] no in-code SI version tripwire (unchanged)
v2 targets SI 0.104 with no version assertion; a future SI API change breaks at populate time, not
import. Relies on `cache_hash` drift detection. *Fix:* a lightweight `__init__`-time SI version-floor
check converts a deep runtime failure into a clear import-time message.

### C6 [Low] maintainability smells, not defects (unchanged)
- `sorting.py` (2457) carries disk-audit/orphan tooling (`find_orphaned_analyzer_folders`) — split candidate.
- `test_single_session_pipeline.py` (11.5k lines) — single merge-conflict surface.
- `utils.py` (1116) mixes ~6 concerns + parks domain helpers (`_dedup_merged_spike_times`,
  `unit_brain_region_df`, the timestamp/frame signal-math trio); a `_lookup_validation.py` /
  `_signal_math.py` split would sharpen cohesion.

---

## 4. Self-critique log (updated)
- **I overweighted C1 and C4 in the first pass** — both were inherited from the (now-verified) stale
  REVIEW-REPORT's framing. The deep-dive *verified v1 behavior first* (as instructed) and corrected
  both: export is a non-issue (FK cascade), provenance is inherited + policy-frozen. Lesson: when a
  concern is "v2 didn't modernize X," check what v1 actually does AND what the surrounding mechanism
  (here, `Export.populate_paper`) guarantees, before rating it a gap.
- **Stale-prior discipline paid off:** I triaged every REVIEW-REPORT finding against current source
  rather than trusting it. Result: nearly all HIGH items fixed; the report was net-misleading as a
  current bug list.
- **Subagent reliance:** core modules read by agents; I independently grep/code-verified every claim
  that drove a recommendation (export mechanism, merge resolver, C3 divergence, v1 parity, A8/A9).
- **Static-only:** no suite run, no populate. C3's blast radius is reasoned from code, not observed.

---

## 5. REVIEW-REPORT.md triage (folded in; report then deleted)

Triaged all findings (A1–A9 + low cluster, B, C/P2, D1–D9, E, F1–F3, P1–P6) of REVIEW-REPORT.md
(2026-06-02) against HEAD (2026-06-07). **Nearly everything is fixed**, including all four
originally-HIGH consumer-boundary silent bugs and the entire export/legacy-CI/test-integrity gap,
each with new dedicated tests. Still-live items below — these are the only carry-forwards.

### STILL-LIVE (carry forward)
| ID | Sev | Claim | Current evidence |
|---|---|---|---|
| **A8** | **High** | `NwbfileHasher` hashes name+attrs but **discards dataset values** (`_ = self.hash_dataset(obj)`) → two NWBs differing only in `ElectricalSeries` values hash identically | `utils/nwb_hash.py:327` — digest computed (255-280) then thrown away. Lives in v2's `cache_hash` recompute path; **warn-only** mitigates (degrades the integrity/recompute guarantee, not a hard silent-accept). Pre-existing upstream. *Fix:* fold `hash_dataset(obj)` into the per-object digest; add a value-difference regression test. |
| **A9** | Med | `LFPElectrodeGroup.cautious_insert` silently drops electrodes missing from `Electrode` | `lfp/lfp_electrode.py:172-184` — no `len()` equality check; wrong electrode set flows into LFP/ripple/decoding. Not v2 code but on-branch. *Fix:* assert/warn naming missing ids. |
| **D7** | Med | Burst-merge tutorial v1 cell calls nonexistent `plot_by_sorting_ids` | `notebooks/py_scripts/12_Burst_Merge_Curation.py:80` + `.ipynb:219`; real method is `plot_by_sort_group_ids` (v0 cell already fixed). One-token rename in both paired files. |
| **P5** | Low-Med | Production code branches on global `test_mode` to skip checks | `analysis/v1/group.py:90` (dup-group guard) + `unit_annotation.py:80` (unit-id validity) still skip under pytest. Most-visible `group.py:236` label filter now documented + covered by `test_filter_units`. Net Low (test_mode=False in prod). |
| low cluster | Low | latent/cosmetic | `artifact.py:1241` `>=` vs v1 strict `>` (measure-zero edge); `curation.py:399-401` reuse guard omits `apply_merge=True` (silent flag no-op on existing root); `utils.py:808-809` consolidate-intervals re-sort (harmless); `common_interval.py:381/588-592/253-255` three latent `Interval`/`IntervalList` issues (no in-tree trigger); `v0/figurl_views/SpikeSortingView.py:43,53` v0 dead code. |

### FIXED since the report (do not re-investigate)
A1/D1 (merge artifact_detection_id drop), A2 (multi-source `fetch_nwb` misalign), A3 (MS5 filter/whiten),
A4 (v0 unbound `get_curated_sorting`), A5 (unseeded `random_spikes` → now `seed=job_kwargs`/0),
A6 (`deprecate_log` kwarg), A7 (`UnitWaveformFeatures` v2 branch now reachable via `_fetch_waveform_v2`),
C/P2 (bulk `insert` now Pydantic-validated on all 4 Lookups), P3 (noise_levels length guard),
idempotency str-vs-UUID dedup, enum re-raise `__cause__`, D2/D3/D4 (export completeness + zero-unit,
new `test_export_safety.py`), D5 (legacy `spikeinterface.metrics` import try/except),
D6 (legacy CI resolution), D8 (`skip_if_no_dlc` string condition). B: no efficiency regressions.

---

## 6. Recommended actions (prioritized)
1. **C3** — canonicalize merged-id assignment to min-contributor order on both paths + add the
   preview==apply equivalence test. *(Small, the only landed-core correctness fix.)*
2. **A8** — fold dataset-value digest into `NwbfileHasher` + value-difference test. *(Highest-severity
   carry-forward; strengthens `cache_hash`/recompute integrity.)*
3. **C2** — extract a v2-owned `resolve_merge_restriction` before Phase 3 concat grows the surface;
   reconsider warn-only multi-curation fan-out.
4. **A9, D7** — one-line fixes (electrode-count guard; notebook method rename).
5. **C4 / C5 / C6** — optional hardening: cheap `CurationV2.insert1` guard; SI version-floor check;
   module splits. None urgent.
