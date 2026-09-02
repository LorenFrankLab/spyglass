# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

File:line refs into live code showing exactly what each phase touches and what it leaves alone. Verified by reading at planning time; re-verify if the tree has moved.

**Phase 1 — curation-scoped analyzer**
- `src/spyglass/spikesorting/v2/_analyzer_cache.py:148` — `analyzer_cache_lock` (reentrant, per-`sorting_id`, cross-process `FileLock`): the existing lock the resolver reuses; **do not** add a second locking scheme.
- `src/spyglass/spikesorting/v2/_analyzer_cache.py:219` — `_publish_sibling` / atomic staging+rename publish, and `analyzer_path:115`, `is_canonical_analyzer_folder_name:64`, `assert_path_safe_waveform_params_name:49`: the cache primitives the resolver extends to key on curation identity.
- `src/spyglass/spikesorting/v2/metric_curation.py:1389-1445` — `CurationEvaluation.make_compute`'s merged-curation `else:` branch already builds **temporary** curation-scoped display (+ metric when `wants_pc`) analyzers over merged spike trains via the shared low-level `build_analyzer` (`:1406`/`:1419`), reconstructing the merged sorting **DB-free** through `_sorting_from_units_nwb` (`:1394`). `make_compute` is a `_parallel_make = True` DB-free worker (`:999-1005`); routing decision `use_fast_path = matches_raw_namespace(...)` at `:1128`. Phase 1 does **not** reroute this method through the DB-backed resolver (that would break the tri-part DB-free contract); instead both this path and the resolver funnel through the same `build_analyzer`, and the resolver adds identity-keyed persistence + locking for interactive use. `source_analyzer_hashes` NULL for the merged path at `:983`.
- `src/spyglass/spikesorting/v2/metric_curation.py:191-216` — `_assert_curation_in_raw_namespace`: the guard that makes analyzer-backed plots raise on merged curations. Phase 1 removes the raise from the plot/read helpers and replaces it with resolver routing (the guard stays only where a caller genuinely needs the raw analyzer).
- `src/spyglass/spikesorting/v2/metric_curation.py:2477-2732` — the analyzer-backed read/plot helpers to route through the resolver: `get_waveforms:2477`, `plot_units_qc:2555`, `get_correlograms:2599`, `plot_correlograms:2609`, `investigate_pair_xcorrel:2622`, `investigate_pair_peaks:2637`, `plot_peak_over_time:2650`, `get_peak_amps:2665`, `plot_burst_pair_metrics:2680`, `plot_si_quality_metrics:2712`, `plot_si_template_metrics:2722`.

**Phase 2 — facade, value objects, strict errors, lifecycle**
- `src/spyglass/spikesorting/v2/_pipeline_types.py:115-137` — the `_RunV2SummaryBase` fields `sorting_id`, `root_curation_id`, `root_merge_id`, `analysis_curation_id|None`, `analysis_merge_id|None`, inherited by `RunV2SingleSessionSummary` (TypedDict, `:152`). Phase 2 wraps the returned summary so `run.root_curation` / `run.analysis_curation` return `CurationRef`s without the caller re-keying `root_curation_id`→`curation_id`.
- `src/spyglass/spikesorting/v2/metric_curation.py:1808` `accept_evaluation_outputs`, `:2023` `use_evaluation_labels`, `:2056` `overlay_evaluation_labels`, `:1928` `preview_merges`, `:1960` `accept_merges`, `:1994` `accept_all_suggested_merges` — the five-plus apply methods the scripted facade re-expresses as `evaluation.merge_and_evaluate(...)` / `.accept_labels(mode=...)` / `.preview_merges(...)` (+ expert `commit_merges(...)`). These stay as the expert layer; Phase 3's review session is the primary hands-on facade.
- `src/spyglass/spikesorting/v2/_metric_curation_nwb.py:53-78` — `_scalar_or_nan`, the metric **write-path** coercion (non-scalar → NaN + warning). Phase 2 flips its except-branch to raise `UnsupportedMetricValueError` by default; legitimate scalar-float NaN keeps the `float(value)` success path. NOT `_metric_curation.py:38-53` (`_is_finite_metric_value`), which is a separate read-side threshold filter left as an explicit decision.
- `src/spyglass/spikesorting/v2/curation.py:78-104` — `CurationV2` heading: `curation_source` enum (`manual`/`analyzer_curation`/`figpack`/`curation_evaluation`), `merges_applied`, `parent_curation_id=-1`, `object_id`, `description`. Phase 2 derives `state` and `operation_type` from these existing columns (no schema).
- `src/spyglass/spikesorting/v2/curation.py:326` `delete` (orphan-lineage guard `:368-380`), `:388` `audit_orphaned_lineage`, `:2008` `has_unapplied_proposed_merges`, `:2044` `is_committed_curation` — the existing lineage/state machinery the lifecycle helpers build on.

**Phase 3 — browser-first FigPack review**
- `src/spyglass/spikesorting/v2/figpack_curation.py:450-471` — the `if upload: view.show(...)` / else `view.save(...)` branch; `:691-697` the `FigPackUploadError` that refuses a hosted upload of a curation carrying pre-existing labels/merges; `:717-719` `_write_seed_annotations` called only for the offline bundle. Phase 3 removes the asymmetry (seed hosted too, per the spike outcome).
- `src/spyglass/spikesorting/v2/figpack_curation.py:341-343` `FigPackCurationNamespaceError`, and `build_curation_view:735`, `fetch_curation_from_uri:774`, `save_curation_from_uri:787`; `figpack_config_hash` imported at `:42`. Phase 3 routes view-building through the Phase 1 resolver (so merged curations open), embeds figure identity, and makes `save_curation_from_uri` verify identity instead of trusting the caller's parent key.
- `notebooks/py_scripts/10_Spike_SortingV1.py:228-290` and `src/spyglass/spikesorting/v1/figurl_curation.py:21-231` — the real UX baseline: researchers generate a seeded FigURL/curation URI, curate labels and merges in the browser, fetch the external JSON, and manually insert the resulting curation. Phase 3 preserves that browser-first mental model while collapsing URI, selection/populate, key, metric-transfer, and import plumbing into a resumable review object.

**Phase 4 — typed annotations (schema)**
- `src/spyglass/spikesorting/v2/curation.py:78-104` — new provenance columns (`created_at`, `created_by`) added to `CurationV2`; three new tables added in a new/extended schema module. See [phase 4](phase-4-unit-annotations.md).
- `src/spyglass/spikesorting/analysis/v1/unit_annotation.py` — the existing downstream `UnitAnnotation` surface; Phase 4's read interface must not collide with or duplicate it (it annotates curated units, not merge-table units).

## Scope and dependency policy

### Goals

- Preserve the lab's **browser-first v1 FigURL mental model** while removing its URI/key/table plumbing: start a review, curate in FigPack, preview the exact change set, commit, and continue into merged-unit verification.
- A curator evaluates, merges, **visually verifies the actual merged unit**, labels, and exports — without reconstructing keys, transferring metric dicts, typing merge IDs in the primary workflow, or calling DataJoint `insert_selection`/`populate` directly.
- One **low-level analyzer builder** (`build_analyzer`) serves evaluation, plotting, and FigPack — retaining two callers by design (the DB-free `make_compute` worker and the DB-connected resolver).
- One small, documented review facade is the paved road; the scripted `evaluate`/plot facade is secondary and table methods remain the expert layer.
- v1's custom-metric flexibility is restored through a typed, immutable extension model — not v1's `metrics=` argument.

### Design principle (SpikeInterface boundary)

Spyglass stays the **durable provenance / workflow layer**; SpikeInterface 0.104.3 is the **pinned computation and interchange layer**; neither becomes the other's data model. Concretely: Spyglass owns durable unit IDs, lineage, NWB, transactions, and the gap-correct merge; SI owns analyzer/extension computation and (behind adapters/tests) its curation model. SI's `curation` API is explicitly experimental — anything consuming it is isolated behind a pinned adapter, never the DB schema.

### Non-Goals

- **No new sorter, matcher, metric, or visualization plugin APIs** (owner-settled; see the review-triage memory).
- **No change to the frozen scientific parameter recipes** or their identity/hashing.
- **Not** restoring v1's `BurstPair` *stored table* — the burst-pair analysis stays as on-demand plot helpers (a separate, deferred decision; out of scope here).
- **Not** touching cross-session matching, general pipeline presets, or the recording/artifact/sort stages except where the facade re-exposes them read-only. `CurationReviewProfile` is a narrow persisted binding for the curation UX, not a new general preset/plugin system.
- **Not** reintroducing `insert_curation(metrics=...)`: it mixed scientific state with computed annotations and caused the HDF5 write failure that retired it.
- **Not** building the SpikeInterface curation-JSON interchange adapter now (`export_si_curation` / `import_si_curation`) — **deferred as a follow-up (YAGNI)**: nothing in Spyglass currently consumes SI's Pydantic curation model. Trigger to build it: a concrete need (a collaborator sharing SI curation JSON, or wanting SI auto-labelers). When built, scope it to labels+merges, fail-loud (`UnsupportedCurationOperationError`) on removals/splits, preserve the format version, and pin SI-0.104.3 contract fixtures — and keep it an adapter, not the data model.

### Dependency policy

No new runtime dependencies. FigPack work stays behind the existing optional `spikesorting-v2-curation` extra (`figpack` + `figpack_spike_sorting`); every FigPack entry point must keep raising the current actionable `ImportError`/install hint when the extra is absent.

## Metrics

**Release gate (end-to-end browser-first user test).** A current v1 FigURL user can run this with no manual key construction, no metric-dict transfer, no typed merge IDs, and no direct `insert_selection`/`populate`. `start_review` resolves and persists the profile's exact evaluation spec, builds/reuses the evaluation, seeds the figure, and returns a resumable review handle:

```python
run = run_v2_pipeline(...)
review = run.start_review(
    source="analysis",                            # never silently falls back to root
    profile="franklab_hippocampus_2026_06",
    upload=True,
)
review.open()                                      # researcher labels + selects merges in FigPack

changes = review.preview_import()                 # pure diff: labels, merges, conflicts, unit-count delta
display(changes)
receipt = changes.commit()                        # UUID + annotation-hash rechecked; child + re-evaluation

if receipt.needs_merge_verification:
    verification = receipt.continue_review()      # actual merged analyzer; same profile/config
    verification.open()
    final = verification.preview_import().commit(
        confirm_no_changes=True                   # an explicit reviewed/no-change result is valid
    ).curation
else:
    final = receipt.curation

final.merge_id
```

The scripted alternative remains supported for reproducible batch/expert work: `curation.evaluate(...) → evaluation.merge_and_evaluate(...) → post.plots.* → accept_labels(...)`. It is not the canonical hands-on tutorial.

Secondary, measurable:
- Rendering a merged unit's waveform + correlogram succeeds where it raised before (Phase 1 acceptance test).
- A hosted **or controlled-bundle** FigPack figure of an already-labeled merged curation round-trips edits back as an identity-verified child (hosted persist is spike-gated; the remote read-back falls back to external controlled storage); a merge import leads directly to a second review over the actual merged analyzer (Phase 3 acceptance test).
- A custom annotation set imports from and exports to a DataFrame and appears in a summary/FigPack column (Phase 4 acceptance test).
- Cache footprint is bounded: orphan sweep + retention policy remove analyzers no live curation references (Phase 1).
- In a moderated usability check, at least three current v1 FigURL users complete the browser-first journey without constructing a DataJoint key or requiring facilitator intervention; all identify the reviewed parent, understand the proposed diff, resume a review, and retrieve the final `merge_id`. Record completion time and errors against the v1 baseline; do not ship a regression hidden by technical tests.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Two analyzer-building paths (resolver + `CurationEvaluation`'s temp build) drift in preprocessing/extension set. | Both funnel through the one shared low-level `build_analyzer` (already called at `metric_curation.py:1406`/`:1419`), so analyzer *content* cannot diverge. `make_compute` stays DB-free (not rerouted through the DB-backed resolver — that would break the tri-part contract); the resolver only adds persistence + locking. |
| Merged-analyzer persistent cache has no row-level drift detection (`CurationEvaluation.source_analyzer_hashes` is NULL for the merged path by design). | Pre-existing property, not introduced here; the cache manifest *adds* drift detection the row lacks. Recorded as a known gap; extending `source_analyzer_hashes` to the merged path is a separate follow-up (Open Question). |
| The existing sort-analyzer orphan sweep deletes curation-scoped cache folders (its `referenced_paths` covers only `Sorting` recipes). | Phase 1 extends the sweep's `referenced_paths` to protect folders referenced by live `CurationV2` rows (or gives them a name outside the sweep's jurisdiction); a validation test asserts a live curation's cache survives a sweep. |
| Merged-analyzer cache grows on a real lab DB (per-curation × recipe × role folders). | Per-curation identity, manifest-validated reuse, the extended orphan sweep, and a **reference-count** reclaim (drop folders no live `CurationV2` references) wired into the existing analyzer recompute/reclaim trio — not a fresh cleanup path. No cross-curation content dedup (folders keyed by the immutable `curation_uuid`). |
| Cache treated as a durable scientific artifact and trusted when stale. | Manifest carries curated unit ids, contributor map, merged spike-content hash, merge-policy version, source hashes, recipe hash, SI version, and extension inventory; load validates and rebuilds on any mismatch. Regeneratable cache, never provenance of record. |
| A process dies mid extension-write, leaving a valid manifest beside a partial extension that a later load trusts. | The published analyzer is **read-only** with its full display-extension set built + validated + atomically published; extra extensions branch to staged derivatives, never in-place writes. A folder whose zarr does not match its `extension_inventory` is treated as corrupt → rebuilt (F3). |
| Strict metric errors break a currently-"working" run that silently NaN-coerced a drifted metric. | The write-path change (`_scalar_or_nan`, `_metric_curation_nwb.py:53-78`) is **unconditional** (no `strict=False` — a hidden switch would silently alter scientific output; preproduction) and uses an explicit 0-dim/numeric check, not bare `float()`. The all-NaN class (`nn_noise_overlap`) is surfaced by **rule-input validation** (`nan_policy` model), not this error. |
| `save_curation_from_uri` attaches one figure's edits to a different curation with coincidentally similar unit ids. | Figure embeds `(sorting_id, curation_uuid, curation_id, figpack_config_hash)`; import verifies the immutable `curation_uuid` (NOT `curation_id`, which is reused after a delete — B1) and refuses a mismatch. Identity-less figures are refused on the paved road (fail-closed); unsafe import is a separate `import_legacy_figpack_curation`. |
| Adopting SI's `merge_units`/`apply_curation` `censor_ms` regresses Spyglass's gap-correct absolute-time dedup on disjoint recordings. | Phase 1's **gating spike (F6)** compares the custom build vs SI `merge_units` on disjoint intervals + sub-0.4 ms cross-contributor duplicates; SI is adopted only on exact parity (strong prior it is not, so the custom build stays). |
| Hosted FigPack cannot safely seed and/or persist pre-existing state (the current code refuses seeded hosted uploads, `figpack_curation.py:691-697`), yet browser-first is the **primary** journey. | Phase 3 begins with a **spike** on the two independent halves (seed, remote persist/read-back). Seed-fail → prebuilt seeded bundle. **Remote-persist-fail → external-controlled-storage read-back (GitHub/kachery-style, mirroring v1), so the primary remote journey still works.** The primary journey never depends on figpack.org's own persistence being proven. Gate the rest of the phase on the spike; don't hard-code `upload=True` as canonical until it resolves. |
| A friendly profile name becomes an in-memory or mutable alias, so the same user action means different science later. | `CurationReviewProfile` is DB-persisted and immutable; changed content requires a new name/hash, and every figure embeds the resolved exact profile snapshot. Delivery settings remain outside it. |
| Browser import hides consequential merge/label behavior or changes between preview and commit. | `preview_import()` is read-only and shows label/merge diffs, unit-count delta, and merge-label conflicts. `commit()` rechecks both `curation_uuid` and the annotations-content hash (TOCTOU guard); unresolved conflicts block rather than choosing silently. |
| Opening a metric-assisted review implicitly accepts every proposal, or an applied merge is re-seeded as pending. | Only committed labels seed editable annotations. Evaluation proposals and applied `merged_from` provenance are read-only cues; they become changes only through explicit browser actions in the current unit namespace. |
| A two-pass merge-verification workflow feels like restarting curation. | `ReviewImportReceipt.continue_review()` carries the same immutable profile, displayed properties, label palette, and publish mode into a view built from the actual merged analyzer; progress and completed stages are visible and resumable. |
| Two reviewers edit the same parent or a newer child appears while a review is open. | Each import creates/reuses a sibling child pinned to the reviewed `curation_uuid`; never silently rebase or overwrite. Preview reports newer children and reviewer provenance so the user can choose deliberately. |
| Phase-2 facade requires `parent_curation: CurationRef` on non-root APIs — a breaking signature change. | Acceptable in preproduction; migrate the notebooks/tests in the same PR. Root sentinel stays only on initial-creation. |

## Rollout Strategy

Ships as **five** sequential PRs (Phase 0–4); each leaves the tree green and independently reviewable. **Phase 0 and Phase 4 are the schema phases**. Phase 0 adds `curation_uuid`, rule `missing_policy`, and the immutable `CurationReviewProfile` lookup; Phase 4 adds annotation tables plus `created_at`/`created_by`. Phases 1–3 build on Phase 0. The expert table-level API stays available, the scripted facade is secondary, and Phase 3 makes the browser review facade the documented default. No feature flag: this is preproduction, and the canonical notebook is rewritten around the real v1-user journey rather than retaining the hand-keyed path in parallel.

## Upstream dependency

`.claude/docs/plans/pr1609-remediation/` (2026-09-02) must land its Phase 1 before Phase 0 here starts: both edit `CurationV2.insert_curation`, and its Phase 1 shifts line numbers in `curation.py`, `metric_curation.py`, `_pipeline_run.py`, and `_pipeline_types.py` (re-verify the integration-point refs above afterwards). Its Phase 2 adds `ConcatMemberCuration`; Phase 3's `commit()` here must populate it for concat-backed sorts, and the release-gate `final.merge_id` is `None` for concat (use `member_merge_ids`). The shipped `CurationReviewProfile` rule set (Phase 0) and the `franklab_*` preset rule set (remediation task C2) are one decision; see that plan's overview.

## Open Questions

1. **Hosted FigPack feasibility — two independent halves.** (a) *Seed*: can a hosted figpack.org figure be seeded with prior curation state? (b) *Persist/read-back*: can a **remote** reviewer's edits to a hosted figure be read back for `preview_import`? These are separate and each needs its own fallback. *Deferred to the Phase 3 spike (its first task).* Fallbacks (so the browser-first **primary** journey survives a failed spike): if (a) fails → publish a prebuilt seeded bundle through controlled storage; **if (b) fails → the remote read-back uses external controlled storage (GitHub/kachery-style, mirroring v1's `gh://LorenFrankLab/sorting-curations/.../curation.json` at `10_Spike_SortingV1.py:264-284`), not figpack.org's own persistence.** The offline-local-bundle path is not a substitute for remote collaboration. Until the spike resolves, the release-gate does not hard-code `upload=True` as canonical.
2. **`metric` analyzer role retention** — keep the whitened metric-role analyzer cached, or rebuild per evaluation? *Current best answer: build it through the shared primitives but leave it ephemeral unless the Phase 1 benchmark shows reuse pays for the disk; the `display` role is the one that must be retained (plots + FigPack read it).*
3. **Retention policy shape** — LRU by disk budget, or reference-count against live curations only? *Current best answer: reference-count (drop analyzers no curation references) in Phase 1; add a disk-budget cap only if footprint measurements demand it.*

## Estimated Effort

Rough diff sizing for executor expectations (no time estimate):
- Phase 0: ~150–250 LOC (identity/rule alterations + immutable review-profile lookup), plus migration and contract tests.
- Phase 1: ~500–800 LOC (resolver + cache extension + rerouting ~11 helpers + reconciling `CurationEvaluation`), plus tests. The heaviest phase.
- Phase 2: ~400–600 LOC (value objects, facade methods, `merge_and_evaluate`, strict-error swap, lifecycle helpers), plus notebook/test migration.
- Phase 3: ~450–700 LOC after the spike (review facade + diff/commit receipts + routing + identity embed/verify + guided continuation + seed symmetry), plus the browser-first notebook and usability exercise.
- Phase 4: ~400–600 LOC (three tables + columns + read interface + DataFrame IO + `alter()`), plus tests.
