# Designs

[← back to PLAN.md](PLAN.md)

Per-component algorithmic detail. Phases reference these by anchor and do not restate them.

- [D1 — Curation-scoped analyzer resolver & cache](#d1)
- [D2 — merge_and_evaluate orchestration](#d2)
- [D3 — Strict metric-value handling](#d3)
- [D4 — FigPack figure-identity verification](#d4)
- [D5 — Typed unit-annotation model](#d5)
- [D6 — Browser-first review orchestration](#d6)

---

## D1 — Curation-scoped analyzer resolver & cache {#d1}

Signature and routing contract: [get_curation_analyzer](shared-contracts.md#get_curation_analyzer). Manifest: [Analyzer cache manifest](shared-contracts.md#analyzer-cache-manifest).

**Where it lives.** New internal module `src/spyglass/spikesorting/v2/_curation_analyzer.py`, sitting on top of the existing `_analyzer_cache.py` primitives. It does **not** fork the cache — it extends the identity used to key it.

**Cache path identity — per-curation, keyed by `curation_uuid` (F1 + B1).** Today `_analyzer_cache.analyzer_path(sorting_id, waveform_params_name)` (`_analyzer_cache.py:115`) keys a folder by sort + recipe. Extend to a **per-curation** path: `(sorting_id, curation_uuid, role, waveform_recipe_hash, spikeinterface_version)`. Use `curation_uuid` (Phase 0), **not** `curation_id`: the numeric id is `max+1` and reused after a delete (`curation.py:1062-1072`), so a `curation_id`-keyed path could be reused across different curation generations. There is **no** cross-curation content dedup (a path that encodes the curation identity cannot collapse distinct curations); `curation_uuid` pins the generation, and the manifest validates the merged spike content. Reuse `is_canonical_analyzer_folder_name` (`:64`) / `assert_path_safe_waveform_params_name` (`:49`) path-safety conventions, but see **Retention** below.

**Build/reuse algorithm — read-only publish + staged derivatives (F3).** A mutable shared cache is not crash-safe even under a lock: a process dying mid extension-write leaves a valid input manifest beside a partially written extension, and a later load can treat it as valid. So the published analyzer is **immutable**: build the display analyzer with its full standard display-extension set in staging, validate it (manifest incl. `extension_inventory`), atomically publish, then treat it as **read-only**. A plot needing an extension outside the standard set gets a **staged derivative** whose identity includes the extension name + param hash — never an in-place write to the published cache. The lock is held only around build→publish, not across caller reads.

```python
def get_curation_analyzer(curation_key, waveform_recipe, role="display", extra_extensions=None):
    namespace = classify(curation_key)   # raw | merged-committed | preview | zero-unit  (DB reads OK — interactive path)
    if namespace == "preview":
        raise <existing "commit the merge first" error>        # curation.py:2144-style message
    if namespace == "zero-unit":
        raise ZeroUnitAnalyzerError(...)                       # concrete; plots render an empty-state view
    if namespace == "raw":
        base = load_or_rebuild_raw_analyzer(sorting_id, waveform_recipe, role)   # existing fast path
    else:  # merged-committed
        path = curation_analyzer_path(curation_key, waveform_recipe, role)       # per-curation identity
        with analyzer_cache_lock(curation_key["sorting_id"]):   # reentrant (_analyzer_cache.py:148); held only around build→publish
            if not (path.exists() and manifest_valid(path, curation_key, waveform_recipe, role)):
                staging = build_merged_analyzer(curation_key, waveform_recipe, role)  # full display extension set
                validate_complete(staging)                     # every inventoried extension present + complete
                write_manifest(staging, current_manifest(...))
                atomic_publish(staging, path)                  # _analyzer_cache._publish_sibling pattern, :219
        base = si.load_sorting_analyzer(path)                  # READ-ONLY published analyzer
    if not extra_extensions:
        return base
    return staged_derivative(base, extra_extensions)           # keyed by (base identity, ext name, param hash); never mutates `base`
```

**The shared unit is the low-level builder `build_analyzer`, not this resolver — the single-low-level-builder invariant ([shared-contracts](shared-contracts.md#get_curation_analyzer)).** `build_analyzer` (in `_sorting_analyzer`) is *already* what both paths call: `CurationEvaluation.make_compute`'s merged branch calls it at `metric_curation.py:1406` (display) and `:1419` (metric, when `wants_pc`) over a `TemporaryDirectory` (the actual merged build lives at **`metric_curation.py:1389-1445`**, inside `make_compute` which starts at `:1191`; the routing decision `use_fast_path = CurationV2.matches_raw_namespace(...)` is at `:1128`). `make_compute` reconstructs the merged sorting **DB-free** via `_sorting_from_units_nwb` (`:1394`), not `get_merged_sorting`, because it is a `_parallel_make = True` spawn worker that resolves no DB inputs (`:999-1005`).

`build_merged_analyzer` (new, in `_curation_analyzer`) is the **DB-connected** wrapper: it may reconstruct the merged sorting via `CurationV2.get_merged_sorting` (`curation.py:2492`) since the interactive resolver has a DB connection, then calls the **same** `build_analyzer`. **Do not reroute `make_compute` through the resolver** and **do not delete its temp build** — that would push a DB dependency into the DB-free worker and break the tri-part contract (`test_tripart_dispatch_active`). Both paths building over the same `build_analyzer` with the same recipe yields byte-equivalent analyzers; the resolver merely adds persistence + locking for interactive reuse.

**Manifest validation** recomputes each field ([contract](shared-contracts.md#analyzer-cache-manifest)) and compares. Critical known footgun to preserve from the Phase-4a cache work: SI zarr stores the recording path **relative**, so the atomic publish must stage the temp analyzer as a hidden sibling of the final slot (the `_publish_sibling` pattern) — staging elsewhere and moving in yields a recordless analyzer on reload. The extracted build must keep this.

**Retention / cleanup — and a hazard the existing sweep creates.** `Sorting.find_orphaned_analyzer_folders` builds its `referenced_paths` set **only** from each `Sorting` row's display recipe (`sorting.py:2600-2609`) and PC-requesting `CurationEvaluationSelection` metric recipes (`:2622-2629`); `disk_dir_paths` is every under-root folder passing `is_canonical_analyzer_folder_name` (`:2648-2654`), and `classify_orphaned_analyzer_folders` marks any disk folder **not** in `referenced_paths` as a disk-side orphan to delete (`_analyzer_cache.py:459-460`). So a curation-scoped folder named to pass the canonical check is seen as garbage and **deleted out from under a live curation** — the opposite of "recognizes the new folders." **Resolution (decided, not a fork):** one **typed cache subsystem** with two *kinds* — `raw` (sort display/metric) and `curation` (per-curation) — behind **one reference collector** (unions `Sorting` display + PC-requesting metric recipes *and* folders referenced by live `CurationV2` rows, keyed by `curation_uuid`) and **one reclaim entry point**. `find_orphaned_analyzer_folders`/`classify_orphaned_analyzer_folders` consume that single collector, so a live curation's folder is never classified as an orphan. Retention = drop folders no live reference names (reference-count under the per-sort lock). Add a validation test that the orphan sweep does **not** delete a live curation's cache. See [Open Question 3](overview.md#open-questions).

**Read-only, not locked-mutate {#m1-locked-mutate}.** Several rerouted plot helpers compute-and-persist extensions and today run under `_locked_display_analyzer`, which holds `analyzer_cache_lock` across load+mutate (`metric_curation.py:2537-2553`, `:2582-2586`). An earlier draft made the resolver a locked-mutate context manager to preserve that. That is **superseded**: a lock serializes writers but does not make a mutable cache crash-safe (a killed extension write leaves a valid manifest beside a partial extension). So the published analyzer is **read-only** with its full display-extension set built at publish time, and a plot needing an extra extension gets a **staged derivative** (algorithm above) rather than mutating the shared folder. Rerouted helpers therefore call `get_curation_analyzer(...)` for a read, and pass `extra_extensions=` when they need something beyond the standard set — no in-place mutation of the cache.

**F6 — SI `merge_units` parity spike (Phase 1 gating first task).** Before committing to the custom `build_merged_analyzer`, spike whether SI 0.104.3's `SortingAnalyzer.merge_units(merge_unit_groups=…, new_unit_ids=…, censor_ms=…, merging_mode="hard", format="zarr")` (verified — `merge_unit_groups` is the required first arg) can build the merged analyzer directly from the raw analyzer, potentially eliminating the rebuild. **Strong prior it cannot be adopted:** Spyglass's `get_merged_sorting` deduplicates in **absolute spike time** so disjoint-recording wall-clock gaps are respected (`curation.py:2492` docstring, verified), whereas SI's `censor_ms` operates on the sorting's **sample** representation — not gap-correct across disjoint recordings. Compare path A (CurationV2 committed sorting → `build_analyzer`) vs path B (raw analyzer → `merge_units(...)`) on disjoint intervals, same-sample duplicates, sub-0.4 ms cross-contributor duplicates, exact unit IDs/spike trains, templates, correlograms, quality metrics, and reload-after-publish. Adopt SI **only** on exact equivalence; otherwise retain the custom build and record the evidence. See [Phase 1](phase-1-curation-analyzer.md).

**M3 — merged-cache provenance gap (accepted, documented).** The persisted merged analyzer is regeneratable cache validated by its manifest, but the `CurationEvaluation` row's `source_analyzer_hashes` is NULL for the merged path by design (`metric_curation.py:983`, `:1451`), so `detect_stale_source` (`:1525-1621`, which only re-hashes when that field is truthy, `:1585`) cannot detect input drift for a merged row. This is a **pre-existing** property of the merged path, not introduced here — the manifest actually *adds* drift detection at the cache layer the row lacks. Phase 1 does not close the row-level gap; it is recorded as [Open Question / risk](overview.md#risks-and-mitigations). Extending `source_analyzer_hashes` to the merged path is a separate follow-up, not in scope.

## D2 — merge_and_evaluate orchestration {#d2}

The paved-road merge path is the **method** `EvaluationResult.merge_and_evaluate(groups)`, which reuses the evaluation's own `spec` (the two recipe names) — one idempotent, resumable call from the evaluated curation → committed merged child → populated evaluation → receipt (`receipt.evaluation` is the post-merge `EvaluationResult`). There is **no** `evaluation_preset` argument (the phantom registry is gone) and **no** no-arg `merged.evaluate()`; the criteria come from `self.spec`. A lower-level function form takes an explicit `EvaluationSpec`. Mirrors `run_v2_pipeline`'s idempotent-orchestrator philosophy; **must not** be called inside a caller-owned DataJoint transaction (the evaluation `populate` manages its own).

**This constraint is NOT already enforced — add the guard.** The existing `sorting.py:903-905` check is artifact-specific (gated on `artifact_detection_id is not None and cls.connection.in_transaction`); `CurationEvaluationSelection.insert_selection` has no such guard, and `transaction_or_noop` (`utils.py:107-114`) *silently nests* when already in a transaction (its own docstring warns "do not call from inside an outer transaction"). So without a new guard, `create_merged_curation`/`insert_selection` would nest silently and only the step-4 `populate` would fail — with a raw DataJoint error, not an actionable message. `merge_and_evaluate` must therefore begin with an explicit `if connection.in_transaction: raise <actionable "populate manages its own transaction; call outside any open transaction">` before step 1.

```python
# EvaluationResult.merge_and_evaluate(groups) — reuses self.spec; self.curation is the parent.
def merge_and_evaluate(self, merge_groups) -> MergeEvaluateReceipt:
    if connection.in_transaction:                          # M6 — add the guard (not already enforced)
        raise <actionable "call outside any open transaction; populate manages its own">
    # 1. validate merge_groups against the parent's unit namespace (each id must exist; >=2 per group)
    validate_merge_ids(self.curation, merge_groups)
    # 2. commit-or-reuse the merged child (reuse_existing=True → idempotent)
    child = CurationV2.create_merged_curation(
        sorting_key=self.curation.as_key(),
        merge_groups=merge_groups,
        parent_curation_id=self.curation.curation_id,
        reuse_existing=True,
    )                                                      # curation.py:1550
    # 3. create-or-reuse its evaluation selection with THE SAME criteria (self.spec)
    sel = CurationEvaluationSelection.insert_by_curation_id(
        child["sorting_id"], child["curation_id"],
        self.spec.metric_params_name, self.spec.auto_curation_rules_name,
    )                                                      # metric_curation.py:890 (NOT from_names — that does not exist)
    # 4. populate (idempotent; heavy)
    CurationEvaluation.populate(sel)
    # 5. receipt: child ref + post-merge EvaluationResult + warnings + per-stage status
    return MergeEvaluateReceipt(child=CurationRef.from_key(child), evaluation=<post-merge EvaluationResult>, ...)
```

Resumability: every step is find-or-create keyed on content, so a re-run after a crash at step 4 reuses steps 2–3 and only re-attempts the populate. The receipt reports which stages were reused vs freshly run (same shape as the `describe_run` receipt).

## D3 — Strict metric-value handling {#d3}

Error type: [UnsupportedMetricValueError](shared-contracts.md#unsupportedmetricvalueerror).

**Target the WRITE-path coercion**, `_scalar_or_nan` at `_metric_curation_nwb.py:53-78` — "Coerce a metric cell to float; non-scalar / non-numeric → NaN." Its current body does `try: return float(value) except (TypeError, ValueError): warn; return NaN`. Replace the except-branch return with a raise:

```python
def _scalar_metric(value, *, context=""):   # strict, UNCONDITIONAL (no strict= flag)
    # explicit scalar test — NOT bare float(value), which coerces a one-element array
    arr = np.asarray(value)
    if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
        return float(arr)                    # legitimate scalar (incl. float NaN) — passes
    raise UnsupportedMetricValueError(
        f"metric write [{context}]: expected scalar numeric, received "
        f"{value!r} (type={type(value).__name__}, "
        f"shape={getattr(value,'shape',None)}, dtype={getattr(value,'dtype',None)})"
    )
```

The legitimate low-spike NaN is a 0-dim float NaN → passes. Only a genuinely non-scalar/object value raises. **No `strict=False` escape hatch:** it silently changes auto-curation output and is preproduction; if a compatibility mode is ever truly needed it must be part of the evaluation recipe identity + persisted provenance, not a hidden flag ([contract](shared-contracts.md#unsupportedmetricvalueerror)).

**What this does NOT do, and where the gap goes:**
- It does **not** touch `_is_finite_metric_value` at `_metric_curation.py:38-53` (the read-side threshold filter). Whether that should also raise is a distinct, explicitly-called-out decision, not folded in silently.
- It does **not** surface the `nn_noise_overlap`-always-NaN class (a scalar float NaN passes the scalar check). That is handled by the **rule-input validation** in the [contract](shared-contracts.md#unsupportedmetricvalueerror): a rule referencing an all-NaN column fails with an actionable error unless it declares an explicit missing-value policy (SI's `threshold_metrics_label_units(nan_policy=…)` model). Add it as a Phase 2 task alongside the write-path change.

## D4 — FigPack figure-identity verification {#d4}

Figure identity schema and the refuse-on-mismatch rule: [Figure identity](shared-contracts.md#figure-identity).

**Embed** at build time (`build_curation_view`, `figpack_curation.py:735`): write `{sorting_id, curation_uuid, curation_id, figpack_config_hash}` into the figure config. `curation_uuid` (Phase 0) is the verification key — **not** `curation_id`, which is `max+1` and reused after a delete (`curation.py:1062-1072`), so an old figure could otherwise pass against an unrelated replacement curation (B1). No `unit_namespace_hash` is embedded (dropped).

**Verify** on import (`save_curation_from_uri`, `figpack_curation.py:787`): recompute `curation_uuid` from the target curation and compare to the figure's embedded identity; on mismatch raise (do not fall back to a caller-supplied parent key). Read mechanic: `save_curation_from_uri` today reads only `<uri>/annotations.json` via `fetch_curation_from_uri` (`:774-784`), but the identity lives in the figure *config*, so verification needs an **added figure-config read** — an explicit task.

**Legacy figures — fail-closed.** The verified direct-import primitive `save_curation_from_uri` gets **no** `allow_legacy` parameter; the primary `FigPackReview` path derives its parent from verified figure identity and never asks the user for a parent key. Unsafe import lives behind `import_legacy_figpack_curation(uri, asserted_parent=…, confirm_unverified_identity=True)`.

**Seed symmetry and semantics** (gated on the Phase 3 spike): today `_write_seed_annotations` runs only for the offline bundle (`figpack_curation.py:717-719`) and a hosted upload of a curation with pre-existing state is refused (`:691-697`). If hosted seed/persist works, seed hosted too; otherwise use the controlled seeded-bundle fallback. Seed only **committed editable state** (current labels). Do not seed evaluation suggestions as accepted annotations: expose proposed labels and merge groups as clearly named read-only columns/overlays until the researcher explicitly acts. Do not seed an already-applied merge's raw contributor provenance as pending `mergeGroups` in a merged namespace; show `merged_from` provenance read-only. This avoids feeding raw contributor IDs absent from the current merged unit namespace back into import.

The figure config also embeds the immutable review-profile snapshot (`review_profile_name`, `profile_hash`, exact `EvaluationSpec`, ordered displayed properties, label palette, and label-import mode), and `figpack_config_hash` includes that snapshot. This lets `FigPackReview.resume(review_id)` reconstruct the context after a process restart without trusting a mutable in-memory alias. `annotations_hash` lives in the returned `CurationChangeSet`; preview remains mutation-free, and a resumed review simply previews again before commit.

## D5 — Typed unit-annotation model {#d5}

Restores v1's custom-metric flexibility as immutable, content-addressed, typed annotation sets — separate from curation identity (so stale annotations can never masquerade as scientific state, the failure v1's `metrics=` caused).

Three tables (new schema module, e.g. `src/spyglass/spikesorting/v2/unit_annotation.py`):

```
UnitAnnotationDefinition            # Lookup: what an annotation IS
  name : varchar(64)
  version : int
  ---
  value_type : enum('float','int','bool','text')
  physical_unit : varchar(32)   # e.g. 'uV', 's', '' ; NOT a curation label
  description : varchar(255)

CurationUnitAnnotationSet           # a producer's run of one definition over one curation
  -> CurationV2                      # the exact curation namespace
  -> UnitAnnotationDefinition
  set_hash : char(64)                # content address of (definition, params, values)
  ---
  producer : varchar(128)
  producer_version : varchar(64)
  producer_parameters : blob         # the NORMALIZED params themselves, not only the hash
  parameters_hash : char(64)
  created_at : timestamp
  created_by : varchar(128)

CurationUnitAnnotationSet.Value     # part: one typed value per unit
  -> master
  -> CurationV2.Unit                 # DB-ENFORCED unit reference (precedent: TrackedUnit.Member, unit_matching.py:1345)
  ---                                # — cascade-deletes with the curation; not app-validation only
  value_float=null : double         # `double`, NOT MySQL `float` — float is single-precision and
  value_int=null : bigint           # breaks exact content-addressing (see the float-precision memory).
  value_bool=null : bool            # Exactly one typed column is set, per the definition's value_type.
  value_text=null : varchar(255)
```

Canonical `set_hash` serialization (specify, do not hand-wave): sort values by `unit_id`; include the definition `name`+`version`+`value_type`; normalize NumPy scalars to Python types; define explicit encodings for `NaN`/`None`/empty-text; then hash the canonical form. Annotation **definitions are immutable by `version`** — changing a definition means a new `version`, never an in-place edit — so a `set_hash` is reproducible.

Invariants:
- Every `Value.unit_id` references a unit in the annotation set's **exact** curation namespace (validate against `CurationV2.get_sorting`'s unit ids). Categorical **curation labels stay separate** (the `CurationLabel` / `UnitLabel` path) — annotations are generic computed quantities, never a back door for labels.
- A set is **immutable and content-addressed** (`set_hash`) — re-running the same producer with the same params/values is idempotent; a change produces a new set, never an in-place update. Store the normalized `producer_parameters`, not only their hash, so a set is self-describing.
- **Common read interface — explicit selection, never implicit "latest".** A curation may carry multiple evaluations (different metric recipes) and multiple annotation sets (same definition). The reader **requires** explicit inputs and never picks a latest value:
  ```python
  read_unit_properties(curation, evaluation=EvaluationRef(...), annotation_sets=[AnnotationSetRef(...), ...])
  ```
  It unions built-in `CurationEvaluation.get_metrics` with the named custom sets keyed on `(curation, unit_id)`, without forcing both into one physical table, and applies **deterministic column-name collision behavior** (e.g. namespaced columns / documented precedence) rather than a silent overwrite.
- DataFrame import/export: `from_dataframe(curation, definition, df)` and `to_dataframe(set_key)`; explicitly-named annotations are addressable as `displayed_unit_properties` in FigPack (Phase 3) and as summary columns.

## D6 — Browser-first review orchestration {#d6}

The user-facing unit is a **review session**, not a FigPack table selection or an evaluation job. Contracts: [CurationReviewProfile](shared-contracts.md#curationreviewprofile) and [FigPackReview/import](shared-contracts.md#figpackreview-and-import-contracts).

**Start/resume.** `RunResult.start_review(source="analysis"|"root", profile, upload=False, ephemeral=False)` is the primary entry point and resolves the requested `CurationRef`; requesting absent analysis raises an actionable error that names `source="root"` rather than silently changing scientific input. `CurationRef.start_review(...)` is the explicit lower-level form. Both resolve the immutable `CurationReviewProfile`, evaluate/reuse its exact spec, and build/reuse a FigPack selection whose config embeds the parent `curation_uuid` and profile snapshot. Current committed labels seed editable state; evaluation suggestions and applied-merge provenance are non-binding display properties. `FigPackReview.resume(review_id)` rehydrates the handle from the stored selection + figure config and reports derived stage status (no mutable workflow-status column).

**Preview.** `preview_import()` loads the figure config and annotations, verifies identity, canonicalizes labels/merge groups, hashes the canonical annotations payload, and returns a mutation-free `CurationChangeSet`. The change set shows before/after labels, merge groups, unit-count delta, unresolved merge-label conflicts, and any newer sibling curations. Unit IDs absent from the pinned parent fail before preview is returned.

**Commit.** `CurationChangeSet.commit()` re-reads the annotations and parent, verifies `annotations_hash` + `curation_uuid`, then creates/reuses the child through the existing factory. A changed figure yields `ReviewChangedSincePreviewError`; a changed/reused parent yields `CurationNotFoundError`; unresolved label conflicts yield `UnresolvedMergeLabelConflictError`. A zero-change payload requires `confirm_no_changes=True`. Commit is not called inside a caller-owned transaction. It returns a receipt with per-stage created/reused status.

**Merge path.** If merge groups are present, commit applies the gap-correct merge and automatically evaluates the child with the profile's exact spec. The receipt marks `needs_merge_verification=True`. `continue_review()` builds a seeded figure from this committed merged curation's own analyzer, carrying the same profile and delivery configuration. That second browser pass is the scientific verification of the actual merged template/correlogram. It is presented as one continuation action, not a manual repetition of setup.

**Label path.** Import applies the profile's explicit `replace`/`overlay` mode. `replace` compares the complete seeded snapshot and therefore supports deliberate label removal; `overlay` is reserved for deliberately partial imports. The preview makes every clearing/addition visible. Contributor labels that disagree across a merge must be resolved in the edited payload or through `conflict_resolutions=` at commit; no implicit precedence. A label-only edit creates a committed child and needs no analyzer rebuild. The final receipt exposes `.curation.merge_id` directly.

**Collaboration.** A review remains pinned to its original `curation_uuid`. Newer children are reported, not silently selected. Two reviewers can commit sibling children; combining or choosing them is an explicit later action. `created_at`/`created_by` (Phase 4 when available) are shown in the preview/receipt.

**Progress.** Every potentially long step emits/returns named stages (`identity_verified`, `edits_loaded`, `child_committed`, `merged_analyzer_built`, `evaluation_populated`, `verification_view_ready`). Re-invocation derives completed stages from durable rows and resumes idempotently. Exceptions retain the review id and name the restart action.
