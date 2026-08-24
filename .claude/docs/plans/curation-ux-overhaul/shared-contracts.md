# Shared Contracts

[← back to PLAN.md](PLAN.md)

Cross-phase types and signatures. Each appears here once; phases link in by anchor. **Do not weaken** the invariants noted per contract.

Index:
- [CurationRef](#curationref) — the one curation handle (Phase 2, used by 3 & 4)
- [RunResult accessors](#runresult-accessors) — `run.root_curation` / `run.analysis_curation` (Phase 2)
- [EvaluationResult](#evaluationresult) — evaluation snapshot (Phase 2)
- [CurationReviewProfile](#curationreviewprofile) — persisted, lab-approved review configuration (Phase 0)
- [FigPackReview and import contracts](#figpackreview-and-import-contracts) — browser-first review, diff, commit, continuation (Phase 3)
- [get_curation_analyzer](#get_curation_analyzer) — the resolver (Phase 1, used by 3)
- [Analyzer cache manifest](#analyzer-cache-manifest) — regeneratable-cache validation (Phase 1)
- [UnsupportedMetricValueError](#unsupportedmetricvalueerror) — strict metric errors (Phase 2)
- [Figure identity](#figure-identity) — FigPack round-trip safety (Phase 3)

---

## CurationRef

Phase 2 introduces one immutable handle for "which curation," replacing hand-built `{"sorting_id": ..., "curation_id": ...}` dicts and the `root_curation_id`→`curation_id` re-keying footgun.

```python
@dataclass(frozen=True)
class CurationRef:
    sorting_id: UUID
    curation_id: int
    curation_uuid: UUID              # the IDENTITY of record (Phase 0); (sorting_id, curation_id) is only the DJ key

    def as_key(self) -> dict:           # bridge to the table/expert layer
        return {"sorting_id": self.sorting_id, "curation_id": self.curation_id}

    @property
    def merge_id(self) -> UUID: ...      # resolves via SpikeSortingOutput; the downstream handle
```

Invariants (do not weaken):
- **`(sorting_id, curation_id)` is NOT a permanent identity.** `_next_curation_id` is `max(existing)+1` and reuses a numeric id after a delete (`curation.py:1062-1072`), so a stale ref could otherwise resolve to a *different, newly-created* curation. `curation_uuid` (immutable, fresh per insert — Phase 0) is the identity carried and verified everywhere; `from_key(...)` resolves the uuid at construction and every consuming operation re-checks it, raising `CurationNotFoundError` if the row is gone **or its uuid no longer matches** (a reused id → a different generation → treated as not-found). Do not claim "never dangles."
- `.merge_id` resolves the **curated** merge id for this exact curation — never silently the root's. Zero-unit and preview curations resolve per the existing `get_spike_times` guards, not by fabricating an id.
- Non-root facade APIs (see Phase 2) take a `parent_curation: CurationRef`; only initial-creation accepts the root sentinel.

## RunResult accessors

`run_v2_pipeline` / `run_v2_pipeline_session` return value gains attribute accessors so the caller never re-keys the summary. The underlying `RunV2SingleSessionSummary` fields (`_pipeline_types.py:115-137`) are preserved; the wrapper adds:

```python
run.root_curation      -> CurationRef        # from (sorting_id, root_curation_id)
run.analysis_curation  -> CurationRef | None  # from (sorting_id, analysis_curation_id); None until curated
run.sorting_id         -> UUID                # unchanged passthrough
```

Invariant: `run.analysis_curation` is `None` (not the root) when the run did not auto-curate — preserving the existing "no bare merge_id, analysis handle is explicit" safety. Item access (`run["sorting_id"]`) remains supported during the notebook migration; the object is the documented surface.

## EvaluationResult

**Snapshot** value object returned by `curation.evaluate(...)` (a point-in-time read, not a deeply-immutable object — `frozen=True` freezes the *bindings*, but `metrics`/`suggested_merges`/`proposed_labels` are a `DataFrame`/`list`/`dict` and remain mutable). Either hand out defensive copies / read-only views (a copied frame, `MappingProxyType`, tuples) or document it explicitly as a snapshot; do not claim immutability. Carries everything the curator reads, so no manual metric-dict transfer.

```python
@dataclass(frozen=True)
class EvaluationSpec:                          # the exact recipe names, carried forward
    metric_params_name: str
    auto_curation_rules_name: str

@dataclass(frozen=True)
class EvaluationResult:
    curation: CurationRef                     # the curation that was evaluated
    spec: EvaluationSpec                       # so merge_and_evaluate reuses the SAME criteria
    evaluation_id: UUID
    metrics: DataFrame                         # per-unit, in this curation's own unit namespace
    suggested_merges: list[list[int]]
    proposed_labels: dict[int, list[str]]
    warnings: tuple[str, ...]
    plots: "EvaluationPlots"                   # accessor object; see Phase 1 routing

    def preview_merges(self, groups) -> CurationRef: ...
    def merge_and_evaluate(self, groups) -> "MergeEvaluateReceipt": ...   # PAVED ROAD: commit merges + re-evaluate w/ self.spec
    def accept_labels(self, mode: Literal["replace", "overlay"]) -> CurationRef: ...
```

**API shape (do not reintroduce the contradictions this resolves, B3).** `curation.evaluate(metric_params_name=…, auto_curation_rules_name=…)` returns an `EvaluationResult` carrying an immutable `EvaluationSpec` (the two names — there is **no** generic evaluation-preset registry; `CurationEvaluationSelection.insert_by_curation_id(sorting_id, curation_id, metric_params_name, auto_curation_rules_name)` takes the two names directly, `metric_curation.py:890` — there is no `from_names` method). The **scripted** merge path is `evaluation.merge_and_evaluate(groups)` — it commits the merges and re-evaluates the result **using `self.spec`**, returning a `MergeEvaluateReceipt` whose `.evaluation` is the post-merge result. The browser-first path resolves those two names through the real persisted `CurationReviewProfile`. A separate expert `commit_merges(groups) -> CurationRef` exists for committing without re-evaluation; do not overload one `accept_merges` to mean both.

Invariants:
- `metrics` is in the evaluated curation's **own** unit namespace (a merged unit's metrics are over its merged template, never inherited from a contributor).
- `accept_labels(mode="replace")` delegates to `use_evaluation_labels` (clears un-proposed labels); `mode="overlay"` delegates to `overlay_evaluation_labels` (keeps existing). The mode is the single visible choice; the two table methods stay as the expert layer.
- `.plots.*` route through the resolver and therefore render **merged** curations (the Phase 1 gap closure) — via controlled helpers, never by handing out the cache-backed analyzer (see [get_curation_analyzer](#get_curation_analyzer)).

## CurationReviewProfile

The browser-first paved road uses one persisted profile name rather than making a researcher recall two independently versioned parameter names plus display and label configuration.

```python
@dataclass(frozen=True)
class ReviewProfileRef:
    review_profile_name: str
    profile_hash: str
    evaluation_spec: EvaluationSpec
    displayed_unit_properties: tuple[str, ...]
    label_options: tuple[str, ...]
    label_import_mode: Literal["replace", "overlay"]  # overlay maps to the expert layer's inherit policy
```

`CurationReviewProfile` is an immutable DataJoint lookup added in Phase 0. A changed metric recipe, rule set, property order, label palette, or import mode requires a new profile name/hash. The resolved values are embedded into each figure/review receipt, so reopening a review never depends on a mutable alias. Runtime delivery choices (`upload`, `ephemeral`, credentials, local destination) are intentionally not profile fields.

## FigPackReview and import contracts

Phase 3 makes the browser review the primary hands-on workflow. `RunResult.start_review(source="analysis"|"root", profile=...)` is the one top-level entry point; it never silently substitutes root when analysis was requested. `CurationRef.start_review(profile=...)` is the explicit lower-level form. Both evaluate or reuse the profile's exact `EvaluationSpec`, build a seeded FigPack view, and return a resumable handle. `EvaluationResult.start_review(...)` is also available and verifies that its spec equals the chosen profile. The scripted `evaluate`/`merge_and_evaluate` path remains supported but is secondary.

```python
@dataclass(frozen=True)
class FigPackReview:
    review_id: UUID                     # existing content-addressed FigPack selection id
    parent: CurationRef                 # includes curation_uuid
    profile: ReviewProfileRef
    evaluation: EvaluationResult
    uri: str
    upload: bool
    ephemeral: bool

    def open(self) -> None: ...
    def preview_import(self) -> "CurationChangeSet": ...

    @classmethod
    def resume(cls, review_id: UUID) -> "FigPackReview": ...

@dataclass(frozen=True)
class CurationChangeSet:
    review: FigPackReview
    annotations_hash: str               # exact bytes/logical payload previewed
    labels_before: Mapping[int, tuple[str, ...]]
    labels_after: Mapping[int, tuple[str, ...]]
    merge_groups: tuple[tuple[int, ...], ...]
    unit_count_before: int
    unit_count_after: int
    label_conflicts: tuple["MergeLabelConflict", ...]
    newer_sibling_curations: tuple[CurationRef, ...]

    def commit(
        self,
        *,
        conflict_resolutions: Mapping[int, tuple[str, ...]] | None = None,
        confirm_no_changes: bool = False,
    ) -> "ReviewImportReceipt": ...

@dataclass(frozen=True)
class ReviewImportReceipt:
    curation: CurationRef
    evaluation: EvaluationResult | None
    changes: CurationChangeSet
    warnings: tuple[str, ...]
    stages: tuple["StageStatus", ...]
    needs_merge_verification: bool

    def continue_review(self) -> FigPackReview: ...
```

Invariants (do not weaken):
- **Preview is pure.** `preview_import()` reads and validates the figure, computes the exact label/merge diff, resulting unit count, conflicts, and newer siblings, and performs no DB/file mutation.
- **Snapshots do not leak mutation.** Collections/DataFrames on review, change-set, and receipt objects are tuples, mapping proxies, or defensive copies; `frozen=True` alone is not presented as deep immutability.
- **Commit is TOCTOU-safe.** `commit()` re-reads the figure and target, verifies the same `curation_uuid` and `annotations_hash` that were previewed, and refuses if either changed. It creates/reuses an immutable child; it never overwrites or silently rebases onto a newer child.
- **Merge-label conflicts are explicit.** If contributors carry incompatible labels and the edited payload does not unambiguously resolve the resulting unit's labels, commit raises an actionable `UnresolvedMergeLabelConflictError`. No arbitrary contributor wins.
- **Suggestions are not edits.** Existing committed labels seed the editable annotation state. Evaluation-proposed labels/merges are shown as clearly named read-only properties/overlays until the researcher explicitly applies them in FigPack; merely opening and importing a review cannot accept every proposal. Already-applied merge groups are read-only provenance in a merged view, never seeded back as pending `mergeGroups`.
- **No-change review is explicit.** A zero-diff commit is refused unless `confirm_no_changes=True`; when confirmed, it records an explicit reviewed/no-change child/receipt rather than pretending an edit occurred.
- **Merge imports re-evaluate.** A committed merge is automatically evaluated with the review profile's exact `EvaluationSpec`; the receipt carries that post-merge result and sets `needs_merge_verification=True`.
- **Continuation is not a restart.** `continue_review()` builds/reuses a view over the actual merged analyzer and carries forward the immutable profile, displayed properties, label palette, delivery mode, and provenance. Completed stages are visible, and `FigPackReview.resume(review_id)` reconstructs the handle after a notebook/process restart.
- **Collaboration branches.** Concurrent reviewers produce sibling children from the pinned parent. Preview reports newer siblings; the system never silently combines reviews or changes the parent.
- **User code never supplies a parent key on import.** Parent identity comes from the verified figure/review handle. The separately named legacy import remains the only asserted-parent escape hatch.

## get_curation_analyzer

The one analyzer resolver, used by evaluation, plotting, and FigPack. Phase 1 owns it; Phase 3 consumes it.

```python
# INTERNAL resolver — not a public "hand me the analyzer" API. Plots go through controlled helpers.
def _resolve_curation_analyzer(
    curation_ref: CurationRef,       # identity carried by curation_uuid, not just (sorting_id, curation_id)
    waveform_recipe: str,            # named recipe; hashed into cache identity
    role: Literal["display", "metric"] = "display",
) -> "si.SortingAnalyzer": ...       # the CACHE-BACKED object; callers must NOT mutate it (ownership rule)

@contextmanager
def curation_analyzer_with_extensions(   # temporary, context-managed derivative for an unusual plot
    curation_ref, waveform_recipe, role="display", *, extra_extensions: "Mapping[str, dict]",
) -> "Iterator[si.SortingAnalyzer]": ...  # cleaned up on exit; never persisted alongside the published cache
```

**Read-only by OWNERSHIP, not by SI type (B4).** SI 0.104.3's `load_sorting_analyzer` has no read-only flag, and a `SortingAnalyzer` exposes `compute(save=True)` / `delete_extension` (verified) — so "read-only" cannot be a type guarantee, only a convention unless enforced structurally. Therefore: the persistent resolver is **internal**; the published cache is treated as immutable **by ownership** (no supported path mutates it); all supported plots route through controlled helpers that never hand the cache-backed analyzer to user code; a caller needing direct analyzer access gets a **detached memory copy** via `save_as(format="memory")`; and a plot needing an extension outside the standard display set uses the **context-managed, temporary** `curation_analyzer_with_extensions` (cleaned up on exit — staged derivatives are temporary, not a second persistent cache). Every supported helper must leave the published folder **byte-identical** (a test asserts this).

Cache identity is **per-curation, keyed by `curation_uuid`** (Phase 0 / [F1](designs.md#d1)): `(sorting_id, curation_uuid, role, waveform_recipe_hash, spikeinterface_version)`. `curation_uuid` — not the reusable `curation_id` — is what makes the path stable across a delete+recreate. No cross-curation content dedup.

Routing contract (do not weaken):
- **Raw-namespace curation** (root or label-only over raw) → reuse the existing cached sorting analyzer (`_analyzer_cache.py` fast path). No rebuild.
- **Committed merged curation** → build-or-reuse a per-curation analyzer over the merged spike trains, published read-only under the per-curation identity above.
- **Preview / uncommitted-merge curation** → raise the existing actionable "commit the merge first" error (a preview has no defined final namespace).
- **Zero-unit curation** → raise a typed `ZeroUnitAnalyzerError` that plot helpers translate to an empty-state view. (This is the decided behavior — not "pick one".)
- `role="display"` → unwhitened, retained on disk (plots + FigPack read it). `role="metric"` → whitened/metric preprocessing; retention per [Open Question 2](overview.md#open-questions).

**Single-low-level-builder invariant (do not weaken).** The unit that must not drift is the low-level `build_analyzer` in `_sorting_analyzer` — the recipe-driven extension/preprocessing build that is *already* the single builder both `CurationEvaluation.make_compute` (DB-free parallel worker, `metric_curation.py:1005`, calling `build_analyzer` at `:1406`/`:1419`) and this resolver call. Phase 1 does **not** route `make_compute` through this resolver: `make_compute` is a DB-free spawn worker (`_parallel_make = True`, resolves no DB inputs, `metric_curation.py:999-1005`) and must stay so — routing it through the DB-backed resolver would reintroduce the DB dependency the tri-part contract forbids (guarded by `test_tripart_dispatch_active`). Instead, both paths funnel through the same `build_analyzer`; `make_compute` keeps building ephemerally for its populate, and this resolver adds identity-keyed **persistence + locking** on top of the same builder for the interactive path. "One builder" — not "one caller."

## Analyzer cache manifest

Every cached curation-scoped analyzer folder carries a manifest; load validates it and rebuilds on any mismatch. The cache is **regeneratable**, never provenance of record.

```python
@dataclass(frozen=True)
class CurationAnalyzerManifest:
    sorting_id: UUID
    curation_uuid: UUID                       # the stable identity (Phase 0) — the id is reusable, the uuid is not
    curation_id: int                          # DJ key only (informational)
    role: str
    curated_unit_ids: tuple[int, ...]         # exact merged unit set
    contributor_map: dict[int, list[int]]     # merged unit_id -> source unit_ids
    merged_spike_content_hash: str            # hash of the exact merged spike-sample frames (F2)
    merge_policy_version: str                 # Spyglass gap-correct dedup impl version (F2/F6)
    source_artifact_hashes: dict[str, str]    # upstream sorting/recording inputs
    waveform_recipe_hash: str
    spikeinterface_version: str
    extension_inventory: dict[str, str]       # extension name -> param hash (F3 crash-safety)
```

Validation rule: a load is valid iff every field matches what the current curation + recipe + environment recompute to. Any mismatch (recipe changed, SI upgraded, unit set drifted, spike content or merge-policy version changed, artifact rehashed) → treat as stale → rebuild. **Crash-safety (F3):** the `extension_inventory` records each computed extension and its param hash; a folder whose zarr does not contain exactly the inventoried, complete extensions (e.g. a process died mid extension-write) is treated as corrupt → rebuild. Partial/corrupt folders (missing manifest, incomplete zarr) → rebuild. Never surface a stale or half-written cache as valid. The manifest validates the *cache*; it is not provenance of record (the `CurationEvaluation` row remains that).

## UnsupportedMetricValueError

Phase 2 replaces the silent non-scalar→NaN coercion on the **metric write path** — `_scalar_or_nan` at `_metric_curation_nwb.py:53-78` (NOT the read-side threshold filter `_is_finite_metric_value` at `_metric_curation.py:38-53`, which is a separate decision, see [D3](designs.md#d3)) — with a domain error, default-on.

```python
class UnsupportedMetricValueError(ValueError):
    """A quality/template metric produced a non-scalar value where a scalar was required."""
    # message names: metric, unit_id, expected ("scalar numeric"), received (repr incl. shape/dtype)
```

Boundary (do not weaken): a **legitimate numeric `NaN`** (a low-spike unit's metric) is still valid and passes through. The check must be an **explicit scalar test** — verify 0-dim / numeric dtype, not a bare `float(value)` (a one-element array coerces cleanly and would slip through). Only a genuinely non-scalar/object value raises.

**Strict is unconditional (do not add an opt-in `strict=False`).** This is preproduction and coercion silently changes auto-curation outcomes; a coerce-and-warn escape hatch is an unrecorded serializer switch that alters scientific output. If a compatibility mode is ever genuinely required, it must be part of the **evaluation recipe identity and persisted provenance**, not a hidden flag.

Scope note (do not overstate): this catches a metric column that *drifted to a non-scalar dtype*. It does **not** catch an all-`NaN` scalar column (the `nn_noise_overlap`-always-NaN class) — a scalar float passes the scalar check. That defect belongs to a **separate rule-input validation** (below), not this error.

### Rule-input validation (the correct home for the all-NaN class)

An auto-curation rule that references a metric column with **no finite values** is governed by a **persisted `missing_policy`** on `AutoCurationRules.Rule` (added in [Phase 0](phase-0-schema-foundation.md) — the current `Rule` schema has no such field and is `extra="forbid"`, so it must be a real column). Default `'error'` (preproduction): **fail-fast** with an actionable error. This is what surfaces the `nn_noise_overlap`-always-NaN defect (a rule silently never firing), which the scalar `UnsupportedMetricValueError` cannot.

Do **not** conflate with SI: SpikeInterface's `threshold_metrics_label_units(nan_policy='fail')` *labels* a NaN unit as failed and never raises. Spyglass's `'error'` is a distinct fail-fast policy. Modes: `'error'` (raise), `'fail'`/`'pass'` (label the unit fail/pass), `'ignore'` (skip the rule for that unit). SI is inspiration, not the semantics.

## Figure identity

Phase 3 embeds identity in every FigPack figure and verifies it on import, so annotations never attach to the wrong curation.

```json
{
  "sorting_id": "…",
  "curation_uuid": "…",           // the identity of record (Phase 0)
  "curation_id": 4,                // DJ key, informational
  "figpack_config_hash": "…"       // existing figpack_config_hash (figpack_curation.py:42)
}
```

Rule (do not weaken): `save_curation_from_uri(uri, ...)` recomputes identity from the target curation and **refuses** a figure whose embedded `curation_uuid` disagrees — it does not trust a caller-supplied parent key, and it does **not** rely on `(sorting_id, curation_id)` (that pair is reusable: `curation_id` is `max+1` and reused after a delete, `curation.py:1062-1072`, so an old figure could otherwise pass against an unrelated replacement curation — B1). `curation_uuid` is immutable per generation, so a figure only ever verifies against the exact curation it was built from.

Backward-compat — **fail-closed** (do not weaken): the verified direct-import primitive `save_curation_from_uri` has **no** `allow_legacy` parameter, and the primary `FigPackReview` path never asks the user for an asserted parent. A figure without embedded identity is refused. Unsafe import of an identity-less figure lives behind a separately-named, explicit operation:

```python
import_legacy_figpack_curation(uri, asserted_parent=..., confirm_unverified_identity=True)
```

Migration compatibility is not a concern here (preproduction), so the paved road does not carry the risk (see [D4](designs.md#d4)).
