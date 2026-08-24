# Phase 2 — Coherent curation API, strict errors, lifecycle

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design D2](designs.md#d2) · [design D3](designs.md#d3)

Builds the value-object and scripted foundation consumed by Phase 3's browser-first paved road. A curator or automation author can work with `CurationRef` and `EvaluationResult` instead of hand-built keys and five near-identical "apply the evaluation" calls. The scripted `merge_and_evaluate` loop remains a supported secondary path; Phase 3's `start_review` session is the primary hands-on workflow. This phase also makes drifted metrics fail loud and exposes lifecycle helpers without new schema.

**Inputs to read first:**

- [_pipeline_types.py:115-137](../../../../src/spyglass/spikesorting/v2/_pipeline_types.py) — `RunV2SingleSessionSummary` fields the run wrapper exposes as `CurationRef`s.
- [metric_curation.py:1808](../../../../src/spyglass/spikesorting/v2/metric_curation.py) `accept_evaluation_outputs`, `:2023` `use_evaluation_labels`, `:2056` `overlay_evaluation_labels`, `:1928` `preview_merges`, `:1960` `accept_merges`, `:1994` `accept_all_suggested_merges` — the expert methods the facade re-expresses (kept, not deleted).
- [curation.py:417](../../../../src/spyglass/spikesorting/v2/curation.py) `insert_curation` (incl. `reuse_existing:426`, non-default-params guard `:492-499`), `:1419` `create_initial_curation`, `:1550` `create_merged_curation`, `:1618` `save_manual_curation`, `:1737` `summarize_curation`, `:326` `delete`+orphan guard, `:388` `audit_orphaned_lineage`, `:2008` `has_unapplied_proposed_merges`, `:2044` `is_committed_curation`.
- [curation.py:78-104](../../../../src/spyglass/spikesorting/v2/curation.py) — the existing `CurationV2` columns (`curation_source` enum, `merges_applied`, `parent_curation_id`) that `state`/`operation_type` derive from.
- [_metric_curation_nwb.py:53-78](../../../../src/spyglass/spikesorting/v2/_metric_curation_nwb.py) — `_scalar_or_nan`, the metric **write-path** coercion to replace with a raise. Legitimate scalar-float NaN keeps its `float(value)` success path. NOT `_metric_curation.py:38-53` (`_is_finite_metric_value`, a read-side threshold filter — separate decision).
- [metric_curation.py:890-911](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — `CurationEvaluationSelection.insert_by_curation_id(sorting_id, curation_id, metric_params_name, auto_curation_rules_name)` takes the two names directly. There is **no** `from_names` method (verified absent); do not reference it. No "evaluation preset" registry for the facade to assume.
- [utils.py:107-114](../../../../src/spyglass/spikesorting/v2/utils.py) — `transaction_or_noop` silently nests inside an outer transaction; `merge_and_evaluate` must add its own `in_transaction` guard (the `sorting.py:903-905` guard is artifact-specific, not reusable).

**Contracts referenced:**

- [CurationRef](shared-contracts.md#curationref), [RunResult accessors](shared-contracts.md#runresult-accessors), [EvaluationResult](shared-contracts.md#evaluationresult) — the value objects; **do not weaken** the "non-root APIs require a `parent_curation`" invariant.
- [UnsupportedMetricValueError](shared-contracts.md#unsupportedmetricvalueerror) — the strict boundary (legitimate NaN still passes).

**Designs referenced:** [D2](designs.md#d2) (`merge_and_evaluate`), [D3](designs.md#d3) (strict metrics).

## Tasks

- Add the value objects `CurationRef` / `EvaluationResult` (and the `MergedCuration`/`MergeEvaluateReceipt` returns) in a new `src/spyglass/spikesorting/v2/curation_api.py` (public facade module). `CurationRef.from_key(...)` validates existence at construction, but every consuming operation **re-checks** and raises a typed `CurationNotFoundError` on a since-deleted row (do NOT claim "never dangles"). `EvaluationResult` is a **snapshot**, not deeply immutable: hand out copies / read-only views of its `metrics`/`suggested_merges`/`proposed_labels`, or document the snapshot semantics — do not claim immutability of the mutable fields. `.merge_id` resolves the curated merge id via `SpikeSortingOutput`.
- Reserve the Phase-3 extension points `RunResult.start_review(source="analysis"|"root", ...)`, `CurationRef.start_review(...)`, and `EvaluationResult.start_review(...)` in the public design/docs, but do not implement FigPack imports here. The run-level method never silently falls back to root. These consume the persisted `CurationReviewProfile` and return the shared `FigPackReview` contract once Phase 3 lands, keeping the final facade coherent without pulling the optional browser dependency into Phase 2.
- Wrap the `run_v2_pipeline` / `run_v2_pipeline_session` return so `run.root_curation` / `run.analysis_curation` return `CurationRef` (or `None`), keeping item access working during migration. Kills the `root_curation_id`→`curation_id` re-keying footgun.
- Facade methods (reconciled API, B3): `curation.evaluate(metric_params_name=…, auto_curation_rules_name=…) -> EvaluationResult` carrying an immutable `EvaluationSpec`. On `EvaluationResult`: `preview_merges(groups)`; **`merge_and_evaluate(groups)`** (the scripted path — commits and re-evaluates with `self.spec`); and `accept_labels(mode="replace"|"overlay")`. Keep separate expert `commit_merges(groups) -> CurationRef`; do not overload `accept_merges`. Phase 3's persisted `CurationReviewProfile` removes the two-name recall burden from the primary browser path.
- Implement `EvaluationResult.merge_and_evaluate(groups)` per [D2](designs.md#d2): **first** raise if `connection.in_transaction` (M6 — NOT already enforced; `transaction_or_noop` silently nests, `utils.py:107-114`), validate ids against `self.curation`'s namespace, commit-or-reuse the merged child, create-or-reuse the evaluation selection **with `self.spec`'s two names**, populate, return the receipt (`CurationRef.from_key(child)` + post-merge `EvaluationResult` + warnings + per-stage status). Idempotent and resumable. There is **no** `evaluation_preset` argument and **no** no-arg `merged.evaluate()`.
- Change non-root facade APIs to require `parent_curation: CurationRef`; keep the root sentinel only on initial-creation. This is a breaking signature change — migrate all in-repo callers and the notebooks in this PR.
- Strict metric errors per [D3](designs.md#d3): replace the coercion in `_scalar_or_nan` at `_metric_curation_nwb.py:53-78` (the **write path**) with `UnsupportedMetricValueError` — **unconditional, no `strict=False` flag** (a hidden coercion switch silently changes scientific output; preproduction). Use an **explicit scalar check** (0-dim + numeric dtype), NOT a bare `float(value)` (a one-element array coerces). Do NOT touch `_is_finite_metric_value` (`_metric_curation.py:38-53`, the read-side filter) — a separate explicit decision.
- **Rule-input validation (reads the Phase 0 `missing_policy`):** an auto-curation rule referencing a metric column with no finite values is governed by the rule's persisted `missing_policy` (added in [Phase 0](phase-0-schema-foundation.md); the `Rule` table had no such field). Default `'error'` → fail-fast with an actionable error. This — not the scalar `UnsupportedMetricValueError` — surfaces the `nn_noise_overlap`-always-NaN defect. **Distinct from SI:** SI's `nan_policy='fail'` *labels* the unit failed and never raises; Spyglass's `'error'` raises (`'fail'`/`'pass'` label, `'ignore'` skips). Depends on Phase 0.
- Schema-free lifecycle helpers on `CurationRef`/`CurationV2`, all derived from existing columns (**no new columns** — `created_at`/`created_by` are Phase 4):
  - Expose **orthogonal** properties, NOT a single conflated `state` enum (F4): `commit_status ∈ {preview, committed}` (from `has_unapplied_proposed_merges`/`is_committed_curation`, `:2008`/`:2044`), `is_root` (`parent_curation_id == -1`), `is_leaf`, `has_committed_children`. **Do NOT expose a derived `superseded`** — child existence alone cannot establish supersession in a branching graph (a root with a committed child is not "superseded"); real supersession needs an explicit preferred/head/supersedes relationship, which requires schema — defer it (a Phase 4 head-pointer, or out of scope).
  - `operation_type` derived from the **actual merge rows + label delta**, not `merges_applied` alone (a preview merge has `merges_applied=False`, like a label-only child): report producer (`curation_source`) and change-kind (label vs merge, read from the merge rows). If a single normalized action value is wanted, persist it rather than infer it ambiguously.
  - Lineage listing/visualization over `parent_curation_id`; `preview_curation_delete()`; supported subtree deletion (leaf-up, reusing the `delete` orphan guard); orphan-lineage + analyzer-cache health report (composing `audit_orphaned_lineage:388` with the Phase 1 orphan audit).
- Docs: document the scripted alternative — run → `CurationRef` → `evaluate` → `merge_and_evaluate` → merged plots → `accept_labels` → `merge_id` — for automation/debugging. Do **not** make it the final canonical hands-on tutorial; Phase 3 rewrites that around the browser-first review. Keep an expert appendix for table methods, never `insert_selection`/`populate` as the user default.

## Deliberately not in this phase

- `created_at`/`created_by` columns and typed annotation tables — Phase 4 (schema).
- FigPack facade wiring — Phase 3.
- Any change to the analyzer/plot internals — those are Phase 1; Phase 2 only surfaces them via `EvaluationResult.plots`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_run_result_curation_refs` | `run.root_curation` / `run.analysis_curation` return correct `CurationRef`s; `analysis_curation` is `None` before curation (not the root). |
| `test_curation_ref_merge_id_is_curated` | `CurationRef.merge_id` for a curated child resolves the child's merge id, never the root's. |
| `test_accept_labels_modes` | `mode="replace"` clears un-proposed labels; `mode="overlay"` keeps existing — matching the two expert methods. |
| `test_merge_and_evaluate_idempotent` | Re-running with the same args reuses the child + selection and does not duplicate rows; receipt reports reused stages. |
| `test_merge_and_evaluate_rejects_bad_ids` | A merge group naming a unit absent from the parent namespace raises before any write. |
| `test_merge_and_evaluate_outside_txn` | Called inside an open transaction, it raises the actionable "populate manages its own transaction" error rather than deadlocking. |
| `test_merge_and_evaluate_reuses_spec` | `EvaluationResult.merge_and_evaluate(groups)` re-evaluates the merged child with the SAME `spec` (recipe names) — no `evaluation_preset` arg, no no-arg `evaluate()`; `receipt.evaluation` is the post-merge result. |
| `test_commit_merges_no_reeval` | `commit_merges(groups)` commits without re-evaluating (a distinct expert method); `accept_merges` is not overloaded to mean both. |
| `test_strict_metric_error` | A non-scalar metric value raises `UnsupportedMetricValueError` naming metric+unit+shape; a legitimate 0-dim float NaN passes; a **one-element array** is rejected (not silently coerced). No `strict=False` path exists. |
| `test_rule_input_all_nan_fails` | An auto-curation rule referencing an all-NaN metric column fails with an actionable error unless it declares a missing-value policy (surfaces the `nn_noise_overlap` class). |
| `test_state_orthogonal_booleans` | `commit_status`/`is_root`/`is_leaf`/`has_committed_children` compute correctly; **no derived `superseded`** is exposed (a root with a committed child is not "superseded"). No new columns. |
| `test_operation_type_from_merge_rows` | `operation_type` reports producer + change-kind read from the actual merge rows/label delta — a preview merge (`merges_applied=False`) is distinguished from a label-only child (F4/M4). |
| `test_curation_ref_not_found` | Consuming a `CurationRef` whose row was deleted after construction raises `CurationNotFoundError`, not a silent wrong result. |
| `test_evaluation_result_snapshot` | Mutating a returned `EvaluationResult.metrics`/`proposed_labels` does not mutate stored state (copies/read-only views), matching the documented snapshot semantics. |
| `test_strict_metric_leaves_read_filter` | The strict write-path raise does not change `_is_finite_metric_value` behavior (read-side threshold filter still warn-and-skips). |
| `test_subtree_delete_leaf_up` | Subtree deletion removes children before parents and never orphans lineage; a childless curation deletes freely. |
| `test_non_root_requires_parent` | A non-root facade call without `parent_curation` raises; initial-creation accepts the root sentinel. |
| `test_scripted_journey_smoke` *(slow)* | The secondary scripted journey runs end-to-end on the fixture without a raw `insert_selection`/`populate` call; Phase 3 owns the canonical browser journey test. |

## Fixtures

Reuse the Phase 1 fixture sort + merge. Add a small tree of curations (root → labeled child → merged grandchild) in `conftest.py` for the lifecycle/state/subtree-delete tests.

## Review

Dispatch `code-reviewer` against the diff. Confirm: facade delegates to (does not duplicate) expert methods; the breaking `parent_curation` change is fully migrated; strict errors preserve legitimate NaN; lifecycle helpers add no columns; scripted docs do not masquerade as the canonical browser workflow; expert appendix present; no plan references in shipped code/docstrings.
