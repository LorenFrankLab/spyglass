# Phase 1c - Parent-state curation composition and evaluation acceptance

[back to PLAN.md](PLAN.md) . [overview](overview.md)

**SHOULD after phase-1b, before Phase 5 UX.** Phase-1b adds final metrics over
existing committed `CurationV2` rows. This phase makes editing those states
coherent: child curations compose from the parent curation namespace, labels
inherit predictably, and `CurationEvaluation` outputs can be explicitly accepted
as committed child curations.

This is intentionally split from phase-1b. Tasks 1-4 of phase-1b deliver the
urgent scientific need (final metrics on committed merged units). The work here
is the broader curation-editing model that Phase 5 UI/FigPack should build on.

## Design contract

1. **Child curations edit the parent state.** If a user branches from a merged
   parent, merge groups and label edits are expressed in the parent's unit
   namespace. Raw `Sorting.Unit` ids are not silently resurrected.
2. **Raw-unit provenance remains queryable.** `CurationV2.MergeGroup` continues
   to answer "which original raw units contributed to this kept unit?" Child
   operations that reference parent units must expand through the parent's raw
   provenance.
3. **Immediate parent-operation provenance is separate.** If the UI/audit trail
   needs "which parent units were merged in this child operation?", store that in
   a distinct part table rather than overloading raw-contributor provenance.
4. **Evaluation acceptance is explicit.** Proposed labels and merge suggestions
   become a new committed `CurationV2` only when the user calls an acceptance
   helper with explicit choices.

## Inputs to read first

- [phase-1b-curation-evaluation.md](phase-1b-curation-evaluation.md) - the
  evaluation table and final-metric contract this phase extends.
- [src/spyglass/spikesorting/v2/curation.py](../../../../src/spyglass/spikesorting/v2/curation.py) - `CurationV2.insert_curation`, `create_merged_curation`,
  `propose_merge_curation`, `MergeGroup`, and `UnitLabel`.
- [src/spyglass/spikesorting/v2/_curation_transforms.py](../../../../src/spyglass/spikesorting/v2/_curation_transforms.py) and [_curation_plan.py](../../../../src/spyglass/spikesorting/v2/_curation_plan.py) - current pure row-shaping and label-key validation.
- [src/spyglass/spikesorting/v2/_units_nwb.py](../../../../src/spyglass/spikesorting/v2/_units_nwb.py) - curated units read/write, merge dedup, sample-frame sidecar, and `obs_intervals`.
- [src/spyglass/spikesorting/v2/metric_curation.py](../../../../src/spyglass/spikesorting/v2/metric_curation.py) - `CurationEvaluation` accessors added in phase-1b.

## Tasks

1. **Support committed child curation from a parent curation namespace.** Today
   `CurationV2.insert_curation` validates merge groups against raw
   `Sorting.Unit`, and `MergeGroup.contributor_unit_id` is FK'd to raw
   `Sorting.Unit`. That is not enough once a user edits a merged parent: a valid
   edit may reference a fresh merged unit id that is not in `Sorting.Unit`.

   Add an explicit parent-composition path:
   - for `parent_curation_id == -1`, keep the current root behavior;
   - for `parent_curation_id != -1`, source unit rows and spike trains from the
     parent `CurationV2` row, not from raw `Sorting.Unit`;
   - interpret `merge_groups` in the **parent curation's unit namespace**;
   - assign new merged ids from `max(parent_unit_ids) + 1`, in canonical
     ascending-min order;
   - write the child curated-units NWB from the parent units NWB so stored frames,
     dedup, labels, and `obs_intervals` compose from the actual parent state.

2. **Preserve raw provenance and optionally store parent-operation provenance.**
   Preserve raw-unit auditability by expanding each child unit's raw contributors
   through the parent's existing `CurationV2.MergeGroup` rows before inserting
   the child's raw-provenance rows.

   If immediate parent-operation provenance is needed for audit/UI, add a small
   part table such as:

   ```python
   class ParentMergeGroup(SpyglassMixinPart):
       definition = """
       -> CurationV2.Unit
       parent_unit_id: int
       ---
       """
   ```

   Validate `parent_unit_id` in Python against the parent curation's unit set.
   Do not FK it to `Sorting.Unit`; merged parent ids may not exist there. Do not
   overload `MergeGroup` to mean both raw contributors and parent operation.

3. **Clarify label composition.** Child curations should inherit parent labels by
   default because that matches the user's mental model: "I am editing this
   curation." Add an explicit policy argument rather than relying on `labels=None`
   ambiguity:

   - `label_policy="inherit"` (default for child rows): start with parent labels,
     then apply supplied overrides/edits;
   - `label_policy="replace"`: supplied labels are the full child label state;
   - root curations behave like today's full-state insert.

   For a committed merge, the merged unit should inherit the union of contributor
   labels unless the caller provides an explicit label override for the merged
   unit id. This prevents labels on absorbed contributors from disappearing when
   a user commits a merge.

4. **Add explicit `CurationEvaluation` acceptance helpers.** Add helpers that
   create committed child `CurationV2` rows from phase-1b evaluation outputs:

   ```python
   CurationEvaluation().materialize_labels(key, ...)
   CurationEvaluation().create_curation(
       key,
       merge_groups=None,  # explicit accepted groups; None means labels only
       use_all_suggested_merges=False,
       ...
   )
   ```

   These helpers create **committed** child `CurationV2` rows. They never create
   a preview row with unapplied proposed merges unless the caller opts into an
   explicit draft method with a clear name such as `create_preview_curation`
   (drafting a merge for review is a valid feature, just not the default
   acceptance path). Do not silently apply all suggested merges; require either
   an explicit `merge_groups` argument or `use_all_suggested_merges=True`.

5. **Remove `AnalyzerCuration` (replaced by `CurationEvaluation`).** Once the
   acceptance helpers above exist, `CurationEvaluation` + `create_curation` /
   `materialize_labels` fully replace the `AnalyzerCuration` auto-curate ->
   `materialize_curation` flow. v2 is pre-production with **no
   backwards-compatibility requirement**, so do not deprecate-and-keep it: delete
   it outright (no deprecation warning, no compatibility window).
   - migrate the remaining callers -- the `run_v2_pipeline` orchestrator's
     auto-curation step, the canonical notebook, `docs/src/Features/SpikeSortingV2.md`,
     the visualization facade/helpers (`visualization.py`, `_visualization.py`,
     `_metric_curation_plots.py`), and the pipeline reporting text -- onto
     `CurationEvaluation` with the acceptance helpers;
   - add an explicit `CurationSource.curation_evaluation` value (or a deliberately
     named successor if the enum is renamed) and have accepted evaluation children
     use it instead of the legacy `analyzer_curation` provenance tag. Keep or map
     the old enum value only if an in-tree fixture still needs to read a pre-1c
     row during the same PR;
   - delete `AnalyzerCuration`, `AnalyzerCurationSelection`, and their tests
     (the phase-1 R27 guard tests go with them -- the raw/merged namespace bug
     they covered no longer has a table to occur on; `CurationEvaluation`'s
     `test_curation_evaluation_rejects_preview_curation` and the namespace
     invariant are the surviving guards);
   - keep the shared DB-free helpers `AnalyzerCuration` used
     (`apply_snr_peak_sign`, `apply_label_rules`, `_compute_metrics`,
     `_compute_merge_groups`) -- move any that lived as `AnalyzerCuration`
     staticmethods onto `CurationEvaluation` or a shared module so nothing is
     lost in the deletion.

6. **Docs and changelog.** Update docs/notebooks to teach:
   - evaluate a committed curation with `CurationEvaluation`;
   - accept selected labels/merges into a new committed child;
   - re-evaluate the final child for final metrics;
   - preview curations are draft/legacy behavior, not canonical downstream
     outputs;
   - `CHANGELOG.md`: `AnalyzerCuration` is removed; `CurationEvaluation` is the
     curation-evaluation table.

## Deliberately not in this phase

- **Final merged metrics.** Already delivered by phase-1b.
- **FigPack/manual UI implementation.** This phase supplies the backend editing
  semantics Phase 5 should present.
- **Persistent curation-scoped analyzer cache.** Still deferred; phase-1b's
  evaluation path remains the owner of analyzer construction policy.
- **Removing preview support entirely.** A preview curation (`apply_merge=False`
  with proposed merges) stays a valid *draft / review-before-commit* state -- it
  is a feature, not a backwards-compat artifact. The canonical acceptance helpers
  just must not create preview rows by default (committed children only); the
  explicit `create_preview_curation` opt-in remains for drafting.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_curation_composition.py::test_child_curation_composes_from_merged_parent` (new) | A child label edit created from a merged parent keeps the parent's fresh merged unit id and does not resurrect absorbed raw contributor units. |
| `test_curation_composition.py::test_child_merge_groups_are_parent_namespace` (new) | A child merge can reference a merged parent unit id; the child writes the correct merged spike train and expands raw provenance through the parent's contributors. |
| `test_curation_composition.py::test_parent_operation_provenance_records_parent_units` (new, if `ParentMergeGroup` is added) | Child merge operation provenance records parent unit ids, including fresh merged parent ids that are not in `Sorting.Unit`. |
| `test_curation_composition.py::test_child_labels_inherit_and_merge` (new) | Child curations inherit parent labels by default; a merged child unit receives the union of contributor labels unless explicitly overridden; `label_policy="replace"` clears inherited labels. |
| `test_curation_evaluation.py::test_evaluation_materialize_creates_committed_child` (new) | `CurationEvaluation().create_curation(..., merge_groups=accepted)` creates a child with `assert_committed_curation` true; it does not create a preview row. |
| `test_curation_evaluation.py::test_evaluation_acceptance_requires_explicit_merge_choice` (new) | Suggested merge groups are not silently applied unless `merge_groups` is passed or `use_all_suggested_merges=True`. |
| `test_curation_evaluation.py::test_evaluation_child_source_is_curation_evaluation` (new) | Accepted evaluation children use the new `curation_source` provenance value, not the legacy `analyzer_curation` tag. |
| `test_pipeline_facade.py::test_run_v2_pipeline_auto_curation_uses_curation_evaluation` (new/adapted) | After the removal, `run_v2_pipeline`'s auto-curation step produces a `CurationEvaluation` (and committed child via the acceptance helper); importing `AnalyzerCuration` / `AnalyzerCurationSelection` fails (table deleted, no lingering callers). |
| `test_visualization_facade.py::test_visualization_uses_curation_evaluation_accessors` (new/adapted) | Plot/facade helpers read metrics and proposed merge groups from `CurationEvaluation`; no visualization path imports `AnalyzerCuration`. |
| (regression) existing curation merge/dedup/disjoint-interval tests | Stored frames, cross-gap dedup, `obs_intervals`, and lazy/applied merge parity remain correct after the parent-composition refactor. |
| (regression) phase-1b `test_final_metrics_recomputed_for_merged_unit` | Final metrics still compute over the final child after acceptance. |

## Fixtures

- Parent-composition tests should include at least one already-merged parent
  whose fresh unit id is not present in `Sorting.Unit`; otherwise the test is
  vacuous.
- Acceptance tests can reuse a small `CurationEvaluation` row with deterministic
  proposed labels/merge groups, avoiding a full sorter rerun where possible.
- Merge/dedup tests should reuse the existing disjoint-interval and two-unit
  fixtures so parent-state writes keep the existing spike-conservation behavior.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Child-curation edits use the parent curation namespace; merged parent unit ids
  are valid inputs.
- Raw contributors remain queryable and are expanded through parent provenance.
- Parent-operation provenance, if added, is stored separately from raw
  contributor provenance.
- Labels do not disappear silently when a merge is committed from a labeled
  parent.
- Evaluation acceptance helpers create committed child curations and never apply
  all suggested merges without explicit user intent.
- Phase-1b evaluation still works unchanged over the final child.
- `AnalyzerCuration` / `AnalyzerCurationSelection` are deleted (no deprecation
  shim), no caller imports them (including docs, plotting helpers, pipeline
  reporting, and tests), accepted evaluation children carry the new curation-source
  provenance, and the shared compute/label helpers they used survive on
  `CurationEvaluation` or a shared module.
