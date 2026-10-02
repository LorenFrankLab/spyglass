# Spike Sorting v2 Module Split Implementation Plan

**Status:** Not started. Split design to be discussed with the owner before
execution (see "Open questions").

**Goal:** Make the four largest v2 table modules readable by moving long method
bodies and private helpers out of the table classes, with no change to behavior,
schemas, or public import paths.

**Architecture:** The size is concentrated in five classes, not spread across
many tables: `CurationV2` (2,958 lines, essentially all of `curation.py`),
`CurationEvaluation` (2,009), `Sorting` (1,919), `Recording` (1,121) and
`SortingSelection` (846). Moving whole tables into new files would leave those
classes as large as they are, so the split keeps every table class where it is
and thins the class bodies instead: public methods stay on the class (they are
the documented API) and delegate to functions in private `_*` modules; private
helpers move out unless tests patch them. Pure computation goes into the
existing pure service modules; code that queries the database or stages files
goes into service modules that import the database layer lazily. This is the
pattern v2 already uses (`_sorting_dispatch`, `_curation_transforms`,
`_recording_nwb`, ...).

**Tech stack:** DataJoint 0.14.9 (`@schema` tables, tri-part `make_fetch` /
`make_compute` / `make_insert`), SpikeInterface 0.104.3, pytest with the v2
Docker MySQL harness.

**Out of scope:** Behavior changes, renames of public methods or tables, schema
changes, moving tables between schemas, and any cleanup that is not a move.

## Constraints (each verified against the code; do not relax)

1. **Table classes stay in their current module.** About 300 files import from
   `recording`, `sorting`, `curation` and `metric_curation` (src, tests, docs,
   notebooks). Each module owns its schema (`recording.py:105`,
   `sorting.py:274`, `curation.py:67`, `metric_curation.py:94`), and DataJoint
   resolves a definition's `-> Parent` references in the declaring module's
   namespace. Moving a table changes all three.
2. **Public methods stay defined on the class.** `docs/mkdocs.yml` configures
   mkdocstrings without `inherited_members`, so a public method moved into a
   mixin or base class disappears from the API reference. Keep each public
   method's signature and docstring on the class; move its body.
3. **`make_fetch`, `make_compute`, `make_insert` stay on the class.** DataJoint
   dispatches the tri-part make by name, and
   `tests/spikesorting/v2/test_integrity.py:238-244` resolves `make_compute`'s
   return annotation through `vars(sys.modules[cls.__module__])`, so the
   carrier NamedTuples (`RecordingComputed`, `SortingComputed`,
   `CurationEvaluationComputed`, ...) stay importable from the table's own
   module.
4. **`make_compute` keeps its staging on the class.** A `make_compute` may make
   exactly one kind of database access: staging its output file through
   `AnalysisNwbfile` (`create`, `get_abs_path`). The tri-part guard permits it
   with `forbid_db_queries(..., allow_staging=True)`
   (`tests/spikesorting/v2/_tripart_helpers.py:43-74`), and all three large
   `make_compute`s rely on it: `Recording` through
   `_compute_recording_artifact` (`recording.py:2036`), `Sorting` through
   `_stage_sorting_artifact` / `_write_units_nwb` (`sorting.py:2493`, `3456`),
   `CurationEvaluation` directly (`metric_curation.py:1266-1270`). So
   `make_compute` stays on the class as the orchestrator: its pure sub-steps
   (array math, analyzer construction, metric computation) move into pure
   modules as functions that take and return plain data, and its staging steps
   keep calling the class's staging methods. Never move a staging call into a
   pure module.
5. **Two separate properties for service modules.**
   - *Cold-import isolation*: importing the module does not import
     `datajoint`, `spyglass.common`, or a v2 table module. Every `_*` service
     module must have it, including ones that query the database at runtime
     (they import lazily, inside functions, as `_recording_nwb`,
     `_sorting_analyzer` and `_analyzer_cache` do). Spawned parallel workers
     import these modules, so a top-level schema import breaks `n_jobs > 1`.
     `tests/spikesorting/v2/test_service_import_contracts.py` enforces it for
     the modules in `_DB_FREE_SERVICE_MODULES` (line 20).
   - *Runtime purity*: the module never queries the database or stages files.
     Only pure modules may receive `make_compute` sub-steps (constraint 4).
     The pure modules today are `_sorting_dispatch`, `_sorting_units`,
     `_sorting_artifact_mask`, `_metric_curation`, `_metric_curation_plots`,
     `_sort_group_planning`, `_recording_preprocessing`,
     `_curation_transforms`, `_curation_plan`, `_curation_routing` and
     `_recipe_catalog`; confirm a destination is pure by reading it before
     moving compute code there.
6. **Patched methods keep class dispatch.** Tests patch class attributes with
   `monkeypatch.setattr` / `mp.setattr` / `patch.object`, often on wrapped
   lines; generate the full list with
   `python .claude/docs/plans/v2-module-split/patch_inventory.py` (an `ast`
   scan, not a grep). Current private patch points on these classes:
   `Sorting._run_sorter` (42 uses), `Sorting._build_analyzer`,
   `Sorting._write_units_nwb`, `Sorting._populate_unit_part`,
   `Sorting._allow_insert`, `Recording._compute_recording_artifact`,
   `Recording._rebuild_nwb_artifact`, `Recording._write_nwb_artifact`,
   `CurationV2._next_curation_id`, `CurationV2._find_matching_child_curation`,
   `CurationEvaluation._compute_metrics`, `CurationEvaluation._display_analyzer`,
   `CurationEvaluation._display_analyzer_key`, `SortingSelection._find_existing_pk`.
   Patched public methods include `get_recording`, `resolve_stored_traces`,
   `get_analyzer`, `add_extensions`, `resolve_source`, `insert_curation`,
   `get_labels`, `get_metrics` and `get_suggested_merge_groups`. Two rules:
   - Every patched name stays a class attribute (a thin delegate is fine).
   - Extracted code calls every patched name *through the class or instance*
     it was handed (`table_cls._find_matching_child_curation(...)`,
     `self._run_sorter(...)`), never through a module-level function. Calling
     the module function directly would bypass the patch and silently change
     what the test exercises.
7. **Stage-result carriers stay `NamedTuple`.** DataJoint splats them and
   DeepHash-compares the fetched carrier across its two fetches.

## Delegation pattern

A public classmethod keeps its signature and docstring; the body moves to a
module-level function that takes the class as its first argument. Calls to
other class methods inside the moved body keep going through that argument, so
patched methods still dispatch through the class (constraint 6).

```python
# curation.py
from spyglass.spikesorting.v2 import _curation_insert


class CurationV2(...):
    @classmethod
    def insert_curation(cls, sorting_key: dict, labels=None, ...) -> dict:
        """<unchanged public docstring>"""
        return _curation_insert.insert_curation(
            cls, sorting_key, labels=labels, ...
        )

    @classmethod
    def _find_matching_child_curation(cls, *args, **kwargs):
        # Patched by tests: stays a class attribute.
        return _curation_insert.find_matching_child_curation(cls, *args, **kwargs)
```

```python
# _curation_insert.py
"""Insert path behind ``CurationV2.insert_curation`` (queries the database)."""


def insert_curation(table_cls, sorting_key, labels=None, ...):
    from spyglass.spikesorting.v2.sorting import Sorting  # lazy: cold import stays clean

    ...
    match = table_cls._find_matching_child_curation(...)  # through the class
    next_id = table_cls._next_curation_id(...)  # patched: through the class
    plan = _build_curation_insert_plan(table_cls, ...)  # unpatched helper: direct call
    ...
```

Move verbatim first, then adjust only what the move requires (`cls` → the
parameter; unpatched private helpers → module functions; patched names stay
`table_cls.<name>`). Do not refactor logic in the same commit.

`make_compute` follows constraint 4 instead: the method body stays on the class
and shrinks to orchestration.

```python
# sorting.py
def make_compute(self, key, fetched) -> SortingComputed:
    sorting = self._run_sorter(...)                    # patched seam
    sorting = _sorting_dispatch.trim_sorting(...)      # pure sub-step, moved
    analyzer = self._build_analyzer(...)               # patched seam
    rows = _sorting_units.unit_rows_from_analyzer(...) # pure sub-step, moved
    staged = self._stage_sorting_artifact(...)         # staging stays on the class
    return SortingComputed(...)
```

## Tasks

Execute one module per PR, in this order (lowest risk first, so the mechanics
are proven before the largest class): recording, sorting, metric_curation,
curation. Line numbers are as of this plan's writing; re-map with `ast` before
starting a module, and re-run `patch_inventory.py`.

### 1. `recording.py` (2,564 → ~1,500)

- **`Recording` artifact build** (`recording.py:1880-2371`): move the bodies of
  `_rebuild_nwb_artifact` (1880, 120 lines), `_compute_recording_artifact`
  (2036, 262), `_write_nwb_artifact` (2341, 31), `_clear_recompute_deleted_flag`
  (2002) and `_recording_provenance_table` (2300) into `_recording_nwb.py`
  (already the NWB-artifact service; cold-import clean, not pure). The three
  patched names stay as delegates, and moved code calls them through the class.
- **`Recording.make_fetch` / `make_insert`** (1315, 172 lines; 1670, 139): move
  the bodies into a new `_recording_tripart.py` (queries the database; lazy
  imports). `make_compute` (1488, 166) stays on the class; move only pure
  sub-steps, if any, into `_recording_preprocessing.py`.
- **`SortGroupV2`** (150-845): `set_group_by_shank` (438, 199) and
  `set_group_by_electrode_table_column` (639, 177) share the post-planning path
  (`_handle_existing`, `_next_sort_group_ids`, `_insert_sort_group_rows`); move
  that shared path and `_cross_team_downstream` (317) into a new
  `_sort_group_insert.py`. Planning already lives in `_sort_group_planning.py`.
- Keep `DriftEstimate`, `PreprocessingParameters`, `RecordingSelection` as they
  are (each under 160 lines).

### 2. `sorting.py` (3,564 → ~2,000)

- **`Sorting` pure compute helpers** (statics, 3292-3564):
  `_apply_artifact_mask`, `_run_clusterless_thresholder`, `_run_si_sorter`,
  `_remove_excess_spikes` → `_sorting_dispatch.py`;
  `_build_unit_rows_from_analyzer` → `_sorting_units.py`. Check each is
  runtime-pure before moving (constraint 5). `_run_sorter`, `_build_analyzer`,
  `_write_units_nwb`, `_populate_unit_part` and `_allow_insert` are patched:
  they stay class attributes; `_build_analyzer`'s body may move to
  `_sorting_analyzer.py` behind the delegate.
- **`Sorting.make_compute`** (2108, 304 lines): stays on the class and becomes
  orchestration per constraint 4; its pure blocks move into the modules above.
- **`Sorting.make_fetch`** (1717, 184) and its static helpers
  `_fetch_motion_correction`, `_fetch_unit_electrode_metadata`,
  `_first_concat_member`, `_resolve_concat_anchor` (1903-2079): new
  `_sorting_fetch.py`. `resolve_anchor_nwb_file_name` (2080) is public: keep it
  on the class as a delegate.
- **`Sorting.find_orphaned_analyzer_folders`** (3065, 176): body →
  `_analyzer_cache.py` (cache path policy already lives there).
- **`SortingSelection.insert_selection`** (880, 272) and
  `_validate_motion_correction_source` (1227): new
  `_sorting_selection_insert.py`. `_find_existing_pk` (1154) is patched: keep
  it as a delegate and call it through the class. The `resolve_*` /
  `load_stored_traces` methods are the public source-resolution API: keep them;
  their bodies may move into `_source_resolution.py`.
- **`SorterParameters.insert`** (317, 142) and
  `insert_default_legacy_si_sorters` (569, 143): row validation and legacy row
  construction → new `_sorter_parameters.py`.

### 3. `metric_curation.py` (2,980 → ~1,800)

- **`CurationEvaluation.make_compute`** (1226, 300): stays on the class as
  orchestration per constraint 4; it keeps the `AnalysisNwbfile` staging and
  `_cleanup_staged_file` / `_write_empty` calls. Its pure blocks and the pure
  helpers `_evaluate_analyzers` (1668), `_spike_counts`,
  `_assert_unit_namespace`, `_assert_merge_membership`, `_compute_merge_groups`
  and `_surface_template_columns` move into `_metric_curation.py`. The body of
  `_compute_metrics` (2258, 293) moves there too, behind the patched
  `_compute_metrics` delegate; moved code calls `_compute_metrics` through the
  class.
- **`make_fetch`** (1022, 203) and `detect_stale_source` (1565, 100): new
  `_metric_curation_fetch.py`.
- **Acceptance workflow** (1861-2257): the private helpers
  `_evaluated_curation_key`, `_resolve_accepted_merges`,
  `_require_merge_acceptance`, `_create_preview_curation` and the body of
  `accept_evaluation_outputs` → new `_evaluation_acceptance.py`. The public
  verbs (`preview_merges`, `accept_merges`, `accept_all_suggested_merges`,
  `use_evaluation_labels`, `overlay_evaluation_labels`) stay. Moved code calls
  the patched `get_labels` and `get_suggested_merge_groups` through the class.
- **Diagnostics** (2660-2980): already thin delegates to `_metric_curation_plots`;
  `_display_analyzer` and `_display_analyzer_key` are patched and stay. Move
  only bodies over ~30 lines (`get_burst_pair_metrics`, `plot_units_qc`).
- **Shipped parameter payloads**: `QualityMetricParameters._default_rows` (332)
  and `AutoCurationRules._default_payloads` (627) are catalog data →
  `_recipe_catalog.py`.

### 4. `curation.py` (3,044 → ~1,400)

- **Insert path** (`curation.py:508-1530`, ~1,000 lines): the body of
  `insert_curation` (508, 354) and its private helpers
  `_normalized_labels`, `_normalized_real_merge_groups`,
  `_normalize_curation_inputs`, `_validate_parent_or_reuse_root`,
  `_resolve_curation_source`, `_build_curation_insert_plan`,
  `_stage_curation_artifact`, `_insert_curation_rows_transaction`,
  `_cleanup_staged_curation_file`, `_assert_child_reuse_for_merge_wrapper` →
  new `_curation_insert.py`. `_build_merge_provenance_rows` (1278, static,
  pure) → `_curation_transforms.py`. `_next_curation_id` and
  `_find_matching_child_curation` are patched: they stay class attributes and
  moved code calls them through the class (see "Delegation pattern").
- **`resolve_restriction`** (2389, 246): body → new `_curation_restriction.py`;
  the routing classifier it uses is already in `_curation_routing.py`.
- **Recording/sorting readers**: bodies of `get_recording` (1973),
  `get_source_recording` (2043), `get_sorting_input_recording` (2086),
  `get_sorting` (2144, 102), `get_merged_sorting` (2764, 75) and the private
  `_load_curation_recording_meta`, `_upstream_recording_row` → new
  `_curation_readers.py`.
- Keep on the class: `delete`, the audits, the creation wrappers
  (`create_initial_curation`, `propose_merge_curation`, `create_merged_curation`,
  `save_manual_curation`), the status predicates and the small accessors.

### Per-module steps (each PR)

1. Record the module's current API: list each table class's members with
   `inspect.signature`, and the public module-level names; save to the
   scratchpad. Run `patch_inventory.py` and save its output.
2. Run the module's validation suites (see Validation slice) and record the
   result.
3. Move code per the tasks above, one cluster per commit.
4. Add every new `_*` module, and every existing destination missing from the
   list, to `_DB_FREE_SERVICE_MODULES` in
   `tests/spikesorting/v2/test_service_import_contracts.py`. The metric PR adds
   `_metric_curation` and `_metric_curation_plots`, which are missing today.
5. Re-run step 1 and diff: every public name keeps the same signature, and every
   name `patch_inventory.py` listed is still a class attribute.
6. Re-run step 2; results must match.
7. Run `python -m pytest --collect-only tests/spikesorting tests/utils
   tests/decoding` and the leakage test.

## Deliberately not in this plan

- Moving `DriftEstimate`, `SortGroupV2` or the parameter Lookups into their own
  modules: they are small or already thin, and moving a table breaks
  constraint 1. Revisit only if a module is still over ~2,000 lines after the
  body moves.
- Mixins for `CurationV2`: they would hide public methods from the API docs
  (constraint 2).
- Renaming `_DB_FREE_SERVICE_MODULES` to reflect that it checks cold-import
  isolation, not runtime purity: a test-name cleanup, separate from the split.
- `session_group.py`, `unit_matching.py`, `artifact.py`, `motion.py`: not in the
  original scope; reassess after the four modules are done.

## Validation slice

Every PR:

| Test | Asserts |
| --- | --- |
| API snapshot diff (step 5) | Every public method keeps its signature; every patched name is still a class attribute |
| `test_service_import_contracts.py` (with step 4's additions) | Every `_*` service module, old and new, imports without the database layer |
| `test_integrity.py` | Tri-part tables keep `make_compute` carriers and staged-output contracts |
| `test_v1_parity.py::test_no_phase_label_leakage_in_runtime_code` | New modules carry no plan vocabulary |

Per module (all under `tests/spikesorting/v2/` unless noted; DB tests use the
Docker harness):

| PR | Suites | Asserts |
| --- | --- | --- |
| recording | `single_session/test_recording.py` (slow), `test_recording_nwb.py`, `test_recording_services.py`, `test_recompute.py`, `test_sort_group_planning.py`, `single_session/test_sort_group.py` | Recording populate, NWB artifact build/rebuild, recompute and sort-group insertion unchanged |
| sorting | `single_session/test_sorting.py` (slow), `test_sorting_dispatch.py`, `test_sorting_contracts.py` (incl. the `allow_staging` guard), `test_sorter_parameters.py`, `test_parameter_identity.py`, `test_analyzer_lifecycle.py`, `test_analyzer_publication.py`, `test_selection_identity.py`, `test_motion_consumers.py` | Sorter dispatch; staging-only DB access in `make_compute`; parameter duplicate rejection and identity; analyzer retry cleanup and preservation of referenced folders; selection reuse |
| metric_curation | `test_curation_evaluation.py` (incl. the `allow_staging` guard), `test_metric_curation_transforms.py`, `test_metric_curation_plots.py`, `test_metric_eligibility.py`, `test_curation_analyzer.py` | Evaluation compute, eligibility, acceptance verbs and diagnostics unchanged |
| curation | `test_curation_composition.py`, `single_session/test_curation_merges.py`, `single_session/test_curation_insert.py`, `test_curation_api.py`, `test_curation_routing.py`, `test_merge_dedup.py` | Insert, reuse, merge and restriction resolution unchanged |

Last PR only: `tests/spikesorting/v2/test_notebook_execution.py` (slow), to
check the end-to-end pipeline still registers tables and runs.

Follow the repo's test-run rules: one pytest session at a time, only the listed
modules (a full v2 sweep runs over an hour).

## Open questions (owner)

- Is the per-module target size (~1,400-2,000 lines) the goal, or should the
  split go further (for example splitting `Sorting` maintenance methods)?
- Should `SortingSelection`'s public `resolve_*` bodies move into
  `_source_resolution.py`, or stay on the class as the readable reference?
- Order: recording → sorting → metric_curation → curation, or curation first
  because it has the most value?

## Review

Before opening each PR, dispatch `code-reviewer` (or equivalent independent
reviewer) against the diff. Confirm:
- Code was moved, not changed: diff the moved bodies against their originals.
- Every constraint above holds; no public name, signature or docstring changed;
  moved code calls every patched name through the class.
- Validation slice tests pass; slow / integration tests are marked.
- New module names and docstrings describe what the code does and do not
  reference this plan.
