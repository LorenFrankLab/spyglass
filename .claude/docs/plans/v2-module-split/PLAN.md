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
helpers move out entirely. Pure logic goes into the existing DB-free service
modules; logic that queries the database goes into private service modules that
import table classes lazily. This is the pattern v2 already uses
(`_sorting_dispatch`, `_curation_transforms`, `_recording_nwb`, ...).

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
   dispatches the tri-part make by name, and `tests/spikesorting/v2/test_integrity.py:238-244`
   resolves `make_compute`'s return annotation through `vars(sys.modules[cls.__module__])`,
   so the carrier NamedTuples (`RecordingComputed`, `SortingComputed`,
   `CurationEvaluationComputed`, ...) must stay importable from the table's own
   module. Their bodies may move. `make_compute` bodies must stay DB-free (the
   tri-part DB-access guard in `tests/spikesorting/v2/_tripart_helpers.py`), so
   they move only into DB-free modules.
4. **Methods tests monkeypatch stay as class attributes** (thin delegates are
   fine). Current patch points: `Sorting._run_sorter` (19 uses),
   `Sorting._write_units_nwb`, `Sorting._populate_unit_part`,
   `Recording._write_nwb_artifact`, `Recording._rebuild_nwb_artifact`,
   `Recording._compute_recording_artifact`, `CurationV2._next_curation_id`,
   `CurationEvaluation._compute_metrics`, `CurationEvaluation._display_analyzer`.
   Re-run `git grep -n "monkeypatch.setattr(" tests/spikesorting` before each
   module and add any new ones to this list.
5. **DB-free modules stay DB-free.** A module documented as pure (the current
   ones: `_sorting_dispatch`, `_sorting_units`, `_sorting_artifact_mask`,
   `_metric_curation`, `_metric_curation_plots`, `_sort_group_planning`,
   `_recording_preprocessing`, `_curation_transforms`, `_curation_plan`,
   `_curation_routing`, `_recipe_catalog`) must not import `datajoint`,
   `spyglass.common`, or a v2 table module. Spawned parallel workers import
   these, so a schema import there breaks `n_jobs > 1`.
6. **Stage-result carriers stay `NamedTuple`.** DataJoint splats them and
   DeepHash-compares the fetched carrier across its two fetches.
7. **No import cycles at module load.** DB-touching service modules import
   table classes inside functions, as `_source_resolution.py` and
   `_recording_nwb.py` do.

## Delegation pattern

A public classmethod keeps its signature and docstring; the body moves to a
module-level function that takes the class as its first argument. Call sites
inside the class keep calling the method, so nothing outside the module changes.

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
```

```python
# _curation_insert.py
"""Insert path behind ``CurationV2.insert_curation`` (queries the database)."""


def insert_curation(table_cls, sorting_key, labels=None, ...):
    from spyglass.spikesorting.v2.sorting import Sorting  # lazy: no cycle

    ...  # body moved verbatim; ``cls._x(...)`` becomes ``_x(table_cls, ...)``
```

Move verbatim first, then adjust only what the move requires (`cls` → the
parameter, private helper calls → module functions). Do not refactor logic in
the same commit.

## Tasks

Execute one module per PR, in this order (lowest risk first, so the mechanics
are proven before the largest class): recording, sorting, metric_curation,
curation. Line numbers are as of this plan's writing; re-map with
`python3 -c` + `ast` before starting a module.

### 1. `recording.py` (2,564 → ~1,500)

- **`Recording` artifact build** (`recording.py:1880-2371`): move the bodies of
  `_rebuild_nwb_artifact` (1880, 120 lines), `_compute_recording_artifact`
  (2036, 262), `_write_nwb_artifact` (2341, 31), `_clear_recompute_deleted_flag`
  (2002) and `_recording_provenance_table` (2300) into `_recording_nwb.py`
  (already the NWB-artifact service). Keep the three patched names as
  delegates (constraint 4).
- **`Recording.make_fetch` / `make_insert`** (1315, 172 lines; 1670, 139): move
  the bodies into a new `_recording_tripart.py` (DB-touching). `make_compute`
  (1488, 166) moves into `_recording_preprocessing.py` only if every line it
  runs is DB-free; otherwise leave it.
- **`SortGroupV2`** (150-845): `set_group_by_shank` (438, 199) and
  `set_group_by_electrode_table_column` (639, 177) share the post-planning path
  (`_handle_existing`, `_next_sort_group_ids`, `_insert_sort_group_rows`); move
  that shared path and `_cross_team_downstream` (317) into a new
  `_sort_group_insert.py`. Planning already lives in `_sort_group_planning.py`.
- Keep `DriftEstimate`, `PreprocessingParameters`, `RecordingSelection` as they
  are (each under 160 lines).

### 2. `sorting.py` (3,564 → ~1,900)

- **`Sorting` compute helpers** (DB-free statics, 3292-3564): `_apply_artifact_mask`,
  `_run_clusterless_thresholder`, `_run_si_sorter`, `_remove_excess_spikes` →
  `_sorting_dispatch.py`; `_build_unit_rows_from_analyzer` → `_sorting_units.py`;
  `_build_analyzer` → `_sorting_analyzer.py`. Keep `_run_sorter`,
  `_write_units_nwb`, `_populate_unit_part` as delegates (constraint 4).
- **`Sorting.make_compute`** (2108, 304 lines): move the body to a function in
  `_sorting_dispatch.py` (it already holds the dispatch path and is DB-free).
- **`Sorting.make_fetch`** (1717, 184) and its static helpers
  `_fetch_motion_correction`, `_fetch_unit_electrode_metadata`,
  `_first_concat_member`, `_resolve_concat_anchor` (1903-2079): new
  `_sorting_fetch.py`. `resolve_anchor_nwb_file_name` (2080) is public: keep it
  on the class as a delegate.
- **`Sorting.find_orphaned_analyzer_folders`** (3065, 176): body →
  `_analyzer_cache.py` (cache path policy already lives there).
- **`SortingSelection.insert_selection`** (880, 272), `_find_existing_pk`
  (1154), `_validate_motion_correction_source` (1227): new
  `_sorting_selection_insert.py`. The `resolve_*` / `load_stored_traces`
  methods are the public source-resolution API: keep them; their bodies may
  move into `_source_resolution.py`.
- **`SorterParameters.insert`** (317, 142) and
  `insert_default_legacy_si_sorters` (569, 143): row validation and legacy row
  construction → new `_sorter_parameters.py`.

### 3. `metric_curation.py` (2,980 → ~1,700)

- **`CurationEvaluation` compute path** (DB-free): `make_compute` body (1226,
  300), `_evaluate_analyzers` (1668), `_spike_counts`, `_assert_unit_namespace`,
  `_assert_merge_membership`, `_compute_merge_groups`, `_surface_template_columns`
  and the `_compute_metrics` body (2258, 293) → `_metric_curation.py`. Keep
  `_compute_metrics` as a delegate (constraint 4).
- **`make_fetch`** (1022, 203) and `detect_stale_source` (1565, 100): new
  `_metric_curation_fetch.py`.
- **Acceptance workflow** (1861-2257): the private helpers
  `_evaluated_curation_key`, `_resolve_accepted_merges`,
  `_require_merge_acceptance`, `_create_preview_curation` and the body of
  `accept_evaluation_outputs` → new `_evaluation_acceptance.py`. The public
  verbs (`preview_merges`, `accept_merges`, `accept_all_suggested_merges`,
  `use_evaluation_labels`, `overlay_evaluation_labels`) stay.
- **Diagnostics** (2660-2980): already thin delegates to `_metric_curation_plots`;
  keep `_display_analyzer` (patched). Move only bodies over ~30 lines
  (`get_burst_pair_metrics`, `plot_units_qc`).
- **Shipped parameter payloads**: `QualityMetricParameters._default_rows` (332)
  and `AutoCurationRules._default_payloads` (627) are catalog data →
  `_recipe_catalog.py`.

### 4. `curation.py` (3,044 → ~1,300)

- **Insert path** (`curation.py:508-1530`, ~1,000 lines): the body of
  `insert_curation` (508, 354) and its private helpers
  `_normalized_labels`, `_normalized_real_merge_groups`,
  `_find_matching_child_curation`, `_normalize_curation_inputs`,
  `_validate_parent_or_reuse_root`, `_resolve_curation_source`,
  `_build_curation_insert_plan`, `_stage_curation_artifact`,
  `_insert_curation_rows_transaction`, `_cleanup_staged_curation_file`,
  `_assert_child_reuse_for_merge_wrapper` → new `_curation_insert.py`.
  `_build_merge_provenance_rows` (1278, static, pure) → `_curation_transforms.py`.
  Keep `_next_curation_id` on the class (constraint 4).
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

1. Record the module's current API: `python -c` listing `dir()` of each table
   class and the public module-level names; save to the scratchpad.
2. Run the module's test suites (see Validation slice) and record the result.
3. Move code per the tasks above, one cluster per commit.
4. Re-run step 1 and diff: every public name must still exist on the same
   class with the same signature (`inspect.signature`).
5. Re-run step 2; results must match.
6. Run `python -m pytest --collect-only tests/spikesorting tests/utils
   tests/decoding` and the leakage test.

## Deliberately not in this plan

- Moving `DriftEstimate`, `SortGroupV2` or the parameter Lookups into their own
  modules: they are small or already thin, and moving a table breaks
  constraint 1. Revisit only if a module is still over ~2,000 lines after the
  body moves.
- Mixins for `CurationV2`: they would hide public methods from the API docs
  (constraint 2).
- `session_group.py`, `unit_matching.py`, `artifact.py`, `motion.py`: not in the
  original scope; reassess after the four modules are done.

## Validation slice

| Test | Asserts |
| --- | --- |
| API snapshot diff (step 4) | Every public method of each table class still exists with an identical signature |
| `tests/spikesorting/v2/test_integrity.py` | Tri-part tables keep `make_compute` carriers and staged-output contracts |
| `tests/spikesorting/v2/test_service_import_contracts.py` | DB-free modules import no schema modules |
| `tests/spikesorting/v2/single_session/` (recording, sorting) | Recording and sort populate unchanged (DB, slow) |
| `test_curation_*.py`, `test_metric_curation_*.py`, `test_curation_evaluation.py` | Curation insert, merge, evaluation and acceptance unchanged (DB) |
| `test_sorting_dispatch.py`, `test_sort_group_planning.py`, `test_recording_services.py` | Moved helpers behave the same |
| `test_v1_parity.py::test_no_phase_label_leakage_in_runtime_code` | New modules carry no plan vocabulary |
| `tests/spikesorting/v2/test_notebook_execution.py` (slow) | End-to-end pipeline still registers tables and runs |

Follow the repo's test-run rules: run one pytest session at a time, target the
listed modules (a full v2 sweep runs over an hour), and use the Docker harness.

## Open questions (owner)

- Is the per-module target size (~1,300-1,900 lines) the goal, or should the
  split go further (for example splitting `Sorting` maintenance methods)?
- Should `SortingSelection`'s public `resolve_*` bodies move into
  `_source_resolution.py`, or stay on the class as the readable reference?
- Order: recording → sorting → metric_curation → curation, or curation first
  because it has the most value?

## Review

Before opening each PR, dispatch `code-reviewer` (or equivalent independent
reviewer) against the diff. Confirm:
- Code was moved, not changed: diff the moved bodies against their originals.
- Every constraint above holds; no public name, signature or docstring changed.
- Validation slice tests pass; slow / integration tests are marked.
- New module names and docstrings describe what the code does and do not
  reference this plan.
