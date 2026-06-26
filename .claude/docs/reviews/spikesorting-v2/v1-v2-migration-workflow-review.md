# Spike Sorting V2 v1-to-v2 migration workflow review

Date: 2026-06-25

Scope: user-facing migration workflow from v1 to v2, including the feature docs,
migration guide, notebooks, merge-table helpers, curation/merge id handoff, and
status messaging for shipped versus pending v2 features.

Method: local docs/source/test inspection plus one independent explorer-agent
pass. No tests were run for this review.

## Executive Summary

The migration surface has the right ingredients: v2 curations register on the
shared `SpikeSortingOutput` merge table, the main v2 guide is much more complete
than the migration page, and tests cover several v1/v2 parity helper contracts.

The main problem is that the docs are internally inconsistent after recent v2
work. Some sections still say UnitMatch and concat are roadmap-only, while other
sections and the code expose them as implemented. More importantly, the main docs
tell downstream users to carry `run_summary["merge_id"]`, which is the root
curation returned by `run_v2_pipeline`; the notebook has the safer warning to use
the final curation/merge id after auto-curation or manual merges.

## Findings

### 1. High: docs steer downstream users to the uncurated root `merge_id`

The main v2 docs show:

- `merge_id = run_summary["merge_id"]  # key off this downstream`
  (`docs/src/Features/SpikeSortingV2.md:204`)
- "For most downstream work, carry `merge_id = run_summary["merge_id"]` forward"
  (`docs/src/Features/SpikeSortingV2.md:981`)

But `run_v2_pipeline` returns the root curation merge row. The independent pass
verified that `_pipeline_run.py` creates the root curation with
`parent_curation_id=-1` and returns that merge row
(`src/spyglass/spikesorting/v2/_pipeline_run.py:407`,
`src/spyglass/spikesorting/v2/_pipeline_run.py:433`). The notebook has the
correct warning to use `final_merge_id`, not `run_summary["merge_id"]`, after
curation (`notebooks/py_scripts/10_Spike_SortingV2.py:490`).

Impact:

- Users who run auto-curation or manual merges can decode/export the uncurated
  root, dropping labels and applied merges.
- v1 users are used to carrying a curated merge id into downstream tables; this
  wording makes the v2 equivalent easy to get wrong.

Recommended fix:

- Update `SpikeSortingV2.md` and `SpikeSortingV2_Migration.md` to distinguish
  the initial/root `merge_id` from the final curated `merge_id`.
- Thread `final_curation` / `final_merge_id` through all downstream examples.
- Keep `run_summary["merge_id"]` documented as the root output returned by the
  pipeline helper, not the default downstream handoff after curation.

### 2. High: migration/status docs still mark shipped UnitMatch and concat workflows as pending

The migration guide says `unit_matching` and `matcher_protocol` are placeholders
(`docs/src/Features/SpikeSortingV2_Migration.md:158`) and that
`ConcatenatedRecording` / `SessionGroup` methods raise `NotImplementedError`
(`docs/src/Features/SpikeSortingV2_Migration.md:161`). The main v2 status block
also lists cross-session unit matching as unavailable
(`docs/src/Features/SpikeSortingV2.md:1055`).

Those sections contradict the implemented code and earlier docs:

- The main v2 docs describe cross-session unit tracking and UnitMatch usage
  (`docs/src/Features/SpikeSortingV2.md:880`).
- `UnitMatchSelection` and `UnitMatch` are real DataJoint tables
  (`src/spyglass/spikesorting/v2/unit_matching.py:197`,
  `src/spyglass/spikesorting/v2/unit_matching.py:438`).
- `SessionGroup` / concat are implemented in
  `src/spyglass/spikesorting/v2/session_group.py`.

Impact:

- Users are told not to use valid v2 migration paths, especially same-day chronic
  concatenate-and-sort and cross-session unit tracking.
- Review/triage work can accidentally treat implemented paths as future
  placeholders.

Recommended fix:

- Make FigPack the only remaining placeholder in these status sections unless a
  newer gap exists.
- Mark concat and UnitMatch as available, with explicit caveats for optional
  matching extras, fixture gates, and current validation scope.
- Add a docs test or grep-based guard for stale phrases like "unit_matching
  placeholder" and "ConcatenatedRecording ... NotImplementedError".

### 3. Medium: notebooks are stale for shipped cross-session/concat workflows

The UnitMatch notebook still says it is running before a DataJoint wrapper is
written (`notebooks/py_scripts/13_UnitMatch_Cross_Session.py:20`) and ends with
an API summary for that future wrapper
(`notebooks/py_scripts/13_UnitMatch_Cross_Session.py:386`). The dev walkthrough
still says concat is not wired and that `ConcatenatedRecording.make` raises today
(`notebooks/10_Spike_SortingV2_dev_walkthrough.ipynb:588`,
`notebooks/10_Spike_SortingV2_dev_walkthrough.ipynb:954`).

Impact:

- Notebook users lack a user-facing example for the now-shipped
  `SessionGroup`, `UnitMatchSelection`, `UnitMatch`, and `TrackedUnit` tables.
- The main docs and notebooks tell different stories about the migration path.

Recommended fix:

- Update the UnitMatch notebook to use the DataJoint wrapper path described in
  `SpikeSortingV2.md`.
- Move the direct UnitMatchPy spike to developer notes or a backend validation
  notebook.
- Refresh the dev walkthrough's concat stage and regenerate py scripts.

### 4. Medium: merge-id restriction strictness is documented incorrectly

The main docs say `SpikeSortingOutput.get_restricted_merge_ids` defaults to every
available source and that unknown restriction keys raise `ValueError`
(`docs/src/Features/SpikeSortingV2.md:1117`). The independent pass verified that
the default multi-source path is lenient in code unless `sources` is explicit
(`src/spyglass/spikesorting/spikesorting_merge.py:286`) and that tests document
that leniency (`tests/spikesorting/v2/test_v1_parity.py:454`).

Impact:

- Migration users may expect typo protection from the default helper but get
  empty or missing v2 results instead.
- This is exactly the class of mistake migration docs should make loud.

Recommended fix:

- Document that default multi-source lookup is lenient.
- Tell v2 migration users to call `get_restricted_merge_ids(..., sources=["v2"])`
  or `get_spiking_sorting_v2_merge_ids(...)` when validating v2 keys.
- Add a docs snippet or test around typo handling for explicit v2 sources.

### 5. Medium-low: the practical v1-to-v2 porting recipe is split across pages

The migration guide lists renamed tables/parameters and a feature status table,
but the actual "carry this object forward" story is scattered across the main
docs, notebooks, and utility helpers. This interacts with the root/final
`merge_id` problem above.

Impact:

- A v1 user porting a notebook must assemble the critical sequence from multiple
  sources: selection, `run_v2_pipeline`, root curation, analyzer curation/manual
  merge, final merge id, downstream query helper, and export.
- The migration guide is not currently executable as a workflow checklist.

Recommended fix:

- Add one compact end-to-end migration recipe:
  `v1 SpikeSortingSelection -> v2 run_v2_pipeline -> root CurationV2 -> final
  CurationV2 -> final merge_id -> downstream query/export`.
- Include the v1/v2 table names next to each step.
- Add a smoke test or docs check that keeps the recipe's public names current.

### 6. Low: `apply_merge` migration wording is stale

The migration guide says v1 used `apply_merges`
(`docs/src/Features/SpikeSortingV2_Migration.md:38`). Current v1 and v2 both use
`apply_merge` (`src/spyglass/spikesorting/v1/curation.py:44`,
`src/spyglass/spikesorting/v2/curation.py:193`).

Impact:

- Small on its own, but it reduces trust in the migration guide precisely where
  users are checking names.

Recommended fix:

- Remove the claim that current v1 used `apply_merges`.
- Keep `apply_merge` as the supported spelling in examples and migration notes.

