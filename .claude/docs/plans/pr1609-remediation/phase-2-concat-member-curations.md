# Phase 2 — Per-member decodable rows for concatenated sorts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#concat-member-curation)

Ships as a new PR against `master` after #1609 merges. Schema-additive (one new schema with one Computed table, one new `SpikeSortingOutput` part); no `alter()` of existing v2 tables. Env: `spyglass_spikesorting_v2`.

**Inputs to read first:**

- [designs.md § concat-member-curation](designs.md#concat-member-curation) — table definition, `key_source`, make steps, merge part, source contract, orchestration.
- [overview.md § Current codebase integration points — Phase 2](overview.md#current-codebase-integration-points) — verified line refs for `SessionGroup.Member`, `MemberSnapshot`, `MemberBoundary`, `split_unit_spike_trains`, the units-NWB writer, and the merge-table source contract.
- `src/spyglass/spikesorting/v2/curation.py:1290-1395` — the transaction pattern (`AnalysisNwbfile().add` → part inserts → `_merge_insert`) to mirror.
- `src/spyglass/spikesorting/v2/session_group.py:1467-1553` — `split_sorting_by_session`, the existing back-map whose math this table productionizes.
- `src/spyglass/spikesorting/spikesorting_merge.py:38-62, 104-152, 353-372, 456-512` — v2 probe pattern, part declarations, and the methods the merge dispatch calls on a source class.
- `src/spyglass/spikesorting/analysis/v1/group.py:63-74, 190-215` — how a downstream consumer reads a merge row (per-`nwb_file_name` group, `fetch_nwb(return_merge_ids=True, multi_source=True)`).
- Phase 1's `CONCAT_MERGE_GATE_MESSAGE` and `CurationV2.audit_concat_merge_rows` in `curation.py` (added by [phase 1 B1](phase-1-pr1609-fixes.md)).

## Tasks

- **T1. Table + make.** New module `src/spyglass/spikesorting/v2/concat_member_curation.py` with schema `spikesorting_v2_concat_curation` and `ConcatMemberCuration` exactly as in the design (definition, `key_source`, `make`). Before writing `key_source`, read `ConcatenatedRecordingSelection.MemberSnapshot.definition` (`session_group.py:482-510`) and use its actual column names for `nwb_file_name` / `recording_id`. In `make`, assert spike conservation across all members of the sort once per curation (sum of per-member counts equals the curated concat count per unit; `split_unit_spike_trains` already asserts this — surface its error, do not swallow). Assert every local frame is `< len(member timestamps)` before indexing. Stage the member `AnalysisNwbfile` under the member `nwb_file_name` and clean it up on any failure (mirror `curation.py` `_cleanup_staged_curation_file`).
- **T2. Merge-table part + source contract.** In `spikesorting_merge.py`: probe `ConcatMemberCuration` beside `_probe_v2_curation` (`:38-62`), add it to `source_class_dict` (`:104-110`) and `_default_merge_sources` (`:92-100`), declare the `SpikeSortingOutput.ConcatMemberCuration` part. Implement `get_recording`, `get_sorting`, `get_sort_group_info`, and mixin `fetch_nwb` on the new class per the design. `get_sort_group_info` must not raise `ConcatBrainRegionAmbiguousError` (regions are per member).
- **T3. Downstream compatibility.** Confirm `SortedSpikesGroup.fetch_spike_data` (`group.py:190-215`), `UnitAnnotation.add_annotation` (`unit_annotation.py:71-87`), `UnitWaveformFeatures.make` dispatch (`decoding/v1/waveform_features.py:187`: `is_v2 = to_camel_case(...) == "CurationV2"`), and `SpikeSortingOutput.assert_decoding_merge_ids_ok` treat a member row correctly. Extend the waveform-features dispatch to `{"CurationV2", "ConcatMemberCuration"}` and route `_fetch_waveform_v2` to the member's recording; `CurationV2.get_sort_metadata` gains a sibling on the new class.
- **T4. Orchestration.** In `_pipeline_run.py` concat path: populate `ConcatMemberCuration` after the curation stage and add `member_merge_ids: dict[str, UUID]` to the summary (`_pipeline_types.py`: new key on the concat summary TypedDict); `describe_run` (`_pipeline_reporting.py`) prints one line per member. `initialize_v2_defaults` unchanged (no new lookups). Remove the Phase-1 warning branch's "planned addition" wording in `CONCAT_MERGE_GATE_MESSAGE` and point it at `ConcatMemberCuration` instead; keep the gate (the concat `CurationV2` row itself is still never registered).
- **T5. Migration for trial DBs.** CHANGELOG release-notes block: `CurationV2.audit_concat_merge_rows()` → delete listed rows → `ConcatMemberCuration.populate()`. Breaking-change bullet: concat sorts now surface downstream as one merge row per member session.
- **T6. Docs + notebook.** `docs/src/Features/SpikeSortingV2.md` concat section (`:1060-1100`): replace step 5 (`split_sorting_by_session`) with "populate `ConcatMemberCuration`; each member gets a `SpikeSortingOutput` row keyed by its `nwb_file_name`; unit ids are shared across members"; remove the Phase-1 caveat bullet. `notebooks/py_scripts/10_Spike_SortingV2_CrossSession.py` Part A (concat): add a cell reading `member_merge_ids` and feeding one into `SortedSpikesGroup` for that member's session; regenerate the `.ipynb` with jupytext. Table inventory in the doc gains the new schema.
- **T7. Storage doc.** `SpikeSortingV2StorageManagement.md § Concatenated recordings` (`:101`): member rows are regenerable from the concat curation NWB; deleting a concat `CurationV2` cascades to its member rows and their analysis files through the FK.

## Deliberately not in this phase

- Re-labeling or re-merging per member: labels and merges are applied once on the concat `CurationV2` and propagated verbatim (owner decision pending in [overview Open Question 2](overview.md#open-questions); default is verbatim).
- Feeding member rows into `UnitMatch`: cross-session tracking keeps consuming the concat curation; members of one concat group are already the same units by construction.
- Any change to `ConcatenatedRecording`'s stored timeline or to `ignore_times=True`: the concat recording stays synthetic; only the member rows are wall-clock.
- Curation UX (viewing merged units, FigPack) for concat sorts → `.claude/docs/plans/curation-ux-overhaul/`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_concat_member_curation.py::test_spike_conservation` (integration, marked `slow`) | For the smoke concat group: per unit, `sum(len(train) for member rows) == len(curated concat train)`. |
| `::test_member_times_within_member_interval` | Every member row's spike times lie inside that member's `IntervalList` valid times and inside `[ts[0], ts[-1]]` of the member `Recording` timestamps; none equals the concat synthetic time for the same spike (gap-shifted members differ by the wall-clock gap). |
| `::test_unit_ids_preserved_across_members` | Set of unit ids on every member row equals the concat curation's kept unit ids; absent units have empty trains and `n_spikes == 0`. |
| `::test_labels_propagated` | `UnitLabel` labels on the concat curation appear on each member NWB's units table. |
| `::test_merge_row_per_member` | One `SpikeSortingOutput.ConcatMemberCuration` row per member; `SpikeSortingOutput().get_recording({"merge_id": ...})` returns the member `Recording`; `get_sorting` frames round-trip through `numpysorting_from_abs_times` to the member timestamps. |
| `::test_sorted_spikes_group_reads_member_row` | `SortedSpikesGroup.fetch_spike_data` on a group containing one member merge id returns the same trains as reading the member NWB directly; `assert_decoding_merge_ids_ok` passes. |
| `::test_waveform_features_dispatch_member_row` | `UnitWaveformFeatures` dispatch treats the member row as a v2 source and extracts from the member recording (smoke, `slow`). |
| `::test_concat_curation_itself_not_registered` | The concat `CurationV2` still has no `SpikeSortingOutput.CurationV2` row (Phase-1 gate retained). |
| `::test_delete_cascades_member_rows` | Deleting the concat `CurationV2` removes its member rows, their merge rows, and their `AnalysisNwbfile` entries (dry-run preview lists them). |
| `tests/spikesorting/v2/test_pipeline_orchestrator*.py` | Concat `run_v2_pipeline` summary carries `member_merge_ids` keyed by `nwb_file_name`; `describe_run` prints them; `root_merge_id is None` still. |
| `tests/spikesorting/v2/test_legacy_runtime_boundary.py::test_default_merge_sources_skip_v2_when_unavailable` | Extended: `ConcatMemberCuration` part is skipped when v2 is unavailable, same as `CurationV2`. |
| Real-data smoke (manual, before merge) | One same-day two-epoch lab group: populate, then decode one member with the existing decoding notebook; check that ripple-locked or position-locked firing looks time-aligned (a synthetic-timeline row would show a constant offset equal to the inter-epoch gap). |

## Fixtures

- Smoke concat group: the fixtures in `tests/spikesorting/v2/test_session_group_concat.py` (two members with a deliberate wall-clock gap between member recordings; if the existing fixture has contiguous members, add a gap so the synthetic-vs-wall-clock difference is nonzero and testable).
- `SortedSpikesGroup` / decoding consumer: `tests/decoding/conftest.py` patterns for building a group from a merge id.
- Real-data slice: one two-epoch session from the lab DB, run under supervision (this is the PR's stated pre-GA testing mode).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the Phase-1 "planned addition" wording; the doc's `split_sorting_by_session` step 5).
- User-facing documentation listed as tasks is updated, not deferred.
