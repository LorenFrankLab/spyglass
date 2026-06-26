# Phase 3b — NWB metadata/lineage provenance

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make the v2 NWB artifacts self-describing (R10). The writers currently persist
results (traces, spikes, pairs, metrics) but no source/sorter/params/lineage, so a
shared NWB can't be understood without the DataJoint DB. Write the **metadata** —
which is cheap — into each writer; for the two **large-array** items store the
producing params, not the array. Depends on **phase-3a** (reuses the same provenance
field names / version strings) **and phase-2** (its **task 5** stores the *resolved*
motion preset that this phase's task 6 writes into the concat NWB).

**Inputs to read first:**

- [shared-contracts.md#producer-provenance-field-set](shared-contracts.md#producer-provenance-field-set) — field names shared with 3a.
- [src/spyglass/spikesorting/v2/_recording_nwb.py:225-245](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py#L225-L245) — recording ElectricalSeries write (`add_acquisition` at 243).
- [src/spyglass/spikesorting/v2/_units_nwb.py:626-664](../../../../src/spyglass/spikesorting/v2/_units_nwb.py#L626-L664) — sort-time Units writer (`add_unit_column`/`add_unit`); the curated writer (`write_curated_units_nwb` / `_write_curated_units_nwb_body`, ~864-945).
- [src/spyglass/spikesorting/v2/_unitmatch_nwb.py:27-92](../../../../src/spyglass/spikesorting/v2/_unitmatch_nwb.py#L27-L92) — `build_pairs_table` (per-pair columns only; no session map/params).
- [src/spyglass/spikesorting/v2/_metric_curation_nwb.py](../../../../src/spyglass/spikesorting/v2/_metric_curation_nwb.py) — `write_analyzer_curation_tables` (~166-196; three result-only scratch tables).
- [src/spyglass/spikesorting/v2/session_group.py](../../../../src/spyglass/spikesorting/v2/session_group.py) — concat write (~876-884) + `MemberBoundary` (~592-600).
- DB rows that already hold the metadata to write: [sorting.py:1162-1169](../../../../src/spyglass/spikesorting/v2/sorting.py#L1162-L1169) (`Sorting.Unit`: Electrode, `peak_amplitude_uv`, `n_spikes`), [curation.py](../../../../src/spyglass/spikesorting/v2/curation.py) `CurationV2.MergeGroup` (~154), `Raw.raw_object_id`.

**Contracts referenced:** [Producer provenance field set](shared-contracts.md#producer-provenance-field-set).

## Tasks

1. **Recording NWB source provenance.** In `_recording_nwb.write_nwb_artifact` (around the ElectricalSeries write, `_recording_nwb.py:230-245`), write the construction provenance as NWB scratch / `general` metadata (not into the data path): raw source `object_id` (`Raw.raw_object_id`), `recording_id`, `preprocessing_params_name`, `sort_group_id`, the resolved reference mode, bad-channel handling, and the `spikeinterface_version`. Use `nwbfile.add_scratch(...)` or a small `general`-level container keyed under a stable name (e.g. `spyglass_v2_recording_provenance`). The metadata is small scalars/strings — write it directly. (The processed series itself stays in `acquisition`; the NWB-10 "processed under acquisition" cosmetic is out of scope.)

2. **Sorting Units NWB: per-unit + source metadata.** In the sort-time Units writer — `write_sorting_units_nwb` / `_write_sorting_units_nwb_body` (`_units_nwb.py:519/566`; NOT `write_units_nwb`) — add unit columns from `Sorting.Unit` data: `peak_amplitude_uv`, `peak_electrode_id`, `n_spikes`, and `brain_region` (via `Sorting.Unit * Electrode * BrainRegion`). Add file-level provenance scratch: source `recording_id`/`concat_recording_id`, `sorter`, sorter params, `artifact_detection_id`, `display_waveform_params_name`, and the phase-3a versions/seed. **Ordering caveat — `Sorting.Unit` does not exist yet at staging.** The sort-time units NWB is staged in `make_compute` (`_stage_sorting_artifact`, `sorting.py:1565`), but the per-unit metadata is computed (`build_sorting_unit_rows`) and inserted into `Sorting.Unit` only later in `make_insert`/`_populate_unit_part` (`sorting.py:2647-2657`). So you cannot read it "from `Sorting.Unit`" at write time. **Compute the per-unit metadata once before NWB staging** (move/share the `build_sorting_unit_rows` computation so it runs in `make_compute` before `_stage_sorting_artifact`), **pass it into `write_sorting_units_nwb`**, and then **reuse the same rows for `Sorting.Unit.insert`** in `make_insert` (don't recompute). This is a writer signature extension *plus* a compute-ordering change; the same thread-DB-resident-values pattern recurs in tasks 3, 4, 6.

3. **Curated Units NWB: merge lineage.** In the curated writer (`_units_nwb.py` ~864-945), serialize the kept→contributor mapping (`CurationV2.MergeGroup`) as a scratch table so the merge lineage is in the file (currently reconstructed only from the DB). For `apply_merge=True` write the kept-unit→contributors map; for `apply_merge=False` write the proposed groups with an explicit "proposed, not applied" flag.

4. **UnitMatch NWB self-description.** In `_unitmatch_nwb.build_pairs_table` (or a companion scratch alongside it), add: `unitmatch_id`, `session_group_name`, `matcher_params_name`, the per-member `(member_index, sorting_id, curation_id, session_start_time)` map, **and the producer provenance fields from the shared contract** (`matcher_backend`, `matcher_backend_version`, `spikeinterface_version`) — re-emitting the same values phase-3a stores on the `UnitMatch` row. The pairs table currently carries only per-pair side ids (`_unitmatch_nwb.py:30-38`); add the session/params/provenance context so the file is interpretable standalone.

5. **CurationEvaluation NWB inputs.** In the metric-curation result writer
   (`write_analyzer_curation_tables` today, renamed if phase-1c moves it to a
   `CurationEvaluation`-named helper), add the metric param set + kwargs, the
   display/metric recipe names, the auto-merge preset/rules, and the evaluated
   `sorting_id`/`curation_id` as scratch metadata alongside the existing result
   tables -- **and re-emit the source provenance phase-3a (ALSC-5) stores on the
   `CurationEvaluation` row**: source analyzer recipe/manifest/hash for raw-fast-
   path evaluations, curation-unit-set identity for merged temp-analyzer
   evaluations, the sorting/recording `content_hash`, and the
   `spikeinterface_version`. (Same row↔NWB consistency as the UnitMatch case.)

6. **Concat NWB reconstructability + large-array params.** In the concat write (`session_group.py` ~876-884; note it receives only a `filtering_description` string today, so this is another signature extension threading `SessionGroup.Member`/`MemberBoundary` data in), write the ordered member provenance (per member: source NWB, `recording_id`, interval, frame start/end) and the `MemberBoundary` data **into the file**, so a split is reconstructable from the NWB (not only from live `SessionGroup.Member`). Note: there is **no persisted per-member "time transform"** — only the cumulative `end_sample` in `MemberBoundary`; `split_sorting_by_session` reconstructs the mapping arithmetically. Write the **frame boundaries** (from which the transform is derived), not a non-existent transform. For the **motion** large-array item (NWB-3): write the resolved motion **preset + kwargs** (from phase-2's stored resolved preset), NOT the displacement field — document the field as DB/derivable. Same principle for waveform/template arrays elsewhere: write the producing params, not the arrays.

7. **Docs.** CHANGELOG: each writer now embeds provenance; note the scratch container names so downstream readers can find them. Add a short "what provenance is in each v2 NWB" subsection to `docs/src/.../SpikeSortingV2.md`.

## Deliberately not in this phase

- **Storing the motion displacement field or waveform/template arrays in NWB** — decided: store the producing params/preset, document the array as derivable (avoids bloating every NWB).
- **Moving the processed recording out of `acquisition`** (NWB-10) — cosmetic, out of scope.
- **Row-level provenance columns** — phase-3a; this phase only *writes* those values into the NWB.
- **Changing any identity** — provenance only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_nwb_provenance.py::test_recording_nwb_carries_source_provenance` (new file — note `tests/spikesorting/v2/test_recording.py` does NOT exist; recording read-back tests live in `test_recording_provenance.py`/`test_disjoint_readback.py`) | after `Recording.populate`, the artifact NWB provenance scratch carries the **full task-1 set**: raw source `object_id`, `recording_id`, `preprocessing_params_name`, **`sort_group_id`, resolved reference mode, bad-channel handling**, and `spikeinterface_version` (assert each, not a subset). |
| `test_nwb_provenance.py::test_units_nwb_carries_per_unit_and_source_metadata` (new) | the sort-time Units NWB has `peak_amplitude_uv`/`peak_electrode_id`/`n_spikes`/`brain_region` unit columns matching `Sorting.Unit`, AND the **full task-2 source scratch**: `recording_id`/`concat_recording_id`, `sorter` + sorter params, `artifact_detection_id`, `display_waveform_params_name`, and the phase-3a versions/seed (assert each). |
| `test_nwb_provenance.py::test_curated_nwb_carries_merge_lineage` (new) | a merged curation's NWB contains the kept→contributor map matching `CurationV2.MergeGroup`; a preview curation marks groups "not applied". |
| `test_nwb_provenance.py::test_unitmatch_nwb_self_describes` (new) | the UnitMatch NWB carries `unitmatch_id`/`session_group_name`/`matcher_params_name` + the member map, **plus the producer provenance fields from the shared contract** (`matcher_backend`, `matcher_backend_version`, `spikeinterface_version`) so the file re-emits the same values phase-3a stored on the row. |
| `test_nwb_provenance.py::test_curation_evaluation_nwb_carries_inputs` (new, task 5) | the CurationEvaluation NWB carries the metric param set + kwargs, the display/metric recipe names, the auto-merge preset/rules, evaluated `sorting_id`/`curation_id`, **and the phase-3a source provenance** (raw-fast-path analyzer recipe + manifest/hash or merged-curation unit-set identity, sorting/recording `content_hash`, `spikeinterface_version`) alongside the result tables. |
| `test_nwb_provenance.py::test_concat_nwb_reconstructs_member_boundaries` (new) | the concat NWB's embedded member provenance + boundaries reproduce `split_sorting_by_session`'s mapping without reading live `SessionGroup.Member`; the resolved motion preset+kwargs are present, the displacement field is not. |
| (regression) `test_recording.py`, `single_session/test_curation_*`, `test_unitmatch.py`, `test_session_group_concat.py` round-trip/read tests | existing read paths unaffected by the added scratch. |

## Fixtures

Reuse `populated_sorting` (`conftest.py:215`) and `populated_sorting_with_curation`
(`conftest.py:312`) for recording/units; `two_session_curated_group`
(`test_unitmatch.py:508`) for UnitMatch; `chronic_2_session_minirec`
(`conftest.py:340`) for concat. Each test reads the produced NWB via `pynwb` and
asserts the provenance container contents.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Each writer embeds the specified provenance; the round-trip tests read it back from the file (not the DB).
- Large arrays (motion field, waveforms) are NOT written — only their producing params; the docstring/CHANGELOG says where the array is derivable.
- Provenance field names match phase-3a / shared-contracts exactly.
- Added scratch does not break existing NWB read paths (regression tests pass); the data path (ElectricalSeries, spike trains) is unchanged.
- Docs list what provenance lives in each v2 NWB; no plan/phase references in code or tests.
