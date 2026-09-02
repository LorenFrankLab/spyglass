# Phase 1 — Fix PR #1609's master-user regressions, v2 correctness gaps, and doc contradictions

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Ships as commits on the existing `spikesorting-v2` branch (PR #1609). One commit per lettered task below; the PR squash-merges, but per-task commits keep the history reviewable. Work in the `spyglass_spikesorting_v2` conda env for v2 tests and `spyglass_spikesorting_legacy` for v0/v1 tests (see the `spikesorting-v2-local-test-env` and `spyglass-master-branch-test-recipe` memories for the Colima/`DOCKER_HOST` recipe; `pytest -p no:xvfb`). Formatting gate is `uvx black@26.5.1 --line-length 80` plus `uvx ruff@0.14.13 check`.

**Inputs to read first:**

- [overview.md § Current codebase integration points](overview.md#current-codebase-integration-points) — every file:line this phase edits, verified.
- [overview.md § Decisions already taken](overview.md#decisions-already-taken-do-not-re-litigate) — raise-vs-warn, no deprecation, `detect_sign=-1` is correct, single PR.
- `src/spyglass/spikesorting/_legacy_runtime.py` — the guard whose docstring promise this phase makes true.
- `src/spyglass/utils/dj_merge_tables.py:587-812` — `fetch_nwb` resolution paths.
- `src/spyglass/spikesorting/v2/curation.py:1290-1395` — `insert_curation` transaction block.
- `tests/utils/test_merge_consumer_boundary.py` — the hermetic two-source merge fixture to extend.

**Designs referenced:** [si-compat-shim](designs.md#si-compat-shim), [es-selection](designs.md#es-selection), [multi-source-raise](designs.md#multi-source-raise), [export-file-retention](designs.md#export-file-retention), [unitannotation-migration](designs.md#unitannotation-migration), [concat-merge-gate](designs.md#concat-merge-gate), [unitmatch-baseline](designs.md#unitmatch-baseline), [file-tracking-lazy-v2](designs.md#file-tracking-lazy-v2).

## Tasks

### A. Master-user regressions (highest priority, all verified)

- **A1. SpikeInterface compat shim for v0/v1 read paths.** Create `src/spyglass/spikesorting/_si_compat.py` per [si-compat-shim](designs.md#si-compat-shim). Route the ten `si.load_extractor` / `from_times_labels` sites through it (list in overview). Do NOT remove any `_require_legacy_si_environment` guard. Then perform the manual cross-env folder check described in the design on one real v0 recording folder and one v0 sorting folder; record the outcome in the commit message. If 0.104 cannot open 0.99 folders, add the wrapped `RuntimeError` described there. Restore CI coverage: revert the `tests/utils/test_merge.py:90-94` change so `test_merge_get_class_invalid` uses `pop_spike_merge` again ONLY if the v1 pipeline can now run under 0.104 (it cannot: populate is guarded), otherwise leave it and instead add `test_v0_v1_read_paths_under_modern_si` (see validation slice). Update `_legacy_runtime.py:12-17` docstring to name the shim as the mechanism that keeps read paths working.
- **A2. ElectricalSeries selection consistent with `Raw`.** Implement [es-selection](designs.md#es-selection): constant + sanitizer in `nwb_helper_fn.py`, `common_ephys.py:299-304` reads the constant, `ingestion.py:200-203` delegates, selection body refuses zero/multiple matches with the `electrical_series_path` hint. Delete the incorrect comment at `nwb_helper_fn.py:386-388`.
- **A3. `Merge.fetch_nwb` raises on multi-source.** Implement [multi-source-raise](designs.md#multi-source-raise). Remove `disable_warning` from the signature and docstring (`dj_merge_tables.py:591`, `:611-616`). Pass `multi_source=True` in `SpikeSortingOutput.get_spike_times` (`spikesorting_merge.py:497`). Grep the repo for any other `fetch_nwb(` caller on a merge table that could span sources (`LFPOutput`, `PositionOutput`, `DecodingOutput`) and confirm each restricts to one source or opts in.
- **A4. `Export.File` retention on re-export.** Implement [export-file-retention](designs.md#export-file-retention) at `common_usage.py:544-547`.
- **A5. `UnitAnnotation` audit + one-time migration.** Implement [unitannotation-migration](designs.md#unitannotation-migration) on `unit_annotation.py`. Add the migration call to the CHANGELOG release-notes alter block (`CHANGELOG.md:3-31`):
  ```python
  # UnitAnnotation.unit_id now stores the NWB unit id (was a positional index).
  # Run ONCE, before writing new annotations:
  from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation
  UnitAnnotation.audit_positional_unit_ids()               # inspect
  UnitAnnotation.migrate_positional_unit_ids(dry_run=False)  # apply once
  ```
  Also add a `## Breaking Changes` bullet explaining the semantic change and that dense-id curations are unaffected.
- **A6. Zero-unit guard on `SortedSpikesGroup.fetch_spike_data`.** At `group.py:205-212`, skip the file when `"spike_times" not in nwb_file[nwb_field_name]` (mirror `spikesorting_merge.py:504-509`) instead of indexing it.

### B. v2 correctness

- **B1. Concat merge-registration gate.** Implement [concat-merge-gate](designs.md#concat-merge-gate): `CONCAT_MERGE_GATE_MESSAGE`, the `register_merge` branch in `insert_curation`, `CurationV2.audit_concat_merge_rows`, the `_pipeline_run.py:804-807` / `:869-871` `fetch` → optional handling, `root_merge_id: "UUID | None"` at `_pipeline_types.py:121`, docstring at `_pipeline_run.py:330-351`, and a `run_summary["warnings"]` entry. Rewrite `tests/spikesorting/v2/test_session_group_concat.py:1480-1521` to assert no `SpikeSortingOutput.CurationV2` row exists for a concat curation, the warning is logged, and `audit_concat_merge_rows()` returns it once a row is force-inserted for the test. Add a caveat bullet to `docs/src/Features/SpikeSortingV2.md:1090-1100` ("Concat sorts are not registered in `SpikeSortingOutput` yet; per-member decodable rows are a planned addition") and a CHANGELOG entry that tells trial users to run `audit_concat_merge_rows()` and delete the listed rows.
- **B2. UnitMatch baseline window.** Replace `_zero_center` per [unitmatch-baseline](designs.md#unitmatch-baseline).
- **B3. `check_all_files` must not declare v2 schemas.** Implement [file-tracking-lazy-v2](designs.md#file-tracking-lazy-v2).
- **B4. Correct stale claims in docstrings.** (a) "DB-free"/"no DB" claims about `make_compute` at `sorting.py:1552`, `:1750-1753`, `:2963` and `metric_curation.py:999`, `:1011`, `:1229`: reword to "no DB writes; upstream INPUTS are resolved in `make_fetch`; `get_recording` / `AnalysisNwbfile.create` perform reads". Do not relocate the reads. (b) Replace `ArtifactDetectionSelection` / `SharedGroupSource` with the live names at `_selection_identity.py:4,216,223,274,321` and `exceptions.py:388`. (c) `sorting.py:2518-2545` `find_orphaned_analyzer_folders` docstring: name the supported cascade path (a delete starting at `Recording`, `RecordingSelection`, or `SortGroupV2` cascades through DataJoint `FreeTable`s and never runs `Sorting.delete`) as the common cause, not only raw SQL.

### C. Docs, params, notebooks

- **C1. `detect_sign` prose.** `docs/src/Features/SpikeSortingV2.md:431-432` and `notebooks/py_scripts/10_Spike_SortingV2.py:168-170`: state that Frank-lab MountainSort rows use downward-only `detect_sign=-1` (matching v0/v1 defaults) and that a bidirectional `0` row can be cloned with `clone_pipeline_preset`. Regenerate the paired `.ipynb` with `jupytext --to notebook` (see `notebooks/README.md:82-88`; the docs copy is a symlink).
- **C2. One auto-curation story.** Resolve [overview Open Question 1](overview.md#open-questions) with the owner first. Then either (recommended) point the three `franklab_*` presets at `franklab_default_auto_curation_2026_06` (`_recipe_catalog.py:559,671,793`) and update `tests/spikesorting/v2/test_pipeline_presets.py` expectations, or leave presets and change the Curation notebook (`10_Spike_SortingV2_Curation.py:244-259`) and doc (`SpikeSortingV2.md:626-630`) to stop calling the ISI set "the default". In both cases add one sentence to the Quickstart (`SpikeSortingV2_Quickstart.md:38-46`) naming the rule set that `auto_curate=True` applies and how to see it (`describe_pipeline_preset`).
- **C3. Quickstart fixes.** Replace the hardcoded `sort_group_id=0` at `SpikeSortingV2_Quickstart.md:40` with a `describe_sort_groups`-driven selection (mirror `10_Spike_SortingV2.py:129-134`); add the Quickstart to `docs/src/Features/index.md:19-24`.
- **C4. Doc scope line.** `docs/src/Features/SpikeSortingV2.md:19`: "v2 ships the single-session chain plus same-day concatenation and cross-session matching" (or equivalent) so it matches the diagrams below it.
- **C5. Storage doc: deletion cascades.** Add a section to `docs/src/Features/SpikeSortingV2StorageManagement.md` (after `## The deletion gate`, `:69`) explaining that upstream deletes bypass `Sorting.delete`'s folder cleanup and that `Sorting.find_orphaned_analyzer_folders()` is the reclaim path after any delete that did not start at `Sorting`.
- **C6. CHANGELOG.** Under `### Breaking Changes` (`:33`): `Merge.fetch_nwb` multi-source raise + removed `disable_warning`; raw `ElectricalSeries` selection now name-filtered and refuses ambiguity; `UnitAnnotation.unit_id` semantics (with the migration pointer from A5); concat curations not registered in `SpikeSortingOutput` (with `audit_concat_merge_rows`); package-wide SI 0.104 pin means v0/v1 populate needs the legacy env while reads work via the shim. Under the v2 section: UnitMatch baseline derivation, `check_all_files` no longer declares v2 schemas.
- **C7. PR description refresh.** `gh pr edit 1609 --body-file <file>` after regenerating the body: artifact stage is `RecordingArtifact*` / `SharedGroupArtifact*` behind `ArtifactDetectionOutput`; 10 schemas, 37 master tables; the "Behavior changes that affect NON-v2 users" list gains the rollout consequence of the pin (second env for v1 sorting), the ES ambiguity raise, the `Export.File` retention rule, and the ride-along changes (`get_nwb_file` subdir, `dj_graph` bridge skip, `v1/recompute` env match) so reviewers see them; "Open questions" replaced by the decision (raise); "Known issues" gains the concat gate and Phase 2 pointer plus the follow-ups (preflight parity, preset SoT, `get_spike_times` query count, cascade-delete folder leak).

### D. Environments

- **D1. DLC / MoSeq env files.** `environment_dlc.yml`, `environment_moseq_cpu.yml`, `environment_moseq_gpu.yml` (`:26,33,35`): mirror `environment.yml` (`numpy>=2,<3`, `scipy>=1.13`, drop the SI-0.99 comment). Attempt `conda env create -n tmp-dlc -f environments/environment_dlc.yml --dry-run` (or `mamba`). If `pytorch<1.12.0` blocks the solve, keep the pins as they are and add a header block (same wording as `environment_spikesorting_legacy.yml:1-20`) stating these envs require the legacy `sed` on `pyproject.toml` before `pip install -e .`, and record the outcome in [overview Open Question 3](overview.md#open-questions). Extend `tests/spikesorting/v2/test_dependency_contract.py` to assert that every `environments/*.yml` either pins `scipy>=1.13` or carries the legacy-sed header comment.

## Deliberately not in this phase

- Per-member decodable rows for concat sorts → [phase 2](phase-2-concat-member-curations.md). Phase 1 only gates and audits.
- Merged-curation visualization, curation write-verb consolidation, FigPack identity verification, NaN rule policy → `.claude/docs/plans/curation-ux-overhaul/`.
- Preflight calling the selection-plan builders; recipe catalog as source of truth for all lookups; `get_spike_times` query count; `TrackedUnit.Member` partition PK → listed as follow-ups in the PR description (C7), not fixed here.
- Automatically cleaning analyzer folders on upstream cascade deletes → documented (B4c, C5) only. Trigger to build: a second lab report of a leaked folder after a `Recording`/`SortGroupV2` delete.
- `ImportedSpikeSorting` waveform features → not a regression (see overview Non-Goals).
- Reverting the `get_nwb_file` subdirectory change, the `dj_graph` bridge skip, or the `v1/recompute` env-match loosening → surfaced in C7 for reviewer judgment only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/test_legacy_modern_si_coexistence.py::test_v0_v1_read_paths_under_modern_si` (new) | Under SI 0.104: `_si_compat.load_extractor` dispatches to `si.load`; `numpy_sorting_from_samples_and_labels` returns a `NumpySorting` with the given unit ids; `SpikeSortingRecording().load_recording` and `Curation.get_curated_sorting` reach the shim (monkeypatch the shim, assert called) rather than raising `AttributeError`. |
| same module, legacy env (`pytest-legacy` job) | Under SI 0.99: shim dispatches to `load_extractor` / `from_times_labels`; existing v0/v1 tests unchanged. |
| `tests/utils/test_nwb_helper_fn.py::test_get_raw_eseries_path_prefers_named_series` (new) | File with `analog-series` + `e-series` under acquisition returns `acquisition/e-series`. |
| `tests/utils/test_nwb_helper_fn.py::test_get_raw_eseries_path_ambiguous_raises` (new) | File with `e-series` + `ephys` raises `ValueError` naming both and the `electrical_series_path` override. |
| `tests/utils/test_nwb_helper_fn.py::test_get_raw_eseries_path_unnamed_raises` (new) | Single acquisition series named `wideband` raises (matches `Raw`, which would ingest nothing). |
| `tests/common/test_ephys.py` (existing `Raw` ingestion test) | Still ingests the fixture's `e-series` after the constant refactor. |
| `tests/utils/test_merge_consumer_boundary.py::test_fetch_nwb_multi_source_raises_without_opt_in` (renamed from `:147`) | `pytest.raises(ValueError, match="spans 2 sources")`; `multi_source=True` returns both files (`:168` unchanged). |
| `tests/utils/test_merge_consumer_boundary.py::test_fetch_nwb_has_no_disable_warning_kwarg` (new) | `inspect.signature(Merge.fetch_nwb)` lacks `disable_warning`. |
| `tests/common/test_usage.py::test_reexport_keeps_dandi_referenced_files` (new) | Insert a `DandiPath` child for a superseded export's `File` row, re-populate `Export`; populate succeeds, the `File` row survives, the `Table` rows are removed, the log line names `DandiPath`. |
| `tests/common/test_usage.py::test_reexport_deletes_unreferenced_files` (new) | Without children, superseded `File` rows are removed. |
| `tests/spikesorting/v1/test_unit_annotation_migration.py` (new, marks `slow` if it needs the v1 fixture; otherwise build a synthetic Units NWB with ids `[2,3,4,10]`) | `audit_positional_unit_ids` returns the merge id with `true_unit_ids=[2,3,4,10]`; dense-id NWB is not listed; `migrate_positional_unit_ids(dry_run=True)` returns `{mid: {0:2, 1:3, 2:4, 3:10}}` and writes nothing; `dry_run=False` rewrites rows and preserves `Annotation` payloads; a stored `unit_id=7` (≥ n_units) aborts with `ValueError` and no rows change. |
| `tests/spikesorting/v1/test_analysis.py` | Unskip `test_analysis_units` if the JAX issue no longer reproduces in the legacy env; otherwise leave and note in commit. |
| `tests/spikesorting/analysis/...::test_fetch_spike_data_skips_zero_unit_file` (new) | A Units table with no `spike_times` column contributes nothing and does not raise. |
| `tests/spikesorting/v2/test_session_group_concat.py` (rewrite `:1480-1521`) | Concat curation creates NO `SpikeSortingOutput.CurationV2` row; `caplog` contains `CONCAT_MERGE_GATE_MESSAGE` prefix; `run_v2_pipeline(concat...)` summary has `root_merge_id is None` and a warning; `audit_concat_merge_rows()` lists a force-inserted row. Marked `integration`. |
| `tests/spikesorting/v2/test_unitmatch.py::test_zero_center_window_scales_with_spike_width` (new) | For `spike_width=18` the baseline uses 4 samples and the peak sample is unchanged after centering; for 90, 22 samples; a constant DC offset is removed exactly. |
| `tests/spikesorting/v2/test_recompute.py::test_v2_recompute_schema_name_literal` (new) | `recompute.schema.database == "spikesorting_v2_recompute"`. |
| `tests/common/test_file_tracking.py::test_check_all_files_does_not_declare_v2_schema` (new, main job) | With v2 schema absent, `_get_v2_deleted_files()` returns empty and `dj.list_schemas()` gains no `spikesorting_v2_*` entry. |
| `tests/spikesorting/v2/test_pipeline_presets.py` | Preset `auto_curation_rules_name` expectations match the C2 decision; `describe_pipeline_preset` shows it. |
| `tests/spikesorting/v2/test_dependency_contract.py::test_env_files_agree_with_pin` (new) | Every `environments/*.yml` either pins `scipy>=1.13` or carries the legacy-sed header. |
| Docs/notebook check (manual) | `grep -n "detect_sign=0"` returns nothing in docs/notebooks; `jupytext --to notebook` regenerated `.ipynb` matches its `.py`; `mkdocs build` clean. |

## Fixtures

- Multi-acquisition NWB files: synthesize in `tests/utils/test_nwb_helper_fn.py` with pynwb (two `ElectricalSeries` under acquisition with distinct names; reuse the fixture style at `:70-83`).
- Sparse-id Units NWB for the migration test: build a minimal NWB with a Units table whose `id` column is `[2,3,4,10]`; insert a `SpikeSortingOutput.ImportedSpikeSorting` row pointing at it via the existing `tests/decoding/conftest.py` synthetic-import pattern (`:111-210`), or write the file to an `AnalysisNwbfile` and register directly.
- Two-source merge: existing `two_source_merge` fixture in `tests/utils/test_merge_consumer_boundary.py`.
- DANDI child rows: insert a `DandiPath` row directly (no network) referencing the superseded `Export.File` row created by the existing `populate_export` fixture.
- Concat smoke group: existing fixtures in `tests/spikesorting/v2/test_session_group_concat.py`.

## Review

Before pushing the final commit of this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against `git diff <phase-start>..HEAD`. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (`_warn_multi_source`, `disable_warning`, the `nwb_helper_fn.py:386-388` comment, the literal `15` baseline).
- User-facing documentation listed as tasks is updated, not deferred; the PR description (C7) reflects the code as it is after this phase.
