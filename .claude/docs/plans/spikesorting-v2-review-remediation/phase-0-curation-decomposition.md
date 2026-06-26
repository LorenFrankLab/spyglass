# Phase 0 — Decompose `CurationV2` source-routing and accessor cores into DB-free service modules

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Behavior-preserving extraction (R38 / MAINT-1, MAINT-2). The goal is **clean seams
before the Phase 5 UX overhaul extends these methods**, not a line-count reduction.
Apply the anti-theater discipline: extract only genuinely pure logic that becomes
directly unit-testable; leave irreducible DataJoint routers on the table. No public
API changes — every public classmethod stays as a thin wrapper, so no call site is
repointed.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/curation.py:1415-1668](../../../../src/spyglass/spikesorting/v2/curation.py#L1415-L1668) — `resolve_restriction`: pure key-classification half is **1511-1595**, DataJoint join assembly is **1597-1668**.
- [src/spyglass/spikesorting/v2/curation.py:1102-1187](../../../../src/spyglass/spikesorting/v2/curation.py#L1102-L1187) — `summarize_curation`: pure PK-validation + return-dict + `is_merge_preview` derivation; DB fetches at 1157-1180.
- [src/spyglass/spikesorting/v2/_curation_plan.py](../../../../src/spyglass/spikesorting/v2/_curation_plan.py) and [_curation_transforms.py](../../../../src/spyglass/spikesorting/v2/_curation_transforms.py) — the established DB-free service-module pattern (module docstring states "opens no database connection and activates no `dj.schema` at import"; lazy SI/pynwb imports inside functions).
- [tests/spikesorting/v2/test_service_import_contracts.py:20-49](../../../../tests/spikesorting/v2/test_service_import_contracts.py#L20-L49) — the `_DB_FREE_SERVICE_MODULES` list a new module must join.

**Contracts referenced:** none (behavior-preserving; no schema/identity change).

## Tasks

1. **Extract the `resolve_restriction` pure half into a new DB-free module `_curation_routing.py`.** Create `src/spyglass/spikesorting/v2/_curation_routing.py` following the `_curation_plan.py` header convention (no `spyglass.common`, no `utils` heavy barrel, no `dj.schema` at import; stdlib + `_enums` only). Move the key-classification / unknown-key / artifact-interval-name / uuid-normalization logic (`curation.py:1511-1595`) into:

   ```python
   class RestrictionPlan(NamedTuple):
       rec_restriction: dict        # may be {}
       concat_restriction: dict     # may be {}
       sort_restriction: dict       # may be {}
       curation_restriction: dict   # may be {}
       artifact_directive: tuple    # ("exclude" | "include" | None, artifact_detection_id | None)

   def classify_and_normalize_restriction(
       key: dict, *, restrict_by_artifact: bool, strict: bool,
   ) -> "RestrictionPlan | None":
       """Classify a user restriction into per-source dicts + an artifact directive.

       Returns None when strict=False and the key carries an unrecognized field
       (the lenient bail-out the table method turns into a None return). Raises
       ValueError on an unrecognized field when strict=True, and on a contradictory
       concat+recording restriction. Pure: no DataJoint, no DB.
       """
   ```

   **Prerequisite (task 1a):** `parse_artifact_detection_interval_list_name` is currently defined **only in `utils.py:653`** (the DB-heavy barrel — `utils.py:12-13` imports `datajoint`+`spikeinterface` at module top), so `_curation_routing` cannot import it without failing the import-boundary test in task 3. First extract `parse_artifact_detection_interval_list_name`, its inverse, and the `_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX` constant (`utils.py:635-664`) into a small DB-free module (a new `_artifact_naming.py`, or fold into existing DB-free `_curation_transforms.py`), re-exported from `utils` for back-compat, and add it to `_DB_FREE_SERVICE_MODULES`. Then `_curation_routing` imports the parser from that DB-free module. `resolve_restriction` (the classmethod) then calls `classify_and_normalize_restriction(...)`, returns `None` if it returns `None`, and assembles the DataJoint joins (current `curation.py:1597-1668`) from the returned dicts. Behavior is identical.

2. **Extract the `summarize_curation` pure formatter into `_curation_plan.py`.** Add:

   ```python
   def build_curation_summary(
       *, sorting_id, curation_id, merges_applied, description,
       labels: dict, merge_id, merge_groups: dict, n_units: int,
   ) -> dict:
       """Assemble the summarize_curation return dict, including the
       is_merge_preview derivation (not merges_applied and any group has >1
       contributor). Pure."""
   ```

   `summarize_curation` keeps its PK-presence `ValueError` (it must raise before any DB access — see test below), performs its fetches (`curation.py:1157-1180`), then returns `build_curation_summary(...)`. The `is_merge_preview` logic (`curation.py:1173-1175`) moves into the helper.

3. **Register the new module(s) for the import-boundary contract.** Add `"_curation_routing"` (and confirm `_curation_plan` is already present) to `_DB_FREE_SERVICE_MODULES` in `tests/spikesorting/v2/test_service_import_contracts.py:20-49`.

4. **Add DB-free unit tests** for both extracted functions (new `tests/spikesorting/v2/test_curation_routing.py`; extend `test_curation_plan.py` for `build_curation_summary`). See validation slice.

## Deliberately not in this phase

- **`get_recording` (`curation.py:1191-1240`) and `get_sort_metadata` (`1670-1702`) are NOT extracted** — they are thin DataJoint source-routers (`fetch1` + `resolve_source` + a `Recording`/`ConcatenatedRecording` cache read). Extracting them would move DB calls into a "service" module or thread table classes as args — pure theater. They stay on the table.
- **`get_sorting` (`1263-1379`) / `get_merged_sorting` (`1754-1829`) compute is NOT re-extracted** — it already delegates to DB-free `_units_nwb` / `_signal_math` helpers. The residual is fetch-orchestration interleaved with `logger.warning` calls that `test_preview_merge_warning.py` monkeypatches via `curation_mod.logger.warning`; moving the warning emission would break that patch for no testability gain. Leave these methods on the table.
- No preview-guard behavior change here — that is phase-1 (R3). This phase must be a **pure refactor**: the preview warning in `get_sorting` stays exactly as-is.
- No new columns, no identity change.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_curation_routing.py::test_classify_rejects_unknown_key_when_strict` | `classify_and_normalize_restriction({"bogus": 1}, restrict_by_artifact=True, strict=True)` raises `ValueError`; `strict=False` returns `None`. |
| `test_curation_routing.py::test_classify_splits_recording_concat_sort_curation_keys` | a key mixing recording + curation fields classifies into the right per-source dicts; a concat+recording key raises (contradictory). |
| `test_curation_routing.py::test_classify_maps_artifact_interval_name_to_directive` | a `restrict_by_artifact` interval-list name normalizes to `("exclude"/"include", artifact_detection_id)`; a raw `artifact_detection_id` uuid string normalizes to a `UUID`. |
| `test_curation_plan.py::test_build_curation_summary_is_merge_preview` | `build_curation_summary(merges_applied=False, merge_groups={0:[1,2]}, ...)["is_merge_preview"] is True`; `merges_applied=True` → `False`; single-contributor groups → `False`. |
| `test_service_import_contracts.py::test_service_module_imports_without_db_layer[_curation_routing]` | `_curation_routing` cold-imports in a subprocess without pulling `spyglass.common` or any v2 schema module. |
| (regression — must pass UNCHANGED) `test_merge_id_selectivity.py::test_get_restricted_merge_ids_discriminates_by_recording`, `test_session_group_concat.py` resolve-restriction cases, `single_session/test_merge_dispatch.py::test_get_restricted_merge_ids_v2_resolves_through_chain` and `::test_merge_dispatch_restrict_by_artifact_honored_in_v2` | `resolve_restriction` routing behavior is byte-identical after extraction. |
| (regression) `test_curation_wrappers.py::test_summarize_curation_fields`, `::test_summarize_unregistered_merge_id_none`, `::test_summarize_curation_requires_full_pk` | `summarize_curation` output + the PK-`ValueError`-before-DB contract are unchanged. |
| (regression) `single_session/test_merge_dispatch.py::test_merge_dispatch_get_recording_works_for_v2`, `test_preview_merge_warning.py::test_get_sorting_warns_on_unapplied_preview_merge` | `get_recording` / `get_sorting` (left on the table) behave identically; the preview warning still fires through the monkeypatched logger. |

Mark the merge-dispatch / session-group / preview tests as integration (they need `dj_conn` + `populated_sorting` / `chronic_2_session_minirec`).

## Fixtures

The new DB-free unit tests need **no** fixtures — call the extracted functions
directly with plain dicts. The regression integration tests reuse the existing
`populated_sorting` (`conftest.py:215`), `populated_sorting_with_curation`
(`conftest.py:312`), and `chronic_2_session_minirec` (`conftest.py:340`) fixtures.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Every task is implemented as specified; the extraction is behavior-preserving (all regression tests pass unchanged).
- The "Deliberately not in this phase" list is honored — `get_recording`/`get_sort_metadata` are not extracted, `get_sorting`/`get_merged_sorting` keep their on-table warning emission, and no preview-guard behavior changed (that's phase-1).
- `_curation_routing` is genuinely DB-free (import-boundary test passes); it imports the artifact-name parser from a DB-free module, not `utils`.
- New DB-free tests exercise real branching (unknown-key strict/lenient, source classification, artifact directive, is_merge_preview), not tautologies; shared setup is in fixtures.
- Docstrings / test names / module names don't reference this plan or "phase 0".
- No call sites needed repointing (public classmethods unchanged); confirm by grepping callers of `resolve_restriction` / the five accessors.
