# Phase 4b — Execution mismatches, dependency pins, security, cleanup

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

A batch of localized hardening fixes (R32, R16, R34, R14, R15, R18, R22, R26, plus
the R9 TEAM-1 footgun). Each is small and independent; grouped into one PR because
none warrants its own.

**Inputs to read first:**

- Dispatch: [src/spyglass/spikesorting/v2/sorting.py:328-331](../../../../src/spyglass/spikesorting/v2/sorting.py#L328-L331), [:2429-2449](../../../../src/spyglass/spikesorting/v2/sorting.py#L2429-L2449), [:504-558](../../../../src/spyglass/spikesorting/v2/sorting.py#L504-L558); [_sorting_dispatch.py:75-88](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L75-L88), [:504-515](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L504-L515).
- Deps: [pyproject.toml:58,69](../../../../pyproject.toml#L58); [environments/environment_spikesorting_v2.yml:56](../../../../environments/environment_spikesorting_v2.yml#L56).
- Security: [src/spyglass/utils/mixins/analysis.py:205-227,300-302,316-318,341-348](../../../../src/spyglass/utils/mixins/analysis.py#L300-L302); [_sorting_dispatch.py:486](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L486); [common/common_nwbfile.py:107](../../../../src/spyglass/common/common_nwbfile.py#L107); v2 writers `_recording_nwb.py:172-175`, `_units_nwb.py:543,735`.
- DB guard: [metric_curation.py:73](../../../../src/spyglass/spikesorting/v2/metric_curation.py#L73), [recompute.py:67](../../../../src/spyglass/spikesorting/v2/recompute.py#L67); contrast [recording.py:90-91](../../../../src/spyglass/spikesorting/v2/recording.py#L90-L91).
- Merge probe: [spikesorting_merge.py:36-44,127-134](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L36-L44).
- Footguns: [recompute.py:1216-1217](../../../../src/spyglass/spikesorting/v2/recompute.py#L1216-L1217); [sorting.py:2274-2278](../../../../src/spyglass/spikesorting/v2/sorting.py#L2274-L2278); [_recording_nwb.py:258-275](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py#L258-L275); [common/common_usage.py:547-550](../../../../src/spyglass/common/common_usage.py#L547-L550).
- TEAM-1: [recording.py:249-285](../../../../src/spyglass/spikesorting/v2/recording.py#L249-L285) (`SortGroupV2._handle_existing`).

**Contracts referenced:** none.

## Tasks

1. **R32-DISP-1 — reject non-local `execution_params` for in-process sorters.** `SorterParameters` validation stores `execution_params` for every sorter (`sorting.py:328-331`), but clusterless routes without it (`:2429-2449`) and preflight then falsely requires a container. In the insert validation, if the sorter is in `_NON_SI_SORTERS` (clusterless), require `execution_params["backend"] == "local"` (or absent) and raise otherwise — so a clusterless row cannot claim a container that runtime ignores.

2. **R32-DISP-3 — legacy seeder must not create unrunnable MATLAB rows.** `insert_default_legacy_si_sorters` (`sorting.py:504-558`) skips curated sorters but inserts local `default` rows for MATLAB sorters that dispatch then rejects (`_sorting_dispatch.py:75-88`). Either skip `MATLAB_SORTERS` in the seeder, or give those rows a container `execution_params` so they are runnable. Prefer skip (the lab inserts MATLAB rows explicitly with a container).

3. **R32-DISP-4 — allowlist the external-whiten interception to the sorters that rely on it.** `_sorting_dispatch.py:504-515` intercepts **any** truthy `whiten`. **The interception must KEEP firing for MountainSort4/5**, which deliberately carry `whiten=True` and route to external float64 whitening (`_params/sorter.py:457-460`). Do NOT key the allowlist on `_INTERNAL_WHITEN_NO_KWARG_SORTERS` (kilosort2_5/3/ironclust, `_params/sorter.py:467`) — those *reject* a truthy `whiten` at insert (`reject_internal_whiten`, `sorting.py:307`) and can never reach the dispatcher with `whiten=True`, so allowlisting them would disable interception for exactly the wrong set. Define the allowlist as "sorters that use external whitening by design" = **MS4/MS5**; only for an *uncurated/generic* sorter should a meaningful `whiten` be passed through unchanged rather than silently rewritten. The validation test must assert MS4/MS5 still intercept AND a generic sorter passes through.

4. **R16 — reconcile dependency contracts.** Pin `numpy>=2,<3` in `pyproject.toml:58` (currently bare). Reconcile the SpikeInterface contract: `pyproject.toml:69` hard-pins `==0.104.3` while `environments/environment_spikesorting_v2.yml:56` allows `>=0.104,<0.105` — make the env file match the hard pin (or document the intentional divergence in one place). No new dependencies.

5. **R34 — permissions, path confinement, trust docs.**
   - Pass `restrict_permission=True` from the v2 writers (`_recording_nwb.py:172-175`, `_units_nwb.py:543,735`) so artifacts are `0o644`, not world-writable `0o666`.
   - In `_sorting_dispatch.py:486`, only `chmod 0o777` the sorter scratch for **container** backends (the UID-mismatch rationale); for local runs drop the chmod (gratuitous).
   - Add basename/path-confinement at the **filename acceptance boundary**, NOT the random-name generator: guard the caller-supplied `nwb_file_name` in `common_nwbfile.py:107` (`raw_dir + "/" + nwb_file_name`, naive concat) and `AnalysisNwbfile.create(nwb_file_name=...)` (`mixins/analysis.py:205`). Do **not** put the check on `__get_new_file_name` (`analysis.py:341-348`) — that generates an internal random name and needs no guard. Reject a `nwb_file_name` with path separators / `..` / absolute path (a basename assertion is simpler and safer than `resolve().relative_to(root)`, which can reject a symlinked base dir). Defense-in-depth — accident prevention under the trusted-operator model; do not escalate to rejecting legitimate basenames.
   - Add a "Security & trust model" subsection to `docs/src/.../SpikeSortingV2.md` stating: `SorterParameters` writers and ingestion are trusted compute operators; the DB is not internet-facing; `execution_params` can pull/run container images by design.

6. **R14 — add the missing DB-safety guard.** Call `_assert_v2_db_safe()` immediately before the `dj.schema(...)` declaration in `metric_curation.py:73` and `recompute.py:67` (mirroring `recording.py:90-91`), so these two modules no longer rely on transitive coverage.

7. **R15 — make the eager v2 merge-table probe visible, do NOT narrow it to `ImportError`.** `spikesorting_merge.py:36-44` catches `Exception` broadly and, on any failure, drops the `CurationV2` part for the process. **The broad `except` is load-bearing and must stay:** `curation.py:60` calls `_assert_v2_db_safe()` at import, which raises **`RuntimeError`** (not `ImportError`) on any non-localhost DB (`utils.py:531,549`) — this is the *expected, caught* path that lets v0/v1 environments on a production (non-localhost) DB load the merge table. Narrowing to `(ImportError, ModuleNotFoundError)` would let that `RuntimeError` propagate and **break `spikesorting_merge` import on every production deployment** (and R14/task 6 widens that surface). The real fix is **visibility, not narrowing**: keep `except Exception`, but `logger.warning(...)` the captured error (so a genuinely unexpected v2 failure is surfaced, not silent) in addition to the existing `_v2_import_error` record. If you do want to distinguish causes, the set to catch-and-tolerate is `(ImportError, ModuleNotFoundError, RuntimeError)` — the `RuntimeError` (DB-safety) path is **not** optional.

8. **R18 — destructive footguns.**
   - `recompute._recent_cutoff` (`recompute.py:1216-1217`): validate `days_since_creation >= 0` at the public entry points (`delete_files`, `recompute.py:624,1039`); raise `ValueError` on a negative value (currently moves the age floor into the future).
   - Orphan analyzer sweep (`sorting.py:2274-2278`): filter `iterdir()` to entries matching the canonical `{sorting_id}__*.zarr` name pattern before considering them for deletion, so a misconfigured root can't delete unrelated subdirectories.
   - DESTR-2 (`_recording_nwb.py:258-275`): the in-place writer unlinks the **canonical** path on failure when `existing_analysis_file_name` is set; the production rebuild now passes `existing=None` (temp-stage), so remove the dead in-place branch (or guard it to refuse unlinking a canonical path).

9. **R22-DOWN-6 — fix the Export.File leak.** `common_usage.py:547-550` calls `delete_quick()` twice on `self.Table` and never deletes `self.File`. Replace the duplicated line with a `self.File` delete so superseded `Export.File` rows are removed.

10. **R26 — conda-export guard.** Wrap `subprocess.check_output(["conda", "env", "export"], ...)` (`mixins/analysis.py:316-318`) in try/except; on failure log a warning and record an "environment capture unavailable" marker instead of aborting the NWB write (so conda-less / uv-only environments can write).

11. **R9 TEAM-1 — surface the cross-team blast radius on sort-group overwrite.** `SortGroupV2._handle_existing` (`recording.py:249-285`) deletes whole-session sort groups, which can cascade through *other teams'* downstream rows. Per the R9 decision (team_name is a tag, no enforcement), the fix is **visibility, not RBAC**: extend the overwrite preview to enumerate all affected downstream rows across teams (which teams' `RecordingSelection`/`Sorting`/curations would be deleted), so the user sees the cross-team impact before confirming. Do not block; just make the blast radius explicit in the preview/error.

12. **Docs.** CHANGELOG entries for each fix; the security-trust subsection (task 5) and the migration note for the dependency pin.

## Additional tasks (Round-3 reviews)

13. **UCI-6 / DOWN-5 — `get_unit_brain_regions` drops chronic-identity disambiguators.** `TrackedUnit.get_unit_brain_regions` fetches `(sorting_id, curation_id, unit_id)` but returns only `(sorting_id, unit_id, region_name)` (`unit_matching.py:984-996`), dropping `curation_id`/`member_index`/`unitmatch_id`/`tracked_unit_id`/`nwb_file_name`/date at exactly the point chronic identity is resolved. Return the disambiguators. (Triaged under R22; R22's only other scheduled task here is DOWN-6. Bundle DOWN-4 `get_pairs` zero-pair schema and DOWN-3=CNEP-2 — note CNEP-2 lands in phase-1 — as the downstream-contract cluster if convenient.)

## Deliberately not in this phase

- **Team access enforcement / RBAC** (R9 decided non-goal) — only the TEAM-1 visibility fix.
- **Image/registry allowlist** (R34 decided non-goal) — only path-confinement + trust docs.
- **PK changes** to session-global keys (R9-PK) — overview Open Question 1.
- **A sorter plugin API** (R33 non-goal) — task 3 only allowlists the existing whiten interception.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sorter_parameters.py::test_clusterless_rejects_container_backend` (new) | a clusterless `SorterParameters` row with `execution_params["backend"]="docker"` raises at insert. |
| `test_sorter_parameters.py::test_legacy_seeder_skips_matlab` (new) | `insert_default_legacy_si_sorters` does not create local MATLAB rows (or creates container ones). |
| `test_sorting_dispatch.py::test_whiten_interception_allowlisted` (new) | a curated MS sorter's `whiten=True` is intercepted (external whitening); a generic sorter's `whiten` is passed through unchanged. |
| `test_dependency_contract.py::test_numpy_pinned_and_si_contracts_agree` (new) | `pyproject` pins `numpy>=2,<3`; the env-file SI spec matches the pyproject pin (parse both). |
| `test_security.py::test_v2_artifacts_not_world_writable` (new) | a populated recording/units artifact has mode `0o644`; local sorter scratch is not `0o777`. |
| `test_security.py::test_nwb_filename_traversal_rejected` (new) | a `nwb_file_name` with `..`/separator/absolute path raises. |
| `test_import_boundaries.py::test_metric_curation_and_recompute_call_db_guard` (new) | importing/declaring `metric_curation` + `recompute` against a non-local DB host raises via `_assert_v2_db_safe` (patch the host). |
| `test_merge_probe.py::test_unexpected_v2_import_error_surfaces` (new) | a non-ImportError raised during the v2 probe is not swallowed (the broad-except narrowing). |
| `test_recompute.py::test_negative_days_rejected` (new) | `delete_files(days_since_creation=-1)` raises `ValueError`. |
| `test_analyzer_lifecycle.py::test_orphan_sweep_ignores_nonmatching_dirs` (new) | a non-`{uuid}__*.zarr` directory under the analyzer root is never a deletion candidate. |
| `test_export.py::test_export_file_rows_removed_on_overwrite` (new) | re-exporting a `paper_id`/`analysis_id` leaves no stale `Export.File` rows. |
| `test_analysis_mixin.py::test_conda_export_failure_does_not_abort_write` (new) | with `conda` absent (patched), the NWB write completes and records the unavailable marker. |
| `test_recording.py::test_sortgroup_overwrite_preview_lists_cross_team_downstream` (new) | the overwrite preview enumerates downstream rows owned by other teams. |
| (regression) the existing `test_sorting_dispatch.py`, `test_recompute.py`, `test_sorter_parameters.py`, export, and import-boundary suites | unchanged behavior on the happy paths. |

## Fixtures

Most tests are unit-level (parse `pyproject`/env file; patch `conda`/host/`temp_dir`;
call validators directly). The permissions, Export.File, and sort-group-overwrite
tests use `populated_sorting` (`conftest.py:215`) / a populated export / a
two-team sort-group setup built from `_ingest_helpers.py`.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Each of the 11 fixes has a test asserting the new behavior; the happy-path regressions pass.
- The security fixes default-close (artifacts `0o644`, traversal rejected) without breaking the container UID path (scratch chmod still applied for container backends).
- The merge-probe narrowing still degrades gracefully in a real v0/v1 env (ImportError path intact).
- TEAM-1 is a visibility fix only (no enforcement added); the R9 / R33 / R9-PK non-goals are honored.
- The Export.File fix deletes `File` (not a third `delete_quick` on `Table`).
- CHANGELOG + the security-trust docs subsection are present; no plan/phase references in code or tests.
