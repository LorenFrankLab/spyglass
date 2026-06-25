# Spike Sorting V2 User/API Ergonomics Review

Date: 2026-06-25

Scope: the user-facing and notebook-facing experience of spikesorting v2:
documentation flow, setup guidance, orchestration helpers, selection helpers,
parameter lookup APIs, curation/matching workflows, error messages, and public
status signals. This is a different lens from scientific reproducibility,
DataJoint/concurrency, test coverage, NWB portability, operational recovery, and
performance/memory scaling.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 has a strong ergonomic core. `initialize_v2_defaults()`,
`run_v2_pipeline()`, `preflight_v2_pipeline()`, `describe_run()`,
`describe_sort_groups()`, `describe_pipeline_presets()`, and the PK-returning
`insert_selection()` helpers give notebook users a much clearer path than raw
DataJoint table manipulation. The dedicated UX smoke test also covers a
"scientist's first hour" path end-to-end, including notebook execution.

The biggest remaining problems are not that the core API is missing; they are
stale or contradictory contracts around it. The storage-management guide tells
users they can safely delete recording artifacts and rely on
`Recording.get_recording()` to rebuild them, while current tests intentionally
pin that missing-cache rebuild raises a checksum error. The main v2 page also
documents UnitMatch and chronic concat workflows as usable, but later status and
migration sections still call unit matching/concat roadmap or placeholder work.

The code-facing API has one sharp inconsistency: `ConcatenatedRecordingSelection`
silently ignores a supplied `concat_recording_id`, unlike the other
deterministic-ID selection helpers that reject mismatched caller-supplied IDs.
Advanced helper diagnostics are also less friendly than the single-session path:
forgotten default rows in analyzer curation or UnitMatch tend to surface as raw
DataJoint FK failures rather than the guided "run initialize_v2_defaults()"
message users get from `RecordingSelection`, `ArtifactDetectionSelection`, and
`SortingSelection`.

## Findings

### 1. High: storage-management docs promise recording-cache recovery that current tests say does not work

`SpikeSortingV2StorageManagement.md` says both recordings and analyzer folders
are regeneratable, and specifically that `Recording.get_recording()` rebuilds a
missing recording from lineage (`docs/src/Features/SpikeSortingV2StorageManagement.md:13-16`).
The workflow then shows `RecordingArtifactRecompute().delete_files(...,
dry_run=False)` followed by "Later: get_recording() rebuilds the deleted
artifact on demand" (`docs/src/Features/SpikeSortingV2StorageManagement.md:56-61`).
The cache-drift policy further says the recording cache uses "warn-and-rebuild,
not fail-closed" (`docs/src/Features/SpikeSortingV2StorageManagement.md:82-89`).

Current tests pin the opposite user outcome. When the cache file is missing,
`Recording().get_recording(...)` rebuilds and then raises a checksum
`DataJointError` (`tests/spikesorting/v2/single_session/test_recording.py:258-318`).
The direct rebuild path also reaches its hash step and raises the checksum error
inside that rebuild (`tests/spikesorting/v2/single_session/test_recording.py:321-349`).

Impact: this is a destructive-operation guide. A lab admin can follow the
documented safe deletion flow, delete a verified recording cache, and later find
the advertised fetch-back path fails. That is a high-severity UX bug even though
the underlying behavior is intentionally fail-closed today.

Fix direction:

- Update the storage guide now: analyzer folders are fetch-back rebuildable, but
  recording cache deletion should be marked unavailable or expert-only until the
  checksum/recovery path is implemented.
- If deletion is still exposed, make the guide show the exact current failure
  mode and recovery options.
- Add a docs/test assertion that storage-management text cannot claim
  `Recording.get_recording()` rebuilds missing artifacts while the checksum
  failure tests remain the expected behavior.

### 2. High: downstream docs steer users toward the root `merge_id`, not the final curated output

The main docs correctly say `run_v2_pipeline` creates an initial/root curation
and later curation should use wrappers such as `AnalyzerCuration.materialize_curation`
or `CurationV2.create_merged_curation`
(`docs/src/Features/SpikeSortingV2.md:445-517`). But the downstream-consumer
section later says "carry `merge_id = run_summary['merge_id']` forward" and uses
that in the accessor table (`docs/src/Features/SpikeSortingV2.md:976-999`).

The notebook has the safer guidance: after the final curation section, it tells
users to key off `final_merge_id`, not `run_summary["merge_id"]`, because the
run-summary merge ID is the uncurated root
(`notebooks/py_scripts/10_Spike_SortingV2.py:490-496`).

Impact: a user who auto-curates or manually merges can accidentally decode,
export, or analyze the uncurated root curation. This is easy to miss because
both merge IDs are valid and downstream APIs will work.

Fix direction:

- In the prose docs, split "root output from `run_v2_pipeline`" from "final
  curated output after curation."
- Show `final_curation = AnalyzerCuration().materialize_curation(sel)` and
  `final_summary = CurationV2.summarize_curation(final_curation)`, then carry
  `final_summary["merge_id"]` downstream.
- Add a docs/notebook smoke assertion that any downstream section after curation
  uses the final curation's merge ID, not the root run summary.

### 3. Medium-high: availability/status docs contradict implemented v2 surfaces

The main v2 page documents chronic same-day concat as available
(`docs/src/Features/SpikeSortingV2.md:784-879`) and cross-session UnitMatch as a
usable optional-extra workflow (`docs/src/Features/SpikeSortingV2.md:880-944`).
The code also has real `SessionGroup`, `ConcatenatedRecording`, `UnitMatch`, and
`TrackedUnit` tables (`src/spyglass/spikesorting/v2/session_group.py:51`,
`src/spyglass/spikesorting/v2/session_group.py:569`,
`src/spyglass/spikesorting/v2/unit_matching.py:438`).

Later in the same docs, the Status section still says cross-session unit
matching is not yet available and remains a placeholder
(`docs/src/Features/SpikeSortingV2.md:1055-1066`). The migration guide likewise
lists `unit_matching` / `matcher_protocol` as import-safe placeholders and marks
concat/session-group and UnitMatch as roadmap items
(`docs/src/Features/SpikeSortingV2_Migration.md:156-169`). The pipeline module
docstring also says richer surfaces such as metrics, concat sorts, and
cross-session matching "come in later versions"
(`src/spyglass/spikesorting/v2/pipeline.py:7-9`).

Impact: users see a complete workflow and then a warning that the same workflow
does not exist. For optional-extra UnitMatch, this is especially confusing:
"not installed" and "not implemented" require different actions.

Fix direction:

- Replace placeholder/roadmap language with a status table: available,
  experimental, optional extra required, or not yet ported.
- Keep `figpack_curation` as the remaining placeholder; tests already document
  that `unit_matching` and `matcher_protocol` are no longer stubs
  (`tests/spikesorting/v2/test_legacy_stub_imports.py:24-39`).
- Add a simple docs lint for stale phrases such as "unit matching ... placeholder"
  when `UnitMatch` remains an importable public table.

### 4. Medium-high: `ConcatenatedRecordingSelection.insert_selection` silently ignores a supplied primary key

The concat selection docstring says a caller-supplied `concat_recording_id` is
ignored and the ID is minted/found by the helper
(`src/spyglass/spikesorting/v2/session_group.py:377-383`). The implementation
then derives `concat_recording_id = deterministic_id(...)` and returns that
value, without checking whether a supplied `concat_recording_id` matched
(`src/spyglass/spikesorting/v2/session_group.py:460-486`).

This differs from the rest of the deterministic-ID UX. For example,
`RecordingSelection` validates a supplied `recording_id` against the derived ID
(`src/spyglass/spikesorting/v2/_selection_plan.py:117-121`), and sorting/artifact
plans document the same explicit-ID mismatch behavior
(`src/spyglass/spikesorting/v2/_selection_plan.py:143-149`,
`src/spyglass/spikesorting/v2/_selection_plan.py:271-277`).

Impact: a notebook that carries a stale fetched dict or typoed
`concat_recording_id` can receive a different deterministic row without a
warning. That is surprising precisely because the rest of v2 has trained users
that supplied IDs are checked.

Fix direction:

- Use `assert_supplied_id_matches(key.get("concat_recording_id"), ...)` in
  `ConcatenatedRecordingSelection.insert_selection`.
- Add tests that a matching UUID/string passes and a mismatched ID raises a
  clear `ValueError`.
- Consider rejecting unrelated extra keys consistently across all selection
  helpers.

### 5. Medium-high: v2 setup prerequisites are too far from the recommended install path

The README quick start recommends `python scripts/install.py`
(`README.md:67-80`). The installer presents minimal/full options with "Basic
spike sorting" under the minimal path (`scripts/install.py:901-912`), but v2's
default path requires the `spikesorting-v2` optional extra
(`pyproject.toml:112-132`). The main v2 page does mention the extra, but only
near the end in the Environment section
(`docs/src/Features/SpikeSortingV2.md:1042-1049`). The notebook assumes a
`spyglass_spikesorting_v2` kernel but does not give the environment creation
command in its opening prerequisites
(`notebooks/py_scripts/10_Spike_SortingV2.py:15-33`).

Impact: a new user can follow the top-level install path, open the v2 notebook,
and only discover missing sorter/runtime dependencies during preflight or
import. Preflight helps after the fact, but the setup path should make the v2
environment explicit before the notebook starts.

Fix direction:

- Move the v2 environment command near the top of the v2 docs and notebook.
- Add an installer profile or README note for "Spike Sorting v2" that installs
  the `spikesorting-v2` extra.
- Add a smoke check that the documented v2 environment can import v2 and pass
  preflight for the default MS5 preset on a fixture.

### 6. Medium: advanced insert helpers have less friendly missing-default diagnostics than the single-session path

The single-session helpers proactively translate missing lookup rows into
guided messages. `RecordingSelection.insert_selection`,
`ArtifactDetectionSelection.insert_selection`, and
`SortingSelection.insert_selection` call `_ensure_lookup_row_exists()` before
inserting (`src/spyglass/spikesorting/v2/recording.py:818-830`,
`src/spyglass/spikesorting/v2/artifact.py:532-545`,
`src/spyglass/spikesorting/v2/sorting.py:786-799`). The shared helper explicitly
exists to avoid opaque DataJoint FK failures and points users at
`initialize_v2_defaults()` (`src/spyglass/spikesorting/v2/_lookup_validation.py:344-382`).

`AnalyzerCurationSelection.insert_selection` and
`UnitMatchSelection.insert_selection` do not perform the same lookup pre-checks
before inserting their referenced metric/rule/matcher rows
(`src/spyglass/spikesorting/v2/metric_curation.py:718-758`,
`src/spyglass/spikesorting/v2/unit_matching.py:313-351`).

Impact: users who get through the main pipeline and then forget defaults for
metrics, auto-rules, analyzer waveform rows, or matcher params can hit raw FK
errors later in a notebook, after expensive upstream work. That makes the
advanced path feel less polished than the main path.

Fix direction:

- Add `_ensure_lookup_row_exists()` calls for `QualityMetricParameters`,
  `AutoCurationRules`, `AnalyzerWaveformParameters`, and `MatcherParameters` at
  the selection-helper boundary.
- Tests should assert the error names the missing table/row and suggests
  `initialize_v2_defaults()` or the table-specific `insert_default()`.

### 7. Medium: the prose quickstart advertises preflight but the main runnable snippet skips it

The "Run your first single-session sort" prose lists Preflight as step 3
(`docs/src/Features/SpikeSortingV2.md:139-146`), and the debugging cookbook
shows how to inspect `report.checks` later
(`docs/src/Features/SpikeSortingV2.md:397-412`). But the main runnable snippet
imports `preflight_v2_pipeline` and then goes directly from
`describe_pipeline_presets()` to `run_v2_pipeline(...)`
(`docs/src/Features/SpikeSortingV2.md:156-210`).

The notebook does show the explicit preflight block before compute, which is the
better teaching path.

Impact: users may miss the best UX surface in v2: the structured report with
`errors`, `warnings`, and predicted IDs. They will still get preflight inside
`run_v2_pipeline(preflight=True)`, but they lose the chance to inspect and fix
before starting the run.

Fix direction:

- Mirror the notebook's explicit preflight block in the main docs before
  `run_v2_pipeline`.
- Add a docs snippet test that the first-run sequence appears in the documented
  order: defaults, sort group, explicit preflight, run, summary, fetch.

### 8. Medium: v2 storage-management docs are not discoverable from the docs nav

`docs/src/Features/SpikeSortingV2StorageManagement.md` exists and contains a v2
storage workflow, but `docs/mkdocs.yml` links only the main v2 and migration
pages (`docs/mkdocs.yml:80-91`). `docs/src/Features/index.md` also links only
the main v2 page (`docs/src/Features/index.md:19-20`).

Impact: lab admins looking for v2 disk-reclamation guidance can easily land on
the older generic recompute docs instead, or never find the v2-specific page.

Fix direction:

- Add the storage-management page to MkDocs navigation, the Features index, the
  main v2 page, and the generic Recompute page.
- Add a link/nav check that every `docs/src/Features/*.md` page is reachable from
  MkDocs nav or an index page.

### 9. Low-medium: `QualityMetricParameters` has an implicit duplicate-content policy

Most v2 parameter lookup tables expose an explicit duplicate-content guard with
an `allow_duplicate_params=True` escape hatch
(`src/spyglass/spikesorting/v2/recording.py:705-720`,
`src/spyglass/spikesorting/v2/sorting.py:235-248`,
`src/spyglass/spikesorting/v2/unit_matching.py:121-140`). The shared guard says
a second name for the same content forks provenance and should raise unless
explicitly allowed (`src/spyglass/spikesorting/v2/_lookup_validation.py:165-190`).

`QualityMetricParameters.insert()` validates metric payloads but does not call
that duplicate-content guard (`src/spyglass/spikesorting/v2/metric_curation.py:281-318`).
The shipped `franklab_default` and `neuropixels_default` rows intentionally
share a payload today (`src/spyglass/spikesorting/v2/metric_curation.py:251-267`),
so the desired policy may be "aliases are allowed here." It is just not explicit.

Impact: user-created duplicate metric recipes can proliferate under different
names, making provenance tables and selection UIs harder to read. If aliases are
intentional for this lookup, users need that documented.

Fix direction:

- Either add the same duplicate guard with a deliberate shipped-alias exception,
  or document `QualityMetricParameters` as alias-friendly and surface duplicates
  clearly in `describe_parameter_rows()`.
- Add tests for shipped aliases and user-created duplicate behavior.

### 10. Low: selection-helper shape and extra-field handling are uneven

`RecordingSelection` rejects extra joined/fetched fields as part of its identity
planning (`src/spyglass/spikesorting/v2/_selection_plan.py:110-121` through
`recording_identity_payload`). `SortingSelection` and
`ArtifactDetectionSelection` validate required/source fields but do not clearly
reject unrelated extras in the same way
(`src/spyglass/spikesorting/v2/_selection_plan.py:130-160`,
`src/spyglass/spikesorting/v2/_selection_plan.py:260-285`). UnitMatch uses a
positional call shape rather than the dict-based `insert_selection(key)` pattern
used by the other selection helpers
(`src/spyglass/spikesorting/v2/unit_matching.py:228-257`).

Impact: users composing helpers from fetched dictionaries get different
behavior across tables: some extras fail fast, some are ignored, and UnitMatch
requires a different mental model.

Fix direction:

- Standardize allowed-key checks across all selection-plan builders.
- Consider a backward-compatible dict/kwargs overload for
  `UnitMatchSelection.insert_selection`.

## Already Solid

- The single-session orchestration layer is genuinely notebook-friendly:
  `run_v2_pipeline()` documents prerequisites, typed failures, idempotency,
  zero-unit behavior, and stage timing
  (`src/spyglass/spikesorting/v2/_pipeline_run.py:83-234`).
- Preflight reports structured checks, warnings, expected IDs, and exact fixes
  (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:168-260`).
- Pipeline/preset reporting helpers are discoverable and tabular:
  `describe_pipeline_presets()`, `describe_pipeline_preset()`,
  `describe_parameter_rows()`, `describe_run()`, and `describe_units()`
  (`src/spyglass/spikesorting/v2/_pipeline_presets.py:76-206`,
  `src/spyglass/spikesorting/v2/_pipeline_reporting.py:42-74`).
- The first-hour UX smoke test exercises defaults, sort group inspection,
  preflight, pipeline run, curation summary, downstream spike-time fetch, and
  notebook execution (`tests/spikesorting/v2/test_ux_smoke.py:1-20`,
  `tests/spikesorting/v2/test_ux_smoke.py:145-239`).
- Selection helpers return PK-only dicts and handle deterministic-ID races for
  the common path (`src/spyglass/spikesorting/v2/recording.py:764-852`,
  `src/spyglass/spikesorting/v2/artifact.py:479-565`,
  `src/spyglass/spikesorting/v2/sorting.py:760-850`).
- Curation wrappers are intent-first and protect users from silent root reuse
  when creating merge/proposal curations
  (`src/spyglass/spikesorting/v2/curation.py:866-1045`).
- UnitMatch selection deliberately pins one curation per member and validates
  ownership instead of using an implicit "latest curation"
  (`src/spyglass/spikesorting/v2/unit_matching.py:197-351`).

## Suggested Fix Order

1. Correct or disable the destructive recording-cache deletion guidance before
   users rely on it.
2. Fix root-vs-final `merge_id` guidance in the main v2 docs and add a docs smoke
   check.
3. Update availability/status language for UnitMatch, concat, and remaining
   placeholders.
4. Make `ConcatenatedRecordingSelection` reject mismatched supplied IDs.
5. Put v2 installation requirements at the top of the v2 docs/notebook and wire
   them into the README or installer.
6. Add friendly missing-default checks to analyzer curation and UnitMatch
   selection helpers.
7. Make storage docs discoverable through MkDocs nav and index pages.
8. Standardize duplicate/extra-field policies for the remaining advanced
   parameter and selection helpers.
