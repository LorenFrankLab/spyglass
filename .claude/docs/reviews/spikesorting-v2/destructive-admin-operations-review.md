# Spike Sorting V2 Destructive and Admin Operations Review

Date: 2026-06-25

Scope: delete overrides, recompute `delete_files()` reclamation, staged-file
cleanup, analyzer-cache removal, artifact-interval cleanup, orphan sweeps,
rerun/overwrite controls, and operator-facing docs/tests for destructive v2
actions. This is a different lens from scientific reproducibility,
DataJoint/concurrency, dependency/runtime safety, schema evolution, and general
operational observability.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 has a lot of good destructive-operation scaffolding already: sort-group
overwrites are opt-in, recompute deletion is dry-run by default, current-env
matches are required before storage reclamation, recent/unknown-age artifacts are
skipped, cancelled deletes preserve side-effect files, and many populate
rollback paths explicitly unlink staged NWBs or partial analyzer folders.

The remaining risks are mostly second-order admin paths. Cleanup side effects
are tied to child table `delete()` overrides, so parent/selection cascades and
`super_delete()` can bypass them. Analyzer deletion and rebuild operations are
not consistently serialized through the analyzer lock. A stale lower-level
recording writer path can unlink an existing canonical file on failure. Several
admin APIs also return only bare path lists or warnings, which is safe enough for
small tests but thin for real storage-reclamation operations.

## What Looks Solid

- `SortGroupV2` refuses silent reruns and requires an explicit destructive
  overwrite path (`src/spyglass/spikesorting/v2/recording.py:138-146`,
  `src/spyglass/spikesorting/v2/recording.py:253-285`).
- `Sorting.delete()` snapshots rows before deletion and removes analyzer cache
  folders only if the DB row actually disappeared, so a cancelled safemode prompt
  preserves the row's folder (`src/spyglass/spikesorting/v2/sorting.py:2084-2154`).
- `ArtifactDetection.delete()` similarly avoids removing `IntervalList` rows when
  the safemode prompt is cancelled (`src/spyglass/spikesorting/v2/artifact.py:1104-1153`).
- Recording recompute deletion is gated by `matched=1`, the current
  `UserEnvironment`, a default seven-day age floor, and a per-recording lock
  around unlink/update (`src/spyglass/spikesorting/v2/recompute.py:481-506`,
  `src/spyglass/spikesorting/v2/recompute.py:1019-1136`).
- The tests cover stale-env refusal, unknown/recent-age skips, full recording
  delete/rebuild round trip, cancelled sorting delete, cancelled artifact delete,
  and analyzer-orphan classification (`tests/spikesorting/v2/test_recompute.py:127-177`,
  `tests/spikesorting/v2/test_recompute.py:540-585`,
  `tests/spikesorting/v2/test_recompute.py:589-639`,
  `tests/spikesorting/v2/test_delete_cleanup.py:121-180`,
  `tests/spikesorting/v2/test_analyzer_lifecycle.py:410-575`).

## Findings

### 1. High: cascaded deletes can bypass v2 side-effect cleanup

`Sorting.delete()` removes analyzer cache folders only when the `Sorting` table's
delete override itself runs (`src/spyglass/spikesorting/v2/sorting.py:2084-2154`).
`ArtifactDetection.delete()` removes artifact `IntervalList` rows only when the
`ArtifactDetection` delete override itself runs
(`src/spyglass/spikesorting/v2/artifact.py:1104-1153`).

Parent or selection deletes can remove those rows through DataJoint cascade
without calling the child override. The test cleanup helper demonstrates this
kind of path explicitly: it deletes `Sorting` and `ArtifactDetection` rows via
`super_delete()` while cleaning sessions
(`tests/spikesorting/v2/_ingest_helpers.py:200-218`). The comments in
`Sorting.find_orphaned_analyzer_folders()` acknowledge that bypassed delete paths
can leak analyzer folders (`src/spyglass/spikesorting/v2/sorting.py:2156-2178`),
but the side effect remains easy to bypass from upstream cleanup scripts.

Impact: deleting `SortingSelection`, `Recording`, session-level fixtures, or
using `super_delete()` in admin scripts can leave 5-50 GB analyzer folders and
artifact `IntervalList` rows behind after the DB rows are gone. The analyzer
leak has a finder; artifact intervals do not yet have an equivalent resumable
orphan cleaner.

Recommended fix: move side-effect cleanup to cleanup-aware parent/selection
delete paths, or introduce a shared "snapshot side effects before cascade,
cleanup after rows actually disappeared" helper that upstream destructive paths
call. Add integration tests that delete through `SortingSelection.delete()` and
`ArtifactDetectionSelection.delete()` or an upstream `Recording` cascade and
assert side effects are either removed or reported by a first-class orphan audit.

### 2. High: direct recording writes into an existing analysis file can unlink the canonical file on failure

`write_nwb_artifact()` still accepts `existing_analysis_file_name`, passes it to
`AnalysisNwbfile().create(..., recompute_file_name=existing_analysis_file_name)`,
and then unlinks the resolved path on any write/hash exception
(`src/spyglass/spikesorting/v2/_recording_nwb.py:167-180`,
`src/spyglass/spikesorting/v2/_recording_nwb.py:258-275`). The higher-level
`Recording._compute_recording_artifact()` exception handler explicitly says the
existing-file rebuild path should not unlink the cache file
(`src/spyglass/spikesorting/v2/recording.py:1918-1930`), but the lower-level
writer can already have done it.

Production `_rebuild_nwb_artifact()` now uses a fresh temp artifact and atomic
`os.replace` only after content-hash match
(`src/spyglass/spikesorting/v2/recording.py:1628-1674`), so the old in-place
writer branch appears to be stale and more dangerous than the main rebuild path.

Impact: an admin/direct caller that passes `existing_analysis_file_name` can turn
a failed in-place write into a missing canonical artifact. That is especially
surprising because the surrounding comments promise the existing cache is not
unlinked on rebuild failure.

Recommended fix: remove or reject the existing-file overwrite branch in
`write_nwb_artifact()`, or reimplement it with the same temp-file publish
semantics as `_rebuild_nwb_artifact()`. Add a targeted test that injects a
failure after `create(recompute_file_name=...)` and verifies the original file is
neither unlinked nor truncated.

### 3. Medium-high: analyzer cache deletion/rebuild paths are not consistently locked or atomic

The analyzer cache module documents a per-sort lock for mutating shared analyzer
Zarr folders, and `AnalyzerCuration` uses that lock around extension computation.
But destructive/admin paths mutate canonical folders directly:

- `get_analyzer()` removes an invalid folder before rebuilding
  (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:210-240`).
- `rebuild_analyzer_folder()` builds into the canonical folder and removes it on
  failure (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:370-399`).
- `_build_analyzer()` cleans partial canonical folders after compute failures
  (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:656-676`).
- `Sorting.delete()` removes cache folders after DB deletion
  (`src/spyglass/spikesorting/v2/sorting.py:2140-2154`).
- `SortingAnalyzerRecompute.delete_files()` removes analyzer folders with
  `shutil.rmtree(..., ignore_errors=True)`
  (`src/spyglass/spikesorting/v2/recompute.py:1139-1180`).

Impact: a storage-reclamation job, row delete, curation job, and rebuild can
interleave on the same canonical folder. The result can be a failed job, partial
folder, or valid folder removed immediately after another process rebuilds it.

Recommended fix: make all analyzer folder mutations acquire the same per-sort
lock, including invalid-folder removal, rebuild, `Sorting.delete`,
`SortingAnalyzerRecompute.delete_files()`, and orphan-folder deletion. For
rebuilds, prefer build-to-temp-sibling plus atomic publish/rename where the Zarr
format allows it. Add barrier or monkeypatch tests that prove destructive paths
acquire the lock and that concurrent rebuild-vs-delete cannot publish a partial
canonical folder.

### 4. Medium-high: artifact interval cleanup is not resumable after a partial delete

`ArtifactDetection.delete()` snapshots owned interval rows, deletes the master,
then removes `IntervalList` rows afterward
(`src/spyglass/spikesorting/v2/artifact.py:1130-1153`). The ownership collection
helper refuses to infer ownership from naming when part rows are missing
(`src/spyglass/spikesorting/v2/_artifact_intervals.py:746-759`), which is the
right behavior before deleting a live master. After a crash between the master
delete and interval cleanup, however, the master and ownership parts are gone.

Impact: a killed process or `IntervalList.delete()` failure can leave v2-owned
artifact interval rows behind with no first-class dry-run finder or resumable
cleanup command. Those rows can confuse interval listings and make repeated
delete/repopulate cycles harder to reason about.

Recommended fix: add an explicit artifact-interval orphan audit that finds v2
artifact interval names with no surviving `ArtifactDetection.ArtifactRemovedInterval`
owner, reports them by default, and deletes only with a force/confirm flag. Add a
test that monkeypatches interval cleanup to fail after the master row is gone,
then verifies the orphan finder reports and removes only v2-owned rows.

### 5. Medium: `force_stale_env=True` is not durably audited

The storage-management docs say stale-env override is "audit-logged"
(`docs/src/Features/SpikeSortingV2StorageManagement.md:71-80`), and the design
notes call for a written justification. The implementation accepts only
`force_stale_env: bool` and emits a warning log line
(`src/spyglass/spikesorting/v2/recompute.py:481-506`,
`src/spyglass/spikesorting/v2/recompute.py:1019-1063`). The current test verifies
that a forced stale-env dry run lists the artifact, but does not check durable
audit state (`tests/spikesorting/v2/test_recompute.py:127-177`).

Impact: an operator can force deletion based on an older environment with no
database-persisted reason, actor, timestamp, stale env id, or current env id.
That weakens the audit trail for the most dangerous recompute override.

Recommended fix: require a `justification=` argument when `force_stale_env=True`
and persist it in an audit part/table or structured row-level blob. Include the
artifact key, stale authorizing env ids, current env id, user, time, dry-run vs
actual delete, and paths. Add tests for missing justification rejection, dry-run
audit, and actual forced deletion under temporary storage.

### 6. Medium: negative `days_since_creation` can bypass the recent-artifact safety gate

`delete_files()` exposes `days_since_creation` with a default of seven days
(`src/spyglass/spikesorting/v2/recompute.py:481-506`,
`src/spyglass/spikesorting/v2/recompute.py:883-904`). The cutoff helper passes
the value straight into `datetime.timedelta`
(`src/spyglass/spikesorting/v2/recompute.py:1066-1078`). A negative value moves
the cutoff into the future, so a known-age authorizing row will generally not be
"too recent."

Impact: an operator typo such as `days_since_creation=-7` can turn the age floor
into an immediate-deletion allowance for freshly created artifacts.

Recommended fix: validate `days_since_creation >= 0` in both public
`delete_files()` paths or in `_recent_cutoff()`. Add recording and analyzer tests
that negative values raise before any dry-run or deletion logic.

### 7. Medium: recompute deletion and disk-space reporting are too opaque for admin use

The destructive helpers return a bare list of paths. Age-gated rows are skipped
with `continue` (`src/spyglass/spikesorting/v2/recompute.py:1111-1117`,
`src/spyglass/spikesorting/v2/recompute.py:1157-1160`), missing recording files
are treated as successfully absent via `unlink(missing_ok=True)` and then marked
deleted (`src/spyglass/spikesorting/v2/recompute.py:1122-1135`), and
`get_disk_space()` reports matched, not-yet-deleted artifacts without applying
the current-env or age gate (`src/spyglass/spikesorting/v2/recompute.py:458-460`,
`src/spyglass/spikesorting/v2/recompute.py:1183-1197`).

Impact: `[]` can mean "nothing matched", "stale env blocked", "too recent",
"unknown age", or "no path existed", depending on where the caller is looking.
`Total: X` can be read as immediately reclaimable space even when the current
deletion call would skip some of it.

Recommended fix: return or optionally log a structured deletion report with
`dry_run_target`, `deleted`, `skipped_stale_env`, `skipped_too_recent`,
`skipped_unknown_age`, `already_missing`, and `failed_remove`. Rename or augment
`get_disk_space()` to distinguish matched space from currently deletable space.
Add tests for each reason code.

### 8. Medium: orphan analyzer cleanup can delete arbitrary directories under the configured analyzer root

`find_orphaned_analyzer_folders()` treats every directory under
`analyzer_cache_root()` as a disk candidate
(`src/spyglass/spikesorting/v2/sorting.py:2273-2284`) and, after confirmation,
deletes every disk-side orphan with `shutil.rmtree`
(`src/spyglass/spikesorting/v2/sorting.py:2317-2341`). The function does not
require the analyzer cache naming shape `{sorting_id}__{waveform_params_name}.zarr`,
even though analyzer paths are produced that way.

Impact: if `spikesorting_v2_analyzer_dir` is accidentally pointed at a broader
shared directory, `dry_run=False` can remove unrelated subdirectories after a
single broad confirmation prompt.

Recommended fix: only auto-delete directories matching the analyzer-cache
pattern with a valid UUID prefix and `.zarr` suffix. Report nonconforming
directories separately as "unknown under analyzer root" and require a different
manual cleanup path. Add tests that a non-matching directory is reported but not
deleted even when the user confirms.

### 9. Medium-low: destructive analyzer orphan sweep lacks a direct yes/no test and public runbook

The analyzer orphan finder has a destructive branch: `dry_run=False` prompts the
user and deletes disk-side orphan folders only
(`src/spyglass/spikesorting/v2/sorting.py:2189-2193`,
`src/spyglass/spikesorting/v2/sorting.py:2317-2341`). Current tests exercise the
dry-run classification paths, including DB-side, disk-side, stale recipe,
referenced metric folders, and zero-unit carveout
(`tests/spikesorting/v2/test_analyzer_lifecycle.py:410-575`,
`tests/spikesorting/v2/test_analyzer_lifecycle.py:689-718`). The migration doc
mentions only the dry-run audit (`docs/src/Features/SpikeSortingV2_Migration.md:114-116`).

Impact: the actual prompt/confirm branch can regress unnoticed. Operators also
do not have a clear public runbook for when to use the destructive sweep versus
recompute deletion.

Recommended fix: add yes/no prompt tests proving `dry_run=False` deletes only
disk-side orphan folders when the user confirms and deletes nothing when the user
cancels. Document the workflow in `SpikeSortingV2StorageManagement.md`, including
the difference between DB-side missing folders, expected reclaimed folders, and
disk-side orphan folders.

### 10. Medium-low: sort-group destructive preview docs do not match the API behavior

The main v2 docs say `delete_existing_entries=True, confirm=False` returns a
`DeletionPreview` (`docs/src/Features/SpikeSortingV2.md:49-53`). The
implementation raises `ValueError` containing the preview and tells the caller
to rerun with `confirm=True`
(`src/spyglass/spikesorting/v2/recording.py:277-283`). The test uses the
separate `SortGroupV2.preview_existing_entries()` method as the returned preview
path (`tests/spikesorting/v2/single_session/test_sort_group.py:56-86`).

Impact: an operator following the docs gets an exception instead of a returned
object from the constructor path. The safe workflow exists, but the runbook names
the wrong call shape.

Recommended fix: either make `confirm=False` return the preview as documented, or
update docs to say: call `SortGroupV2.preview_existing_entries(nwb_file_name)`
first, then rerun the grouping helper with `delete_existing_entries=True,
confirm=True`.

### 11. Low-medium: `remove_matched()` uses `delete_quick()` on potentially populated recompute selections

Both recompute selection cleanup methods compute redundant selections and then
call `delete_quick()` (`src/spyglass/spikesorting/v2/recompute.py:311-327`,
`src/spyglass/spikesorting/v2/recompute.py:745-760`). A redundant stale or
failed selection can still have a `matched=0` recompute child row.

Impact: cleanup can fail on foreign-key constraints, or at least bypass the
usual cautious-delete/audit style used elsewhere in destructive operations. The
comment explains avoiding the matched child row, but it does not prove no other
child recompute rows exist.

Recommended fix: restrict quick deletion to selections with no recompute child
row, or intentionally cautious-delete/cascade child recompute rows first. Add a
test with one current `matched=1` row and one populated stale or `matched=0`
sibling selection.

