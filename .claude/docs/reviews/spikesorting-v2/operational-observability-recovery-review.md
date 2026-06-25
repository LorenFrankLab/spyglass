# Spike Sorting V2 Operational Observability and Recovery Review

Date: 2026-06-25

Scope: operational behavior of spikesorting v2 under long-running populates,
failed jobs, reclaimed files, partial cleanup, analyzer-cache races, and
operator-facing diagnostics. This is a different lens from the prior scientific
correctness, DataJoint/concurrency, NWB portability, and test coverage reviews:
the question here is whether an operator can tell what happened, recover safely,
and avoid turning a transient failure into hidden drift or unrecoverable state.

Method: local code inspection plus two independent explorer agents. This review
is read-only except for this document.

## Executive Summary

Spike sorting v2 has a strong operational foundation. The pipeline has read-only
preflight, per-stage status and timing, warnings in run summaries, stage-aware
`PipelineStageError` exceptions with partial summaries, a `describe_run` receipt,
zero-unit handling, staged-file cleanup, analyzer orphan detection, and recompute
tables that gate deletion on a current-environment match.

The remaining risks are concentrated around recovery promises and observability
at the boundary between database state and large filesystem artifacts. The
highest-priority issue is recording artifact reclamation: `delete_files()` and
the public storage-management docs describe deleted recording artifacts as
recoverable via `Recording.get_recording()`, but current tests explicitly pin
that the missing-cache path rebuilds and then raises a DataJoint checksum error.
That makes "verified deletion" operationally unsafe until the rebuild path is
made byte-compatible or fail-closed with a typed, actionable recovery error.

The next tier is analyzer-cache mutation. The code has a useful per-sort lock
for curation, but lower-level rebuild/delete paths still write or remove the
canonical zarr folder directly, without taking that same lock and without
atomic staging. That is risky on shared cache roots and multi-job environments.

## Findings

### 1. High: recording artifact reclamation is advertised as recoverable, but the current fetch-back path fails

`RecordingArtifactRecompute.delete_files()` says the regeneratable
`AnalysisNwbfile` is removed and "`Recording.get_recording` rebuilds it on
demand" (`src/spyglass/spikesorting/v2/recompute.py:448-462`). The user-facing
storage guide says the same after a dry-run/real delete pair
(`docs/src/Features/SpikeSortingV2StorageManagement.md:56-60`).

The current behavior is different. `Recording.get_recording()` calls
`_rebuild_nwb_artifact()` when the file is missing
(`src/spyglass/spikesorting/v2/recording.py:1502-1507`). The rebuild writes into
the existing canonical `analysis_file_name`
(`src/spyglass/spikesorting/v2/recording.py:1565-1582`) and the hash helper uses
`AnalysisNwbfile().get_hash()` through the normal checksum-validated path
(`src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:117-136`). Existing tests
explicitly pin that this raises a checksum `DataJointError`, including from
inside the rebuild hash step
(`tests/spikesorting/v2/single_session/test_recording.py:258-318`,
`tests/spikesorting/v2/single_session/test_recording.py:321-348`).

Impact: an operator can run the recommended current-environment recompute flow,
delete a verified recording artifact, and later fail to read the row through the
advertised self-heal path. Worse, the attempted rebuild writes into the canonical
filename before the checksum failure, so recovery has to restore the prior file
or complete a deliberate checksum-reconciliation operation.

Fix direction:

- Stop advertising recording deletion as transparently fetch-back recoverable
  until the path is true.
- Rebuild to a temporary path, hash by direct path or a checksum-bypass path
  intended for verification, compare against the stored row, and only then
  atomically replace or reconcile the external checksum under an explicit
  recovery operation.
- If byte-identical rebuild is not guaranteed yet, raise a v2-specific typed
  recovery error before returning data, with exact next steps: restore backup,
  rerun recompute, or delete/repopulate downstream rows.
- Add an integration test that performs a real
  `RecordingArtifactRecompute().delete_files(..., dry_run=False,
  days_since_creation=0)` and then asserts either successful
  `Recording().get_recording(rec_key)` or the new typed actionable error without
  leaving partial/corrupt bytes.

### 2. Medium-High: analyzer cache rebuild/delete paths bypass the per-sort lock and write canonical folders directly

The cache layer correctly identifies analyzer folders as large regeneratable
scratch and centralizes paths under `analyzer_cache_root()`
(`src/spyglass/spikesorting/v2/_analyzer_cache.py:1-54`). It also provides a
per-sort lock and documents the concurrent zarr mutation hazard
(`src/spyglass/spikesorting/v2/_analyzer_cache.py:92-135`). `AnalyzerCuration`
uses that lock around `Sorting().get_analyzer(...)` and metric computation
(`src/spyglass/spikesorting/v2/metric_curation.py:920-933`).

Lower-level cache mutation does not consistently use the same coordination.
`Sorting.get_analyzer()` can remove an invalid canonical folder and rebuild it
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:210-257`).
`build_analyzer()` writes directly to the canonical zarr folder with
`overwrite=True` (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:587-602`).
`Sorting.delete()` removes cache folders after row deletion
(`src/spyglass/spikesorting/v2/sorting.py:2138-2153`), and recompute deletion
removes analyzer folders with `shutil.rmtree(..., ignore_errors=True)`
(`src/spyglass/spikesorting/v2/recompute.py:1065-1106`).

Impact: a concurrent rebuild, curation, recompute deletion, or row delete can
observe, overwrite, or remove partial canonical analyzer state. On one machine,
the existing lock solves this for curation jobs that enter through
`AnalyzerCuration`, but not for direct `get_analyzer()`, explicit rebuild,
`Sorting.delete()`, or recompute cleanup. The lock doc also notes it does not
coordinate across machines sharing an NFS analyzer directory
(`src/spyglass/spikesorting/v2/_analyzer_cache.py:110-114`).

Fix direction:

- Acquire the same per-sort lock around `load_or_rebuild_analyzer`,
  `rebuild_analyzer_folder`, `Sorting.delete` cache removal, and recompute
  analyzer deletion.
- Prefer building into a temporary sibling folder and renaming into place only
  after `si.load_sorting_analyzer` succeeds.
- Add a barrier/monkeypatch concurrency test for rebuild-vs-rebuild and
  rebuild-vs-delete, asserting no partial canonical folder remains.
- For shared HPC cache roots, consider a DataJoint row/advisory lock or an
  explicit documented "single host mutates analyzer cache" constraint. The
  current `filelock` caveat should be treated as an operational risk, not just a
  comment.

### 3. Medium: artifact interval cleanup can become non-resumable after a partial delete

`ArtifactDetection.delete()` gathers owned `IntervalList` rows before deleting
the master, deletes the master, then removes those interval rows afterward
(`src/spyglass/spikesorting/v2/artifact.py:1104-1153`). This is the right order
for a normal delete because the ownership part rows disappear with the master.

The recovery gap is the crash window after the master cascade and before
`remove_artifact_interval_rows(...)` finishes. Once the master and part rows are
gone, the helper explicitly refuses to infer ownership from naming alone
(`src/spyglass/spikesorting/v2/_artifact_intervals.py:744-759`). The interval
rows are still identifiable by the v2 artifact interval naming/pipeline
convention (`src/spyglass/spikesorting/v2/_artifact_intervals.py:517-525`), but
there is no explicit orphan scanner or cleanup command for this state.

Impact: a killed process or `IntervalList.delete()` failure can leave v2-owned
artifact interval rows behind with no first-class, resumable cleanup path. That
is not likely to corrupt sorting output, but it does leave operational debris
and makes repeated delete/repopulate cycles harder to reason about.

Fix direction:

- Add a dry-run orphan finder for v2 artifact interval rows that are not
  referenced by `ArtifactDetection.ArtifactRemovedInterval`.
- Require a force flag to delete inferred orphans, and restrict it to rows whose
  names/pipeline match the v2 artifact-detection convention.
- Alternatively, persist a tombstone/cleanup row before deleting the master so a
  later cleanup can resume from explicit ownership rather than naming inference.
- Add a test that monkeypatches interval cleanup to fail after `super().delete()`
  and then verifies the orphan finder reports and can safely remove only those
  rows.

### 4. Medium: failed pipeline receipts lose completed-stage timing and stage rows

`run_v2_pipeline()` maintains `stage_seconds` as a local dict and only attaches
it to `run_summary` at the end of a successful run
(`src/spyglass/spikesorting/v2/_pipeline_run.py:295-308`,
`src/spyglass/spikesorting/v2/_pipeline_run.py:428-435`). If a later stage
fails, `PipelineStageError.partial_run_summary` carries stable ids and warnings,
but not the timings for completed stages. In the session wrapper, failed groups
are preserved with their partial summary
(`src/spyglass/spikesorting/v2/_pipeline_run.py:613-642`), but
`describe_run(list_result)` renders failed entries as a single group row plus
warnings, not completed partial stage rows
(`src/spyglass/spikesorting/v2/_pipeline_reporting.py:720-799`).

Impact: for a failed multi-shank session, the receipt cannot answer a common
operator question: which stages completed, and where did the time go? This is
especially painful when a group fails after a long recording or sorting stage.

Fix direction:

- Attach `run_summary["stage_seconds"] = stage_seconds` incrementally before each
  stage call or immediately after each stage completes.
- Have `describe_run(session_result)` render partial stage rows for failed
  groups when a partial summary includes `stage_seconds` or `*_status` fields.
- Extend `test_pipeline_observability.py` and `test_describe_run.py` with a
  sorting-failure case that expects recording/artifact stage rows and timings on
  the failed receipt.

### 5. Medium: preflight expected ids do not distinguish selection rows from computed rows

`PreflightReport.expected_ids` is documented as deterministic selection PKs plus
an `exists` flag (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:197-206`).
The implementation checks only `RecordingSelection`,
`ArtifactDetectionSelection`, and `SortingSelection`
(`src/spyglass/spikesorting/v2/_pipeline_preflight.py:610-624`).

Impact: after a crash that inserted a selection row but failed before the
computed table row landed, preflight can report `exists=True` even though the
stage still needs `populate()`. The actual pipeline remains idempotent and will
populate the missing row, but the operator-facing "what state am I in?" report
is ambiguous.

Fix direction:

- Replace or augment `exists` with `selection_exists`, `computed_exists`, and
  `next_action`.
- Use the expected ids to tell the operator whether the next action is
  `Recording.populate(...)`, `ArtifactDetection.populate(...)`,
  `Sorting.populate(...)`, or "already computed".
- Add a test with a selection-only row and no computed row.

### 6. Medium: recompute deletion no-ops are hard to interpret

Deletion is guarded well: it refuses unmatched rows, defaults away from stale
environment matches, and age-gates recent artifacts
(`src/spyglass/spikesorting/v2/recompute.py:957-1001`,
`src/spyglass/spikesorting/v2/recompute.py:1019-1106`). The operator surface is
thin, though. The age gate silently skips NULL or too-recent `created_at` rows
(`src/spyglass/spikesorting/v2/recompute.py:1041-1044`,
`src/spyglass/spikesorting/v2/recompute.py:1084-1086`), and the public return is
only a list of paths. `unlink(missing_ok=True)` and `rmtree(ignore_errors=True)`
also make "already missing before this cleanup" indistinguishable from "removed
by this call" in the returned path list
(`src/spyglass/spikesorting/v2/recompute.py:1050-1061`,
`src/spyglass/spikesorting/v2/recompute.py:1093-1105`).

Impact: when `delete_files()` returns `[]`, an operator cannot tell whether
nothing matched, everything was too recent, the environment was stale, files were
already missing, or removal failed and was left for retry. The lower-level code
usually preserves safety, but the explanation is not exposed.

Fix direction:

- Return or optionally log structured outcomes: `deleted`, `dry_run_target`,
  `skipped_too_recent`, `skipped_unknown_age`, `already_missing`,
  `failed_remove`, `blocked_stale_env`, and reclaimable bytes.
- Keep the current list-of-paths behavior only as a compatibility wrapper if
  needed.
- Add tests that assert age-gate and already-missing cases report the exact
  reason and relevant `created_at`/cutoff.

### 7. Low-Medium: long-running work and self-healing rebuilds need more breadcrumbs

The pipeline records per-stage durations after the fact and logs batch summaries,
which is useful (`src/spyglass/spikesorting/v2/_pipeline_run.py:287-436`,
`src/spyglass/spikesorting/v2/_pipeline_run.py:669-701`). Single long stages can
still look quiet while they are running. `_run_stage()` times and wraps failures
but does not log stage start/end/failure with the key ids
(`src/spyglass/spikesorting/v2/_pipeline_run.py:41-80`).

Analyzer self-heal is similarly quiet. An invalid analyzer folder logs a warning
before removal/rebuild (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:211-240`),
but a missing folder with default `rebuild=True` rebuilds without a cache-miss
breadcrumb (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:242-257`).
Recording cache rebuild on missing files is also only visible indirectly unless
it mismatches (`src/spyglass/spikesorting/v2/recording.py:1502-1507`,
`src/spyglass/spikesorting/v2/recording.py:1583-1591`).

Impact: operators can mistake a slow sorter, analyzer rebuild, or recording
write for a hung job. Post-run summaries help after success or failure, but they
do not help during the expensive wait.

Fix direction:

- Log stage start/end/failure with stage name, key ids, computed/reused status,
  and elapsed seconds.
- Log cache-miss rebuild start/end for recording and analyzer artifacts,
  including path, reason, and elapsed seconds.
- Add `caplog` tests for mocked successful/failing stages and missing analyzer
  rebuilds.

## Already Solid

- Preflight is read-only, fast, and complete rather than first-failure-only. It
  checks missing session/raw/interval/team/sort-group state, parameter rows,
  sorter installation/runtime, analyzer waveform parameter rows, and warning-only
  configuration smells (`tests/spikesorting/v2/test_preflight.py`).
- Pipeline failures are wrapped as `PipelineStageError` with the original
  exception chained and a partial run summary
  (`src/spyglass/spikesorting/v2/_pipeline_run.py:41-80`).
- Zero-unit sorts are represented explicitly rather than falling through as
  missing output; warnings are included in run summaries and `describe_run`
  receipts (`src/spyglass/spikesorting/v2/_pipeline_run.py:356-389`,
  `tests/spikesorting/v2/test_describe_run.py`).
- Sorting, curation, analyzer curation, and UnitMatch all have meaningful staged
  file cleanup on ordinary exceptions. Several cleanup paths are tested.
- Analyzer cache path policy is centralized and DB-free, and there is a useful
  orphan classifier plus `Sorting.find_orphaned_analyzer_folders()`.
- Recompute deletion has the right safety principle: a current-environment
  `matched=1` row is required by default, and stale matches do not silently
  authorize deletion.

## Suggested Fix Order

1. Make recording artifact deletion/rebuild honest and fail-closed. This is the
   only finding where public docs currently describe a recovery path that tests
   say does not work.
2. Put all analyzer canonical-folder mutation behind one lock and preferably
   atomic staging/rename.
3. Improve failed-run receipts by threading partial `stage_seconds` into
   `PipelineStageError` and rendering partial stage rows in `describe_run`.
4. Add structured recompute deletion outcomes and preflight
   `selection_exists`/`computed_exists`/`next_action`.
5. Add stage and self-heal breadcrumbs once the recovery semantics are sound.

## Good Next Review Dimensions

- Performance and memory scaling: chunk sizes, iterator behavior, analyzer
  extension costs, concat/member scaling, and whether defaults are safe for long
  sessions.
- API ergonomics and migration safety: how natural the v2 public surface is for
  v1 users, and where examples/docs lead users into footguns.
- Security and storage permissions: container bind mounts, `chmod 777` temp
  folders, shared cache roots, and external-store permissions.
- Data lifecycle policy: what should be canonical, regeneratable, backed up,
  reclaimed, or explicitly never deleted.
