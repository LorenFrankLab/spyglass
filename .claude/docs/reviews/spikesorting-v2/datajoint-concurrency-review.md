# Spike Sorting V2 DataJoint and Concurrency Review

Date: 2026-06-25

## Scope

This review examines Spike Sorting V2 along the DataJoint correctness and
concurrency dimension. The focus is on whether database rows, filesystem cache
artifacts, and populate/recompute workflows remain consistent under concurrent
workers, duplicate submissions, or manual recompute operations.

This was a read-only review. No tests were run and no code was modified as part
of the review.

Independent review agents were used to cross-check two perspectives:

- DataJoint correctness and transaction boundaries
- Filesystem cache concurrency and race behavior

## Findings

### High: Concurrent Sorting.populate can delete a valid analyzer cache

`run_v2_pipeline` calls `Sorting.populate(..., reserve_jobs=False)` in
`src/spyglass/spikesorting/v2/_pipeline_run.py`. `Sorting.make_compute` derives a
deterministic analyzer folder from `sorting_id`, and analyzer creation writes to
that folder with `overwrite=True`. If two workers process the same sorting key,
both can build the same canonical analyzer folder.

The problematic sequence is:

1. Worker A and worker B both start the same `Sorting` job.
2. Both build or overwrite the same analyzer folder.
3. Worker A wins the database insert.
4. Worker B hits duplicate insert handling.
5. Worker B cleanup removes the shared analyzer folder, which is now the valid
   analyzer cache for Worker A's committed row.

Relevant code:

- `src/spyglass/spikesorting/v2/_pipeline_run.py`: `Sorting.populate(..., reserve_jobs=False)`
- `src/spyglass/spikesorting/v2/sorting.py`: deterministic analyzer path in `make_compute`
- `src/spyglass/spikesorting/v2/_sorting_analyzer.py`: `si.create_sorting_analyzer(..., overwrite=True)`
- `src/spyglass/spikesorting/v2/sorting.py`: duplicate handling and `_cleanup_staged_sorting_artifacts`

Recommended fix:

- Build analyzers in per-worker temporary folders.
- Publish to the canonical analyzer folder only after the database insert winner
  is known, or hold a per-`sorting_id` analyzer-build lock through build and
  insert.
- Duplicate losers should clean only their private staged artifacts.

### High: Analyzer cache mutations are not consistently serialized

Analyzer cache mutation is protected in some paths but not consistently across
all paths that can load, rebuild, delete, or extend the same analyzer folder.

Examples:

- `_sorting_analyzer.load_or_rebuild_analyzer` can rebuild a folder.
- `_sorting_analyzer.rebuild_analyzer_folder` can delete/recreate a folder.
- `Sorting.add_extensions` can mutate analyzer extensions.
- `recompute._delete_analyzer_folders` can remove analyzer folders.
- `AnalyzerCuration` uses `analyzer_curation_lock`, which shows the codebase
  already has a locking model for one analyzer mutation path.

Without a general analyzer-cache mutation lock, one worker can read or write a
Zarr analyzer while another worker removes, rebuilds, or extends it.

Recommended fix:

- Promote the existing analyzer curation lock concept into a general
  analyzer-cache lock.
- Acquire that lock for analyzer rebuild, extension mutation, recompute analyzer
  deletion, invalid-folder cleanup, and any path that publishes or removes a
  canonical analyzer folder.

### High: Recording cache rebuild and recompute deletion are not serialized

`Recording.get_recording` can rebuild a missing recording NWB artifact into the
canonical `analysis_file_name`. Recompute deletion can unlink that same artifact.
Those operations are not guarded by a shared per-recording lock.

This creates several race windows:

- A rebuild can write into a file while recompute deletion is checking or
  unlinking it.
- Recompute can mark a file deleted while a concurrent rebuild recreates it.
- A reader can see a partially rewritten or missing HDF5 file.
- A rebuilt artifact can remain installed even if its content hash no longer
  matches the expected row.

Relevant code:

- `src/spyglass/spikesorting/v2/recording.py`: `get_recording`
- `src/spyglass/spikesorting/v2/recording.py`: `_rebuild_nwb_artifact`
- `src/spyglass/spikesorting/v2/_recording_nwb.py`: `write_recording_nwb`
- `src/spyglass/spikesorting/v2/recompute.py`: recording file deletion and
  `deleted=1` updates

Recommended fix:

- Add a per-`recording_id` file lock shared by recording rebuild, recording read
  repair, and recompute deletion.
- Rebuild into a temporary file and atomically replace the canonical artifact
  only after successful write and content verification.
- Treat rebuilt content-hash mismatch as a hard error instead of returning the
  mismatched artifact.
- Ensure recompute deletion performs a presence-aware update while holding the
  same artifact lock.

This overlaps with the recording-content fingerprint plan. That plan should
explicitly include the lock and atomic-publish requirement, not only semantic
fingerprint validation.

### Medium: CurationV2 child curation id allocation has a check-then-insert race

`CurationV2.insert_curation` allocates `curation_id` by fetching existing ids and
choosing `max(existing) + 1`. It then stages curated-unit NWB work before the
final insert transaction.

Two clients inserting child curations for the same `sorting_id` can choose the
same `curation_id`. One insert succeeds, and the other fails late after doing
expensive file staging. Cleanup is best-effort, but the operation is not
race-idempotent.

Relevant code:

- `src/spyglass/spikesorting/v2/curation.py`: `insert_curation`
- `src/spyglass/spikesorting/v2/curation.py`: `_build_curation_insert_plan`
- `src/spyglass/spikesorting/v2/curation.py`: `_stage_curation_artifact`
- `src/spyglass/spikesorting/v2/curation.py`: `_insert_curation_rows_transaction`

Recommended fix:

- Allocate child `curation_id` under a per-`sorting_id` lock or allocator row.
- Alternatively, catch duplicate-key failures, clean private staged artifacts,
  re-fetch the conflicting curation, and retry or return the existing curation
  when the payload is identical.

### Medium: TrackedUnit.make does graph derivation inside the populate transaction

`TrackedUnit.make` performs multiple database fetches and derives tracked-unit
graphs inside a monolithic DataJoint `make`. Although the final insert uses
`transaction_or_noop`, DataJoint's populate machinery still wraps `make` in an
outer transaction.

For large unit-matching graphs, this can hold transaction locks much longer than
needed and increases the chance of populate stalls or deadlocks.

Relevant code:

- `src/spyglass/spikesorting/v2/unit_matching.py`: `TrackedUnit.make`
- `src/spyglass/spikesorting/v2/unit_matching.py`: final insert block

Recommended fix:

- Refactor `TrackedUnit` to the common V2 tri-part pattern:
  - `make_fetch`: read all database state needed for computation
  - `make_compute`: derive tracked units outside the transaction
  - `make_insert`: insert master and part rows atomically

### Low: AutoCurationRules.insert_rules is not race-idempotent

`AutoCurationRules.insert_rules` checks whether a ruleset already exists before
starting its insert transaction. Concurrent callers can both observe a missing
ruleset, then one succeeds and the other raises a duplicate-key error even when
the payload is identical and the caller requested `skip_duplicates=True`.

Relevant code:

- `src/spyglass/spikesorting/v2/metric_curation.py`: `AutoCurationRules.insert_rules`
- `src/spyglass/spikesorting/v2/metric_curation.py`: `insert_default`

Recommended fix:

- Catch duplicate-key failures around the insert transaction.
- Re-fetch the stored ruleset.
- Return successfully if the stored payload matches the requested payload.
- Raise a conflict only when the duplicate name maps to different rule content.

## Positive Observations

Several V2 paths already have good correctness structure:

- Many selection tables use content-addressed identifiers and duplicate-key
  recovery.
- Several computed tables separate database fetch, external compute, and final
  insert.
- The code already has helper concepts such as `transaction_or_noop` and
  analyzer locks, which can be reused instead of inventing a new concurrency
  model.

The main gap is not the database schema alone. It is the boundary between
database rows and shared filesystem artifacts. Canonical NWB files and canonical
analyzer folders need the same ownership, staging, and locking guarantees as the
DataJoint rows that point to them.

## Suggested Priority

1. Fix analyzer cache staging and locking.
2. Add recording artifact locks and atomic rebuild/publish semantics.
3. Make `CurationV2` child id allocation race-safe.
4. Refactor `TrackedUnit.make` to shorten transaction scope.
5. Make `AutoCurationRules.insert_rules` duplicate-safe.

