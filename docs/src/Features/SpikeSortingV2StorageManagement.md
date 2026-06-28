# Spike Sorting v2 — Storage Management

Chronic recordings make spike-sorting artifacts large. Spyglass v2 lets you
reclaim that disk space *safely*: an artifact is deleted only after a verified
round-trip showing it can be regenerated from its stored lineage. Two artifact
families have recompute machinery:

- the preprocessed **recording** (`Recording`, an NWB-resident
  `ElectricalSeries` inside an `AnalysisNwbfile`); and
- the per-sort **SortingAnalyzer** folder (`Sorting`, a `binary_folder` of
  waveform/template/extension data).

Both are regeneratable: `Recording.get_recording()` rebuilds a missing
recording from its `RecordingSelection` lineage, and `Sorting.get_analyzer()`
rebuilds a missing analyzer folder from the stored sort. The recompute tables
verify that regeneration *before* anything is deleted.

## The recompute trios

| Recording | SortingAnalyzer | Role |
| --- | --- | --- |
| `RecordingArtifactVersions` | `SortingAnalyzerVersions` | Inventory dependencies + a reference content hash. |
| `RecordingArtifactRecomputeSelection` | `SortingAnalyzerRecomputeSelection` | Plan an attempt under a labeled `UserEnvironment` (a `rounding` precision applies to the analyzer trio only). |
| `RecordingArtifactRecompute` | `SortingAnalyzerRecompute` | Regenerate, compare content hashes, record `matched` / `deleted`. |

The comparison uses reproducible **content** — for recordings the content
fingerprint (traces, timestamps, persisted probe geometry, and scaling metadata)
that defines `Recording.content_hash`, and for analyzers the deterministic
extension data (excluding the stochastic `noise_levels` estimate). Neither uses
a whole-file digest, which folds in volatile NWB metadata (`object_id`,
timestamps) and is not reproducible across regenerations. The recording
identity has no `rounding` knob — its precision is fixed by the fingerprint's
`TRACE_ROUNDING` / `TIMESTAMP_ROUNDING` constants; `rounding` applies only to
the analyzer extension comparison.

## Workflow

```python
from spyglass.spikesorting.v2.recompute import (
    RecordingArtifactVersions,
    RecordingArtifactRecomputeSelection,
    RecordingArtifactRecompute,
)

rec_key = {"recording_id": ...}

# 1. Inventory.
RecordingArtifactVersions.populate(rec_key)

# 2. Plan an attempt under the current environment.
RecordingArtifactRecomputeSelection.attempt_all(rec_key)

# 3. Regenerate + compare.
RecordingArtifactRecompute.populate(rec_key)

# 4. See what can be reclaimed.
RecordingArtifactRecompute().get_disk_space(rec_key)

# 5. Reclaim (matched=1 + current environment + > days_since_creation old).
RecordingArtifactRecompute().delete_files(rec_key, dry_run=True)    # preview
RecordingArtifactRecompute().delete_files(rec_key, dry_run=False)   # delete

# Later: get_recording() rebuilds + reconciles the deleted artifact on demand.
```

The `SortingAnalyzer*` trio mirrors this for analyzer folders; deletion removes
the folder, which `Sorting.get_analyzer()` rebuilds on the next access.

## The deletion gate (do not weaken)

`delete_files()` refuses to delete unless there is a `matched=1` recompute row
**whose `env_id` is the current `UserEnvironment`**:

- A `matched=0` row never authorizes deletion.
- A `matched=1` row from a *different* environment (e.g. a verification that
  succeeded months ago under an older SpikeInterface pin) is not evidence the
  current environment can regenerate the artifact. The default raises
  `StaleEnvMatchedError` naming the stale env(s). Pass `force_stale_env=True`
  (audit-logged) to override deliberately.
- Recently-created artifacts are skipped (`days_since_creation`, default 7).

A `matched=0` recompute also records which objects differ in the `Name`
(missing-from-old/new) and `Hash` (differing) part tables for review.

## Cache-drift policy

The recording cache is **content-addressed and fail-closed on drift**.
`Recording.get_recording()` rebuilds only when the cached file is missing; the
rebuild regenerates to a private temp file, fingerprints it, and installs it
(atomic `os.replace`) only if the fingerprint matches the row's stored
`content_hash` — then reconciles the DataJoint `~external` byte checksum so the
next checksum-validated read succeeds. If the rebuild **diverges** from
`content_hash` (e.g. a SpikeInterface/BLAS upgrade, an edited raw NWB, or
changed upstream construction inputs), it raises `RecordingContentDriftError`
(naming the file and the recovery options) and never serves the drifted bytes —
the canonical slot is left untouched. Rebuild, read-repair, and recompute
deletion are serialized per recording (`recording_artifact_lock`) and publish
atomically, so a reclamation can never race a rebuild. There is no hash-mutating
`repair()`.

## Concatenated recordings

The cross-session `ConcatenatedRecording` cache (a motion-corrected, unwhitened
`ElectricalSeries` stitched from the ordered member recordings) is a third
regeneratable family, with the same fail-closed lifecycle as the single-session
recording — plus a frozen member set tied into its identity.

- **Identity = the ordered member set.** `concat_recording_id` is content-
  addressed from the group + parameter names **and** a SHA-256 of the ordered
  *logical* member set (`member_set_hash`). When you create the selection,
  `ConcatenatedRecordingSelection.insert_selection` freezes each member's logical
  identity and its resolved `Recording` (`recording_id` + `content_hash`) into the
  `ConcatenatedRecordingSelection.MemberSnapshot` part. A different ordered member
  set is therefore a *different* concat; editing `SessionGroup.Member` afterward
  does not change or invalidate an existing concat — it just mints a new id on the
  next `insert_selection`.
- **Reads use the frozen snapshot.** Materialization and split read the frozen
  `MemberSnapshot`, never the live group, so a later group edit cannot silently
  re-point an existing concat. If a frozen member's underlying `Recording` is gone
  or its `content_hash` has drifted from the snapshot, materialization/rebuild
  raises `MissingRecordingForConcatError` / `ConcatMemberDriftError` rather than
  building from changed inputs.
- **Verify-on-read + rebuild-on-missing.** `ConcatenatedRecording.get_recording()`
  mirrors `Recording.get_recording()`: a missing cache file is rebuilt through a
  locked (`concat_recording_artifact_lock`), atomic (`os.replace`),
  content-`hash`-verified path, raising `RecordingContentDriftError` on a
  fingerprint mismatch instead of serving drifted bytes (the canonical slot is
  left untouched). A motion-corrected concat is only byte-reproducible insofar as
  `correct_motion` is deterministic; an irreproducible rebuild fails loudly here
  rather than silently.
- **Split-back conserves spikes.** `split_sorting_by_session()` back-maps a
  concat-frame sorting into per-member local frames; it asserts one strictly-
  increasing boundary per frozen member and that every input spike lands in
  exactly one member, raising `ConcatSplitError` rather than dropping spikes that
  fall outside a member's range.
- **Recompute/reclamation: deferred.** There is no `ConcatenatedRecordingArtifact*`
  recompute trio yet — a concat cache that is deleted out of band is rebuilt and
  verified on demand by `get_recording()`, which covers correctness. A dedicated
  audit + `delete_files` reclamation surface (the analogue of the recording trio
  above) is deferred until concat outputs are first retained at scale, and should
  reuse the shared recompute helpers rather than a bespoke table family.

## Admin surface

`attempt_all`, `remove_matched` (on the `*RecomputeSelection` tables) and
`with_names`, `get_parent_key`, `recheck`, `get_disk_space`,
`update_secondary` (on the `*Recompute` tables) port the v1
`RecordingRecompute` operations.

## Test safety

Destructive recompute paths must run under a temporary `SPYGLASS_BASE_DIR`
(never shared lab storage); tests prefer `dry_run=True` unless deletion is the
behavior under test (issue #1573).
