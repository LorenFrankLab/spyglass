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

## Admin surface

`attempt_all`, `remove_matched` (on the `*RecomputeSelection` tables) and
`with_names`, `get_parent_key`, `recheck`, `get_disk_space`,
`update_secondary` (on the `*Recompute` tables) port the v1
`RecordingRecompute` operations.

## Test safety

Destructive recompute paths must run under a temporary `SPYGLASS_BASE_DIR`
(never shared lab storage); tests prefer `dry_run=True` unless deletion is the
behavior under test (issue #1573).
