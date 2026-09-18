# Spike Sorting v2 — Storage Management

Chronic recordings make spike-sorting artifacts large. Spyglass v2 lets you
reclaim that disk space *safely*: an artifact is deleted only after a verified
round-trip showing it can be regenerated from its stored lineage. Two artifact
families have recompute machinery:

- the preprocessed **recording** (`Recording`, an NWB-resident
    `ElectricalSeries` inside an `AnalysisNwbfile`); and
- the per-sort **SortingAnalyzer** folder (`Sorting`, a `binary_folder` of
    waveform/template/extension data).

Both are regeneratable: `Recording.get_recording()` rebuilds a missing recording
from its `RecordingSelection` lineage, and `Sorting.get_analyzer()` rebuilds a
missing analyzer folder from the stored sort. The recompute tables verify that
regeneration *before* anything is deleted.

## The recompute trios

| Recording                             | SortingAnalyzer                     | Role                                                                                                          |
| ------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `RecordingArtifactVersions`           | `SortingAnalyzerVersions`           | Inventory dependencies + a reference content hash.                                                            |
| `RecordingArtifactRecomputeSelection` | `SortingAnalyzerRecomputeSelection` | Plan an attempt under a labeled `UserEnvironment` (a `rounding` precision applies to the analyzer trio only). |
| `RecordingArtifactRecompute`          | `SortingAnalyzerRecompute`          | Regenerate, compare content hashes, record `matched` / `deleted`.                                             |

The comparison uses reproducible **content** — for recordings the content
fingerprint (traces, timestamps, persisted probe geometry, and scaling metadata)
that defines `Recording.content_hash`, and for analyzers the deterministic
extension data, including the seed-pinned `noise_levels` estimate. Neither uses
a whole-file digest, which folds in volatile NWB metadata (`object_id`,
timestamps) and is not reproducible across regenerations. A legacy analyzer
inventory that omitted `noise_levels`, or records it as unseeded, is reported as
`matched=0` with an explicit `legacy/unverifiable` message instead of a
corruption-like hash mismatch. The recording identity has no `rounding` knob —
its precision is fixed by the fingerprint's `TRACE_ROUNDING` /
`TIMESTAMP_ROUNDING` constants; `rounding` applies only to the analyzer
extension comparison.

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
RecordingArtifactRecompute().delete_files(rec_key, dry_run=True)  # preview
RecordingArtifactRecompute().delete_files(rec_key, dry_run=False)  # delete

# Later: get_recording() rebuilds + reconciles the deleted artifact on demand.
```

The `SortingAnalyzer*` trio mirrors this for analyzer folders; deletion removes
the folder, which `Sorting.get_analyzer()` rebuilds on the next access. A
rebuild retires the prior `SortingAnalyzerVersions` generation and its dependent
verdicts, then inventories the published folder again. `attempt_all()` also
refreshes legacy inventories and folders rebuilt by DB-free workers using a
path/size/mtime fingerprint, without reading waveform payloads.

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

A completed comparison with `matched=0` records which objects differ in the
`Name` (missing-from-old/new) and `Hash` (differing) part tables. Attempts that
are explicitly skipped, legacy/unverifiable, or fail regeneration instead put
the reason in `err_msg` and have no synthetic diff rows.

## Analyzer cache layout and memory

Every analyzer cache folder is a SpikeInterface `binary_folder` analyzer named
`{sorting_id}__{payload}.analyzer` under the analyzer root
(`dj.config["custom"]["spikesorting_v2_analyzer_dir"]`, else
`<temp_dir>/spikesorting_v2/analyzers`). The payload is the waveform recipe name
for a sort's own display/metric analyzer, or
`curation_{uuid}_{role}_{recipe_hash}_si_{si_hash}` for a committed merged
curation's per-generation cache; a derivative that carries extra extensions with
specific parameters appends `_ext_{request_hash}` and is reused on every later
request with the same parameters.

`binary_folder` is deliberate: it is the only SpikeInterface 0.104 format whose
waveform extraction writes straight into a memmapped `waveforms.npy` (the `zarr`
and `memory` formats extract into a shared-memory buffer sized for the whole
waveform volume and then copy it). Every load goes through
`load_analyzer_folder`, which maps `waveforms.npy` lazily instead of reading the
whole volume into RAM. Measured on a 415 MB waveform volume
(`test_analyzer_memory.py`): extraction peaks at ~1.5x the volume (file-backed
dirty pages included) versus ~2.3x for `zarr`; a lazy load plus one unit's read
costs ~0.17x versus >= 1x for an eager load. The scientific waveform sample
(`max_spikes_per_unit`, the recipe window, sparsity) is never reduced to meet a
memory target -- only the storage and load paths changed.

The analyzer's recording reference is saved as `recording.pickle`, including
artifact exclusions and recipe preprocessing. This avoids SI 0.104.3's JSON
loss of structured artifact intervals, including after probe projection. It
stores extractor parameters and source paths; it does not copy the recording's
trace data. Reopened derivative analyzers use the same serialization policy.
An older cache whose recording cannot reload is treated as invalid and rebuilt
through the usual cache recovery path.

Folders written under the pre-launch `.zarr` convention are not read; they are
disposable and rebuild on first access. Delete stale `*.zarr` folders under the
analyzer root by hand.

Scratch disk: budget for the preprocessed recording (`Recording`), the sorter's
own scratch under `temp_dir` while it runs, the display and (when PC metrics are
requested) metric analyzers, plus one per-generation cache (and its derivatives)
per committed merged curation you review. The `SortingAnalyzerRecompute` tables
report folder sizes.

Sorting computes its analyzer in a private build directory and derives unit
metadata from that same build. Only the successful database insert can install
it in the canonical slot. A competing attempt that loses its insert discards
its own build and leaves the committed sorting's analyzer unchanged. Analyzer
extraction runs outside the database transaction; publication uses directory
renames. During a rebuild, budget space for both the old and new cache.

## Shared storage and multiple workers

Single-machine and multi-node workers use the same storage ownership protocol.
For multiple nodes, configure all workers to use the same MySQL database,
durable recording/analysis paths, and `spikesorting_v2_analyzer_dir`. The shared
mount must provide cross-host POSIX file locking and directory renames within
the cache filesystem. Review bundles shared between hosts need the same locking
support. Do not place a worker's lock directory on local scratch while its
analyzers live on shared storage.

Validate locking on the actual server/client mounts before deploying multiple
nodes: hold a cache lock on node A and verify that a short acquisition on node B
times out; release or terminate A and verify that B can acquire it. Then run the
two-worker publication, rebuild, and interruption tasks in the
[release-readiness audit](../../plans/spikesorting-v2-release-readiness-audit.md).
A successful local-filesystem test does not establish shared-mount behavior.
Lock errors propagate rather than allowing unprotected cache writes.

## Release workload measurement

`tests/spikesorting/v2/scripts/measure_release_workflow.py` is the repeatable
release run: it executes the supported workflow (prepare → sort → auto-label →
review bundle and reopen → merge → reevaluate → waveform inspection → Phy export
→ unit selection → warm rerun) on one NWB file against a private MySQL container
and base dir, sampling the whole process tree's RSS (workers included) and the
scratch-disk footprint, and writes a JSON receipt with stage timings, peak
memory, peak scratch, bundle bytes and the effective configuration. Run it on a
lab Linux machine with one representative 1–3 h tetrode recording and one ≥1 h
probe recording to set the supported machine budgets; short synthetic fixtures
are not evidence of long-recording capacity.

## Upstream deletion cascades and analyzer folders

`Sorting.delete()` removes the corresponding regeneratable analyzer folder when
the delete starts at `Sorting`. An upstream delete is different: deleting a
`Recording`, `RecordingSelection`, or `SortGroupV2` cascades to `Sorting`
through DataJoint `FreeTable` objects, which remove the database row without
calling the Python `Sorting.delete()` override. The same bypass occurs with raw
SQL or other out-of-band deletion. In those cases the database cascade is
correct, but the 5–50 GB analyzer folder can remain on disk.

After any deletion that did not start at `Sorting`, audit the analyzer cache and
then reclaim only the disk-side orphans you reviewed:

```python
from spyglass.spikesorting.v2.sorting import Sorting

report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
report["disk_side"]  # canonical folders with no surviving reference
report["staging"]  # abandoned build/trash folders, including interrupted jobs

# Interactive confirmation; deletes disk-side/staging folders, never DB rows.
Sorting.find_orphaned_analyzer_folders(dry_run=False)
```

The report also distinguishes DB-side rows whose expected folder is absent and
folders intentionally reclaimed through `SortingAnalyzerRecompute`. Build
ownership is checked with locks, not process IDs or elapsed time; active builds
are skipped and ownership is checked again before cleanup. Killed processes
release their locks, making their hidden build/trash directories reclaimable.
The audit never auto-deletes a database row. Finish upstream cascades and run
canonical-orphan reclamation when sorting/rebuild jobs are idle.

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
    `ConcatenatedRecordingSelection.insert_selection` freezes each member's
    logical identity, resolved `Recording` (`recording_id` + `content_hash`),
    and exact `artifact_detection_id` (or explicit `None`) into the
    `ConcatenatedRecordingSelection.MemberSnapshot` part. Member artifact
    choices participate in `member_set_hash`; changing a mask creates a
    different concat and downstream sort. A different ordered member set is also
    a different concat; editing `SessionGroup.Member` afterward does not change
    or invalidate an existing concat — it just mints a new id on the next
    `insert_selection`.
- **Reads use the frozen snapshot.** Materialization and split read the frozen
    `MemberSnapshot`, never the live group, so a later group edit cannot
    silently re-point an existing concat. If a frozen member's underlying
    `Recording` is gone or its `content_hash` has drifted from the snapshot,
    materialization/rebuild raises `MissingRecordingForConcatError` /
    `ConcatMemberDriftError` rather than building from changed inputs.
- **Artifact dependencies and valid time are preserved.** The selected member
    detections are foreign-key dependencies; ordinary deletion refuses
    referenced detections, while explicit `cascade_delete()` removes dependent
    concat and sorting outputs. Materialization and rebuild apply the same masks
    before motion correction and preserve masked samples afterward.
    `obs_intervals` stores valid time on the concat timeline;
    `MemberBoundary.member_valid_times` stores it in original session timestamps
    for member exports.
- **Checksum-validated reads + content-verified rebuild-on-missing.**
    `ConcatenatedRecording.get_recording()` mirrors `Recording.get_recording()`:
    a present cache file is read through `AnalysisNwbfile`'s `~external`
    byte-checksum validation, and a missing cache file is rebuilt through a
    locked (`concat_recording_artifact_lock`), atomic (`os.replace`),
    content-`hash`-verified path, raising `RecordingContentDriftError` on a
    fingerprint mismatch instead of installing drifted bytes (the canonical slot
    is left untouched). A motion-corrected concat is only byte-reproducible
    insofar as `correct_motion` is deterministic; an irreproducible rebuild
    fails loudly here rather than silently.
- **Split-back conserves spikes.** `split_sorting_by_session()` back-maps a
    concat-frame sorting into per-member local frames; it asserts one strictly-
    increasing boundary per frozen member and that every input spike lands in
    exactly one member, raising `ConcatSplitError` rather than dropping spikes
    that fall outside a member's range.
- **Per-member curated outputs are regenerable.** `ConcatMemberCuration` derives
    each session-aligned Units table entirely from the chosen concat
    `CurationV2` NWB, frozen member boundaries, and member recording timestamps.
    The supported `ConcatMemberCuration.delete()` and `CurationV2.delete()`
    paths list the member rows, merge IDs, and analysis files in a dry run;
    after a confirmed delete they remove the downstream member/merge rows and
    reclaim only `AnalysisNwbfile` entries (and external files) that became true
    orphans. Administrative bypasses such as `super_delete`, raw SQL, or an
    upstream `FreeTable` cascade do not call these Python cleanup hooks; after
    such a bypass, review `AnalysisNwbfile().cleanup(dry_run=True)` before
    applying cleanup.
- **Recompute/reclamation: deferred.** There is no
    `ConcatenatedRecordingArtifact*` recompute trio yet — a concat cache that is
    deleted out of band is rebuilt and verified on demand by `get_recording()`,
    which covers correctness. A dedicated audit + `delete_files` reclamation
    surface (the analogue of the recording trio above) is deferred until concat
    outputs are first retained at scale, and should reuse the shared recompute
    helpers rather than a bespoke table family.

## Admin surface

`attempt_all`, `remove_matched` (on the `*RecomputeSelection` tables) and
`with_names`, `get_parent_key`, `recheck`, `get_disk_space`, `update_secondary`
(on the `*Recompute` tables) port the v1 `RecordingRecompute` operations.

## Test safety

Destructive recompute paths must run under a temporary `SPYGLASS_BASE_DIR`
(never shared lab storage); tests prefer `dry_run=True` unless deletion is the
behavior under test (issue #1573).
