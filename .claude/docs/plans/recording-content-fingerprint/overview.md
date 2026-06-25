# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Design rationale (the "why" behind every decision below):
[../spikesorting-v2/recording-content-fingerprint-design.md](../spikesorting-v2/recording-content-fingerprint-design.md).

## Current codebase integration points

Line numbers verified 2026-06-25 against the post-merge branch (`origin/master`
PR #1600 merged at `c2913b4a`; WIP analyzer/orphan fix at `96e20f98`).

**Phase 0 — recording-source correctness (prerequisite):**

- `src/spyglass/spikesorting/v2/_recording_nwb.py:32` —
  `raw_eseries_path_and_timestamp_mode` picks the **first** acquisition
  `ElectricalSeries` (`:39-40`); must resolve from
  `Raw.raw_object_id` (`src/spyglass/common/common_ephys.py:291`,
  `Raw.nwb_object:377-389`). Threaded into `read_raw_nwb_recording` at
  `recording.py:1697-1704`.
- `src/spyglass/spikesorting/v2/_recording_geometry.py:70` —
  `channel_names[int(c)]` indexes the electrodes table by electrode id as a row
  position; must use the row mapping the write side uses
  (`_nwb_metadata_helpers.py:69` `electrode_table_region` →
  `get_electrode_indices`). `restrict_recording` slice/rename at
  `_recording_restriction.py:558-571`.

**Phase 1 — the fix:**

- `src/spyglass/spikesorting/v2/_recompute.py:53` — `hash_recording_traces`:
  reused as the `traces` component; needs explicit little-endian encoding
  ([:74](../../../../src/spyglass/spikesorting/v2/_recompute.py) uses native
  `tobytes()`). `combined_hash` ([:80](../../../../src/spyglass/spikesorting/v2/_recompute.py)),
  `compare_hash_dicts` ([:89](../../../../src/spyglass/spikesorting/v2/_recompute.py)) reused as-is.
- `src/spyglass/spikesorting/v2/_recording_nwb.py:61` — `write_nwb_artifact`:
  today returns `cache_hash` from `_hash_nwb_recording`
  ([:215](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py)); change to
  return the readback `content_hash` fingerprint. Electrode region at
  [:189](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py), object_id at
  [:208](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py).
- `src/spyglass/spikesorting/v2/recording.py:1020` — `Recording` `cache_hash:
  char(64)` column → `content_hash`. `make_insert` stores it
  ([:1329](../../../../src/spyglass/spikesorting/v2/recording.py),
  [:1467](../../../../src/spyglass/spikesorting/v2/recording.py)). `get_recording`
  ([:1482](../../../../src/spyglass/spikesorting/v2/recording.py)) and
  `_rebuild_nwb_artifact` (~`:1549`) gain reconciliation. `RecordingComputed`
  NamedTuple carries `cache_hash` ([:932](../../../../src/spyglass/spikesorting/v2/recording.py),
  [:960](../../../../src/spyglass/spikesorting/v2/recording.py),
  [:1302](../../../../src/spyglass/spikesorting/v2/recording.py)) → rename.
- `src/spyglass/spikesorting/v2/session_group.py:589` — `ConcatenatedRecording`
  `cache_hash: char(64)` → `content_hash` (rides the shared writer at
  [:876](../../../../src/spyglass/spikesorting/v2/session_group.py); `make_insert`
  [:900](../../../../src/spyglass/spikesorting/v2/session_group.py)). `get_recording`
  ([:963](../../../../src/spyglass/spikesorting/v2/session_group.py)) unchanged —
  concat reclamation is out of scope (design §4).
- `src/spyglass/spikesorting/v2/recompute.py:328` — `RecordingArtifactRecompute`:
  `make_compute` ([:387](../../../../src/spyglass/spikesorting/v2/recompute.py))
  anchors `matched` on `Recording.content_hash`; `delete_files`
  ([:448](../../../../src/spyglass/spikesorting/v2/recompute.py)) wording revert.
  `RecordingArtifactRecomputeSelection` ([:261](../../../../src/spyglass/spikesorting/v2/recompute.py))
  `rounding=4: int` PK ([:267](../../../../src/spyglass/spikesorting/v2/recompute.py))
  removed; `attempt_all` ([:274](../../../../src/spyglass/spikesorting/v2/recompute.py)).
  Diff machinery `_insert_comparison` ([:940](../../../../src/spyglass/spikesorting/v2/recompute.py)).
- `src/spyglass/utils/mixins/analysis.py:820` — `_resolve_external` (merged from
  #1600): the byte-only checksum-refresh primitive reused as-is. **Do not** route
  through `get_hash(resolve=True)` ([:743](../../../../src/spyglass/utils/mixins/analysis.py),
  which resolves even on mismatch, [:816](../../../../src/spyglass/utils/mixins/analysis.py)).
- `src/spyglass/common/common_file_tracking.py:226` — `if analysis_file_name in
  deleted_files: return None` fires *before* the `Path(fname).exists()` check at
  [:234](../../../../src/spyglass/common/common_file_tracking.py); make it
  presence-aware.
- `src/spyglass/spikesorting/v2/exceptions.py:309` — add
  `RecordingContentDriftError` after the last exception.
- `src/spyglass/spikesorting/v2/_analyzer_cache.py:92` — `analyzer_curation_lock`:
  the per-sort `filelock.FileLock` pattern mirrored for the new
  `recording_artifact_lock(recording_id)` (held by `get_recording` read-repair,
  `_rebuild_nwb_artifact`, and `delete_files`).

**Phase 2 — v1-parity hardening (reference, not modified):**

- `src/spyglass/spikesorting/v1/recompute.py:93` — `this_env` /
  [`_has_matching_env`:116](../../../../src/spyglass/spikesorting/v1/recompute.py)
  (env-compat gate); [`attempt_all`:310](../../../../src/spyglass/spikesorting/v1/recompute.py)
  (`limit`, `force_attempt`); [`_check_xfail`:365](../../../../src/spyglass/spikesorting/v1/recompute.py).
  `RecordingArtifactVersions` already inventories `nwb_deps` (v2 recompute.py).

**Left alone:** `SortingAnalyzer` zarr path (no external checksum; rebuild
already round-trips), terminal NWBs (curated/units/metric/unit-match), the
analyzer self-heal + orphan-classification work already committed at `96e20f98`.

## Scope and dependency policy

### Goals

- **(Phase 0)** `Recording.make` reads the *right* signal: raw source pinned to
  `Raw.raw_object_id`, channels mapped by electrode-table row index — so the
  fingerprint identifies the intended recording, not whatever the first
  acquisition series / row-position lookup happened to return.
- Recording delete → on-demand rebuild round-trips: `get_recording` succeeds
  after `delete_files`, with the `~external` checksum reconciled.
- The fingerprint includes the **resolved source object id** (from Phase 0) so
  two recordings built from different raw series never collide.
- A single scientific identity (`content_hash`) drives recompute matching,
  delete authority, and rebuild reconciliation — they can no longer disagree.
- Never silently serve drifted bytes: a rebuild that diverges from the stored
  `content_hash` raises loudly.
- Serialize recording-artifact mutation: rebuild, read-repair, and recompute
  deletion hold a shared per-`recording_id` lock and publish atomically, so
  concurrent workers cannot race the canonical file (DataJoint/concurrency
  review, 2026-06-25).

### Non-Goals

- `ConcatenatedRecording` reclamation (rebuild path, recompute table,
  `delete_files`) — concat gets `content_hash` for free via the shared writer
  but a missing concat file stays a hard error (design §4).
- Byte-for-byte deterministic NWB writing — infeasible across SI/BLAS/platform;
  the fingerprint is representation-blind instead.
- Touching the `SortingAnalyzer` / terminal-NWB paths.
- The broader **input/recipe fingerprint** for *populate-time* drift detection,
  and runtime/seed/software provenance capture (scientific-reproducibility review
  §§3–8). Our output `content_hash` is a *rebuild-time* drift detector — a
  rebuild whose output differs (changed seed, preprocessing order, source,
  intervals) fails closed via `RecordingContentDriftError`. It does **not** catch
  upstream changes at *populate* time (a fresh populate stores a fresh hash with
  nothing to compare against). Closing that gap (validating construction inputs
  before populate) is a separate provenance workstream. Phase 0 + the
  `source_object_id` fingerprint field are the only pieces of it folded here.
- Analyzer-cache locking, UnitMatch identity, curation lineage, and NWB
  self-describing provenance tables (operational, reproducibility, and
  portability reviews) — separate workstreams.

### Dependency policy

No new runtime dependencies. Relies on `origin/master` PR #1600 (merged at
`c2913b4a`): `_resolve_external` and the dataset-content-inclusive
`NwbfileHasher`. h5py ≥ 3 is already required by the env.

## Metrics

- The two pinning tests at
  [test_recording.py:258/:320](../../../../tests/spikesorting/v2/single_session/test_recording.py)
  flip from asserting a checksum `DataJointError` to asserting a successful
  rebuild + read.
- Round-trip integration test: `populate → recompute matched →
  delete_files(dry_run=False) → get_recording` returns traces byte-equal (within
  `TRACE_ROUNDING`) to the pre-delete content.
- Fingerprint determinism: identical recording → identical `content_hash` across
  repeated reads; perturbed traces/timestamps/gain/channel-order/geometry →
  different hash.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Concurrent rebuild vs. recompute deletion race the canonical file (unlink-races-write, reader sees half-written HDF5) | Shared per-`recording_id` `recording_artifact_lock` across rebuild/read-repair/delete + temp-file `os.replace` atomic publish (design §3.5/§3.6). Single-machine / shared-FS scope (`filelock`), as with `analyzer_curation_lock`. |
| Recompute "current file == fresh rebuild" authorizes a delete the rebuild can't honor against the row identity | Anchor `matched` on `combined_hash(fresh) == Recording.content_hash` (design §3.6 Medium 1) — same invariant the rebuild enforces. |
| A failed checksum refresh leaves a byte-different file under a stale checksum, wedging `get_recording` | All-or-nothing cleanup: unlink the slot unless fingerprint-match **and** `_resolve_external` both succeed (design §3.5 High 2). |
| A rebuilt file stays hidden from integrity scans under a stale `deleted=1` | Presence-aware skip in `common_file_tracking` + best-effort `deleted=0` clear (design §3.6 Medium 4). |
| Fingerprint sources geometry from an SI readback surface that drifts across versions | Hash the persisted `ElectricalSeries.electrodes` region directly; `get_channel_locations` is parity-only (design §3.2 Medium 2). |
| `content_hash` rename churns 42 refs across 6 v2 files | Mechanical; covered by compile + the full v2 suite. |
| Depending on `_resolve_external` (a `_`-prefixed method master could rename) | Documented recompute primitive; reliance smoke-test pins the contract (design §3.3). |

## Rollout Strategy

All-at-once, no feature flag, no migration — v2 is pre-release with no users and
schema edits are free (`[[spikesorting-v2-schema-policy]]`). Phase 1 is the
complete correctness fix and is independently shippable; Phase 2 is additive
operational hardening on top. The whole-file `cache_hash` is removed in Phase 1
(no parallel `cache_hash`/`content_hash` period).

## Open Questions

0. **Opt-in `verify_content_hash` on existing-file reads (portability review §9).**
   RESOLVED — won't do. `get_recording` re-validates the DataJoint `~external`
   byte checksum on every read (via the default `get_abs_path`), which already
   catches ordinary on-disk corruption; an opt-in re-fingerprint would only add
   the narrow "valid-but-content-diverged" case (manual file replacement, a
   restore that desynced the checksum table, untrusted cross-host storage), at
   the cost of reading every trace on each read. Marginal value off the normal
   path, not required for reclamation correctness — dropped rather than carried
   as a flag.
1. **At-creation provenance scope (Phase 2).** RESOLVED — dropped in Phase 2
   with evidence: `RecordingArtifactVersions` is `dj.Computed` and reads
   `nwb_deps` from the immutable file, so lazy populate captures identical
   values regardless of timing; pip-env provenance already lives in the
   selection's `UserEnvironment` FK; and deletion authority is the
   `content_hash` recompute match, never provenance. At-creation logging would
   add no fidelity. Original question for reference: log env into
   `RecordingArtifactVersions` at recording creation (v1 parity), or leave
   recompute to populate it lazily? Best answer: lazy is sufficient for
   correctness; at-creation logging is the v1-parity nicety — deferred to
   Phase 2's own judgment, not a Phase 1 blocker.

## Estimated Effort

- Phase 0: ~80–150 LOC src (raw-source resolver via `Raw.raw_object_id`;
  electrode-id→row channel mapping) + ~150 LOC tests (multi-series + shuffled-id
  fixtures). Independently shippable correctness fix.
- Phase 1: ~250–350 LOC src (new `_recording_fingerprint.py` ~120; edits across
  `recording.py`/`_recording_nwb.py`/`recompute.py`/`session_group.py`/
  `common_file_tracking.py`/`exceptions.py`) + ~250 LOC tests. Wide but
  mechanical on the rename; the fingerprint + reconciliation is the real logic.
- Phase 2: ~120–180 LOC src + ~120 LOC tests.
