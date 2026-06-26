# Phase 4c — Concat artifact lifecycle & member-set integrity

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

`ConcatenatedRecording` is missing the integrity guarantees `Recording` already has
(R40, CONCS-2), and its identity floats free of its actual member set (R39 / CONCS-1,
CONCS-3). This phase brings concat to parity: verify-on-read + rebuild + recompute
lifecycle, a frozen member snapshot tied into identity/split, and split-back spike
conservation. Independent of phases 1–3, but **lands after phase-4a**: this phase
assumes phase-4a's concat-compatibility checks (electrode/region/fs — R30/CONCS-4)
are in place and adds the lifecycle/identity layer on top. Both edit
`_concat_recording.py`, so phase-4c rebases onto phase-4a.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/session_group.py:353-358](../../../../src/spyglass/spikesorting/v2/session_group.py#L353-L358) (`_IDENTITY_FIELDS` → `concat_recording_id` at `:466`), [:589](../../../../src/spyglass/spikesorting/v2/session_group.py#L589) (stored `content_hash`), [:592-600](../../../../src/spyglass/spikesorting/v2/session_group.py#L592-L600) (`MemberBoundary`), [:756-758](../../../../src/spyglass/spikesorting/v2/session_group.py#L756-L758) (`make_fetch` reads live members), [:963-992](../../../../src/spyglass/spikesorting/v2/session_group.py#L963-L992) (`get_recording` — no exists/rebuild/verify), [:994-1058](../../../../src/spyglass/spikesorting/v2/session_group.py#L994-L1058) (`split_sorting_by_session`).
- [src/spyglass/spikesorting/v2/_concat_recording.py:204-214](../../../../src/spyglass/spikesorting/v2/_concat_recording.py#L204-L214) (`split_unit_spike_trains` — drops out-of-range frames).
- **The Recording pattern to mirror:** [recording.py:1526-1528](../../../../src/spyglass/spikesorting/v2/recording.py#L1526-L1528) (rebuild-on-missing), [:1571-1700](../../../../src/spyglass/spikesorting/v2/recording.py#L1571-L1700) (`_rebuild_nwb_artifact` — locked, atomic, content-hash-verified), [_recording_fingerprint.py:236-274](../../../../src/spyglass/spikesorting/v2/_recording_fingerprint.py#L236-L274) (`recording_artifact_lock`), [recompute.py:221-299](../../../../src/spyglass/spikesorting/v2/recompute.py#L221-L299) (`RecordingArtifactVersions`/`RecordingArtifactRecompute`).

**Contracts referenced:** [Master-row identity immutability](shared-contracts.md#master-row-identity-immutability) (phase-2 guards the `SessionGroup` *master*; this phase adds the member-set freeze).

## Tasks

1. **Freeze the ordered member set into concat identity (R39 / CONCS-1).** `concat_recording_id` is content-addressed from group + param *names* only (`session_group.py:353-358`), while `make_fetch`/`split`/anchor read **live** `SessionGroup.Member`. Add a `MemberSnapshot` part to `ConcatenatedRecording` (or extend `MemberBoundary`) capturing, per `member_index`, the member's logical identity — `nwb_file_name`, `sort_group_id`, `interval_list_name`, `recording_id`, and the member's `content_hash`/sample count. Fold a hash of that ordered snapshot into the `concat_recording_id` identity payload **or** (cheaper, no id change) store it and **validate current-vs-snapshot at `make_fetch`/`get_recording`/`split`**, raising a `ConcatMemberDriftError` if live members diverge from the snapshot. Prefer folding into identity (pre-production, no data) so a different member set is a different `concat_recording_id`.

2. **Verify `content_hash` on read + rebuild-on-missing (R40 / CONCS-2, the cheap half).** `ConcatenatedRecording.get_recording` (`session_group.py:963-992`) currently reads the stored NWB with no `Path.exists()` / hash check. Mirror `Recording.get_recording` (`recording.py:1526-1528`): on missing file, rebuild via a locked atomic path; on present file, the rebuild path verifies the recomputed fingerprint against the stored `content_hash` and raises `RecordingContentDriftError` on mismatch. Reuse `recording_artifact_lock` keyed on `concat_recording_id` (or a concat-specific lock) and the temp-stage + `os.replace` atomic publish. **This half should land even if task 3 is deferred** — without it a missing/stale concat NWB reads garbage silently.

3. **Concat recompute tables (R40, the heavy half — DECIDE timing).** Add `ConcatenatedRecordingArtifactVersions` + `ConcatenatedRecordingArtifactRecompute` mirroring the recording analogues (`recompute.py:221-299`): env/SI-version inventory, byte-deterministic recompute audit, and `delete_files` reclamation with the same age/env/`force_stale_env` gates and atomic locked deletion. **If real concat data will not be retained before Phase 5, this task may be deferred** (gate: first retention of concat outputs) — but tasks 1, 2, 4 are not deferrable.

4. **Split-back spike conservation (R39 / CONCS-3).** `split_unit_spike_trains` (`_concat_recording.py:204-214`) keeps only `frames[(frames>=start)&(frames<end)]` and silently drops the rest; `split_sorting_by_session` (`session_group.py:994-1058`) never validates the boundaries. Add: assert exactly one boundary per member, strictly-increasing boundaries, final boundary == concat sample count; and assert every input spike is assigned to exactly one member (sum of per-member counts == input count) — raise `ConcatSplitError` on any drop. With task 1's frozen snapshot the boundaries can no longer silently drift.

5. **Docs.** CHANGELOG: concat now verifies `content_hash` on read + rebuilds on missing; member set is frozen into identity; split-back conserves spikes. Add a concat lifecycle/storage subsection to `SpikeSortingV2StorageManagement.md` (coordinate with the phase-5 docs-nav fix).

## Deliberately not in this phase

- **Concat compatibility checks** (electrode/region/fs — CONCS-4) — phase-4a R30; this phase assumes those land and adds the *lifecycle/identity* layer.
- **NWB member-boundary provenance** (writing boundaries into the file) — phase-3b task 6; this phase is the DB-side lifecycle, not the NWB writer.
- **Split-back persistence bridge** (CONCS-6, materializing per-member `Sorting` rows) — enhancement, out of scope; split stays in-memory.
- **Multi-day concat success-path tests** (CONCS-7) — phase-6.
- **Shared-artifact member freezing** (AVTM-1, the sibling of R39) — phase-2 (same pattern, artifact side).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_session_group_concat.py::test_concat_id_changes_with_member_set` (new) | two SessionGroups identical in name+params but differing in member identity produce **different** `concat_recording_id` (or, if snapshot-validate chosen, a post-materialization member edit raises `ConcatMemberDriftError` at `get_recording`/`split`). |
| `test_session_group_concat.py::test_concat_get_recording_rebuilds_on_missing` (new) | deleting the concat NWB then calling `get_recording` rebuilds it (no error); a corrupted/drifted file raises `RecordingContentDriftError`. Mirrors `test_get_recording_rebuilds_on_missing_cache`. |
| `test_session_group_concat.py::test_concat_split_conserves_all_spikes` (new) | **per-spike/frame membership** is preserved — every input (unit, frame) appears in exactly one member's output and the back-mapped frames match (not just summed counts, which a drop+duplicate could satisfy); a boundary set that doesn't cover the full range raises `ConcatSplitError`. |
| `test_recompute.py::test_concat_recompute_round_trip` (new, if task 3 done) | concat `delete_files(dry_run=False)` removes the folder, marks `deleted=1`, and rebuilds — mirrors the recording round-trip. |
| (regression) `test_session_group_concat.py` existing concat populate/split/region tests | unchanged behavior on the happy path. |

Mark concat populate/rebuild tests integration (need `chronic_2_session_minirec`).

## Fixtures

`chronic_2_session_minirec` (`conftest.py:~341`) for concat populate/split/rebuild;
the member-drift test mutates a `SessionGroup.Member` row after materialization;
the split-conservation test can use a synthetic multi-member `NumpySorting` with
known per-member spike counts (DB-light).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Concat `get_recording` verifies `content_hash` on read and rebuilds on missing, using the same lock + atomic-publish shape as the recording side (task 2 landed even if task 3 is deferred).
- Member-set drift is closed: either `concat_recording_id` folds the member snapshot, or current-vs-snapshot is validated at fetch/split with a raised error — not left to live unguarded reads.
- Split-back raises on any dropped spike; boundary coverage is asserted.
- If task 3 is deferred, the deferral is recorded with its trigger (first concat-data retention); task 2's verify-on-read is NOT deferred.
- No duplication with phase-4a's concat-compat checks; no plan/phase references in code or tests.
