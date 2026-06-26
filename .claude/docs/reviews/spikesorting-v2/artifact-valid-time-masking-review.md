# Spike Sorting V2 Artifact Detection and Valid-Time Masking Review

Date: 2026-06-25

Scope: artifact-detection parameter semantics, shared artifact groups, generated
valid-time intervals, sort-time artifact masking, artifact interval ownership,
concat/artifact selection boundaries, and public docs/tests around those paths.

Method: local static source/test/docs inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run the test suite.

## Executive Summary

This area is much stronger than it was earlier in the v2 cycle. Artifact
detection is chunked, gain-aware, gap-aware, and heavily tested against disjoint
recordings. Empty masks fail loudly before the sorter sees an all-zero recording,
and the artifact interval ownership part table is used by delete/read helpers to
avoid guessing ownership from interval-list names.

The remaining risks are mostly lifecycle and consumer-boundary issues. Shared
artifact groups validate their member recordings at creation time, but an
`ArtifactDetectionSelection` identity for a shared group is only the group name
plus params, and later computation/validation reads live group membership. Also,
`Sorting.make_fetch()` consumes artifact valid-times by generated
`IntervalList` name rather than through the stricter ownership helper that the
artifact layer already provides. Those are the two most important fixes.

## What Looks Solid

- `ArtifactDetectionParamsSchema` documents and validates the key scientific
  semantics: amplitude threshold in microvolts, per-frame cross-channel z-score,
  OR behavior when both thresholds are set, `detect=False`, row-level
  `job_kwargs`, positive `removal_window_ms`, and `min_length_s`
  (`src/spyglass/spikesorting/v2/_params/artifact_detection.py:10-88`).
- Artifact parameter rows are Pydantic-validated and duplicate-content guarded
  before insertion (`src/spyglass/spikesorting/v2/artifact.py:157-187`).
- Detection scans chunks by default and returns compact run ranges rather than
  one frame index per flagged sample
  (`src/spyglass/spikesorting/v2/_artifact_intervals.py:42-160`).
- Detection builds valid intervals per recorded chunk, so disjoint recordings do
  not get artifact-removed valid-times spanning wall-clock gaps
  (`src/spyglass/spikesorting/v2/_artifact_intervals.py:450-514`).
- Sort-time masking rejects empty, malformed, unsorted, and overlapping
  `valid_times` before masking (`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:90-123`).
- Tests cover disjoint gap handling, removal-window spillover, join behavior,
  empty masks, chunked-vs-in-memory equivalence, job-kwargs propagation, and
  bounded scan memory (`tests/spikesorting/v2/test_disjoint_artifact.py:170-340`,
  `tests/spikesorting/v2/test_artifact_intervals.py:331-935`).
- Shared artifact groups have strong normal-path insert checks: populated
  members, one session, one sampling frequency, exact timestamp fingerprint,
  matching sample counts, and matching dtypes
  (`src/spyglass/spikesorting/v2/artifact.py:226-437`).
- The artifact read/delete helper treats `ArtifactRemovedInterval` ownership as
  authoritative and raises when ownership rows are missing or contradictory
  (`src/spyglass/spikesorting/v2/_artifact_intervals.py:597-720`,
  `tests/spikesorting/v2/single_session/test_artifact.py:1168-1252`,
  `tests/spikesorting/v2/test_shared_artifact_group.py:491-615`).

## Findings

### 1. Medium-high: shared artifact group membership is not frozen into artifact identity

`artifact_detection_id` for a shared group is derived from
`artifact_detection_params_name` and `shared_artifact_group_name` only
(`src/spyglass/spikesorting/v2/_selection_identity.py:209-263`,
`src/spyglass/spikesorting/v2/_selection_plan.py:260-339`). Population then
fetches the current `SharedArtifactGroup.Member` rows and scans their current
recordings (`src/spyglass/spikesorting/v2/artifact.py:799-840`,
`src/spyglass/spikesorting/v2/artifact.py:927-956`). Later,
`SortingSelection` also checks current group membership when deciding whether an
artifact pass can be linked to a recording
(`src/spyglass/spikesorting/v2/sorting.py:964-979`).

Impact:

- If members are added after an artifact detection has populated, the newly
  added recording can be allowed to use a mask that was never computed over its
  channels.
- If members are removed, an existing artifact detection keeps the same UUID and
  interval-list name even though the logical source set has changed.
- If the row is deleted and repopulated after member edits, the same
  `artifact_detection_id` can compute different valid-times.
- The expensive compatibility checks in `insert_group()` only protect the member
  set that existed at insert time.

Recommended fix:

- Freeze the scanned member set under `ArtifactDetection` itself, for example an
  `ArtifactDetection.MemberSnapshot` part containing sorted `recording_id`s and
  time-axis/content fingerprints, and validate it before masking.
- Or include a deterministic member-set fingerprint in the shared-group
  artifact identity.
- Guard `SharedArtifactGroup.Member` direct edits once a group is referenced, or
  document the group as immutable and require a new group name for any member
  change.
- Add tests for add/remove member after populate, both for sorting validation
  and repopulation identity.

### 2. Medium: `Sorting.make_fetch()` bypasses artifact interval ownership

The artifact layer has a strict read helper that fetches
`ArtifactDetection.ArtifactRemovedInterval` rows first and raises if a populated
artifact row has no owned interval rows
(`src/spyglass/spikesorting/v2/_artifact_intervals.py:645-684`). Delete cleanup
is intentionally built around the same ownership part rather than reconstructing
interval names (`tests/spikesorting/v2/single_session/test_artifact.py:1168-1252`).

`Sorting.make_fetch()` does not use that helper. For a recording source with an
artifact pass, it directly reconstructs the generated interval-list name and
fetches `IntervalList.valid_times` by `(nwb_file_name, interval_list_name)`
(`src/spyglass/spikesorting/v2/sorting.py:1250-1265`). The sorter then masks
from that array (`src/spyglass/spikesorting/v2/sorting.py:1502-1511`).

Impact:

- A partially deleted/corrupted `ArtifactDetection` row with missing ownership
  parts can still be consumed if a same-name `IntervalList` row exists.
- A manually inserted `IntervalList` with the UUID-derived name can influence a
  sort without being owned by the artifact row.
- The code already has the stricter helper, so this is an inconsistent consumer
  boundary rather than a missing abstraction.

Recommended fix:

- Fetch artifact valid-times in `Sorting.make_fetch()` through
  `ArtifactDetection().get_artifact_removed_intervals(..., as_dict=True)` or the
  underlying `read_artifact_removed_intervals()` helper.
- For shared-group artifacts, select the entry matching the sorting recording's
  `nwb_file_name`.
- Add tests where the artifact master exists but the ownership part is missing,
  and where a stray same-name `IntervalList` exists, asserting sorting fails
  loudly before sorter execution.

### 3. Medium: non-empty `valid_times` are under-validated at the mask boundary

`apply_artifact_mask()` validates empty input, shape, `end < start`, and
start-sorted/non-overlapping intervals
(`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:90-123`). It does not
explicitly reject NaN/Inf values or intervals outside the recording envelope
before the complement walker maps times to frames
(`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:174-201`).
`frames_for_times()` documents monotonic finite timestamp assumptions but accepts
query times as raw float64 values
(`src/spyglass/spikesorting/v2/_signal_math.py:598-651`).

The normal `ArtifactDetection._detect_artifacts()` path generates finite
in-bounds intervals, so this is mainly a consumer-boundary risk for hand-built,
legacy, or corrupted `IntervalList` rows. It matters because `Sorting.make_fetch()`
currently reads that `IntervalList` directly.

Impact:

- A NaN or Inf boundary can map to a frame boundary without a clear error,
  producing a no-op, over-mask, or under-mask depending on position.
- Out-of-envelope valid intervals can overstate the observed duration written to
  Units NWBs, because `_write_units_nwb()` trusts non-`None` `obs_intervals`
  (`src/spyglass/spikesorting/v2/_units_nwb.py:592-655`).
- The code already has strict finite-timestamp helpers elsewhere, so allowing
  non-finite interval boundaries is out of character for v2's interval math.

Recommended fix:

- Require all `valid_times` values to be finite before the complement walk.
- Require intervals to lie within `[t_first, t_last]`, with a small
  sample-period tolerance at the edges if needed.
- Decide whether zero-length valid intervals should be rejected or normalized
  away; if retained, test their behavior explicitly.
- Add tests for NaN, Inf, outside-before-start, outside-after-end, and
  zero-length rows.

### 4. Medium-low: concat artifact behavior is clear in code but weak at the public boundary

The selection plan rejects `concat_recording_id` combined with
`artifact_detection_id` (`src/spyglass/spikesorting/v2/_selection_plan.py:181-194`),
and there is a pure plan test for that rejection
(`tests/spikesorting/v2/test_selection_plan.py:247-258`). At runtime, concat
sorts have `obs_intervals = None` and the mask block is skipped
(`src/spyglass/spikesorting/v2/sorting.py:1268-1272`,
`src/spyglass/spikesorting/v2/sorting.py:1489-1511`).

The code comment says concat sorts "reuse per-member Recording artifacts"
(`src/spyglass/spikesorting/v2/_selection_plan.py:138-141`). That is easy to
misread as "member artifact masks are applied to the concat recording." They are
not applied today.

Impact:

- A future refactor could preserve the pure plan test while weakening the actual
  `SortingSelection.insert_selection()` behavior.
- Users may believe concat sorts inherit member artifact masks, while the
  runtime explicitly treats concat as no artifact pass.
- A bypassed inconsistent row would still sort unmasked unless compute rechecks
  the invariant.

Recommended fix:

- Add a DB/API test that calls `SortingSelection.insert_selection()` with
  `concat_recording_id + artifact_detection_id` and asserts the clear
  `ValueError`.
- Revalidate in `Sorting.make_fetch()` or `make_compute()` that concat sources
  have no artifact source part.
- Reword the comment/docs to say concat artifact masking is currently
  unsupported; member masks are not applied to concat sorts.

### 5. Medium-low: artifact/shared-group semantics are underdocumented for users

The source schema carries the real artifact parameter contract, including
cross-channel z-score semantics, OR behavior, `detect=False`, `job_kwargs`, and
`min_length_s` (`src/spyglass/spikesorting/v2/_params/artifact_detection.py:10-88`).
The docs summarize `ArtifactDetection` as "amplitude-threshold artifact
intervals" and show only the `"default"` row in the stage-by-stage example
(`docs/src/Features/SpikeSortingV2.md:71-73`,
`docs/src/Features/SpikeSortingV2.md:594-598`).

The API docstring for `ArtifactDetectionParameters` also says defaults ship only
`"none"` and `"default"` (`src/spyglass/spikesorting/v2/artifact.py:125-140`),
but the catalog includes production 100 uV and 50 uV variants
(`src/spyglass/spikesorting/v2/_recipe_catalog.py:216-245`).

Shared artifact groups have even more important construction constraints:
populated recordings, one session, same sampling frequency, exact timestamps,
sample count, and dtype (`src/spyglass/spikesorting/v2/artifact.py:226-437`).
The tests cover these well, including multi-member union behavior
(`tests/spikesorting/v2/test_shared_artifact_group.py:124-236`,
`tests/spikesorting/v2/single_session/test_artifact.py:1493-1685`), but the
public docs only briefly mention the feature.

Impact:

- Users have to infer scientific behavior and compatibility constraints from
  source/tests.
- The `"none"` preset, zero-artifact warning, empty-valid-times failure mode,
  and `job_kwargs` tuning are easy to miss.
- Shared-group users may discover expensive timestamp compatibility checks only
  after attempting insertion.

Recommended fix:

- Add a docs section for artifact parameters with the shipped presets, units,
  z-score limitations, OR semantics, `detect=False`, `min_length_s`, empty-mask
  behavior, and `job_kwargs`.
- Add a shared-group example showing `SharedArtifactGroup.insert_group`,
  `ArtifactDetectionSelection.insert_selection`, per-member sorting selections,
  and `get_artifact_removed_intervals(..., as_dict=True)`.
- Update the `ArtifactDetectionParameters` docstring to point at the catalog or
  `describe_pipeline_presets()` rather than listing only two rows.

### 6. Low: the disjoint seam heuristic has one acknowledged single-frame edge

`apply_artifact_mask()` drops width-one complement ranges that look like pure
inter-chunk gaps so a disjoint recording does not zero the final good sample of
the previous chunk (`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:203-229`).
The comment explicitly notes the remaining edge: a genuine one-sample artifact
exactly on a chunk's final sample can be treated as a gap boundary
(`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:209-211`).

The normal detector path uses positive `removal_window_ms`, so a detected
artifact is widened and this single-frame edge is mostly unreachable from
validated v2 artifact parameters (`src/spyglass/spikesorting/v2/_params/artifact_detection.py:72-75`).

Impact: hand-built or corrupted artifact valid-times can leave one final-frame
boundary artifact unmasked even though `obs_intervals` may exclude it.

Recommended fix:

- Keep the current heuristic if the risk is acceptable, but pin the edge with a
  test so the tradeoff stays explicit.
- Longer term, carry frame-native artifact ranges or base-interval metadata so
  the mask consumer can distinguish pure inter-chunk gaps from real
  one-frame artifact tails.

## Follow-On Leads For Other Review Dimensions

- **Concat/session-group semantics:** concat sorts currently have no artifact
  mask and no concat-wide observation intervals. This overlaps the concat review
  space more than artifact detection itself.
- **NWB export/provenance:** sort-time Units NWBs write `obs_intervals`, but the
  curated Units writer currently rebuilds units with spike times/ids and does
  not pass `obs_intervals` (`src/spyglass/spikesorting/v2/_units_nwb.py:649-655`,
  `src/spyglass/spikesorting/v2/_units_nwb.py:895-903`). That deserves a
  focused NWB/export provenance pass.
- **Docs/API ergonomics:** artifact and shared-group docs should be folded into
  Phase 5 docs work, alongside clearer concat artifact messaging.
