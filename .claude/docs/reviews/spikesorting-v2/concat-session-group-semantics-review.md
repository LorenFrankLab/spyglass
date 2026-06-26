# Spike Sorting V2 concat/session-group semantics review

## Scope

This review covers the chronic/session-group concatenation path in Spike Sorting
V2:

- `SessionGroup` and `ConcatenatedRecording`
- concat-backed `Sorting`, `CurationV2`, analyzer reconstruction, and reporting
- split-back semantics from a concat sorting to member sessions
- tests and user-facing documentation for chronic/same-day and multi-day use

The review intentionally focuses on semantic correctness and lifecycle risks, not
general style.

## Method

Two independent passes reviewed the concat/session-group path:

1. A source-focused pass over `session_group.py`, `_concat_recording.py`,
   `sorting.py`, `curation.py`, `_sorting_analyzer.py`, reporting, and recompute
   support.
2. A tests/docs/API pass over concat test coverage, public docs, downstream
   consumer tests, and the documented chronic workflow.

The findings below are the synthesized issues that survived both passes.

## Executive summary

The concat path has a good conceptual split between single-recording sortings and
concat-backed sortings. It also has strong guardrails around duplicate members,
missing upstream recordings, source-kind separation, motion sample-count drift,
and several concat-aware accessors.

The biggest remaining risk is that concat identity is deterministic but not
content-addressed over the ordered member inputs. `concat_recording_id` is minted
from the group name and parameter names, while later compute, fetch, split, and
anchor resolution read the current `SessionGroup.Member` rows. If group membership
changes after a selection or artifact is created, the same concat id can silently
refer to different scientific inputs.

The second large gap is lifecycle parity: `Recording` now has missing-file
rebuild/content reconciliation, but `ConcatenatedRecording` does not. Since concat
recordings are persisted NWB artifacts and can feed sorting, curation, and
reporting, the same storage/recompute concerns apply.

## What looks solid

- `SortingSelection` has explicit source-kind separation, so a row cannot be both
  recording-backed and concat-backed. See
  `src/spyglass/spikesorting/v2/sorting.py:1007` and
  `src/spyglass/spikesorting/v2/_selection_identity.py:266`.
- `SessionGroup.create_group()` rejects empty groups, caller-supplied recording
  dates, duplicate logical members, and multi-day groups unless explicitly
  allowed. See `src/spyglass/spikesorting/v2/session_group.py:85`.
- `ConcatenatedRecordingSelection.insert_selection()` checks that every current
  group member has a populated `Recording` before inserting the concat selection.
  See `src/spyglass/spikesorting/v2/session_group.py:431`.
- Motion correction has useful policy gates: `auto` resolves to `rigid_fast` for
  same-day groups and rejects multi-day groups. See
  `src/spyglass/spikesorting/v2/_concat_recording.py:93`.
- The concat builder rejects unsupported SpikeInterface job keys and tests cover
  that behavior. See `src/spyglass/spikesorting/v2/_concat_recording.py:290` and
  `tests/spikesorting/v2/test_concat_recording.py:190`.
- Several downstream accessors are concat-aware rather than accidentally treating
  concat sortings as ordinary session-local sortings. Examples:
  `CurationV2.get_recording()` at
  `src/spyglass/spikesorting/v2/curation.py:1210`,
  `CurationV2.resolve_restriction()` at
  `src/spyglass/spikesorting/v2/curation.py:1416`, and
  `Sorting.get_unit_brain_regions()` at
  `src/spyglass/spikesorting/v2/sorting.py:2345`.
- UnitMatch currently rejects concat-backed sortings, which is safer than
  silently matching on a synthetic concat timeline. See
  `src/spyglass/spikesorting/v2/unit_matching.py:1014`.

## Findings

### 1. High: concat identity can drift when `SessionGroup.Member` changes

`ConcatenatedRecordingSelection` derives `concat_recording_id` from group owner,
group name, preprocessing parameters, and motion parameters:

- `src/spyglass/spikesorting/v2/session_group.py:350`
- `src/spyglass/spikesorting/v2/session_group.py:460`

It does not include the ordered member identities, resolved `recording_id`s,
member sample counts, or any fingerprint of the upstream recording contents.
Later stages then read the current group membership:

- `make_fetch()` fetches current `SessionGroup.Member` rows and uses the current
  first member as anchor: `src/spyglass/spikesorting/v2/session_group.py:750`.
- `_resolve_member_recording_keys()` resolves current members to current
  `Recording` rows: `src/spyglass/spikesorting/v2/session_group.py:602`.
- `make_compute()` loads current member recordings and stores boundaries:
  `src/spyglass/spikesorting/v2/session_group.py:786`.
- `split_sorting_by_session()` maps stored `MemberBoundary.member_index` rows
  back onto current members: `src/spyglass/spikesorting/v2/session_group.py:994`.
- `Sorting._first_concat_member()` and concat anchor resolution also read the
  current first member: `src/spyglass/spikesorting/v2/sorting.py:1313`.

`MemberBoundary` stores only `member_index` and `end_sample`, not the member's
logical identity or resolved recording id:

- `src/spyglass/spikesorting/v2/session_group.py:580`

This means a group edit can make an existing `concat_recording_id`, existing NWB
artifact, stored boundaries, and split-back keys disagree about which scientific
inputs were concatenated. The most dangerous case is an edit that preserves member
indices but swaps a member identity: the id remains stable, while future fetch,
sort, split, and anchor operations can read a different membership than the one
used to create the stored artifact.

Recommended fix:

- Freeze ordered member inputs into concat identity. A minimal version would add a
  selection part table containing ordered member identity, resolved `recording_id`,
  and the upstream recording content fingerprint.
- Include the member snapshot/fingerprint in `concat_recording_id`.
- Validate at fetch/compute/split time that current group membership matches the
  snapshotted member set, or stop reading live membership once the snapshot
  exists.
- Add tests that mutate group membership after selection insertion and after
  concat population, then assert deterministic failure rather than silent reuse.

### 2. High: concat recordings lack Recording's rebuild/recompute lifecycle

`Recording.get_recording()` has missing-file recovery and artifact/content
reconciliation:

- `src/spyglass/spikesorting/v2/recording.py:1504`
- `src/spyglass/spikesorting/v2/recording.py:1571`

The recompute framework also has version tracking for recording artifacts:

- `src/spyglass/spikesorting/v2/recompute.py:217`

By contrast, `ConcatenatedRecording.get_recording()` directly reads the stored
analysis NWB file:

- `src/spyglass/spikesorting/v2/session_group.py:963`

No concat-specific artifact version table, missing-file rebuild path, or content
fingerprint reconciliation was found. This creates lifecycle asymmetry: a regular
`Recording` can recover when its persisted artifact is missing or stale, while a
concat recording becomes a hard dependency on the existing analysis file.

This matters more because concat recordings feed later expensive artifacts:

- concat-backed `Sorting.make_compute()` reads `ConcatenatedRecording`:
  `src/spyglass/spikesorting/v2/sorting.py:1489`
- analyzer reconstruction reads concat recordings:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:260`
- reporting uses concat duration:
  `src/spyglass/spikesorting/v2/_pipeline_reporting.py:479`

Recommended fix:

- Add concat artifact version/recompute support parallel to `Recording`, keyed by
  the snapshotted concat inputs and concat parameters.
- Implement a locked rebuild path for missing concat NWB artifacts.
- Store and verify a concat content fingerprint for the persisted artifact.
- Add tests for missing-file rebuild, stale-file detection, and concurrent rebuild
  behavior.

### 3. Medium: split-back can silently drop spikes outside stored boundaries

`split_unit_spike_trains()` assigns spikes to member intervals using half-open
frame ranges and returns only frames that fit within a member boundary:

- `src/spyglass/spikesorting/v2/_concat_recording.py:176`

Frames outside all member ranges are not reported as errors. This is reasonable
for a small pure helper only if callers have already proven that the boundaries
cover the sorting timeline exactly. The higher-level caller does not currently
make that proof explicit:

- `split_sorting_by_session()` fetches current members and stored boundaries:
  `src/spyglass/spikesorting/v2/session_group.py:994`

It does not validate that:

- there is exactly one boundary for each snapshotted member
- boundaries are strictly increasing
- the final boundary equals the concat recording's sample count
- every spike in the concat sorting was assigned to exactly one member

When combined with the live-member drift in finding 1, this can convert stale or
corrupt boundaries into silent spike loss during split-back.

Recommended fix:

- Validate boundary count, order, and final coverage before splitting.
- Count input and output spikes per unit and raise if any spike is dropped.
- Tie boundaries to snapshotted member identities, not just `member_index`.
- Add tests for truncated boundaries, extra boundaries, reordered members, and a
  spike at or beyond the final boundary.

### 4. Medium: concat compatibility checks omit sampling and scaling invariants

`assert_concat_compatible()` checks channel ids and channel locations/geometry:

- `src/spyglass/spikesorting/v2/_concat_recording.py:217`

The corresponding tests cover those dimensions:

- `tests/spikesorting/v2/test_concat_recording.py:250`

No explicit checks were found for sampling frequency, dtype, channel gains,
offsets, units, or other scaling-related properties. SpikeInterface may fail or
coerce in some of these cases, but the concat API currently does not produce a
clear Spyglass-level error before concatenation.

The risk is scientific rather than just ergonomic: mixed sample rates or different
scaling metadata can make a concat recording appear valid while downstream timing
or amplitude assumptions are no longer uniform.

Recommended fix:

- Extend `assert_concat_compatible()` to check sampling frequency and relevant
  scaling metadata.
- Decide whether dtype differences are hard errors or explicitly normalized.
- Add tests for sample-rate mismatch and gain/offset mismatch.

### 5. Medium: concat merge IDs can look session-local to downstream consumers

The docs state that V2 merge IDs keep downstream code mostly unchanged:

- `docs/src/Features/SpikeSortingV2.md:978`

For ordinary single-session sortings, that is the intended compatibility story.
For concat-backed sortings, however, the merge id represents a synthetic timeline
that can span multiple recording intervals or sessions. Some metadata accessors
resolve concat sortings by anchoring to the first member:

- `Sorting.resolve_anchor_nwb_file_name()`:
  `src/spyglass/spikesorting/v2/sorting.py:1391`
- `CurationV2.get_sort_metadata()`:
  `src/spyglass/spikesorting/v2/curation.py:1671`
- `CurationV2.get_sort_group_info()`:
  `src/spyglass/spikesorting/v2/curation.py:1939`

The downstream consumer tests reviewed exercise recording-backed V2 sortings, but
not concat-backed merge IDs in session-scoped downstream tables:

- `tests/spikesorting/v2/test_downstream_consumers.py:367`

This creates an API risk: a concat merge id can pass through generic merge-id
interfaces while representing data that is not session-local in the same way as a
single recording. The current brain-region guard is good, but it does not cover
all downstream consumers that may assume one NWB file, one interval list, one
session, or one sort-group metadata row.

Recommended fix:

- Document concat merge IDs as synthetic concat-timeline artifacts, not drop-in
  replacements for every session-scoped downstream consumer.
- Add guards in downstream-facing helpers where session-local assumptions are
  required.
- Add at least one downstream consumer test that attempts to use a concat-backed
  merge id and verifies the intended behavior.

### 6. Medium-low: split-back is in-memory only, leaving no persistence bridge to per-member workflows

The chronic workflow documents `split_sorting_by_session()` as the way to map a
concat sorting back to member sessions:

- `docs/src/Features/SpikeSortingV2.md:845`

The current implementation returns in-memory `NumpySorting` objects:

- `src/spyglass/spikesorting/v2/session_group.py:994`

That is useful for inspection and custom workflows, but it does not create
session-local `Sorting` or `CurationV2` rows. Meanwhile, some downstream
cross-session workflows require ordinary per-session sortings; UnitMatch, for
example, rejects concat-backed sortings:

- `src/spyglass/spikesorting/v2/unit_matching.py:1014`

The docs mention UnitMatch near the chronic workflow caveats, but the workflow
does not yet provide a persistence bridge from split-back output into the
session-local V2 tables.

Recommended fix:

- Either document split-back as inspection/export only, or add a supported API to
  materialize split member sortings/curations with preserved unit lineage.
- If materialization is added, define how concat unit ids map to member-local unit
  ids and how the original concat sorting id is recorded as provenance.
- Add tests for split materialization, or explicitly test that no persistence is
  claimed.

### 7. Medium-low: multi-day concat support is documented ahead of test coverage

The docs describe multi-day concat as experimental and opt-in:

- `docs/src/Features/SpikeSortingV2.md:786`

Tests cover multi-day gating and `auto` rejection:

- `tests/spikesorting/v2/test_session_group_concat.py:653`

No explicit success-path test was found for an allowed multi-day group with an
explicit motion preset or no motion. The default motion parameters also expose
only `"none"`, `"auto_default"`, and `"rigid_fast_default"`:

- `src/spyglass/spikesorting/v2/session_group.py:269`

This is probably acceptable while the docs call multi-day support experimental,
but it should be intentional. Without a success-path test, regressions can turn
"experimental but supported with opt-in" into "documented but accidentally
unusable".

Recommended fix:

- Add one lightweight multi-day success-path test with `allow_multi_day=True` and
  explicit no-motion or explicit rigid-fast settings.
- Keep the documentation clear that `auto` is same-day only.
- If multi-day concat is not ready, narrow the docs to say it is planned rather
  than supported.

### 8. Low: concat NWB observation intervals are not directly readback-tested

Concat-backed `Sorting.make_fetch()` sets `obs_intervals=None`:

- `src/spyglass/spikesorting/v2/sorting.py:1228`

`Sorting.make_compute()` therefore skips artifact masking and writes units over
the full concat recording:

- `src/spyglass/spikesorting/v2/sorting.py:1489`

Reporting uses `ConcatenatedRecording.total_duration_s` as the observed duration:

- `src/spyglass/spikesorting/v2/_pipeline_reporting.py:479`

This matches the current documented caveat that concat artifact removal is not
implemented. The missing piece is test coverage that reads the resulting NWB
units for a concat-backed sorting and asserts the expected full-recording
observation interval. Single-recording tests cover NWB observation intervals more
directly; concat mostly gets indirect duration checks.

Recommended fix:

- Add a concat-backed sorting NWB readback test that asserts the unit
  `obs_intervals` behavior.
- When concat artifact masking is implemented, use that test as the regression
  point for per-member valid-time mapping.

## Follow-on review leads

- Curated Units NWB export/provenance: whether curated outputs preserve
  observation intervals, source identity, and split/concat provenance.
- Analyzer lifecycle/storage contracts: analyzer extension persistence,
  recompute, stale artifact detection, and missing analyzer recovery.
- Downstream/session-scoped consumer safety: audit all merge-id consumers for
  implicit single-session assumptions.
- Delete/recompute behavior across source, concat, sorting, curation, analyzer,
  and external files.
- Documentation/API drift for newly added schemas and chronic workflow entry
  points.

