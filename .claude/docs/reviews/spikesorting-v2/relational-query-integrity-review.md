# Spike Sorting V2 Relational and Query Integrity Review

Date: 2026-06-25

Scope: DataJoint relational invariants in spikesorting v2: deterministic
selection identity, source-part exclusivity, parent/part consistency, merge-id
query routing, concat/session-group provenance, UnitMatch/TrackedUnit graph
integrity, export/query selectivity, and direct-write hardening. This review
focuses on whether stored rows remain true as relational facts over time.

Method: local static source/test/docs inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

The normal v2 insert paths are much stronger than the raw schema alone. The
selection-plan builders derive deterministic IDs from explicit logical payloads,
source parts make recording-vs-concat provenance queryable, `resolve_source()`
guards catch missing or duplicated source parts before compute, and
`UnitMatchSelection` revalidates per-member curation ownership before running a
matcher.

The remaining integrity risks are mostly "relational drift" and bypass/update
holes. Several tables make a deterministic ID mean "the row currently named by
these mutable secondary columns" rather than "the immutable input identity this
ID was minted from." `SessionGroup.Member` is especially important: concat,
sorting anchors, and UnitMatch provenance all point back to the current member
rows, while materialized artifacts store only group names or `member_index`
boundaries. That can retarget existing outputs if group membership changes.

## What Looks Solid

- `_selection_plan.py` validates source shape before table insertion, including
  exactly one `recording_id` vs `concat_recording_id` for sorting selections.
- `_selection_identity.py` normalizes payloads and includes source kind in
  sorting identity, avoiding recording/concat aliasing.
- `ArtifactDetectionSelection.resolve_source()` and
  `SortingSelection.resolve_source()` explicitly reject zero or multiple source
  parts at materialization time
  (`src/spyglass/spikesorting/v2/artifact.py:669-708`,
  `src/spyglass/spikesorting/v2/sorting.py:1007-1044`).
- `SortingSelection` models artifact detection as a zero-or-one part table
  instead of a nullable master FK, and `resolve_artifact_detection()` centralizes
  the optional-part read (`src/spyglass/spikesorting/v2/sorting.py:1046-1073`).
- `UnitMatchSelection` hashes pinned member curations and revalidates both the
  hash and group membership in `_find_existing_pk()` / `make_fetch()`
  (`src/spyglass/spikesorting/v2/unit_matching.py:353-434`,
  `src/spyglass/spikesorting/v2/unit_matching.py:503-574`).
- `CurationV2.resolve_restriction()` is the single owner of v2 merge-query
  routing. It handles broad no-source queries by using the `SortingSelection`
  master so concat-backed curations are not dropped
  (`src/spyglass/spikesorting/v2/curation.py:1416-1668`).

## Findings

### 1. High: deterministic selection IDs can be retargeted through `update1`

`SelectionMasterInsertGuard` blocks direct `insert` / `insert1` unless the
validated helper passes `allow_direct_insert=True`, but it does not block
`update1` (`src/spyglass/spikesorting/v2/utils.py:171-254`). The same file
already has `ImmutableParamsLookup.update1()` for parameter rows because editing
content under an existing key redefines downstream deterministic IDs
(`src/spyglass/spikesorting/v2/utils.py:257-309`). The same hazard applies to
selection masters: secondary columns are the human-readable identity that broad
queries use, while the UUID primary key is what downstream computed rows hold.

Examples:

- `ConcatenatedRecordingSelection` stores `session_group_owner`,
  `session_group_name`, `preprocessing_params_name`, and
  `motion_correction_params_name` under a deterministic `concat_recording_id`
  (`src/spyglass/spikesorting/v2/session_group.py:342-358`).
- `SortingSelection` stores sorter parameters under `sorting_id`, while source
  identity lives in part tables (`src/spyglass/spikesorting/v2/sorting.py:650-710`).
- `UnitMatchSelection` stores group, matcher, and `curation_set_hash` under
  `unitmatch_id` (`src/spyglass/spikesorting/v2/unit_matching.py:197-225`).

Impact: a direct update can make an existing UUID appear to represent a
different logical selection without recomputing any dependent artifact. Queries
through the mutated selection metadata can then return stale `Recording`,
`Sorting`, `CurationV2`, `UnitMatch`, or `TrackedUnit` rows under the wrong
identity.

Fix direction:

- Add a default-deny `update1` guard to `SelectionMasterInsertGuard`, mirroring
  `ImmutableParamsLookup`.
- If a maintenance escape hatch is needed, require an explicit keyword such as
  `allow_selection_mutation=True` and document that it is only safe for rows
  with no live downstream references.
- Add guard tests for each deterministic selection master.

### 2. High: `SessionGroup.Member` is live provenance for materialized concat and UnitMatch outputs

`SessionGroup.Member` stores the ordered member identity for a group, but the
master key is only `(session_group_owner, session_group_name)` and the part key
adds only `member_index`; the actual member identity fields are secondary
columns (`src/spyglass/spikesorting/v2/session_group.py:65-83`).

Downstream rows do not freeze that member set:

- `ConcatenatedRecordingSelection` identity is only group name plus
  preprocessing/motion names, not a member-set hash
  (`src/spyglass/spikesorting/v2/session_group.py:350-358`).
- `ConcatenatedRecording.make_fetch()` reads the current `SessionGroup.Member`
  rows, and `make_insert()` stores only `MemberBoundary(member_index,
  end_sample)` (`src/spyglass/spikesorting/v2/session_group.py:750-784`,
  `src/spyglass/spikesorting/v2/session_group.py:926-951`).
- `split_sorting_by_session()` later fetches the current members again and
  aligns stored boundaries by `member_index`
  (`src/spyglass/spikesorting/v2/session_group.py:1030-1058`).
- Sorting concat anchors are resolved from the current first member
  (`src/spyglass/spikesorting/v2/sorting.py:1312-1418`).
- `UnitMatchSelection` and `UnitMatch` pin curation choices by `member_index`,
  but complete multi-session provenance is documented as queryable through the
  current `SessionGroup.Member` rows
  (`src/spyglass/spikesorting/v2/unit_matching.py:197-225`,
  `src/spyglass/spikesorting/v2/unit_matching.py:437-447`).

Impact: after a concat or UnitMatch artifact is populated, editing/replacing a
member row can make the stored artifact's provenance point at a different NWB
file, sort group, interval, or team. In the concat path, split sorting can label
spike trains under the wrong session identity or fail late if indices no longer
line up. In the sorting path, the anchor NWB/electrode resolution can move to a
different first member than the one used to write the artifact.

Fix direction:

- Prefer making `SessionGroup` immutable after creation: block direct
  `Member.insert/update/delete` except through an explicit replacement workflow
  that also invalidates or remints dependent selections.
- If groups must be mutable, persist a `member_set_hash` or frozen member
  snapshot under `ConcatenatedRecordingSelection` and `UnitMatchSelection`.
- Have concat split/anchor and UnitMatch accessors assert that the current group
  hash matches the stored hash, or read from the frozen snapshot instead of the
  live group table.
- Add tests that materialize concat and UnitMatch rows, mutate/forge member rows,
  and assert either stable frozen provenance or loud mismatch errors.

### 3. High: `CurationV2` direct writes can bypass parts and merge registration

`CurationV2` is a plain manual table (`src/spyglass/spikesorting/v2/curation.py:64-88`).
The safe path is `insert_curation()`, which validates parentage, writes the
curated-units NWB, inserts `Unit`, `UnitLabel`, and `MergeGroup` parts, and
registers the row into `SpikeSortingOutput.CurationV2` inside the same DB
transaction (`src/spyglass/spikesorting/v2/curation.py:192-230`,
`src/spyglass/spikesorting/v2/curation.py:815-842`).

There is no corresponding guard on direct `CurationV2.insert1()` or `update1()`.
A bypass can create a master row with no units, no merge provenance, no merge
registration, or mutate `parent_curation_id`, `object_id`, or `merges_applied`
after dependent rows exist.

Impact: downstream consumers route through the merge table and part rows, while
the curation master can be made to disagree with both. This is a relational
integrity hole even if the user-facing API path is correct.

Fix direction:

- Add direct-write guards to `CurationV2` master insertion and `update1`, with a
  maintenance escape hatch if needed.
- Add an integrity checker that verifies: master has `Unit` rows, every unit has
  merge provenance, root/child parentage is valid, the analysis NWB object is
  present, and `SpikeSortingOutput.CurationV2` registration exists.
- Test that direct master writes are rejected by default.

### 4. Medium-high: concat-backed curations are underselected by preprocessing restrictions

`ConcatenatedRecordingSelection` includes `preprocessing_params_name`
(`src/spyglass/spikesorting/v2/session_group.py:342-348`), but
`CurationV2.resolve_restriction()` classifies `preprocessing_params_name` as a
single-recording key only (`src/spyglass/spikesorting/v2/curation.py:1511-1518`).
Concat keys are limited to `concat_recording_id`, session-group fields, and
`motion_correction_params_name` (`src/spyglass/spikesorting/v2/curation.py:1524-1529`).
The resolver rejects any restriction that mixes concat keys with recording keys
(`src/spyglass/spikesorting/v2/curation.py:1604-1612`).

Impact: a restriction like `{"preprocessing_params_name": "..."}` searches only
recording-backed curations and silently omits concat-backed curations produced
with the same preprocessing recipe. A restriction combining
`concat_recording_id` with `preprocessing_params_name` can be rejected as
contradictory even though both fields are valid on
`ConcatenatedRecordingSelection`.

Fix direction:

- Treat `preprocessing_params_name` as a shared source key.
- For preprocessing-only restrictions, union recording-backed and concat-backed
  branches.
- For concat plus preprocessing, route through `ConcatenatedRecordingSelection`
  rather than rejecting the key mix.
- Add tests through both `CurationV2.resolve_restriction()` and
  `SpikeSortingOutput.get_restricted_merge_ids()`.

### 5. Medium: source-part exclusivity is enforced at read time but not audited or find-existing safe

Artifact and sorting selections model source identity with mutually exclusive
part tables. `resolve_source()` correctly raises unless exactly one source part
exists (`src/spyglass/spikesorting/v2/artifact.py:669-708`,
`src/spyglass/spikesorting/v2/sorting.py:1007-1044`). But the find-existing
paths look only through the requested source part:

- `ArtifactDetectionSelection._find_existing_pk()` joins master to the chosen
  source part and returns the deterministic ID if found
  (`src/spyglass/spikesorting/v2/artifact.py:586-631`).
- `SortingSelection._find_existing_pk()` does the same for the requested
  recording or concat source, then checks optional artifact state
  (`src/spyglass/spikesorting/v2/sorting.py:852-914`).

The maintenance helper only reports zero-source orphans:
`find_orphaned_masters()` returns masters whose source-part count sums to zero,
not masters with multiple source parts
(`src/spyglass/spikesorting/v2/utils.py:317-351`).

Impact: a bypassed extra source-part row can remain invisible to
`prune_orphaned_selections()` and can be accepted by `insert_selection()` as an
existing deterministic row for the requested source. The error is only surfaced
later when a populate/accessor calls `resolve_source()`, and broad joins can see
the malformed master before that happens.

Fix direction:

- Before returning an existing deterministic PK, count all mutually exclusive
  source parts and raise if the row has anything other than exactly one source.
- Add a general `audit_source_part_integrity()` helper that reports zero-source
  and multi-source masters separately.
- Add bypass tests for artifact and sorting selections with both source parts
  attached.

### 6. Medium: `UnitMatch.Pair` FKs do not constrain rows to the selected curation universe

Generated UnitMatch insertion canonicalizes matcher output against the pinned
member curations, which is good. The part table itself, however, only declares
projected FKs to any `CurationV2.Unit` for each endpoint
(`src/spyglass/spikesorting/v2/unit_matching.py:459-480`). The comment says the
FK guarantees "a pair cannot reference a unit absent from the pinned curation,"
but the schema-level FK only guarantees that the units exist somewhere in
`CurationV2.Unit`.

Impact: a direct `UnitMatch.Pair.insert1()` can reference real units from
unrelated curations and still satisfy DataJoint FKs. Later `TrackedUnit`
derivation may fail, or direct pair queries may report cross-session matches
outside the `UnitMatchSelection.MemberCuration` universe.

Fix direction:

- Add a part-level insert guard or integrity checker requiring both endpoints to
  belong to the selected `UnitMatchSelection.MemberCuration` set, and requiring
  cross-member rather than same-member pairs.
- Add a test that direct insertion of an existing but unpinned unit is rejected
  by the guard/checker.

### 7. Medium-low: `TrackedUnit` re-derives the matchable universe from current labels

`UnitMatch.make_fetch()` freezes the matchable unit IDs at fetch time and passes
them into matcher execution
(`src/spyglass/spikesorting/v2/unit_matching.py:580-625`). `TrackedUnit.make()`
does not read a persisted frozen universe; it rebuilds `node_universe` from the
current `CurationV2` labels
(`src/spyglass/spikesorting/v2/unit_matching.py:864-907`).

The code comment expects a loud failure when a stored `UnitMatch.Pair` references
a unit that is no longer matchable. That is true for paired units because graph
derivation sees an edge endpoint outside the node universe. But unmatched
singletons have no pair row. If an unmatched unit is relabeled after UnitMatch
population but before `TrackedUnit.populate()`, it can simply disappear from the
current node universe and therefore from tracked-unit output.

Impact: paired stale units fail loudly, while singleton stale units can drift
silently. The tracked graph can stop representing the same unit universe the
matcher actually saw.

Fix direction:

- Persist the frozen matchable universe used by `UnitMatch` as a part table or
  compact serialized artifact, and have `TrackedUnit` derive from it.
- Alternatively, treat pinned curations as immutable after UnitMatch population
  and enforce that at the curation-label write boundary.
- Add DB-boundary tests for paired and singleton relabeling between UnitMatch and
  TrackedUnit.

## Coverage and Documentation Follow-ups

These are lower-level review items that reinforce the findings above:

- Concat merge-id helper tests assert inclusion, not exclusivity. Add exact
  result tests for `concat_recording_id`, session-group fields, and
  `motion_correction_params_name`.
- Export lineage tests are root-only. Add a root-plus-child curation fixture and
  assert exporting the child `merge_id` includes the child units file and
  excludes the root/sibling units files.
- `ConcatenatedRecordingSelection.insert_selection()` still ignores a supplied
  `concat_recording_id` while other deterministic selection helpers reject a
  mismatched supplied ID (`src/spyglass/spikesorting/v2/session_group.py:377-383`).
  This was already captured in the API review, but it is also an identity
  consistency issue.
- The docs/notebook still risk steering users from root `merge_id` to downstream
  workflows instead of final curated `merge_id`; this is already tracked in the
  combined triage as R2.

## Suggested Fix Order

1. Add `update1` guards to deterministic selection masters and direct-write
   guards to `CurationV2`.
2. Decide the `SessionGroup.Member` immutability vs frozen-snapshot policy before
   retaining concat or UnitMatch outputs.
3. Fix `CurationV2.resolve_restriction()` so preprocessing restrictions cover
   concat-backed curations.
4. Add source-part integrity auditing and strengthen find-existing checks.
5. Harden `UnitMatch.Pair` and freeze or guard the UnitMatch-to-TrackedUnit unit
   universe.
