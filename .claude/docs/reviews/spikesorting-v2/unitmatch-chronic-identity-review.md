# Spike Sorting V2 UnitMatch / chronic identity semantics review

Date: 2026-06-25

Scope: cross-session unit identity in the v2 UnitMatch path, including
`SessionGroup` member pinning, per-member curation identity, matcher input
provenance, pair graph integrity, `TrackedUnit`, and the portable NWB artifact.
This pass does not re-review concat-and-sort except where the two workflows share
chronic/session-group assumptions.

Method: local source/test inspection plus one independent explorer-agent pass.
No tests were run for this review.

## Executive Summary

The core shape is sound: UnitMatch pins one `CurationV2` row per
`SessionGroup.Member`, validates that those curations belong to the selected
members, freezes matchable units for the heavy matcher compute step, canonicalizes
matcher output before normal insertion, and derives tracked units through a
deterministic graph policy.

The remaining risks are about making the biological identity contract durable at
every boundary. The biggest issue is that `UnitMatch` and `TrackedUnit` do not
consume exactly the same frozen unit universe: `UnitMatch.make_fetch` freezes the
matchable set, while `TrackedUnit.make` re-derives it from current labels later.
There are also places where DataJoint schema FKs are weaker than the code-level
contract, and where effective matcher inputs are not fully snapshotted with the
produced graph.

## What Looks Solid

- `UnitMatchSelection.insert_selection` validates one curation per session-group
  member and rejects curations owned by the wrong member
  (`src/spyglass/spikesorting/v2/unit_matching.py:228`).
- The selection hash is recomputed from `MemberCuration` rows before populate,
  so raw-inserted master/part mismatches are caught
  (`src/spyglass/spikesorting/v2/unit_matching.py:555`).
- `UnitMatch.make_fetch` freezes the member plan, chronological recording date,
  and matchable unit ids before heavy compute
  (`src/spyglass/spikesorting/v2/unit_matching.py:580`).
- Matcher output is canonicalized before normal insertion: unpinned curation
  pairs, same-member pairs, and reversed duplicates are rejected
  (`src/spyglass/spikesorting/v2/_matcher_graph.py:99`).
- The backend has explicit guards for same-probe geometry mismatch and silent
  UnitMatchPy session drops
  (`src/spyglass/spikesorting/v2/_unitmatch_backend.py:239`,
  `src/spyglass/spikesorting/v2/_unitmatch_backend.py:271`).

## Findings

### 1. High: `TrackedUnit` can drift from the actual UnitMatch run universe

`UnitMatch.make_fetch` freezes `matchable_unit_ids` for the matcher run
(`src/spyglass/spikesorting/v2/unit_matching.py:580`). Later,
`TrackedUnit.make` intentionally re-derives the node universe from current
curation labels (`src/spyglass/spikesorting/v2/unit_matching.py:867`,
`src/spyglass/spikesorting/v2/unit_matching.py:899`).

That loud-fails if a relabeled unit participates in a stored pair, but it can
silently change singleton behavior for unmatched units. A unit that was in the
matcher's frozen universe but later receives an excluded label can disappear
from tracked-unit output; a unit that becomes matchable later can appear as a new
singleton even though the matcher never considered it.

Impact:

- `TrackedUnit` is not guaranteed to be a partition of the exact unit universe
  used by the matcher run.
- Downstream chronic identity can change after `UnitMatch` population without a
  new UnitMatch run, especially for unmatched units.

Recommended fix:

- Persist a frozen per-run unit universe, for example
  `UnitMatch.MatchableUnit` or an NWB member/unit table.
- Have `TrackedUnit` consume that frozen universe instead of calling
  `CurationV2.get_matchable_unit_ids` again.
- Add a regression test that relabels an unmatched unit between `UnitMatch` and
  `TrackedUnit` populate and asserts the chosen contract.

### 2. High: `UnitMatch.Pair` schema FKs do not enforce pinned-curation graph integrity

The `Pair` part table FKs each endpoint to global `CurationV2.Unit` projections
(`src/spyglass/spikesorting/v2/unit_matching.py:471`). The docstring says this
guarantees a pair cannot reference a unit absent from the pinned curation
(`src/spyglass/spikesorting/v2/unit_matching.py:462`), but the schema only
guarantees that each endpoint exists in some `CurationV2.Unit` row. It does not
encode that the endpoint curation is one of this `UnitMatchSelection`'s
`MemberCuration` rows, nor that pairs are cross-member, unique, or canonical.

Normal `make_insert` is protected because it passes matcher output through
`canonicalize_match_pairs` (`src/spyglass/spikesorting/v2/_matcher_graph.py:99`).
Direct maintenance inserts into the part table are not protected by that helper.

Impact:

- Raw or maintenance inserts can add pairs for unpinned curations under an
  existing `unitmatch_id`.
- Duplicate/reversed pairs can corrupt the graph that `TrackedUnit` later uses.

Recommended fix:

- Override `Pair.insert` / `insert1` or add a validated insertion helper that
  checks endpoints against `UnitMatchSelection.MemberCuration`, rejects
  same-member/reversed duplicates, and validates probability ranges.
- Add a test that a pair for an existing but unpinned curation is rejected at the
  table boundary.

### 3. Medium-high: effective matcher and graph parameters are not frozen with the produced graph

The selection identity stores `matcher_params_name` and `curation_set_hash`, and
the hash only covers `(member_index, sorting_id, curation_id)`
(`src/spyglass/spikesorting/v2/_matcher_graph.py:37`). `UnitMatch` stores the
analysis NWB pointer, pair count, and runtime
(`src/spyglass/spikesorting/v2/unit_matching.py:450`). `TrackedUnit` later
fetches the current `MatcherParameters` row to interpret the graph thresholds
(`src/spyglass/spikesorting/v2/unit_matching.py:882`).

Separately, important effective inputs live outside the stored params snapshot:
`extract_unitmatch_bundle` has defaults for `ms_before`, `ms_after`,
`max_spikes_per_unit`, and `seed`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:107`), and the backend
starts from UnitMatchPy's dynamic `default_params.get_default_param()`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:232`).

Impact:

- Replacing or mutating a `MatcherParameters` row can reinterpret existing pairs
  with new `tracked_unit_threshold` or `max_strict_nodes`.
- UnitMatchPy default changes, waveform-window changes, or bundle-seed changes
  can alter results without necessarily changing `unitmatch_id` or artifact
  provenance.

Recommended fix:

- Persist a matcher-parameter fingerprint or full effective snapshot on
  `UnitMatch`, including graph thresholds, UnitMatchPy defaults that are relied
  on, bundle extraction params, dependency versions, and resolved job kwargs.
- Have `TrackedUnit` read that frozen snapshot, not the live lookup row.
- Treat matcher parameter rows as immutable after first use, including replacing
  `replace=True` mutation paths with explicit new parameter names.

### 4. Medium: the UnitMatch NWB artifact is pair-only and not self-describing

`_unitmatch_nwb.py` writes only a `unit_match_pairs` table with side sorting ids,
curation ids, unit ids, match probability, drift estimate, and FDR estimate
(`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:1`,
`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:29`). The DataJoint tables carry
the richer context, but the NWB artifact is described as the exportable analysis
artifact (`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:12`).

Impact:

- Outside DataJoint, the NWB does not say which `SessionGroup` was matched, what
  member order was used, what recording dates were frozen, what matcher params
  were effective, or which matchable units were considered.
- A user can read pair rows but cannot audit whether the artifact represents the
  same biological identity universe as the database row.

Recommended fix:

- Add a structured UnitMatch provenance object/table to the NWB artifact:
  session group, ordered members, pinned curations, recording dates, matchable
  unit universe, matcher name/version/params, bundle params, dependency versions,
  and graph policy.
- Keep the pair table as the compact edge list, but make it portable with the
  context required to interpret those edges.

### 5. Medium: same-probe/channel-universe assumptions fail late

UnitMatch requires a shared channel geometry. The backend checks channel-position
shape/value compatibility after session bundles have already been extracted
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:239`). The expensive bundle
extraction starts in `_extract_and_match` before that backend-level rejection
(`src/spyglass/spikesorting/v2/unit_matching.py:780`).

Impact:

- Cross-day groups with different bad-channel exclusions, channel order, or probe
  projection can fail after significant waveform extraction work.
- The selection layer does not expose a clear preflight reason tied to specific
  members/channels.

Recommended fix:

- Preflight ordered channel ids and channel positions in `insert_selection` or
  `UnitMatch.make_fetch`.
- Report mismatched members and channels before extraction.
- Document the required common channel universe for chronic matching.

### 6. Medium-low: tracked-unit brain-region output drops disambiguators

`TrackedUnit.get_unit_brain_regions` fetches `sorting_id`, `curation_id`, and
`unit_id`, but returns only `sorting_id`, `unit_id`, and `region_name`
(`src/spyglass/spikesorting/v2/unit_matching.py:984`,
`src/spyglass/spikesorting/v2/unit_matching.py:996`).

Impact:

- Consumers lose `curation_id`, `member_index`, session/date, and source NWB
  context at the point where chronic identity is being resolved.
- The result is harder to join back to a specific pinned member, especially in
  mixed curation workflows.

Recommended fix:

- Include `unitmatch_id`, `tracked_unit_id`, `member_index`, `curation_id`,
  `nwb_file_name`, and recording date in the returned DataFrame.
- Keep the current three-column view only as a convenience projection.

