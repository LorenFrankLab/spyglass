# Phase 4c — UnitMatch across independently sorted daily concatenations

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs-motion-and-matching.md#matching-independently-sorted-recording-groups)

**Status:** Planned owner-requested feature addition (2026-09-19), not implemented. Depends on phase 4a and phase 3c's effective-source/lineage contract; also inherits phases 3a/3b's geometry and observed-time invariants. The original single-session matcher remains a supported input shape. Original merge-gate scheduling is recorded separately in PLAN.md.

**Inputs to read first:**

- `src/spyglass/spikesorting/v2/unit_matching.py` — `UnitMatchSelection.MemberCuration`, `_curation_member_identity`, `_validate_member_curations`, frozen `MatchableUnit`, pair validation, bundle extraction and `TrackedUnit` session/region resolution.
- `_unit_match_planning.py`, `_pipeline_run.py::run_v2_unit_match`, `_pipeline_types.py` — candidate discovery, one-curation-per-member planning, and receipts.
- `_matcher_graph.py` — chronological ordering, same-session rejection, pair canonicalization, strict clique partition and original-session counting.
- `matcher_protocol.py`, `_unitmatch_backend.py`, `_unitmatch_nwb.py` — self-contained bundle protocol and exported pair identities; the backend has no concat-source restriction.
- `session_group.py::ConcatenatedRecordingSelection.MemberSnapshot`, `MemberBoundary`, `concat_member_curation.py` and `_concat_recording.py::split_unit_spike_trains` — parent unit identity and original-member maps. Member `get_recording` currently returns the pre-correction cache; it is not an implicit matching source.
- `curation.py` and the phase-3c source service — selected processed traces, curation generation, parent sorting and electrode provenance.
- Historical evidence: `git show 9343433f:src/spyglass/spikesorting/v2/unit_matching.py` introduced the concat guard. The earlier remediation plan's claim that matching already consumes concat curations is inaccurate; use the shared design's correction.
- `tests/spikesorting/v2/test_unitmatch.py`, `test_unitmatch_backend.py`, concat-member/source tests, and `notebooks/py_scripts/10_Spike_SortingV2_CrossSession.py`.

**Designs referenced:** [matching recording groups](designs-motion-and-matching.md#matching-independently-sorted-recording-groups), [per-unit halves](designs.md#per-unit-halves), [independent motion stage](designs-motion-and-matching.md#independent-motion-stage).

## Tasks

- **Explicit matching inputs.** Replace the mandatory one-original-member/one-curation model with an ordered `UnitMatchSelection.Input` and frozen `InputRecording` membership/provenance. One input is a complete independently curated single-recording or same-day concat sort, including the pinned curation generation and applied motion/masks. Existing SessionGroup-based discovery becomes an adapter; add direct daily-sort selection to the planner. Version/recreate v2 schemas and receipts explicitly.
- **Identity and ownership validation.** Hash the selected curations, generation/source snapshots and matcher settings. At selection, compute and pair-insert boundaries, validate exact source membership, duplicate sorts/generations, shared constituent sessions, overlapping original acquisition spans, and effective electrode geometry. Reject a concat plus its own member sort. Preserve existing wrong-source, duplicate/reversed pair and frozen-universe defenses. Do not merely delete the concat guard or identify a concat by its first member.
- **Original-time ordering.** Freeze the full original recording map and UTC acquisition times; order matcher inputs chronologically using the complete input's source times and a deterministic tie-break. No live-group lookup may change an existing run. Define pair orientation independently of chronological feed order.
- **Bundles from the selected parent sort.** Extract one bundle per daily concat from the phase-3c effective unwhitened source and parent curation. Use phase 4a's temporal split and exclude waveform support crossing acquisition/member boundaries or artifact exclusions. Sampling caps, returned exclusions and warnings remain per matching input. If all units are excluded, name that input and fail before writing its bundle; never shift input-to-session mapping.
- **Preserve parent unit identity.** Keep pair and graph nodes keyed by `(sorting_id, curation_id, unit_id)` with pinned generation provenance. Do not insert one graph node per `ConcatMemberCuration` export or generate within-parent matches. Keep both-directions acceptance, the strict partition and singleton handling unchanged.
- **Tracking back to original members.** Store/expose each tracked parent's member projections, sparse/reordered unit IDs, original clocks, observation support and per-member regions. Distinguish number of matching inputs from original sessions with detected spikes; an empty exported train must not increase detected-session counts. Update `n_sessions_observed` semantics/name and all readers explicitly; never duplicate the anchor member's region across the group.
- **Protocol/export/reporting audit.** Keep the matcher backend consuming directories and echoing curated-unit identities; update input terminology and metadata as needed. Update NWB input maps, pairing exports, curation hashes, receipts, diagnostics, selection introspection, cleanup/deletion protection and reconstruction of provenance without the live SessionGroup. Report whether each bundle used applied motion and its exact reference.
- **Combined workflow and evidence.** Exercise two independently sorted daily concatenations with real UnitMatchPy, then the opt-in mask/concat/motion/sort/curate/match workflow. Extend evaluation to gradual waveform change, intermittent units, disappearance/reappearance and a three-day conflicting-match case. Record incorrect identities, recall and singleton rates on fixed populations; synthetic dropout recovery from phase 4a is not a long-duration tracking validation.
- **Docs with implementation.** Update `SpikeSortingV2.md`, migration and storage docs, the cross-session notebook/script, and CHANGELOG. Show daily concat -> daily curation -> cross-day matching -> original-member analysis. State supported input shapes, overlap restrictions and counting semantics. Remove the stale claim that this worked before this phase; mark the earlier remediation-plan statements as superseded when documenting implementation history.

## Deliberately not in this phase

- Matching two windows of one original NWB session, or comparing alternative sorts of overlapping source data.
- Multi-day concatenations as individual matching inputs; the supported new input is a same-day assembly matched to other independent inputs.
- Changing UnitMatch thresholds, strict graph grouping, waveform mean/median choice, or adding a new matcher backend.
- Re-sorting or re-curating members during matching; explicit source curations are prerequisites.

## Validation slice

Proposed targets in `tests/spikesorting/v2/test_unitmatch_concat.py`, with shared fixtures in the existing v2 fixture modules.

| Test/experiment | Required assertion |
| --- | --- |
| `test_match_inputs_accept_single_and_daily_concat_sorts` | Single/single, single/concat and concat/concat selections validate and resolve their exact parent curation and processing result |
| `test_match_inputs_reject_duplicate_and_overlapping_sources` | Same parent twice, different curations of one parent, concat plus its member, and shared original sessions fail before bundle extraction |
| `test_matching_snapshot_survives_live_group_changes` | Frozen members, curation generation, motion reference, dates and unit universe determine the run; later edits cannot silently change it |
| `test_daily_bundle_uses_corrected_parent_and_valid_support` | Real extracted values come from the selected parent trace frame; no window crosses a member boundary or gap; partial-member units get two valid halves |
| `test_concat_units_are_not_duplicated_in_match_graph` | Each parent unit occurs once, independent of number of exported members; no artificial within-parent edges |
| `test_tracked_units_map_to_original_member_times_and_regions` | Gapped clocks, unequal member lengths and differing member regions resolve correctly; empty trains do not imply detection; distinct-session counts do not count intervals twice |
| `test_corrected_geometry_is_checked_before_matching` | Incompatible retained channel sets/positions fail clearly; original geometry matching is insufficient |
| `test_concat_pairs_and_input_provenance_round_trip` | NWB and DB preserve pair identities, expanded membership, source/correction references and chronological ordering with shuffled discovery order |
| `test_daily_concat_matches_planted_units` (`slow`, real UnitMatchPy) | Two independently sorted two-member days recover planted correspondences with fixed distractors; existing phase-4a gates still pass |
| Three-day long-duration benchmark (`slow`) | Reports false identities and recall under drift/disappearance/reappearance, keeps strict conflicting-edge behavior, and meets a committed held-out acceptance manifest |
| Full corrected-daily-sort workflow (scheduled/manual) | Original-member analysis after motion, sorting, matching and reload returns the planted identities/times; off/no-motion controls and scientific metrics are recorded |

## Fixtures and execution

- Deterministic two-day/two-member synthetic data, with a three-day extension, sparse unit IDs, gaps and a unit absent from selected members. Plant known correspondences and distractors; do not make every unit match every other unit.
- Real SI extraction and UnitMatchPy in the matcher-extra lane; small source/identity tests in regular v2 CI. The real-sorter + motion chain belongs in the scheduled/manual lane, with small structural coverage per PR.
- Commit seed/fixture manifests, define numeric scientific gates before held-out execution, and report missing real-data evidence explicitly. Phase 5 owns lane wiring; this phase owns assertions.

## Review

Before opening the implementation PR, obtain independent review of expanded ownership, parent-versus-member identity, time/geometry frames, counting semantics, and the real matcher evidence. Verify that existing single-session behavior remains valid, overlap guards survive direct-insert attempts, and the feature works through original-member analysis rather than stopping at a successful backend call.
