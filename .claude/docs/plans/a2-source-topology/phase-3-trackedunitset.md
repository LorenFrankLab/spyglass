# Phase 3 — `TrackedUnit` → `TrackedUnitSet` (computed-set-with-parts)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

> **SPUN OFF — this is a standalone PR against `spikesorting-v2`, NOT part of the
> A2 comparison spike** (2026-08-19 review: orthogonal to the topology change,
> ships on its own merits, and keeping it in the spike confounded the topology
> isolation). It fixes a real defect regardless of the A2 go/no-go. This file is
> its design spec; it does not enter the spike's parity gates or ledger.

Fix the "one `UnitMatch` key → N master rows → partial-deletion never
regenerates" defect by remodeling to a one-row-per-key computed set with the
tracked units and members as parts. **Correction (test-power/scientific review):**
the identity test must compare the **member partition** (frozenset of frozensets of
`(sorting_id, curation_id, unit_id)`) + `median_match_probability` keyed by
member-set — NOT positional `tracked_unit_id`; and `test_partial_delete_regenerates`
must assert deleting the **master** regenerates while deleting a **part row then
populate is a no-op** (the documented trap), matching the design residual.

**Inputs to read first:**
- [unit_matching.py:1322-1466](../../../../src/spyglass/spikesorting/v2/unit_matching.py) — `TrackedUnit` definition (1335-1343), `Member` part (1345-1351), and `make()` (1353-1466) whose loop `self.insert(master_rows)` (1441-1466) inserts N masters/key.
- [curation.py:152,210](../../../../src/spyglass/spikesorting/v2/curation.py) — `-> CurationV2.Unit` / `-> Sorting.Unit.proj(...)` (part-as-FK-target precedent for the composite Member FK).
- The **consumer set** (Task 3): `spikesorting_merge.py`, `_pipeline_run.py`, `curation.py`, `sorting.py`, `unit_matching.py`, `_matcher_graph.py`, `exceptions.py`, `_params/matcher.py`; notebook `10_Spike_SortingV2_CrossSession`; test files touching `TrackedUnit`.

**Contracts referenced:** none from the spike — this is a **standalone PR**, so
it does not use the spike's `shared-contracts.md` gates or topology ledger. Its
own correctness bar is the validation slice below (member-partition identity +
the partial-delete regeneration semantics).

**Designs referenced:** [designs.md#trackedunitset](designs.md#trackedunitset) — table declarations + the one-master-N-parts `make()` skeleton.

## Tasks
- Declare `TrackedUnitSet` (`-> UnitMatch`, **1 row/key**, `--- n_tracked_units, policy_used`).
- Part `TrackedUnitSet.TrackedUnit` (`-> master; tracked_unit_id: int; --- n_sessions_observed, median_match_probability`).
- Part `TrackedUnitSet.Member` — **composite FK to the part** (`-> TrackedUnitSet.TrackedUnit; -> CurationV2.Unit`), NOT `-> master; tracked_unit_id;…` (that leaves `tracked_unit_id` unenforced).
- Rewrite `make()`: derive the tracked units, then insert **one** master + N `TrackedUnit` + M `Member` rows in one transaction. `key_source` stays `UnitMatch`.
- Sweep the enumerated consumer set: replace direct `TrackedUnit` reads with `TrackedUnitSet.TrackedUnit`; add/adjust a natural-key accessor (`get_tracked_units(unit_match_key)`) so consumers never reach into the part by hand.
- Note the residual in the class docstring: individual part-row deletion won't retrigger populate (presence = the single master); recompute = delete the master.
- **Change accounting (PR-local, NOT the spike ledger):** *deleted* = the old top-level `TrackedUnit` Computed table + its N-masters `self.insert(master_rows)` (batched at [unit_matching.py:1465](../../../../src/spyglass/spikesorting/v2/unit_matching.py)); *added* = the `TrackedUnitSet` master + two parts + the `get_tracked_units` accessor; *unchanged* = the `make()` derivation logic. This is a standalone PR — it does not enter the spike's topology ledger.

## Deliberately not in this phase
- Topology / artifact / import-light — Phases 1-2-4. This phase touches only the tracked-unit table + its consumers.

## Validation slice
| Test | Asserts |
| --- | --- |
| `test_one_master_per_unitmatch_key` | `make()` inserts exactly one `TrackedUnitSet` master per `UnitMatch` key (regardless of N tracked units). |
| `test_master_delete_regenerates` | deleting the **master** and re-populating restores the full set (the old N-masters partial-population trap is gone); deleting a **part row** then populating is a **no-op** (the documented trap). |
| `test_member_fk_enforced` | inserting a `Member` with a nonexistent `tracked_unit_id` raises an integrity error (composite FK enforced). |
| `test_trackedunitset_partition_identity` | injected cross-session: the member **partition** (frozenset of frozensets of `(sorting_id, curation_id, unit_id)`) + `median_match_probability` keyed by member-set match A1's `TrackedUnit` results (NOT positional `tracked_unit_id`). |
| `test_consumers_updated` | each enumerated consumer resolves via the natural-key accessor (no direct part access); cross-session notebook cell runs. |

## Fixtures
A two-session `UnitMatch` fixture (reuse the cross-session test fixtures) driving
`derive_tracked_units`; injected sortings so tracked-unit identity is deterministic.

## Review
Dispatch `code-reviewer`. Confirm: exactly one master per key; the composite
`Member` FK enforces `tracked_unit_id`; the full consumer set is swept (no stale
`TrackedUnit` reads); the identity test compares the member **partition** (not
positional `tracked_unit_id`) against A1; the partial-delete test encodes the
documented no-op trap; no plan-milestone names in artifacts. (Standalone PR — no
spike ledger/gates.)
