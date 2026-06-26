# Phase 2 — Identity & relational integrity

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Close the identity/schema gaps that are cheapest to fix while pre-production: master
rows can be silently retargeted (R28), runtime-semantics aliases change meaning under
a fixed version (R6), `TrackedUnit` re-derives its universe instead of using the
frozen one (R7), and schema-init isn't an upgrade workflow (R8). One PR; **five**
related task groups (A–E, the last added by the Round-3 reviews; review per group is fine).

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/utils.py:171-254](../../../../src/spyglass/spikesorting/v2/utils.py#L171-L254) (`SelectionMasterInsertGuard`) and [:257-309](../../../../src/spyglass/spikesorting/v2/utils.py#L257-L309) (`ImmutableParamsLookup.update1` — the pattern to mirror); [:317-351](../../../../src/spyglass/spikesorting/v2/utils.py#L317-L351) (`find_orphaned_masters`).
- [src/spyglass/spikesorting/v2/curation.py:64-65](../../../../src/spyglass/spikesorting/v2/curation.py#L64-L65), [session_group.py:55-83](../../../../src/spyglass/spikesorting/v2/session_group.py#L55-L83) — plain `dj.Manual` masters.
- [src/spyglass/spikesorting/v2/_params/preprocessing.py:7,106-135](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L106-L135); [_selection_identity.py:40-46](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L40-L46); [_concat_recording.py:22,93-146](../../../../src/spyglass/spikesorting/v2/_concat_recording.py#L93-L146); [recording.py:2200-2223](../../../../src/spyglass/spikesorting/v2/recording.py#L2200-L2223) (`DriftEstimate`).
- [src/spyglass/spikesorting/v2/unit_matching.py:438-457](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L438-L457) (`UnitMatch` def), [:587-625](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L587-L625) (freeze), [:864-907](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L864-L907) (`TrackedUnit.make` re-derivation).
- [src/spyglass/spikesorting/v2/_lookup_validation.py:83-93,264-282](../../../../src/spyglass/spikesorting/v2/_lookup_validation.py#L83-L93); [__init__.py:18-61](../../../../src/spyglass/spikesorting/v2/__init__.py#L18-L61); [sorting.py:320-323](../../../../src/spyglass/spikesorting/v2/sorting.py#L320-L323) (the one backfill that exists).

**Contracts referenced:** [Master-row identity immutability](shared-contracts.md#master-row-identity-immutability) — this phase implements it.

## Tasks

### Group A — R28 master-row immutability + source-part audit

1. **Add an `update1` deny to `SelectionMasterInsertGuard`.** Mirror `ImmutableParamsLookup.update1` (`utils.py:282-309`):

   ```python
   def update1(self, row, *, allow_master_mutation=False):
       """Reject in-place mutation: a selection master's secondary columns feed
       its deterministic id; mutating them retargets the id under live
       dependents. Insert a new selection instead."""
       if not allow_master_mutation:
           raise dj.errors.DataJointError(
               f"In-place update1 of {self.__class__.__name__} is not supported: "
               "its identity is derived from these columns. Insert a new selection "
               "via insert_selection(). Pass allow_master_mutation=True only for a "
               "deliberate maintenance edit of a row with no live references."
           )
       super().update1(row)
   ```

   This automatically covers all six masters that mix the guard in: `RecordingSelection` (`recording.py:745`), `ArtifactDetectionSelection` (`artifact.py:441`), `SortingSelection` (`sorting.py:651`), `UnitMatchSelection` (`unit_matching.py:198`), `ConcatenatedRecordingSelection` (`session_group.py:330`), `AnalyzerCurationSelection` (`metric_curation.py:646`).

2. **Guard `CurationV2` and `SessionGroup` direct writes.** These are plain `dj.Manual` (`curation.py:65`, `session_group.py:65`). Add insert + `update1` denial that routes legitimate writes through the factory methods. Give each master an `insert`/`insert1`/`update1` override (a small shared mixin, e.g. `FactoryOnlyMaster`, or reuse `SelectionMasterInsertGuard`'s mechanism) that raises unless a keyword bypass is passed; then thread the bypass through the **two exact factory insert sites**: `CurationV2.insert_curation`'s master insert at **`curation.py:820`** (`cls.insert1(master_row)` → `cls.insert1(master_row, allow_master_mutation=...)`/the chosen bypass kwarg) and `SessionGroup.create_group`'s master insert at **`session_group.py:208`** (`cls.insert1({...})`). DataJoint forwards `insert1`'s `**kwargs` to `insert`, so the bypass reaches the override. **If you add the guard but forget the keyword at these two sites, `insert_curation`/`create_group` break immediately** — the regression tests (`test_curation_*`, `test_session_group_concat`) will catch it, but wire both sites. (`UnitLabel` already validates its own inserts — leave it.)

3. **Add a multi-source integrity audit.** `find_orphaned_masters` (`utils.py:347-351`) flags only **zero-source** masters; a master with **two** source-part rows is caught lazily only when `resolve_source` runs. Add a sibling:

   ```python
   def audit_source_part_integrity(master_table, part_tables) -> list[dict]:
       """Return masters whose source-part row count != 1 (0 = orphan, >1 =
       ambiguous source). Complements find_orphaned_masters' zero-only check."""
   ```

   **Critical — count only the recording-source parts.** For `SortingSelection`, count **only `[RecordingSource, ConcatenatedRecordingSource]`** (the XOR-exactly-one pair). `ArtifactDetectionSource` (`sorting.py:696-710`) is a deliberately **independent zero-or-one** part — a valid sorting with an artifact pass has it, so including it would make every artifact-bearing sorting count==2 and be falsely flagged. (`find_orphaned_masters`/`prune_orphaned_selections` already pass only the recording-source pair, `sorting.py:996-998`.) For `ArtifactDetectionSelection`, count only `[RecordingSource, SharedArtifactGroupSource]`. The validation test must include a "valid Recording+Artifact sorting is NOT flagged" case, not only the two-recording-sources case.

### Group B — R6 runtime-alias versioning

4. **Mark the preprocessing runtime-order change as a distinct version and enforce it going forward.** Bump `PREPROCESSING_SCHEMA_VERSION` (`_params/preprocessing.py:7`) so the current filter→reference runtime order is a versioned state (the v3 change shipped under an unchanged version — `:106-135`). Re-seed the shipped default rows at the new version. Add/extend a test that pins the documented runtime semantics to a checked-in expected version so a **future** silent order change fails CI. **Scope honestly:** this does NOT retroactively identify which *existing* dev rows predate the order flip — a pre-flip and a post-flip row are both `schema_version=3` with a **byte-identical blob** (the order change moved no field, `:119`), so content-based detection is structurally impossible. The bump forces a re-seed of shipped defaults and pins *future* detection; it is not a back-detector. (Identity stays name-based — R6's full-recipe-as-identity is a decided non-goal.)

5. **Store the resolved motion/drift preset, not just the alias.** `resolve_motion_correction` maps `"auto"`→`AUTO_SAME_DAY_PRESET="rigid_fast"` (`_concat_recording.py:22,136-145`); `DriftEstimate` hardcodes `_DEFAULT_PRESET="dredge_fast"` (`recording.py:2200`). Persist the **resolved** preset string as a secondary attribute on the concat row. **Carrier threading required:** the preset is resolved in `make_compute` (`session_group.py:835`) but `ConcatRecordingComputed` (`:550-565`) and `make_insert` (`:900-911`) do not carry it — add a `motion_preset` field to that NamedTuple, thread it through the return, and extend `make_insert`'s `insert1` dict + the column. `DriftEstimate` already stores `motion_preset` and its `_DEFAULT_PRESET` is a concrete preset (not an alias), so it likely already satisfies this — just confirm. Add an `AUTO_SAME_DAY_PRESET_VERSION` module constant so a change to the alias mapping is a visible, test-pinned bump rather than a silent re-meaning.

### Group C — R7 frozen matchable universe

6. **Persist the frozen matchable universe and have `TrackedUnit` consume it.** The matchable set is frozen in `UnitMatch.make_fetch` (`unit_matching.py:592-623`) but only transiently in `member_plan`; `TrackedUnit.make` re-derives `node_universe` from **current** labels (`:890-906`), so a relabel between the two stages silently drops singletons. Add a part table to persist the frozen set:

   ```python
   class MatchableUnit(SpyglassMixin, dj.Part):
       definition = """
       -> master                 # UnitMatch
       member_index: int
       sorting_id: uuid
       curation_id: int
       unit_id: int
       """
   ```

   **Carrier threading (required — `make_insert` does not currently receive the frozen set).** The frozen `matchable_unit_ids` lives only in `make_fetch`/`make_compute`'s `member_plan`; the `UnitMatchComputed` carrier (`unit_matching.py:85-96`) and `make_insert` (`:683-691`) do not carry it. So: add the frozen matchable triples as a field on the `UnitMatchComputed` NamedTuple (keep it a NamedTuple per the stage-result constraint — do not convert to dataclass), thread it from `make_compute`, and have `make_insert` write `MatchableUnit` from it. `TrackedUnit.make` (`:890-906`) then reads `node_universe` from `UnitMatch.MatchableUnit & key` instead of calling `CurationV2().get_matchable_unit_ids(...)`. **Canonicalize on read** to match `derive_tracked_units`' node identity: `MatchableUnit` stores `(sorting_id, curation_id, unit_id)` but `node_universe` tuples are `(str(sorting_id), int(curation_id), int(unit_id))`. Keep `derive_tracked_units`' loud failure for edges outside the (now frozen) universe. **Target a singleton (no-`Pair`) unit in the relabel test** — a matched-unit relabel already raises today and would pass for the wrong reason.

### Group D — R8 schema-init upgrade safety

7. **Generalize the outer-version backfill.** Only `SorterParameters` backfills the outer `params_schema_version` from the blob (`sorting.py:320-323`); the single-schema Lookups rely on the column default, so `_assert_schema_version_matches` early-returns when the column is omitted (`_lookup_validation.py:83-93`). Move the backfill into `validate_lookup_rows` (`_lookup_validation.py:130-139`) as a default `per_row_hook` behavior: when `params_schema_version` is absent, fill it from the validated blob's `schema_version` before the drift check, for every Lookup.

8. **Add a stale-default content audit.** `initialize_v2_defaults` (`__init__.py:18-61`) re-runs each `insert_default()` idempotently and never compares a stored same-name row's *content* to the shipped content (`reject_duplicate_parameter_content` skips existing-PK rows — `_lookup_validation.py:264-281`). Add:

   ```python
   def verify_v2_default_catalog(*, strict: bool = False) -> list[dict]:
       """For each shipped default name already in the DB, compare the stored
       content fingerprint to the shipped content. Return mismatches (stale
       defaults). Raise DuplicateParameterContentError-style error if strict."""
   ```

   Call it (non-strict, log warnings) at the end of `initialize_v2_defaults`, and expose it for an admin to run strict. This closes SCHEMA-2/SCHEMA-3 — it flags a stored same-name **shipped-default** row whose content diverged from the reseeded default. (It does **not** back-detect which old *dev* rows predate the preprocessing order flip — same version, identical blob; see task 4. Don't claim that.)

9. **Docs.** CHANGELOG entries for: the new `update1` deny on masters; the preprocessing version bump (note dev rows must be re-seeded); the new `UnitMatch.MatchableUnit` part; `verify_v2_default_catalog`. Update the migration doc (`docs/src/.../SpikeSortingV2_Migration.md`) with the version-bump + re-seed note.

## Additional tasks (Round-3 reviews)

### Group A (extends R28)

10. **UCI-2 / REL-6 / CLIFE-5 — constrain `UnitMatch.Pair` to the pinned-curation universe.** `Pair` FKs each endpoint to `CurationV2.Unit.proj(...)` globally (`unit_matching.py:475-476`), so the schema only guarantees the unit exists in *some* curation, not in this selection's `MemberCuration`. Normal `make_insert` is safe only because it runs `canonicalize_match_pairs`; a raw/maintenance `Pair.insert` bypasses that. Add a validated `Pair.insert`/`insert1` override (or an FK to `MemberCuration`) that checks both endpoints against the selection's `MemberCuration`, rejects same-member and reversed-duplicate edges, and validates probability ranges. (R28 as originally written did not cover the Pair FK.)

### Group E — shared-artifact member-set freeze (R39, artifact side)

11. **AVTM-1 — freeze `SharedArtifactGroup` membership into `artifact_detection_id`.** The shared-group artifact identity payload is `{source_kind, artifact_detection_params_name, shared_artifact_group_name}` only (`_selection_identity.py:259-263`); `ArtifactDetection.make_compute` reads **live** `SharedArtifactGroup.Member` and the part is unguarded, so the scanned set can change under a fixed `artifact_detection_id`. Apply the **same member-snapshot pattern phase-4c uses for concat**: fold an ordered member-set fingerprint into the artifact identity, or store a snapshot and validate current-vs-snapshot at compute. (Sibling of phase-4c CONCS-1; keep the two implementations consistent.)

### Group C (extends R7)

12. **UCI-5 — geometry preflight before dense bundle extraction.** The channel-position compatibility check lives inside the backend `match()` (`_unitmatch_backend.py:239-257`), which runs only *after* `extract_unitmatch_bundle` builds dense per-session bundles (`unit_matching.py:794`). Add an ordered channel-id/position preflight to `UnitMatchSelection.insert_selection` (or `make_fetch`) so a cross-day geometry mismatch fails fast, before the expensive extraction.

## Deliberately not in this phase

- **Full recording construction-recipe-as-identity** (R6 full) — decided non-goal; identity stays name-based, this phase only makes semantics changes *detectable*/versioned.
- **Child-curation composition semantics** (R7 composition / CLIFE-2) — a Phase-5 FigPack design decision (overview "Phase 5 adjustments"); here only the frozen-universe drift is fixed.
- **PK changes** to session-global keys (R9-PK) — see overview Open Question 1; not in this plan.
- **Provenance columns** (effective seed, versions) — phase-3a.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_selection_identity.py::test_update1_rejected_all_masters` (new) | parametrized over all six selection masters + `CurationV2` + `SessionGroup`: `cls.update1(changed_row)` raises `DataJointError` match `"not supported"`; `allow_master_mutation=True`/factory-bypass succeeds. Mirrors `test_sorter_parameters_update1_rejected_in_place`. |
| `test_selection_identity.py::test_curationv2_direct_insert_rejected` (new) | `CurationV2.insert1(row)` raises; `CurationV2.insert_curation(...)` still succeeds (factory bypass works). Same for `SessionGroup.create_group`. |
| `test_integrity.py::test_audit_source_part_integrity` (new) | dual *recording*-source rows are flagged; a zero-source master is flagged; **a valid `RecordingSource + ArtifactDetectionSource` sorting is NOT flagged** (artifact part excluded from the count); a clean single-recording-source master is not flagged. |
| `test_params_validation.py::test_preprocessing_runtime_version_pinned` (new/extended) | the preprocessing runtime-order version equals a checked-in expected constant **AND** the actual runtime call order is pinned (extend `test_preprocessing_order.py`'s `["bandpass_filter","common_reference","remove_channels"]` order-signature assertion), so a future order change without a version bump fails — pinning the constant alone is insufficient. |
| `test_concat_recording.py::test_resolved_motion_preset_persisted` (new) | a concat built with `preset="auto"` stores the resolved `"rigid_fast"` (not `"auto"`) and the alias-version constant. |
| `test_unitmatch.py::test_tracked_unit_uses_frozen_universe_after_relabel` (new) | freeze a UnitMatch, relabel a member unit, populate `TrackedUnit`: the node universe matches the frozen `MatchableUnit` rows (singleton not dropped) — **fails before** the frozen-universe change. |
| `test_parameter_identity.py::test_outer_version_backfilled_for_all_lookups` (new) | inserting a row with `params_schema_version` omitted backfills it from the blob for **every** param Lookup the task covers — Preprocessing/Artifact/Motion/Waveform **and Matcher + QualityMetric + AutoCurationRules** (or, if those use a bespoke insert path, assert the backfill is applied there too); drift check still trips on an explicit mismatch. |
| `test_pipeline_run.py::test_verify_v2_default_catalog_flags_stale` (new) | seed a default, mutate the stored blob, `verify_v2_default_catalog()` returns the mismatch; a clean catalog returns `[]`. |
| `test_pipeline_run.py::test_initialize_v2_defaults_runs_catalog_audit` (new) | `initialize_v2_defaults` invokes `verify_v2_default_catalog` (non-strict) and logs a warning when a stale default is present — the wiring, not just the helper. |
| `test_unitmatch.py::test_pair_insert_rejects_unpinned_curation` (new, UCI-2) | a raw `UnitMatch.Pair.insert` referencing a unit outside the selection's `MemberCuration`, a same-member or reversed-duplicate edge, **or an out-of-range `match_probability`**, raises; the canonical `make_insert` path is unaffected. |
| `test_artifact_integration.py::test_shared_group_member_set_frozen` (new, AVTM-1) | editing `SharedArtifactGroup.Member` after materialization changes the `artifact_detection_id` (or raises a member-drift error at compute) — the scanned set can't change under a fixed id. |
| `test_unitmatch.py::test_geometry_preflight_fails_before_extraction` (new, UCI-5) | a member with mismatched channel ids/positions is rejected at `insert_selection`/`make_fetch`, before any dense bundle extraction runs. |
| (regression) `single_session/test_curation_*`, `test_session_group_concat.py`, `test_unitmatch.py::test_tracked_unit_make_seeds_singletons`, `test_initialize_v2_defaults_is_idempotent` | factory inserts, concat, normal TrackedUnit populate, and idempotent init all still pass. |

## Fixtures

Master `update1`/insert guards and the source-part audit are mostly DB-free or use
the minimal `dj_conn` + a single inserted selection. The frozen-universe test reuses
`two_session_curated_group` (`test_unitmatch.py:508`) + a relabel of one member. The
schema-init tests reuse the default-seeding fixtures. The concat-preset test reuses
`chronic_2_session_minirec` (`conftest.py:340`).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `update1` is rejected on every selection master AND `CurationV2`/`SessionGroup`; the existing `insert_curation`/`create_group` paths still insert (factory bypass wired correctly, not left broken).
- The frozen-universe change persists `MatchableUnit` and `TrackedUnit` reads it; the relabel-divergence test fails on pre-change code.
- The preprocessing version bump re-seeds shipped rows and the runtime-version pin test would catch a future silent change.
- The outer-version backfill applies to all Lookups, not just `SorterParameters`; the drift check still trips on explicit mismatch.
- `verify_v2_default_catalog` detects same-name content drift and is wired into `initialize_v2_defaults` non-strict.
- No PK changes; provenance columns are not added here (that's 3a).
- CHANGELOG + migration-doc updates present; no plan/phase references in code or tests.
