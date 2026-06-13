# Overview — Scope, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All paths are `src/spyglass/spikesorting/` unless noted. Line numbers verified
against the working tree at planning time; re-confirm before editing (the files
move).

**Phase 1 — reference defaults (grouping helpers):**

- `v2/recording.py:275` — `SortGroupV2.set_group_by_shank`: change
  `reference_mode="none"` / `reference_electrode_id=None` defaults to
  auto-inherit-from-config; add a v1-style `references: dict | None` arg;
  resolve a `(reference_mode, reference_electrode_id)` *per electrode group*
  instead of one scalar for the whole call.
- `v2/recording.py:360-382` — the current `omit_ref_electrode_group` branch
  (only fires for the call-wide `reference_mode == "specific"` scalar):
  rewrite to use each group's *resolved* specific reference.
- `v2/recording.py:463` — `SortGroupV2.set_group_by_electrode_table_column`:
  same default change + auto inheritance + global override; **no** per-group
  `references` mapping (out of scope — see Non-Goals).
- `v2/recording.py:420-448` / `:582-608` — the master-row build loops that
  currently write the single call-wide `reference_mode` /
  `reference_electrode_id`: write the per-group resolved values instead.
- `v2/utils.py:329` — `assert_reference_not_member`: already raises when a
  `"specific"` reference is a sort-group member. Reuse it at group-creation
  time (it currently only runs at materialization, `_recording_materialization.py:183`).
- `v2/utils.py:201` — `_validate_reference_fields`: unchanged. Still the
  insert-time guard that `reference_mode ∈ {none, global_median, specific}`
  and `reference_electrode_id` is non-null **iff** `specific`. The resolver's
  output must satisfy it.
- `v2/utils.py:194` — `ReferenceMode = Literal["none","global_median","specific"]`:
  unchanged. **No `"auto"` member is added** — auto is helper-level only.
- `common/common_ephys.py:81` — `Electrode.original_reference_electrode = -1`:
  read-only source of the per-electrode configured reference (sentinel int).
- `spikesorting/utils.py:13` — v1 `get_group_by_shank`: read-only reference for
  the algorithm Phase 1 mirrors. **Note the bug at line 90**: on mixed
  references v1 *constructs* `ValueError(...)` but never `raise`s it (it falls
  through with `sort_ref_id` unbound on that branch). Phase 1 raises for real.

**Phase 2 — filter-before-reference (runtime + provenance):**

- `v2/_recording_materialization.py:341` — `apply_pre_motion_preprocessing`:
  currently reference→filter (`:362-398`). Reorder to filter→reference→drop-ref.
- `v2/_recording_materialization.py:402` — `filtering_description`: currently
  lists `common reference …; bandpass filter …` (`:415-423`). Reorder to
  `bandpass filter …; common reference …`.
- `v2/_recording_materialization.py:60` — `restrict_recording`: **untouched.**
  It already includes the specific reference channel in the slice (`:186-201`)
  and calls `assert_reference_not_member` (`:183`); the ref channel is dropped
  after referencing inside `apply_pre_motion_preprocessing`. Confirm the
  docstring at `:95-98` still reads correctly after the reorder (it does —
  "dropped after referencing" stays true).
- `v2/_params/preprocessing.py:8-14, 40-59, 94-106` — docstrings that assert
  "references first, then bandpass … load-bearing … do not reorder to filter
  then reference." These now describe the *removed* behavior; correct them.
- `v2/recording.py:621` — `PreprocessingParameters` Lookup, `:632`
  `params_schema_version=3`, `:641` `_DEFAULT_CONTENTS`: **unchanged version.**
  Per the pre-release schema policy we keep `schema_version=3` (the params blob
  shape is identical — only the runtime *interpretation* of the order changes).
  Dev rows are regenerated, not migrated.
- `docs/src/Features/SpikeSortingV2_Migration.md`, `CHANGELOG.md` — document
  both v1↔v2 behavior changes (see Rollout).

## Scope and dependency policy

### Goals

- v2 grouping helpers inherit `Electrode.original_reference_electrode` by
  default (per electrode group), matching the useful default v1 has.
- Sentinel semantics stay v1-compatible: `None`/`-1` → no reference, `-2` →
  global median, `>= 0` → specific electrode.
- Explicit overrides remain: `reference_mode="none"` forces no reference;
  `"global_median"` forces CMR; `"specific"` + id forces a fixed reference.
- Auto-derivation fails *loud* on ambiguity (mixed configured references in one
  group) and on a specific reference that is itself a sort-group member.
- Preprocessing applies **bandpass filter first, then reference** — the
  signal-processing-preferred order. This changes numeric output ONLY on the
  `global_median` branch with `operator="median"` (median is non-linear); for
  `specific` / `none` / `average` the filter and reference steps are linear and
  commute, so output is unchanged.
- Both behavior changes are documented as v1→v2 differences for users.

### Non-Goals

- **No new `"auto"` `reference_mode` stored in the DB.** Auto is resolved in
  the helper to one of the three existing modes; the column never sees `"auto"`.
- **No per-group `references` mapping for `set_group_by_electrode_table_column`.**
  It gets auto-inheritance + a single global override only. Add the per-group
  mapping later if a concrete use case appears.
- **No `params_schema_version` bump.** (Spec asked for v3→v4; overridden by the
  project pre-release policy — see Open Question 1.)
- **No backwards-compatibility shim** for the old `reference_mode="none"`
  default or the old reference-first order. Pre-production; regenerate dev rows.
- No change to `SortGroupV2` / `PreprocessingParameters` table *structure*
  (columns, FKs).

### Dependency policy

No new dependencies. Uses existing `spikeinterface.preprocessing`
(`bandpass_filter`, `common_reference`) and numpy.

## Metrics

- Reference resolution: DB-free `resolve_group_reference` tests cover every
  sentinel branch + the mixed and in-group raises; one integration test covers
  the helper-level missing-`references`-key raise.
- Preprocessing order: stub/mocked tests assert `bandpass_filter` is invoked
  before `common_reference` on the specific and global-median paths (a
  code-structure guard — the order is only numerically observable on the
  median branch).
- `filtering_description` returns `"bandpass filter … Hz; common reference (…)"`.
- Full v2 suite green in a DataJoint + SpikeInterface environment; the
  `global_median`/`median` numeric baseline is regenerated and its comparison
  encodes the intentional order divergence (specific-reference sessions
  re-materialize identically and need no re-baselining).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Changing the helper default silently alters every existing caller's sort groups (references now inherited, not "none"). | Pre-production; the change is intended. Phase 1 updates the fixtures/tests that create default sort groups so the new inherited references are the asserted expectation, not an accident. |
| Filter-before-reference changes numeric output on the `global_median`/`median` branch → its v1↔v2 *numeric* parity breaks there. (`specific` / `none` / `average` are linear and stay identical.) | Phase 2 re-captures the `global_median` numeric baseline and updates that comparison so it no longer asserts trace-identical parity with v1's reference-first order. Param-level parity (`test_parity_canonical`) is unaffected (params unchanged) and stays. |
| Auto-references make "specific reference is a sort-group member" common (a tetrode referenced to one of its own wires). | `assert_reference_not_member` already raises; Phase 1 runs it at group-creation time so the error surfaces at `set_group_by_*` instead of deep in `make`. `omit_ref_electrode_group` is the documented escape hatch. |
| Keeping `schema_version=3` while behavior changes could confuse a reader into thinking nothing changed. | The params *blob* genuinely did not change shape; only the runtime order did. Phase 2 adds a schema-history docstring note recording the order change at v3, and documents it in the migration doc + CHANGELOG. |
| Existing deterministic selection ids / cached recordings can point at rows produced under the old semantics because neither `SortGroupV2`'s PK nor `PreprocessingParameters`' PK changes. | Treat this as a dev-state invalidation, not a migration: delete/recreate affected `SortGroupV2` rows and delete downstream v2 `RecordingSelection` / `Recording` / artifact / sorting / curation rows plus cached analysis files before repopulating. No stale row from the old no-reference or reference-first behavior may remain in the validation DB. |

## Rollout Strategy

All-at-once, no feature flag (pre-production). Dev databases are regenerated
rather than migrated. The rollout checklist is:

1. Delete affected downstream v2 rows from curation/sorting/artifact/recording
   outputs back through `RecordingSelection` for the sessions under test.
2. Delete or archive cached analysis files / folders produced from the old
   reference-first or default-no-reference behavior.
3. Drop and re-run `SortGroupV2.set_group_by_*` so helper-created rows carry the
   inherited reference defaults.
4. Re-populate `Recording` and downstream v2 outputs from the clean state.

Acceptance gate: after regeneration, no validation fixture or dev DB row used by
the v2 suite may depend on a pre-change `RecordingSelection` id or cached
recording artifact.

**User-facing documentation ships inside each phase** (not deferred):

- Phase 1 → `SpikeSortingV2_Migration.md` + `CHANGELOG.md`: v2 grouping helpers
  now inherit the configured reference by default (matching v1), replacing the
  earlier v2 `"none"` default; mixed configured references now raise (v1
  silently mis-referenced).
- Phase 2 → `SpikeSortingV2_Migration.md` + `CHANGELOG.md`: v2 now bandpass-
  filters **before** referencing — an intentional divergence from v1's
  reference-then-filter order — which changes preprocessed/sorted output
  numerically versus v1.

## Open Questions

1. **`params_schema_version` bump — resolved.** Spec requested v3→v4. Overridden
   by the recorded pre-release policy ("do not bump `params_schema_version`
   pre-release; edit defs in place / regenerate dev rows"). **Decision: keep
   `schema_version=3`**, regenerate dev rows, document the order change in the
   schema-history docstring + migration doc. Revisit only if the params *blob*
   shape changes.

## Estimated Effort

- Phase 1: ~150-250 LOC across `v2/recording.py` (two helpers + a resolver) and
  `v2/utils.py` (resolver + reuse of `assert_reference_not_member`), plus
  ~150 LOC of pure-resolver unit tests and fixture/expectation updates.
- Phase 2: ~40 LOC of runtime/provenance change in
  `v2/_recording_materialization.py`, docstring corrections in
  `_params/preprocessing.py`, ~80 LOC of order tests, plus baseline regen and
  doc/CHANGELOG entries.
