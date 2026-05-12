# Lessons learned — designing a new Spyglass pipeline

Notes from building the v3 spike sorting pipeline plan (Spring 2026). Intended as input to the upstream Spyglass skill at `~/.claude/skills/spyglass/` and as a checklist for the next new-pipeline effort. Plan artifacts that grounded these lessons live at `.claude/docs/plans/spikesorting-v3/`.

## Process lessons

### Validate the design with `code_graph.py` BEFORE implementing

Single biggest win of the v3 plan process. Concrete pattern:

1. **Precondition check** — for every Spyglass table the new pipeline FKs into, run `code_graph.py describe <table>` and record the actual PK / FK / column types. Catches drift between the design assumption and the current state. Findings from the v3 pass are at [`.claude/docs/plans/spikesorting-v3/precondition-check.md`](plans/spikesorting-v3/precondition-check.md).

2. **Draft schema validation** — materialize the proposed schemas as a single Python file at `src/<package>/v<N>/_draft.py` (or under a plan-artifact directory + symlinked into the src tree). NO `@schema` decoration; just class declarations with their `definition` strings and `make()` raising `NotImplementedError`. Then run with file paths relative to the code-graph source root:
   - `python /path/to/spyglass-skill/scripts/code_graph.py --src src describe <NewTable> --file spyglass/<package>/v<N>/_draft.py --json` — verifies the `definition` parses, FKs resolve, PK structure matches the design.
   - `python /path/to/spyglass-skill/scripts/code_graph.py --src src path --up <NewTable> --file spyglass/<package>/v<N>/_draft.py --json` — walks every ancestor through the upstream Spyglass tree. Unresolved FK names show up here.
   - `python /path/to/spyglass-skill/scripts/code_graph.py --src src path --down <NewTable> --file spyglass/<package>/v<N>/_draft.py --json` — walks every descendant; useful for confirming the planned fan-out (e.g., "CurationV3 should have 4 downstream consumers across Phases 2/4/5").
   - Cycles, name collisions with v0/v1, and missing parts all surface. Review the JSON `warnings` block; any unaccounted `heuristic_resolution` warning is a blocker. Some current warnings are expected only when explicitly documented, e.g. `AnalysisNwbfile` resolving between `common_nwbfile.py` and `custom_nwbfile.py`, or draft v3 class names colliding with v0/v1 names.

3. **Recurring per-phase check** — every phase's "Review" section ends with: "`code_graph.py describe` returns clean output for every new table; `path --up/--down` chains match the design DAG; JSON warnings are empty or explicitly accounted for."

Concrete findings this caught for v3:
- `BrainRegion.region_id` is `smallint auto_increment` PK, NOT `region_name` — plan text referred to "BrainRegion row named 'Unknown'" as if it were the key; corrected to "insert the row and use its auto-generated region_id".
- `ElectrodeGroup` ALSO has `-> BrainRegion`; brain region exists at TWO levels (group + electrode), important for multi-region probes.
- `bad_channel` and `contact_side_numbering` are `enum("True", "False")` strings on Spyglass tables (not int/bool); v3 helpers filtering Spyglass `Electrode` must use string comparison.
- v1 `CurationV1.object_id` is `varchar(72)` — wider than the plan's `varchar(40)` draft; widened for parity.
- `SpikeSortingOutput.source_class_dict` exists in TWO places (module-level + inherited from `_Merge`); the dispatch test must verify which one is consulted at runtime.
- `CurationV3.UnitLabel` had to become a separate part table after review caught that a scalar `curation_label: varchar(32)` could not represent the documented `labels: dict[int, list[str]]` API or v1's indexed multi-label NWB column.

**Skill-level recommendation**: add a "Schema design validation" section to the spyglass skill router with this pattern. Currently the skill teaches `code_graph.py` mostly for DEBUGGING existing pipelines (one-off "where is X declared"); the new-pipeline use case is a different workflow but uses the same tool.

### State the design's assumptions explicitly and verify each one

The v3 plan went through 7 rounds of review. Most rounds found cross-document drift — claims in one file contradicted by another. The pattern that worked was:

- **Capture every design decision as an explicit `Invariant — do not weaken` line in shared-contracts.md.** When subsequent phases reference it, they link by anchor. Removes ambiguity about "is this still the rule?"
- **Make decision-output tables (e.g., "Phase 0 storage benchmark decides binary-cache vs NWB-only via this matrix") rather than ungrounded commitments.** The "binary cache is faster" claim was repeated as fact for many rounds before getting flagged as SI-community folklore — the matrix forces measurement.
- **Convert vague "won't fix" footnotes to explicit traceability tables.** v3's "v1 GitHub issues v3 closes" + "issues NOT fixed in v3" tables came after the third round of reviewers wondered what the plan's scope actually was.

### Read GitHub issue threads, don't trust an agent's summary

In an early round I dispatched a subagent to mine ~20 spike-sorting issues and trusted its summaries. The reviewer caught that issue #133 was misrepresented (I said "LabTeam in PK blocks shared sorts"; the real issue is `rmtree` of another team's recording folder when two teams run identical sorts). Re-reading every cited issue directly via `gh issue view N` was the only reliable check.

**Skill-level recommendation**: when the spyglass skill suggests mining issues for design feedback, advise reading threads directly. Agent summaries miss the comments where root causes get pinned down.

### Use the `spyglass-skill` discipline for destructive ops in new code

The spyglass skill teaches inspect-before-destroy for ad-hoc work. The same pattern applies to NEW table classmethods that can cascade-delete. v3's `SortGroupV3.set_group_by_*(delete_existing_entries=True, confirm=True)` returns a `DeletionPreview` (rows to delete + downstream cascade row counts + reclaimable disk + cross-team-owned rows) and raises unless `confirm=True` is also passed. Adopting this in the schema design — not just at the ad-hoc-tool layer — closes the silent-overwrite class of bugs the v1 `set_group_by_shank` had.

## Schema design lessons specific to Spyglass

### Zero-migration is achievable but requires forward-compatibility decisions upfront

The "no `alter()` calls across phases" constraint forced anticipating later phases in earlier ones. Concretely:

- Phase 1's `SortingSelection` had to declare **both** nullable FKs (`-> [nullable] Recording` and `-> [nullable] ConcatenatedRecording`) even though Phase 1 only used `Recording`. Phase 3 then just lifts the `NotImplementedError` guard in `insert_selection()`; the schema is unchanged.
- Same pattern for `ArtifactDetectionSelection` (nullable Recording vs SharedArtifactGroup).
- `Sorting.Unit`, `CurationV3.Unit`, and `CurationV3.UnitLabel` part tables had to land in Phase 1 even though Phase 2 / 4 consume them.

Recording this as a contract (shared-contracts § Zero-Migration Schema Forward-Compatibility) — with a per-phase table of "Phase 1 decisions that anticipate Phase N" — made it reviewable.

### Polymorphic FKs via nullable typed FK pair + XOR

When a Computed table can take input from one of N upstream tables, the temptation is a "loose" discriminator column (`source: enum(...), source_id: uuid`). DataJoint can't enforce referential integrity on this — the resolution falls to runtime validation that's easy to bypass.

The pattern that worked: **two nullable typed FKs with XOR enforced in `insert_selection()`**. DataJoint enforces both FKs structurally; the helper enforces "exactly one non-null". v3 uses this for SortingSelection (Recording XOR ConcatenatedRecording) and ArtifactDetectionSelection (Recording XOR SharedArtifactGroup).

### UUID-keyed Selection + Computed pair, not composite PK

The Selection / Computed pair (v1's pattern) factored out cleanly when both share a UUID PK. The Computed inherits the Selection's PK via `-> Selection`, giving a single-column FK target for downstream tables. Composite-PK FKs from downstream onto a Selection table get unwieldy fast (v0's `SpikeSortingRecording` was keyed by 6 columns).

### Cross-session tables anchor on a deterministic member

`AnalysisNwbfile` has a single non-null `-> Nwbfile` parent ([common_nwbfile.py:630](src/spyglass/common/common_nwbfile.py#L630)). A computed table that spans multiple sessions (v3's `ConcatenatedRecording`, `UnitMatch`) still needs ONE Nwbfile to anchor the analysis NWB. The rule v3 adopted: **always use the first SessionGroup.Member's nwb_file_name (ordered by member_index) as the deterministic anchor**. Cross-session provenance stays queryable through the explicit `SessionGroup.Member` part, not through the analysis NWB's session.

Same rule applies to per-unit Electrode FKs on concat-source `Sorting.Unit` — `Electrode` inherits `ElectrodeGroup`, so the effective key includes `nwb_file_name`, `electrode_group_name`, and `electrode_id`. A concat sort has one Electrode row per member for the same physical channel identity. v3 anchors `Sorting.Unit -> Electrode` to the first member; per-session brain regions for tracked units are derived from `TrackedUnit.Member` walking each member separately.

### Brain-region tracing wants a part table, not a join helper

v1's `CurationV1.get_sort_group_info()` does `fetch(limit=1)` on the SortGroup-Electrode-BrainRegion join — fine for single-region tetrodes but wrong for multi-region polymer probes. The lab feedback was "incredibly hard to trace a unit back to a brain region."

v3 fix: persist per-unit `(electrode_id, peak_amplitude_uV)` to a `Sorting.Unit` part table at sort time. Brain region is then a constant-time `Sorting.Unit * Electrode * BrainRegion` join. `CurationV3.Unit` mirrors through merges. Adding the part table costs ~5 columns and ~50 LOC; the UX improvement (one-call `get_unit_brain_regions(merge_key)` on `SpikeSortingOutput`) is large.

Generalizable lesson: **whenever a "trace X back to Y" query is hard for users, ask whether the answer can be precomputed at compute-table-make time and stored on a part table.** Per-fetch joins are silently expensive AND silently wrong (limit=1, fetch_first, etc.).

### Model list-valued user concepts as part tables, not scalar columns

v1 curation labels are list-valued per unit: one unit can carry labels like `["noise", "reject"]`, and unlabeled units have an empty list. An early v3 draft documented `labels: dict[int, list[str]]` but stored `curation_label=NULL: varchar(32)` on `CurationV3.Unit`. That mismatch would have forced either lossy serialization, an undocumented scalar-only API, or a later schema migration.

The corrected pattern is `CurationV3.UnitLabel`, one row per `(unit_id, label)`, with no rows for unlabeled units. Filtering helpers then get named semantics:

- `get_unit_brain_regions(include_labels=[...])` includes units with any requested label.
- `get_matchable_unit_ids(exclude_labels={"reject", "noise", "artifact"})` excludes any unit carrying an excluded label, even if it also has `accept`, and includes unlabeled / `accept` / `mua` units.

Generalizable lesson: **when the Python API says `list[...]`, the DataJoint schema needs a part table or a real list-valued storage decision.** A nullable scalar column is a red flag under a zero-migration policy.

## Pydantic-validated parameters

Every Spyglass Lookup table has a `params: blob` column. v1's typo-at-populate failure mode (user inserts misspelled param name; everything looks fine until `populate()` raises somewhere deep) is fixable by validating `params` against a Pydantic model at `insert1` time.

Pattern: per-table `_params/<table_name>.py` with `<TableName>ParamsSchema(BaseModel)`. Lookup table overrides `insert1` to call `model_validate(row["params"])` and serialize via `model_dump()`. Versioned schemas via `params_schema_version: int` secondary attribute. Documented in `shared-contracts.md § Pydantic Parameter Schema Convention`.

Catches: typos, wrong types, missing required fields, out-of-range values, JSON-incompatibility (the NaN issue from #1556). All at insert time, before any populate runs.

## Storage / artifact-management lessons

- **Treat "binary cache vs NWB-wrapped" as a measured decision, not a belief.** v3 plan deferred this to a Phase 0 benchmark with an explicit decision matrix; both storage designs are documented conditionally. Folklore from external communities (SI's "binary is faster") doesn't transfer without measurement.
- **Recompute machinery is real production infrastructure**, not a nice-to-have. v1's `RecordingRecompute` (env-tracked deterministic-recompute verification + `delete_files(matched=1)` storage reclamation) is what makes "delete X TB of old caches; regenerate on demand" safe. v3 ports the three-table pattern.
- **Side artifacts (binary caches, analyzer folders) need rebuild-without-delete helpers.** The v3 `Recording.get_recording()` and `Sorting.get_analyzer()` regenerate missing files via private `_rebuild_*` methods, NEVER via `(self & key).delete_quick(); self.populate(key)` — which would cascade through downstream curation / merge rows and destroy provenance.

## NWB / ingestion lessons

- **The ElectricalSeries name matters for ingestion**: Spyglass's `Raw._source_nwb_object_name` lookup matches one of `["e-series", "electricalseries", "ephys", "electrophysiology"]` ([common_ephys.py:289-294](../../src/spyglass/common/common_ephys.py#L289-L294)). Any NWB-producing converter (real lab tooling, MEArec fixture wrappers) must hit this list — the default in `MEArecRecordingInterface` may differ and silently produce un-ingestable files.
- **Use `trodes_to_nwb` as the structural reference** for any new NWB-producing converter for Spyglass ingestion. Its YAML metadata schema (experimenter, subject, electrode_groups, ntrode_electrode_group_channel_map, etc.) is what Spyglass downstream tooling expects.

## What I'd put in the spyglass skill

Concrete additions (or strengthening of existing sections):

1. **New section: "Designing a new pipeline / new tables"** (currently `custom_pipeline_authoring.md` covers extending existing pipelines but doesn't have the "design before implement" workflow).
   - Subsection: "Validate schema designs with `code_graph.py`" — recipes for precondition check, draft schema validation, recurring per-phase check.
   - Subsection: "Forward-compatibility patterns" — nullable typed FK pair + XOR; UUID-keyed Selection/Computed pairs; cross-session anchor rules.
2. **Strengthen `merge_methods.md`** with the "source_class_dict has two locations" gotcha — when registering a new part on a merge master, verify which one the dispatch methods actually consult.
3. **Add to `feedback_loops.md`**: "When designing destructive class-method APIs (anything that could cascade-delete), inspect-before-destroy belongs in the schema, not just at the ad-hoc CLI layer."
4. **Add to `runtime_debugging.md`** (and possibly a new `common_mistakes.md` entry): the `enum("True", "False")` string convention on `Electrode.bad_channel`, `Probe.contact_side_numbering`, etc. — easy to miss when porting filter code from NWB-DataFrame contexts to DataJoint contexts.
5. **New skill or section**: "Designing for multi-session analyses" — covers the anchor-to-first-member rule for `AnalysisNwbfile`, FK design for tables that span multiple Sessions, the explicit-curation-pinning pattern (vs implicit "latest" lookups).

Most generalizable single recommendation: **make `code_graph.py describe + path --up/--down` a routine part of new-table review, not an optional debug step.** It catches structural errors that markdown review misses, doesn't need the database, runs in seconds.
