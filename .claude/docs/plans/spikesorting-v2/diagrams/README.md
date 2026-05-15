# Spike Sorting v2 — Table Structure Diagrams

These diagrams show the DataJoint table structure that v2 introduces at each phase/checkpoint. Each diagram is **cumulative**: it shows the tables that exist after that checkpoint is complete, including baseline Spyglass tables that v2 depends on.

These files are reference diagrams, not execution instructions. If process
wording in a diagram conflicts with `PLAN.md` or a phase/checkpoint file, follow
the active plan text.

Diagrams use Mermaid `erDiagram` syntax — GitHub renders them natively. To view locally, install a Mermaid-capable previewer or paste into [mermaid.live](https://mermaid.live).

## Reading order

| File | Phase | What it covers |
| --- | --- | --- |
| [00-baseline.md](00-baseline.md) | (before any v2 work) | Existing Spyglass tables v2 plugs into: `Session`, `Electrode`, `BrainRegion`, `AnalysisNwbfile`, `LabTeam`, `IntervalList`, `SpikeSortingOutput` merge master. |
| [01-phase-1.md](01-phase-1.md) | Phase 1 | Single-session sort MVP: `SortGroupV2`, `Recording`, `ArtifactDetection`, `Sorting`, `CurationV2`. Includes the forward-compat concat scaffolding (`SessionGroup`, `ConcatenatedRecording` declared but not populated). |
| [02-phase-2.md](02-phase-2.md) | Phase 2 | Quality metrics + auto-curation (`AnalyzerCuration`) and recompute verification tables (`RecordingArtifactRecompute*`, `SortingAnalyzerRecompute*`). |
| [03-phase-3.md](03-phase-3.md) | Phase 3 | Same-day chronic concat — no new tables, `ConcatenatedRecording.make()` body fills in. Schema is identical to Phase 1; diagram highlights what activates. |
| [04-phase-4.md](04-phase-4.md) | Phase 4 | Cross-session unit tracking: `UnitMatch`, `TrackedUnit`. |
| [05-phase-5.md](05-phase-5.md) | Phase 5 | FigPack curation UI: `FigPackCuration`. No other schema changes (orchestrator helpers are Python-only). |

## Conventions

### Table tier annotations

Every table is annotated with its DataJoint tier:

| Tier | Meaning |
| --- | --- |
| **Manual** | Inserted by user code (or `insert_selection()` helpers). |
| **Lookup** | Inserted once with `contents`; holds named parameter sets. |
| **Computed** | Populated by a `make()` method from upstream selection rows. |
| **Imported** | Populated from an external source (typically an NWB file). |
| **Part** | Sub-table of a master; shares the master's primary key plus one additional attribute. |

### Relationship notation

Mermaid ER syntax:

| Symbol | Meaning |
| --- | --- |
| `||--o{` | One-to-many (parent has many children). Used for foreign keys. |
| `||--||` | One-to-one. Used for Computed tables keyed exactly off their upstream Selection. |
| `}o--o{` | Many-to-many. Not used in DataJoint directly — appears via Part tables. |

### v2 cross-phase symbols

- **NEW IN PHASE N**: tables introduced in the current phase.
- **PHASE 1 (FORWARD-COMPAT)**: tables declared in Phase 1 whose `make()` body raises `NotImplementedError` until a later phase. The zero-migration policy requires FK targets to exist when the FK is declared.
- **(EXTERNAL)**: pre-existing Spyglass tables that v2 does not modify.
- **(MERGE PART)**: a part of `SpikeSortingOutput` that v2 adds; the merge master is unchanged.

### What the diagrams don't show

- Secondary attributes (only PKs and key FKs are shown for readability). See [designs.md](../designs.md) for full schemas.
- Indexes and constraints (Mermaid ER doesn't render these).
- The Python-only orchestrator helpers `run_v2_pipeline()` / `run_v2_unit_match()` — they are functions, not tables.
- The `SpikeSortingOutput.CurationV1`, `.ImportedSpikeSorting`, `.CuratedSpikeSorting` parts that v2 does not touch.

## Source of truth

The schema source of truth is [designs.md](../designs.md). When in doubt, the design doc wins.
