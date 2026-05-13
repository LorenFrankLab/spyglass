# Spike Sorting v2 Implementation Plan

**Status:** Not started.

A next-generation spike sorting pipeline for Spyglass that (1) targets SpikeInterface ≥0.104 with the SortingAnalyzer API, keeps the v1 production sorters (MountainSort 4 and `clusterless_thresholder`), adds MS5 and KS4, and preserves a custom-row path for other SpikeInterface sorters, (2) supports cross-session unit tracking via a pluggable matcher backend (UnitMatchPy first, DeepUnitMatch via the same plugin slot later), (3) handles chronic recordings — same-day concatenation as the default path, with multi-day opt-in (sort-then-match via Phase 4 UnitMatch is the recommended cross-day workflow) — (4) makes unit → brain region tracing first-class via a `Sorting.Unit` part table populated at sort time (fixes the v1 multi-region under-reporting bug), (5) reduces user-facing table touchpoints from ~7 to ~2 via a `run_v2_pipeline()` convenience API with Pydantic-validated parameters and FigPack as the curation UI, and (6) plugs into the existing `SpikeSortingOutput` merge table so downstream consumers (decoding, ripple, MUA, `SortedSpikesGroup`) keep working unchanged. **Zero schema migration**: every table is final from the phase that introduces it. v0 and v1 stay in-tree indefinitely.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need v1 feature parity boundaries?** [feature-parity.md](feature-parity.md).
5. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
6. **Need upstream-repo line refs / on-disk format details?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, risks, rollout strategy, open questions.
- [shared-contracts.md](shared-contracts.md) — SortingAnalyzer layout convention, Pydantic parameter schema, MatcherProtocol plugin interface.
- [designs.md](designs.md) — schema designs for each v2 table (Recording, Sorting, Curation, AnalyzerCuration, Recompute, SessionGroup, UnitMatch).
- [feature-parity.md](feature-parity.md) — explicit v1 parity matrix, including intentional departures.
- Phases (each ships as a separable PR):
  - [phase-0-scaffolding.md](phase-0-scaffolding.md) — foundation work split into Phase 0a (module/CI/code-graph scaffolding) and Phase 0b (fixtures, storage benchmark, v1 baseline capture); no v2 pipeline tables.
  - [phase-1-modern-single-session.md](phase-1-modern-single-session.md) — SortingAnalyzer-based single-session sort end-to-end; new `SpikeSortingOutput.CurationV2` part.
  - [phase-2-analyzer-curation.md](phase-2-analyzer-curation.md) — metrics + auto-merge + burst-pair consolidated into `AnalyzerCuration`, plus recompute verification tables for storage reclamation.
  - [phase-3-session-group-concat.md](phase-3-session-group-concat.md) — `SessionGroup` table + `ConcatenatedRecording` for same-day chronic recordings.
  - [phase-4-unitmatch-cross-session.md](phase-4-unitmatch-cross-session.md) — pluggable matcher backend with UnitMatchPy; polymer validation gate, with Neuropixels/tetrode informational checks.
  - [phase-5-ux-overhaul.md](phase-5-ux-overhaul.md) — `run_v2_pipeline()` API, FigPack curation, notebook rewrite, v1/v2 path-selection docs.
- [appendix.md](appendix.md) — SpikeInterface 0.99→0.104 migration cheat sheet, UnitMatchPy integration notes, MountainSort 5 install + sorter param table.
