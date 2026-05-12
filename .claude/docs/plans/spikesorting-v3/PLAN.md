# Spike Sorting v3 Implementation Plan

**Status:** Not started.

A next-generation spike sorting pipeline for Spyglass that (1) targets SpikeInterface ≥0.104 with the SortingAnalyzer API, (2) supports cross-session unit tracking via a pluggable matcher backend (UnitMatch first, DeepUnitMatch later), (3) handles chronic recordings through session-group concatenation and lazy streaming, (4) reduces user-facing table touchpoints from ~7 to ~2 via a `run_v3_pipeline()` convenience API and Pydantic-validated parameters, and (5) plugs into the existing `SpikeSortingOutput` merge table so downstream consumers (decoding, ripple, MUA, `SortedSpikesGroup`) keep working unchanged. v0 and v1 stay in-tree indefinitely.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
5. **Need upstream-repo line refs / on-disk format details?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, risks, rollout strategy, open questions.
- [shared-contracts.md](shared-contracts.md) — SortingAnalyzer layout convention, Pydantic parameter schema, MatcherProtocol plugin interface.
- [designs.md](designs.md) — schema designs for each v3 table (Recording, Sorting, Curation, MetricCuration, SessionGroup, UnitMatch).
- Phases (each ships as a separable PR):
  - [phase-0-scaffolding.md](phase-0-scaffolding.md) — module layout, SI ≥0.104 dep migration, baseline-capture fixtures for v1 parity.
  - [phase-1-modern-single-session.md](phase-1-modern-single-session.md) — SortingAnalyzer-based single-session sort end-to-end; new `SpikeSortingOutput.CurationV3` part.
  - [phase-2-analyzer-curation.md](phase-2-analyzer-curation.md) — metrics + auto-merge + burst-pair consolidated into `AnalyzerCuration`.
  - [phase-3-session-group-concat.md](phase-3-session-group-concat.md) — `SessionGroup` table + `ConcatenatedRecording` for same-day chronic recordings.
  - [phase-4-unitmatch-cross-session.md](phase-4-unitmatch-cross-session.md) — pluggable matcher backend with UnitMatchPy; tetrode validation gate.
  - [phase-5-ux-overhaul.md](phase-5-ux-overhaul.md) — `run_v3_pipeline()` API, FigPack curation, notebook rewrite, v1 sunset roadmap.
- [appendix.md](appendix.md) — SpikeInterface 0.99→0.104 migration cheat sheet, UnitMatchPy integration notes, MountainSort 5 install + sorter param table.
