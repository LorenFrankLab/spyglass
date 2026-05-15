# Spike Sorting v2 Implementation Plan

**Status:** Not started.

A next-generation spike sorting pipeline for Spyglass that (1) targets SpikeInterface ≥0.104 with the SortingAnalyzer API, keeps the v1 production sorters (MountainSort 4 and `clusterless_thresholder`), adds MS5 and KS4, and preserves a custom-row path for other SpikeInterface sorters, (2) supports cross-session unit tracking via a pluggable matcher backend (UnitMatchPy first, DeepUnitMatch via the same plugin slot later), (3) handles chronic recordings — same-day concatenation as the default path, with multi-day opt-in (sort-then-match via Phase 4 UnitMatch is the recommended cross-day workflow) — (4) makes unit → brain region tracing first-class via a `Sorting.Unit` part table populated at sort time (fixes the v1 multi-region under-reporting bug), (5) reduces user-facing table touchpoints from ~7 to ~2 via a `run_v2_pipeline()` convenience API with Pydantic-validated parameters and FigPack as the curation UI, and (6) plugs into the existing `SpikeSortingOutput` merge table so downstream consumers (decoding, ripple, MUA, `SortedSpikesGroup`) keep working unchanged. **Zero schema migration**: every table is final from the phase that introduces it. v0 and v1 stay in-tree indefinitely.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase/checkpoint?** Open the matching phase file and start with its `Executor Checklist`. Treat the checklist as the implementation contract for that slice of work; use the longer Tasks / Validation sections as reference.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need v1 feature parity boundaries?** [feature-parity.md](feature-parity.md).
5. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
6. **Need upstream-repo line refs / on-disk format details?** [appendix.md](appendix.md).

For lab users deciding whether to use v2, start with the Phase 5 user-facing
surface ([phase-5-ux-overhaul.md](phase-5-ux-overhaul.md)), then the
`run_v2_pipeline()` design in [designs.md](designs.md#run_v2_pipeline-orchestrator),
then the notebook/docs produced by that phase. The phase files are implementation
plans, not the final user guide.

## Execution safety

All v2 implementation work uses the [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) contract:

- Develop and validate in an isolated `uv` virtualenv. Do not install SpikeInterface 0.104, UnitMatchPy, MEArec, or FigPack into a shared/base environment.
- Run DataJoint integration tests against an isolated database by default. Prefer the existing pytest Docker MySQL path (`tests/conftest.py` starts a Docker-backed test server and uses `database.prefix = "pytests"`); the repo's `docker-compose.yml` is the manual local fallback.
- Treat production-connected real-data checks as optional smoke tests, not the main validation target. They require an explicit env-var gate, must write only to test schemas/temp analysis directories, and must not delete or mutate production rows.

All implementation artifacts also use the [Code Artifact Naming](shared-contracts.md#code-artifact-naming) contract: code, tests, notebooks, docstrings, comments, and user-facing docs must describe behavior/components, not the plan phase that introduced them.

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, risks, rollout strategy, open questions.
- [shared-contracts.md](shared-contracts.md) — SortingAnalyzer layout convention, Pydantic parameter schema, MatcherProtocol plugin interface.
- [designs.md](designs.md) — schema designs for each v2 table (Recording, Sorting, Curation, AnalyzerCuration, Recompute, SessionGroup, UnitMatch, FigPack curation, and pipeline helpers).
- [feature-parity.md](feature-parity.md) — explicit v1 parity matrix, including intentional departures.
- Execution phases/checkpoints (PR boundaries are chosen by the project owner; phases are not required to ship as standalone PRs):
  - [phase-0-scaffolding.md](phase-0-scaffolding.md) — foundation work split into Phase 0a (module/CI/code-graph scaffolding) and Phase 0b (fixtures and v1 baseline capture); no v2 pipeline tables.
  - [phase-0c-si-0104-prerequisite.md](phase-0c-si-0104-prerequisite.md) — required prerequisite checkpoint that bumps the SI runtime for v2 and makes the legacy v0/v1 runtime boundary explicit before Phase 1 can land.
  - [phase-1-modern-single-session.md](phase-1-modern-single-session.md) — SortingAnalyzer-based single-session sort end-to-end; new `SpikeSortingOutput.CurationV2` part.
  - [phase-2-analyzer-curation.md](phase-2-analyzer-curation.md) — metrics + auto-merge + burst-pair consolidated into `AnalyzerCuration`, plus recompute verification tables for storage reclamation.
  - [phase-3-session-group-concat.md](phase-3-session-group-concat.md) — `SessionGroup` table + `ConcatenatedRecording` for same-day chronic recordings.
  - [phase-4-unitmatch-cross-session.md](phase-4-unitmatch-cross-session.md) — pluggable matcher backend with UnitMatchPy; polymer validation gate.
  - [phase-5-ux-overhaul.md](phase-5-ux-overhaul.md) — `run_v2_pipeline()` sorting API, `run_v2_unit_match()` helper, FigPack curation, notebook rewrite, v1/v2 path-selection docs. Split into Phase 5a (FigPack feasibility spike — verifies the FigPack spike-sorting API and edited-curation round trip, replaces the `PHASE5A_CONTRACT_STUB` markers) and Phase 5b (orchestrator extension, FigPack tables, notebooks, docs). Phase 5b is gated on Phase 5a, mirroring the Phase 4a/4b split.
- [appendix.md](appendix.md) — SpikeInterface 0.99→0.104 migration cheat sheet, UnitMatchPy integration notes, MountainSort 5 install + sorter param table.

## Dependency DAG

```text
Phase 0a scaffolding/code graph
  -> Phase 0b fixtures/baseline
  -> Phase 1 single-session v2 MVP

Phase 0c SI 0.104 compatibility boundary + dependency bump
  -> Phase 1 single-session v2 MVP

Phase 1 single-session v2 MVP
  -> Phase 2 AnalyzerCuration + recompute verification
  -> Phase 3 SessionGroup + ConcatenatedRecording

Phase 3 SessionGroup + ConcatenatedRecording
  -> Phase 4a UnitMatchPy technical spike
      -> Update appendix + shared-contracts + designs from 4a findings
          -> Re-run code_graph on the revised draft schema
              -> Phase 4b UnitMatch cross-session tracking implementation
                  -> Phase 5a FigPack feasibility spike
                      -> Replace PHASE5A_CONTRACT_STUB markers from 5a findings
                          -> Phase 5b UX/FigPack tables/notebooks/docs
```

Execution happens on the long-lived `spikesorting-v2` integration branch with checkpoint commits. Phase 1 is the first runtime v2 pipeline checkpoint and requires Phase 0a, Phase 0b, and Phase 0c. Phase 0c is a hard gate because Phase 1 imports and runs SpikeInterface 0.104 APIs while legacy v0/v1 active-runtime workflows must either be guarded with clear legacy-environment messages or explicitly proven compatible. Checkpoint commits may be grouped into larger review PRs if the gating order and validation evidence remain clear.
