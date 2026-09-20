# PR #1609 Review Fixes — Implementation Plan

**Status:** Phases 1 and 2 implemented on `spikesorting-v2` (phase 1: 3b1f6974..59829a9f; phase 2: 59829a9f..84f99145 incl. owner-review fixes; 2026-09-18/19), PRs not yet opened; open items: moseq extras cannot resolve (keypoint-moseq pins panel==0.14.4 vs base panel>=1.4, owner decision); dropped-frame run splitting on position clocks flagged for a real-session spot check; merge-level v2 accessors raise when the curated recording is unavailable. Phases 3a-6 not started.

Fixes the merge-blocking findings from the 2026-09-18 full-branch review of PR #1609 (spikesorting-v2 vs master, head cc9a7953): five silent-science defects in the v2 signal path, cross-session matching, and observed-time layer; seven compatibility/dependency failures that break non-spike-sorting users (DLC, MoSeq) and v0/v1 read paths; then the CI lanes that let those regressions through, and the stale user docs. Each phase ships as its own PR against the `spikesorting-v2` branch. v2 is pre-production: schema changes are allowed (owner, 2026-09-18), and there is no backwards-compatibility or deprecation obligation for v2 tables, caches, or receipts. v0/v1 behavior must be preserved.

**Scope addition (owner, 2026-09-19; planned, not implemented).** Add optional, validated motion correction independently of concatenation (phase 3c), and UnitMatch between independently sorted daily concatenations (phase 4c). These are explicit feature additions beyond the original review fixes. Their design, scientific validation, and documentation are in this plan; they ship as separate PRs. The original merge-gate agreement below remains recorded as-is: the request to add these features does not by itself set their before/after-#1609 merge scheduling.

**Merge scope (owner, 2026-09-18).** Required before PR #1609 merges: everything that makes spikesorting-v2 correct, preserves existing functionality this PR touched, and demonstrates both with tests — phases 1, 2, 3a, 3b, 4a, 4b, 5, and the *required* subset of phase 6 (README receipt key, CHANGELOG factual corrections and omissions, stale table names in user-facing text, `TODO.md`, scaffolding tokens in shipped source). Optional cleanup, tracked but not blocking: the phase-6 items marked optional (test renames, regex widening, CHANGELOG restructuring, docstring completeness, column-comment wording) and the deferred `mountainsort4` environment reorganization noted in phase 1.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, designs it depends on, tasks, validation slice, and fixtures.
2. **Need a per-component design?** [designs.md](designs.md).
3. **Need broader scope / risks / dependency policy / open questions?** [overview.md](overview.md).
4. **Need the original review evidence (measurements, reproduction commands, the full Important/Suggestion backlog)?** [appendix-review-findings.md](appendix-review-findings.md).
5. **Working on independent motion correction or matching daily sorts?** Read [designs-motion-and-matching.md](designs-motion-and-matching.md) plus phase 3c or 4c. This includes the verified upstream distinctions and supersedes the older remediation plan's claim that concat-backed matching already works.

## Dependencies for the added features

- Phases 3a + 3b → **3c**: correction consumes normalized geometry, original clocks, masks, and persisted valid/continuity spans.
- Phases 3c + 4a → **4c**: matching daily sorts uses the shared effective-source/lineage contract and per-unit temporal halves.
- Phase 4a remains independently deliverable; it does not enable motion correction or concat-backed matching.
- Phases 3c/4c own their tests, benchmarks and feature docs. Phase 5 adds their CI execution when they land; phase 6 audits the resulting docs. Original phase-5/6 work need not wait for the feature additions.

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, dependency policy (the numpy-pin decision), risks, open questions, effort.
- [designs.md](designs.md) — filter-before-restriction, valid-sample statistics (explicit valid frame ranges, per-span sampling), geometry normalization before probe construction, per-unit UnitMatch halves, per-metric missingness eligibility, MUA detection per contiguous observed run.
- [designs-motion-and-matching.md](designs-motion-and-matching.md) — independent estimation/application, source lineage versus effective traces, scientific recipe validation, and matching inputs spanning multiple original recordings.
- Phases (each ships as a separable PR):
  - [phase-1-dependencies-ci.md](phase-1-dependencies-ci.md) — numpy pin relaxation, scipy declaration, DLC/MoSeq extras and env files, black.
  - [phase-2-shared-modules.md](phase-2-shared-modules.md) — v0/v1 waveform reads, annotation-migration boundary, legacy sorter defaults, merge fetch_nwb partial drop, MUA NaN guard.
  - [phase-3a-recording-stage.md](phase-3a-recording-stage.md) — filter before time restriction; normalize 3D geometry to a distinct 2D plane before any probe is built and persist it; every contact unique.
  - [phase-3b-sorting-stage.md](phase-3b-sorting-stage.md) — valid frame ranges threaded and persisted; whitening/noise/nn-noise from within-span samples; observed-interval endpoints; interval-set guards; zero-spike units.
  - [phase-3c-motion-correction.md](phase-3c-motion-correction.md) — **added feature**: off/estimate/apply modes, persisted estimates and corrected recordings, shared single/concat implementation, and probe-specific validation.
  - [phase-4a-unitmatch-halves.md](phase-4a-unitmatch-halves.md) — per-unit temporal cross-validation halves.
  - [phase-4b-metric-missingness.md](phase-4b-metric-missingness.md) — computation failure vs expected missingness in auto-curation; SI metric warnings; double-precision params columns.
  - [phase-4c-concat-unitmatch.md](phase-4c-concat-unitmatch.md) — **added feature**: one matching input per independent single/daily-concat sort, frozen constituent provenance, and tracking back to original members.
  - [phase-5-validation-lanes.md](phase-5-validation-lanes.md) — real v0/v1 read-path fixtures, legacy waveform-features test, acceptance probes as opt-in tests, matcher fixture wiring.
  - [phase-6-docs.md](phase-6-docs.md) — README receipt key, CHANGELOG consolidation, stale table names, scaffolding tokens, TODO.md.
- [appendix-review-findings.md](appendix-review-findings.md) — the verified review report this plan is derived from.
