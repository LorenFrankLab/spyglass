# Curation UX Overhaul Implementation Plan

**Status:** Not started.

Makes hands-on spike-sorting curation in the v2 pipeline something a lab member can do without holding the internal data model in their head. The **primary user journey is browser-first**, matching the lab's real v1 FigURL workflow: start one review from a curation, inspect/label/merge in FigPack, preview the imported change set, commit it, and continue directly into a post-merge verification review when needed. The scripted evaluation/plot facade remains a supported secondary path for automation, debugging, and expert use. Today a curator must build DataJoint keys by hand, choose between several near-identical "apply the evaluation" methods, remember to re-key downstream off the curated child, and cannot *look at* a merged unit's waveform or correlogram at all. This plan closes that gap at the architectural root, adds a persisted review profile and first-class review session, brings FigPack to identity-verified round-trip parity, and restores v1's custom-metric flexibility through typed annotations without its NWB failure mode.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained: upstream files to read, contracts/designs it depends on, tasks, validation slice, fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — curation/evaluation/review value objects, resolver signature, error type, and figure-identity schema shared across phases.
- [designs.md](designs.md) — resolver + cache internals, scripted `merge_and_evaluate`, browser review orchestration, strict-metric handling, figure identity, and annotation model.
- Phases (each ships as a separable PR, in this order):
  - [phase-0-schema-foundation.md](phase-0-schema-foundation.md) — immutable `curation_uuid` identity, persisted rule `missing_policy`, and immutable `CurationReviewProfile` rows. Small, additive schema the later phases depend on.
  - [phase-1-curation-analyzer.md](phase-1-curation-analyzer.md) — one curation-scoped analyzer resolver behind evaluation, plotting, and FigPack; closes the merged-unit visualization gap.
  - [phase-2-curation-facade.md](phase-2-curation-facade.md) — immutable value objects + scripted facade foundation, `merge_and_evaluate`, strict metric errors, and schema-free lifecycle helpers.
  - [phase-3-figpack-parity.md](phase-3-figpack-parity.md) — the primary browser-first `start_review → preview_import → commit → continue_review` workflow, backed by the resolver and identity-verified local + hosted-or-controlled-bundle FigPack round trips (hosted persist is spike-gated; the remote read-back falls back to external controlled storage, mirroring v1's GitHub/kachery path).
  - [phase-4-unit-annotations.md](phase-4-unit-annotations.md) — typed, content-addressed unit-annotation tables + curation provenance columns (schema).
