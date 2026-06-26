# Spike Sorting V2 Review Remediation — Implementation Plan

**Status:** Not started.

Remediation of the verified findings from the 26-document Spike Sorting V2 review
triage ([../../reviews/spikesorting-v2/TRIAGE.md](../../reviews/spikesorting-v2/TRIAGE.md)),
sequenced as independently-shippable PRs that land **before** the Phase 5 UX
overhaul. This plan fixes silent-wrong-science bugs, bakes reproducibility and
identity guarantees into the schema while the pipeline is still pre-production
(no users, schema freeze lifted), hardens operational/security surfaces, and
decomposes the monolithic `CurationV2` table class so Phase 5 builds on clean
seams. The owner-agreed disposition behind every phase is recorded in the triage
"Decisions" section.

This plan is **separate from** the main build plan in
[../spikesorting-v2/](../spikesorting-v2/); its phase numbers are local to this
directory. The "Phase 5 UX overhaul" referenced throughout is
[../spikesorting-v2/phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md),
which a companion edit (see overview "Phase 5 adjustments") updates in place.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / sequencing?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, sequencing, integration points, risks, the Phase-5 adjustments.
- [shared-contracts.md](shared-contracts.md) — the provenance field set (phases 3a/3b) and the master-row immutability invariant (phase 2).
- Phases (each ships as a separable PR):
  - [phase-0-curation-decomposition.md](phase-0-curation-decomposition.md) — behavior-preserving extraction of `CurationV2.resolve_restriction`'s pure routing half + the `summarize_curation` formatter into DB-free service modules (R38). The other accessors (`get_recording`/`get_sorting`/`get_merged_sorting`/`get_sort_metadata`) stay on the table. Largely independent of phases 1–4 (phase-4a's REL-4 task must land **after** it); must merge before the Phase 5 UX work.
  - [phase-1-correctness.md](phase-1-correctness.md) — **MUST, ship first.** Silent-wrong-science fixes: boundary-sample loss, DC-offset reattach, SNR polarity, AnalyzerCuration namespace, sorter-output-freed, preview-curation guard (R29, R27, R4, R3).
  - [phase-2-identity-integrity.md](phase-2-identity-integrity.md) — master-row immutability, runtime-alias versioning, TrackedUnit frozen universe, schema-init upgrade safety (R28, R6, R7, R8).
  - [phase-3a-row-provenance.md](phase-3a-row-provenance.md) — effective seed, producing versions, UnitMatch bundle params, matcher backend version into computed rows (R1, R33-matcher).
  - [phase-3b-nwb-provenance.md](phase-3b-nwb-provenance.md) — source/sorter/params/lineage metadata into the NWB writers; params-not-arrays for motion/waveforms (R10).
  - [phase-4a-cache-concurrency-compat.md](phase-4a-cache-concurrency-compat.md) — analyzer-cache lock + atomic publish, concat compatibility, bypass revalidation, temp routing (R5, R30, R31, R12) + Round-3 analyzer/temp tasks (ALSC-3/4, CLUST-2).
  - [phase-4b-execution-deps-security.md](phase-4b-execution-deps-security.md) — dispatch mismatches, dependency pins, permissions/path/trust-docs, db-guard, merge probe, footguns, Export.File leak, conda guard (R32, R16, R34, R14, R15, R18, R22, R26) + UCI-6 downstream disambiguators.
  - [phase-4c-concat-lifecycle-integrity.md](phase-4c-concat-lifecycle-integrity.md) — **new (Round 3):** concat verify-on-read + rebuild + recompute parity, frozen member-set identity, split-back spike conservation (R40, R39-concat, CONCS-3).
  - [phase-6-scientific-validation-ci-gates.md](phase-6-scientific-validation-ci-gates.md) — **new (Round 3):** publish gating fixtures + make ship criteria required in CI, drift/auto-merge/ground-truth science gates, fixture-realism bands (R41 / SVFR-1..6) + Round-3 science test gaps.

Round-3 reviews also added tasks to phases 1, 2, and 3a (see each phase's "Additional tasks (Round-3 reviews)" section) and doc items to the Phase-5 edit.
