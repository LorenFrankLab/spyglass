# Spikesorting-v2 Pipeline Fixes Implementation Plan

**Status:** Not started.

Fixes the spikesorting-v2-pipeline-relevant defects found by the multi-agent review (`../spikesorting-v2/REVIEW-REPORT.md`): two silent-wrong-result bugs at the v2→downstream boundary, the sort/param validation + reproducibility gaps, export-safety for v2 outputs, hardening of under-tested-but-correct behaviors, and the on-branch CI breakages that gate the branch from merging. Each phase ships as an independent PR. Phase numbers are stable identities; the **recommended execution sequence is 5 → 2 → 1 → 3 → 6 → 4** (CI-unblock → risk → dependency) — see [overview.md](overview.md) for the rationale and dependency policy.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained (inputs to read, tasks with inline code + file:line, validation slice, fixtures, review checklist).
2. **Need broader scope / integration points / risks / rollout?** [overview.md](overview.md).
3. **Need the underlying evidence for a finding?** `../spikesorting-v2/REVIEW-REPORT.md` (groups A–F + named patterns).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, rollout, effort.
- Phases (each ships as a separable PR):
  - [phase-1-consumer-boundary.md](phase-1-consumer-boundary.md) — **A1** `artifact_id` silently dropped in merge-id resolution + **A2** multi-source `fetch_nwb(return_merge_ids=True)` misalignment. Highest risk; do first.
  - [phase-2-sorter-params.md](phase-2-sorter-params.md) — MS5 `filter`/`whiten` toggles, bulk-`insert` validation (P2), `noise_levels` length guard (P3), analyzer `random_spikes` seed, enum `from exc`.
  - [phase-3-export-safety.md](phase-3-export-safety.md) — D2/D3/D4: investigation spike → route-or-document accessors → export-completeness + zero-unit export tests.
  - [phase-4-test-hardening.md](phase-4-test-hardening.md) — exact-value tests for verified behaviors (interval frames, gain, SharedArtifactGroup union, lazy-vs-applied merge, idempotency row-counts, consumer alignment) + stale-doc reconciliation. No prod logic change (except one test-integrity line).
  - [phase-5-ci-infra.md](phase-5-ci-infra.md) — legacy-SI import shim (D5), resolvable legacy env (D6), DLC skip-condition fix (D8). Restores branch CI.
  - [phase-6-clusterless-waveform-features.md](phase-6-clusterless-waveform-features.md) — make `UnitWaveformFeatures` (clusterless decoding input) work for v2 sorts under SI 0.104 via a `SortingAnalyzer` path. Real regression fix; depends on the v2 analyzer being stable (Phase 2).
