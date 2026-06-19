# Spike Sorting v2 — UX follow-ups Implementation Plan

**Status:** Refreshed 2026-06-19 against the current branch.

Branch audit summary: the public run-summary API rename has mostly already
landed (`run_v2_pipeline` uses a `run_summary` local and
`PipelineStageError.partial_run_summary` exists), but residual `manifest`
language remains in tests/comments and possibly some non-primary docs. The
worth-doing work is still Phase 2a (scientific catalog/provenance correction),
Phase 3 (interval discovery + docs polish), and Phase 4 (session-level runner).
Phase 2b remains gated and should not be implemented without the analyzer-curation
phase and scientific sign-off.

Ordered, separately-shippable improvements to the `spyglass.spikesorting.v2`
user-facing surface, identified in a UX audit of the orchestration layer:
(1) finish the residual "manifest" → "run summary" cleanup left after the main
API rename; (2a) correct v2's shipped "franklab" parameter blobs to the real
DB-attested production recipes (region-based preproc — hippocampus 600 Hz /
cortex 300 Hz, the MS4 `franklab_probe_*` family by adjacency_radius × rate,
100/50 µV artifact), make them dated/fingerprinted/inspectable, and set
MountainSort4 as the probe default; (2b) **gated** recipes that can't ship yet —
Set A's downstream curation stages (need the analyzer-curation phase) and
unattested recipes (need scientific sign-off); (3) close
the interval-discovery and documentation-polish gaps on top of that canonical
catalog; (4) add `run_v2_pipeline_session()`, a session-level batch runner that
sorts every sort group in a session instead of forcing the user to loop one
`sort_group_id` at a time. All changes are additive or pre-release renames; the
returned-dict keys and `merge_id` consumers are unaffected (Phase 2a does change
the derived ID *values* — acceptable pre-release; see overview).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, tasks, validation slice, and fixtures.
2. **Need broader scope / risks / the run-summary key contract / the selection-identity caveat?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, the run-summary dict contract, the name-based selection-identity caveat, risks, open questions.
- [appendix.md](appendix.md) — SpikeInterface 0.104.3 deep reference (sorter/preproc/SortingAnalyzer/metrics/motion internals, with SI file:line and divergences). Load when implementing against SI — it resolves `detect_threshold` semantics, the `nn_advanced` rename, the custom `isi_violation` fraction, the analyzer recompute cascade, and the motion/torch dependencies.
- Phases (each ships as a separable PR, in order):
  - [phase-1-run-summary-rename.md](phase-1-run-summary-rename.md) — mostly landed; finish residual "manifest" → "run summary" cleanup in tests/comments/docs. No pipeline behavior change.
  - [phase-2a-parameter-names-and-fingerprints.md](phase-2a-parameter-names-and-fingerprints.md) — correct v2's blobs to the DB-attested recipes (region preproc 600/300 Hz, MS4 `franklab_probe_*` family by radius × rate, 100/50 µV artifact), date/fingerprint them, MS4 probe default, duplicate-content guard, `describe_parameter_rows()`. **Lands before Phase 3/4** so the catalog is settled; corrections change derived selection IDs (see overview).
  - [phase-2b-deferred-and-unattested.md](phase-2b-deferred-and-unattested.md) — **GATED.** Recipes that can't ship yet: Set A's downstream curation stages (wait on the analyzer-curation phase) and unattested recipes — Neuropixels tuning, tetrode-20 kHz, MS5 probe (wait on scientific sign-off). Depends on 2a; not a UX follow-up.
  - [phase-3-discovery-and-polish.md](phase-3-discovery-and-polish.md) — `describe_intervals()`, notebook placeholder fix, zero-unit docstring cross-refs, gated-stub note, and docs/notebook surfacing of the canonical catalog. The helper itself can land independently; docs/notebook preset names should follow 2a.
  - [phase-4-session-runner.md](phase-4-session-runner.md) — `run_v2_pipeline_session()` multi-sort-group batch runner with per-group outcome reporting. The code can land after the Phase 1 cleanup; docs/examples should follow 2a's dated preset names.
