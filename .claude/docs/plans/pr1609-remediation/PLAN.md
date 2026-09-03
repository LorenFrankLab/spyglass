# PR #1609 Remediation Plan

**Status:** Implementation complete on `spikesorting-v2`; manual real-data
validation pending.

Fixes the merge-blocking regressions and documentation contradictions found in the 2026-09-01 architecture/UX review of PR #1609 (Spike Sorting v2, branch `spikesorting-v2`), and then makes concatenated (same-day chronic) sorts usable downstream by splitting a curated concat sort back into one decodable, wall-clock-aligned row per member session. By owner decision, both phases land as reviewable commits on the existing preproduction PR, including the additive Phase 2 tables. Automated validation completed on 2026-09-03; the lab-data compatibility and time-aligned concat decoding checks remain manual pre-merge gates.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, designs it depends on, tasks, validation slice, and fixtures.
2. **Need a per-component design (code)?** [designs.md](designs.md).
3. **Need broader scope / risks / decisions already made?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, decisions already taken (raise-vs-warn, no deprecation windows, single PR), risks, open questions.
- [designs.md](designs.md) — code for the SpikeInterface compat shim, ElectricalSeries selection, multi-source raise, Export.File retention, UnitAnnotation audit + migration, concat merge gate, UnitMatch baseline window, and the concat member-curation table.
- Phases (implemented as reviewable commits on this PR):
  - [phase-1-pr1609-fixes.md](phase-1-pr1609-fixes.md) — commits onto `spikesorting-v2` / PR #1609: master-user regressions, v2 correctness fixes, doc/param reconciliation, PR description refresh.
  - [phase-2-concat-member-curations.md](phase-2-concat-member-curations.md) — per-member decodable rows for concat sorts (schema-additive).
