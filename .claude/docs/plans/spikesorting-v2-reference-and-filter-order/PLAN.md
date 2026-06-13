# Spike-Sorting v2 Reference Defaults & Filter-Before-Reference Plan

**Status:** Not started.

Make v2's default referencing behavior inherit each electrode's configured
reference (the way v1 does) instead of defaulting to no reference, and change
preprocessing to the signal-processing-preferred order — bandpass filter first,
then reference subtraction. The pipeline is pre-production, so this ships as a
clean behavior change with dev rows regenerated rather than migrated.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is
   self-contained: upstream files to read, tasks, validation slice, fixtures.
2. **Need broader scope / risks / rollout / open questions?**
   [overview.md](overview.md).

There is no `shared-contracts.md` / `designs.md` / `appendix.md` — the two
phases touch mostly different files, and the one shared idea (the
`reference_mode` sentinel→column mapping) is small enough to live in
phase-1 and is *consumed* (not re-derived) by the existing runtime.

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, risks,
  rollout, open questions.
- Phases (each ships as a separable PR):
  - [phase-1-reference-defaults.md](phase-1-reference-defaults.md) —
    auto-inherit references in the `SortGroupV2` grouping helpers; sentinel
    mapping; mixed / in-group reference fail-early; v1-style `references` dict;
    `omit_ref_electrode_group` rewrite. No schema change.
  - [phase-2-filter-before-reference.md](phase-2-filter-before-reference.md) —
    reorder preprocessing to bandpass→reference→drop-ref; fix
    `filtering_description`; correct the docstrings that assert v1
    reference-first parity; invalidate old dev selection/cache state; regenerate
    dev rows + v1↔v2 numeric baseline.
