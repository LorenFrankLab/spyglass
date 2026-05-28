# spikesorting-v2 Review-Fixes Implementation Plan

**Status:** Not started.

Addresses every finding from the 6-agent `master..HEAD` review of the
`spikesorting-v2` branch (scientific-correctness, silent-failure, type-design,
code-quality, comment-accuracy, test-coverage). The findings are fixed across
three independently-shippable PRs: correctness/error-handling, type/schema
design, and tests/docs. No finding is deferred — but several are resolved by
*verifying the existing behavior is correct and documenting it* rather than by
changing code, and that distinction is called out per-finding so the executor
does not "fix" an intentional design into a regression.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is
   self-contained: inputs to read, tasks with file:line anchors, validation
   slice, fixtures.
2. **Need the full finding ledger, scope, decisions, or the fix-type taxonomy?**
   [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, the 40-finding ledger mapped to phases, the three settled design decisions, and the fix-type taxonomy (confirmed-bug / verify-first / fix-by-documenting / restructure).
- Phases (each ships as a separable PR):
  - [phase-1-correctness-and-error-handling.md](phase-1-correctness-and-error-handling.md) — silent-failure & correctness fixes (C1–C7) + runtime-contract fixes + MEDIUM error-handling. The merge-blocking behavioral PR.
  - [phase-2-type-and-schema-design.md](phase-2-type-and-schema-design.md) — `ArtifactSource` part table, `SortGroupV2` reference-mode split, `noise_levels`/`CurationLabel` reconciliation, and all Pydantic-schema tightening.
  - [phase-3-tests-and-docs.md](phase-3-tests-and-docs.md) — coverage additions, tautological-test cleanup, and all comment/doc-accuracy fixes.
