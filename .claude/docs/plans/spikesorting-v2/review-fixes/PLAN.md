# spikesorting-v2 Review-Fixes Implementation Plan

**Status:** Phase 1 complete; Phases 2–7 pending.

Two layers of findings, addressed across seven independently-shippable PRs:

- **Original 6-agent `master..HEAD` review (40 findings)** — scientific-correctness, silent-failure, type-design, code-quality, comment-accuracy, test-coverage. Phases 1–3 cover this layer (the C/R/E/T/F/V/Q/D series).
- **Two-pass parity audit (183 confirmed findings, 70 untested branches, 8 stub gaps)** — multi-agent v1↔v2 audit across every module, plus a follow-up audit covering the sorting.py module that timed out in the first pass. Phases 4–7 cover this layer (the A series), with the audit JSON in [.claude/audits/](../../../audits/) and the readable index in [.claude/audits/COMBINED_SYNTHESIS.md](../../../audits/COMBINED_SYNTHESIS.md).

The audit overlaps the original review on several items (per-shank reference electrode → existing T2; `noise_levels` → T3; session_group → V3; etc.); duplicate items stay with their original phase and are not re-listed in the A series. The fix-type taxonomy ([overview.md § Fix-type taxonomy](overview.md#fix-type-taxonomy)) applies uniformly across both layers — some A items are DOCUMENT or VERIFY-FIRST, not BUG; do not blind-fix.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is
   self-contained: inputs to read, tasks with file:line anchors, validation
   slice, fixtures.
2. **Need the full finding ledger, scope, decisions, or the fix-type taxonomy?**
   [overview.md](overview.md).
3. **Need the audit catalog (for an A-series task)?**
   [.claude/audits/COMBINED_SYNTHESIS.md](../../../audits/COMBINED_SYNTHESIS.md)
   has the human-readable index;
   [.claude/audits/spikesorting-v2-parity-audit.json](../../../audits/spikesorting-v2-parity-audit.json)
   and
   [.claude/audits/spikesorting-v2-sorting-parity.json](../../../audits/spikesorting-v2-sorting-parity.json)
   are the per-finding records with verifier reasoning.

## Files

- [overview.md](overview.md) — scope, the finding ledger (40 original + A-series additions) mapped to phases, the settled design decisions, and the fix-type taxonomy.
- Phases (each ships as a separable PR; suggested execution order is sequential, but Phases 4–7 are independent of each other and can be parallelized after Phase 1 lands):
  - [phase-1-correctness-and-error-handling.md](phase-1-correctness-and-error-handling.md) — **[DONE]** silent-failure & correctness fixes (C1–C7) + runtime-contract fixes + MEDIUM error-handling.
  - [phase-2-type-and-schema-design.md](phase-2-type-and-schema-design.md) — `ArtifactSource` part table, `SortGroupV2` reference-mode split, `noise_levels`/`CurationLabel` reconciliation, and all Pydantic-schema tightening (T1–T10, F1–F2).
  - [phase-3-tests-and-docs.md](phase-3-tests-and-docs.md) — coverage additions (V1–V5), tautological-test cleanup (Q1–Q3), and all comment/doc-accuracy fixes (D1–D8).
  - [phase-4-audit-correctness-and-parity.md](phase-4-audit-correctness-and-parity.md) — audit-derived behavioral fixes the original review missed (A1–A16): `_apply_artifact_mask` empty-`valid_times` raise, Franklab preset back-compat aliases, MS4 schema sentinel acceptance, install-gated `SorterParameters._DEFAULT_CONTENTS`, `numpy.Inf` global mutation, `sorter_temp_dir.cleanup()` exception masking, regression test pinning the existing zero-unit `get_analyzer` guard (audit's "phantom analyzer_folder" premise was disproved on re-read; the column is `varchar(255)` NOT NULL and the guard already short-circuits before path lookup), motion-correction defaults wiring.
  - [phase-5-production-scale-readiness.md](phase-5-production-scale-readiness.md) — `ChunkRecordingExecutor` restoration for artifact detection (the audit's largest item), AnalysisNwbfile rebuild + hash-mismatch tests, `channel_name` real-NWB fixture, tetrode-geometry gate negatives, SI version pin + KS4 snapshot test, analyzer-folder disk-leak audit job (A17–A23).
  - [phase-6-coverage-backfill.md](phase-6-coverage-backfill.md) — test-only PR covering the audit's 70 untested branches (excluding existing V1–V5 and excluding stub-`NotImplementedError` shapes per the project decision) plus parity pins for intentional-justified items that lack regression tests (A24–A31).
  - [phase-7-migration-docs-and-stub-roadmap.md](phase-7-migration-docs-and-stub-roadmap.md) — mostly-docs PR (+ two small additions): CHANGELOG enumerating v1→v2 breaking changes; `ImportError`-raising `__getattr__` shims + roadmap docstrings on the four stub modules (`metric_curation`, `figpack_curation`, `unit_matching`, `matcher_protocol`); opt-in `SorterParameters.insert_default_legacy_si_sorters()` classmethod for v1 workflow porting; migration guide section; stale-`v1/*` ref sweep (A32–A36). The two NEW symbols (stub `__getattr__`, classmethod) do not modify any existing code path.

## Awareness of the parent plan

The parent [.claude/docs/plans/spikesorting-v2/](../PLAN.md) is the epic this child plan extends. **Parent Phase 0a/0b/0c and Phase 1 are done** (per the project's `.remember/today-*.md` log). Parent Phases 2 (AnalyzerCuration), 3 (SessionGroup/ConcatenatedRecording), 4 (UnitMatch), and 5 (UX/FigPack) are pending.

- **Parent Phase 1b** is the runtime-regressions layer that includes R1–R18 and B1–B7. Several review-fixes Phase 4–7 tasks coordinate with it; the sequencing notes per task are explicit.
- **Parent Phase 3** implements `ConcatenatedRecording.make` / `SessionGroup.create_group` / `is_multi_day` / `MotionCorrectionParameters` consumers. When it lands, Phase 6 A31's invariant tests stay valid; Phase 4 A11 (`Sorting.make` key_source antijoin) flips from "skip silently" to "process normally." Both are pinned with explicit `# TODO(parent-Phase-3)` markers in the test code.
- **Parent Phase 4** implements `unit_matching` and `matcher_protocol`. When it lands, Phase 7 A33's `__getattr__` stub for those modules is deleted and replaced by real exports.
