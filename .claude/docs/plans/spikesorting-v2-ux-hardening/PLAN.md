# Spike Sorting v2 — First-Hour UX Hardening Plan

**Status:** Not started.

Makes the *already-shipped* v2 single-session pipeline pleasant for a scientist's first hour: a discoverable preset catalog, a fast fail-early preflight check, an observable run manifest with per-stage timing/status and a stage-aware exception, friendly named curation wrappers, a less-surprising test harness, and a runnable canonical notebook gated by an end-to-end UX smoke test. It adds usability surface on top of the current MVP (post Phase 1 of the master plan); it does **not** add scientific features.

> **This is a separate plan from `.claude/docs/plans/spikesorting-v2/`.** That directory holds the master pipeline roadmap (its "Phase 1–5" mean entirely different things — Phase 2 metrics, Phase 3 concat, Phase 4 UnitMatch, Phase 5 FigPack UI). The phases *here* are independent of those and land on the current MVP. Where the two overlap (the user-facing notebook and the preset surface), this plan owns the minimal version now and the master plan's [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) extends it later (FigPack cells, richer presets, concat/cross-session). See [overview.md § Relationship to the master roadmap](overview.md#relationship-to-the-master-roadmap).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained: upstream files to read, contracts it depends on, tasks, validation slice, fixtures, review gate.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md) — the manifest schema, preflight-report schema, `PipelineStageError`, and stage-status vocabulary.
3. **Need broader scope / risks / roadmap relationship?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points (file:line), roadmap relationship, risks, metrics.
- [shared-contracts.md](shared-contracts.md) — manifest schema, preflight report schema, `PipelineStageError`, stage-status values.
- Phases (each ships as a separable PR):
  - [phase-1-preset-discovery.md](phase-1-preset-discovery.md) — `describe_presets()` accessor beside `list_presets()`.
  - [phase-2-preflight.md](phase-2-preflight.md) — `preflight_v2_pipeline()` + `run_v2_pipeline(..., preflight=True)`.
  - [phase-3-observability.md](phase-3-observability.md) — per-stage status/timing/warnings in the manifest + `PipelineStageError` with partial manifest.
  - [phase-4-curation-wrappers.md](phase-4-curation-wrappers.md) — `create_root_curation` / `preview_merge_curation` / `apply_merge_curation` / `summarize_curation`.
  - [phase-5-test-harness.md](phase-5-test-harness.md) — pytest harness friction fixes (verify-first).
  - [phase-6-canonical-notebook-and-smoke-gate.md](phase-6-canonical-notebook-and-smoke-gate.md) — runnable single-session notebook, docs quickstart, and the end-to-end UX smoke-test release gate.

## Dependency DAG

```text
phase-1 preset-discovery ─┐
phase-2 preflight ────────┤
phase-3 observability ────┤   (phase-3 sequenced after phase-2: both edit run_v2_pipeline)
phase-4 curation-wrappers ┤
                          └─> phase-6 canonical-notebook + UX smoke gate  (exercises 1–4 end-to-end)

phase-5 test-harness ─ (independent infra; may land in any order; landing early de-frictions running the new tests)
```

Each phase is an independent PR: the suite is green at every merge boundary, and no phase needs a later phase's code to be reviewable. Phase 6 is the capstone — it imports the surfaces from Phases 1–4, so it lands last.
