# Spike Sorting v2 — UX follow-ups Implementation Plan

**Status (2026-06-19):** Phases 1, 2a, 3, and 4 are **complete and committed**
on branch `spikesorting-v2`. Phase 3's docs/docstring polish landed, but its
`describe_intervals` helper was **dropped by decision** (interval discovery is
cross-pipeline — see the Phase 3 entry). The only remaining work is Phase 2b,
which stays **gated** (needs the analyzer-curation phase + scientific sign-off).

- **Phase 1 — run-summary rename: ✅ complete** (`db3cf1bf`, `c5710036`,
  `8ac81914`).
- **Phase 2a — parameter catalog + content identity: ✅ complete**
  (`669789ac` → `ab877245` → `0f7eb32c` → `b2a01e65` → `a7b201cb`). Shipped:
  region preproc (hippocampus 600 / cortex 300 Hz, 1.5 ms min-segment),
  rate-keyed MS4 family (MS4 was the default; a later PR-review fix moved the
  shipped default to MS5 because MS4's ml_ms4alg backend needs numpy<2 -- see
  below), 100/50 µV artifact rows, dated
  names + content fingerprints, the duplicate-content guard
  (`allow_duplicate_params` escape hatch), `describe_parameter_rows()`, and a
  preflight sampling-rate check. **Decisions that diverged from this plan as
  written:** MS5 ships as `recommendation_status="alternative"` (NOT
  "comparison"); the 500 µV artifact row keeps the name `default` (not renamed
  "bugfix"); MS4 sorter rows are rate-keyed and region-agnostic (region lives
  on the preproc row, since MS4 runs `filter=False`). An **experimental
  Neuropixels Kilosort4 preset** (`franklab_neuropixels_ks4_2026_06`, matched
  to the AIND `aind-ephys-spikesort-kilosort4` recipe — community-grounded, not
  lab-attested) was brought forward from 2b.
- **Phase 2b — gated.** NP-KS4 partially addressed above; the rest (Set A
  downstream curation, tetrode-20 kHz, MS5 probe) stays deferred pending the
  analyzer-curation phase + scientific sign-off.
- **Phase 3 — docs/docstring polish: ✅ landed, minus `describe_intervals`**
  (`f9e7fb07` → `8efe1fb1`). Shipped: zero-unit `get_sorting`/`get_analyzer`
  docstring cross-refs; gated-stub notes on the
  `SessionGroup`/`ConcatenatedRecording` forward declarations; a
  `describe_pipeline_presets()` filter example for discovering the experimental
  Neuropixels/Kilosort4 preset by name (the dated
  `franklab_neuropixels_ks4_2026_06` from 2a — no new preset names); and the
  `your_session.nwb` notebook/doc placeholder fix. **Dropped by decision:** the
  planned `describe_intervals(nwb_file_name)` helper. Summarizing a session's
  `IntervalList` rows is cross-pipeline (every pipeline takes an
  `interval_list_name`), so it does not belong on the spikesorting user surface;
  if wanted later, add it as a generic `IntervalList`/`common` helper (mirroring
  the existing `IntervalList.plot_intervals`). `interval_list_name` stays a
  required arg.
- **Phase 4 — session preflight + runner: ✅ complete** (`15941a14`). Shipped:
  `preflight_v2_pipeline_session()` (read-only whole-session check aggregating
  per-group `preflight_v2_pipeline` into a new `PreflightSessionReport`) and
  `run_v2_pipeline_session()` (batch runner over all/selected sort groups with
  per-group `outcome` reporting), plus a shared `_resolve_session_sort_group_ids`
  target resolver. Both require an explicit `pipeline_preset`; the runner
  fail-fasts or collects per-group preflight/sort failures
  (`continue_on_error`), catching exactly `PipelineStageError` /
  `PreflightError` / `ZeroUnitSortError` and letting input/unexpected errors
  stop the batch. Sequential, `list[dict]` return. Docs + user notebook gained a
  whole-session example. **Deliberately not done:** cross-group parallelism, a
  DataFrame return type.

All four UX-followup phases are now landed; only Phase 2b remains gated.

Branch audit summary: the public run-summary API rename has landed
(`run_v2_pipeline` uses a `run_summary` local and
`PipelineStageError.partial_run_summary` exists). Phase 2b remains gated and
should not be implemented without the analyzer-curation phase and scientific
sign-off.

Ordered, separately-shippable improvements to the `spyglass.spikesorting.v2`
user-facing surface, identified in a UX audit of the orchestration layer:
(1) finish the residual "manifest" → "run summary" cleanup left after the main
API rename; (2a) correct v2's shipped "franklab" parameter blobs to the real
DB-attested production recipes (region-based preproc — hippocampus 600 Hz /
cortex 300 Hz, the MS4 `franklab_probe_*` family by adjacency_radius × rate,
100/50 µV artifact), make them dated/fingerprinted/inspectable (a later
PR-review fix moved the shipped `run_v2_pipeline` default to MS5, since MS4's
`ml_ms4alg` backend needs numpy<2 — MS4 stays the production probe recipe);
(2b) **gated** recipes that can't ship yet —
Set A's downstream curation stages (need the analyzer-curation phase) and
unattested recipes (need scientific sign-off); (3) close
the interval-discovery and documentation-polish gaps on top of that canonical
catalog; (4) add `preflight_v2_pipeline_session()` plus
`run_v2_pipeline_session()`, so a user can validate and then sort every sort
group in a session instead of forcing the user to loop one `sort_group_id` at a
time. All changes are additive or pre-release renames; the returned-dict keys and
`merge_id` consumers are unaffected (Phase 2a does change the derived ID *values*
— acceptable pre-release; see overview).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, tasks, validation slice, and fixtures.
2. **Need broader scope / risks / the run-summary key contract / the selection-identity caveat?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, the run-summary dict contract, the name-based selection-identity caveat, risks, open questions.
- [appendix.md](appendix.md) — SpikeInterface 0.104.3 deep reference (sorter/preproc/SortingAnalyzer/metrics/motion internals, with SI file:line and divergences). Load when implementing against SI — it resolves `detect_threshold` semantics, the `nn_advanced` rename, the custom `isi_violation` fraction, the analyzer recompute cascade, and the motion/torch dependencies.
- Phases (each ships as a separable PR, in order):
  - ✅ **[phase-1-run-summary-rename.md](phase-1-run-summary-rename.md)** (complete) — residual "manifest" → "run summary" cleanup in tests/comments/docs. No pipeline behavior change.
  - ✅ **[phase-2a-parameter-names-and-fingerprints.md](phase-2a-parameter-names-and-fingerprints.md)** (complete) — corrected v2's blobs to the DB-attested recipes (region preproc 600/300 Hz, rate-keyed MS4 family, 100/50 µV artifact), dated/fingerprinted, duplicate-content guard, `describe_parameter_rows()`, preflight sampling-rate check, + an experimental Neuropixels KS4 preset. (2a made MS4 the default; a later PR-review fix moved the shipped default to MS5 since MS4's `ml_ms4alg` backend needs numpy<2 — MS4 stays the production recipe.) See the Status block above for decisions that diverged from this file as written.
  - [phase-2b-deferred-and-unattested.md](phase-2b-deferred-and-unattested.md) — **GATED.** Recipes that can't ship yet: Set A's downstream curation stages (wait on the analyzer-curation phase) and unattested recipes — Neuropixels tuning, tetrode-20 kHz, MS5 probe (wait on scientific sign-off). Depends on 2a; not a UX follow-up.
  - ✅ **[phase-3-discovery-and-polish.md](phase-3-discovery-and-polish.md)** (landed, minus `describe_intervals`) — zero-unit docstring cross-refs, gated-stub notes, the notebook placeholder fix, and `describe_pipeline_presets()`-based surfacing of the canonical KS4/NPX preset. The planned `describe_intervals()` helper was **dropped by decision** (interval discovery is cross-pipeline; promote to a `common`/`IntervalList` helper if needed).
  - ✅ **[phase-4-session-runner.md](phase-4-session-runner.md)** (complete) — `preflight_v2_pipeline_session()` read-only whole-session validation plus `run_v2_pipeline_session()` multi-sort-group batch runner with per-group outcome reporting (shared `_resolve_session_sort_group_ids`, `PreflightSessionReport`). Sequential, `list[dict]` return; explicit `pipeline_preset` required.
