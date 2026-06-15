# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

File:line refs into the existing v2 code showing exactly what each phase touches and what it leaves alone. Anchored to live code at planning time — re-verify before editing (line numbers drift).

- `src/spyglass/spikesorting/v2/pipeline.py:47` — `list_presets()`: Phase 1 adds a sibling `describe_presets()`; `list_presets` is preserved unchanged.
- `src/spyglass/spikesorting/v2/pipeline.py:29-45` — `_Preset` Pydantic model (`preproc_params_name`, `artifact_params_name`, `sorter`, `sorter_params_name`): Phase 1 extends it with optional human-facing fields (`intended_use`, `threshold_units`, `notes`) and populates them for the three built-in presets at `pipeline.py:64-83`.
- `src/spyglass/spikesorting/v2/pipeline.py:86-291` — `run_v2_pipeline()`: Phase 2 adds a `preflight: bool = True` parameter and an early preflight call; Phase 3 wraps each stage with timing/status capture and changes the returned manifest. The current manifest keys (`pipeline.py:283-291`) are **preserved**; new keys are added alongside them.
- `src/spyglass/spikesorting/v2/pipeline.py:208-235` — the three `insert_selection` → `populate` stage blocks (recording, artifact, sorting): Phase 3 instruments these; Phase 2 reads (does not insert) the same upstream tables for existence checks.
- `src/spyglass/spikesorting/v2/_selection_identity.py:106` — `deterministic_id(kind, payload)`, DB-free and content-addressed: Phase 2 reuses it to pre-compute the exact selection PKs a run will produce, for the preflight `expected_ids` field, without inserting.
- `src/spyglass/spikesorting/v2/recording.py:959` — `RecordingSelection.insert_selection()`; `recording.py:338` — `SortGroupV2.set_group_by_shank()`: Phase 2 reads the upstream tables these depend on (`Session`, `IntervalList`, `LabTeam`, `SortGroupV2`, `PreprocessingParameters`) read-only; it does not call `insert_selection`.
- `src/spyglass/spikesorting/v2/sorting.py:402-409` — the `installed_sorters()`-backed availability gate (binary-present check, stricter than `available_sorters()` spelling check at `sorting.py:195-211`): Phase 2 reuses the strict gate so a spelled-valid sorter whose binary is absent is caught before populate.
- `src/spyglass/spikesorting/v2/curation.py:197` — `CurationV2.insert_curation()` (the expert API): Phase 4 adds named classmethod wrappers next to it; `insert_curation` is preserved unchanged as the expert surface.
- `src/spyglass/spikesorting/v2/curation.py:1043` — `CurationV2.get_merge_groups()`, `curation.py:89` — the `merges_applied` field: Phase 4's `summarize_curation` reads these.
- `src/spyglass/spikesorting/v2/exceptions.py:75-89` — typed exception module: Phase 2 adds `PreflightError`; Phase 3 adds `PipelineStageError`.
- `src/spyglass/spikesorting/v2/__init__.py:14` — `initialize_v2_defaults()`: unchanged; the canonical notebook (Phase 6) calls it as step 1.
- `tests/conftest.py:489-501` — `pytest_configure` initializes the shared `DataDownloader` near the end of configure, and `pytest_unconfigure` references the `SERVER` global unconditionally under `if TEARDOWN:`; `tests/conftest.py:461` is where `SERVER` is assigned in `pytest_configure`: Phase 5 guards teardown and verifies pure-helper runs do not start shared downloads. `tests/data_downloader.py:50-72` — downloader construction eagerly starts `curl` processes; Phase 5 includes it in the no-spurious-download reproduction. `pyproject.toml:180-220` — `addopts` (`-p no:warnings` at `:185`) and `filterwarnings`: Phase 5 reconciles these (verify-first). `tests/spikesorting/v2/conftest.py:123-155` — `pytest_sessionstart` downloads the smoke fixture unconditionally even for pure-helper tests: Phase 5 gates the download.

## Scope and dependency policy

### Goals

- A scientist can discover what each preset does (`describe_presets()`) without reading module source.
- A misconfigured run fails in **seconds** with a structured, actionable report — not minutes/hours into `populate()` with an opaque FK or sorter error.
- A completed run reports, per stage, whether work was **computed** vs **reused** and how long it took; a failed run raises a stage-aware exception carrying the partial manifest.
- Basic curation (initial curation, proposed merges, applied merges) has discoverable, named entry points; the expert `insert_curation` stays for power users.
- The v2 test suite runs without spurious downloads or error-masking teardown.
- A new user can run one notebook end-to-end (defaults → sort group → preflight → pipeline → curation summary → downstream fetch), and CI proves that path stays working.

### Non-Goals

- **No FigPack / web curation UI.** Owned by [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) (master roadmap).
- **No concat / cross-session / UnitMatch surface.** Owned by master-roadmap Phases 3–5. `run_v2_pipeline` stays single-session here.
- **No new presets** (KS4, sameday-concat, `register_preset`). Owned by master-roadmap Phase 5. This plan only *describes* the three presets that already ship.
- **No auto-curation / metrics** (`auto_curate=True`). Owned by master-roadmap Phase 2.
- **No schema migrations.** Adding optional fields to the in-memory `_Preset` Pydantic model and adding methods/exceptions is not a DataJoint schema change; no `params_schema_version` bump, no table-`definition` edits. (Consistent with the project's pre-release schema policy.)
- **No removal of `insert_curation`.** The wrappers are thin sugar over it.

### Dependency policy

No new third-party dependencies. `pandas` (used by `describe_presets`) is already a core Spyglass dependency; import it lazily inside the function to keep `import spyglass.spikesorting.v2` free of heavy imports, matching the existing lazy-import convention in `__init__.py:1-11`.

## Relationship to the master roadmap

This plan and `.claude/docs/plans/spikesorting-v2/` are complementary, not competing:

| Surface | This plan (now, on the MVP) | Master roadmap (later) |
| --- | --- | --- |
| Preset catalog | `describe_presets()` over the 3 shipping presets | Phase 5: `register_preset`, KS4 + concat presets, richer `_params/preset.py` model |
| `run_v2_pipeline` | preflight + observability (single-session) | Phase 5: `auto_curate`, concat mode, `figpack`, `run_v2_unit_match` |
| Curation | named Python wrappers over `insert_curation` | Phase 5: FigPack web curation UI |
| Notebook | minimal runnable single-session `.ipynb` | Phase 5: adds FigPack + cross-session cells to the **same** notebook |

**One notebook, not two.** Phase 6 created `notebooks/10_Spike_SortingV2.ipynb`; master-roadmap Phase 5 should extend that notebook rather than create a competing one. This is flagged in [phase-6](phase-6-canonical-notebook-and-smoke-gate.md#deliberately-not-in-this-phase).

**Manifest-drift note (for the master-roadmap executor, not an action here):** [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) and `designs.md` describe a manifest with keys `recording` / `artifact_detection` / `sorting` / `initial_curation` that **never shipped** — the actual manifest uses `recording_id` / `artifact_id` / `sorting_id` / `curation_id` (`pipeline.py:283-291`). This plan builds on the shipped keys and preserves them (see [shared-contracts.md § Pipeline manifest](shared-contracts.md#pipeline-manifest-schema)). The master roadmap's manifest design should be reconciled to the shipped keys when Phase 5 is executed.

## Metrics

- **Preflight speed:** `preflight_v2_pipeline()` returns in < ~1 s on the smoke fixture's session (no `populate`, no recording materialization). Asserted by a wall-clock bound in the Phase 2 validation slice.
- **Preflight accuracy:** for a valid configuration, `expected_ids` equals the selection PKs a subsequent `run_v2_pipeline` actually produces (round-trip assertion in Phase 2). `curation_id` is not precomputed because it is assigned by `CurationV2.insert_curation`.
- **Idempotency preserved:** after Phases 2–3, two identical `run_v2_pipeline` calls still return equal manifests modulo the timing field, and insert no duplicate rows (regression assertion).
- **Notebook gate:** `notebooks/10_Spike_SortingV2.ipynb` executes end-to-end against the smoke fixture in CI with zero errors (Phase 6).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Preflight checks drift from what `populate()` actually requires (a check passes but the run still fails, or vice-versa). | Preflight reuses the *same* identity/payload path (`deterministic_id`) and the *same* strict sorter gate (`installed_sorters()`) the real code uses; where `insert_selection` builds the identity payload inline, Phase 2 extracts a shared payload builder so the two cannot diverge. See [phase-2](phase-2-preflight.md). |
| `preflight=True` default breaks an existing caller whose configuration preflight wrongly rejects (false positive). | Preflight failures raise a typed `PreflightError` listing each failed check with the exact fix; callers can pass `preflight=False` to bypass. The default is justified because the alternative is a worse failure deep in `populate()`. |
| Per-stage timing is misread as compute cost on an idempotent re-run (re-runs no-op in `populate`, so `stage_seconds≈0`). | The manifest carries a per-stage **status** (`computed`/`reused`) alongside `stage_seconds`; the field is documented as "wall-clock spent *this call*," not cumulative compute. See [shared-contracts.md](shared-contracts.md#stage-status-values). |
| Changing the manifest shape breaks the master-roadmap Phase 5 tests / downstream readers. | Current keys are strictly preserved; only additive keys (`*_status`, `stage_seconds`, `warnings`) are introduced. A regression test pins the original keys' presence and values. |
| Curation wrappers obscure the merge DAG (a proposed/applied merge wrapper always roots a new curation, losing the parent-child branch model). | Merge wrappers accept an optional `parent_curation_id` (default `-1`) so merges can branch off an existing initial curation, matching `insert_curation`'s DAG semantics. See [phase-4](phase-4-curation-wrappers.md). |
| Phase 5 harness "fixes" encode a problem that isn't actually there (the SERVER/addopts claims are hypotheses). | Phase 5 is **verify-first**: each fix has a reproduction step that must fail before the fix and pass after; a claim that cannot be reproduced is dropped, not patched speculatively. |

## Rollout Strategy

All-additive, no feature flag needed. Every change is opt-in or backward-compatible:

- `describe_presets`, `preflight_v2_pipeline`, and the curation wrappers are new symbols — nothing depends on them yet.
- `run_v2_pipeline` gains `preflight=True` (new default behavior is a *fast safety check*, fully bypassable) and additive manifest keys (existing keys unchanged).
- Harness changes are test-infra only.

No deprecation window is required: nothing is removed or repurposed. The one user-visible behavior change — `preflight=True` by default — is a guard that runs before any work and is documented in the `run_v2_pipeline` docstring and CHANGELOG (Phase 2).

## Open Questions

1. **`describe_presets` return type — DataFrame vs list-of-dicts?** Current best answer: return a `pandas.DataFrame` (notebook-friendly, the dominant Spyglass return convention for catalog accessors), imported lazily. If a caller wants raw data, `.to_dict("records")` is one call away. Settled unless the executor finds a no-pandas constraint.
2. **`expected_ids` depth in preflight — exact deterministic IDs, or just "exists/will-create" booleans?** Current best answer: compute exact IDs via `deterministic_id` (the helper is DB-free and already used by `insert_selection`), and annotate each with whether the row already exists. Acceptable fallback if payload extraction proves entangled: resolved param names + per-stage existence booleans, with the deterministic-ID upgrade deferred. Decision recorded in [phase-2](phase-2-preflight.md).
3. **Curation wrappers — classmethods on `CurationV2` or free functions?** Current best answer: classmethods on `CurationV2`, matching the existing `insert_curation` / `get_merge_groups` / `get_merged_sorting` placement, so they're discoverable from the table. Settled.

## Estimated Effort

Small-to-moderate, dominated by tests and the notebook. Rough diff sizing:

- Phase 1: ~60 LOC src + ~80 LOC tests.
- Phase 2: ~150 LOC src (preflight + report dataclass + `PreflightError` + payload-builder extraction) + ~200 LOC tests.
- Phase 3: ~120 LOC src (instrumentation + `PipelineStageError`) + ~150 LOC tests.
- Phase 4: ~120 LOC src (4 wrappers) + ~180 LOC tests.
- Phase 5: ~40 LOC test-infra changes + repro notes.
- Phase 6: notebook (~10 code cells) + ~120 LOC docs + ~120 LOC smoke test.
