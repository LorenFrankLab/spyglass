# Phase 5 — UX overhaul: pipeline orchestrator, FigPack, notebook rewrite

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#run_v3_pipeline-orchestrator)

The capstone phase. Adds the `run_v3_pipeline()` convenience function (35-cell notebook → 8-cell notebook), Pydantic-validated parameter presets, FigPack as the v3 curation UI, and the full v3 notebook walkthrough. **v1 is NOT sunset by this plan**; v0 and v1 continue to coexist with v3 indefinitely (per resolved decision in overview).

**Inputs to read first:**

- All files implemented in Phases 1–4. Phase 5 doesn't add new tables; it adds convenience layers on top.
- [notebooks/10_Spike_SortingV1.ipynb](notebooks/10_Spike_SortingV1.ipynb) — the v1 notebook v3 replaces.
- [notebooks/11_Spike_Sorting_Analysis.ipynb](notebooks/11_Spike_Sorting_Analysis.ipynb) — downstream consumer notebook; should work unchanged with v3 outputs.
- [src/spyglass/spikesorting/v1/figurl_curation.py](src/spyglass/spikesorting/v1/figurl_curation.py) — FigURL pattern to mirror in FigPack.
- [.claude/docs/plans/spikesorting-v3/appendix.md § FigPack vs FigURL](appendix.md#figpack-vs-figurl) — migration policy.

**Contracts referenced:**

- [`insert_selection()` Return-Value Normalization](shared-contracts.md#insert_selection-return-value-normalization) — `run_v3_pipeline()` relies on this contract to be idempotent.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — Phase 5 introduces a `Preset` Pydantic schema that bundles parameter names.

**Designs referenced:** [`run_v3_pipeline()` Orchestrator](designs.md#run_v3_pipeline-orchestrator), [FigPackCuration](designs.md#figpackcuration).

## Tasks

- **EXTEND `pipeline.py`** (Phase 1 shipped a minimal version with 3 presets covering recording → artifact → sorting → initial curation). Phase 5 adds the missing stages and broadens the preset set per [designs.md § `run_v3_pipeline()` Orchestrator](designs.md#run_v3_pipeline-orchestrator):
  - **Add `auto_curate=True` parameter** — wires up Phase 2's `AnalyzerCuration` stage and the materialization step.
  - **Add `session_group_name` parameter (optional)** — when set, the orchestrator routes through Phase 3's `ConcatenatedRecording` instead of `Recording`. Mutually exclusive with the single-session inputs (`sort_group_id` etc.).
  - **Add `unit_match=False` parameter (optional, requires `session_group_name`)** plus `unit_match_curation_choices` — wires up Phase 4's UnitMatch path while preserving the explicit `MemberCuration` pinning contract. The orchestrator must raise if `unit_match=True` and choices are omitted; it never auto-selects "latest" curations.
  - **Add `figpack=False` parameter (optional)** — wires up the FigPack curation stage below.
  - **Expand `PRESETS`** to include Phase 5's full set:
    - `franklab_tetrode_mountainsort4`, `franklab_tetrode_mountainsort5`, `clusterless_thresholder_default` (carried over from Phase 1)
    - `franklab_probe_kilosort4` (new)
    - `franklab_tetrode_clusterless` (new — combines threshold detection + Phase 2 metrics)
    - `franklab_chronic_single_day` (new — uses SessionGroup + ConcatenatedRecording for same-day chronic)
  - `register_preset(name, preset_dict)` — public API for labs to add custom presets without modifying v3 source.
  - Each preset must reference Lookup-table row names that ALREADY EXIST. Phase 5 inserts these baseline rows via `insert_default()` calls in `__init__.py`.

- **Implement `_params/preset.py`** Pydantic model:
  ```python
  class PresetSchema(BaseModel):
      model_config = ConfigDict(extra="forbid")
      preproc_params_name: str
      artifact_params_name: str
      sorter: str  # validated against SI's available_sorters
      sorter_params_name: str
      metric_params_name: str
      auto_curation_rules_name: str
      description: str = ""
  ```
  Validates at preset-registration time that every referenced Lookup row exists (raises a clear error if a parameter set is missing). This catches the typo-at-populate failure mode entirely.

- **FigPack feasibility check FIRST**, before implementing anything. Phase 5 declares FigPack as the v3 curation UI per resolved decision #2 — but the implementer must verify the upstream package is usable before writing the table. Tasks:
  1. Confirm `figpack` is installable from PyPI under the expected name.
  2. Verify the `figpack.spike_sorting.build_curation_view(analyzer)` API (or its current equivalent) exists and works.
  3. Test on a single example: build a curation view from a v3 SortingAnalyzer end-to-end and publish to FigPack. Round-trip a known labels dict back via `fetch_curation_from_uri`.

  **If FigPack is not usable at implementation time**: STOP Phase 5 and escalate to the project owner. Per resolved decision #2, FigPack is the v3 curation UI; the plan does not silently fall back to FigURL. Surface the blocker rather than ship a degraded UI. Possible resolutions (decided by project owner, not the implementer): wait for FigPack release; pin to a specific FigPack commit; add a contribution to upstream FigPack.

- **Implement `figpack_curation.py`** per [designs.md § FigPackCuration](designs.md#figpackcuration):
  - `FigPackCurationSelection` Manual + `FigPackCuration` Computed.
  - `make()` uses the verified FigPack API. Publishes view, stores returned URI.
  - `FigPackCuration.fetch_curation_from_uri(uri) -> tuple[dict, list]` — round-trip labels + merge_groups from FigPack back into v3.
  - Gate `figpack` import inside the module with a clear install message if absent (helps users who haven't installed the `spikesorting-v3-curation` extra).

- **Add `figpack` to `pyproject.toml`** as an optional dependency:
  ```toml
  optional-dependencies.spikesorting-v3-curation = ["figpack>=X.Y"]
  ```
  Pin minimum version once verified at implementation time.

- **Write the v3 notebook**: new file `notebooks/13_Spike_SortingV3.ipynb`. Target ≤10 code cells:
  1. Imports + DataJoint config (1 cell).
  2. Insert team + session (1 cell).
  3. `SortGroupV3().set_group_by_shank(nwb_file_name=...)` (1 cell).
  4. `manifest = run_v3_pipeline(..., preset="franklab_tetrode_mountainsort5")` (1 cell).
  5. Print the manifest (1 cell).
  6. Optional: launch FigPack curation, retrieve labels (2 cells).
  7. Optional: insert into `SortedSpikesGroup` for downstream analysis (1 cell).

  Compare with [notebooks/10_Spike_SortingV1.ipynb](notebooks/10_Spike_SortingV1.ipynb) (35 code cells). Phase 5 success metric: ≤10 cells.

- **Cross-session notebook**: new file `notebooks/14_Spike_Sorting_CrossSession.ipynb`. Walks through:
  1. Build a `SessionGroup` for 3 same-day sessions.
  2. Run `run_v3_pipeline()` for each session individually.
  3. Run `UnitMatch` across the group.
  4. Inspect `TrackedUnit` membership.
  5. Decoding with `TrackedUnit`-indexed spikes (link to downstream notebook).

- **Documentation overhaul**:
  - Promote `docs/src/Pipelines/SpikeSorting/v3.md` from "new pipeline" to "recommended for new work". Add a top banner.
  - Update root README "Quick example" snippet to use `run_v3_pipeline()`.
  - Keep `docs/src/Pipelines/SpikeSorting/v1.md` accessible and live; do NOT mark v1 as deprecated. v0 and v1 remain populated paths for legacy data.
  - Add a new docs page: `docs/src/Pipelines/SpikeSorting/choosing-v1-vs-v3.md` — for users deciding which path to use. TL;DR: existing v1 sorts stay queryable through v1; new sorts go to v3 via `run_v3_pipeline`.
  - CHANGELOG.md: "v3 spike sorting is the recommended path for new work. `run_v3_pipeline()` reduces typical sort setup to a single function call. FigPack is the v3 curation UI. v0 and v1 remain supported indefinitely."

- **No v1 sunset criteria.** Per the resolved design decision in [overview.md](overview.md), v0 and v1 stay in-tree indefinitely. Phase 5 simply documents that v3 is the recommended path for new sorts; v1 docs stay live and unmarked-as-deprecated.

- **End-to-end integration test** `tests/spikesorting/v3/test_run_pipeline.py`:
  - `test_run_v3_pipeline_minirec_clusterless` — calls `run_v3_pipeline(...)`, asserts manifest has all 5 stages + valid merge_id, downstream `SpikeSortingOutput.get_spike_times(...)` returns sane arrays. This is a plumbing/integration guard only; minirec is not a sort-correctness or parity oracle.
  - `test_run_v3_pipeline_idempotent` — call `run_v3_pipeline(...)` twice with identical args; second call returns the same manifest (no duplicate inserts).
  - `test_preset_validation_catches_missing_lookup_rows` — define a preset referencing a nonexistent param name; `register_preset` raises with a clear "row 'foo' not found in PreprocessingParameters" message.

- **Notebook smoke test**: `tests/notebooks/test_spike_sorting_v3_notebook.py` — uses `jupytext` (already a docs optional dep at [pyproject.toml](pyproject.toml)) to execute `notebooks/13_Spike_SortingV3.ipynb` cell-by-cell against the `minirec` fixture. Marked slow.

## Deliberately not in this phase

- **No v0/v1 source removal.** Per resolved decision in overview, v0 and v1 stay in-tree indefinitely; this plan never sunsets them. v3 does not back-port to v1's FigURL flow either.
- **No DeepUnitMatch.** Phase 4 ships only UnitMatch; DeepUnitMatch is future work via the same `MatcherProtocol` plugin.
- **No v1-to-v3 data migration tooling.** Users keep using v1 for their existing v1 sorts; new sorts go through v3. Whether to write a one-shot "convert v1 CurationV1 row to a v3 CurationV3 row" helper is decided separately.
- **No schema changes to existing v3 tables.** Per the zero-migration policy, Phase 5 only ADDS new tables (`FigPackCurationSelection`, `FigPackCuration`, and the `_params/preset.py` registrations). Any change to Phase 1–4 table definitions is forbidden.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_preset_schema_validation` | `PresetSchema(preproc_params_name="bogus", ...)` succeeds at construction but `register_preset(name, dict)` raises because the Lookup row doesn't exist. Valid preset registers cleanly. |
| `test_run_v3_pipeline_minirec_clusterless` (slow, integration) | Single call produces a valid merge_id; downstream spike-time fetch works. Does NOT assert sort correctness or parity against minirec. |
| `test_run_v3_pipeline_idempotent` | Two calls with identical args return identical manifests. No duplicate rows inserted (count check on every Selection table before/after second call). |
| `test_run_v3_pipeline_manifest_complete` | Manifest contains entries for all 5 stages (recording, artifact, sorting, initial_curation, auto_curation) and a final `merge_id`. |
| `test_register_preset_catches_typos` | Registering a preset with `preproc_params_name="defaut"` (typo) raises clearly. |
| `test_figpack_curation_make_publishes_uri` (slow, integration; optional) | Skipped if `figpack` not installed; otherwise asserts `FigPackCuration.populate(key)` returns a non-empty URI. |
| `test_figpack_round_trip_labels` (slow, integration; optional) | Publish a FigPack view with known labels; `fetch_curation_from_uri()` recovers them. |
| `test_v3_notebook_executes` (slow, integration) | `jupytext` executes `notebooks/13_Spike_SortingV3.ipynb` against `minirec` with no errors. Cell count ≤10 verified programmatically. |
| `test_cross_session_notebook_executes` (slow, integration, optional) | Executes `notebooks/14_Spike_Sorting_CrossSession.ipynb` if a multi-session fixture is available. |

## Fixtures

- **`minirec`** — used for the orchestrator end-to-end test.
- **`figpack` package** — optional install; tests gate on its presence.
- **No new conftest entries** — Phase 5 reuses fixtures from Phases 1–4.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — v1 is NOT removed, no v0/v1 schema touched, no existing v3 table altered.
- Validation slice tests pass; slow / integration tests are marked.
- `notebooks/13_Spike_SortingV3.ipynb` is ≤10 code cells (verify by running `jq '.cells | map(select(.cell_type == "code")) | length' notebooks/13_Spike_SortingV3.ipynb`).
- `run_v3_pipeline()` is idempotent (the manifest comparison test passes).
- FigPack feasibility was verified before implementation began (or the project owner was escalated if FigPack proved unusable — no silent fallback).
- All docs tasks landed: v3.md banner, README snippet, `choosing-v1-vs-v3.md` decision page.
- CHANGELOG.md mentions Phase 5 deliverables (orchestrator, FigPack, notebook rewrite).
- Sanity: `git diff src/spyglass/spikesorting/v0/ src/spyglass/spikesorting/v1/` is empty — no v0/v1 source touched.
- Sanity: `git diff` against any Phase 1–4 table `definition` strings is empty — zero-migration policy honored.
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
