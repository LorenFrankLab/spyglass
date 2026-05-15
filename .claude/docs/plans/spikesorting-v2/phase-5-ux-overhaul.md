# Phase 5 — UX overhaul: pipeline orchestrator, FigPack, notebook rewrite

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#run_v2_pipeline-orchestrator)

The capstone phase. Adds the `run_v2_pipeline()` convenience function (35-cell notebook → ≤10-cell notebook), Pydantic-validated parameter presets, FigPack as the v2 curation UI, and the full v2 notebook walkthrough. **v1 is NOT sunset by this plan**; v0 and v1 continue to coexist with v2 indefinitely.

## Executor Checklist

- Verify the current FigPack spike-sorting API first; stop and escalate if edited curation state cannot be persisted and fetched.
- Extend `pipeline.py` with full presets, auto-curation, concat routing, FigPack routing, and separate `run_v2_unit_match()`.
- Implement preset validation and optional FigPack dependency gates.
- Implement `FigPackCurationSelection`, `FigPackCuration`, URI generation, and curation round-trip only against the verified FigPack API.
- Rewrite notebooks/docs so v2 is easier to use while v0/v1 remain available.
- Run end-to-end notebook/orchestrator tests in the isolated database. Production-connected real-data smoke is optional and must use the explicit production-smoke gate with test schemas/temp output directories.
- Run the Phase 5 validation goals plus `code_graph.py describe/path` for FigPack tables.

**Inputs to read first:**

- All files implemented in Phases 1–4. Phase 5 adds only its own FigPack tables (`FigPackCurationSelection`, `FigPackCuration`) and preset registrations; it must not alter any Phase 1–4 table definitions.
- [notebooks/10_Spike_SortingV1.ipynb](notebooks/10_Spike_SortingV1.ipynb) — the v1 notebook v2 replaces.
- [notebooks/11_Spike_Sorting_Analysis.ipynb](notebooks/11_Spike_Sorting_Analysis.ipynb) — downstream consumer notebook; should work unchanged with v2 outputs.
- [src/spyglass/spikesorting/v1/figurl_curation.py](../../../../src/spyglass/spikesorting/v1/figurl_curation.py) — FigURL pattern to mirror in FigPack.
- [.claude/docs/plans/spikesorting-v2/appendix.md § FigPack vs FigURL](appendix.md#figpack-vs-figurl) — migration policy.

**Global invariants apply:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) and [Code Artifact Naming](shared-contracts.md#code-artifact-naming).

**Phase-specific contracts referenced:**

- [`insert_selection()` Return-Value Normalization](shared-contracts.md#insert_selection-return-value-normalization) — `run_v2_pipeline()` relies on this contract to be idempotent.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — Phase 5 introduces a `Preset` Pydantic schema that bundles parameter names.
- [Custom Exception Classes](shared-contracts.md#custom-exception-classes) — pipeline input-mode errors and FigPack empty-sort errors use the shared exception module.

**Designs referenced:** [`run_v2_pipeline()` Orchestrator](designs.md#run_v2_pipeline-orchestrator), [FigPackCuration](designs.md#figpackcuration).

## Tasks

- **EXTEND `pipeline.py`** (Phase 1 shipped a minimal version with 3 presets covering recording → artifact → sorting → initial curation). Phase 5 adds the missing stages and broadens the preset set per [designs.md § `run_v2_pipeline()` Orchestrator](designs.md#run_v2_pipeline-orchestrator):
  - **Add `auto_curate=False` parameter** — wires up Phase 2's `AnalyzerCuration` stage and materialization only when the caller opts in. The default remains initial-curation-only so a convenience call does not silently commit suggested labels/merges into a new `CurationV2`. If `auto_curate=True`, the returned manifest must include both the `AnalyzerCuration` suggestion row and the materialized child `CurationV2` row.
  - **Add `concat_session_group_owner` + `concat_session_group_name` parameters (optional pair)** — when set together, the orchestrator routes through Phase 3's `ConcatenatedRecording` instead of `Recording`. They are mutually exclusive with the single-session inputs (`nwb_file_name`, `sort_group_id`, `interval_list_name`, `team_name`) and are only for same-day / explicit-opt-in concatenated sorting. The owner field is required because `SessionGroup` names are team-namespaced. In concat mode, `team_name` is invalid; member teams come from `SessionGroup.Member.team_name`. Concat-mode manifests contain `concat_recording` in place of `recording` and omit `artifact_detection` until concat-wide artifact semantics land. These parameters are deliberately NOT reused for UnitMatch.
  - **Add `run_v2_unit_match()` helper** — a separate convenience function for Phase 4's sort-then-match path. Signature: `run_v2_unit_match(session_group_owner, session_group_name, matcher_params_name="unitmatch_default", curation_choices=None) -> dict`. It requires explicit `curation_choices` keyed by `SessionGroup.member_index`, calls `UnitMatchSelection.insert_selection(..., curation_choices=...)`, populates `UnitMatch` and `TrackedUnit`, and returns a manifest with `unitmatch_id`. It never auto-selects "latest" curations. Keeping this separate prevents the concat-sorting workflow from being confused with the per-member curation workflow required by UnitMatch.
  - **Add `figpack=False` parameter (optional)** — wires up the FigPack curation stage below. `figpack=True` publishes/builds the curation view, returns the URI in the manifest, and exits; it does not block waiting for interactive edits. Users call `FigPackCuration.fetch_curation_from_uri()` after labeling.
  - **Expand `PRESETS`** to include Phase 5's full set:
    - `franklab_tetrode_mountainsort4`, `franklab_tetrode_mountainsort5`, `franklab_tetrode_clusterless_thresholder` (carried over from Phase 1)
    - `franklab_probe_kilosort4` (new)
    - `franklab_tetrode_mountainsort5_sameday_concat` (new — uses SessionGroup + ConcatenatedRecording for same-day chronic; sets `motion_correction_params_name="auto_default"`)
  - Built-in preset names follow `{lab}_{probe_or_modality}_{sorter_or_workflow}` plus an optional topology suffix. External labs should use the same pattern, e.g. `berke_intan_mountainsort5` or `franklab_tetrode_mountainsort5_sameday_concat`.
  - `register_preset(name, preset_dict)` — public API for labs to add custom presets without modifying v2 source.
  - Each preset must reference Lookup-table row names that ALREADY EXIST. Phase 5 inserts these baseline rows via `insert_default()` calls in `__init__.py`.

- **Add the final `run_v2_pipeline()` docstring** from [designs.md § `run_v2_pipeline()` Orchestrator](designs.md#run_v2_pipeline-orchestrator). The docstring must include `Parameters`, `Returns`, and `Raises` sections; state that exactly one input mode is required; list the single-session fields and concat fields; say concat mode rejects `team_name`; and quote the `PipelineInputError` message for mixed/missing/partial modes.

- **Implement `_params/preset.py`** Pydantic model. Required fields: `preproc_params_name`, `artifact_params_name`, `sorter`, `sorter_params_name`, `metric_params_name`, and `auto_curation_rules_name`. Optional fields: `motion_correction_params_name` and `description`. Binding behavior: extra fields are forbidden; `sorter` is validated against SI's available sorters; preset registration validates that every referenced Lookup row exists and raises clearly if a parameter set is missing. `motion_correction_params_name` is optional for ordinary single-session presets and required only for presets intended for concat session groups. This catches the typo-at-populate failure mode entirely.

- **FigPack feasibility check FIRST**, before implementing anything. FigPack is the v2 curation UI, but the implementer must verify the upstream package is usable before writing the table. Tasks:
  1. Confirm the actual installable package set. Current upstream uses the core `figpack` package plus a spike-sorting extension package (`figpack-spike-sorting` on PyPI, imported as `figpack_spike_sorting` in the upstream repository); do not assume `figpack` alone provides spike-sorting views.
  2. Verify the current spike-sorting extension API. Do not assume the stale example `figpack.spike_sorting.build_curation_view(analyzer)` or `view.publish()` exists; pin the real import path, view-construction API, and upload method in this plan before writing the DataJoint table.
  3. Test on a single example: build a curation view from a v2 SortingAnalyzer end-to-end and publish/upload to FigPack. Round-trip a known labels dict and merge-groups representation back via the verified API or documented state file. If FigPack can display a curation view but cannot persist edited curation state in a retrievable form, stop and escalate before schema finalization.
  4. Record the verified FigPack and `figpack-spike-sorting` versions in this phase's PR description and in the optional dependency lower bounds.

  **If FigPack is not usable at implementation time**: STOP Phase 5 and escalate to the project owner. The plan does not silently fall back to FigURL. Surface the blocker rather than ship a degraded UI. Possible resolutions (decided by project owner, not the implementer): wait for FigPack release; pin to a specific FigPack commit; add a contribution to upstream FigPack.

- **Implement `figpack_curation.py`** per [designs.md § FigPackCuration](designs.md#figpackcuration):
  - `FigPackCurationSelection` Manual + `FigPackCuration` Computed.
  - `FigPackCurationSelection.insert_selection(curation_key, label_options=None, metrics=None, upload=True, ephemeral=False)` mirrors v1's explicit curation-UI identity instead of storing only `-> CurationV2`. The selection row stores `figpack_config_hash` plus label options, requested metrics, and upload mode so repeated calls are idempotent and multiple UI configurations for the same curation are representable without querying blob equality.
  - Default `label_options` must use the v2 enum labels (`["mua", "accept", "noise"]`), not FigURL-era `"good"`, unless Phase 5 explicitly adds a validated alias.
  - `make()` uses the verified FigPack API. Publishes/uploads the view, stores returned URI.
  - `build_curation_view(curation_key, label_options=None, metrics=None, upload=True, ephemeral=False) -> str` or an equivalently named helper is the v2 analog of `FigURLCurationSelection.generate_curation_uri`: it creates/inserts the `FigPackCurationSelection` row, populates `FigPackCuration`, and returns the URI. The exact helper internals wrap the verified FigPack API from the feasibility check.
  - `FigPackCuration.fetch_curation_from_uri(uri) -> tuple[dict, list]` — round-trip labels + merge_groups from FigPack back into v2, but only after the feasibility check proves that edited state is persisted and retrievable.
  - Gate `figpack` and `figpack_spike_sorting` imports inside the module with a clear install message if absent (helps users who haven't installed the `spikesorting-v2-curation` extra).

- **Add FigPack packages to `pyproject.toml`** as optional dependencies:
  ```toml
  optional-dependencies.spikesorting-v2-curation = [
      "figpack>=X.Y",
      "figpack-spike-sorting>=A.B",
  ]
  ```
  Pin minimum versions once verified at implementation time. Do not add these to core Spyglass dependencies.

- **Write the v2 notebook**: new file `notebooks/13_Spike_SortingV2.ipynb`. Target ≤10 code cells:
  1. Imports + DataJoint config (1 cell).
  2. Insert team + session (1 cell).
  3. `SortGroupV2().set_group_by_shank(nwb_file_name=...)` (1 cell).
  4. `manifest = run_v2_pipeline(..., preset="franklab_tetrode_mountainsort5")` (1 cell).
  5. Print the manifest (1 cell).
  6. Optional: launch FigPack curation, retrieve labels (2 cells).
  7. Optional: insert into `SortedSpikesGroup` for downstream analysis (1 cell).

  Compare with [notebooks/10_Spike_SortingV1.ipynb](notebooks/10_Spike_SortingV1.ipynb) (35 code cells). Phase 5 success metric: ≤10 cells.

- **Cross-session notebook**: new file `notebooks/14_Spike_Sorting_CrossSession.ipynb`. Walks through:
  1. Build a `SessionGroup` for 3 same-day sessions.
  2. Run `run_v2_pipeline()` for each session individually.
  3. Run `run_v2_unit_match(session_group_owner, session_group_name, curation_choices=...)` across the group.
  4. Inspect `TrackedUnit` membership.
  5. Decoding with `TrackedUnit`-indexed spikes (link to downstream notebook).

- **Documentation overhaul**:
  - Promote `docs/src/Features/SpikeSortingV2.md` from "new pipeline" to "recommended for new work". Add a top banner.
  - Update root README "Quick example" snippet to use `run_v2_pipeline()`.
  - Keep existing v0/v1 docs, API references, and notebooks accessible and live; do NOT mark v1 as deprecated. v0 and v1 remain populated paths for legacy data.
  - Add a new docs page: `docs/src/Features/ChoosingSpikeSortingV1VsV2.md` — for users deciding which path to use. TL;DR: v2 is recommended for new work; v1 remains supported and available. Existing v1 sorts stay queryable through v1.
  - In `docs/src/Features/ChoosingSpikeSortingV1VsV2.md`, include the import path explicitly: external or ground-truth NWB Units still use the existing `ImportedSpikeSorting` workflow and appear in `SpikeSortingOutput.ImportedSpikeSorting`; they are not reinserted as `CurationV2` rows.
  - Add CHANGELOG entry noting the full v2 orchestrator, FigPack curation, notebook rewrite, v2's recommended-for-new-work designation, v0/v1 continued support, and the opt-in `auto_curate=True` path.

- **No v1 sunset criteria.** Per the resolved design decision in [overview.md](overview.md), v0 and v1 stay in-tree indefinitely. Phase 5 simply documents that v2 is the recommended path for new sorts; v1 docs stay live and unmarked-as-deprecated.

- **End-to-end integration test** `tests/spikesorting/v2/test_run_pipeline.py`:
  - `test_run_v2_pipeline_minirec_clusterless` — calls `run_v2_pipeline(...)` with default `auto_curate=False`, asserts manifest has the expected single-session stages (`recording`, `artifact_detection`, `sorting`, `initial_curation`) plus valid `merge_id`; downstream `SpikeSortingOutput.get_spike_times(...)` returns sane arrays. This is a plumbing/integration guard only; minirec is not a sort-correctness or parity oracle.
  - `test_run_v2_pipeline_auto_curate_opt_in` — calls `run_v2_pipeline(..., auto_curate=True)` and asserts the manifest adds `auto_curation` plus a materialized child curation row.
  - `test_run_v2_pipeline_concat_manifest` — calls concat mode on a small same-day `SessionGroup`; asserts the manifest contains `concat_recording`, `sorting`, `initial_curation`, and optional `auto_curation`, and does not contain `artifact_detection`.
  - `test_run_v2_pipeline_idempotent` — call `run_v2_pipeline(...)` twice with identical args; second call returns the same manifest (no duplicate inserts).
  - `test_run_v2_pipeline_rejects_mixed_single_and_concat_inputs` — passing both single-session inputs and `concat_session_group_owner` / `concat_session_group_name` raises a clear error before any insert.
  - `test_run_v2_pipeline_requires_concat_owner_and_name` — passing only one concat session-group field raises before insert.
  - `test_run_v2_unit_match_requires_explicit_curations` — `run_v2_unit_match(..., curation_choices=None)` raises and never auto-pins latest curations.
  - `test_preset_validation_catches_missing_lookup_rows` — define a preset referencing a nonexistent param name; `register_preset` raises with a clear "row 'foo' not found in PreprocessingParameters" message.

- **Notebook smoke test**: `tests/spikesorting/v2/test_notebooks.py` — uses `jupytext` (already a docs optional dep at [pyproject.toml](../../../../pyproject.toml)) to execute `notebooks/13_Spike_SortingV2.ipynb` cell-by-cell against the `minirec` fixture. Marked slow.

## Deliberately not in this phase

- **No v0/v1 source removal.** v0 and v1 stay in-tree indefinitely; this plan never sunsets them. v2 does not back-port to v1's FigURL flow either.
- **No DeepUnitMatch.** Phase 4 ships only UnitMatch; DeepUnitMatch is future work via the same `MatcherProtocol` plugin.
- **No v1-to-v2 data migration tooling.** Users keep using v1 for their existing v1 sorts; new sorts go through v2. Whether to write a one-shot "convert v1 CurationV1 row to a v2 CurationV2 row" helper is decided separately.
- **No schema changes to existing v2 tables.** Per the zero-migration policy, Phase 5 only ADDS new tables (`FigPackCurationSelection`, `FigPackCuration`, and the `_params/preset.py` registrations). Any change to Phase 1–4 table definitions is forbidden.

## Validation goals

Behaviors the Phase 5 validation goals must cover. Implementer chooses test names and splits.

1. **Preset validation**: `register_preset` raises clearly on a typo or missing Lookup row before the preset is usable.
2. **`run_v2_pipeline` single-session end-to-end** (slow, integration): one call produces a valid merge_id; downstream `get_spike_times` returns sane arrays; the default manifest contains `recording`, `artifact_detection`, `sorting`, `initial_curation`, and final `merge_id`. A separate opt-in call with `auto_curate=True` adds `auto_curation` and the materialized child curation.
3. **`run_v2_pipeline` idempotency**: two identical calls return the same manifest; no duplicate rows inserted (count check on every Selection table).
4. **Single-session vs concat input mode**: mixing single-session inputs with concat session-group inputs raises before any insert; supplying only one of `concat_session_group_owner` / `concat_session_group_name` raises before any insert; concat happy path returns a manifest with `concat_recording` and no `artifact_detection`.
5. **`run_v2_unit_match`**: requires explicit `curation_choices` (raises if missing — never auto-picks "latest"); two calls with identical args return the same `unitmatch_id`.
6. **FigPack feasibility-gated** (slow, integration, optional): with the `spikesorting-v2-curation` extra installed, `FigPackCuration.populate(key)` returns a non-empty URI; zero-unit curation publishes an empty view or raises a clear `EmptySortingError` (never missing-column `KeyError`); published labels round-trip via `fetch_curation_from_uri`.
7. **Notebook smoke** (slow, integration): `jupytext` executes `notebooks/13_Spike_SortingV2.ipynb` against `minirec` with no errors; programmatic check confirms code-cell count ≤10. Cross-session notebook executes when a multi-session fixture is available (optional).

## Commands to run

```bash
source .venv-spikesorting-v2/bin/activate
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

python - <<'PY'
import importlib.util
assert importlib.util.find_spec("figpack") is not None
assert importlib.util.find_spec("figpack_spike_sorting") is not None
PY

pytest tests/spikesorting/v2/test_run_pipeline.py -q
pytest tests/spikesorting/v2/test_notebooks.py -q
test "$(jq '.cells | map(select(.cell_type == "code")) | length' notebooks/13_Spike_SortingV2.ipynb)" -le 10

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe FigPackCurationSelection --file spyglass/spikesorting/v2/figpack_curation.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe FigPackCuration --file spyglass/spikesorting/v2/figpack_curation.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up FigPackCuration --file spyglass/spikesorting/v2/figpack_curation.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --down FigPackCuration --file spyglass/spikesorting/v2/figpack_curation.py --json

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 docs notebooks README.md CHANGELOG.md
git diff --exit-code -- src/spyglass/spikesorting/v0 src/spyglass/spikesorting/v1
```

## Fixtures

- **`minirec`** — used for the orchestrator end-to-end test.
- **`spikesorting-v2-curation` optional extra** — optional install; tests gate on both `figpack` and `figpack_spike_sorting` imports.
- **No new conftest entries** — Phase 5 reuses fixtures from Phases 1–4.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — v1 is NOT removed, no v0/v1 schema touched, no existing v2 table altered.
- Validation goals are covered; slow / integration tests are marked.
- `notebooks/13_Spike_SortingV2.ipynb` is ≤10 code cells (verify by running `jq '.cells | map(select(.cell_type == "code")) | length' notebooks/13_Spike_SortingV2.ipynb`).
- `run_v2_pipeline()` is idempotent (the manifest comparison test passes).
- `run_v2_unit_match()` is idempotent by `(session_group_owner, session_group_name, matcher_params_name, curation_set_hash)` and does not conflate UnitMatch with concatenated sorting.
- FigPack feasibility was verified before implementation began (or the project owner was escalated if FigPack proved unusable — no silent fallback).
- All docs tasks landed: `docs/src/Features/SpikeSortingV2.md` banner, README snippet, `docs/src/Features/ChoosingSpikeSortingV1VsV2.md` decision page.
- CHANGELOG.md mentions the delivered user-facing capabilities (orchestrator, FigPack, notebook rewrite), the opt-in `auto_curate=True` path, and v0/v1 continued support without referencing plan phases.
- Sanity: `git diff src/spyglass/spikesorting/v0/ src/spyglass/spikesorting/v1/` is empty — no v0/v1 source touched.
- Sanity: `git diff` against any Phase 1–4 table `definition` strings is empty — zero-migration policy honored.
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
