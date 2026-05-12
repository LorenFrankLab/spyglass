# Phase 5 — UX overhaul: pipeline orchestrator, FigPack, notebook rewrite, v1 sunset roadmap

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#run_v3_pipeline-orchestrator)

The capstone phase. Adds the `run_v3_pipeline()` convenience function (35-cell notebook → 8-cell notebook), Pydantic-validated parameter presets, FigPack curation as an alternate UI path, the full v3 notebook walkthrough, and the documented sunset criteria for v1.

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

- **Implement `pipeline.py`** per [designs.md § `run_v3_pipeline()` Orchestrator](designs.md#run_v3_pipeline-orchestrator). Specific:
  - `run_v3_pipeline(nwb_file_name, sort_group_id, interval_list_name, team_name, preset, skip_artifact=False, auto_curate=True) -> dict` — the orchestrator. Returns a manifest dict containing every `(stage, key)` it inserted/populated plus final `merge_id`.
  - `PRESETS: dict[str, PresetSchema]` — named bundles of parameter rows. Default presets:
    - `franklab_tetrode_mountainsort5`
    - `franklab_tetrode_clusterless`
    - `franklab_probe_kilosort4`
    - `clusterless_threshold_default`
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

- **Implement `figpack_curation.py`** per [designs.md § FigPackCuration](designs.md#figpackcuration):
  - `FigPackCurationSelection` Manual + `FigPackCuration` Computed.
  - `make()` uses `figpack.spike_sorting.build_curation_view(analyzer)` (verify exact import path against current FigPack release before implementation). Publishes view, stores returned URI.
  - `FigPackCuration.fetch_curation_from_uri(uri) -> tuple[dict, list]` — round-trip labels + merge_groups from FigPack back into v3.
  - Optional dependency: gate `figpack` import with a clear install message if absent.

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
  - Mark `docs/src/Pipelines/SpikeSorting/v1.md` (or whatever path the v1 docs live at) with a "legacy" banner — but keep all v1 docs accessible.
  - Add a new docs page: `docs/src/Pipelines/SpikeSorting/v1-to-v3-migration.md` — for users with v1 sorts wondering what changes for new sorts (TL;DR: v1 stays accessible; new sorts go to v3 via `run_v3_pipeline`).
  - CHANGELOG.md: "v3 spike sorting is the recommended path. `run_v3_pipeline()` reduces typical sort setup to a single function call. FigPack curation added. v1 remains supported indefinitely (no removal in this release)."

- **v1 sunset trigger documentation** (NOT v1 removal — documentation of WHEN v1 might be removed):
  - New section in `docs/src/Pipelines/SpikeSorting/v1.md` titled "Sunset criteria": "v1 source will be deprecated when (a) v3 has been the default in docs for ≥6 months, (b) no v1 populate calls have been observed in production logs for ≥3 months (measured via a query in `spyglass.spikesorting.v1.SpikeSortingRecording.populate()` log mining), and (c) all lab production data has either been re-sorted in v3 or is read-only legacy archive."
  - Actual v1 removal is **a future plan**, NOT in this scope. The Phase 5 PR explicitly documents this.

- **End-to-end integration test** `tests/spikesorting/v3/test_run_pipeline.py`:
  - `test_run_v3_pipeline_minirec_clusterless` — calls `run_v3_pipeline(...)`, asserts manifest has all 5 stages + valid merge_id, downstream `SpikeSortingOutput.get_spike_times(...)` returns the same units that direct v3 path would (Phase 1's parity test, but through the orchestrator).
  - `test_run_v3_pipeline_idempotent` — call `run_v3_pipeline(...)` twice with identical args; second call returns the same manifest (no duplicate inserts).
  - `test_preset_validation_catches_missing_lookup_rows` — define a preset referencing a nonexistent param name; `register_preset` raises with a clear "row 'foo' not found in PreprocessingParameters" message.

- **Notebook smoke test**: `tests/notebooks/test_spike_sorting_v3_notebook.py` — uses `jupytext` (already a docs optional dep at [pyproject.toml](pyproject.toml)) to execute `notebooks/13_Spike_SortingV3.ipynb` cell-by-cell against the `minirec` fixture. Marked slow.

## Deliberately not in this phase

- **No v1 source removal.** Documented sunset criteria only.
- **No FigURL deprecation in v3.** v3 ships with FigPack ADDED, not FigURL replaced. FigURL stays usable.
- **No multi-day chronic support.** Phase 3's documented limitation stays.
- **No DeepUnitMatch.** Phase 4 still ships only UnitMatch.
- **No automated metrics on lab production data.** The "no v1 populates in 3 months" sunset criterion is documented but not instrumented in this PR.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_preset_schema_validation` | `PresetSchema(preproc_params_name="bogus", ...)` succeeds at construction but `register_preset(name, dict)` raises because the Lookup row doesn't exist. Valid preset registers cleanly. |
| `test_run_v3_pipeline_minirec_clusterless` (slow, integration) | Single call produces a valid merge_id; spike times match Phase 1 parity baseline. |
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
- The "Deliberately not in this phase" list is honored — v1 is NOT removed in this PR.
- Validation slice tests pass; slow / integration tests are marked.
- `notebooks/13_Spike_SortingV3.ipynb` is ≤10 code cells (verify by running `jq '.cells | map(select(.cell_type == "code")) | length' notebooks/13_Spike_SortingV3.ipynb`).
- `run_v3_pipeline()` is idempotent (the manifest comparison test passes).
- All docs tasks landed: v3.md banner, README snippet, migration page, sunset criteria documented in v1.md.
- CHANGELOG.md mentions Phase 5 deliverables.
- Sanity: `git diff src/spyglass/spikesorting/v0/ src/spyglass/spikesorting/v1/` is empty — no v0/v1 source removed.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
