# Phase 5 — UX overhaul: pipeline orchestrator, FigPack, notebook extension

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#run_v2_pipeline-orchestrator)

The capstone phase. Extends the current canonical single-session notebook
(`notebooks/10_Spike_SortingV2.ipynb`, created by the UX-hardening addendum)
with the remaining user-facing v2 surfaces: full `run_v2_pipeline()`
orchestration, Pydantic-validated parameter presets, FigPack as the v2 curation
UI, and v1/v2 path-selection docs. It is split into **Phase 5a FigPack
feasibility** and **Phase 5b implementation/docs** so the unknown FigPack
curation-state API is verified before any DataJoint table is written. **v1
source is NOT removed by this plan**; existing v0/v1 rows continue to coexist
with v2. Active v0/v1 runtime workflows use the legacy SI 0.99 environment
unless Phase 0c explicitly ports a narrow shim.

## Executor Checklist

- Complete Phase 5a first: verify the current FigPack spike-sorting API and edited-curation state round trip, replace all `PHASE5A_CONTRACT_STUB` markers, and stop/escalate if the round trip cannot be made to work.
- Extend `pipeline.py` with full presets, auto-curation, concat routing, FigPack routing, and separate `run_v2_unit_match()`.
- Implement preset validation and optional FigPack dependency gates.
- Implement `FigPackCurationSelection`, `FigPackCuration`, URI generation, and curation round-trip only against the verified FigPack API.
- Extend the canonical v2 notebook/docs so v2 is easier to use while v0/v1
  remain available.
- Run the source documentation-density pass (trim over-commenting, promote prose into rendered NumPy sections), coordinated with [DOCSTRING-AUDIT.md](DOCSTRING-AUDIT.md) so the two passes don't fight.
- Run end-to-end notebook/orchestrator tests in the isolated database. Production-connected real-data smoke is optional and must use the explicit production-smoke gate with test schemas/temp output directories.
- Run the Phase 5 validation goals plus `code_graph.py describe/path` for FigPack tables.

**Inputs to read first:**

- All files implemented in Phases 1–4. Phase 5 adds only its own FigPack tables (`FigPackCurationSelection`, `FigPackCuration`) and preset registrations; it must not alter any Phase 1–4 table definitions.
- [notebooks/10_Spike_SortingV2.ipynb](../../../../notebooks/10_Spike_SortingV2.ipynb) — the canonical v2 single-session notebook to extend; do not create a competing single-session notebook.
- [notebooks/10_Spike_SortingV1.ipynb](../../../../notebooks/10_Spike_SortingV1.ipynb) — the v1 notebook whose long manual workflow the v2 notebook replaces.
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

### Phase 5a — FigPack feasibility spike

- **PHASE5A_CONTRACT_STUB — no FigPack DataJoint implementation while this marker remains.** Phase 5a must replace this marker in `phase-5-ux-overhaul.md`, `designs.md`, and `appendix.md` with observed API details before Phase 5b starts. Reviewers should use `rg "PHASE5A_CONTRACT_STUB" .claude/docs/plans/spikesorting-v2` as the stop/go check.

- **Verify FigPack outside DataJoint first.** Use the isolated v2 curation environment and one existing v2 `SortingAnalyzer` fixture. Deliverables:
  1. Confirm the actual installable package set. Current upstream uses the core `figpack` package plus a spike-sorting extension package (`figpack-spike-sorting` on PyPI, imported as `figpack_spike_sorting` in the upstream repository); do not assume `figpack` alone provides spike-sorting views.
  2. Verify the current spike-sorting extension API. Do not assume stale examples such as `figpack.spike_sorting.build_curation_view(analyzer)` or `view.publish()` exist; pin the real import path, view-construction API, upload/publish method, and local/offline display method in this plan before writing the DataJoint table.
  3. Test on a single example: build a curation view from a v2 SortingAnalyzer end-to-end and publish/upload or save it through the verified API.
  4. Round-trip a known labels dict and merge-groups representation back through the verified API or documented state file. If FigPack can display a curation view but cannot persist edited curation state in a retrievable form, stop and escalate before schema finalization.
  5. Record the verified FigPack and `figpack-spike-sorting` versions, exact import statements, minimal working code snippet, upload/publish behavior, and curation-state retrieval path in `tests/spikesorting/v2/resolver/figpack-runtime.md`.
  6. Replace the `PHASE5A_CONTRACT_STUB` markers in this file, `designs.md`, and `appendix.md` with the observed contract. Include the exact helper shape that `FigPackCuration.make()` and `FigPackCuration.fetch_curation_from_uri()` will wrap.

  **If FigPack is not usable at implementation time**: STOP Phase 5 and escalate to the project owner. The plan does not silently fall back to FigURL. Possible resolutions (decided by project owner, not the implementer): wait for FigPack release; pin to a specific FigPack commit; add a contribution to upstream FigPack; or remove FigPack from this release scope.

### Phase 5b — implementation and docs

- **Do not start Phase 5b while `PHASE5A_CONTRACT_STUB` remains anywhere under `.claude/docs/plans/spikesorting-v2/`.**

- **EXTEND `pipeline.py`** (Phase 1 shipped a minimal version with 3 presets covering recording → artifact → sorting → initial curation). Phase 5 adds the missing stages and broadens the preset set per [designs.md § `run_v2_pipeline()` Orchestrator](designs.md#run_v2_pipeline-orchestrator):
  - **Add `auto_curate=False` parameter** — wires up Phase 2's `AnalyzerCuration` stage and materialization only when the caller opts in. The default remains initial-curation-only so a convenience call does not silently commit suggested labels/merges into a new `CurationV2`. If `auto_curate=True`, the returned manifest must include both the `AnalyzerCuration` suggestion row and the materialized child `CurationV2` row.
  - **Add `require_units=False` parameter (review-fix C5)** — controls zero-unit behavior. **Default (`require_units=False`): a zero-unit sort writes an empty-but-real curation + merge row and returns a FULL manifest** with real `curation_id` / `merge_id` and `n_units=0` plus a loud warning — zero units is a legitimate quiet-shank result, not an error, and the empty-but-real row stays merge-keyable like any other sort. **`require_units=True` raises `ZeroUnitSortError`** instead. This is the orchestrator surface of the C5 contract (graceful-by-default, opt-in to raise) — do NOT make raising the default, and do NOT return a `None` `merge_id` (the earlier partial-manifest design was superseded once the empty-`NumpySorting` read path made the row buildable; see `shared-contracts.md § Empty / NaN / Boundary Invariants`).
  - **Add `concat_session_group_owner` + `concat_session_group_name` parameters (optional pair)** — when set together, the orchestrator routes through Phase 3's `ConcatenatedRecording` instead of `Recording`. They are mutually exclusive with the single-session inputs (`nwb_file_name`, `sort_group_id`, `interval_list_name`, `team_name`) and are only for same-day / explicit-opt-in concatenated sorting. The owner field is required because `SessionGroup` names are team-namespaced. In concat mode, `team_name` is invalid; member teams come from `SessionGroup.Member.team_name`. A `SessionGroup.Member` is a sorting member tuple, not a whole NWB/day, so one day can contribute multiple interval members and long recordings split across NWB files can contribute multiple members. Concat-mode manifests contain `concat_recording` in place of `recording`; the concat `SortingSelection` simply has **no `ArtifactDetectionSource` row** (post-review-fixes — there is no `artifact_detection_id=None` to set), and the manifest omits `artifact_detection` until concat-wide artifact semantics land. These parameters are deliberately NOT reused for UnitMatch.
  - **Add `run_v2_unit_match()` helper** — a separate convenience function for Phase 4's sort-then-match path. Signature: `run_v2_unit_match(session_group_owner, session_group_name, matcher_params_name="unitmatch_default", curation_choices=None) -> dict`. It requires explicit `curation_choices` keyed by `SessionGroup.member_index`, calls `UnitMatchSelection.insert_selection(..., curation_choices=...)`, populates `UnitMatch` and `TrackedUnit`, and returns a manifest with `unitmatch_id`. It never auto-selects "latest" curations. Keeping this separate prevents the concat-sorting workflow from being confused with the per-member curation workflow required by UnitMatch.
  - **Add `figpack=False` parameter (optional)** — wires up the FigPack curation stage below. `figpack=True` publishes/builds the curation view, returns the URI in the manifest, and exits; it does not block waiting for interactive edits. Users call `FigPackCuration.fetch_curation_from_uri()` after labeling.
  - **Expand `PRESETS`** to include Phase 5's full set:
    - `franklab_tetrode_mountainsort4`, `franklab_tetrode_mountainsort5`, `franklab_tetrode_clusterless_thresholder` (carried over from Phase 1)
    - `franklab_probe_kilosort4` (new)
    - `franklab_tetrode_mountainsort5_sameday_concat` (new — uses SessionGroup + ConcatenatedRecording for same-day chronic; sets `motion_correction_params_name="auto_default"`)
  - Built-in preset names follow `{lab}_{probe_or_modality}_{sorter_or_workflow}` plus an optional topology suffix. External labs should use the same pattern, e.g. `berke_intan_mountainsort5` or `franklab_tetrode_mountainsort5_sameday_concat`.
  - `register_preset(name, preset_dict)` — public API for labs to add custom presets without modifying v2 source.
  - `clone_preset(base_name, new_name, **overrides)` — the "tune one knob" path that closes the flexibility cliff (a user wanting a single non-default value should not hand-write a full Pydantic params dict). It resolves `base_name` to its preprocessing / artifact-detection / sorter parameter rows, applies `**overrides` (dotted keys into the relevant stage blob, e.g. `bandpass_filter.freq_min=700`), inserts the derived parameter rows under fresh names, and registers `new_name` via `register_preset`. Binding behavior: (a) **validates** every derived row against its Pydantic schema before inserting (a bad override raises the same teaching error as a direct insert); (b) **does not mutate** the base preset or its parameter rows — it only adds new rows + a new preset registration; (c) **refuses ambiguous/duplicate names** — raises if `new_name` already exists in the preset registry, or if a derived parameter-row name collides with an existing row whose content differs (mirror the `DuplicateParameterContentError` opt-in rather than silently forking provenance). An override that targets a key not present in the base stage schema raises rather than silently adding it. Pairs with `describe_pipeline_preset(name)` (the pre-Phase-5 helper that unpacks a preset's full validated values) so a user can inspect a base preset, clone it with one change, and inspect the result.
  - Each preset must reference Lookup-table row names that ALREADY EXIST. Phase 5 inserts these baseline rows via `insert_default()` calls in `__init__.py`.
  - **Clusterless preset docs (post-review-fixes T3):** the `franklab_tetrode_clusterless_thresholder` preset points at the `default` clusterless `SorterParameters` row, which sets `threshold_unit="uv"` (100 µV — v1 behavior). Preset/notebook docs should state the clusterless **default threshold is in µV**, and that the synthetic `smoke_clusterless_5uv` row uses `threshold_unit="mad"` (MAD multiplier). Do not describe the clusterless threshold ambiguously as "5σ".

- **Add the final `run_v2_pipeline()` docstring** from [designs.md § `run_v2_pipeline()` Orchestrator](designs.md#run_v2_pipeline-orchestrator). The docstring must include `Parameters`, `Returns`, and `Raises` sections; state that exactly one input mode is required; list the single-session fields and concat fields; say concat mode rejects `team_name`; and quote the `PipelineInputError` message for mixed/missing/partial modes.

- **Implement `_params/preset.py`** Pydantic model. Required fields: `preprocessing_params_name`, `sorter`, `sorter_params_name`, `metric_params_name`, and `auto_curation_rules_name`. Optional fields: `artifact_detection_params_name` (default None), `motion_correction_params_name`, and `description`. Binding behavior: extra fields are forbidden; `sorter` is validated against SI's available sorters; preset registration validates that every referenced Lookup row exists and raises clearly if a parameter set is missing. `motion_correction_params_name` is optional for ordinary single-session presets and required only for presets intended for concat session groups. **`artifact_detection_params_name` is optional and MUST be None for concat presets** — concat sorts run no artifact detection (no `ArtifactDetectionSource` row), so the orchestrator forbids consuming `artifact_detection_params_name` in concat mode: a concat preset (or concat-mode call) carrying a non-None `artifact_detection_params_name` raises a clear preset/`PipelineInputError` rather than silently ignoring it. Single-session presets that omit `artifact_detection_params_name` simply skip artifact detection (equivalent to `skip_artifact=True`). This catches the typo-at-populate failure mode entirely. **Schema-version expectations (post-review-fixes):** the referenced `PreprocessingParameters` rows validate at `PreprocessingParamsSchema` v3 (bandpass Optional, whiten default None) and clusterless `SorterParameters` rows at `ClusterlessThresholderSchema` v4 (with `threshold_unit`). Preset validation must accept these current versions; a preset referencing a row pinned at a stale `params_schema_version` should fail the version-match guard with a clear message rather than silently validating.

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

- **Extend the canonical v2 notebook**: update
  `notebooks/10_Spike_SortingV2.ipynb`. Do **not** create another competing
  single-session walkthrough. Keep the main single-session path ≤10 code cells
  by consolidating cells as needed:
  1. Imports + DataJoint config (1 cell).
  2. Parameters for an already-ingested session (1 cell).
  3. Initialize defaults, insert team, create/inspect sort groups, including
     `describe_sort_groups` and `plot_sort_group_geometry` (1 cell).
  4. Choose/inspect a preset (1 cell).
  5. Preflight and run `run_v2_pipeline(..., pipeline_preset="franklab_tetrode_mountainsort5")` (1-2 cells).
  6. Inspect the manifest and curation summary (1 cell).
  7. Fetch downstream outputs by `merge_id` (1 cell).
  8. Optional: launch FigPack curation and retrieve labels if this can be kept
     inside the same ≤10-code-cell budget; otherwise link to the FigPack docs
     or a dedicated curation snippet rather than bloating the first-hour path.
  9. Optional: insert into `SortedSpikesGroup` for downstream analysis if it
     fits the same budget.

  Compare with [notebooks/10_Spike_SortingV1.ipynb](../../../../notebooks/10_Spike_SortingV1.ipynb) (35 code cells). Phase 5 success metric: ≤10 cells.

- **Cross-session notebook**: new file `notebooks/14_Spike_Sorting_CrossSession.ipynb`. Walks through:
  1. Build a `SessionGroup` for 3 same-day sessions.
  2. Run `run_v2_pipeline()` for each session individually.
  3. Run `run_v2_unit_match(session_group_owner, session_group_name, curation_choices=...)` across the group.
  4. Inspect `TrackedUnit` membership.
  5. Decoding with `TrackedUnit`-indexed spikes (link to downstream notebook).

- **Documentation overhaul**:
  - Promote `docs/src/Features/SpikeSortingV2.md` from "new pipeline" to "recommended for new work". Add a top banner.
  - Update root README "Quick example" snippet to use `run_v2_pipeline()`.
  - Keep existing v0/v1 docs, API references, and notebooks accessible and live; do NOT remove v1 docs. Make the environment boundary explicit: active v0/v1 population and curation workflows use the legacy SI 0.99 environment unless Phase 0c explicitly ported that surface.
  - Add a new docs page: `docs/src/Features/ChoosingSpikeSortingV1VsV2.md` — for users deciding which path to use. TL;DR: v2 is recommended for new work under the SI 0.104 environment; active v1 runtime workflows require the legacy SI 0.99 environment unless Phase 0c explicitly ported that surface. Existing v1 sorts stay queryable through v1 where possible.
  - In `docs/src/Features/ChoosingSpikeSortingV1VsV2.md`, include the import path explicitly: external or ground-truth NWB Units still use the existing `ImportedSpikeSorting` workflow and appear in `SpikeSortingOutput.ImportedSpikeSorting`; they are not reinserted as `CurationV2` rows.
  - Add CHANGELOG entry noting the full v2 orchestrator, FigPack curation, canonical notebook extension, v2's recommended-for-new-work designation, v0/v1 continued support, and the opt-in `auto_curate=True` path.

- **Source documentation-density pass (readability follow-up).** The v2 source is ~46% documentation (≈36% docstring lines + ≈10% `#` comment lines, measured across [`src/spyglass/spikesorting/v2/`](../../../../src/spyglass/spikesorting/v2/)). That density was the right investment during the v1→v2 port, but it now hurts scannability: several functions wrap a handful of logic lines in multi-paragraph inline rationale (e.g. [recording.py:1202-1258](../../../../src/spyglass/spikesorting/v2/recording.py#L1202-L1258) — ~40 comment lines around ~15 lines of code). Trim for signal without losing the standalone "why". This is a comments/docstrings-only pass — **no logic changes**.
  - **Decoupled from v1 removal — by necessity.** This plan keeps v0/v1 in-tree indefinitely (see "Deliberately not in this phase"), so the v1-divergence commentary is NOT deleted as dead archaeology — v1 still exists for readers to compare against. Instead, **condense and relocate**: leave a one-line pointer at the call site (e.g. "v1 dropped the final sample here; see [v1-v2-divergences.md](v1-v2-divergences.md)") and move the multi-paragraph "v1 did X / v2 does Y" reasoning into the function or module docstring, or into [v1-v2-divergences.md](v1-v2-divergences.md). The inline comment should say what a reader needs at that line, not re-derive the port history.
  - **Delete pure restatement.** Comments that paraphrase the next line of code with no added "why" are removed; genuinely non-obvious numerical/algorithmic rationale stays (Nyquist-check placement, `searchsorted`-vs-affine frame mapping, the FP-rounding clamp). When in doubt, keep.
  - **Coordinate with [DOCSTRING-AUDIT.md](DOCSTRING-AUDIT.md); do not fight it.** That audit found the *opposite* gap — contract details living in prose instead of rendered NumPy `Parameters`/`Returns`/`Raises` sections. This pass must **promote** that prose into sections (closing the audit's Medium Patterns 1–5), not strip it. Net effect: fewer freeform comment paragraphs, more structured docstring sections — same or better information, more scannable. Run the two passes together to avoid double-touching the same files.
  - **Named hotspots.** The tri-part `make_*` compute bodies and the longest service functions — `detect_artifacts` ([_artifact_intervals.py](../../../../src/spyglass/spikesorting/v2/_artifact_intervals.py)), `run_clusterless_thresholder` / `run_si_sorter` ([_sorting_dispatch.py](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py)), `build_analyzer` ([_sorting_analyzer.py](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py)), `apply_pre_motion_preprocessing` ([_recording_preprocessing.py](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py)) — carry the densest comment-to-code ratio and are the primary targets.
  - **Measure, don't threshold.** Re-measure the documentation ratio after the pass and record before/after in the PR description. There is NO hard target — over-trimming a numerically subtle pipeline is worse than over-documenting it; the goal is scannability, not a percentage.
  - **Scope guard.** Source comments/docstrings only. No logic edits, and no `definition`-string column-comment changes (zero-migration policy; see [DOCSTRING-AUDIT.md](DOCSTRING-AUDIT.md) "Excluded from the fix pass"). The density pass diff must show only comment/docstring lines.

- **No v1 source-removal criteria.** Per the resolved design decision in [overview.md](overview.md), v0 and v1 stay in-tree indefinitely. Phase 5 documents that v2 is the recommended path for new sorts under SI 0.104 and that active v1 runtime workflows use the legacy SI 0.99 environment unless explicitly ported; v1 docs stay live with that environment boundary.

- **End-to-end integration test** `tests/spikesorting/v2/test_run_pipeline.py`:
  - `test_run_v2_pipeline_minirec_clusterless` — calls `run_v2_pipeline(...)` with default `auto_curate=False`, asserts manifest has the expected single-session stages (`recording`, `artifact_detection`, `sorting`, `initial_curation`) plus valid `merge_id`; downstream `SpikeSortingOutput.get_spike_times(...)` returns sane arrays. This is a plumbing/integration guard only; minirec is not a sort-correctness or parity oracle.
  - `test_run_v2_pipeline_auto_curate_opt_in` — calls `run_v2_pipeline(..., auto_curate=True)` and asserts the manifest adds `auto_curation` plus a materialized child curation row.
  - `test_run_v2_pipeline_concat_manifest` — calls concat mode on a small same-day `SessionGroup`; asserts the manifest contains `concat_recording`, `sorting`, `initial_curation`, and optional `auto_curation`, and does not contain `artifact_detection`. Also asserts the concat `SortingSelection` has no `ArtifactDetectionSource` row (not an `artifact_detection_id=None` column).
  - `test_run_v2_pipeline_zero_units_manifest` (review-fix C5) — on a zero-unit sort, the default (`require_units=False`) returns a full manifest with real `curation_id` / `merge_id` and `n_units=0` and emits a warning; `require_units=True` raises `ZeroUnitSortError`. Asserts the default does NOT raise, the manifest is well-formed, and the empty-but-real merge row is resolvable through `SpikeSortingOutput`.
  - `test_run_v2_pipeline_idempotent` — call `run_v2_pipeline(...)` twice with identical args; second call returns the same manifest (no duplicate inserts).
  - `test_run_v2_pipeline_rejects_mixed_single_and_concat_inputs` — passing both single-session inputs and `concat_session_group_owner` / `concat_session_group_name` raises a clear error before any insert.
  - `test_run_v2_pipeline_requires_concat_owner_and_name` — passing only one concat session-group field raises before insert.
  - `test_run_v2_unit_match_requires_explicit_curations` — `run_v2_unit_match(..., curation_choices=None)` raises and never auto-pins latest curations.
  - `test_preset_validation_catches_missing_lookup_rows` — define a preset referencing a nonexistent param name; `register_preset` raises with a clear "row 'foo' not found in PreprocessingParameters" message.

- **Notebook smoke test**: extend the existing UX smoke coverage or add
  `tests/spikesorting/v2/test_notebooks.py` — uses `jupytext` (already a docs
  optional dep at [pyproject.toml](../../../../pyproject.toml)) to execute
  `notebooks/10_Spike_SortingV2.ipynb` cell-by-cell against the `minirec` or
  smoke fixture. Marked slow.

- **Document GitHub-hosted curation JSON ingress workflow (audit followup NB-N5 from Phase 1b sweep).** v1 supports loading curation labels/merge_groups from a GitHub-hosted `curation.json` via `FigURLCuration.generate_curation_uri` and `gh://.../curation.json` URIs (v1 notebook cells `924cdfce`, `b2e9b018`). Phase 5's FigPack design replaces FigURL but does not document the equivalent v2 workflow for loading external curation JSONs. Common Frank-lab pattern for sharing manual curations between users.

  Pick one of the two paths (implementer's choice; Phase 5a feasibility evidence may inform):
  - (a) **Extend `CurationV2.insert_curation`** to accept a `curation_uri: str | None = None` kwarg that, when set, reads `labels` and `merge_groups` from a JSON at the URI (supports `gh://`, `file://`, and `http(s)://`). The dict keys map directly to `insert_curation`'s `labels` and `merge_groups` args.
  - (b) **Document the manual "read JSON → dict" pattern** in the Phase 5 notebook + `SpikeSortingV2.md` so users know v2's curation surface is "Python dict in" rather than "URI in", and example code shows how to convert a v1-shaped curation.json to v2's `insert_curation(labels=..., merge_groups=...)` call.

  Both paths preserve the v1 workflow; (a) is closer to v1's API surface, (b) is simpler to implement.

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
4. **Single-session vs concat input mode**: mixing single-session inputs with concat session-group inputs raises before any insert; supplying only one of `concat_session_group_owner` / `concat_session_group_name` raises before any insert; concat happy path returns a manifest with `concat_recording` and no `artifact_detection` (the concat `SortingSelection` simply has no `ArtifactDetectionSource` row).
4b. **Zero-unit manifest (review-fix C5)**: default `require_units=False` writes an empty-but-real curation + merge row and returns a full manifest (real `curation_id` / `merge_id`, `n_units=0`) with a warning and does not raise; `require_units=True` raises `ZeroUnitSortError`.
5. **`run_v2_unit_match`**: requires explicit `curation_choices` (raises if missing — never auto-picks "latest"); two calls with identical args return the same `unitmatch_id`.
6. **FigPack feasibility-gated** (slow, integration, optional): Phase 5a has replaced all `PHASE5A_CONTRACT_STUB` markers and recorded `figpack-runtime.md`; with the `spikesorting-v2-curation` extra installed, `FigPackCuration.populate(key)` returns a non-empty URI or verified local artifact reference; zero-unit curation publishes an empty view or raises a clear `EmptySortingError` (never missing-column `KeyError`); published labels round-trip via `fetch_curation_from_uri`.
7. **Notebook smoke** (slow, integration): `jupytext` executes `notebooks/10_Spike_SortingV2.ipynb` against `minirec` or the smoke fixture with no errors; programmatic check confirms code-cell count ≤10. Cross-session notebook executes when a multi-session fixture is available (optional).
8. **Source documentation-density pass (readability)**: the density-pass diff touches only comment/docstring lines (no logic change — verify the diff carries no executable-line edits); the v2 documentation ratio is re-measured and before/after recorded in the PR; the DOCSTRING-AUDIT.md Medium Patterns 1–5 are closed by promoting prose into rendered NumPy `Parameters`/`Returns`/`Raises` sections rather than by deletion; v1-divergence rationale is condensed/relocated (not deleted), since v0/v1 stay in-tree. No `definition`-string column comments are edited.

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
test -z "$(rg 'PHASE5A_CONTRACT_STUB' .claude/docs/plans/spikesorting-v2 || true)"

pytest tests/spikesorting/v2/test_run_pipeline.py -q
pytest tests/spikesorting/v2/test_notebooks.py -q
test "$(jq '.cells | map(select(.cell_type == "code")) | length' notebooks/10_Spike_SortingV2.ipynb)" -le 10

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

Before opening or reviewing the implementation PR that contains this checkpoint, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — v1 is NOT removed, no v0/v1 schema touched, no existing v2 table altered.
- Validation goals are covered; slow / integration tests are marked.
- `notebooks/10_Spike_SortingV2.ipynb` is ≤10 code cells (verify by running `jq '.cells | map(select(.cell_type == "code")) | length' notebooks/10_Spike_SortingV2.ipynb`).
- `run_v2_pipeline()` is idempotent (the manifest comparison test passes).
- `run_v2_unit_match()` is idempotent by `(session_group_owner, session_group_name, matcher_params_name, curation_set_hash)` and does not conflate UnitMatch with concatenated sorting.
- FigPack feasibility was verified in Phase 5a before implementation began, `tests/spikesorting/v2/resolver/figpack-runtime.md` records the working API and round trip, and no `PHASE5A_CONTRACT_STUB` markers remain. If FigPack proved unusable, the project owner was escalated and no degraded fallback shipped.
- All docs tasks landed: `docs/src/Features/SpikeSortingV2.md` banner, README snippet, `docs/src/Features/ChoosingSpikeSortingV1VsV2.md` decision page.
- Source documentation-density pass landed: the density-pass diff is comment/docstring-only (no logic change); the v2 documentation ratio is re-measured with before/after in the PR; v1-divergence rationale was condensed/relocated (not deleted) given v0/v1 stay in-tree; DOCSTRING-AUDIT.md Medium Patterns were closed by promoting prose into rendered NumPy sections; no `definition`-string column comments were touched.
- CHANGELOG.md mentions the delivered user-facing capabilities (orchestrator, FigPack, canonical notebook extension), the opt-in `auto_curate=True` path, v0/v1 source/query coexistence, and the legacy-environment boundary for active v0/v1 runtime workflows without referencing plan phases.
- Sanity: `git diff src/spyglass/spikesorting/v0/ src/spyglass/spikesorting/v1/` is empty — no v0/v1 source touched.
- Sanity: `git diff` against any Phase 1–4 table `definition` strings is empty — zero-migration policy honored.
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
