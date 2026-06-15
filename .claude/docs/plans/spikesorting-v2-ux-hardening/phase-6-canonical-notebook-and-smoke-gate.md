# Phase 6 — Canonical notebook + UX smoke-test release gate

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The capstone. Ship one runnable single-session notebook that walks the whole happy path using the surfaces from Phases 1–4, and an automated **UX smoke test** that proves "a scientist's first hour works" — defaults → sort group → preflight → pipeline → curation summary → downstream fetch — on the smoke fixture in CI. This phase imports `describe_presets`, `preflight_v2_pipeline`, the enriched manifest, and the curation wrappers, so it lands **last**.

**Contracts referenced:**

- [Pipeline manifest schema](shared-contracts.md#pipeline-manifest-schema) — the notebook prints it; the smoke test asserts the stable + additive keys.
- [Preflight report schema](shared-contracts.md#preflight-report-schema) — the notebook shows a passing report; the smoke test asserts `report.ok`.

**Inputs to read first:**

- [pipeline.py](../../../../src/spyglass/spikesorting/v2/pipeline.py) — `describe_presets` (Phase 1), `preflight_v2_pipeline` + `preflight=True` (Phase 2), enriched manifest (Phase 3).
- [curation.py](../../../../src/spyglass/spikesorting/v2/curation.py) — `create_initial_curation` / `propose_merge_curation` / `create_merged_curation` / `summarize_curation` (Phase 4).
- [__init__.py:14-44](../../../../src/spyglass/spikesorting/v2/__init__.py#L14-L44) — `initialize_v2_defaults()`, step 1 of the notebook.
- [tests/spikesorting/v2/conftest.py:181-273](../../../../tests/spikesorting/v2/conftest.py#L181-L273) — the existing populate setup the smoke test mirrors (session ingest, team, sort group, defaults).
- The master-roadmap [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) notebook task — confirm the target filename `notebooks/10_Spike_SortingV2.ipynb` so Phase 5 *extends* this notebook rather than creating a competing one.

## Tasks

- **Write `notebooks/10_Spike_SortingV2.ipynb`** — minimal single-session walkthrough, **≤10 code cells**. Author it via `jupytext` paired mode (see the `jupyter-notebook-editor` workflow) so it's diffable. Cells:
  1. Imports + DataJoint config.
  2. `initialize_v2_defaults()`; insert the `LabTeam` row; (note that the session is assumed already ingested via `insert_sessions` — link to the ingestion notebook rather than re-teaching it).
  3. `SortGroupV2.set_group_by_shank(nwb_file_name=...)`.
  4. `describe_presets()` — show the catalog so the user picks a preset by understanding, not guessing.
  5. `report = preflight_v2_pipeline(...)`; print `report` — show a green check before committing minutes to populate. Mention `preflight=False` exists.
  6. `manifest = run_v2_pipeline(..., preset="franklab_tetrode_mountainsort5")`.
  7. Print `manifest` — call out `merge_id` (the downstream key), `n_units`, the `*_status` (computed vs reused), and `stage_seconds`.
  8. `CurationV2.summarize_curation(manifest)` — Phase 4 explicitly normalizes a full manifest to `{"sorting_id": ..., "curation_id": ...}` before querying, so the notebook can pass the pipeline result directly. Show units/labels/merge state; mention `create_initial_curation` / `propose_merge_curation` / `create_merged_curation` for manual curation.
  9. Downstream: resolve the sort through `SpikeSortingOutput` and fetch spike times (the "it actually produced usable data" payoff). Optionally insert into `SortedSpikesGroup`.
  - Keep markdown cells for narration; the **≤10 limit is code cells only**. Include the zero-unit and rerun behaviors as short markdown notes (per the UX plan), not extra code cells.
- **Docs quickstart:** add a "Run your first single-session sort" section to `docs/src/Features/SpikeSortingV2.md` mirroring the notebook's path in prose (defaults → sort group → preflight → pipeline → summary → fetch). Do **not** promote v2 to "recommended for new work" or rewrite the README quick-example — that promotion is owned by master-roadmap Phase 5. CHANGELOG entry: the v2 single-session notebook + quickstart.
- **Write the UX smoke test** `tests/spikesorting/v2/test_ux_smoke.py` — the release gate, marked `slow` + `integration`. It runs the *exact* first-hour path programmatically against the smoke fixture (not the notebook — that's the next task), asserting at each step:
  1. `initialize_v2_defaults()` is idempotent (callable twice, no error).
  2. `describe_presets()` returns the catalog including the preset under test.
  3. `preflight_v2_pipeline(...)` returns `report.ok is True` on the configured session, and `report.expected_ids` is populated.
  4. `run_v2_pipeline(..., preflight=True)` returns a manifest with all stable + additive keys; `report.expected_ids["recording_id"]["id"] == manifest["recording_id"]` (preflight predicted the real PK).
  5. `summarize_curation(manifest)` returns a coherent summary; `merge_id` matches the manifest and the same call with a minimal curation key returns the same summary.
  6. The sort is resolvable through `SpikeSortingOutput` and `get_spike_times` (or the documented accessor) returns sane arrays.
  7. A second `run_v2_pipeline(...)` is idempotent (manifest equal modulo `stage_seconds`/`*_status`; `*_status` now `reused`).
- **Notebook execution smoke test** — extend `test_ux_smoke.py` (or add `tests/spikesorting/v2/test_notebooks.py`) to execute `notebooks/10_Spike_SortingV2.ipynb` cell-by-cell against the smoke fixture using `jupytext` (already a docs optional dep), and assert the code-cell count ≤10 (`jq '.cells | map(select(.cell_type=="code")) | length' notebooks/10_Spike_SortingV2.ipynb`). Marked `slow`. Skip cleanly (not fail) when the smoke fixture is absent locally; CI provides it.

## Deliberately not in this phase

- **No FigPack cells, no cross-session/concat cells.** Master-roadmap Phase 5 *adds* those to this same notebook. This phase ships the single-session core only. (Leave a short markdown placeholder note "interactive curation: coming via FigPack" so Phase 5 has an obvious insertion point — optional.)
- **No `register_preset`, no KS4 preset in the notebook** — only the three shipping presets (Phase 1 scope).
- **No README promotion / "recommended for new work" banner** — master-roadmap Phase 5 owns the v1-vs-v2 positioning docs.
- **No new fixture** — reuse the smoke fixture. If it's too quiet to produce units reliably, use the same preset/params the existing `populated_sorting` fixture uses, which is known to produce a sort.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_ux_smoke_first_hour` (slow, integration) | The full defaults→sort-group→preflight→pipeline→summary→fetch path runs end-to-end with the asserts listed in the task; this is the "does a scientist's first hour work?" gate. |
| `test_ux_smoke_preflight_predicts_ids` (slow, integration) | `preflight` `expected_ids` match the recording/artifact/sorting PKs the manifest returns (ties Phases 2+3 together at the user surface without pretending preflight predicts `curation_id`). |
| `test_ux_smoke_idempotent` (slow, integration) | Re-running the path returns an equal manifest (modulo timing/status) and inserts no duplicate rows. |
| `test_user_notebook_executes` (slow) | `notebooks/10_Spike_SortingV2.ipynb` executes cell-by-cell with no errors against the smoke fixture. |
| `test_user_notebook_cell_budget` | The notebook has ≤10 code cells. (Fast; pure file inspection — does not need the DB.) |

## Fixtures

Reuse `mearec_polymer_smoke` and the populate-setup pattern from [tests/spikesorting/v2/conftest.py:181-273](../../../../tests/spikesorting/v2/conftest.py#L181-L273). The smoke test should set up its *own* clean session (so the idempotency/`computed`→`reused` assertions are deterministic) rather than depending on the package-scoped `populated_sorting`, which is pre-populated. Real-data smoke beyond the MEArec fixture is optional and gated (per the master plan's production-smoke policy) — not required for this gate.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- The notebook is ≤10 code cells (`jq` check) and executes clean against the smoke fixture.
- The notebook uses the Phase 1–4 surfaces (`describe_presets`, `preflight_v2_pipeline`, the enriched manifest, `summarize_curation`) — it is the real first-hour path, not a stripped demo.
- The UX smoke test asserts the *whole* chain including the preflight→manifest ID round-trip and idempotency, and is marked slow/integration.
- It is the SAME notebook filename master-roadmap Phase 5 will extend (no second competing notebook); the "Deliberately not in this phase" boundaries are honored (no FigPack, no README promotion).
- Docs quickstart added; CHANGELOG updated; nothing references this plan or its phases in notebook/test/doc names.
- The smoke test sets up its own session for deterministic idempotency assertions and skips cleanly when the fixture is absent.
