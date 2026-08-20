# Phase 4 — Import-light `SpikeSortingOutput.CurationV2` declaration

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

> **SPUN OFF — standalone PR against `spikesorting-v2`, NOT part of the A2 spike**
> (2026-08-19 review: orthogonal to the topology change; ships on its own merits).
> This file is its design spec; it does not enter the spike's parity gates/ledger.

Make the v2 merge-part declaration import-light so `SpikeSortingOutput` declares
the identical schema regardless of dependency skew — no more conditional shape.
**Test hygiene (test-power review):** run the import-light checks in a
**subprocess** (or `importlib.reload` with explicit `sys.modules` eviction of the
whole chain) — a module already imported by an earlier test makes a module-top
`spikeinterface`-raises monkeypatch never re-fire, so the test passes vacuously.
And `test_import_light_schema_identical` must **structurally** compare
headings/FK/dispatch-map between the two import modes, not merely assert import
didn't raise.

**Inputs to read first:**
- [spikesorting_merge.py:37-150](../../../../src/spyglass/spikesorting/spikesorting_merge.py) — `_probe_v2_curation` (37-62), `source_class_dict` (104-110), the conditional `CurationV2` part (143-150), and the `if CurationV2 is None` guards (172, 337, 383, 435).
- [utils.py:14](../../../../src/spyglass/spikesorting/v2/utils.py) — `import spikeinterface as si` (the single module-top heavy import in the declaration chain; only use at [utils.py:756](../../../../src/spyglass/spikesorting/v2/utils.py) `si.get_global_job_kwargs()`).
- [curation.py:101,210](../../../../src/spyglass/spikesorting/v2/curation.py) — `CurationV2.definition` FKs `-> Sorting` / `-> Sorting.Unit` (why the whole `-> Sorting` declaration chain must import light).

**Contracts referenced:** none from the spike — **standalone PR**; it does not
use the spike's `shared-contracts.md` gates or topology ledger.

## Tasks
- Move `import spikeinterface as si` from `utils.py:14` (module top) to the call site at `utils.py:756`. Confirm it is the only module-top heavy (SI/torch) import across the `CurationV2 -> Sorting` declaration chain (`recording/artifact/session_group/_units_nwb/_sorting_dispatch/_sorting_analyzer/utils` + transitively imported `_sorting_units/_sorting_artifact_mask/_recipe_catalog/_curation_transforms/_signal_math/_params/*`). If any other module-top heavy import exists, defer it too.
- Declare `SpikeSortingOutput.CurationV2` **unconditionally**; delete `_probe_v2_curation` (37), the `CurationV2, _v2_import_error = …` module global (62), `_raise_v2_unavailable` (65), and **every** `if CurationV2 is (not) None` guard — the complete list is **97, 109, 143, 172, 337, 383, 435** (Phase 4's own `test_probe_removed` grep must pass).
- **Scope the claim to SI/torch (do not overstate).** Import-light removes the SI/torch-skew failure only. The `CurationV2 -> Sorting` chain hard-imports **pydantic** at module top via every `_params/*` (`_params/sorter.py:32`, etc.), and `_probe_v2_curation`'s docstring named "SpikeInterface / **Pydantic** skew" as a degradation reason — so removing it means a pydantic-skew failure now hard-breaks `spikesorting_merge` for v0/v1-only users. Either also defer the `_params/*` pydantic imports, or **document pydantic as a hard `spikesorting_merge` dependency** and state that `test_import_light_schema_identical` covers the SI/torch axis only.
- **Docs:** note in [SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md) that v2 results are queryable without the compute stack (torch/SI) — only *running* v2 needs them.

## Deliberately not in this phase
- No topology / artifact / tracked-unit changes.
- No change to `CurationV2`'s compute behavior — only import timing (heavy imports stay in method bodies).

## Validation slice
| Test | Asserts |
| --- | --- |
| `test_import_light_schema_identical` (slow) | in a torch/SI-less env (or with SI import made to raise on module-top), `import spikesorting_merge` compiles the **identical** schema (same part/FK/dispatch map) as the full-import case. |
| `test_no_schema_import_side_effects` | importing the declaration chain triggers no `@schema` DB connection / import-time side effect beyond the intended registration. |
| `test_query_without_compute_stack` | a v2 sort registered by a full-stack worker is queryable (`get_spike_times`) from the import-light path. |
| `test_probe_removed` | `_probe_v2_curation` and the `if CurationV2 is None` guards no longer exist (grep). |

## Fixtures
A torch/SI-less import harness (e.g., monkeypatch `spikeinterface` import to raise
at module-top, or a minimal env) to prove import-light; a pre-registered v2 sort
for the query-without-compute test.

## Review
Dispatch `code-reviewer`. Confirm: only import *timing* changed, no behavior;
the schema is byte-identical across full vs import-light; all conditional guards
removed (no dead `CurationV2 is None` branches); the import-light checks run in a
subprocess and compare schema structurally; the query-without-torch doc note
added. (Standalone PR — no spike ledger/gates.)
