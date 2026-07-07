# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

The canonical design (schema of every table, field-by-field mappings, the full
`ndx-fiber-photometry` 0.2.3 / `ndx-ophys-devices` 0.3.1 schema-coverage table,
and the rationale for every decision across 13 review passes) lives in
**`docs/superpowers/specs/2026-07-07-fiber-photometry-ingestion-design.md`**
(the "design doc" throughout this plan). This plan does not restate the schema;
it points into that doc and specifies the executable breakdown, exact
integration points, and validation.

## Current codebase integration points

New module `src/spyglass/common/common_photometry.py` (schema `common_photometry`)
is added. Existing files get additive registrations, **plus one behavioral gate
in `common_optogenetics`** (Open Question 1, resolved to "gate"): the optogenetics
fiber tables must skip photometry-referenced fibers so they don't error on
photometry files (see the optogenetics-cross-processing risk).

- `src/spyglass/common/__init__.py:51-53` — optogenetics import block; **add** a
  parallel `from spyglass.common.common_photometry import (...)` block after it.
- `src/spyglass/common/__init__.py:93-96` — `__all__` entries; **add** the new
  table names. Preserve everything else.
- `src/spyglass/common/populate_all_common.py:33-38` — optogenetics import
  block; **add** a `common_photometry` import.
- `src/spyglass/common/populate_all_common.py:205-206` — parent-node list
  (`OpticalFiberDevice`, `Virus`): **add** the six device tables here (they have
  no Session dependency).
- `src/spyglass/common/populate_all_common.py:233-235` — Session-dependent list:
  **add** `FiberPhotometryConfig` then `FiberPhotometryResponseSeries` after it
  (config depends on Session + the six device tables; the response series
  depends on config).
- `pyproject.toml:105-118` — `optional-dependencies.test`; **add**
  `"ndx-fiber-photometry==0.2.3",` (fixture-build dependency only — ingestion
  never imports it; see design doc "Optional-dependency guarantee").
- `src/spyglass/common/common_optogenetics.py:313` (`OpticalFiberDevice`) and
  `:346` (`OpticalFiberImplant`) — **add a `get_nwb_objects()` override to each**
  (the gate, phase-1): `OpticalFiberImplant` drops photometry-referenced fiber
  *instances*; `OpticalFiberDevice` drops only *models not needed by a remaining
  (non-photometry) fiber* — see the exact rule in
  [shared-contracts.md#opto-gate](shared-contracts.md#opto-gate).
  Backward-compatible: a file with no `FiberPhotometry` container is
  unaffected. This is the only *behavioral* change to existing code.
- `src/spyglass/common/common_ephys.py:286-389` — `Raw` is the template for the
  signal-reference table (`SpyglassIngestion, dj.Imported`, `raw_object_id`,
  `_nwb_table = Nwbfile` at :297, `nwb_object()` at :377). Untouched — read only.
- `src/spyglass/utils/mixins/ingestion.py` and `.../mixins/fetch.py` — the
  ingestion/fetch machinery the new tables build on. Untouched — read only; exact
  refs in [appendix.md](appendix.md).

## Scope and dependency policy

### Goals

- Ingest `ndx-fiber-photometry` metadata (devices, indicators, per-fiber config)
  into queryable DataJoint tables, and expose the recorded fluorescence via
  `fetch_nwb()` / `fetch1_dataframe()` without copying arrays into the database.
- Ship in a new `common_photometry` module on the **current** dependency floor
  for `core` 2.9.0 files (no pynwb/hdmf bump). The only edit outside the new
  module is the optogenetics `get_nwb_objects()` gate (behavioral, no schema
  change).

### Non-Goals

- **No analysis / derived-signal pipeline** (dF/F, ratiometric isosbestic
  correction, motion correction, downsampling, merge tables). That is a separate
  future PR — do not add it while "in here."
- **No schema changes to `common_optogenetics`** — no new columns, no altered
  types. The design decoupled from it (photometry has its own tables). The **one
  allowed** change is behavioral: a `get_nwb_objects()` gate on the two
  optogenetics fiber tables so they skip photometry fibers (Open Question 1).
- **No pynwb/hdmf floor bump** — the 2.10.0 read path is blocked upstream and is
  out of scope (see Dependency policy + Risks).
- `CommandedVoltageSeries`, viral-vector/injection linkage, and excitation-source
  operational scalars are deferred + warned, not modeled (design doc, Scope /
  Schema coverage).

### Dependency policy

- `ndx-fiber-photometry==0.2.3` → `optional-dependencies.test` **only**. It is
  needed solely to *build* the test fixture; ingestion matches NWB types by
  class-name string and gates on file-embedded namespaces, so it is never
  imported at ingest time. This is verified — see design doc "Optional-dependency
  guarantee (verified)".
- **Do not** add `ndx-fiber-photometry` to core `dependencies`, and **do not**
  bump `pynwb`/`hdmf`. Reading the *sample* files (which embed `core` 2.10.0)
  would require `pynwb>=4.0.0`/`hdmf>=6.1.0`, which is currently unsatisfiable
  because `ndx-franklab-novela` (a mandatory core dep) pins `hdmf<5`. The feature
  targets `core` 2.9.0 files (pynwb 3.1.x producers); full analysis in the design
  doc "Dependencies" section.

## Metrics

- All Validation-slice tests in both phases pass on the current floor
  (`pynwb 3.1.3` / `hdmf 4.3.1`), using a `core` 2.9.0 fixture.
- The Schema-coverage table in the design doc is fully realized: every field
  marked ✅/◐ round-trips through ingest→fetch; every ⚠️ field triggers a warning
  when populated; every ⛔ type/field is absent.
- Package-absent guarantee: ingestion of the fixture succeeds with
  `ndx_fiber_photometry` uninstalled, and `"ndx_fiber_photometry" not in
  sys.modules` after the ingest call.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Existing `common_optogenetics` fiber tables run on photometry files and can **error** on model-less / sparse-model+complete-insertion fibers → `populate_all_common` returns an `InsertError` failure state, and `rollback_on_fail=True` `super_delete`s the whole Nwbfile (destroying photometry inserts) | **Gate** (Open Question 1, resolved): both optogenetics fiber tables get a `get_nwb_objects()` override — `OpticalFiberImplant` drops photometry fiber *instances*; `OpticalFiberDevice` drops only models *not needed by a remaining non-photometry fiber* (exact rule: [shared-contracts.md#opto-gate](shared-contracts.md#opto-gate)) — so they produce zero `InsertError` on photometry files. Phase-1 test asserts `populate_all_common()` returns **no new** `InsertError` for **all** photometry fixtures (incl. model-less, complete-insertion, and mixed-modality shared-model), under both `rollback_on_fail` values. |
| Device tables matching generic `ndx-ophys-devices` objects would ingest non-photometry devices | Device `get_nwb_objects()` is photometry-ref-scoped (returns `[]` without a `FiberPhotometry` container; collects only `FiberPhotometryTable`-referenced objects). See [shared-contracts.md](shared-contracts.md#ref-scoped-get_nwb_objects). |
| `_expected_duplicates=True` duplicate validation crashes on blob/array values | `[2]`-vector specs stored as scalar min/max pairs, not blobs. [shared-contracts.md](shared-contracts.md). |
| `Device.model` is optional → `AttributeError` folding model specs | Null-safe `.model` extraction. [shared-contracts.md](shared-contracts.md#null-safe-model). |
| Object-ref resolution in `to_dataframe()` behaves unexpectedly | Already verified on real files + a prototype fixture round-trips on pynwb 3.1.3 (design doc "Validated against real data"; prototype at `docs/superpowers/specs/2026-07-07-fiber-photometry-fixture-prototype.py`). |

## Rollout Strategy

All-at-once, additive. No feature flag, no backwards-compatibility surface (new
schema + new module; no existing public API changes). A stock install on the
current floor gains the tables; a non-photometry file is a clean no-op for them.
Phase-1 ships the metadata layer (query the setup); phase-2 adds signal retrieval.

## Open Questions

1. **Optogenetics cross-processing** — **RESOLVED: gate.** The optogenetics fiber
   tables error on some photometry fibers, which makes `populate_all_common`
   report failure and (with `rollback_on_fail=True`) destroy the photometry
   inserts. Owner chose to gate: both `OpticalFiberDevice` and
   `OpticalFiberImplant` get a `get_nwb_objects()` override excluding
   photometry-referenced fibers (phase-1). Backward-compatible; the only
   behavioral change to `common_optogenetics`.

## Estimated Effort

~700–1000 LOC total across both phases: phase-1 ≈ 6 device tables + config +
custom `get_nwb_objects`/override + fixture builder + ~10 tests + wiring
(~500–700); phase-2 ≈ response series + `.Fiber` part + `fetch1_dataframe` +
~5 tests + wiring (~200–300).
