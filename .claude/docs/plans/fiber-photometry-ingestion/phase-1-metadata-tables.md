# Phase 1 — Device catalog + `FiberPhotometryConfig` (ingest & query the setup)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Ships the metadata layer: six reusable device/indicator tables and the per-fiber
config table, ingested by `populate_all_common`, plus a `core` 2.9.0 test
fixture and the test suite. After this phase a user can ingest a photometry NWB
and query the full experimental setup (which fiber, indicator, excitation source,
detector, filters, wavelengths, insertion). No signal retrieval yet (phase-2).

**Inputs to read first:**

- `docs/superpowers/specs/2026-07-07-fiber-photometry-ingestion-design.md` —
  **the schema**. Sections: "Architecture" → "Reusable device tables" and
  "Session-specific config"; "Schema coverage" (the authoritative field
  disposition); "Testing" (tests 1–13); Decisions table.
- [appendix.md](appendix.md) — `IngestionMixin` refs (`get_nwb_objects`,
  `generate_entries_from_nwb_object`, `_key_has_required_attrs`,
  `_extension_requirements`, `_unequal_vals`) and the fixture prototype.
- `src/spyglass/common/common_optogenetics.py` — closest existing
  `SpyglassIngestion` example; also **edited** this phase (the `get_nwb_objects()`
  gate on `OpticalFiberDevice`/`OpticalFiberImplant` — see the gate task below).
- `docs/superpowers/specs/2026-07-07-fiber-photometry-fixture-prototype.py` —
  the verified starting point for the fixture builder.

**Contracts referenced:**

- [Config schema](shared-contracts.md#config-schema) — PK and phase-2-relied-on
  invariants; **do not weaken**.
- [Ref-scoped `get_nwb_objects()`](shared-contracts.md#ref-scoped-get_nwb_objects)
  — the photometry-scoping invariant for all six device tables.
- [Optogenetics fiber-table gate](shared-contracts.md#opto-gate) — the
  behavioral change to `common_optogenetics` (Open Question 1, resolved).
- [Null-safe `.model`](shared-contracts.md#null-safe-model),
  [Duplicate-validation safety](shared-contracts.md#dup-safety).

## Tasks

- **Module skeleton.** Create `src/spyglass/common/common_photometry.py`:
  `schema = dj.schema("common_photometry")`, imports (`datajoint`,
  `SpyglassIngestion`, `Session`, `Nwbfile`, `logger`). Add a
  `_referenced_devices(nwb_file, column_names, ...)` helper implementing the
  three-step ref-scoped collection from
  [shared-contracts.md#ref-scoped-get_nwb_objects](shared-contracts.md#ref-scoped-get_nwb_objects),
  and a `model_attr(name)` null-safe folder from
  [shared-contracts.md#null-safe-model](shared-contracts.md#null-safe-model).

- **Six device tables** (`Indicator`, `ExcitationSource`, `Photodetector`,
  `DichroicMirror`, `OpticalFilter`, `OpticalFiber`), each `SpyglassIngestion,
  dj.Manual`, keyed by `<device>_name`, `_expected_duplicates = True`,
  `_extension_requirements = {"ndx-fiber-photometry": "0.2.3", "ndx-ophys-devices":
  "0.3.1"}`. Columns per the design doc "Reusable device tables" (reusable spec
  only; every optional field nullable; `[2]`-vectors as min/max scalar pairs;
  `source_class`/`filter_class` derived discriminators; the five model-backed
  tables also carry `manufacturer`/`model_number`/`model_description`). Each
  overrides `get_nwb_objects()` via `_referenced_devices(...)` with its column
  name(s); field mapping is declarative using `model_attr(...)` for model specs.

- **`FiberPhotometryConfig`** (`SpyglassIngestion, dj.Manual`). PK and columns
  per [shared-contracts.md#config-schema](shared-contracts.md#config-schema) and
  the design doc. **Object discovery**: set `_source_nwb_object_type =
  "FiberPhotometryTable"` so the mixin's default `get_nwb_objects()` collects all
  `FiberPhotometryTable` objects (and returns `[]` — a clean no-op — for a file
  with none); `_source_nwb_object_type` is otherwise a mixin property that
  **raises `NotImplementedError`** if unset ([ingestion.py:79](../../../src/spyglass/utils/mixins/ingestion.py)).
  Set `_extension_requirements = {"ndx-fiber-photometry": "0.2.3",
  "ndx-ophys-devices": "0.3.1"}` (it reads the ndx-fiber-photometry table and
  ndx-ophys-devices object refs) — required so the mixin's post-`get_nwb_objects()`
  version check gates below-min files (`test_below_min_version_warns`).
  Implement a custom `generate_entries_from_nwb_object()` that,
  for each `FiberPhotometryTable` and each row: sets `location ← row.location`
  (required), resolves device FKs from the referenced objects' `.name`, stores
  the session-local fiber fields (`optical_fiber_description`, insertion incl.
  `depth`/`position_reference`, all nullable), reads the optional
  columns/refs only when present-and-non-null, and **warns** (once, naming them)
  on (a) unmodeled table columns and (b) populated-but-unmodeled device-object
  attributes (excitation operational fields, `Indicator.viral_vector_injection`).
  Matches **all** `FiberPhotometryTable` objects and takes
  `fiber_photometry_name` from each table's parent container.

- **Optogenetics fiber-table gate** — per
  [shared-contracts.md#opto-gate](shared-contracts.md#opto-gate), add a
  `get_nwb_objects()` override to `common_optogenetics.OpticalFiberDevice`
  (`common_optogenetics.py:313`) and `OpticalFiberImplant` (`:346`):
  `OpticalFiberImplant` drops photometry-referenced fiber **instances**;
  `OpticalFiberDevice` drops only `OpticalFiberModel`s **not needed by a remaining
  (non-photometry) fiber** (keep a model if any surviving fiber still references
  it — do **not** drop every photometry-referenced model, which would break a
  shared-model mixed-modality file). Factor the "fibers referenced by any
  `FiberPhotometryTable`" collection into a shared helper (importable by both
  modules) rather than duplicating it. Behavioral only — **no schema change** to
  `common_optogenetics`. This is the resolution of Open Question 1.

- **`populate_all_common` wiring** — `populate_all_common.py`: add the import
  (near `:33-38`); add the six device tables to the parent-node list (near
  `:205-206`); add `FiberPhotometryConfig` to the Session-dependent list (near
  `:233-235`), after the device tables. (`FiberPhotometryResponseSeries` is added
  in phase-2.)

- **Exports** — `common/__init__.py`: add the `common_photometry` import block
  (after `:53`) and the table names to `__all__` (near `:96`).

- **Test-dependency** — `pyproject.toml`: add `"ndx-fiber-photometry==0.2.3",` to
  `optional-dependencies.test` (`:105-118`). Do **not** touch core `dependencies`
  or `pynwb`/`hdmf` (overview → Dependency policy).

- **Fixture builder** — `tests/common/conftest.py` (or a `tests/common/`
  fixture module): a builder that writes a **`core` 2.9.0** photometry NWB with
  pynwb 3.1.x, starting from the prototype
  (`docs/superpowers/specs/2026-07-07-fiber-photometry-fixture-prototype.py`) and
  extended to cover: two fibers, a `DichroicMirror`, a `BandOpticalFilter`, an
  `EdgeOpticalFilter` (with the optional config filter/dichroic columns
  populated), a `PulsedExcitationSource`, a `notes` column, a model-less
  referenced device, a "sparse model + complete insertion" fiber, and a
  **mixed-modality** file (a photometry fiber plus a separate non-photometry
  `OpticalFiber` sharing the same `OpticalFiberModel`, for the gate over-prune
  test). Parametrize so tests can request the sub-fixtures they need.

- **Docs** — add a short "Fiber photometry" subsection to the relevant common-
  tables doc (and a CHANGELOG entry) covering: what gets ingested, the `core`
  2.9.0 data-production constraint (design doc "Dependencies"), and that the only
  `common_optogenetics` change is the behavioral `get_nwb_objects()` gate (no
  schema change). No notebook yet (retrieval example lands in phase-2 where
  `fetch1_dataframe` exists).

## Deliberately not in this phase

- **`FiberPhotometryResponseSeries`, `.Fiber`, `fetch1_dataframe`, signal
  retrieval** — phase-2. The config PK is frozen here for phase-2 to FK into.
- **Any analysis/dF/F** — future PR (overview → Non-Goals).
- **Schema changes to `common_optogenetics`** — the gate is a `get_nwb_objects()`
  override only; **no** new/altered columns.
- **pynwb/hdmf bump** — out of scope.

## Validation slice

Maps to the design doc "Testing" items. All run on the current floor with the
`core` 2.9.0 fixture.

| Test | Asserts |
| --- | --- |
| `test_device_and_config_ingest` | device + config rows appear; config device FKs (incl. `-> OpticalFiber`) resolve; session-local fiber fields (`optical_fiber_description`, insertion incl. `depth`/`position_reference`) stored, null where source null |
| `test_location_from_row` | `config.location == row.location` (e.g. `'DLS'`), **not** the fiber description |
| `test_ref_scoped_no_op` | a file with `ndx-ophys-devices` `OpticalFiber`/`Indicator` but **no** `FiberPhotometry` container → device tables produce **no rows, no warning, no exception** |
| `test_below_min_version_warns` | photometry objects present but `ndx-fiber-photometry` below min → warning + no rows; and `get_nwb_objects()` on a missing-column table returns `[]` (no raise) |
| `test_device_reingest_and_cross_session` | second file reusing a device `name` with same reusable spec → clean skip, no `DuplicateError`; blob-free vector columns don't raise "truth value ambiguous" |
| `test_null_fiber_metadata` | model-less fiber → `OpticalFiber` row with null model cols (no `AttributeError`); null `pitch`/`roll`/`yaw`/`description` stored null; required `location` present |
| `test_populate_all_common_gate` | full `populate_all_common()` on **all** photometry fixtures — sample-like, **model-less**, and **sparse-model+complete-insertion** — returns **no new `InsertError` keys** (the gate excludes photometry fibers from the optogenetics tables); photometry rows present. Run with `rollback_on_fail=True` too and assert the Nwbfile + photometry rows **survive** (no `super_delete`) |
| `test_opto_gate_noop_on_non_photometry` | a **pure-optogenetics** file (`OpticalFiber` with complete fields, no `FiberPhotometry` container) → the gated `OpticalFiberDevice`/`OpticalFiberImplant` still ingest it exactly as before (gate returns the default set); regression guard that the gate is a no-op without a `FiberPhotometry` container |
| `test_opto_gate_mixed_modality_shared_model` | a **mixed** file with a photometry fiber **and** a separate non-photometry (optogenetics) fiber that **shares the same `OpticalFiberModel`** → the shared model is **kept** (a remaining fiber needs it), the optogenetics fiber ingests into `OpticalFiberImplant` with a resolvable `-> OpticalFiberDevice` FK, and `populate_all_common` returns **no new `InsertError`**. Guards the over-prune bug in [shared-contracts.md#opto-gate](shared-contracts.md#opto-gate) |
| `test_package_absent_import_safety` | ingest the pre-built fixture with `ndx_fiber_photometry` uninstalled → rows present; `"ndx_fiber_photometry" not in sys.modules` after ingest (build fixture in a separate step) — *marked slow/subprocess* |
| `test_subtype_and_optional_roundtrip` | `PulsedExcitationSource` → `source_class='pulsed'`; base/`Band`/`Edge` filters → correct `filter_class`; the full model-spec set (`wavelength_min/max`, `reflection_band_*`, `slope_*`, fiber `numerical_aperture`/`ferrule_*`) round-trips; `notes` stored |
| `test_unmodeled_warns` | populated `commanded_voltage_series` column, `ExcitationSource.power_in_W`, or `Indicator.viral_vector_injection` → warning naming it; ingest still succeeds |

## Fixtures

Synthesized in `tests/common/` via the fixture builder (above), written with
pynwb 3.1.x so they embed `core` 2.9.0. No real-data slice is checked in (the
1.3 GB sample files are `core` 2.10.0 and unreadable on the floor — overview →
Dependency policy); the synthetic fixture is the parity target and is
representative because it was derived from the real files' structure.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff.
Confirm:
- Every task implemented as specified; the six device tables are ref-scoped (no
  generic `nwb_file.objects` matching) and store reusable spec only.
- "Deliberately not in this phase" honored — no `FiberPhotometryResponseSeries`,
  no dF/F; the only `common_optogenetics` change is the `get_nwb_objects()` gate
  (no schema change), and it is a no-op for non-photometry files.
- Validation-slice tests pass; slow/subprocess tests marked.
- Tests exercise behavior, not tautologies; shared fixture setup is in
  `conftest`, not copy-pasted (`testing-anti-patterns`).
- No docstring/test/module name references this plan or "phase 1".
- `pyproject` change is limited to the `test` extra; core deps and `pynwb`/`hdmf`
  untouched.
- CHANGELOG + docs subsection updated in this PR, not deferred.
