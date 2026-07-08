# Appendix — machinery & external-schema references

[← back to PLAN.md](PLAN.md)

Line refs verified against the repo at plan time (branch
`feature/fiber-photometry-ingestion`, off `master` @ `8a352448`). Re-confirm if
the executor is on a later commit.

## `SpyglassIngestion` / `IngestionMixin` — `src/spyglass/utils/mixins/ingestion.py`

The base the device + config + response tables build on.

- `class IngestionMixin` — `:22`. Class attributes the tables set:
  `_expected_duplicates` (`:47`), `_source_nwb_object_name` (`:50`),
  `_extension_requirements` (`:52`).
- `table_key_to_obj_attr` — `:56`. The declarative field mapping. Values may be a
  string attr, an `(attr, default)` tuple, or a **callable** `v(obj)` — the
  callable form is how model specs are folded (see
  [shared-contracts.md#null-safe-model](shared-contracts.md#null-safe-model)).
- `generate_entries_from_nwb_object()` — `:108`. Default DynamicTable path
  (`to_dataframe().itertuples()`) is at `:120-132`. **Raises `ValueError` when a
  mapped nested object is `None`** at `:142-145` — the reason model-less fibers
  break the optogenetics implant mapping, and why config/response need custom
  overrides rather than the declarative path.
- `get_nwb_objects()` — `:168`. Default matches `nwb_file.objects` by
  `_source_nwb_object_type`. **Override this** per
  [shared-contracts.md#ref-scoped-get_nwb_objects](shared-contracts.md#ref-scoped-get_nwb_objects).
- `insert_from_nwbfile()` — `:221`. Ordering that matters: fetch objects
  (`:249`) → **early-return with no warning if none** (`:250-251`) → **then**
  `check_extension_requirements()` (`:255`). This ordering is why the version
  warning only fires when photometry objects are present, and why
  `get_nwb_objects()` must be defensive on old schemas.
- `_key_has_required_attrs()` — `:321`. **Drops** (does not raise) an entry whose
  non-nullable attr is `None`. This is why nullable columns matter for
  sparse-but-valid files, and why the optogenetics tables silently drop sparse
  photometry fibers.
- `validate_duplicates()` / `validate1_duplicate()` — `:380` / `:407`.
  `_unequal_vals()` at `:460` compares with plain `!=` → the blob-comparison
  hazard ([shared-contracts.md#dup-safety](shared-contracts.md#dup-safety)).
- `check_extension_requirements()` — `:467`. Compares file-embedded namespace
  versions to `_extension_requirements`; warns + returns `False` if below min.

## `fetch_nwb` resolution — `src/spyglass/utils/mixins/fetch.py`

- `:72-84` — `fetch_nwb()` resolves the NWB file via `_nwb_table` **or** a literal
  `-> Nwbfile` in the definition, else raises `NotImplementedError`. The response
  table FKs `-> Session`, so it **must** set `_nwb_table = Nwbfile`.

## `Raw` — the signal-table template — `src/spyglass/common/common_ephys.py`

- `class Raw(SpyglassIngestion, dj.Imported)` — `:286`. `_nwb_table = Nwbfile`
  at `:297`. `table_key_to_obj_attr` (declarative, incl. callables) at `:311`.
  `nwb_object(key)` accessor at `:377`. Copy the **class shape**
  (`SpyglassIngestion, dj.Imported` + `_nwb_table = Nwbfile`) for
  `FiberPhotometryResponseSeries` (phase-2) — **but not** `nwb_object`'s body:
  `Raw` fetches `raw_object_id` by `nwb_file_name` only (`:386`) because it is
  one row per file; the response series is many-per-file, so its `nwb_object(key)`
  must fetch by the full key (`response_series_object_id`). See
  [phase-2-signal-reference.md](phase-2-signal-reference.md).

## Optogenetics pattern + the gate — `src/spyglass/common/common_optogenetics.py`

No schema change; phase-1 adds a behavioral `get_nwb_objects()` **gate** to
`OpticalFiberDevice`/`OpticalFiberImplant` (see
[shared-contracts.md#opto-gate](shared-contracts.md#opto-gate)). Otherwise this
module is the closest existing example of `SpyglassIngestion` tables and a custom
`generate_entries_from_nwb_object` override.

- `OpticalFiberImplant` — `:346`; its `generate_entries_from_nwb_object`
  override (assigns a per-file index) — `:391`. `OpticalFiberDevice` — `:313`
  (the NOT-NULL fiber fields that make it drop sparse photometry fibers).
- `OptogeneticProtocol` (`:14`) shows a table with a custom `make`.

## External NWB extension schemas

Installed spec YAMLs (read at audit time; versions pinned in the design doc's
Schema-coverage section):

- `ndx-fiber-photometry` 0.2.3 — `.../site-packages/ndx_fiber_photometry/spec/ndx-fiber-photometry.extensions.yaml`.
  Key types: `FiberPhotometryTable` (required + optional columns), `FiberPhotometry`
  (LabMetaData), `FiberPhotometryResponseSeries` (`data`, optional
  `fiber_photometry_table_region`), `FiberPhotometryIndicators`.
- `ndx-ophys-devices` 0.3.1 — `.../site-packages/ndx_ophys_devices/spec/ndx-ophys-devices.extensions.yaml`.
  Device model/instance types + `FiberInsertion` + `Indicator`.

The design doc's **Schema coverage** table is the authoritative field-by-field
disposition (✅ / ◐ / ⚠️ / ⛔) — do not re-derive it; implement to it.

## Verified fixture prototype

`docs/superpowers/specs/2026-07-07-fiber-photometry-fixture-prototype.py` — a
minimal `ndx-fiber-photometry` NWB builder that **round-trips on pynwb 3.1.3**
(embeds `core` 2.9.0), verified at design time. Phase-1's fixture builder starts
from this and extends it (filters, dichroic, pulsed source, null/edge cases).
Gotchas it already resolved: `DeviceModel`/`add_device_model` exist in 3.1.3;
`DynamicTableRegion` imports from `hdmf.common`; `FiberPhotometryTable.add_row`
takes object-ref columns directly.
