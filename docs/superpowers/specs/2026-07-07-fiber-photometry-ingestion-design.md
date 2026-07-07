# Fiber-Photometry Ingestion — Design

**Date:** 2026-07-07
**Status:** Approved design, pending implementation plan
**Branch:** `feature/fiber-photometry-ingestion` (off `master`)

## Goal

Give Spyglass the ability to ingest fiber-photometry data recorded with the
[`ndx-fiber-photometry`](https://github.com/catalystneuro/ndx-fiber-photometry)
NWB extension: the experimental setup (devices, indicators, per-fiber
configuration) becomes queryable DataJoint metadata, and the recorded
fluorescence traces stay in the NWB file and are retrieved on demand via
`fetch_nwb()`.

This mirrors the existing `common_optogenetics` module, which ingests the
`ndx-ophys-devices` types (`ViralVector`, `ViralVectorInjection`, `OpticalFiber`)
that `ndx-fiber-photometry` also builds on.

## Scope

**In scope (this PR):**
- Reusable device/indicator metadata tables for the photometry-specific
  `ndx-ophys-devices` types.
- A per-fiber configuration table (the NWB `FiberPhotometryTable`).
- A signal-reference table for `FiberPhotometryResponseSeries` — stores the NWB
  `object_id`, not the array, retrievable via `fetch_nwb()`.
- `populate_all_common` wiring, exports, tests, and a short docs/notebook cell.

**Out of scope (deliberate, follow-up PRs):**
- Any analysis/derived-signal pipeline (dF/F, motion correction, downsampling,
  merge tables). A metadata + raw-reference layer is the correct first
  increment; a versioned `photometry/v1/` pipeline is a natural second PR.
- `CommandedVoltageSeries` ingestion (secondary signal used only by modulated-
  excitation rigs — YAGNI for now; the config table's optional reference to it
  is simply not modeled yet).
- Relocating the shared `ndx-ophys-devices` tables into a neutral module. We
  reuse `common_optogenetics`' tables **in place** (see Decisions).

## Background: the `ndx-fiber-photometry` data model

Three layers (verified against `ndx-fiber-photometry` 0.2.3 and
`ndx-ophys-devices` ≥ 0.3.1):

1. **Devices** (`ndx-ophys-devices`, in `nwbfile.devices` / device models):
   `OpticalFiberModel`/`OpticalFiber`, `ExcitationSourceModel`/`ExcitationSource`,
   `PhotodetectorModel`/`Photodetector`, `DichroicMirror`,
   `BandOpticalFilter`/`EdgeOpticalFilter`, `Indicator`, `ViralVector`,
   `ViralVectorInjection`. Uses a model-instance pattern (model = specs,
   instance = actual hardware).
2. **Config** (`ndx-fiber-photometry`, in `nwbfile.lab_meta_data`): a
   `FiberPhotometry` container holding a `FiberPhotometryTable` (`DynamicTable`,
   one row per fiber/channel with object-references to the devices + indicator +
   excitation/emission wavelengths + location + optional coordinates).
3. **Signal** (`ndx-fiber-photometry`, in `nwbfile.acquisition`):
   `FiberPhotometryResponseSeries` (a `TimeSeries`; `data` is `[time]` or
   `[time, n_fibers]`, with a `fiber_photometry_table_region`
   `DynamicTableRegion` back into the config table).

## Decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Depth | Metadata + signal **reference** | Makes fluorescence retrievable while following Spyglass's object-id convention; one focused PR. |
| Shared devices | **Reuse `common_optogenetics` tables in place** | `Virus`, `VirusInjection`, `OpticalFiberDevice`, `OpticalFiberImplant` already ingest the shared `ndx-ophys-devices` types. No duplication, no schema migration of the alpha-stage optogenetics tables. |
| Fiber link | `FiberPhotometryConfig` **FKs into `OpticalFiberImplant`** | Single source of truth for the implanted fiber + its stereotactic coordinates. `OpticalFiberImplant` already runs in `populate_all_common`. |
| Model/instance | **Collapse** to one reusable table per device type | Leaner than a model+instance split; spec fields are folded in from `.model` via the object-ref mapping (the same trick `OpticalFiberImplant` uses to read `model.name`). |
| Extension dep | `ndx-fiber-photometry` in the **`test` extra only** | Ingestion never imports the package (see Optional-Dependency Guarantee); it's needed only to build the test fixture. |

## Architecture

New module `src/spyglass/common/common_photometry.py`, schema
`common_photometry`. Reuses the `SpyglassIngestion` mixin exactly as
`common_optogenetics` does.

### Reusable device tables

Keyed by device `name`, `_expected_duplicates = True`,
`_extension_requirements = {"ndx-ophys-devices": "0.3.1"}`. Each reads the
`ndx-ophys-devices` **instance** and folds in useful `.model` spec fields via
the `table_key_to_obj_attr` object-ref mapping.

- **`Indicator`** ← `Indicator` (NWBContainer): `label`, `description`,
  `manufacturer`.
- **`ExcitationSource`** ← `ExcitationSource` (Device) + its model:
  `source_type`, `excitation_mode` (from model), `power_in_W` (instance).
- **`Photodetector`** ← `Photodetector` (Device) + its model: `detector_type`,
  `gain`, `gain_unit`.
- **`DichroicMirror`** ← `DichroicMirror` (Device) + model: cut-on/cut-off
  wavelengths (optional fields).
- **`OpticalFilter`** ← `BandOpticalFilter` / `EdgeOpticalFilter`: `filter_type`
  plus center/bandwidth or cut wavelength. One table covering both subtypes
  (matched by class-name string; subtype-specific fields nullable).

`ExcitationSourceModel`/`PhotodetectorModel` etc. are **not** separate tables;
their spec fields are denormalized onto the reusable instance table. This
matches the optogenetics precedent and keeps the table count to five new device
tables.

### Session-specific config — `FiberPhotometryConfig`

The NWB `FiberPhotometryTable`, one row per fiber/channel. Because it is a
`DynamicTable`, the mixin's `to_dataframe()` path yields one entry per row.

```
FiberPhotometryConfig
  -> Session
  fiber_id: int                      # row index within the FiberPhotometryTable
  ---
  -> OpticalFiberImplant             # common_optogenetics (reused)
  -> Indicator
  -> ExcitationSource
  -> Photodetector
  -> DichroicMirror                  # nullable FK (optional in NWB)
  -> OpticalFilter.proj(emission_filter_name='filter_name')   # nullable
  -> OpticalFilter.proj(excitation_filter_name='filter_name') # nullable
  location: varchar(255)
  excitation_wavelength_in_nm: float
  emission_wavelength_in_nm: float
  coordinates=null: blob             # optional 3-vector for multi-fiber arrays
```

Object-references in the row (`indicator`, `optical_fiber`, `excitation_source`,
…) resolve to the referenced NWB objects in `to_dataframe()`; the mapping uses
callables (`lambda row: row.indicator.name`) to extract the FK values. The
optional device references (dichroic mirror, filters) are handled with the
mixin's tuple-with-default mapping so missing references null the FK rather than
raise.

### Signal reference — `FiberPhotometryResponseSeries`

`dj.Imported`, following [`Raw`](../../../src/spyglass/common/common_ephys.py)
(`raw_object_id: varchar(40)` + `nwb_object()` accessor).

```
FiberPhotometryResponseSeries
  -> Session
  response_series_object_id: varchar(40)   # NWB object id, for fetch_nwb()
  ---
  name: varchar(80)
  description: varchar(2000)
  num_samples: int

  class Fiber(dj.Part):                     # the fiber_photometry_table_region
    -> master
    column_index: int                       # data column -> fiber
    ---
    -> FiberPhotometryConfig
```

The `.Fiber` part makes the column↔fiber mapping relational (from the NWB
`DynamicTableRegion`) instead of an opaque blob, so a 2-D `[time, n_fibers]`
series is fully interpretable from DataJoint alone.

Retrieval helper: `fetch1_dataframe()` returns a `pandas.DataFrame` indexed by
timestamps with one column per fiber (labeled by the `.Fiber` mapping),
assembled from `fetch_nwb()`.

## Optional-dependency guarantee

The explicit requirement — *do not import `ndx-fiber-photometry` unless the user
needs it* — is satisfied by the existing ingestion machinery; no new mechanism
is required.

- **Type matching is by class-name string.**
  [`is_nwb_obj_type`](../../../src/spyglass/utils/nwb_helper_fn.py) compares
  `nwb_object.__class__.__name__ == "FiberPhotometryResponseSeries"`. pynwb
  reconstructs typed containers from the spec **embedded in the NWB file**, so
  matching works even when the extension package is not installed. No `import
  ndx_fiber_photometry` anywhere on the ingest path.
- **Gating is by file-embedded namespace.**
  [`check_extension_requirements`](../../../src/spyglass/utils/mixins/ingestion.py)
  reads the file's namespace versions (`get_file_namespaces`, via
  `load_namespaces`) and compares against `_extension_requirements`. A file
  lacking the `ndx-fiber-photometry` / `ndx-ophys-devices` namespace (or below
  the min version) is **skipped with a warning, not an error**.

Consequences:
- `ndx-fiber-photometry==0.2.3` is added to `optional-dependencies.test` only,
  needed solely to *write* the fixture NWB. It is **not** a core dependency.
  (`ndx-ophys-devices` is already a core dependency for other reasons; we do not
  add it.)
- A stock Spyglass install ingests a photometry NWB correctly.
- A stock install ingesting a non-photometry NWB does nothing photometry-related
  (the device/config/signal tables find no matching objects and no-op).

## Integration

- **`populate_all_common.py`**: add the five device tables as parent nodes
  (alongside `OpticalFiberDevice`, `Virus`); `FiberPhotometryConfig` after
  `OpticalFiberImplant` **and** the device tables; `FiberPhotometryResponseSeries`
  after `FiberPhotometryConfig`.
- **`common/__init__.py`**: export the new tables.
- **`pyproject.toml`**: `ndx-fiber-photometry==0.2.3` in the `test` extra.

## Testing

Fixture-driven, in `tests/common/` (fixture builder uses the `test` extra):

1. **Fixture builder** writes a minimal photometry NWB — two devices of each
   type, a two-fiber `FiberPhotometryTable`, and one `[time, 2]`
   `FiberPhotometryResponseSeries`.
2. **Ingestion**: device, config, and signal rows all appear;
   `FiberPhotometryConfig` FKs resolve to the right devices and to
   `OpticalFiberImplant`; the `.Fiber` part maps both columns.
3. **Retrieval**: `fetch_nwb()` / `fetch1_dataframe()` returns the traces with
   the expected shape and per-fiber column labels.
4. **Gating contract**: ingesting a file **without** the namespace is skipped
   cleanly (warning, no rows, no exception) — the core "don't require the
   extension" guarantee.
5. **Idempotency**: re-ingesting the same file inserts nothing new and does not
   trip `_expected_duplicates` validation.

## Risks / open implementation details

- **Object-ref resolution in `to_dataframe()`**: how pynwb surfaces object
  references and the `DynamicTableRegion` in the itertuples rows needs to be
  pinned down empirically when writing the mapping (callable vs. attribute).
  This is the main implementation unknown.
- **Optional FKs**: nullable foreign keys for the optional dichroic mirror /
  filters must use the mixin's default-aware mapping so absent references null
  cleanly.
- **Filter subtypes**: `BandOpticalFilter` vs `EdgeOpticalFilter` share one
  `OpticalFilter` table with nullable subtype fields; confirm both class-name
  strings are matched.
- **`num_samples` / timestamps**: `FiberPhotometryResponseSeries` may use
  `rate`+`starting_time` or explicit `timestamps`; the retrieval helper must
  handle both.
```
