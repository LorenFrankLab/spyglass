# Fiber-Photometry Ingestion — Design

**Date:** 2026-07-07
**Status:** Design validated against real NWB files; **open prerequisite** — a
pynwb/hdmf floor bump (`pynwb>=4.0.0`) is required to read the files (see Risks)
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
- Five reusable device/indicator metadata tables (`Indicator`,
  `ExcitationSource`, `Photodetector`, `DichroicMirror`, `OpticalFilter`). The
  last two cover the extension's *optional* filter/dichroic references — modeled
  now (front-loaded) so files that populate them need no future schema change.
- A per-fiber configuration table (the NWB `FiberPhotometryTable`) capturing the
  required columns plus nullable FKs for the optional `dichroic_mirror`,
  `emission_filter`, `excitation_filter` references and the optional
  `coordinates`.
- A signal-reference table for `FiberPhotometryResponseSeries` — stores the NWB
  `object_id`, not the array, retrievable via `fetch_nwb()`.
- A small **additive** enhancement to `common_optogenetics.OpticalFiberImplant`:
  a nullable `optical_fiber_object_id` column so the photometry config can
  resolve the implant FK (see Decisions).
- **Graceful-degradation safeguard:** the config override ignores any
  `FiberPhotometryTable` column it does not model and **logs a warning naming
  those columns**, so a genuinely novel/unmodeled column is visible rather than
  silently dropped.
- `populate_all_common` wiring, exports, tests, and a short docs/notebook cell.

**Out of scope (deliberate, follow-up PRs):**
- Any analysis/derived-signal pipeline (dF/F, ratiometric isosbestic
  correction, motion correction, downsampling, merge tables). A metadata +
  raw-reference layer is the correct first increment; a versioned
  `photometry/v1/` pipeline is a natural second PR.
- `CommandedVoltageSeries` (an acquisition `TimeSeries`, and the optional
  `commanded_voltage_series` config reference). A secondary signal used only by
  modulated-excitation rigs; absent in the sample data. When a file provides it,
  the graceful-degradation safeguard warns that the `commanded_voltage_series`
  column is unmodeled. Add as an additive signal table when needed.
- Viral-vector / injection linkage for indicators — absent in the sample data
  (`Indicator.viral_vector_injection == None`; no `FiberPhotometryViruses`). The
  `ndx-ophys-devices` `Indicator` has an optional `viral_vector_injection` link;
  when a file provides it, this becomes an additive nullable FK from `Indicator`
  to the reused `common_optogenetics.VirusInjection`. Deferred, not designed
  away.
- Relocating the shared `ndx-ophys-devices` tables into a neutral module. We
  reuse `common_optogenetics`' tables **in place**.

## Background: the `ndx-fiber-photometry` data model

Three layers (verified against `ndx-fiber-photometry` 0.2.3 and
`ndx-ophys-devices` 0.3.1):

1. **Devices** (`ndx-ophys-devices`, in `nwbfile.devices` / `nwbfile.device_models`):
   model-instance pattern — `OpticalFiberModel`/`OpticalFiber`,
   `ExcitationSourceModel`/`ExcitationSource`, `PhotodetectorModel`/`Photodetector`,
   plus `Indicator`. Instances carry per-channel description; **spec fields live
   on the model**.
2. **Config** (`ndx-fiber-photometry`, in `nwbfile.lab_meta_data`): a
   `FiberPhotometry` container holding a `FiberPhotometryTable` (`DynamicTable`,
   one row per fiber/channel with object-references to the devices + indicator +
   excitation/emission wavelengths + location).
3. **Signal** (`ndx-fiber-photometry`, in `nwbfile.acquisition`):
   `FiberPhotometryResponseSeries` (a `TimeSeries`; a `fiber_photometry_table_region`
   `DynamicTableRegion` points back into the config table).

## Validated against real data

Two example files (`sub-400_ses-119247.nwb`, `sub-400_ses-119974.nwb`, 1.3 GB
each) were inspected in an isolated `uv` venv (`pynwb` 4.0.0 / `hdmf` 6.1.0)
**with the extension package NOT installed**. Findings:

- **No-import claim proven (on pynwb 4.0.0).** With `ndx_fiber_photometry`
  uninstalled, pynwb reconstructed every typed object from the file-embedded
  spec; `__class__.__name__` matched (`FiberPhotometryResponseSeries`,
  `Indicator`, `OpticalFiber`, …) and all spec fields (`source_type`,
  `numerical_aperture`, `fiber_insertion.insertion_position_ap_in_mm`,
  `Indicator.label`, …) were accessible. This is exactly the surface Spyglass
  ingestion uses.
- **Dependency-floor blocker (measured).** The files embed the `core` 2.10.0
  schema. Spyglass's current floor `pynwb>=3.1.3` / `hdmf>=3.4.6` **cannot read
  them**: `pynwb 3.1.3` (the highest 3.x, pulling `hdmf 4.3.1`) fails with
  `TypeError: DatasetBuilder ... incorrect type for 'data' (got 'uint64')`,
  before any type matching. Only `pynwb>=4.0.0` / `hdmf>=6.x` read the files.
  So ingestion requires a **major pynwb/hdmf floor bump** (see Risks) — the
  "stock install ingests correctly" claim below holds only on the bumped floor.
- **Object-ref resolution works.** `FiberPhotometryTable.to_dataframe()` returns
  the referenced device containers as cell values, so `row.indicator.name` etc.
  yield the FK values. (Previously the design's main unknown — now resolved.)
- **Config table is flat (7 columns):** `location`,
  `excitation_wavelength_in_nm`, `emission_wavelength_in_nm`, `indicator`,
  `optical_fiber`, `excitation_source`, `photodetector`. No dichroic / filter /
  coordinates / notes columns; no such devices present.
- **Both files:** 4 fibers (DLS, DMS, NAc, PL) × 2 excitation wavelengths
  (isosbestic 415/420 nm + signal 490 nm) = 8 config rows and 8 **1-D**
  `FiberPhotometryResponseSeries`, each ~26.7 M samples `float64`, `unit='V'`,
  `rate≈6024.1 Hz` + `starting_time` (no `timestamps`), each region referencing
  a **single** config row.
- **`OpticalFiber.fiber_insertion`** is populated (bregma-referenced AP/ML/DV,
  hemisphere) → optogenetics' `OpticalFiberImplant` mapping will ingest it.
- **Implant identity gap** (see Decisions): `OpticalFiberImplant`'s `implant_id`
  is a positional counter and it stores only the *model* name (shared by all 4
  fibers), so the config table needs another key to resolve the correct implant.

## Decisions (resolved during brainstorming + validation)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Depth | Metadata + signal **reference** | Makes fluorescence retrievable via object-id; one focused PR. ~1.7 GB/session confirms not copying arrays into DataJoint. |
| Shared devices | **Reuse `common_optogenetics` tables in place** | Already ingest the shared `ndx-ophys-devices` types. No duplication, no migration. |
| Device tables | **Five** (`Indicator`, `ExcitationSource`, `Photodetector`, `DichroicMirror`, `OpticalFilter`) | First three are exercised by the sample data. `DichroicMirror`/`OpticalFilter` cover the extension's optional filter/dichroic refs — **front-loaded** so future files with them need no `alter()`; validated only against a synthetic fixture (no real data populates them). Two tables need custom multi-class matchers: `ExcitationSource` (+ `PulsedExcitationSource`) and `OpticalFilter` (base + `Band` + `Edge`), since `is_nwb_obj_type` is exact-match and would silently miss subtypes. |
| Unmodeled columns | **Ignore + warn** | Even with the optional refs front-loaded, a truly novel column (e.g. `commanded_voltage_series`) must not silently vanish; the override logs it. |
| Model/instance | **Collapse** to one reusable table per type | Spec fields folded from `.model` via object-ref mapping (verified accessible); leaner than a model+instance split. |
| Fiber link | `FiberPhotometryConfig` **FKs into `OpticalFiberImplant`** | Single source of truth for the implanted fiber + its stereotactic coordinates. |
| Implant resolution | **Add nullable `optical_fiber_object_id` to `OpticalFiberImplant`** | Additive **schema migration** (DataJoint `alter()`, no PK change / no data copy); pre-existing rows are `NULL` and need re-ingest/backfill to cross-link. Lets the config resolve the implant by the fiber instance's `object_id`, robustly. |
| Extension dep | `ndx-fiber-photometry==0.2.3` in the **`test` extra only** | Ingestion never imports it (verified); needed only to build the test fixture. |
| Config PK | `(Session, fiber_photometry_name, fiber_id)` | `fiber_photometry_name` disambiguates multiple `FiberPhotometry` containers per file (rare) so two tables' row `id`s can't collide; degenerates to one constant name in the common case. |
| Re-ingestion | Device tables `_expected_duplicates=True`; config/response `False` | Matches existing Spyglass session-specific tables (`Raw`, `VirusInjection`); idempotency via file-level `reinsert`, not per-row skip. |

## Architecture

New module `src/spyglass/common/common_photometry.py`, schema
`common_photometry`, built on the `SpyglassIngestion` mixin like
`common_optogenetics`. **Not** a pure declarative reuse: three device tables use
only the declarative `table_key_to_obj_attr` mapping; `ExcitationSource` and
`OpticalFilter` override `get_nwb_objects()` to match their **subtype** class
names, since `is_nwb_obj_type` does exact string equality and would silently miss
subtypes (`ExcitationSource` must also catch `PulsedExcitationSource`;
`OpticalFilter` must catch base `OpticalFilter`, `BandOpticalFilter`, and
`EdgeOpticalFilter`); and `FiberPhotometryConfig` and
`FiberPhotometryResponseSeries` require custom
`generate_entries_from_nwb_object()` overrides, because they need work the
declarative mapping cannot express — a DB lookup to resolve the
`OpticalFiberImplant` FK, the optional per-row filter/dichroic refs, and the
`DynamicTableRegion` → config-row mapping.
Precedent exists in the same base:
[`OpticalFiberImplant.generate_entries_from_nwb_object`](../../../src/spyglass/common/common_optogenetics.py)
overrides to assign `implant_id`, and `OptogeneticProtocol` uses a custom
`make`. Note the mixin **raises** `ValueError` on a missing nested object-ref
([`ingestion.py`](../../../src/spyglass/utils/mixins/ingestion.py)), so any
optional ref must be handled in the override, not mapped as a separate object
key.

### Reusable device tables

Keyed by device `name`, `_expected_duplicates = True`,
`_extension_requirements = {"ndx-ophys-devices": "0.3.1"}`. Each reads the
`ndx-ophys-devices` **instance** and folds in useful `.model` spec fields via
the `table_key_to_obj_attr` object-ref mapping (as `OpticalFiberImplant` reads
`model.name`). Model-carried fields use callables
(`lambda o: o.model.source_type`); optional fields use the mixin's
`(attr, default)` tuple form so missing values null cleanly.

- **`Indicator`** ← `Indicator`: `indicator_name`←`name`, `label`, `description`,
  `manufacturer`.
- **`ExcitationSource`** ← `ExcitationSource` **or** `PulsedExcitationSource`
  (+ model), via a custom `get_nwb_objects()` matching both: `excitation_source_name`
  ←`name`, `description` (instance, per-channel), `manufacturer`, `source_type`,
  `excitation_mode` (from model), a `source_class` `enum('continuous','pulsed')`
  discriminator, and nullable pulsed-only fields `pulse_rate_in_Hz`,
  `peak_power_in_W`, `peak_pulse_energy_in_J`. Matters because `excitation_source`
  is a **required** config ref — an unmatched pulsed source would break the FK.
- **`Photodetector`** ← `Photodetector` (+ model): `photodetector_name`←`name`,
  `description`, `manufacturer`, `detector_type`, `gain` (from model, nullable).
- **`DichroicMirror`** ← `DichroicMirror` (+ model): `dichroic_mirror_name`
  ←`name`, `manufacturer`, `cut_on_wavelength_in_nm`, `cut_off_wavelength_in_nm`
  (from model, all nullable). Default class-name matcher.
- **`OpticalFilter`** ← base `OpticalFilter`, `BandOpticalFilter`, **or**
  `EdgeOpticalFilter` (+ model), via a custom `get_nwb_objects()` matching all
  three class names (the config target type is the base `OpticalFilter`, so a
  plain instance must match too): `optical_filter_name`←`name`, `filter_class`
  `enum('base','band','edge')` (from the matched class), `filter_type` (from
  `OpticalFilterModel`), `manufacturer`, and the subtype-specific nullable fields
  `center_wavelength_in_nm` / `bandwidth_in_nm` (band) and `cut_wavelength_in_nm`
  (edge). Specs live on the model; folded via callables. Untested against real
  data (no sample file populates filters) — exercised by a synthetic fixture only.

### Optogenetics enhancement (additive)

`common_optogenetics.OpticalFiberImplant` gains one nullable secondary column:

```text
optical_fiber_object_id=null: varchar(40)   # NWB object id of the OpticalFiber instance
```

mapped in `table_key_to_obj_attr["self"]` from the instance `object_id`. This is
no PK change and no data copy, but it **is** an additive schema migration
(DataJoint `alter()`). Pre-existing `OpticalFiberImplant` rows get `NULL` for the
new column, so cross-linking a file whose implants were ingested **before** this
change requires re-ingesting/backfilling those rows. Within a single
`populate_all_common` run the value is populated as the implant ingests (before
the config reads it), so freshly-ingested files resolve fine.

### Session-specific config — `FiberPhotometryConfig`

The NWB `FiberPhotometryTable`, one row per fiber/channel. The custom override
iterates `FiberPhotometryTable.to_dataframe()` rows itself — it does **not** rely
on the mixin's default `DynamicTable` recursion (which cannot do the FK lookups
and optional-ref handling below).

```text
FiberPhotometryConfig
  -> Session
  fiber_photometry_name: varchar(64) # lab-meta container name (usually one/file)
  fiber_id: int                      # the FiberPhotometryTable row `id` (see note)
  ---
  -> OpticalFiberImplant             # common_optogenetics (reused); resolved via
                                     #   row.optical_fiber.object_id
  -> Indicator
  -> ExcitationSource
  -> Photodetector
  -> [nullable] DichroicMirror                                       # optional ref
  -> [nullable] OpticalFilter.proj(emission_filter_name='optical_filter_name')   # optional
  -> [nullable] OpticalFilter.proj(excitation_filter_name='optical_filter_name') # optional
  location: varchar(255)
  excitation_wavelength_in_nm: float
  emission_wavelength_in_nm: float
  coordinates=null: blob             # optional 3-vector for multi-fiber arrays
```

Ingestion is a custom `generate_entries_from_nwb_object()` override (per the
Architecture note). Required FK values come from the resolved object-refs
(`row.indicator.name`, `row.excitation_source.name`, `row.photodetector.name`);
the `OpticalFiberImplant` FK is resolved with a DB lookup on
`(nwb_file_name, optical_fiber_object_id = row.optical_fiber.object_id)` — hence
the additive column above. The device tables and `OpticalFiberImplant` must
ingest before this table (`populate_all_common` ordering), guarded in the
override.

**Optional refs** (`dichroic_mirror`, `emission_filter`, `excitation_filter`,
`coordinates`) are read only when the column exists on the `FiberPhotometryTable`
**and** the per-row value is non-null; otherwise the FK/attr is left null (never
mapped as a separate object key, which would raise). **Any other column** the
override does not recognize triggers a one-time warning naming it (the
graceful-degradation safeguard) so no metadata is silently dropped.

**Re-ingestion.** `FiberPhotometryConfig` and `FiberPhotometryResponseSeries`
are session-specific, so — like `Raw`, `VirusInjection`, and `OpticalFiberImplant`
— they leave `_expected_duplicates = False` (the flag means "shared across
sessions"; the reusable device tables set it `True`). Idempotency for these is
therefore governed by the standard file-level flow (`insert_sessions` skips
already-ingested files; `reinsert=True` re-runs), **not** by per-row duplicate
validation; a naive second `insert_from_nwbfile` on the same file would
`DuplicateError` (caught and logged by `populate_all_common`), exactly as for the
existing session-specific tables.

**`fiber_id`** is the `FiberPhotometryTable`'s row `id` (the `DynamicTable`
identifier), which may be **non-consecutive** — not a positional 0..n counter.
This matters because the response-series `DynamicTableRegion` stores **positional
row indices**, so the region → config mapping must translate positional index →
row `id` via the table's id ordering.

**`fiber_photometry_name`** disambiguates the (rare) case of multiple
`FiberPhotometry` lab-meta containers in one file: without it, two tables' row
`id` 0 would collide on `(Session, fiber_id)`. The config table matches **all**
`FiberPhotometryTable` objects (`get_nwb_objects` over the file) and takes the
name from each table's parent container; the response-series `.Fiber` FK carries
the same name (resolved from the region's target table). Usually there is exactly
one container, so this degenerates to a single constant name.

### Signal reference — `FiberPhotometryResponseSeries`

`dj.Imported`, following [`Raw`](../../../src/spyglass/common/common_ephys.py)
(`raw_object_id: varchar(40)` + `nwb_object()` accessor).

```text
FiberPhotometryResponseSeries
  -> Session
  response_series_object_id: varchar(40)   # NWB object id, for fetch_nwb()
  ---
  name: varchar(80)
  description: varchar(2000)
  num_samples: bigint                       # ~2.7e7 in sample data
  unit: varchar(16)

  class Fiber(dj.Part):                     # the fiber_photometry_table_region
    -> master
    region_index: int                       # position within the region (0-based)
    ---
    -> FiberPhotometryConfig                # the referenced config row
```

Ingestion is a custom `generate_entries_from_nwb_object()` override: for each
positional index in `series.fiber_photometry_table_region.data`, translate to
the config row `id` (per the non-consecutive-id note above) and emit a `.Fiber`
part row. In the sample data each series references exactly one config row (one
`.Fiber` part row); the part table generalizes to the extension's optional 2-D
`[time, n_fibers]` case (multiple part rows) without schema change.

Retrieval helper `fetch1_dataframe()` returns a `pandas.DataFrame` indexed by
time — computed from `starting_time` + `arange(num_samples) / rate` (the sample
data uses `rate`; the helper also handles explicit `timestamps`) — with the
fluorescence column(s) labeled by the `.Fiber` → config `location` /
wavelength, assembled from `fetch_nwb()`.

## Optional-dependency guarantee (verified)

The explicit requirement — *do not import `ndx-fiber-photometry` unless the user
needs it* — is satisfied by the existing ingestion machinery and **verified on
the real files** (read succeeded with the package uninstalled):

- **Type matching is by class-name string.**
  [`is_nwb_obj_type`](../../../src/spyglass/utils/nwb_helper_fn.py) compares
  `nwb_object.__class__.__name__ == "FiberPhotometryResponseSeries"`. pynwb
  reconstructs typed containers from the spec **embedded in the NWB file**, so
  matching (and spec-field access) works without the extension installed.
- **Gating is by file-embedded namespace.**
  [`check_extension_requirements`](../../../src/spyglass/utils/mixins/ingestion.py)
  reads namespace versions (`get_file_namespaces` → `load_namespaces`) and
  compares against `_extension_requirements`. A file lacking the namespace (or
  below min version) is **skipped with a warning, not an error**.

Consequences:
- `ndx-fiber-photometry==0.2.3` goes in `optional-dependencies.test` only —
  needed solely to *write* the fixture NWB. Not a core dependency.
  (`ndx-ophys-devices` is already a core dependency; unchanged.)
- A stock Spyglass install **on the bumped pynwb/hdmf floor** (see the
  dependency-floor blocker above and Risks) ingests a photometry NWB correctly
  without the extension package; ingesting a non-photometry file is a clean no-op
  for these tables. Note the two guarantees are distinct: *not importing the
  extension* is proven; *reading the file at all* needs `pynwb>=4.0.0`.

## Integration

- **`populate_all_common.py`**: `Indicator`, `ExcitationSource`, `Photodetector`,
  `DichroicMirror`, `OpticalFilter` as parent nodes (alongside
  `OpticalFiberDevice`, `Virus`); `FiberPhotometryConfig` after
  `OpticalFiberImplant` **and** the device tables; `FiberPhotometryResponseSeries`
  after `FiberPhotometryConfig`.
- **`common/__init__.py`**: export the new tables.
- **`pyproject.toml`**: `ndx-fiber-photometry==0.2.3` in the `test` extra.
  Optionally bump the existing core dep `ndx-ophys-devices` →
  `ndx-ophys-devices>=0.3.1` for consistency with the `_extension_requirements`
  we reference. (Not strictly required: ingestion never imports the package, and
  gating reads the *file's* embedded namespace, not the installed version — so
  this is a consistency change only.)
- **`pyproject.toml` floor bump (load-bearing, not cosmetic)**: `pynwb>=4.0.0`
  and `hdmf>=6.x` are **required** to read `core` 2.10.0 photometry files (see the
  dependency-floor risk). This is the one dependency change that gates the feature
  working at all, and it must be validated suite-wide.

## Testing

Fixture-driven, in `tests/common/` (fixture builder uses the `test` extra and
mirrors the real files at reduced size — e.g. 2 fibers × 2 wavelengths, short
`rate`-based series):

1. **Fixture builder** writes a minimal photometry NWB: excitation-source /
   photodetector / optical-fiber models + instances (with `fiber_insertion`),
   an `Indicator`, **plus** a `DichroicMirror`, a `BandOpticalFilter`, and an
   `EdgeOpticalFilter` (with models) to exercise the front-loaded optional path
   and the multi-class matcher; a `FiberPhotometry` lab-meta container with a
   `FiberPhotometryTable` whose optional `dichroic_mirror` / `emission_filter` /
   `excitation_filter` / `coordinates` columns are populated; and 1-D
   `FiberPhotometryResponseSeries` with a single-row `fiber_photometry_table_region`.
2. **Ingestion**: device, config, and signal rows appear; `FiberPhotometryConfig`
   FKs resolve to the right devices and to the correct `OpticalFiberImplant`
   (via `optical_fiber_object_id`); the `.Fiber` part maps the region row.
3. **Retrieval**: `fetch_nwb()` / `fetch1_dataframe()` returns the trace with the
   expected length and time axis derived from `rate` + `starting_time`.
4. **Gating contract**: ingesting a file **without** the namespace is skipped
   cleanly (warning, no rows, no exception).
5. **Re-ingestion semantics** (per-table, matching existing Spyglass behavior):
   the shared **device** tables (`_expected_duplicates=True`) skip consistent
   pre-existing rows on a second `insert_from_nwbfile` (and validate divergence);
   the **session-specific** config/response tables (`_expected_duplicates=False`,
   like `Raw`/`VirusInjection`) are governed by the file-level `reinsert` flow —
   a naive re-`insert_from_nwbfile` raises `DuplicateError`. Test both: device
   re-ingest is a clean no-op; session-table re-ingest matches the established
   `reinsert` path.
6. **Optogenetics regression**: `OpticalFiberImplant` still ingests unchanged
   (new column nullable/defaulted); existing optogenetics tests pass.
7. **Package-absent import safety** (the core "don't require the extension"
   guarantee — a normal test-extra run has the package installed, so it cannot
   prove this on its own): assert `"ndx_fiber_photometry" not in sys.modules`
   immediately after running the ingestion path on a pre-built fixture (build
   the fixture in a separate step/subprocess so its import doesn't pollute
   `sys.modules`). Complement with a CI job that installs Spyglass **without**
   the photometry package **on the bumped pynwb/hdmf floor** (`pynwb>=4.0.0`) and
   ingests a committed fixture file. (Validated manually during design on
   `pynwb 4.0.0`: the real files read fully with `ndx_fiber_photometry`
   uninstalled; `pynwb 3.1.3` cannot read them — see the dependency-floor risk.)
8. **Front-loaded optional path + subtype matching**: with the full fixture,
   `DichroicMirror` and `OpticalFilter` rows ingest (matcher captures base
   `OpticalFilter`, `BandOpticalFilter`, **and** `EdgeOpticalFilter`;
   `filter_class` set correctly); a `PulsedExcitationSource` ingests into
   `ExcitationSource` with `source_class='pulsed'` and pulsed fields populated;
   the config's `dichroic_mirror` / `emission_filter` / `excitation_filter` FKs
   resolve and `coordinates` is stored.
9. **Graceful-degradation safeguard**: a fixture whose `FiberPhotometryTable`
   carries an **unmodeled** column (e.g. a `commanded_voltage_series` ref)
   ingests the core row and **logs a warning naming that column**; no exception,
   no silent drop.
10. **Edge cases** (explicit): (a) 1-D single-row-region response series (the
    sample-data shape); (b) **non-consecutive** `FiberPhotometryTable` row `id`s
    → region positional-index translation is correct; (c) **two optical fibers
    sharing one model** (the sample data — all four share `DoricFlatFiber400um`)
    → `OpticalFiberImplant` resolution by `optical_fiber_object_id` picks the
    right implant; (d) a config table **lacking** the optional filter/dichroic
    columns (the sample-data shape) ingests with those FKs left null.

## Risks / open implementation details

- **pynwb/hdmf floor bump (cross-cutting prerequisite)**: reading these `core`
  2.10.0 files requires `pynwb>=4.0.0` / `hdmf>=6.1.0` (measured — the current
  floor `pynwb>=3.1.3` cannot read them). Until resolved, the feature does not
  work on a stock current-floor install.
  - *Spyglass-code impact of pynwb 4.0: negligible* (changelog + `src/` scan):
    Python 3.10+ already required; the removed `ProcessingModule.get_data_interface`
    is **not** used (Spyglass's same-named helper uses `.data_interfaces.get()`);
    `NWBHDF5IO(extensions=)` / `ic_electrodes` / `add_container` unused;
    `pynwb.validate(paths=)` hits are `dandi.validate`, not pynwb; `add_scratch`/
    `ScratchData` still exist (only removed args, not used); `BehavioralEvents`
    (DIO/sensors) is deprecated-not-removed and still **reads** fine.
  - *The actual unknown is transitive*: `hdmf>=6.1.0` pulls pandas-3.0-era
    constraints; whether SpikeInterface / DataJoint / the other `ndx-*` pins
    co-resolve needs a **resolve-and-run spike** (plan phase 0). Given the
    negligible code impact, if the spike is clean the bump is small enough to be
    its own quick prerequisite PR that photometry builds on.
- **`OpticalFiberImplant` ordering vs `FiberPhotometryConfig`**: the config FK
  resolution depends on `OpticalFiberImplant` (and its new `object_id` column)
  being populated first — enforced by `populate_all_common` ordering and a guard
  in the config `make`.
- **Multiple excitation sources sharing a model**: 4 instances / ≤3 models; the
  collapsed device table keys by instance name and denormalizes model specs —
  acceptable, but re-ingest validation (`_expected_duplicates`) must compare
  folded values consistently.
- **`num_samples` / timing source**: sample data uses `rate` + `starting_time`;
  the retrieval helper must also handle explicit `timestamps` for other rigs.
- **Region cardinality**: sample data is 1-D single-row regions; the `.Fiber`
  part supports multi-row (2-D) regions but that path is untested against real
  multi-fiber files.
- **Front-loaded optional devices are unvalidated against real data**: neither
  sample file populates `DichroicMirror`/`OpticalFilter` or the optional config
  columns, so the `OpticalFilter` multi-class matcher, the `filter_class`
  discriminator, the model-spec fold for band vs edge, and the nullable optional
  FKs are exercised **only by a synthetic fixture**. Revisit the field mapping
  when a real filter/dichroic-bearing file is available.
- **Optional per-row refs**: an optional column may exist on the table while an
  individual row's ref is null; the override must null the FK per-row, not only
  per-column.
