# Fiber-Photometry Ingestion — Design

**Date:** 2026-07-07
**Status:** Design validated against real NWB files, review-converged, and
**viable on Spyglass's current dependency floor** — with one data-production
constraint: files must embed NWB `core` 2.9.0 (written with pynwb 3.1.x), not
`core` 2.10.0. Verified end-to-end (write + read, with and without the extension
installed) on pynwb 3.1.3. The sample files happen to be `core` 2.10.0 (pynwb-4
toolchain); reading *those* needs a pynwb/hdmf bump that is currently blocked
upstream — but the data doesn't inherently require 2.10.0. See Dependencies.
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
  required columns, FKs to the device tables, the optical fiber's **name +
  insertion metadata stored locally** (all nullable), plus nullable FKs for the
  optional `dichroic_mirror`, `emission_filter`, `excitation_filter` references
  and the optional `coordinates`.
- A signal-reference table for `FiberPhotometryResponseSeries` — stores the NWB
  `object_id`, not the array, retrievable via `fetch_nwb()`.
- **No changes to `common_optogenetics`.** The config stores the fiber identity
  and insertion metadata itself rather than FK-ing into `OpticalFiberImplant`
  (see Decisions) — so this PR is self-contained in `common_photometry`.
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
  to `common_optogenetics.VirusInjection`. Deferred, not designed away.
- Reusing/relocating `common_optogenetics` tables. Earlier drafts FK-ed the
  config into `OpticalFiberImplant`; that reuse repeatedly forced changes to the
  optogenetics tables (positional-id resolution, then relaxing required fields
  the extension marks optional), so the design was **decoupled** — photometry is
  self-contained and touches no optogenetics table.

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
- **Core-schema version matters (measured).** The *sample* files embed `core`
  2.10.0 and the floor `pynwb 3.1.3` cannot read them (fails with `TypeError:
  DatasetBuilder ... 'uint64'`). But an equivalent FP file written as `core`
  2.9.0 (pynwb 3.1.x) **does** read on the floor — verified. See Dependencies:
  the fix is a data-production constraint (produce `core` 2.9.0), not a bump.
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
- **`OpticalFiber.fiber_insertion`** is partly populated: bregma-referenced
  AP/ML/DV and hemisphere present, but **`pitch`/`roll`/`yaw` are `None`** (and
  the fiber model's `active_length`/`ferrule_*` are `None`). `common_optogenetics`
  declares those `NOT NULL` and would drop the fiber — the reason the config
  stores fiber identity + insertion **locally as nullable fields** rather than
  FK-ing into `OpticalFiberImplant` (see Decisions).

## Decisions (resolved during brainstorming + validation)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Depth | Metadata + signal **reference** | Makes fluorescence retrievable via object-id; one focused PR. ~1.7 GB/session confirms not copying arrays into DataJoint. |
| Optogenetics coupling | **None — self-contained** | Reusing `OpticalFiberImplant` repeatedly forced optogenetics changes (positional-id, then required-vs-optional fields), so the fiber link was decoupled; `common_photometry` touches no optogenetics table. |
| Device tables | **Five** (`Indicator`, `ExcitationSource`, `Photodetector`, `DichroicMirror`, `OpticalFilter`) | First three are exercised by the sample data. `DichroicMirror`/`OpticalFilter` cover the extension's optional filter/dichroic refs — **front-loaded** so future files with them need no `alter()`; validated only against a synthetic fixture (no real data populates them). Two tables need custom multi-class matchers: `ExcitationSource` (+ `PulsedExcitationSource`) and `OpticalFilter` (base + `Band` + `Edge`), since `is_nwb_obj_type` is exact-match and would silently miss subtypes. |
| Unmodeled columns | **Ignore + warn** | Even with the optional refs front-loaded, a truly novel column (e.g. `commanded_voltage_series`) must not silently vanish; the override logs it. |
| Model/instance | **Collapse** to one reusable table per type | Spec fields folded from `.model` via object-ref mapping (verified accessible); leaner than a model+instance split. |
| Fiber link | **Store fiber name + insertion metadata locally** on `FiberPhotometryConfig` (all nullable), no FK to `OpticalFiberImplant` | The extension marks all `FiberInsertion` fields optional and the real data leaves pitch/roll/yaw (and fiber `active_length`/ferrule) null, which the optogenetics tables reject; local nullable storage ingests the real data and keeps the PR self-contained. Small duplication if a fiber is also used by optogenetics (rare, content-consistent). |
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
declarative mapping cannot express — extracting the fiber's name + nested
`fiber_insertion.*` fields, the optional per-row filter/dichroic refs, and the
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
  -> Indicator
  -> ExcitationSource
  -> Photodetector
  -> [nullable] DichroicMirror                                       # optional ref
  -> [nullable] OpticalFilter.proj(emission_filter_name='optical_filter_name')   # optional
  -> [nullable] OpticalFilter.proj(excitation_filter_name='optical_filter_name') # optional
  optical_fiber_name: varchar(80)    # the OpticalFiber instance name (local, no FK)
  location: varchar(255)             # OpticalFiber.description (nullable in source)
  hemisphere=null: enum('left','right')  # insertion metadata (all nullable per spec)
  ap_location=null: float
  ml_location=null: float
  dv_location=null: float
  pitch=null: float
  roll=null: float
  yaw=null: float
  excitation_wavelength_in_nm: float
  emission_wavelength_in_nm: float
  coordinates=null: blob             # optional 3-vector for multi-fiber arrays
```

Ingestion is a custom `generate_entries_from_nwb_object()` override (per the
Architecture note). Device FK values come from the resolved object-refs
(`row.indicator.name`, `row.excitation_source.name`, `row.photodetector.name`).
The **fiber is stored locally, not FK-ed**: `optical_fiber_name ←
row.optical_fiber.name`, `location ← row.optical_fiber.description`, and the
insertion fields from `row.optical_fiber.fiber_insertion.*`
(`insertion_position_ap/ml/dv_in_mm`, `hemisphere`, `insertion_angle_pitch/roll/
yaw_in_deg`) — each nullable, since the extension marks them optional and the
real data leaves several null. The device tables must ingest before this table
(`populate_all_common` ordering), guarded in the override.

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
- **Gating is by matching objects, then embedded namespace.** `insert_from_nwbfile`
  first calls `get_nwb_objects()`; if **no** objects of the target type are found
  (the usual non-photometry case) it returns immediately — a **clean no-op, no
  warning** ([ingestion.py:248-251](../../../src/spyglass/utils/mixins/ingestion.py)).
  Only when matching objects **are** present does it call
  `check_extension_requirements`, which warns and skips **iff the extension
  namespace is below `_extension_requirements` min version**. Because our tables
  match by class name, "objects present" implies "namespace present", so the
  warning path is effectively "photometry file, but `ndx-fiber-photometry`
  older than the required version" — not "file without the namespace".

Consequences:
- `ndx-fiber-photometry==0.2.3` goes in `optional-dependencies.test` only —
  needed solely to *write* the fixture NWB. Not a core dependency.
  (`ndx-ophys-devices` is already a core dependency; unchanged.)
- A stock Spyglass install ingests a **`core` 2.9.0** photometry NWB correctly
  without the extension package (verified on the floor pynwb 3.1.3); ingesting a
  non-photometry file is a clean no-op for these tables. The two guarantees are
  distinct: *not importing the extension* is proven on both 3.1.3 and 4.0; a
  `core` **2.10.0** file additionally needs `pynwb>=4.0.0` (see Dependencies).

## Integration

- **`populate_all_common.py`**: `Indicator`, `ExcitationSource`, `Photodetector`,
  `DichroicMirror`, `OpticalFilter` as parent nodes; `FiberPhotometryConfig`
  after the five device tables (no `OpticalFiberImplant` dependency —
  self-contained); `FiberPhotometryResponseSeries` after `FiberPhotometryConfig`.
- **`common/__init__.py`**: export the new tables.
- **`pyproject.toml`**: `ndx-fiber-photometry==0.2.3` in the `test` extra.
  Optionally bump the existing core dep `ndx-ophys-devices` →
  `ndx-ophys-devices>=0.3.1` for consistency with the `_extension_requirements`
  we reference. (Not strictly required: ingestion never imports the package, and
  gating reads the *file's* embedded namespace, not the installed version — so
  this is a consistency change only.)
- **`pyproject.toml`: no floor bump required** for the supported (`core` 2.9.0)
  path — the feature works on the current `pynwb>=3.1.3` / `hdmf>=3.4.6` floor.
  (A `pynwb>=4.0.0` / `hdmf>=6.1.0` bump would only be needed to read `core`
  2.10.0 files directly, and is currently blocked upstream — see Dependencies.)

## Testing

Fixture-driven, in `tests/common/` (fixture builder uses the `test` extra and
mirrors the real files at reduced size — e.g. 2 fibers × 2 wavelengths, short
`rate`-based series):

1. **Fixture builder** writes a minimal photometry NWB **with pynwb 3.1.x so it
   embeds `core` 2.9.0** (matching the supported floor): excitation-source /
   photodetector / optical-fiber models + instances (with `fiber_insertion`),
   an `Indicator`, **plus** a `DichroicMirror`, a `BandOpticalFilter`, and an
   `EdgeOpticalFilter` (with models) to exercise the front-loaded optional path
   and the multi-class matcher; a `FiberPhotometry` lab-meta container with a
   `FiberPhotometryTable` whose optional `dichroic_mirror` / `emission_filter` /
   `excitation_filter` / `coordinates` columns are populated; and 1-D
   `FiberPhotometryResponseSeries` with a single-row `fiber_photometry_table_region`.
   (A minimal version of this builder is already prototyped and round-trips on
   pynwb 3.1.3 — see the design spike.)
2. **Ingestion**: device, config, and signal rows appear; `FiberPhotometryConfig`
   device FKs resolve to the right devices, `optical_fiber_name` + insertion
   fields are stored locally (null where the NWB source is null), and the
   `.Fiber` part maps the region row.
3. **Retrieval**: `fetch_nwb()` / `fetch1_dataframe()` returns the trace with the
   expected length and time axis derived from `rate` + `starting_time`.
4. **Gating contract**: (a) a non-photometry file (no matching objects) is a
   **clean no-op — no rows, no exception, no warning** (the mixin early-returns
   before the namespace check); (b) a file that *does* contain photometry objects
   but carries `ndx-fiber-photometry` **below** the `_extension_requirements` min
   version is skipped **with a warning** and no rows. (Distinct from the config
   override's own unmodeled-column warning.)
5. **Re-ingestion semantics** (per-table, matching existing Spyglass behavior):
   the shared **device** tables (`_expected_duplicates=True`) skip consistent
   pre-existing rows on a second `insert_from_nwbfile` (and validate divergence);
   the **session-specific** config/response tables (`_expected_duplicates=False`,
   like `Raw`/`VirusInjection`) are governed by the file-level `reinsert` flow —
   a naive re-`insert_from_nwbfile` raises `DuplicateError`. Test both: device
   re-ingest is a clean no-op; session-table re-ingest matches the established
   `reinsert` path.
6. **Null insertion metadata** (the exact real-data shape decoupling fixes): a
   fiber whose `pitch`/`roll`/`yaw` (and model `active_length`/`ferrule_*`) are
   `None` ingests cleanly, with those config columns stored null. `common_optogenetics`
   is untouched, so there is no optogenetics change to regress.
7. **Package-absent import safety** (the core "don't require the extension"
   guarantee — a normal test-extra run has the package installed, so it cannot
   prove this on its own): assert `"ndx_fiber_photometry" not in sys.modules`
   immediately after running the ingestion path on a pre-built fixture (build
   the fixture in a separate step/subprocess so its import doesn't pollute
   `sys.modules`). Complement with a CI job that installs Spyglass **without**
   the photometry package **on the current floor** (`pynwb 3.1.3`) and ingests a
   committed **`core` 2.9.0** fixture file. (Verified during design: a `core`
   2.9.0 FP file reads on pynwb 3.1.3 with `ndx_fiber_photometry` uninstalled —
   class-name match and object-refs intact.)
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
    → each config row stores its own `optical_fiber_name` + insertion, so a
    shared model never conflates fibers; (d) a config table **lacking** the
    optional filter/dichroic columns (the sample-data shape) ingests with those
    FKs left null.

## Dependencies

- **Data-production constraint: files must be `core` 2.9.0 (verified, no bump
  needed).** `ndx-fiber-photometry` data does **not** inherently require `core`
  2.10.0 — it uses only `TimeSeries` + `DynamicTable` (+ `DeviceModel`, which
  exists in `core` 2.9.0). Verified end-to-end on **pynwb 3.1.3** (Spyglass's
  floor): a minimal FP file written with pynwb 3.1.x embeds `core` 2.9.0 and
  reads back correctly — response series, all 7 config columns, and object-ref
  resolution — **and** the no-import guarantee holds (the same file reads with
  the extension package *uninstalled*, class-name match and refs intact). So the
  feature ships on the **current** dependency floor, provided producers write
  with pynwb 3.1.x. The design's `_extension_requirements` gate on the *extension*
  namespaces (present in 2.9.0 files), never on the core version.
- **Why the sample files don't read today (and the 2.10.0 path is blocked).** The
  provided example files were written with a pynwb-4 toolchain, so they embed
  `core` 2.10.0, which needs `pynwb>=4.0.0` → `hdmf>=6.1.0`. But
  **`ndx-franklab-novela` (all releases incl. latest 0.2.4, a mandatory core
  Spyglass dep) pins `hdmf<5`**, so pynwb 4 is currently unsatisfiable in
  Spyglass — a hard upstream conflict, not a Spyglass-code problem (pynwb 4.0's
  code impact is negligible: `get_data_interface` is a local helper using
  `.data_interfaces.get()`, `extensions=`/`ic_electrodes` unused,
  `validate(paths=)` hits are `dandi`, `BehavioralEvents` still reads). So:
  - *Preferred path*: produce/convert photometry data with pynwb 3.1.x (`core`
    2.9.0). Works today. **Existing 2.10.0 files must be regenerated from source**
    with a pynwb-3.1.x toolchain (you cannot down-convert 2.10.0→2.9.0 in one env,
    since pynwb 4 and 3.1.x can't coexist).
  - *Alternative (out of Spyglass's control)*: wait for `ndx-franklab-novela` to
    release an `hdmf`-6-compatible version, then bump `pynwb>=4.0.0` /
    `hdmf>=6.1.0` (and `ndx-optogenetics` `==0.3.0`→`0.4.0`), enabling direct
    2.10.0 reads.

## Risks / open implementation details

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
