# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts referenced by both phases. Each appears once here; phases link in by
anchor. **Do not weaken** the invariants marked as such.

Full field-by-field schema and the exhaustive coverage table are in the design
doc (`docs/superpowers/specs/2026-07-07-fiber-photometry-ingestion-design.md`,
sections "Architecture" and "Schema coverage"). This file captures only what
both phases must agree on.

## `FiberPhotometryConfig` schema {#config-schema}

Built in phase-1; phase-2's `FiberPhotometryResponseSeries.Fiber` part FKs into
it. The primary key and the columns phase-2 depends on **must not change** once
phase-1 ships.

Primary key: `(nwb_file_name, fiber_photometry_name, fiber_id)` where
`fiber_photometry_name` is the `FiberPhotometry` lab-meta container name and
`fiber_id` is the `FiberPhotometryTable` row `id` (the `DynamicTable` identifier,
which may be **non-consecutive** — not a positional counter).

Full column list: design doc, "Session-specific config — `FiberPhotometryConfig`".
Invariants phase-2 relies on:

- `fiber_id` is the table row **`id`**, while a response series'
  `DynamicTableRegion` stores **positional** row indices — phase-2 must translate
  positional index → row `id` via the table's id ordering to resolve the
  `.Fiber` FK. **Do not** redefine `fiber_id` as positional.
- One config row per `(container, table-row)`; the six device FKs plus the
  session-local fiber fields are all set by phase-1's custom override.

## Reference-scoped `get_nwb_objects()` for device tables {#ref-scoped-get_nwb_objects}

**Invariant — do not weaken.** Every device table (`Indicator`,
`ExcitationSource`, `Photodetector`, `DichroicMirror`, `OpticalFilter`,
`OpticalFiber`) overrides `get_nwb_objects()` to:

1. Return `[]` if the file has no `FiberPhotometry` lab-meta container (so
   non-photometry / pure-optogenetics files are a clean no-op — this is the fix
   for the High finding in review 11; device tables must **not** match generic
   `nwb_file.objects` of the `ndx-ophys-devices` type).
2. Otherwise collect the distinct objects referenced by the relevant
   `FiberPhotometryTable` column(s), deduped by `name`:
   - `Indicator` ← `indicator`; `ExcitationSource` ← `excitation_source`;
     `Photodetector` ← `photodetector`; `OpticalFiber` ← `optical_fiber`;
     `DichroicMirror` ← `dichroic_mirror`; `OpticalFilter` ← `emission_filter`
     ∪ `excitation_filter`.
3. Be **defensive**: `get_nwb_objects()` runs *before*
   `check_extension_requirements()` in the mixin, so on a file whose
   `FiberPhotometry`/`FiberPhotometryTable` predates the 0.2.3 shape (missing a
   column/attr), return `[]` rather than raise — the version gate on the config /
   response tables then emits the below-min warning.

Because the referenced object *is* the subtype, this subsumes class-name
matching: `excitation_source` may reference `ExcitationSource` **or**
`PulsedExcitationSource`; `emission_filter`/`excitation_filter` may reference base
`OpticalFilter`, `BandOpticalFilter`, or `EdgeOpticalFilter`. No
`is_nwb_obj_type` matching is needed; the `source_class` / `filter_class`
discriminators are derived from the referenced object's class.

A shared helper (e.g. `_referenced_devices(nwb_file, column_names, class_filter=None)`)
implements 1–3 once; each device table calls it with its column name(s).

## Null-safe `.model` extraction {#null-safe-model}

`Device.model` is **optional** in core NWB. Model-derived device columns must be
folded via null-safe callables so a model-less referenced device stores nulls
rather than raising `AttributeError`:

```python
def model_attr(name):
    return lambda o: getattr(o.model, name, None) if getattr(o, "model", None) is not None else None
```

## Duplicate-validation safety for device tables {#dup-safety}

Device tables set `_expected_duplicates = True`, so re-ingest/cross-session
validation compares secondary values with plain `!=`
(`ingestion.py` `_unequal_vals`). Two consequences the schema already encodes —
**do not reintroduce the hazards**:

- **No array/blob columns.** `[2]`-vector specs (`wavelength_range`,
  `reflection_band`, `transmission_band`) are stored as scalar **min/max pairs**
  (`wavelength_min_nm`/`wavelength_max_nm`, etc.), because `np.ndarray != np.ndarray`
  is array-valued and raises "truth value ambiguous" in the validator.
- **Reusable spec only.** Device tables store only values stable for a given
  device `name` across sessions (model-derived specs, identity, discriminators).
  Per-session/instance fields (per-channel `description`, `power_in_W`,
  `intensity_in_W_per_m2`, `exposure_time_in_s`, pulsed operational params) are
  **not** stored here — they would trip divergence validation. See design doc,
  Decisions → "Model/instance".
