# Phase 2 — `FiberPhotometryResponseSeries` + retrieval (get the traces)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Ships the signal layer on top of phase-1: a signal-reference table storing the
NWB `object_id` (not the array) for each `FiberPhotometryResponseSeries`, a
`.Fiber` part mapping each data column to its config row, and a
`fetch1_dataframe()` retrieval helper. After this phase a user can retrieve the
recorded fluorescence as a time-indexed DataFrame with per-fiber columns.

**Inputs to read first:**

- `docs/superpowers/specs/2026-07-07-fiber-photometry-ingestion-design.md` —
  section "Signal reference — `FiberPhotometryResponseSeries`" (table + `.Fiber`
  part + override + `fetch1_dataframe` behavior incl. the optional-region and
  empty-`.Fiber` fallbacks).
- `src/spyglass/common/common_ephys.py:286-389` — `Raw`, the template
  (`_nwb_table = Nwbfile` at `:297`, `nwb_object()` at `:377`). See
  [appendix.md](appendix.md).
- [appendix.md](appendix.md) — `fetch.py:72-84` (why `_nwb_table` is required)
  and the mixin refs.

**Contracts referenced:**

- [Config schema](shared-contracts.md#config-schema) — the `.Fiber` part FKs into
  `FiberPhotometryConfig`; **`fiber_id` is the row `id`, region data is
  positional** → translate positional index → `id`.

## Tasks

- **`FiberPhotometryResponseSeries`** (the **master**) in `common_photometry.py`
  (`SpyglassIngestion, dj.Imported`, following `Raw`). PK
  `(nwb_file_name, response_series_object_id)`; secondary `name`, `description`,
  `comments` (nullable), `num_samples` (`bigint`), `unit`. **Set
  `_nwb_table = Nwbfile`** (required — the table FKs `-> Session`; see
  [appendix.md](appendix.md) `fetch.py:72-84`) and
  `_extension_requirements = {"ndx-fiber-photometry": "0.2.3"}`. Add a
  `nwb_object(key)` accessor — **use the full key**, `(self & key).fetch1(
  "response_series_object_id")` (or `key["response_series_object_id"]` directly),
  **not** `Raw`'s `nwb_file_name`-only fetch. `Raw` gets away with the latter
  because it is one row per file; photometry files have **many** response series
  per file, so a `nwb_file_name`-only fetch would `fetch1()` multiple rows / return
  the wrong object. **The master owns object discovery**: override
  `get_nwb_objects()` to be photometry-ref-scoped like phase-1 (return `[]`
  without a `FiberPhotometry` container; collect the
  `FiberPhotometryResponseSeries` objects), so non-photometry files no-op.

- **Custom `generate_entries_from_nwb_object()` override on the master**: returns
  an `IngestionEntries` dict with **both** the master row **and** its `.Fiber`
  rows (parent before child, per the mixin contract). Master row:
  name/description/comments/num_samples/unit + `object_id`. `.Fiber` rows: for
  each positional index in `series.fiber_photometry_table_region.data`, translate
  to the config row `id` (per
  [shared-contracts.md#config-schema](shared-contracts.md#config-schema)) and
  emit one `.Fiber` entry. **If `fiber_photometry_table_region is None`**
  (optional in the schema): emit the master row and **no `.Fiber` rows**, and
  **warn** — do not skip or raise.

- **`FiberPhotometryResponseSeries.Fiber` part** — PK `(-> master, region_index)`;
  secondary `-> FiberPhotometryConfig`. **Populated only by the master override
  above — it has no independent ingestion.** Do **not** give the part its own
  `get_nwb_objects()`; the mixin fetches objects from the table being inserted,
  so a part-level discovery would ingest response-series objects out of context.

- **`fetch1_dataframe()` retrieval helper** — returns a time-indexed
  `pandas.DataFrame` from `fetch_nwb()`, time axis from `starting_time` +
  `arange(num_samples)/rate` (also handle explicit `timestamps`). Column labels
  from the `.Fiber` → config row with the deterministic fallback
  `f"{location or optical_fiber_name}_{int(excitation_wavelength_in_nm)}nm"`
  (disambiguate by `fiber_id`); **when `.Fiber` is empty**, fall back to generic
  `f"{series.name}_col{i}"` labels rather than raising.

- **`populate_all_common` wiring** — add `FiberPhotometryResponseSeries` to the
  Session-dependent list **after** `FiberPhotometryConfig` (the entry phase-1
  added near `populate_all_common.py:233-235`).

- **Exports** — add `FiberPhotometryResponseSeries` to the `common_photometry`
  import block and `__all__` in `common/__init__.py`.

- **Docs** — extend the phase-1 docs subsection with a retrieval example
  (`(FiberPhotometryResponseSeries & key).fetch1_dataframe()`), and add a short
  notebook cell or docs snippet showing the round-trip. CHANGELOG entry for the
  signal layer.

## Deliberately not in this phase

- **Any analysis/derived signal** (dF/F, isosbestic correction, downsampling) —
  future PR. This phase stops at retrieving the raw trace.
- **Copying array data into DataJoint** — the table stores `object_id` only;
  arrays stay in the NWB file and are read via `fetch_nwb()` (overview → Goals).
- **`CommandedVoltageSeries`** — deferred + warned (design doc, Scope).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_response_series_ingest` | one master row per `FiberPhotometryResponseSeries`; `object_id`/`num_samples`/`unit` correct; `.Fiber` maps the region's positional index → the right config `fiber_id` |
| `test_fetch1_dataframe_roundtrip` | `fetch1_dataframe()` returns a time-indexed frame; length == `num_samples`; time axis from `rate`+`starting_time`; values match `fetch_nwb()` data; column labeled by `location`+wavelength |
| `test_optional_region_none` | a series with no `fiber_photometry_table_region` → master row inserted, **no `.Fiber` rows**, warning (not skip/raise); `fetch1_dataframe()` returns generic `f"{series.name}_col{i}"` labels |
| `test_nwb_table_set` | `fetch_nwb()` succeeds (regression guard that `_nwb_table = Nwbfile` is set — without it the mixin raises `NotImplementedError`) |
| `test_multiple_series_per_file` | a file with **several** `FiberPhotometryResponseSeries` (the sample shape — 8 per file): each row's `nwb_object(key)` / `fetch1_dataframe()` resolves the **correct** series (guards the `Raw`-style `nwb_file_name`-only fetch, which would `fetch1()` multiple rows) |
| `test_multi_row_region` | a 2-D `[time, n_fibers]` series with a multi-row region → one `.Fiber` row per referenced config row; columns labeled per fiber — *marked if it needs a bespoke fixture* |

## Fixtures

Extend phase-1's fixture builder (`tests/common/`) with: a 1-D single-row-region
series (the sample-data shape), a series **without** a region, and (for
`test_multi_row_region`) a 2-D series with a multi-row region. All `core` 2.9.0.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff.
Confirm:
- `_nwb_table = Nwbfile` is set; `fetch_nwb()`/`fetch1_dataframe()` work.
- Optional-region and empty-`.Fiber` paths behave as specified (no skip/raise;
  generic labels).
- Positional-index → `fiber_id` translation is correct (not treating region data
  as ids).
- "Deliberately not in this phase" honored — no analysis code, no array copy.
- Validation-slice tests pass; bespoke-fixture tests marked.
- Tests exercise behavior, not tautologies; fixtures in `conftest`
  (`testing-anti-patterns`).
- No docstring/test/module name references this plan or "phase 2".
- Retrieval docs/notebook example + CHANGELOG updated in this PR.
