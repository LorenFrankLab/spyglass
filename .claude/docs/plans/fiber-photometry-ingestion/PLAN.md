# Fiber-Photometry Ingestion Implementation Plan

**Status:** Not started.

Adds the ability to ingest `ndx-fiber-photometry` NWB data into Spyglass: a
self-contained `common_photometry` module that stores the experimental setup
(devices, indicators, per-fiber configuration) as queryable DataJoint tables and
exposes the recorded fluorescence traces via `fetch_nwb()` / `fetch1_dataframe()`
without copying arrays into the database. Ships on the current dependency floor
for `core` 2.9.0 files. Adds no `common_optogenetics` schema — the only change
there is a behavioral `get_nwb_objects()` gate so its fiber tables skip
photometry fibers.

The full design (schema, field mappings, exhaustive schema-coverage table, and
the rationale from 13 review passes) lives in
`docs/superpowers/specs/2026-07-07-fiber-photometry-ingestion-design.md` — the
plan points into it rather than restating it.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is a
   self-contained execution prompt (inputs to read, contracts, tasks, validation,
   fixtures, review).
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / dependency policy / risks / open questions?**
   [overview.md](overview.md).
4. **Need machinery line-refs or external-schema pointers?**
   [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals/non-goals, dependency policy (test-extra
  only; no pynwb bump), integration points, risks, open questions.
- [shared-contracts.md](shared-contracts.md) — `FiberPhotometryConfig` schema,
  ref-scoped `get_nwb_objects()`, null-safe `.model`, duplicate-validation safety.
- [appendix.md](appendix.md) — `IngestionMixin`/`fetch`/`Raw` line-refs and the
  ndx extension spec pointers + verified fixture prototype.
- Phases (each ships as a separable PR):
  - [phase-1-metadata-tables.md](phase-1-metadata-tables.md) — six device tables
    + `FiberPhotometryConfig` + fixture + wiring; ingest & query the setup.
  - [phase-2-signal-reference.md](phase-2-signal-reference.md) —
    `FiberPhotometryResponseSeries` + `.Fiber` + `fetch1_dataframe`; get the traces.
  - [phase-3-injection-metadata.md](phase-3-injection-metadata.md) — nullable
    `FiberPhotometryConfig -> VirusInjection` link (reuse the shared, session-scoped
    `common_optogenetics.VirusInjection`; no new table); injection provenance.
    Design: `docs/superpowers/specs/2026-07-08-fiber-photometry-injection-metadata-design.md`.

## Deliberately not in this plan

- **Analysis / derived-signal pipeline** (dF/F, ratiometric isosbestic
  correction, motion correction, downsampling, merge-table output) — a natural
  follow-up PR once the metadata + raw-reference layer lands. Revisit when a
  consumer needs derived photometry signals.
- **pynwb/hdmf floor bump** to read `core` 2.10.0 files directly — blocked
  upstream (`ndx-franklab-novela` pins `hdmf<5`); see overview → Dependency policy.
