# Phase 3 — injection metadata (link the fiber's indicator to its viral injection)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Adds viral-injection provenance to the photometry pipeline by **linking**
`FiberPhotometryConfig` to Spyglass's existing session-scoped
`common_optogenetics.VirusInjection` — not by modeling a new injection table. After
this phase, "which construct / titer / site was injected for this photometry fiber"
is one join, and a photometry file that carries injection metadata populates the
shared virus tables (single source of truth).

**Inputs to read first:**

- `docs/superpowers/specs/2026-07-08-fiber-photometry-injection-metadata-design.md` —
  the approved design (rationale, the reuse-in-place decision, the sparse-parent-virus
  document-and-defer decision). This phase implements it.
- [src/spyglass/common/common_photometry.py:311-321](../../../src/spyglass/common/common_photometry.py) —
  `_UNMODELED_ATTRS`; `"indicator": ("viral_vector_injection",)` at `:320` is removed
  by this phase (it becomes modeled).
- [src/spyglass/common/common_photometry.py:325-354](../../../src/spyglass/common/common_photometry.py) —
  the `FiberPhotometryConfig` definition (the FK is added after `-> OpticalFiber` at `:338`).
- [src/spyglass/common/common_photometry.py:361-422](../../../src/spyglass/common/common_photometry.py) —
  `generate_entries_from_nwb_object`; the per-row `entry` dict at `:386-407` gains
  `injection_object_id`, resolved from `record["indicator"]` (already read at `:390`).
- [src/spyglass/common/common_optogenetics.py:247-269](../../../src/spyglass/common/common_optogenetics.py) —
  `Virus` (`← ViralVector`); [src/spyglass/common/common_optogenetics.py:271-305](../../../src/spyglass/common/common_optogenetics.py) —
  `VirusInjection` (`← ViralVectorInjection`, PK `(nwb_file_name, injection_object_id)`).
  **Definitions unchanged by this phase.** Note both require fields the ndx types mark
  optional (`Virus.description`; `VirusInjection.description`/`pitch`/`roll`/`yaw`).
- [tests/common/_photometry_fixture.py:118-125](../../../tests/common/_photometry_fixture.py) —
  `_fiber_photometry` (builds the `FiberPhotometry` lab-meta container; gains the
  optional virus/injection slots); [tests/common/_photometry_fixture.py:289-393](../../../tests/common/_photometry_fixture.py) —
  `build_minimal` (gains an `injection` param).
- [tests/common/conftest.py:552-560](../../../tests/common/conftest.py) —
  the `insert_photometry` factory defaults `raise_err=True`; the sparse-parent-virus
  test must pass `raise_err=False`.

**Contracts referenced:**

- [Config schema](shared-contracts.md#config-schema) — this phase **adds** a nullable
  secondary FK to `FiberPhotometryConfig`; it does **not** change the PK or any column
  phase-2's `.Fiber` FK depends on, so phase-2 is unaffected.

## Tasks

- **Import + FK (no cycle).** In `common_photometry.py`, add
  `from spyglass.common.common_optogenetics import VirusInjection` to the common-import
  block ([:29-31](../../../src/spyglass/common/common_photometry.py)). `common_optogenetics`
  imports only the DB-free `_photometry_nwb` (not `common_photometry`), and is imported
  before `common_photometry` in `common/__init__.py` (`:51` vs `:58`), so the FK resolves
  at declaration time with no cycle. Add the nullable FK after `-> OpticalFiber` (`:338`):

  ```text
      -> OpticalFiber
      -> [nullable] VirusInjection        # viral injection delivering this indicator (if any)
      location: varchar(255)              # the FiberPhotometryTable row site, e.g. 'DLS'
  ```

  The FK's parent PK `nwb_file_name` is already in the config PK, so this adds only a
  nullable `injection_object_id` secondary column.

- **Resolve the link defensively in the override; drop it from the warn list.**
  Remove `"indicator": ("viral_vector_injection",),` from `_UNMODELED_ATTRS` (`:320`) —
  it is now modeled, not deferred (leave the `"excitation_source"` operational-field
  warnings). In `generate_entries_from_nwb_object`, inside the per-row loop
  ([:383-415](../../../src/spyglass/common/common_photometry.py)), resolve the injection
  from the indicator and set the FK **only if the `VirusInjection` row exists** (scoped
  by the session key), else leave it `None`:

  ```python
  for row_id, record in zip(df.index, df.to_dict("records")):
      fiber = record["optical_fiber"]
      insertion = getattr(fiber, "fiber_insertion", None)
      # Injection is optional (0/1 per indicator) and lives in the shared
      # VirusInjection table. Link only if that row actually ingested: a sparse
      # injection/virus is dropped upstream, and an unconditional FK would dangle.
      injection = getattr(record["indicator"], "viral_vector_injection", None)
      injection_object_id = None
      if injection is not None:
          injection_key = {**base_key, "injection_object_id": injection.object_id}
          if VirusInjection & injection_key:
              injection_object_id = injection.object_id
      entry = dict(
          base_key,
          fiber_photometry_name=fiber_photometry_name,
          fiber_id=int(row_id),
          injection_object_id=injection_object_id,
          indicator_name=record["indicator"].name,
          # ... (existing fields unchanged) ...
      )
  ```

  This query is safe: `VirusInjection` is populated before `FiberPhotometryConfig` in
  `populate_all_common` (`:249` before `:252`), so its rows exist when the override runs.
  No gate on `VirusInjection` is added — it *should* ingest photometry injections
  (they are real injections); the only config-side failure mode (dangling FK) is
  removed by the existence check.

- **`populate_all_common` / exports.** No change — `Virus` (`:216`, parent node),
  `VirusInjection` (`:249`), `FiberPhotometryConfig` (`:252`) are already ordered
  correctly, and `VirusInjection` is already exported from `common/__init__.py`.

- **Fixture support (complete code — the container wiring is non-obvious).** Extend the
  builder to carry a spec-conformant injection. Add a helper and thread it through
  `_fiber_photometry` and `build_minimal`:

  ```python
  def _viral_injection(od, fp, suffix, *, sparse_virus=False):
      """A (ViralVector, ViralVectorInjection) pair for a photometry file.

      ``sparse_virus`` omits ``description`` on the vector — spec-valid (optional in
      ndx) but NOT-NULL in Spyglass's ``Virus``, i.e. the sparse-parent case that
      makes ``Virus`` drop the parent.
      """
      vv_kwargs = dict(
          name="ViralVector" + suffix, construct_name="AAV-dLight3.8",
          manufacturer="Addgene", titer_in_vg_per_ml=1.5e13,
      )
      if not sparse_virus:
          vv_kwargs["description"] = "dLight3.8 AAV"
      vv = od.ViralVector(**vv_kwargs)
      inj = od.ViralVectorInjection(
          name="ViralVectorInjection" + suffix, description="NAcc injection",
          location="NAcc", hemisphere="left", reference="bregma",
          ap_in_mm=1.7, ml_in_mm=1.7, dv_in_mm=-6.0,
          pitch_in_deg=0.0, roll_in_deg=0.0, yaw_in_deg=0.0,
          volume_in_uL=0.4, viral_vector=vv,
      )
      return vv, inj
  ```

  Give `_fiber_photometry` (`:118`) optional slots (mirrors the ndx-optogenetics layout):

  ```python
  def _fiber_photometry(fp, table, indicator, viral_vector=None, injection=None):
      kwargs = dict(
          name="fiber_photometry",
          fiber_photometry_table=table,
          fiber_photometry_indicators=fp.FiberPhotometryIndicators(indicators=[indicator]),
      )
      if injection is not None:
          kwargs["fiber_photometry_viruses"] = fp.FiberPhotometryViruses(
              viral_vectors=[viral_vector]
          )
          kwargs["fiber_photometry_virus_injections"] = fp.FiberPhotometryVirusInjections(
              viral_vector_injections=[injection]
          )
      return fp.FiberPhotometry(**kwargs)
  ```

  Add an `injection` param to `build_minimal` (`:289`): `injection=None` |
  `"complete"` | `"sparse_virus"`. When set, build the pair via `_viral_injection`,
  pass `viral_vector_injection=injection` to the `Indicator`, and pass
  `viral_vector`/`injection` into `_fiber_photometry`. (The `Indicator` is created in
  `build_minimal` around `:345`; add the link there.)

- **Docs.** Extend the `common_photometry` docs subsection (`docs/src/Features/Ingestion.md`,
  the Fiber-photometry section) noting `FiberPhotometryConfig` links a fiber's indicator
  to its viral injection in the shared `VirusInjection`, with the retrieval join:
  `FiberPhotometryConfig * VirusInjection` (site: titer/location/coords) and
  `FiberPhotometryConfig * VirusInjection * Virus` (construct). Add a CHANGELOG entry
  under the existing photometry bullets.

## Deliberately not in this phase

- **A thin `fetch_injection()` accessor on `FiberPhotometryConfig`** — optional sugar
  over the documented join; skip unless a consumer needs it.
- **Any change to `Virus`/`VirusInjection` definitions** — reused unchanged (mitigation
  options 2/3 for the sparse-parent case are deferred; see the design doc).
- **The old flat `ndx-fiber-photometry` device schema** — out of scope (community
  converges on the `ndx-ophys-devices` split; overview → Non-goals).
- **dF/F / isosbestic / ratiometric analysis** — still a separate future PR.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_injection_populates_shared_tables` | a photometry file with an injection populates `VirusInjection` (titer/location correct) and its parent `Virus` (`construct_name` correct) from the `FiberPhotometryVirusInjections` / `FiberPhotometryViruses` containers |
| `test_config_injection_link` | a config row whose indicator has an injection has `injection_object_id` set; `(FiberPhotometryConfig * VirusInjection)` yields the right titer/location and `(… * Virus)` the right `construct_name` |
| `test_no_injection_frank_shape` | a file with **no** injection (e.g. `build_full`) ingests cleanly; `injection_object_id` is `None`; no `InsertError` |
| `test_sparse_injection_no_dangling_fk` | an injection missing a `VirusInjection` NOT-NULL field is dropped by `VirusInjection`; the config link is left `None`; no `InsertError` |
| `test_sparse_parent_virus_photometry_survives` | a **complete** injection with a sparse parent `ViralVector` (no `description`), ingested with **`raise_err=False`** + `rollback_on_fail=False`: `Virus`/`VirusInjection` get no row (the pre-existing opto `-> Virus` `InsertError`, logged), **but the photometry config/response rows survive** and `injection_object_id` is `None`. Pins the document-and-defer behavior |

All are `@pytest.mark.slow` (they ingest via `insert_photometry` / `populate_all_common`).

## Fixtures

Extend `tests/common/_photometry_fixture.py` per the Tasks above: the `_viral_injection`
helper, virus/injection slots on `_fiber_photometry`, and the `injection` param on
`build_minimal` (`None` | `"complete"` | `"sparse_virus"`). Synthetic-but-spec-conformant
(the `FiberPhotometry.fiber_photometry_viruses` / `…_virus_injections` slots are defined
by ndx-fiber-photometry 0.2.3) — the same rigor as the existing photometry fixtures; no
real new-schema file carries injection yet.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent) against
the diff. Confirm:
- The FK is nullable and set **only** when the `VirusInjection` row exists (session-key
  scoped); no dangling-FK path.
- `indicator.viral_vector_injection` is removed from `_UNMODELED_ATTRS`; the
  excitation-source warnings still fire.
- `Virus`/`VirusInjection` definitions are untouched; `populate_all_common` ordering is
  unchanged.
- The sparse-parent-virus test uses `raise_err=False` **and** `rollback_on_fail=False`
  and asserts photometry survival (not the destructive rollback path).
- Validation-slice tests pass; all marked `slow`.
- Tests exercise behavior, not tautologies; fixtures in the shared builder
  (`testing-anti-patterns`).
- No docstring/test/module name references this plan or "phase 3".
- Retrieval docs + CHANGELOG updated in this PR.
