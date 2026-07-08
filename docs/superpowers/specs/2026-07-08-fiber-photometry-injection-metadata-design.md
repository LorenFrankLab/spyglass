# Fiber-photometry injection metadata — design

**Status:** Approved (design); not yet implemented.

Follow-up to `2026-07-07-fiber-photometry-ingestion-design.md`. Adds viral-injection
provenance to the fiber-photometry pipeline by **linking** photometry configuration
to Spyglass's existing, session-scoped `VirusInjection` table — not by modeling a
new injection table.

## Context

`common_photometry` is intended to be the community-shared Spyglass photometry
schema (serving multiple labs — e.g. Frank and Berke — not one lab's pipeline).
Injection metadata (which construct, where, what titer/volume) is a real user need:
the Berke lab's fork stores it, and the `ndx-ophys-devices` `Indicator` carries an
optional `viral_vector_injection` link for exactly this. Phase-1 deliberately
deferred it (warned as unmodeled) because the one available real file (Frank,
`sub-400_ses-119974.nwb`) does not populate it. This design defines how to model it
when a file does, using infrastructure that already exists.

### The shared types are already modeled

`ViralVector` and `ViralVectorInjection` are **modality-agnostic** `ndx-ophys-devices`
containers (used by both photometry indicators and optogenetic effectors). Spyglass
already ingests them:

- `common_optogenetics.Virus` — `_source_nwb_object_type = "ViralVector"`,
  `_expected_duplicates = True` (reusable virus catalog).
- `common_optogenetics.VirusInjection` — `_source_nwb_object_type =
  "ViralVectorInjection"`, session-scoped (PK `(nwb_file_name, injection_object_id)`),
  with `-> Virus`, `location`, `hemisphere`, `ap/ml/dv_location`, `pitch/roll/yaw`,
  `volume`, `titer`. Its declarative mapping already pulls `titer` from the nested
  `viral_vector` link.

Both are already in `populate_all_common`: `Virus` as a parent node; `VirusInjection`
in the Session-dependent block **before** `FiberPhotometryConfig`.

### The extension defines where photometry injections live

`ndx-fiber-photometry` 0.2.3's `FiberPhotometry` (LabMetaData) container has two
optional slots that mirror `ndx-optogenetics` exactly:

- `FiberPhotometryViruses` — holds `ViralVector` objects,
- `FiberPhotometryVirusInjections` — holds `ViralVectorInjection` objects.

The `Indicator.viral_vector_injection` link points to an injection stored there. Because
hdmf refuses to write an orphan (unparented) linked container, a **valid** file that
carries injection metadata necessarily stores the `ViralVectorInjection` (and its
`ViralVector`) as real, parented objects — hence they are in `nwb_file.objects`, and
the existing `Virus` / `VirusInjection` class-name matchers already ingest them with
**no change**. (Verified: the ndx spec defines these containers; the opto test fixture
parents injections the same way and `VirusInjection` ingests them.)

## Decision

**Reuse `common_optogenetics.Virus` / `VirusInjection` in place** (single source of
truth); **do not** add a photometry-specific injection table (that would duplicate the
same biological fact across two schemas). **Do not** relocate `Virus`/`VirusInjection`
to a shared module — that is a destructive migration of already-populated tables in
DataJoint (a table's identity is its `schema.table_name`) and is out of scope.

Add exactly one thing: a **nullable link from `FiberPhotometryConfig` to
`VirusInjection`**.

### Why the link belongs on `FiberPhotometryConfig`, not `Indicator`

The NWB attribute lives on the `Indicator`, but Spyglass's `Indicator` is a **reusable
catalog** keyed by indicator name (`_expected_duplicates = True`), whereas an injection
is a **session-specific surgical event** (`VirusInjection` is `-> Session` +
`injection_object_id`). Putting the FK on `Indicator` would mix reusable reagent
identity with a per-session event. `FiberPhotometryConfig` is the correct
session/fiber/channel context; it already holds the `-> Indicator` reference from which
the injection is resolved.

Injection is per-indicator (0/1). Fibers sharing an indicator share the same
`VirusInjection` row — the FK is a pointer, not copied data, so there is no
denormalization concern.

## Schema change

One nullable FK on `FiberPhotometryConfig`:

```text
-> [nullable] VirusInjection   # viral injection delivering this fiber's indicator
```

`VirusInjection`'s PK is `(nwb_file_name, injection_object_id)`; `nwb_file_name` is
already in the config PK, so the FK adds only a nullable `injection_object_id`
secondary column. This is a schema change to `FiberPhotometryConfig`, but the table is
new on this (undeployed) feature branch — so it is a plain definition edit, with **no
`alter()` / migration** required.

## Ingestion

- `Virus` / `VirusInjection`: **table definitions unchanged.** For a photometry file
  whose `FiberPhotometry` carries `FiberPhotometryViruses` /
  `FiberPhotometryVirusInjections`, they already ingest the `ViralVector` /
  `ViralVectorInjection` objects (parented → discoverable). A photometry-only file with
  injections now populates the shared virus tables — the intended single-source outcome.
  **Caveat — these tables are stricter than the ndx types**, so "free" holds only for
  **complete** virus/injection rows: `Virus.description` is NOT-NULL but *optional* in
  `ViralVector`, and `VirusInjection.description` / `pitch` / `roll` / `yaw` are NOT-NULL
  but *optional* in `ViralVectorInjection`. A spec-valid-but-sparse virus/injection is
  dropped by `_key_has_required_attrs` (see the sparse-parent risk below).
- `FiberPhotometryConfig.generate_entries_from_nwb_object`: resolve the injection from
  the indicator and set the FK **defensively**:

  ```python
  injection = getattr(record["indicator"], "viral_vector_injection", None)
  injection_object_id = None
  if injection is not None:
      injection_key = {**base_key, "injection_object_id": injection.object_id}
      if VirusInjection & injection_key:        # only if actually ingested
          injection_object_id = injection.object_id
  entry["injection_object_id"] = injection_object_id
  ```

  - Resolve with the **session key** (`{**base_key, "injection_object_id": ...}`), not
    `injection_object_id` alone.
  - Set it **only if that row exists**, else leave `None`. `VirusInjection`'s columns
    are all NOT-NULL, so a **sparse** injection is silently dropped by
    `_key_has_required_attrs`; setting the FK unconditionally would then dangle →
    `InsertError`. The existence check makes the link null in that case instead.
  - This query is safe: `VirusInjection` runs before `FiberPhotometryConfig` in
    `populate_all_common`, so its rows exist by the time the config override runs.
- Remove `indicator.viral_vector_injection` from `common_photometry._UNMODELED_ATTRS`
  (it is now modeled, not deferred; keep the warning only for genuinely unmodeled
  attributes such as the excitation-source operational fields).
- **No gate on `VirusInjection` for the config link.** Unlike the optical-fiber gate
  (photometry fibers are *foreign* to the optogenetics implant tables), an injection is
  an injection regardless of modality — `VirusInjection` ingesting a photometry injection
  is desired. The config-side failure mode (sparse *injection* → dropped → dangling
  config FK) is fully handled by the nullable + existence-checked link above. A **second,
  upstream** failure mode — a sparse parent *virus* causing a `VirusInjection` →
  `-> Virus` `InsertError` — is *not* fixed by the config check and is a pre-existing
  opto behavior; handled by **document + defer** (see the sparse-parent risk below).

## Retrieval

Injection *site* details (`titer`, `location`, `ap`/`ml`/`dv`, `volume`) are on
`VirusInjection`: `FiberPhotometryConfig * VirusInjection`. The **`construct_name`** lives
on `Virus` (which `VirusInjection` FKs), so add `* Virus` when construct metadata is
needed: `FiberPhotometryConfig * VirusInjection * Virus`. Optionally add a thin
`FiberPhotometryConfig` accessor returning that joined row for a key; otherwise document
the join. No array data is copied; this is low-cardinality reference metadata.

## `populate_all_common` ordering (already correct)

`Virus` (parent node) → `VirusInjection` (Session-dependent block) →
`FiberPhotometryConfig` (same block, after `VirusInjection`). The config's FK resolves
because `VirusInjection` is populated first. No ordering change needed.

## Fixtures & validation

Extend the photometry fixture builder to add (spec-conformant) `FiberPhotometryViruses`
+ `FiberPhotometryVirusInjections` to the `FiberPhotometry` container and give an
indicator a `viral_vector_injection` link. (Synthetic-but-spec-true, exactly like every
existing photometry fixture — no real new-schema file with injection exists yet.)

| Test | Asserts |
| --- | --- |
| Injection populates the shared tables | `VirusInjection` gets a row (titer/location correct) and `Virus` gets the parent (construct correct) from the photometry file's `FiberPhotometryVirusInjections` / `FiberPhotometryViruses` |
| Config link resolves | a config row whose indicator has an injection has `injection_object_id` set; `(config * VirusInjection)` yields the right titer/location and `(config * VirusInjection * Virus)` the right `construct_name` |
| No-injection (Frank shape) | a file with no injection ingests cleanly; `injection_object_id` is `None`; no `InsertError` |
| Sparse injection → no dangling FK | an injection missing a NOT-NULL field is dropped by `VirusInjection`; the config link is left `None`; no `InsertError` |
| Sparse parent virus (photometry stays safe) | a **complete** injection whose parent `ViralVector` lacks a NOT-NULL field (e.g. `description`), ingested with **`raise_err=False`** and `rollback_on_fail=False` (the `insert_photometry` fixture defaults `raise_err=True`, which would *propagate* the error instead of logging it — the test must override both): pins that `VirusInjection` fails to insert it (the pre-existing opto `-> Virus` `InsertError`, logged) **but the photometry rows survive** — config/response ingest and the config `injection_object_id` is `None`. Documents the contained behavior; robust handling deferred |

## Risks / verify during implementation

- **Nullable FK sharing `nwb_file_name`.** `-> [nullable] VirusInjection` reuses the
  non-null `nwb_file_name` (from the config PK) and adds a nullable `injection_object_id`.
  MySQL treats the FK as unenforced when `injection_object_id` is NULL; when set, both
  columns must match. This is the standard nullable-FK pattern, but verify DataJoint
  declares it as intended.
- **Cross-module import / declaration order.** `common_photometry` must import
  `VirusInjection` from `common_optogenetics` for both the FK declaration and the
  existence check. `common_optogenetics` depends only on the DB-free `_photometry_nwb`
  (not `common_photometry`), so there is no cycle — but confirm import order in
  `common/__init__.py` resolves `VirusInjection` before `FiberPhotometryConfig` is
  declared.
- **Sparse parent virus → `VirusInjection` `InsertError` (decision: document + defer).**
  Spyglass's `Virus`/`VirusInjection` require several fields the ndx types mark optional
  (above). A spec-valid file with a **complete injection but a sparse parent
  `ViralVector`** (e.g. no `description`) makes `Virus` drop the parent, after which
  `VirusInjection` inserts an injection whose `-> Virus` FK is unresolvable →
  `InsertError`. This originates in the **opto tables**, is independent of (and not fixed
  by) the config-side check, and under `populate_all_common(rollback_on_fail=True)` would
  `super_delete` the whole `Nwbfile`. It is a **pre-existing** opto behavior this feature
  merely *exposes* by routing photometry injections through these tables.

  **Chosen handling: document + defer.** The "Sparse parent virus" test pins the current
  behavior and, critically, that the **photometry side stays safe** — in the default
  `rollback_on_fail=False` path the config link is left `None` and the photometry
  config/response rows survive (the opto `InsertError` is contained, not catastrophic).
  Robust handling is left to a follow-up *when a real injection file demands it* (matching
  the design-against-real-data approach used throughout). Two paths remain available then,
  not taken now: a **behavioral gate** on `VirusInjection.get_nwb_objects` that skips an
  injection whose parent virus would be dropped (no schema change, but changes opto
  ingestion behavior); or **relaxing** the ndx-optional columns to nullable on
  `Virus`/`VirusInjection` (a schema change to deployed opto tables → migration).

## Non-goals

- No new injection table; no relocation/migration of `Virus`/`VirusInjection`.
- No **column/definition** changes to `Virus`/`VirusInjection` (that would be
  mitigation option 3, explicitly out of scope). A *behavioral* gate (option 2) remains
  open pending the sparse-parent-virus decision. No changes to the device tables, the
  response-series layer, or the optical-fiber gate.
- No dF/F / isosbestic / ratiometric analysis (still a separate future PR).
- No support for the old flat `ndx-fiber-photometry` device schema (community
  converges on the `ndx-ophys-devices` split).
