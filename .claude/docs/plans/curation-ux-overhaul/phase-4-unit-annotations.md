# Phase 4 — Extensible unit annotations (schema)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design D5](designs.md#d5)

Restores v1's custom-metric flexibility — attaching arbitrary computed quantities to units — as a typed, immutable, content-addressed extension model, kept **separate** from curation identity so stale annotations can never masquerade as scientific state (the failure v1's `insert_curation(metrics=...)` caused via HDF5 write). **This is the schema phase:** it opens a deliberate, owner-approved window in the v2 "re-freeze for lab trials" policy.

**Inputs to read first:**

- [curation.py:78-104](../../../../src/spyglass/spikesorting/v2/curation.py) — the `CurationV2` heading `created_at`/`created_by` are added to; `get_sorting:1903` for the unit-id namespace validation.
- [metric_curation.py:1727](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — `CurationEvaluation.get_metrics`: the built-in metric source the common read interface unions with custom annotations.
- [analysis/v1/unit_annotation.py](../../../../src/spyglass/spikesorting/analysis/v1/unit_annotation.py) — the existing downstream `UnitAnnotation` (annotates merge-table units); the new tables must not collide with or duplicate it — these annotate *curated* units in a curation namespace.
- The DataJoint no-self-referential-FK / schema-policy memories — for how lineage/immutability are expressed without forbidden constructs, and the ALTER/`alter()` mechanics.

**Contracts referenced:**

- [CurationRef](shared-contracts.md#curationref) — annotation sets are keyed to an exact curation.

**Designs referenced:** [D5](designs.md#d5) (the three-table model + read interface).

## Tasks

- Add `created_at` (timestamp) and `created_by` (varchar) columns to `CurationV2` (`curation.py:78-104`) via a DataJoint `alter()`; document the migration in the PR (additive with defaults — existing rows take the default, no backfill of historical authorship). Surface both through the Phase 2 lifecycle helpers (they were derived-only before; now they read real columns where present).
- Create the annotation schema module (e.g. `src/spyglass/spikesorting/v2/unit_annotation.py`) with `UnitAnnotationDefinition`, `CurationUnitAnnotationSet`, and `CurationUnitAnnotationSet.Value` per [D5](designs.md#d5). `Value` uses a **real `-> CurationV2.Unit` FK** (DB-enforced + cascade-deletes with the curation; precedent `TrackedUnit.Member`, `unit_matching.py:1345`) — not app-validation only. Enforce: one typed column set per the definition's `value_type`; content-addressed `set_hash` with the **canonical serialization from D5** (sort by unit_id, include definition name+version+type, normalize NumPy scalars, defined NaN/None/text encoding); definitions **immutable by `version`**; sets immutable (re-run = new set). Schema details: numeric values use **`double`, not MySQL `float`** (single-precision breaks content-addressing); all `varchar` get **explicit lengths**; store the **normalized `producer_parameters`** blob, not only `parameters_hash`; curation **labels stay in the `CurationLabel`/`UnitLabel` path** — annotations are never a back door for labels. (Identity references carry `curation_uuid` per [Phase 0](phase-0-schema-foundation.md) where a durable external reference is needed.)
- Support scalar `float` / `int` / `bool` / `text` value types. Reject a value whose runtime type disagrees with the definition's `value_type`.
- DataFrame IO: `from_dataframe(curation, definition, df)` (validates unit ids + dtype, computes `set_hash`, inserts idempotently) and `to_dataframe(set_key)`.
- Common read interface — **explicit selection, never implicit "latest"** (F4/reviewer): `read_unit_properties(curation, evaluation=EvaluationRef(...), annotation_sets=[...])` requires the caller to name which evaluation and which annotation sets; a curation with multiple evaluations/sets is disambiguated by the caller, not by a silent latest pick. It unions built-in `CurationEvaluation.get_metrics` with the named custom sets keyed on `(curation, unit_id)` without a physical join table, applying **deterministic column-name collision behavior** (namespaced columns or documented precedence, never a silent overwrite). This is what summaries and FigPack `displayed_unit_properties` (Phase 3) consume.
- Expose selected annotations in `CurationV2.summarize_curation` output and as FigPack `displayed_unit_properties` (extends Phase 3's display path; no FigPack schema change needed).
- Extend the Phase-3 review entry points with explicit `annotation_sets=[AnnotationSetRef(...), ...]`. Fold each selected set's immutable `set_hash` into the review/figure config hash and pass only those sets to `read_unit_properties`; never resolve an annotation by name or "latest" implicitly. **An empty `annotation_sets` selection MUST leave the review/figure config hash byte-identical to the Phase-3 value** — so Phase 4 landing does not silently change the identity of every pre-existing Phase-3 figure (only figures that actually select an annotation set get a new identity). Profile-level `displayed_unit_properties` may name the desired column layout, but the curation-specific set selection remains an explicit review input.
- Docs: add a short "custom unit annotations" section to the curation notebook/reference — define an annotation, compute values into a DataFrame, import, read back through the common interface, display in a summary/FigPack column. State explicitly that annotations are **not** curation labels and do not affect curation identity or the merge_id.

## Deliberately not in this phase

- Restoring v1's `insert_curation(metrics=...)` — permanently out of scope (mixes scientific state with computed annotations; caused the HDF5 failure).
- Restoring v1's stored `BurstPair` table — a separate, deferred decision (see [overview non-goals](overview.md#non-goals)).
- Any change to the frozen scientific parameter recipes or their hashing.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_created_columns_alter` | The `alter()` adds `created_at`/`created_by`; pre-existing rows read the defaults; `insert_curation` populates them going forward. |
| `test_annotation_namespace_validation` | A `Value.unit_id` absent from the curation's unit set is rejected. |
| `test_annotation_set_content_addressed` | Re-importing identical (definition, params, values) is idempotent (same `set_hash`, no duplicate); changing a value yields a new set, leaving the old one intact. |
| `test_annotation_immutable` | An in-place update of a committed set is refused. |
| `test_value_type_enforced` | A text value under a `float` definition (and vice versa) is rejected. |
| `test_dataframe_round_trip` | `to_dataframe(from_dataframe(...))` returns the same unit→value mapping and dtype. |
| `test_common_read_interface` | The unified reader returns both `CurationEvaluation` built-ins and a custom set for the same curation, keyed on `(curation, unit_id)`, without a physical join table. |
| `test_read_requires_explicit_selection` | With two evaluations and two annotation sets present, the reader requires explicit `evaluation=`/`annotation_sets=` and never picks a "latest"; a name collision is resolved deterministically (namespaced/precedence), not silently overwritten. |
| `test_double_precision_content_address` | A `value_float` near float32's precision limit round-trips exactly (stored as `double`) so `set_hash` is stable — a `float` column would drift. |
| `test_annotation_in_summary_and_figpack` | A selected annotation appears in `summarize_curation` output and as a FigPack `displayed_unit_properties` column. |
| `test_annotation_not_curation_identity` | Adding/removing an annotation set does not change the curation's `merge_id` or identity hash. |

## Fixtures

Reuse the Phase 1/2 curation tree. Synthesize a small annotation DataFrame (e.g. a per-unit `custom_score`) in `conftest.py` covering float/int/bool/text and an out-of-namespace unit id for the rejection test.

## Review

Dispatch `code-reviewer` against the diff. Confirm: the `alter()` is additive and documented; annotation sets are immutable + content-addressed + namespace-validated; the common reader unions built-ins and customs without a forced physical merge; annotations provably do not touch curation identity/merge_id; the new tables don't duplicate the downstream `UnitAnnotation`; docs state the not-a-label distinction; no plan references in shipped code.
