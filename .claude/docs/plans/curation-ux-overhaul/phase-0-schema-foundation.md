# Phase 0 — Schema foundation (identity + rule policy + review profiles)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Small schema additions that later phases depend on for a **correct** identity, a **persistable** missing-value policy, and a single user-facing review profile instead of two versioned parameter names plus display configuration. This phase opens the schema window with Phase 4 and ships first.

**Inputs to read first:**

- [curation.py:1062-1072](../../../../src/spyglass/spikesorting/v2/curation.py) — `_next_curation_id`: `max(existing)+1`, explicitly **not** serialized and **reused** after a delete (or on a concurrent-insert retry). This is why `(sorting_id, curation_id)` is not a permanent identity (B1).
- [curation.py:78-104](../../../../src/spyglass/spikesorting/v2/curation.py) — the `CurationV2` heading `curation_uuid` is added to; `insert_curation:417` (where it is generated) and `reuse_existing:426` (a reused row keeps its existing uuid).
- [metric_curation.py:476-500](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — `AutoCurationRules.Rule` (metric/operator/threshold/label only; `ImmutableParamsLookup`, `insert` unsupported → rows go through `insert_rules`).
- [_params/metric_curation.py:311-324](../../../../src/spyglass/spikesorting/v2/_params/metric_curation.py) — `AutoCurationRuleSchema`, `ConfigDict(extra="forbid")` — a new field must be added here too.
- [unit_matching.py:1345](../../../../src/spyglass/spikesorting/v2/unit_matching.py) — `TrackedUnit.Member`'s `-> CurationV2.Unit` FK: precedent that a part table can DB-enforce a unit reference (used by Phase 4).
- `QualityMetricParameters` and `AutoCurationRules` — the two immutable recipe rows a `CurationReviewProfile` binds for the browser-first paved road. The profile is a real lookup, not the nonexistent evaluation-preset registry or an in-memory alias.

**Contracts referenced:**

- [CurationRef](shared-contracts.md#curationref) (carries `curation_uuid`), [CurationReviewProfile](shared-contracts.md#curationreviewprofile), [Analyzer cache manifest](shared-contracts.md#analyzer-cache-manifest) and [Figure identity](shared-contracts.md#figure-identity) (both key on `curation_uuid`), and [UnsupportedMetricValueError § rule-input validation](shared-contracts.md#unsupportedmetricvalueerror) (`missing_policy`).

## Tasks

- **`curation_uuid` (B1).** Add an immutable `curation_uuid : uuid` column to `CurationV2` via `alter()` (unique per row). `insert_curation` generates a **fresh** token per created row (not content-derived — a delete+recreate of byte-identical content must yield a *new* generation, so the old ref/figure/cache correctly stop matching); a `reuse_existing` hit returns the existing row's uuid. Backfill existing rows with fresh uuids in the migration. `(sorting_id, curation_id)` stays the ergonomic DataJoint key; `curation_uuid` is the **identity of record** carried by `CurationRef`, analyzer cache paths + manifests, FigPack figure identity, and receipts/annotation references.
- **`missing_policy` (B2).** Add `missing_policy : enum('error','fail','pass','ignore')` to `AutoCurationRules.Rule` (default `'error'` for preproduction) and to `AutoCurationRuleSchema` (`_params/metric_curation.py`, which is `extra="forbid"`). Include it in rule **normalization, content identity, and idempotency** comparisons (so two rule sets differing only by policy are distinct). Define the modes precisely and **do not conflate with SI**: SI's `threshold_metrics_label_units(nan_policy='fail')` *labels* a NaN unit as failed and never raises; Spyglass's `'error'` is **fail-fast** (raise an actionable error when a referenced column has no finite values). `'fail'`/`'pass'` label the unit; `'ignore'` skips the rule for that unit.
- **`CurationReviewProfile` (UX1).** Add an immutable, DB-persisted lookup in a DB-only module such as `review_profile.py` (no FigPack optional import), addressed by `review_profile_name` and binding `-> QualityMetricParameters`, `-> AutoCurationRules`, ordered built-in `displayed_unit_properties`, ordered `label_options`, and an explicit label-import mode (`replace` or `overlay`, mapped to the expert layer's `replace`/`inherit`). `replace` consumes the figure's complete seeded label snapshot and permits intentional label removal; `overlay` is only for deliberately partial annotation imports. Normalize the lists, validate every built-in property/label, compute `profile_hash`, reject duplicate content under a second name unless explicitly authorized for shipped aliases, and require a new profile name for changed content. Publishing location (`upload`/local), credentials, ephemeral mode, and curation-specific annotation-set selections are runtime review inputs and **do not** belong in the profile. Ship `franklab_hippocampus_2026_06` (dated, matching the existing preset namespace) resolving the approved Frank-lab rows with `replace`. This is not a general preset/plugin system.
- **Reuse, don't reinvent (name the existing bindings).** A new table is warranted only because the pieces it consolidates are each insufficient alone, and the executor must reuse them rather than re-implement:
  - the `(metric_params_name, auto_curation_rules_name)` binding already exists in-memory as `_PipelinePreset` (`_pipeline_presets.py:22`, fields `:47-48`) and per-curation as `CurationEvaluationSelection` (`metric_curation.py:770`) — the new table exists over `_PipelinePreset` **because** it must be DB-persisted, immutable, and content-addressed, which the in-memory preset is not (state that justification in the module docstring);
  - `label_options` + `displayed_unit_properties` and their `default_label_options()` / `normalize_displayed_unit_properties()` helpers already ship on `FigPackCurationSelection` (`_figpack_curation.py:35`, `:102`) — **reuse the helpers**, do not re-implement list normalization;
  - the `replace`/`overlay` import mode maps to the existing `CurationV2.label_policy` `replace`/`inherit` (`curation.py:429`, documented `:521-528`) — reuse it, don't invent a parallel enum.
- Docs: note all three additions in the PR (additive `alter()`s with defaults, the `curation_uuid` backfill, and the net-new immutable profile lookup).

## Deliberately not in this phase

- Any use of `curation_uuid` in the resolver/FigPack/CurationRef — those consume it in Phases 1–3.
- The `start_review`/FigPack session implementation — Phase 3 consumes `CurationReviewProfile`; Phase 0 only makes profiles persistent and queryable.
- The rule-input **validation logic** that reads `missing_policy` — that is Phase 2; Phase 0 only makes the field persistable.
- The annotation tables and `created_at`/`created_by` — Phase 4.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_curation_uuid_immutable_and_unique` | Every `CurationV2` row has a unique `curation_uuid`; it is set once by `insert_curation` and never rewritten. |
| `test_curation_id_reuse_yields_new_uuid` | Deleting the highest curation and inserting a new one **reuses the numeric `curation_id`** but produces a **different `curation_uuid`** (the generation guard for B1). |
| `test_reuse_existing_keeps_uuid` | `insert_curation(..., reuse_existing=True)` on identical content returns the existing row and its existing `curation_uuid`. |
| `test_rule_missing_policy_persists` | A `Rule` inserted with `missing_policy='error'` round-trips; `AutoCurationRuleSchema` accepts the field; two rule sets differing only in `missing_policy` have distinct identities. |
| `test_review_profile_persisted_and_immutable` | A profile resolves the exact metric/rule rows and ordered display/label configuration; an identical insert is idempotent, changed content under the same name is refused, and the row survives a kernel restart. |
| `test_review_profile_hash_covers_semantics` | Changing either recipe FK, property/label order, or label-import mode changes `profile_hash`; runtime delivery options and curation-specific annotation sets are not accepted or hashed. |
| `test_review_profile_separates_delivery_options` | `upload`/credentials/ephemeral are rejected as profile fields; they remain per-review runtime options. |
| `test_alter_additive` | The `alter()`s add the columns; pre-existing rows read the defaults / backfilled uuids; no data loss. |

## Fixtures

Reuse the existing v2 fixture sort. The reuse test builds a small curation tree, deletes the highest child, and re-inserts to force numeric-id reuse.

## Review

Dispatch `code-reviewer` against the diff. Confirm: `curation_uuid` is immutable + fresh-per-insert and backfilled; `missing_policy` is in schema/model/identity; review profiles are DB-persisted, immutable, normalized, and bind real recipe rows without absorbing delivery configuration; the alterations are additive and documented; no Phase-3 review orchestration leaks in; no plan references in shipped code.
