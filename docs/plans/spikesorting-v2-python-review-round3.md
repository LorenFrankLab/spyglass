# Spike sorting v2: remaining Python and API consistency issues

Reviewed at `a35bfd34`. This is a read-only source review; only this review note
was added. The first three findings are bounded fixes. The fourth is a lower
priority design cleanup, not a reason to delay release.

## 1. Finish the lossless identifier rule at unit-matching inputs

`_unit_match_planning.py:216–219` still applies `int()` to a caller's manual
curation choice before validating membership. A direct call to the real
`build_unit_match_plan` with an available curation 1 and a manual choice of
`curation_id=1.9` produces a valid plan selecting curation 1, with `ok=True`.

The direct selection API repeats this pattern at `unit_matching.py:337–340`,
for both member indices and curation IDs. The runner repeats normalization
for its warning summary at `_pipeline_run.py:1539–1542`.

This is another occurrence of the identifier issue from the previous review.
The new `_lossless_int` rule protects the curation facade, but these public
unit-matching inputs do not use it. A valid membership check after truncation
cannot detect that the caller supplied a different value.

Suggested scope:

- Apply the established integer-input rule to caller-supplied manual choices
  and direct unit-matching choices, before membership checks or writes.
- Share the small normalization rule where useful; do not create a general
  validation framework or replace casts of database integer columns.
- Keep planner and direct-selection behavior consistent. Derive runner
  reporting from the normalized choices where practical.
- Cover a fractional ID, a boolean, and valid Python/NumPy integer inputs at
  these boundaries. Preserve the existing member-ownership checks.

## 2. Make policy snapshots actually read-only

`analysis_selection.py:49–66` wraps the outer policy catalog in
`MappingProxyType`, but its nested dictionaries and lists remain mutable.
`UnitSelectionReceipt` is frozen, and its policy gets a mapping proxy at line
457, but that policy's lists also remain mutable.

Two reproduced consequences:

```python
custom = dict(V2_UNIT_SELECTION_POLICIES["v2_accepted_single_units"])
custom["exclude_labels"].remove("mua")
# Also removes "mua" from the shipped policy catalog.

receipt.policy["exclude_labels"].clear()
# Succeeds, while included_unit_ids/excluded_units retain the old verdicts.
```

Calling the actual `ensure_v2_unit_selection_policies` with a stubbed table
after the first operation generated a seed row without the MUA exclusion.
Since insertion skips existing rows, a fresh database can get different
default policy content from one already seeded. The second operation makes
the receipt's policy description inconsistent with its recorded verdicts;
it does not itself change the stored database policy or downstream group.

Suggested scope: use tuple label values and read-only inner mappings for the
catalog and receipt snapshot. Convert to lists at the existing database
insertion boundary, and adjust the policy annotations accordingly. A small
local constructor is enough; no recursive freezing utility is needed. Verify
that deriving a custom policy cannot alter the defaults and that a receipt's
policy cannot be edited independently of its verdicts.

## 3. Give visualization errors remediation for the analyzer they use

`_visualization.py:267–269` recommends
`Sorting().add_extensions(sorting_key, missing_list)` for every missing display
extension. That mutates the raw sort's analyzer. It cannot add an extension
to the separate analyzer used for a committed merged curation.

A concrete affected path is `plot_si_quality_metrics` for a merged curation:
the standard merged display extension set does not include `quality_metrics`.
The resolver reads that curation's base cache, so adding the extension to the
raw sort does not satisfy the next plot call. The user incurs computation and
still gets the error. The already-supported `compute_missing=True` option
requests the correct curation-scoped derivative.

Suggested scope: remove or qualify the raw-sort instruction and give advice
supported by the calling visualization. For the SI metric widgets, the
existing `compute_missing=True` route and official `plot_metrics` alternative
are sufficient. Update the formatter test that currently requires the text
`add_extensions`; asserting an obsolete method name preserves the problem.
Do not change analyzer ownership or cache publication to fit the error text.

This finding was established by tracing the actual caller, resolver, extension
catalog, and `Sorting.add_extensions`; it was not exercised against a live
database or a real merged analyzer during this review.

## 4. Optional: give UnitMatchPlan one authoritative set of choices

`_unit_match_planning.py:111–112` stores the chosen curations twice:
`curation_choices` supplies execution, while `rows` supplies `as_dataframe()`.
Both are public and mutable. The builder copies the selected IDs into each
representation separately at lines 330–339.

After building a root plan, the following was reproduced:

```python
plan.curation_choices[0]["curation_id"] = 1
plan.as_dataframe()  # Still displays curation 0.
plan.ok              # Still True.
```

The runner consumes the edited choice, not the displayed one. Its later
database validation still applies; this is a disagreement between review and
execution, not a bypass of membership validation. Editing a returned plan is
not currently documented, so this is lower priority than the first three.

If changing it, store each member's selected identity once and derive the
execution mapping and review table from that representation. Prefer a
read-only plan snapshot whose manual alternatives go through the existing
manual planner, so warnings/errors remain meaningful. Merely adding
`frozen=True` would leave the nested dictionaries and lists mutable. Avoid
adding setters, mutation hooks, or a separate plan-editing framework.

## Verification and release scope

The selected existing suites passed: **33 passed, 1 integration test
deselected** (`test_analysis_selection.py`, `test_visualization_registry.py`,
and `test_unit_match_planning.py`, excluding integration/database markers).
They do not cover the reproduced mutation and fractional-ID cases. The
behavior probes ran in an isolated Python process; policy insertion used a
stub table. No MySQL integration tests were run.

Fix the remaining identifier boundary, policy snapshots, and misleading
visualization advice in small changes. Treat the plan representation as an
optional follow-up. None requires a schema migration or reopening the merge
engine, analyzer cache design, or production v1 interfaces.
