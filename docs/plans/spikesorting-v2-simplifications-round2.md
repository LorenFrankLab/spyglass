# Spike sorting v2: additional simplification candidates

Audited at `fd8e571c`, after the seven items in
`spikesorting-v2-simplifications.md` were implemented. These are additional
maintenance opportunities, not release blockers or a request to redesign the
pipeline. This audit changes no pipeline code or tests.

If another cleanup patch is wanted, prioritize items 1–3. Item 4 is a small
query refactor that needs database validation. Item 5 is useful but can wait
until merge handling is next being changed.

## 1. Share preparation of validated parameter rows

Evidence:

- `metric_curation.py:327–368` prepares QualityMetricParameters rows for
  insertion. `_pipeline_reporting.py:458–495` repeats the payload assembly,
  validation, default filling, and conversion into stored columns.
- `metric_curation.py:494–524` prepares AutoCurationRules master and part rows.
  `_pipeline_reporting.py:498–537` repeats the master preparation for the
  default-catalog audit.

The reporting helper docstrings explicitly say they mirror insertion. A new
validated field or default currently requires synchronized edits to both
paths, or the audit can report false drift or fail to check a new field.

Extract two small DB-free preparation functions: one returning a normalized
quality-metric row, the other returning a normalized rules master and its
ordered part rows. Use them in insertion and default verification. Suitable
homes are the existing metric parameter/service modules; a new generic
parameter-table framework is unnecessary.

Keep duplicate checks, relational reads, transactions, and writes in the table
methods. Preserve omitted-versus-explicit `template_metric_columns`, schema
versions, job kwargs, rule order, and missing-value policy. Share preparation
of expected values; keep the comparison against fetched stored values intact.

Verification: current writer and audit code produced identical rows for all
three shipped quality defaults and four rules masters in a DB-free probe.
After refactoring, run parameter validation/identity tests and the existing
`test_verify_v2_default_catalog_flags_validated_and_part_drift` integration
test. Check an explicit non-default template-column payload as well as omitted
columns.

There is related reflection in `_shipped_default_rows` at
`_pipeline_reporting.py:540`. If changing that dispatch, wire the eight known
tables to explicit row providers in the existing catalog list. Missing stored
defaults remain legitimate (for example, an unavailable sorter); malformed
internal default metadata should not silently become an empty audit. This is
optional follow-up, not required for the shared preparation functions.

## 2. Remove a bounded set of obsolete private Recording forwarding methods

Evidence: `recording.py:2247–2315` contains four methods that only forward their
arguments to existing services:

- `_spikeinterface_channel_ids` -> `_recording_geometry.spikeinterface_channel_ids`
- `_fetch_sort_group_probe_info` -> `_recording_geometry.fetch_sort_group_probe_info`
- `_maybe_apply_tetrode_geometry` -> `_recording_geometry.maybe_apply_tetrode_geometry`
- `_filtering_description` -> `_recording_preprocessing.filtering_description`

Their docstrings largely explain why the forwarding methods survived extraction.
Repository search finds table-internal callers and v2 tests; the channel-ID
forwarder has no runtime caller in the repository. These methods add a second
place to navigate and maintain without adding behavior.

Call the services directly, migrate the tests to those functions, and remove
these four draft-private methods. Keep a sensible import alias or rename the
local filtering-description variable to avoid shadowing the service function.

This preserves the service algorithms and keeps calls in their existing
fetch/compute phase. Public conveniences such as plotting methods and
`get_analyzer` remain useful entry points. Do not use this item as a reason to
remove every forwarding method, runtime adapter, or v0/v1 interface.

Verification: direct filtering-description coverage already exists in
`test_recording_services.py`. Retain channel-name mapping, tetrode geometry,
stable probe-info ordering, and preprocessing provenance assertions while
moving their imports to the service layer. Preserve any deliberate test
injection at the actual call site.

## 3. Stop passing values that the receiving function never uses

An AST read-use check, followed by caller inspection, found these concrete
cases:

| Value | Receiver | Bounded change |
| --- | --- | --- |
| `model_cls` | `_lookup_validation.py:52`, `_assert_schema_version_matches` | Remove the unused argument and its inaccurate fallback documentation. Its only repository caller already supplies a validated row. Keep the outer/inner version comparison. |
| `sort_group_channel_ids` | `_recording_preprocessing.py:45` / `:148`, `apply_temporal_preprocessing` / `apply_spatial_preprocessing` | Done: the pre-motion helper was split into these two and neither takes the channel list. The recording is already channel-restricted here. The channel list stays where restriction and geometry actually use it. |
| `rounding` | `recompute.py:152`, `_recompute_compute` | Remove only this generic helper argument. Keep rounding in analyzer selection/provenance and `_recompute_analyzer_hashes`; the regeneration closure already captures it there. |
| `analyzer_folder` | `sorting.py:1945`, `Sorting.make_insert` | Remove it from `SortingComputed`, its construction, and the matching insertion signature. `make_compute` already derives the unit rows, and insertion neither reads nor cleans the shared analyzer folder. |

For the final case, keep the local `analyzer_folder` used to build/publish the
analyzer and derive unit metadata. The change removes only its trip into
insertion. Update the now-stale `SortingComputed` explanation about insertion
loading/cleaning that folder.

Do not mechanically remove unused `key` parameters from DataJoint methods:
those are part of the framework's calling convention. Likewise, adjust both
sides of the positional `SortingComputed` -> `make_insert` contract together.

Verification: preprocessing-order and outer-version tests, recompute outcome
tests, `test_sorting_computed_matches_make_insert_signature`, and existing
sorting insertion/cleanup tests. No database schema or identity change is
needed for these argument removals.

## 4. Express UnitMatch choice discovery as a relational query

Evidence: `_pipeline_run.py:1713–1739`, inside
`_unit_match_member_choices`, fetches recording IDs, converts them to a list
of restrictions, fetches sorting IDs, converts those to another restriction
list, and finally fetches curations. Two nested emptiness branches guard this
intermediate representation.

Within the existing member loop, restrict CurationV2 through the relationship
between RecordingSelection and SortingSelection.RecordingSource:

```python
member_sortings = SortingSelection.RecordingSource * (
    RecordingSelection & member_id
)
choices = (CurationV2 & member_sortings).fetch(
    # Keep the existing projected columns and ordering.
)
```

This replaces up to three reads per member with one and naturally returns no
curations when the member has no matching recording or sort. It avoids Python
ID-list materialization and its empty-list cases. The member loop and its
output format can stay as they are; a new query framework is unnecessary.

Preserve the FULL member identity, including team_name, the recording-source
restriction, choice ordering, and empty choices for unsorted members. Do not
silently add latest/committed/valid-for-matching filtering during this refactor.

Verification: database-backed choice discovery and
`test_describe_unit_match_choices_excludes_other_team`, plus an unsorted member
and multiple matching recordings. The query change has not been run against a
database during this audit.

A separate optional improvement follows the same principle:
`_pipeline_geometry.py:98–118` fetches electrodes once per group, whereas the
neighboring `_sort_group_geometry_rows` already fetches all session members
together. Batch the summary's membership read and group the rows locally if
that reporting function is being touched; preserve empty groups and nullable
geometry fields.

## 5. Give committed merged-unit ID assignment one owner

Evidence: the same max(source IDs) + 1 numbering rule, with groups ordered by
minimum contributor ID, is implemented independently in:

- `_curation_transforms.py:519–531` for stored curation rows;
- `review_api.py:617–627` for predicted label-conflict IDs;
- `review_api.py:705–718` for the labels submitted on commit.

The implementations currently agree. Their agreement is important: a reviewer
resolves a conflict for a predicted new unit ID, and the writer must attach
those labels to that same merged unit.

Extract a small pure function that assigns committed IDs to validated merge
groups, and use it at these three sites. Preserve outer group ordering by
minimum contributor. Preserve each caller's contributor ordering: sorting
contributors in the writer would also change the peak-channel tie breaker
when amplitudes are equal. Keep preview leader IDs (`min(group)`) and lazy
spike reconstruction semantics unchanged.

Do not turn this into a merge-engine rewrite or remove external-input
validation and fresh generation checks. The scripted facade rejects an empty
merge request, while label-only browser imports and root curations legitimately
have no merges; a shared helper must not erase that distinction.

Verification: reversed order of two merge groups, noncontiguous unit IDs,
preview/applied parity, labels attached to the predicted IDs, and conflicting
inherited labels. Existing curation-plan, curation-merges, and review tests
cover much of this. Given the scientific identity contract, this item can wait
until merge code next needs a change.

## Audit validation and scope

Forty focused tests passed: preprocessing order, review label import, direct
filtering provenance, outer schema versions, and UnitMatch planning. The row
preparation probe compared the current real writer bodies with the current
audit helpers, intercepting execution before any database operation.

These are baseline checks, not validation of an implemented refactor. No full
database workflow, browser session, or long-recording run was performed.

Keep deferring broad pipeline decomposition, cache lifecycle redesign, and
NWB compatibility removal. The existing distinction between float-tolerant
stored-value comparisons and stricter in-memory comparisons is intentional;
superficial similarity alone is not a reason to merge those functions.
