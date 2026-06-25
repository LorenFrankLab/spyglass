# Spike Sorting V2 Serialization and Payload Compatibility Review

Date: 2026-06-25

Scope: lookup-row payload schemas, `schema_version` behavior, JSON-compatible
parameter serialization, duplicate-content contracts, NWB interchange helpers,
empty-result shapes, and docs/tests that define public payload expectations.
This review intentionally focuses on data interchange and versioned payload
contracts, not recompute correctness, destructive-operation safety, or raw
performance.

Method: local static code/docs/test inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

V2 has a promising serialization foundation: most parameter tables share a
single lookup-row validation helper, parameter identity uses deterministic
canonical JSON, and the NWB helpers are increasingly DB-free and covered by
round-trip tests. The strongest pieces are the canonical fingerprint path,
selection fingerprinting, immutable-update guard, and explicit sample-index
metadata for units.

The main gap is that compatibility is still more convention than contract.
`schema_version` is checked for inner/outer equality, but not for supported
versions. JSON compatibility is enforced mainly as a side effect of fingerprint
generation, so opt-out or decomposed parameter tables can still admit payloads
that are hard to export, compare, or migrate. A few public accessors also return
different shapes for empty or source-dependent results, which makes downstream
code harder to write defensively.

## What Looks Solid

- Shared validation exists for most lookup rows through
  `validate_lookup_rows()` (`src/spyglass/spikesorting/v2/_lookup_validation.py`),
  which gives v2 a single place to tighten payload compatibility rules.
- Parameter fingerprints use canonical JSON with sorted keys and
  `allow_nan=False`, giving byte-stable identity for JSON-native payloads
  (`src/spyglass/spikesorting/v2/_parameter_identity.py:55-75`).
- Duplicate-content checks compare semantic payloads instead of only row names,
  and `ImmutableParamsLookup.update1()` prevents silent mutation of parameter
  blobs after insert (`src/spyglass/spikesorting/v2/_lookup_validation.py`).
- NWB payload code is increasingly isolated into helpers such as
  `_metric_curation_nwb.py`, `_unitmatch_nwb.py`, and `_units_nwb.py`, which
  makes DB-free round-trip testing feasible.
- Unit serialization records sample-index metadata and handles several
  no-data/empty-data cases intentionally, including empty Units tables and
  metric NaN/None coercion.
- Reporting helpers expose explicit DataFrame column constants for several
  user-facing tables, which is the right pattern for stable downstream
  contracts.

## Findings

### 1. High: `schema_version` is a loose tag, not a supported-version contract

`_assert_schema_version_matches()` only verifies that the outer
`params_schema_version` column equals the inner `params["schema_version"]` when
the outer column is present (`src/spyglass/spikesorting/v2/_lookup_validation.py:52-93`).
It does not reject stale or future versions. The Pydantic models also declare
`schema_version: int = CURRENT_SCHEMA_VERSION` rather than a literal supported
version, for example in preprocessing, sorter, motion, analyzer waveform,
metric curation, and matcher parameter models
(`src/spyglass/spikesorting/v2/_params/preprocessing.py:138-140`,
`src/spyglass/spikesorting/v2/_params/sorter.py:59-60`,
`src/spyglass/spikesorting/v2/_params/motion_correction.py:103-106`,
`src/spyglass/spikesorting/v2/_params/analyzer_waveform.py:41-43`,
`src/spyglass/spikesorting/v2/_params/metric_curation.py:213-215`,
`src/spyglass/spikesorting/v2/_params/matcher.py:38-41`).

There is a second edge case: when `params_schema_version` is omitted,
`_assert_schema_version_matches()` returns without checking the inner value.
Several tables rely on a DataJoint column default to tag the row as current
version, so an omitted outer value plus a stale or future inner
`schema_version` can be stored as a current-version row. `SorterParameters` has
a table-specific sentinel/backfill path, but the pattern is not enforced for all
lookup tables (`src/spyglass/spikesorting/v2/sorting.py:203-360`,
`src/spyglass/spikesorting/v2/recording.py:683-735`,
`src/spyglass/spikesorting/v2/artifact.py:123-187`,
`src/spyglass/spikesorting/v2/session_group.py:251-321`,
`src/spyglass/spikesorting/v2/unit_matching.py:121-166`).

Impact: future clients, old scripts, or hand-written inserts can create rows
that appear compatible with the current code but were never interpreted by a
version-aware migration path. That makes recompute, export, and duplicate
detection less trustworthy because the version tag stops meaning "validated as
this version."

Recommended fix:

- Make each current schema model reject unsupported versions, either with
  `Literal[CURRENT_SCHEMA_VERSION]` or a model validator that explicitly allows
  only known versions.
- If older versions are intentionally supported, add a version-dispatch and
  migration step before normalizing rows into current models.
- Move the sorter-style outer-version backfill/reject behavior into shared
  validation so every lookup table handles omitted outer versions consistently.
- Add tests for future, stale, and missing-outer-version rows across every
  parameter table, including decomposed tables such as metric curation rules.

### 2. Medium-high: JSON portability is a fingerprint side effect, not a first-class insert invariant

`_validate_params()` runs Pydantic validation and returns `model_dump()` output,
but it does not independently enforce JSON-native, finite payload values
(`src/spyglass/spikesorting/v2/_lookup_validation.py:29-49`). `_jsonable_blob()`
and `parameter_fingerprint()` do enforce a canonical JSON round trip, but they
are reached mainly through duplicate-content checks
(`src/spyglass/spikesorting/v2/_lookup_validation.py:142-162`,
`src/spyglass/spikesorting/v2/_parameter_identity.py:55-75`).

That creates holes:

- `reject_duplicate_parameter_content()` returns immediately when
  `allow_duplicate_params=True`, so rows inserted through that escape hatch also
  skip the JSON/fingerprint path
  (`src/spyglass/spikesorting/v2/_lookup_validation.py:165-270`).
- `QualityMetricParameters.insert()` validates reconstructed payloads but does
  not call the duplicate/fingerprint guard
  (`src/spyglass/spikesorting/v2/metric_curation.py:285`).
- `AutoCurationRules.insert_rules()` uses `_jsonable_blob()` only for same-name
  idempotency comparison, not as a universal payload invariant
  (`src/spyglass/spikesorting/v2/metric_curation.py:423`).
- `_resolved_job_kwargs()` blindly merges overrides with `dict.update()` and
  does not validate mapping type, key type, JSON compatibility, or finite
  values (`src/spyglass/spikesorting/v2/utils.py:607-632`).

Impact: DataJoint blobs can accept values that are convenient in Python but not
portable as versioned payloads, such as numpy arrays, numpy scalars, sets,
non-string keys, NaN/Inf, or custom objects. These may later fail during export,
fingerprint comparison, documentation/report generation, or migration, and the
failure may occur far away from the original insert.

Recommended fix:

- Add a shared `assert_jsonable_finite_blob()` step after Pydantic validation
  and before insert, independent of duplicate detection.
- Run it for `params`, `job_kwargs`, `execution_params`, metric kwargs, auto
  merge kwargs, preset kwargs, and rule blobs.
- Keep `allow_duplicate_params=True` scoped only to the duplicate-content
  policy; it should not bypass serialization safety.
- Validate job-kwargs overrides as mappings with string keys and JSON-compatible
  finite values before merging.
- Add negative tests for NaN/Inf, numpy arrays, sets, custom objects, non-string
  keys, and duplicate-allowed inserts across every parameter family.

### 3. Medium: Duplicate-content behavior differs from the documented contract

The public docs describe duplicate-content rejection broadly: parameter rows
should be immutable, fingerprinted, and protected from duplicate semantic
payloads (`docs/src/Features/SpikeSortingV2.md:371-395`). That matches several
core lookup tables, but not all v2 parameter-like tables.

`QualityMetricParameters.insert()` reconstructs and validates the Pydantic
model, then inserts decomposed metric/default/override rows without the shared
duplicate-content guard (`src/spyglass/spikesorting/v2/metric_curation.py:209-318`).
`AutoCurationRules.insert_rules()` prevents same-name drift, but it does not
reject identical rule content under different names; tests intentionally insert
that scenario (`src/spyglass/spikesorting/v2/metric_curation.py:365-510`).

Impact: users reading the docs will assume name uniqueness and content identity
mean the same thing across all parameter tables. In practice, some tables allow
semantic duplicates by design or by omission. That makes `describe` output,
triage reports, and recompute plans harder to interpret consistently.

Recommended fix:

- Decide whether metric/rule tables should follow the same duplicate-content
  policy as the core lookup tables.
- If yes, add content fingerprints or shared duplicate guards to
  `QualityMetricParameters` and `AutoCurationRules`, with an explicit
  `allow_duplicate_params` escape hatch if needed.
- If no, narrow the docs to say exactly which tables enforce duplicate-content
  rejection, and teach `describe_parameter_rows()` to surface duplicates in
  metric/rule tables as informational warnings.
- Add tests for same-content/different-name metric parameters and auto-curation
  rules so the chosen policy is locked down.

### 4. Medium: Empty UnitMatch pairs lose the DataFrame schema

The UnitMatch NWB writer knows the pair table schema through explicit string,
integer, and float column lists (`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:29-38`).
`build_pairs_table([])` writes an empty table with concrete columns, but
`read_pairs()` returns `[]` for an empty result
(`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:110`). `UnitMatch.get_pairs()`
then wraps that list as `pd.DataFrame(read_pairs(...))`, producing a DataFrame
with no columns (`src/spyglass/spikesorting/v2/unit_matching.py:821-829`).

Tests cover the zero-pair length but not the empty DataFrame schema
(`tests/spikesorting/v2/test_unitmatch.py:410-413`,
`tests/spikesorting/v2/test_unitmatch.py:1066-1083`).

Impact: downstream code that selects expected pair columns works for non-empty
matches and fails for valid empty matches. Empty outputs are common in matching
pipelines, so this is a user-visible compatibility edge.

Recommended fix:

- Define a public/internal `PAIR_COLUMNS` constant and use
  `pd.DataFrame(rows, columns=PAIR_COLUMNS)` in `UnitMatch.get_pairs()`.
- Add tests that empty and non-empty `get_pairs()` results expose the same
  columns in the same order, with predictable dtypes where feasible.
- Consider making `read_pairs()` return a structured empty representation rather
  than a bare list if external callers rely on that helper directly.

### 5. Medium-low: NWB curation-label payloads have multiple shapes and the docs do not name them

`_units_nwb.py` documents that sort-time unit rows carry placeholder
`curation_label` values shaped like `["uncurated"]`
(`src/spyglass/spikesorting/v2/_units_nwb.py:533-539`). The implementation
intentionally writes a scalar `"uncurated"` at sort time
(`src/spyglass/spikesorting/v2/_units_nwb.py:610-655`). Later curation writes
can use ragged label arrays, while no-label cases need separate handling because
HDMF does not always round-trip empty ragged columns cleanly.

Impact: internal readers may already handle these cases, but external NWB
consumers and docs readers have to infer the possible shapes. That increases the
chance that an apparently harmless export/import or notebook path breaks on a
newly sorted, uncured, or empty-label dataset.

Recommended fix:

- Update `_units_nwb.py` docstrings and public docs to describe the exact label
  shapes currently emitted for sorted, curated, and no-label rows.
- Add a small reader helper that normalizes scalar/string/ragged/no-label cases
  into one Python representation for downstream callers.
- Add tests for pre-curation scalar labels, post-curation multi-label rows, and
  no-label/empty-label rows through the public read path.

### 6. Medium-low: artifact interval return shape depends on source path by default

`ArtifactDetection.get_artifact_removed_intervals()` can return a numpy array
for single-recording artifacts or a dictionary keyed by NWB file for shared
artifact groups, unless callers pass `as_dict=True`
(`src/spyglass/spikesorting/v2/artifact.py:1053-1092`,
`src/spyglass/spikesorting/v2/_artifact_intervals.py:597-630`).

Impact: callers that work on a single source can break when a selection is moved
to a shared artifact group, even though the conceptual payload is still
"artifact-removed intervals." This also complicates serialization docs because
the stable interchange shape is opt-in rather than default.

Recommended fix:

- Prefer a stable dictionary shape for new public APIs, or add an explicitly
  named helper such as `get_artifact_removed_intervals_by_nwb()`.
- If backward compatibility requires the current default, document it next to
  the method and add tests that assert both single-recording and shared-group
  shapes.
- Use the stable shape in examples and downstream v2 code.

### 7. Low: public docs understate the exported payload schema and overstate setup coverage

The main v2 docs say several payloads are DataFrames or are exported "the same"
way as v1, but they do not describe the stable columns, empty-result behavior,
or restore/import limitations for v2-specific NWB payloads
(`docs/src/Features/SpikeSortingV2.md:512-514`,
`docs/src/Features/SpikeSortingV2.md:935-937`,
`docs/src/Features/SpikeSortingV2.md:1015-1040`). Setup docs also list
`SharedArtifactGroup` with Pydantic-style parameter tables and omit seeded
parameter families such as analyzer waveform, motion correction, and matcher
parameters (`docs/src/Features/SpikeSortingV2.md:54-58`,
`docs/src/Features/SpikeSortingV2.md:174-176`,
`notebooks/py_scripts/10_Spike_SortingV2.py:100-102`).

Impact: this is less likely to corrupt data, but it makes compatibility
expectations ambiguous for users writing notebooks, exports, or migration tools.

Recommended fix:

- Add a compact payload-schema appendix for exported v2 NWB content:
  recording, sorting units, curation labels, quality metrics, UnitMatch pairs,
  and empty-result shapes.
- Make the setup docs list all seeded/default parameter tables accurately, and
  separate true parameter blobs from grouping/selection tables.
- Document whether export currently supports full restore/import for each v2
  payload or only file capture.
