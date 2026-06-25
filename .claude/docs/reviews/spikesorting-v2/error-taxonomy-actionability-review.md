# Spike Sorting V2 Error Taxonomy and Actionability Review

Date: 2026-06-25

Scope: exception taxonomy, preflight/actionable failures, public workflow error
surfaces, direct table workflows, malformed input behavior, artifact load/read
failures, batch result reporting, and test coverage for user-facing failure
modes. This is a different lens from import boundaries, ownership, destructive
operations, dependency compatibility, and docs link integrity.

Method: local static code/test/docs inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

Spike Sorting V2 has a much stronger error foundation than the older pipeline:
there is a v2 exception module, preflight collects many actionable setup
failures before populate, stage failures are wrapped with partial run context,
and duplicate/source-part invariants usually raise clear domain errors instead
of raw DataJoint messages.

The remaining gaps cluster around public paths that bypass the orchestrated
`run_v2_pipeline(..., preflight=True)` happy path: chronic `SessionGroup`
creation, direct table-by-table sorting, artifact readback helpers, UnitMatch
compute, malformed public input dictionaries, and the explicit
`preflight=False` bypass. These paths can still leak `KeyError`, `IndexError`,
DataJoint `fetch1` / FK errors, Pydantic `ValidationError`, SpikeInterface
runtime errors, or low-level file/NWB errors without the v2 key, stage, path,
member index, or repair hint.

## What Looks Solid

- `src/spyglass/spikesorting/v2/exceptions.py` defines many domain-specific
  exceptions with docstrings that name the failed invariant and intended user
  action.
- `PipelineStageError` preserves the failed stage, partial run summary,
  original exception type, and chained traceback
  (`src/spyglass/spikesorting/v2/exceptions.py:167-198`,
  `src/spyglass/spikesorting/v2/_pipeline_run.py:63-79`).
- Preflight checks many common setup problems before writes: session, `Raw`,
  selected interval, team, sort group, sort-group electrodes, parameter rows,
  samplerate, sorter runtime/backend, and analyzer parameters
  (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:340-400` and nearby).
- Selection/source-part tables consistently guard bypassed or duplicate rows
  with actionable errors, including `DuplicateSelectionError` and
  `SchemaBypassError`.
- Cleanup catches are usually cleanup-and-reraise, so they avoid masking the
  original failure.

## Findings

### 1. High: preflight misses the required `"raw data valid times"` interval

`preflight_v2_pipeline()` verifies that the selected sort interval exists
(`src/spyglass/spikesorting/v2/_pipeline_preflight.py:376-386`) and now verifies
that the common `Raw` row exists (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:369-375`).
However, `Recording.make_fetch()` also unconditionally fetches the fixed
`IntervalList` row named `"raw data valid times"`
(`src/spyglass/spikesorting/v2/recording.py:1109-1115`).

If that fixed interval is missing, preflight can pass and populate later fails
with a lower-level DataJoint fetch error. That contradicts the preflight
contract for a common partial-ingestion prerequisite.

Recommended fix:

- Add a `raw_valid_times_exists` preflight check with a message that names
  `"raw data valid times"` and points to re-running common ingestion.
- Add a defensive `Recording.make_fetch()` guard so direct `Recording.populate`
  also fails with a typed/actionable error.
- Add a regression next to `test_preflight_missing_raw`
  (`tests/spikesorting/v2/test_preflight.py:246-265`) that simulates a missing
  fixed interval and asserts the preflight report names it.

### 2. High: `SessionGroup.create_group()` can leak `KeyError` and raw FK errors

The chronic/concat workflow documents passing member dictionaries into
`SessionGroup.create_group()`, but the method directly indexes
`member["nwb_file_name"]` while deriving dates
(`src/spyglass/spikesorting/v2/session_group.py:148-159`). It then inserts all
members without prevalidating that each member has required fields or that its
`SortGroupV2`, `IntervalList`, and `LabTeam` FK rows exist
(`src/spyglass/spikesorting/v2/session_group.py:207-215`).

The existing atomicity test deliberately exercises a failed member insert
(`tests/spikesorting/v2/test_session_group_concat.py:246-255`), but the user
still sees a low-level failure rather than a member-indexed explanation.

Impact: a typo in a documented workflow can produce `KeyError` or DataJoint
integrity errors without saying which member is malformed or which upstream row
to create.

Recommended fix:

- Introduce a typed `SessionGroupInputError` or `SessionGroupMemberError`.
- Validate all member dictionaries before the transaction: required fields,
  no extra date field, FK existence, duplicate logical members, and date policy.
- Report member indexes and missing upstream rows in one actionable message.
- Replace broad-error assertions with typed-error assertions and keep the
  no-partial-insert atomicity check.

### 3. Medium: supported direct workflows lack the pipeline's stage wrapper

The orchestrated pipeline wraps any stage failure in `PipelineStageError`
(`src/spyglass/spikesorting/v2/_pipeline_run.py:63-79`). Direct table workflows
remain supported in docs and tests, but several of those direct paths surface
raw dependency/runtime errors:

- `run_si_sorter()` calls `sis.run_sorter(...)` directly and only uses `finally`
  for global-job-kwarg restoration
  (`src/spyglass/spikesorting/v2/_sorting_dispatch.py:573-590`).
- Public readback helpers such as `Recording.get_recording()` and
  `Sorting.get_sorting()` can expose DataJoint, filesystem, SpikeInterface, or
  NWB/HDF5 exceptions without the relevant v2 key or artifact path
  (`src/spyglass/spikesorting/v2/recording.py:1520-1535`,
  `src/spyglass/spikesorting/v2/sorting.py:1817-1830`).
- `CurationV2.get_curated_sorting()` reads Units NWB data through helper
  functions that can fail without curation/file context
  (`src/spyglass/spikesorting/v2/curation.py:1354-1364`).
- UnitMatch member extraction and backend matching are not wrapped by
  `run_v2_pipeline`, so backend/library failures can lack `unitmatch_id`,
  matcher name, and member index.

Recommended fix:

- Add typed wrappers for direct execution/readback failures, for example
  `SorterExecutionError`, `RecordingArtifactLoadError`, `SortingArtifactLoadError`,
  `UnitsNWBReadError`, and `UnitMatchStageError`.
- Preserve the original exception via chaining and include table key, stage,
  analysis file/object id, sorter/backend, or member index as appropriate.
- Add monkeypatch tests for failed `sis.run_sorter`, `AnalysisNwbfile.get_abs_path`,
  `se.read_nwb_recording`, Units NWB readers, and UnitMatch backend calls.

### 4. Medium: malformed public input dictionaries still leak raw Python errors

Several public entry points validate semantic errors well, but still assume the
outer shape of caller-provided dictionaries:

- `CurationV2.insert_curation()` immediately reads `sorting_key["sorting_id"]`
  (`src/spyglass/spikesorting/v2/curation.py:311`), so a malformed or joined row
  can raise bare `KeyError` before the method reaches its clearer sorting-row
  validation.
- `UnitMatchSelection.insert_selection()` builds `choices_by_member` by calling
  `curation_choices.items()` and indexing `choice["sorting_id"]` /
  `choice["curation_id"]`
  (`src/spyglass/spikesorting/v2/unit_matching.py:291-294`), so a list, scalar,
  or missing field yields `AttributeError` / `KeyError`.
- Lookup-table parameter validation delegates directly to Pydantic
  (`src/spyglass/spikesorting/v2/_lookup_validation.py:29-49`,
  `src/spyglass/spikesorting/v2/_lookup_validation.py:130-138`), which is rich
  but not wrapped with table name, row name/index, schema version, or the
  relevant insert helper.

Recommended fix:

- Add outer-shape validation before semantic validation for public dictionary
  APIs. Error messages should name the method, required fields, and an example
  key shape.
- Wrap Pydantic `ValidationError` in a v2 parameter validation error that
  includes table name, row identity, schema version, and the original validation
  summary as the chained cause.
- Add tests for missing `sorting_id`, malformed `curation_choices`, and one bad
  row in a bulk parameter insert.

### 5. Medium: v2 has many named exceptions but no common public base

The exception module says callers should be able to catch specific v2 failure
modes rather than parsing bare `ValueError` / `RuntimeError`
(`src/spyglass/spikesorting/v2/exceptions.py:1-6`). The classes themselves
inherit directly from built-in exceptions
(`src/spyglass/spikesorting/v2/exceptions.py:12-167`), and several public
workflow errors still intentionally use bare `ValueError`.

Impact: batch scripts cannot reliably say "catch all expected Spike Sorting V2
operator errors but let unexpected bugs crash" without enumerating every class
and still missing bare public-input errors.

Recommended fix:

- Add a common `SpikeSortingV2Error` base.
- For backward compatibility, consider `SpikeSortingV2ValueError(ValueError,
  SpikeSortingV2Error)` and `SpikeSortingV2RuntimeError(RuntimeError,
  SpikeSortingV2Error)` compatibility bases, then migrate existing named
  exceptions under them.
- Add tests that representative public workflow failures remain compatible with
  `ValueError` / `RuntimeError` where callers may already depend on that, while
  also being catchable as `SpikeSortingV2Error`.

### 6. Medium-low: `preflight=False` intentionally creates a second error world

`test_run_pipeline_preflight_bypass` asserts that a missing team with
`preflight=False` raises a raw DataJoint `IntegrityError`, specifically not
`PreflightError` (`tests/spikesorting/v2/test_preflight.py:703-721`). That is a
real bypass contract, not an accidental bug.

The tradeoff is that a public orchestrator flag turns common setup mistakes back
into low-level DataJoint errors. Users may set `preflight=False` for speed or
retry behavior and lose the otherwise strong actionability guarantees.

Recommended fix:

- Keep full preflight optional, but add cheap insert-boundary guards for common
  FK mistakes: missing `LabTeam`, `Raw`, selected interval, sort group, and
  parameter rows.
- Alternatively rename/document the flag as an expert/debug bypass and make the
  consequence explicit in user docs.

### 7. Medium-low: batch failure results drop structured stage/cause fields

`run_v2_pipeline_session(..., continue_on_error=True)` stores
`error_type`, `error`, `partial_run_summary`, and warnings for failures
(`src/spyglass/spikesorting/v2/_pipeline_run.py:624-640`). For a
`PipelineStageError`, the object also has structured `stage` and
`original_type` attributes (`src/spyglass/spikesorting/v2/exceptions.py:180-198`),
but the batch result does not expose them except through string parsing.

Recommended fix:

- Add optional `stage` and `original_error_type` fields to failed batch result
  rows when the exception carries those attributes.
- Add a continue-on-error regression that verifies a wrapped stage failure is
  machine-readable without parsing `error`.

## Suggested Repair Order

1. Add the missing `"raw data valid times"` preflight/direct guard.
2. Harden `SessionGroup.create_group()` input and FK validation before the
   transaction.
3. Add outer-shape validation for `CurationV2.insert_curation()`,
   `UnitMatchSelection.insert_selection()`, and lookup parameter rows.
4. Add direct-workflow wrappers for sorter execution, artifact readback, Units
   NWB readback, and UnitMatch backend failures.
5. Add a common v2 exception base and structured batch failure fields.

