# Spike Sorting V2 Maintainability and Module-Boundary Review

Date: 2026-06-25

Scope: code organization, table-class responsibility boundaries, source-routing
cohesion, tri-part carrier contracts, pipeline/catalog duplication, private vs
public helper surfaces, test fixture architecture, notebook/docs contributor
ergonomics, and stale status text. This review focuses on future change risk,
not immediate runtime correctness.

Method: local static code/docs/test inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

V2 is not an unstructured rewrite. A lot of the hard work is already factored
into focused helpers: `_selection_plan`, `_selection_identity`,
`_sorting_dispatch`, `_sorting_analyzer`, `_units_nwb`, `_recording_*`,
`_metric_curation_*`, `_pipeline_*`, and the Pydantic parameter modules. The
comments also do a good job naming load-bearing invariants, especially around
tri-part populate, deterministic ids, source parts, and filesystem cleanup.

The maintainability risk is that several extractions stopped just short of a
stable internal architecture. Public DataJoint table classes still carry schema,
validation, compute orchestration, staging cleanup, accessors, plotting
delegates, and audit/query routing in one place. The pipeline facade has split
implementation files, but stage/preset/default-table knowledge is repeated
across run, preflight, reporting, docs, and tests. Tests are broad and useful,
but many integration tests rebuild the same expensive chain by hand, and some
fixture/download behavior depends on source-text scanning rather than explicit
test metadata.

## What Looks Solid

- Complex compute bodies are increasingly served by pure or DB-light helpers:
  `_recording_restriction.py`, `_recording_geometry.py`,
  `_recording_nwb.py`, `_sorting_dispatch.py`, `_sorting_units.py`,
  `_unitmatch_nwb.py`, and `_matcher_graph.py`.
- The `pipeline.py` facade re-exports public orchestration names while keeping
  implementation split across `_pipeline_*` modules.
- `initialize_v2_defaults()` gives users one default-seeding entry point and
  currently includes all eight seeded lookup families.
- Tests already recognize the positional tri-part carrier hazard for several
  tables in `tests/spikesorting/v2/test_sorting_contracts.py`.
- Test comments are unusually good at explaining why guard tests exist, which
  will make cleanup safer.
- Fixture cleanup helpers such as `_ingest_helpers._clean_session_v2()` are
  documented and centralize some otherwise fragile teardown logic.

## Findings

### 1. High: source-routing logic is centralized, but too monolithic

`CurationV2.resolve_restriction()` is the single owner of broad v2 restriction
routing, which is the right architectural instinct. The problem is size and
mixed responsibilities: the method owns allowed-key definitions, artifact
interval-name parsing, UUID normalization, strict-vs-lenient unknown-key
behavior, recording-vs-concat source dispatch, mixed-source rejection, optional
artifact anti-joins, and final `CurationV2` query assembly
(`src/spyglass/spikesorting/v2/curation.py:1416-1668`).

Impact: adding another source family, another source key, or another artifact
restriction rule requires editing one long method with many interacting cases.
The prior concat work shows this is a real risk: broad merge queries can look
correct while silently dropping one source family if the route is incomplete.

Recommended cleanup:

- Extract a small source-routing registry or spec module. It should declare:
  source-family keys, source table joins, mutually exclusive key groups, and
  optional artifact semantics.
- Keep `CurationV2.resolve_restriction()` as the public shim, but have it call
  named parsing and route-building helpers.
- Add table-driven tests for recording source, concat source, no source, mixed
  source rejection, artifact UUID, artifact interval-name mapping,
  `artifact_detection_id=None` anti-join, strict unknown keys, and lenient
  unknown keys.

### 2. High: public table classes still mix too many responsibilities

Several table classes remain very large and multi-purpose:

- `CurationV2` starts at `src/spyglass/spikesorting/v2/curation.py:65` and
  carries curation insertion, merge semantics, NWB staging, public accessors,
  merge-query routing, summaries, and downstream metadata helpers.
- `Sorting` starts at `src/spyglass/spikesorting/v2/sorting.py:1114` and owns
  tri-part sorting, analyzer build/rebuild, unit-part population, staged-file
  cleanup, analyzer-cache deletion, orphan scans, accessors, and brain-region
  summaries.
- `Recording` starts at `src/spyglass/spikesorting/v2/recording.py:995` and
  owns tri-part preprocessing, truncation guards, cache rebuilds, accessors,
  delete previews, and drift-estimate neighbors.
- `AnalyzerCuration` starts at
  `src/spyglass/spikesorting/v2/metric_curation.py:827` and mixes metric
  compute, auto-merge, NWB writing, materialization, DataFrame accessors, and
  plotting delegates.

Impact: future edits have a broad blast radius. A reviewer touching a small
accessor or plotting shim must understand adjacent compute, cleanup, and schema
invariants in the same class. Large classes also make it harder to define which
methods are stable notebook APIs and which are internal service hooks.

Recommended cleanup:

- Move restriction resolution, curation summaries, analyzer-cache maintenance,
  staged-artifact cleanup, and plotting/accessor delegates into focused service
  modules.
- Keep table methods as compatibility shims where notebooks already call them.
- Add smoke tests for the public table-method surface before moving bodies, so
  internal extraction does not break notebook imports.
- Prefer "table owns schema and DataJoint transaction boundary; service owns
  pure or DB-light behavior" as the review rule for new code.

### 3. Medium: tri-part boundaries rely on positional tuple protocols and weak dict shapes

Tri-part populate carriers are `NamedTuple`s that DataJoint unpacks
positionally into `make_compute(key, *fetched)` or `make_insert(key, *computed)`.
Examples include `SortingFetched` and `SortingComputed`
(`src/spyglass/spikesorting/v2/sorting.py:96-169`), `RecordingFetched` and
`RecordingComputed` (`src/spyglass/spikesorting/v2/recording.py:895-947`),
`ArtifactFetched` and `ArtifactComputed`
(`src/spyglass/spikesorting/v2/artifact.py:86-118`),
`ConcatRecordingFetched` and `ConcatRecordingComputed`
(`src/spyglass/spikesorting/v2/session_group.py:531-565`),
`AnalyzerCurationFetched` and `AnalyzerCurationComputed`
(`src/spyglass/spikesorting/v2/metric_curation.py:181-206`), and
`UnitMatchFetched` and `UnitMatchComputed`
(`src/spyglass/spikesorting/v2/unit_matching.py:65-92`).

Some carriers also contain loosely shaped `dict` and `list[dict]` payloads, for
example concat `member_plan` and UnitMatch `member_plan`. The code comments
correctly call this a wire contract, and tests pin some signature alignment in
`tests/spikesorting/v2/test_sorting_contracts.py:50-133`.

Impact: adding or reordering fields can misbind adjacent string/dict slots
without a useful type error unless every signature and alignment test is updated.
The current contract tests cover several important carriers, but not all
tri-part computed tables.

Recommended cleanup:

- Use frozen dataclasses or explicit payload objects internally, with a tiny
  `to_make_args()` adapter only at the DataJoint boundary if positional unpacking
  is unavoidable.
- Define `TypedDict` or dataclasses for repeated plan dictionaries such as
  concat member plans and UnitMatch member plans.
- Extend the signature-alignment tests to every tri-part carrier:
  `ArtifactDetection`, `ConcatenatedRecording`, `UnitMatch`, recompute families,
  and drift estimate.
- Add tests that validate required keys and value types for each `member_plan`
  shape.

### 4. Medium: pipeline orchestration has parallel catalogs of stage and parameter knowledge

The split `_pipeline_*` modules improved readability, but stage/default-table
knowledge still exists in several independent places:

- `run_v2_pipeline()` hardcodes the single-session stage chain and preset fields
  (`src/spyglass/spikesorting/v2/_pipeline_run.py:83-436`).
- `preflight_v2_pipeline()` re-derives prerequisite checks, expected ids,
  parameter rows, analyzer waveform rows, runtime checks, and warnings
  (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:263-636`).
- `describe_parameter_rows()` knows the seeded lookup tables, shipped default
  sources, preset-fold columns, duplicate-fingerprint scope, and row summaries
  (`src/spyglass/spikesorting/v2/_pipeline_reporting.py:42-445`).
- `initialize_v2_defaults()` has its own list of seeded tables
  (`src/spyglass/spikesorting/v2/__init__.py:18-61`).

Impact: adding a stage, seeded table, preset metadata field, or alternate
orchestrator path requires synchronized edits across run, preflight, reporting,
docs, and tests. Drift has already appeared in user docs and notebook comments:
some setup text names only preprocessing/artifact/sorter defaults even though
the initializer now seeds more tables.

Recommended cleanup:

- Introduce a shared descriptor registry for pipeline stages and default lookup
  tables. The registry should feed default initialization, preflight checks,
  parameter-row reporting, and docs snippets where possible.
- Keep stage-specific logic in small functions, but avoid repeating the table
  list and preset-field list by hand.
- Add a test asserting every table seeded by `initialize_v2_defaults()` appears
  in `describe_parameter_rows()` and in docs/examples that describe seeded
  defaults.
- Add a test that `preflight_v2_pipeline()` and `run_v2_pipeline()` derive the
  same deterministic selection ids for every shipped preset.

### 5. Medium: slow integration setup is duplicated across test modules

The same "copy fixture, clean session, initialize defaults, create LabTeam, make
sort groups, populate Recording, populate ArtifactDetection, populate Sorting"
recipe appears in multiple places: the parent fixture
(`tests/spikesorting/v2/conftest.py:215`), the single-session fixture
(`tests/spikesorting/v2/single_session/conftest.py:70`), and bespoke tests such
as `test_preview_merge_warning.py`, `test_merge_id_selectivity.py`, and
`test_artifact_integration.py`.

Impact: row names, cleanup order, fixture names, and default seeding can drift
between integration tests. New contributors have to copy a long recipe to add a
single slow test, which makes small behavior tests look more expensive and more
fragile than they need to be.

Recommended cleanup:

- Extend `_ingest_helpers.configure_v2_run_inputs()` into a small integration
  builder, for example `populate_single_session_chain(...)`.
- Let the builder accept narrow overrides for sorter params, artifact params,
  labels, and zero-unit handling.
- Cover the builder itself with one focused test, then migrate a few modules
  first to prove the API is pleasant before touching all slow tests.

### 6. Medium: fixture-fetch detection is source-text coupled

`tests/spikesorting/v2/conftest.py` decides which test modules need the smoke
fixture by scanning module source for a literal fixture name and by maintaining a
hard-coded shared-fixture allowlist (`tests/spikesorting/v2/conftest.py:77-137`).
Several tests still hand-roll local fixture-path checks and skip messages, for
example `test_preview_merge_warning.py` and `test_ux_smoke.py`.

Impact: a test can use an imported constant, wrapper fixture, or helper that
needs a fixture without containing the magic literal. That can pass locally,
skip in CI, or fail only after collection depending on download state. The rule
is also hard for new contributors to discover.

Recommended cleanup:

- Replace source scanning with explicit metadata such as
  `@pytest.mark.v2_fixture("mearec_polymer_smoke")` or a
  `v2_fixture_path(name)` fixture that calls `ensure_fixture`.
- Add a regression test where a module uses an imported constant or wrapper
  fixture and still triggers fixture fetch.
- Keep the existing source-scan path temporarily as a warning-only transition if
  migrating all tests at once is too noisy.

### 7. Medium-low: private/public boundaries are blurred by compatibility wrappers

Several underscored helpers are private by naming but treated as stable monkeypatch
or compatibility surfaces in tests. Examples include sorting/analyzer helpers
around `src/spyglass/spikesorting/v2/sorting.py:1855` and
`src/spyglass/spikesorting/v2/sorting.py:1985`, plus plotting/accessor delegates
on `AnalyzerCuration` around
`src/spyglass/spikesorting/v2/metric_curation.py:1419`. Tests import many
underscore modules directly; a local scan found hundreds of private-module or
direct-bypass references across `tests/spikesorting/v2`.

Some of that is appropriate for pure service modules, but it is not documented
which underscore modules are deliberately tested internal units and which are
legacy shims that should not gain more call sites.

Impact: contributors cannot easily tell which helpers can be moved, renamed, or
deprecated. Tests may freeze incidental private APIs while public notebook
contracts remain under-described.

Recommended cleanup:

- Add a short "supported internal service modules" note for v2 contributors:
  which underscore modules are unit-test targets, which are compatibility shims,
  and which are private implementation details.
- Prefer new tests against service modules for pure behavior, and public table
  methods only for user-facing contracts.
- Add `__all__` where facade modules intentionally expose public helpers.
- Centralize corrupt-state builders that use `allow_direct_insert=True` so
  bypass intent is explicit and column trivia does not spread through tests.

### 8. Low: parameter lookup insertion patterns remain duplicated

Several parameter lookup tables share the same shape: `insert1()` delegates to
`insert()`, `insert()` normalizes rows through `validate_lookup_rows()`, optional
duplicate-content guards run, and `insert_default()` loads shipped defaults.
Examples include artifact, recording, sorting, motion, analyzer waveform, and
matcher parameter tables (`src/spyglass/spikesorting/v2/artifact.py:157`,
`src/spyglass/spikesorting/v2/recording.py:706`,
`src/spyglass/spikesorting/v2/sorting.py:235`,
`src/spyglass/spikesorting/v2/session_group.py:290`,
`src/spyglass/spikesorting/v2/unit_matching.py:112`).

Impact: new lookup tables can drift in duplicate-content, schema-version,
`job_kwargs`, or default-row behavior. Some drift is already visible in the
serialization review for decomposed metric/rule tables.

Recommended cleanup:

- Add a small validated-lookup helper or mixin with table-specific hooks:
  name fields, schema model, default rows, sorter scoping, and extra validation.
- Keep unusual decomposed tables explicit, but make their deviation from the
  shared contract obvious in code and docs.
- Run the same schema-version, duplicate-content, JSONability, and
  `allow_duplicate_params` tests against every lookup family.

### 9. Low: docs and facade status text lag the shipped surface

`pipeline.py` says metrics, auto-curation, concat sorts, cross-session matching,
and UI hooks "come in later versions" even though many of those surfaces now
exist elsewhere in the package or docs (`src/spyglass/spikesorting/v2/pipeline.py:6-9`).
The feature docs describe UnitMatch as shipped, then later say cross-session
unit matching is not yet available (`docs/src/Features/SpikeSortingV2.md:888`,
`docs/src/Features/SpikeSortingV2.md:1055`). Some planning docs still describe
Phase 4/UnitMatch surfaces as placeholders or planned work.

Impact: this is not a runtime bug, but it slows contributor decisions. People
cannot tell whether to extend a feature, wait for a phase, or update stale
placeholder text.

Recommended cleanup:

- Update `pipeline.py` to say the orchestrator covers the single-session path
  and point to the separate table/API surfaces for metrics, concat, matching,
  and visualization.
- Add a "current shipped surface" banner to the public feature doc.
- Move historical phase docs behind an archive/status preface, or add a short
  "this file is historical" header where appropriate.
- Add a docs smoke check for contradictory status phrases such as "UnitMatch
  not yet available" when `UnitMatchSelection` exists.

### 10. Low: the canonical notebook is becoming both beginner path and advanced reference

The main notebook starts as a single-session walkthrough, then continues through
auto-curation, manual merge, cell typing, visualization, and whole-session
sorting (`notebooks/py_scripts/10_Spike_SortingV2.py:245-539`). The current
notebook smoke tests execute content, but they do not guard a first-hour
ergonomics target or code-cell budget.

Impact: new users and contributors have one long artifact to maintain. Advanced
examples can become stale blockers for the beginner path, and beginner fixes can
accidentally disturb advanced workflow docs.

Recommended cleanup:

- Split the "first sort" notebook from optional curation/visualization and
  whole-session extension notebooks, or mark optional sections clearly.
- Add a notebook smoke assertion for the intended first-hour path: imports,
  defaults, preflight, one sort, summary, and downstream fetch.
- Keep advanced workflow tests, but let them fail against advanced notebooks
  rather than making the beginner walkthrough carry every scenario.
