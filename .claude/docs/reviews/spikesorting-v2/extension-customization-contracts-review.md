# Spike Sorting V2 Extension and Customization Contracts Review

Date: 2026-06-25

Scope: user/lab extension points in spikesorting v2: custom sorter rows and
execution backends, pipeline presets and recipe catalogs, analyzer waveform
recipes, quality metric and auto-curation rules, matcher backends, curation
labels, visualization helpers, fixture/contributor extension contracts, docs,
and tests. This review focuses on whether advertised customization surfaces are
real, stable, teachable contracts.

Method: local static source/docs/test inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

V2 has several strong customization primitives. Parameter rows are validated at
insert time, duplicate-content guards protect provenance, sorter execution
provenance is explicit, auto-curation rule sets are inserted atomically, and
`CurationV2` deliberately allows custom labels through an opt-in guard. Those
are good foundations for lab-specific workflows.

The main issue is that some surfaces are advertised as extensible before the
runtime contract is actually open. `MatcherProtocol` is clean and pure, but
`UnitMatch` table execution still always builds a UnitMatch-specific bundle
before dispatching to a matcher. Pipeline presets are a static in-code catalog,
not a user-registerable API. Custom waveform rows are accepted by the lookup
table, but sort-time display analyzer selection is still resolved through a
private preprocessing-to-waveform map. Several smaller gaps are documentation
or provenance issues: custom rule labels fail late unless users know to pass
`allow_custom_labels=True`, visualization discovery is a closed catalog, and old
fixture-plan text conflicts with the current sidecar-ground-truth contract.

## What Looks Solid

- `MatcherProtocol` itself is narrow, pure Python, and testable without
  DataJoint (`src/spyglass/spikesorting/v2/matcher_protocol.py:1-19`).
- `register_matcher()` has a default collision guard and explains why replacing
  a name is dangerous when table rows store only the matcher name
  (`src/spyglass/spikesorting/v2/matcher_protocol.py:106-147`).
- `SorterParameters` separates scientific params from execution provenance via
  validated `execution_params`, and reserves container/runtime keys so they
  cannot be smuggled through `params` or `job_kwargs`
  (`src/spyglass/spikesorting/v2/sorting.py:203-233`,
  `src/spyglass/spikesorting/v2/_params/sorter.py:507-570`).
- `QualityMetricParameters` validates metric names against installed
  SpikeInterface, distinguishes metric names from output columns, and checks PCA
  metric consistency (`src/spyglass/spikesorting/v2/_params/metric_curation.py:195-310`).
- `AutoCurationRules.insert_rules()` validates the master row and ordered rule
  rows together, and blocks direct master/part insertion
  (`src/spyglass/spikesorting/v2/metric_curation.py:364-481`).
- `CurationV2.UnitLabel` uses `varchar` plus Python-side validation so labs can
  add custom labels with `allow_custom_labels=True` without a schema migration,
  while still catching typoed canonical labels by default
  (`src/spyglass/spikesorting/v2/curation.py:108-152`).
- The visualization facade is careful about display-vs-metric analyzer routing
  and missing-extension errors (`src/spyglass/spikesorting/v2/visualization.py:1-44`,
  `src/spyglass/spikesorting/v2/_visualization.py:43-83`).

## Findings

### 1. High: matcher plugins are not backend-agnostic in the table execution path

`MatcherProtocol` says a matcher consumes wrapper-prepared bundles and returns
`MatchPair` objects (`src/spyglass/spikesorting/v2/matcher_protocol.py:31-97`).
That interface is good, but the only table execution path bakes in UnitMatch
bundle extraction before dispatching to the registered matcher:
`UnitMatch._extract_and_match()` imports `extract_unitmatch_bundle`
(`src/spyglass/spikesorting/v2/unit_matching.py:761-763`) and always calls it
for each session (`src/spyglass/spikesorting/v2/unit_matching.py:794-799`).
`extract_unitmatch_bundle()` then calls `_require_unitmatch()`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:149`), so a custom matcher
that does not need UnitMatchPy still inherits a UnitMatchPy dependency in normal
`UnitMatch.populate()` execution.

The integration test proves this pressure point: the fixture matcher can run
without UnitMatchPy only by monkeypatching the private
`_unitmatch_backend.extract_unitmatch_bundle` helper
(`tests/spikesorting/v2/test_unitmatch.py:1215-1222`).

Impact: labs cannot add a materially different matcher backend through the
public protocol alone. They must accept the UnitMatch directory layout, install
UnitMatchPy, or patch/edit private code.

Fix direction:

- Make bundle preparation part of the registered backend contract, for example
  `prepare_session(...) -> SessionMatcherInput`, a `BundleBuilder`, or a richer
  protocol that owns both extraction and matching.
- Keep the current UnitMatch bundle builder as the built-in implementation.
- Add an end-to-end custom matcher test that runs through
  `MatcherParameters` / `UnitMatch.populate()` without UnitMatchPy and without
  monkeypatching `_unitmatch_backend`.

### 2. High: custom pipeline presets are not a public extension surface

Users can insert custom parameter rows, but the one-shot pipeline API accepts
only the static `_PIPELINE_PRESETS` dictionary. `_PIPELINE_PRESETS` is built once
from `_recipe_catalog.pipeline_preset_specs()`
(`src/spyglass/spikesorting/v2/_pipeline_presets.py:66-73`), and
`run_v2_pipeline()` rejects any name absent from that dictionary
(`src/spyglass/spikesorting/v2/_pipeline_run.py:256-263`). Preflight uses the
same private static catalog (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:307-327`).

The docs blur this boundary. The preset section says a Docker row and other
rates are "a user-insertable `SorterParameters` row away"
(`docs/src/Features/SpikeSortingV2.md:309-316`), and the stage-by-stage section
is titled "custom pipeline preset" (`docs/src/Features/SpikeSortingV2.md:566-569`).
But stage-by-stage driving is not a preset registration mechanism; it bypasses
`run_v2_pipeline()`.

Impact: a lab can create valid preprocessing/artifact/sorter rows but cannot
make a named, discoverable, preflightable, one-shot pipeline bundle without
editing `_recipe_catalog.py` or monkeypatching `_PIPELINE_PRESETS`.

Fix direction:

- Either add a public `register_pipeline_preset(name, spec, replace=False)` API
  with validation, collision rules, and provenance guidance, or promote presets
  to a Lookup seeded by shipped defaults.
- If static presets are intentional for now, rename the docs section to
  "stage-by-stage custom rows" and explicitly say `run_v2_pipeline()` accepts
  only shipped presets today.
- Test that a registered custom preset appears in `list_pipeline_presets()`,
  `describe_pipeline_presets()`, `preflight_v2_pipeline()`, and
  `run_v2_pipeline()` if registration is implemented.

### 3. Medium-high: custom analyzer waveform rows are validated, but sort-time selection is not first-class

`AnalyzerWaveformParameters` supports custom validated waveform recipes
(`src/spyglass/spikesorting/v2/sorting.py:582-648`). But the sort-time display
recipe is explicitly not a free per-sort knob: it is resolved from the source
preprocessing recipe and is not part of `sorting_id`
(`src/spyglass/spikesorting/v2/sorting.py:1134-1146`). `Sorting.make_fetch()`
always calls `waveform_params_for_preprocessing(preprocessing_params_name)`
(`src/spyglass/spikesorting/v2/sorting.py:1274-1287`).

That resolver is a private static map. Unknown, custom, multi-region, `default`,
`default_neuropixels`, and `no_filter` preprocessing recipes all fall back to
the wider cortex pair (`src/spyglass/spikesorting/v2/_recipe_catalog.py:491-517`),
and tests assert that a `"some_custom_recipe"` gets the cortex fallback
(`tests/spikesorting/v2/test_analyzer_waveform_params.py:146-155`).

Impact: a lab can insert a custom waveform row and can use metric-waveform
overrides in some analyzer-curation paths, but normal sorting cannot select a
custom display analyzer recipe through `SortingSelection`, pipeline presets, or
a documented resolver API. Custom preprocessing rows silently inherit cortex
windows unless core code changes.

Fix direction:

- Document the current contract explicitly: display recipe selection is a
  shipped resolver only; metric recipe override is the supported custom route.
- Add a custom metric-waveform-row docs example and test.
- If custom display windows are intended, add a public resolver/Lookup mapping
  preprocessing recipes to display/metric waveform rows, or add optional
  waveform recipe fields to selection/preset APIs.

### 4. Medium: matcher registry replacement/provenance can re-route existing rows

`MatcherParameters` stores `matcher`, `params`, and schema/job fields; it does
not store the backend module/class/version that `matcher` resolved to
(`src/spyglass/spikesorting/v2/unit_matching.py:99-110`). The registry guard
acknowledges the risk: replacing a registered name can make existing rows
dispatch to different code (`src/spyglass/spikesorting/v2/matcher_protocol.py:118-124`).

Two details make this more fragile than the docstring implies:
`register_matcher(..., replace=True)` is public and tests it as a supported
override (`tests/spikesorting/v2/test_matcher_protocol.py:71-83`), but every
lookup calls `register_default_matchers()` (`src/spyglass/spikesorting/v2/matcher_protocol.py:167-195`),
and the built-in UnitMatch registration always passes `replace=True`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:394-409`). A lab override
under the built-in `"unitmatch"` name can therefore be overwritten by the next
default-registration path.

Impact: persisted matcher rows are reproducible only if the process registry is
recreated identically. A name collision, import-order difference, or deliberate
`replace=True` can route an old `MatcherParameters` row to different backend
code without any row-level provenance change.

Fix direction:

- Register defaults only when absent, or replace only when the existing backend
  is the known built-in object/version.
- Store optional backend provenance on `MatcherParameters` or on `UnitMatch`
  results: module, class, package version, and perhaps a backend-declared
  `backend_version`.
- Add tests that explicit replacement behavior is either preserved across
  `_registered_matchers()` / `get_matcher()` or explicitly disallowed for
  built-in names.

### 5. Medium: custom sorter support is a generic SpikeInterface escape hatch, not a full sorter plugin API

`SorterParameters` documents a useful escape hatch: users can insert rows for
any installed SpikeInterface sorter, with `GenericSorterParamsSchema` as the
fallback (`src/spyglass/spikesorting/v2/sorting.py:203-213`). Insert-time
validation checks names against `spikeinterface.sorters.available_sorters()`,
the curated schema set, or `_NON_SI_SORTERS`
(`src/spyglass/spikesorting/v2/sorting.py:277-301`).

The non-SI and special-policy routes are not plugin surfaces. The only non-SI
sorter is hard-coded as `clusterless_thresholder`
(`src/spyglass/spikesorting/v2/sorting.py:369-373`), internal-whitening and
container-only sorter policies live in hard-coded sets
(`src/spyglass/spikesorting/v2/_params/sorter.py:440-469`,
`src/spyglass/spikesorting/v2/_sorting_dispatch.py:31-45`), and custom runtime
behavior requires editing `_sorting_dispatch.py`.

Impact: the current custom sorter contract is good for "try a known
SpikeInterface sorter with arbitrary params." It is not a supported API for a
lab-specific non-SI sorter or a sorter requiring custom pre/post hooks. Users
can overestimate what "custom sorter row" means.

Fix direction:

- Document the supported boundary: arbitrary SpikeInterface sorter rows are
  accepted; non-SI sorters and custom runtime hooks require source changes.
- If non-SI sorter plugins are desired, add a sorter adapter registry with
  params schema, execution policy, preprocessing hooks, and result conversion.
- Add tests for the intended generic-SI custom path and for rejection of unknown
  non-SI sorter names.

### 6. Medium: custom auto-curation labels can pass rule insertion but fail when materialized

`AutoCurationRules.Rule.label` is a free string validated only for length
(`src/spyglass/spikesorting/v2/_params/metric_curation.py:311-330`), so a rule
set can be inserted with a lab-specific label. Later,
`AnalyzerCuration.materialize_curation()` defaults `allow_custom_labels=False`
(`src/spyglass/spikesorting/v2/metric_curation.py:1383-1417`) and delegates to
`CurationV2.insert_curation()`, whose label validator rejects labels outside the
canonical set unless `allow_custom_labels=True`
(`src/spyglass/spikesorting/v2/curation.py:496-532`).

The notebook mentions custom labels generally
(`notebooks/py_scripts/10_Spike_SortingV2.py:247-252`), and the docs mention
`AutoCurationRules.insert_rules(...)` (`docs/src/Features/SpikeSortingV2.md:522-530`),
but neither connects custom rule labels to the materialization flag.

Impact: a custom rule set can compute metrics and proposed labels successfully,
then fail only when the user tries to commit the proposals into `CurationV2`.
That is a late failure for a configuration issue.

Fix direction:

- Document canonical-label custom rule examples and a custom-label example using
  `AnalyzerCuration().materialize_curation(sel, allow_custom_labels=True)`.
- Consider validating rule labels against `CurationLabel` by default at
  `insert_rules()` time, with an explicit `allow_custom_labels=True` override
  matching `CurationV2`.
- Add tests for custom rule-label materialization failure and success.

### 7. Low-medium: visualization discovery is static, not extensible

`available_visualizations()` is backed by a private static tuple
(`src/spyglass/spikesorting/v2/_visualization.py:86-242`) and the facade exports
a fixed `__all__` (`src/spyglass/spikesorting/v2/visualization.py:56-73`).
Labs can call existing helpers with custom kwargs or write external plotting
code, but there is no way to register a lab-specific helper into the discovery
catalog.

Impact: the facade is useful as a closed catalog. It is not an extension point,
even though "available visualizations" can read like a discoverable registry.

Fix direction:

- Either document the facade as closed and source-controlled, or add
  `register_visualization(...)` with key-type validation, collision rules, and
  tests that custom helpers appear in `available_visualizations()`.

### 8. Low-medium: old fixture-generation plan conflicts with current test fixture contract

The Phase 0 scaffolding plan says MEArec ground-truth units should be written to
`nwbfile.units` and imported via `ImportedSpikeSorting`
(`.claude/docs/plans/spikesorting-v2/phase-0-scaffolding.md:199-202`). Current
tests assert the opposite contract: ground truth lives in
`ProcessingModule("ground_truth")["units"]`, `nwbfile.units` must be empty, and
`ImportedSpikeSorting` is intentionally not invoked
(`tests/spikesorting/v2/test_fixture_ingestion.py:22-32`,
`tests/spikesorting/v2/test_fixture_ingestion.py:88-120`).

Impact: this is not a runtime bug, but it is a contributor extension hazard.
Someone regenerating or extending fixtures from the old plan could put planted
truth in the canonical units slot and poison sorter-output/parity workflows.

Fix direction:

- Mark the old Phase 0 fixture text as superseded or update it to the sidecar
  ground-truth contract.
- Add a lightweight docs consistency check around fixture ground-truth placement
  if these plan docs remain active contributor references.

