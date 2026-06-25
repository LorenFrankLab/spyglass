# Spike Sorting V2 Import-Time and Lazy-Boundary Review

Date: 2026-06-25

Scope: import-time side effects, DataJoint schema activation boundaries,
optional/heavy dependency imports, public facade import behavior, service-module
contracts, v0/v1 coexistence through `SpikeSortingOutput`, docs/notebook import
guidance, and test coverage for import-only safety. This is a different lens
from dependency version compatibility, docs link integrity, destructive
operations, and multi-user ownership.

Method: local static code/docs inspection plus two independent explorer-agent
reviews. This review is read-only except for this document. I did not run broad
imports or tests; one agent reported that its shell lacked `datajoint`, which is
itself a reminder that import-boundary tests need controlled subprocess
environments.

## Executive Summary

V2 has a strong direction: the package root is light, several service modules
are explicitly kept free of schema activation, visualization imports table
classes inside functions, and the UnitMatch backend keeps `UnitMatchPy` behind a
cached runtime guard. Those patterns are worth preserving.

The remaining import-boundary risks are concentrated in a few chokepoints.
Schema modules often import the `spyglass.common` package root, which activates
many unrelated common schemas and can run prepopulation at import. The shared
`SpikeSortingOutput` merge table probes v2 at module import, so v0/v1 and
decoding imports can still fan into v2. The `v2.utils` compatibility barrel
imports DataJoint and SpikeInterface at module import, so modules that look
"pure" or "DB-free" inherit a heavier dependency boundary than their docs/tests
make obvious.

## What Looks Solid

- `import spyglass.spikesorting.v2` is intentionally light: the package root
  imports only `_enums` eagerly, and `initialize_v2_defaults()` imports table
  modules only when called (`src/spyglass/spikesorting/v2/__init__.py:3-15`,
  `src/spyglass/spikesorting/v2/__init__.py:39-56`).
- The visualization facade keeps schema table classes and SpikeInterface widget
  imports inside plotting/export functions, while `available_visualizations()`
  is backed by a DB-free registry (`src/spyglass/spikesorting/v2/visualization.py:41-56`,
  `src/spyglass/spikesorting/v2/_visualization.py:1-25`).
- `UnitMatchPy` is not imported by the built-in backend until matching or bundle
  extraction actually needs it; `_require_unitmatch()` provides the actionable
  optional-extra error path (`src/spyglass/spikesorting/v2/_unitmatch_backend.py:73-89`).
- Worker/service modules such as `_artifact_compute.py`, `bad_channels.py`,
  `_recording_nwb.py`, `_recording_geometry.py`, `_analyzer_cache.py`, and
  `matcher_protocol.py` document and mostly follow the "no schema activation at
  import" pattern.
- There is already a subprocess import-contract test for service modules that
  asserts they do not import `spyglass.common` or v2 schema modules
  (`tests/spikesorting/v2/test_service_import_contracts.py:1-91`).

## Findings

### 1. High: v2 schema modules import the `spyglass.common` package root

Several v2 table modules import symbols through `from spyglass.common import ...`
instead of importing narrow source modules:

- `recording.py` imports `IntervalList`, `LabTeam`, and `Session`
  (`src/spyglass/spikesorting/v2/recording.py:27`).
- `sorting.py` imports `IntervalList`
  (`src/spyglass/spikesorting/v2/sorting.py:30`).
- `artifact.py` imports `IntervalList` and `Session`
  (`src/spyglass/spikesorting/v2/artifact.py:35`).
- `session_group.py` imports `IntervalList`, `LabTeam`, and `Session`
  (`src/spyglass/spikesorting/v2/session_group.py:23`).
- `unit_matching.py` imports `Session`
  (`src/spyglass/spikesorting/v2/unit_matching.py:41`).

The `spyglass.common` package root imports a broad set of behavior, device,
ephys, interval, lab, NWB, optogenetics, position, region, sensor, session,
subject, task, user, and helper modules (`src/spyglass/common/__init__.py:3-85`).
It also calls `prepopulate_default()` at import time when the global
`prepopulate` setting is enabled (`src/spyglass/common/__init__.py:141-142`).

Impact: importing one v2 schema module can activate a much larger common schema
surface than requested, and in some configurations can run prepopulation during
import. That makes v2 imports harder to reason about in notebooks, docs builds,
subprocess workers, and safety guards.

Recommended fix: replace root imports with narrow module imports, for example
`spyglass.common.common_interval.IntervalList`,
`spyglass.common.common_lab.LabTeam`,
`spyglass.common.common_session.Session`, and
`spyglass.common.common_ephys.Raw`. Add a static or subprocess test that v2
schema modules import only the common modules they directly FK, not the
`spyglass.common` aggregator.

### 2. High: `SpikeSortingOutput` eagerly probes v2 during shared merge-table import

`spikesorting_merge.py` imports `spyglass.spikesorting.v2.curation.CurationV2`
at module import so it can conditionally declare the `SpikeSortingOutput.CurationV2`
part table (`src/spyglass/spikesorting/spikesorting_merge.py:24-44`,
`src/spyglass/spikesorting/spikesorting_merge.py:127-134`). The import is
wrapped so v0/v1-only environments keep working, and the original exception is
saved for later v2-specific errors. That is a useful compatibility pattern.

The cost is that v0/v1 and decoding code paths that only need the shared merge
table still attempt a v2 import. If v2 fails once, the process defines
`SpikeSortingOutput` without the v2 part until restart. The broad
`except Exception` also treats a real v2 import bug the same way it treats an
expected non-local DB guard or missing dependency.

Impact: v0/v1-only users still pay the v2 import/guard/heavy-dependency probe.
In a long-lived process, a transient or buggy v2 import failure can make v2
merge outputs unavailable for the lifetime of that process.

Recommended fix: separate expected "v2 intentionally unavailable" conditions
from unexpected import bugs. At minimum, pre-check the v2 DB-host guard before
importing the full curation schema and narrow the caught exception types. Longer
term, consider a v2-specific merge registration path or a dedicated lightweight
FK declaration module so unrelated downstream imports do not probe the full v2
curation stack. Add a subprocess test proving v1/decoding imports do not import
v2 unless v2 is available or explicitly requested.

### 3. Medium-high: `v2.utils` is a heavy compatibility barrel for otherwise light helpers

`utils.py` imports both DataJoint and SpikeInterface at module import
(`src/spyglass/spikesorting/v2/utils.py:12-13`). The SpikeInterface import is
used by `_resolved_job_kwargs()` for `si.get_global_job_kwargs()`
(`src/spyglass/spikesorting/v2/utils.py:608-632`), but the module also re-exports
many pure helpers from `_signal_math`, `_reference_resolution`,
`_lookup_validation`, and `_nwb_metadata_helpers`
(`src/spyglass/spikesorting/v2/utils.py:15-50`).

As a result, modules that import a pure helper through `utils` inherit DataJoint
and SpikeInterface at import. Examples include `_recipe_catalog.py`, which
documents "No DB connection or `dj.schema` activation at import" but imports
`_validate_params` through `utils`
(`src/spyglass/spikesorting/v2/_recipe_catalog.py:11-16`,
`src/spyglass/spikesorting/v2/_recipe_catalog.py:39`), and
`_sort_group_planning.py`, which imports reference-resolution helpers through
`utils` (`src/spyglass/spikesorting/v2/_sort_group_planning.py:1-21`).

Impact: the existing boundary is "no `spyglass.common` / no v2 schema module,"
not "minimal dependencies." That distinction is easy to miss. Public imports
such as `spyglass.spikesorting.v2.pipeline` can require the broader v2 runtime
even for DB-free catalog discovery, because `pipeline.py` imports
`_pipeline_presets`, `_pipeline_presets` imports `_recipe_catalog`, and
`_recipe_catalog` imports `utils`
(`src/spyglass/spikesorting/v2/pipeline.py:34-62`,
`src/spyglass/spikesorting/v2/_pipeline_presets.py:16`,
`src/spyglass/spikesorting/v2/_recipe_catalog.py:39`).

Recommended fix: stop routing pure-helper imports through `utils` in modules
that claim to be dependency-light. Import `_validate_params` directly from
`_lookup_validation`, reference helpers directly from `_reference_resolution`,
and signal helpers directly from `_signal_math`. Move `import spikeinterface as
si` inside `_resolved_job_kwargs()`. Then strengthen
`test_service_import_contracts.py` to assert either absence or allowed presence
of `datajoint`, `spikeinterface`, and `spyglass.utils` per module.

### 4. Medium: common NWB table declaration pulls file-I/O runtime dependencies early

V2 schema modules need `AnalysisNwbfile` FKs, but importing
`spyglass.common.common_nwbfile` imports `h5py`, `pandas`, `pynwb`,
`spikeinterface`, HDMF, and `tqdm` before declaring the DataJoint schema
(`src/spyglass/common/common_nwbfile.py:6-14`,
`src/spyglass/common/common_nwbfile.py:41-45`).

Impact: merely declaring a table with an `AnalysisNwbfile` FK pays for the
NWB/SpikeInterface file-I/O stack before any NWB file is read or written. This
is broader than v2 and may be accepted Spyglass-wide behavior, but it limits how
light v2 schema imports can become.

Recommended fix: treat this as a shared longer-term refactor rather than a v2
blocker. Split table definitions and file-I/O helpers where practical, or move
NWB/SI imports into the methods that actually open files. If that is too large,
document that v2 schema modules cannot be import-light while they FK
`AnalysisNwbfile`.

### 5. Medium: public facade dependencies are not pinned by import-boundary tests

The package root promises that table/helper modules are imported lazily and that
optional runtime dependencies are avoided until a submodule is used
(`src/spyglass/spikesorting/v2/__init__.py:3-12`). Public submodules are allowed
to import more, but the current tests mostly assert no database query or no
schema/common import, not the dependency surface itself.

For example, `_pipeline_presets.py` imports Pydantic at module import
(`src/spyglass/spikesorting/v2/_pipeline_presets.py:14`), and
`visualization.py` imports `_visualization`, which imports pandas at module
import (`src/spyglass/spikesorting/v2/visualization.py:49`,
`src/spyglass/spikesorting/v2/_visualization.py:25`). `pyproject.toml` includes
many scientific dependencies directly, including SpikeInterface and PyNWB, but
does not list `pandas` or `pydantic` directly in core or the v2 extras
(`pyproject.toml:43-77`, `pyproject.toml:112-146`).

Impact: CI may pass because transitive dependencies are present, while the
actual public import contract is not explicit. If a resolver or upstream package
stops pulling pandas or Pydantic transitively, public v2 imports can fail before
an actionable v2 error path runs.

Recommended fix: either declare direct dependencies that public imports require,
or move those imports behind function-local guards with clear install messages.
Add subprocess import-only tests for `spyglass.spikesorting.v2.pipeline` and
`spyglass.spikesorting.v2.visualization` in dependency-blocked or dependency-
mocked environments.

### 6. Medium-low: `metric_curation.py` and `recompute.py` rely on transitive DB-host guard coverage

Most v2 schema modules call `_assert_v2_db_safe()` immediately before
`dj.schema(...)` (`src/spyglass/spikesorting/v2/recording.py:90-91`,
`src/spyglass/spikesorting/v2/sorting.py:198-199`,
`src/spyglass/spikesorting/v2/curation.py:60-61`,
`src/spyglass/spikesorting/v2/artifact.py:82-83`,
`src/spyglass/spikesorting/v2/unit_matching.py:61-62`,
`src/spyglass/spikesorting/v2/session_group.py:46-47`). `metric_curation.py`
and `recompute.py` declare schemas without their own direct guard
(`src/spyglass/spikesorting/v2/metric_curation.py:73`,
`src/spyglass/spikesorting/v2/recompute.py:64-66`).

Today they are guarded transitively because they import guarded v2 schema
modules first. That is easy to break during refactors and contradicts the
contract in `_assert_v2_db_safe()` that every schema module calls it before
schema registration (`src/spyglass/spikesorting/v2/utils.py:513-557`).

Impact: low immediate risk, but a fragile safety invariant around the most
important import-time guard.

Recommended fix: import `_assert_v2_db_safe` and call it directly immediately
before `schema = dj.schema(...)` in both modules. Add a static test that every
file containing `schema = dj.schema("spikesorting_v2_...")` also contains a
direct `_assert_v2_db_safe()` call before schema declaration, or is explicitly
allowlisted with a reason.

### 7. Medium-low: docs and notebooks describe stale UnitMatch import behavior

The migration docs still say `figpack_curation`, `unit_matching`, and
`matcher_protocol` are import-safe placeholders
(`docs/src/Features/SpikeSortingV2_Migration.md:153-160`). The main v2 docs
also still mark cross-session unit matching as unavailable in the status section
(`docs/src/Features/SpikeSortingV2.md:1055-1060`). In code, `unit_matching.py`
is a real schema module and creates `spikesorting_v2_unit_matching`
(`src/spyglass/spikesorting/v2/unit_matching.py:39-62`). The stub tests already
know `unit_matching` and `matcher_protocol` are no longer stubs
(`tests/spikesorting/v2/test_legacy_stub_imports.py:24-39`).

The UnitMatch notebook is also historical: it says the work is before a
DataJoint wrapper is written and imports `UnitMatchPy` at top level
(`notebooks/py_scripts/13_UnitMatch_Cross_Session.py:15-21`,
`notebooks/py_scripts/13_UnitMatch_Cross_Session.py:61`), bypassing the
production backend's lazy optional-extra guard.

Impact: users can follow docs that promise placeholder import behavior or
top-level `UnitMatchPy` imports even though the production path is now a real
DataJoint wrapper with a guarded backend.

Recommended fix: update migration/status docs so only `figpack_curation` is
described as a stub, and label the old UnitMatch notebook as historical/internal
or rewrite it to use `SessionGroup`, `UnitMatchSelection`, and `UnitMatch`. Add a
docs/notebook scan that fails if `unit_matching` or `matcher_protocol` reappear
in placeholder language, or if public notebooks contain top-level
`import UnitMatchPy`.

## Suggested Priority

1. Replace v2 `from spyglass.common import ...` imports with narrow common-module
   imports.
2. Move pure-helper imports off the `utils` barrel and make the SpikeInterface
   import in `_resolved_job_kwargs()` function-local.
3. Add direct `_assert_v2_db_safe()` calls to `metric_curation.py` and
   `recompute.py`.
4. Tighten `SpikeSortingOutput`'s v2 probe so expected unavailability is
   separated from real import bugs.
5. Update UnitMatch docs/notebooks and strengthen subprocess import-boundary
   tests to cover dependency leaks, not just schema/common leaks.

