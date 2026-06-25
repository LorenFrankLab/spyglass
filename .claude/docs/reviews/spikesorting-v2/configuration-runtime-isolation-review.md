# Spike Sorting V2 Configuration and Runtime-Isolation Review

Date: 2026-06-25

Scope: configuration precedence, DataJoint/Spyglass settings, environment
variables, path roots, temporary and cache directories, SpikeInterface global
state, `job_kwargs`, sorter container `execution_params`, v2 DB safety guards,
recompute environment gates, and mutable process-global registries. This is a
different lens from import boundaries, ownership, destructive operations,
performance, and error taxonomy, though some findings touch those surfaces.

Method: local static code/docs/test inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

V2 has several strong isolation patterns: sorter container execution is
first-class tracked provenance, sorter scratch is isolated per sort,
SpikeInterface global job kwargs are restored after sorter runs, analyzer-cache
path policy is centralized, the v2 DB guard blocks accidental schema registration
on non-local hosts, and recompute/delete requires a current-environment match by
default.

The weaker spots are mostly "ambient configuration" seams. Effective
`job_kwargs` can come from mutable process/global state without becoming part of
row identity. Spyglass directory settings are often imported or cached as module
constants, so same-interpreter config reloads can leave v2 writing through stale
paths. Analyzer cache folders are keyed by deterministic `sorting_id` plus
recipe, but not by database/runtime namespace or a manifest. Some schema modules
still rely on transitive v2 DB guard coverage. Several large temporary
directories use system temp rather than Spyglass temp. Docs surface many of the
right knobs late or indirectly, so users may only discover them after a failed
run.

## What Looks Solid

- Sorter execution provenance is separated from scientific sorter params:
  `SorterParameters.execution_params` tracks backend/image/install controls, and
  reserved execution keys are rejected from sorter params and `job_kwargs`
  (`src/spyglass/spikesorting/v2/_params/sorter.py:331-591`,
  `src/spyglass/spikesorting/v2/sorting.py:215-229`).
- `run_si_sorter()` creates a per-sort scratch root under Spyglass temp,
  scopes the NumPy `np.Inf` compatibility patch, installs SI global job kwargs
  only for the run, restores the previous global state, and cleans the temp dir
  in `finally` (`src/spyglass/spikesorting/v2/_sorting_dispatch.py:467-621`).
- Analyzer cache policy is centralized in `_analyzer_cache.py`; relocation is
  explicitly treated as a cache miss and has focused tests
  (`src/spyglass/spikesorting/v2/_analyzer_cache.py:1-55`,
  `tests/spikesorting/v2/test_analyzer_lifecycle.py:308-345`).
- Container runtime preflight is explicit and does not silently fall back from a
  containerized preset to local execution
  (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:536-564`).
- Recompute/delete has a current-environment gate via `StaleEnvMatchedError`;
  stale historic matches do not authorize deletion by default
  (`src/spyglass/spikesorting/v2/recompute.py:16-20`,
  `src/spyglass/spikesorting/v2/recompute.py:1173-1205`).
- The standalone v2 test bootstrap is conservative about base directories,
  schema prefixes, and production-smoke reads
  (`tests/spikesorting/v2/test_env.py:1-225`,
  `tests/spikesorting/v2/test_bootstrap_safety_guards.py:1-94`).

## Findings

### 1. High: effective `job_kwargs` can come from mutable global/config state

`parameter_fingerprint()` includes per-row `job_kwargs` in parameter identity
(`src/spyglass/spikesorting/v2/_parameter_identity.py:78-125`). That says
`job_kwargs` are provenance-bearing enough to distinguish rows.

At compute time, however, `_resolved_job_kwargs()` starts from the current
SpikeInterface global kwargs, then applies
`dj.config["custom"]["spikesorting_v2_job_kwargs"]`, then applies per-row blobs
(`src/spyglass/spikesorting/v2/utils.py:607-632`). Sorting, analyzer extension
builds, artifact detection, concat/motion correction, and UnitMatch all consume
that resolved dict (`src/spyglass/spikesorting/v2/sorting.py:1519-1563`,
`src/spyglass/spikesorting/v2/artifact.py:900-919`,
`src/spyglass/spikesorting/v2/session_group.py:827-849`,
`src/spyglass/spikesorting/v2/unit_matching.py:768-799`).

Impact: two processes can compute the same table rows with different ambient SI
globals or DataJoint custom config, while only the per-row layer is captured in
row identity. If these values are pure resource controls, they should not fork
identity at all. If any can affect chunk boundaries, random sampling, temporary
layout, or extension output, the ambient layers need provenance.

Recommended fix:

- Decide the contract explicitly:
  - resource-only kwargs are execution hints and should not affect row identity;
    in that case ignore `si.get_global_job_kwargs()` by default and validate
    `dj.config` kwargs as resource-only; or
  - effective kwargs are provenance, in which case persist or fingerprint the
    resolved ambient layers.
- Keep `random_seed` out of ambient config unless it is persisted, since v2
  already treats it as a reproducibility knob in per-row `job_kwargs`.
- Add tests that set conflicting SI globals and DataJoint custom kwargs with
  identical rows and assert either the output identity/provenance changes or the
  ambient state is ignored.

### 2. High: Spyglass directory configuration is cached into module/class globals

Spyglass settings are loaded at import/startup
(`src/spyglass/settings.py:123-129`,
`src/spyglass/settings.py:638-650`), and common helpers import directory values
as module constants, for example `common_nwbfile.py` imports `analysis_dir` and
`raw_dir` directly (`src/spyglass/common/common_nwbfile.py:16-18`).
`AnalysisMixin` also caches `_analysis_dir` on the class via
`_cached_analysis_dir` (`src/spyglass/utils/mixins/analysis.py:106-189`).

Impact: a notebook, test, worker, or fixture script that reloads config or
changes `SPYGLASS_BASE_DIR` / directory settings after imports can keep reading
or writing v2 artifacts through stale raw/analysis/temp roots. This is especially
dangerous for v2 because recording NWBs and units NWBs are large, expensive
artifacts and because cache/recompute behavior assumes path resolution is
consistent.

Recommended fix:

- Replace imported directory strings in artifact paths with runtime accessors
  where possible.
- Clear dependent class/module caches when `load_config(force_reload=True)` is
  used.
- Add tests that import v2/common NWB helpers, reload config to new raw/analysis
  dirs, and assert subsequent artifact creation and path resolution use only the
  new directories.

### 3. High: analyzer cache paths are not namespaced by database/runtime context

Analyzer folders resolve to:

`analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.zarr"`

The root comes from `dj.config["custom"]["spikesorting_v2_analyzer_dir"]` or
`Path(temp_dir) / "spikesorting_v2" / "analyzers"`
(`src/spyglass/spikesorting/v2/_analyzer_cache.py:35-89`). The `sorting_id`
itself is deterministic from logical selection content, not from database host,
database prefix, analysis root, raw NWB path, or runtime environment
(`src/spyglass/spikesorting/v2/_selection_identity.py:1-38`).

Impact: two databases, two users, or a test and production process sharing one
analyzer root can resolve the same deterministic `sorting_id`/recipe to the same
folder. `Sorting.get_analyzer()` will load an existing folder before rebuilding
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:210-257`), so a cache folder
from another namespace can be reused if the folder is loadable.

Recommended fix:

- Namespace analyzer paths by an explicit stable context such as database host +
  prefix/schema + configured analysis/base root, or require a v2 analyzer-cache
  namespace config value.
- Add an analyzer manifest inside each folder with sorting id, waveform params
  name, DB namespace, source NWB/object identifiers, Spyglass/SI versions, and
  relevant parameter fingerprints. Reject/rebuild on manifest mismatch.
- Add a test that creates the same deterministic `sorting_id` under two
  namespaces sharing one root and verifies they do not load each other's cache.

### 4. Medium-high: v2 DB safety guard is incomplete and mostly import-time

`_assert_v2_db_safe()` refuses non-local DB hosts unless
`SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1`
(`src/spyglass/spikesorting/v2/utils.py:513-557`). Many schema modules call it
before `dj.schema(...)`, including recording, artifact, sorting, curation,
session_group, and unit_matching
(`src/spyglass/spikesorting/v2/recording.py:90-91`,
`src/spyglass/spikesorting/v2/artifact.py:82-83`,
`src/spyglass/spikesorting/v2/sorting.py:198-199`).

`metric_curation.py` and `recompute.py` declare v2 schemas without a direct guard
(`src/spyglass/spikesorting/v2/metric_curation.py:73`,
`src/spyglass/spikesorting/v2/recompute.py:67`). They are usually protected
transitively today because they import guarded v2 modules first, but that is
fragile. The guard also runs at import/schema declaration time, not necessarily
at later mutation or destructive entry points after a long-lived process changes
DataJoint config or connection state.

Recommended fix:

- Add explicit `_assert_v2_db_safe()` calls immediately before every
  `dj.schema("spikesorting_v2_*")`, including `metric_curation.py` and
  `recompute.py`.
- Add a static test that every v2 schema module guards locally before schema
  declaration.
- Re-check the active connection/host at high-risk mutation and destructive
  operations, especially recompute delete paths.

### 5. Medium: the v2 DB guard is not discoverable before import-time failure

The user docs and notebook import guarded v2 modules after ordinary DataJoint
setup assumptions (`notebooks/py_scripts/10_Spike_SortingV2.py:24-42`). The
environment section documents SI requirements
(`docs/src/Features/SpikeSortingV2.md:1042-1053`), but does not explain the v2
DB host guard, allowed hosts, override env var, or restart/reimport guidance.

Impact: users connected to a normal lab DataJoint host may hit a `RuntimeError`
during import before reaching any setup explanation. The error message is
actionable for tests, but not enough as production documentation.

Recommended fix:

- Add a "V2 database safety guard" setup block before v2 imports in the docs and
  notebook.
- Explain allowed hosts, `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1`, when it
  is appropriate, and why a kernel/process restart is safest after changing it.
- Clarify that `bootstrap_v2_test_environment()` is for fixture/test scripts, not
  general production setup.
- Add a docs/notebook smoke assertion that the env var and guard text appear.

### 6. Medium: large temporary work dirs bypass configured Spyglass temp

Sorter scratch is correctly rooted under `spyglass.settings.temp_dir`
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:467-483`). But UnitMatch
bundle extraction uses `tempfile.TemporaryDirectory(prefix="unitmatch_")` without
a configured directory (`src/spyglass/spikesorting/v2/unit_matching.py:754-780`),
and analyzer recompute uses `tempfile.mkdtemp(prefix="v2_analyzer_recompute_")`
without a configured directory (`src/spyglass/spikesorting/v2/recompute.py:1107-1110`).

Impact: large waveform bundles or analyzer rebuilds can land in system `/tmp`
rather than lab/HPC scratch. That can fill small volumes, be purged unexpectedly,
or make container/shared-filesystem behavior inconsistent with sorter scratch.

Recommended fix:

- Anchor UnitMatch and analyzer-recompute temp dirs under `spyglass.settings.temp_dir`
  or a v2-specific scratch config key.
- Add tests that set `SPYGLASS_TEMP_DIR`/Spyglass temp config and assert these
  temp roots are created there and cleaned on success and failure.

### 7. Medium: storage roots and cache behavior are underdocumented and not preflighted

`spikesorting_v2_analyzer_dir` is a major storage knob, but docs mostly surface
only the global `spikesorting_v2_job_kwargs` setting late in the performance
section (`docs/src/Features/SpikeSortingV2.md:1082-1090`). The docs do not yet
give a practical storage model for `SPYGLASS_BASE_DIR`, raw/analysis/temp dirs,
analyzer cache roots, writability, shared-vs-local storage, cleanup, rebuilds, or
orphan discovery.

Preflight checks database rows and sorter/container runtime, but it does not
check the effective analysis/temp/analyzer roots for existence, writability, or
available location class (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:358-400`
and nearby).

Recommended fix:

- Add an "Execution resources and storage" section near the first run example.
- Show a notebook cell printing effective raw/analysis/temp/analyzer roots.
- Explain that recording/unit NWBs are canonical artifacts, analyzer folders are
  regeneratable scratch, and moving the analyzer root creates cache misses.
- Add preflight warnings or failures for missing/unwritable temp and analyzer
  roots.

### 8. Medium: NWB artifact creation hard-depends on `conda env export`

`AnalysisMixin._logged_env_info()` shells out to `conda env export`
(`src/spyglass/utils/mixins/analysis.py:311-320`), and `_alter_spyglass_version()`
writes that string into analysis NWBs
(`src/spyglass/utils/mixins/analysis.py:306-309`). V2 recording and units NWB
writers use `AnalysisNwbfile.create`, so this shared behavior is in the v2 write
path.

Impact: v2 artifact writes can fail in venv/uv/pip-only containers where `conda`
is absent or broken. Even when it works, the broad exported environment is a
machine-specific log, not a stable minimal runtime fingerprint.

Recommended fix:

- Make environment capture best-effort for artifact creation.
- Fall back to structured `importlib.metadata`, Python/platform, Spyglass,
  DataJoint, PyNWB, HDF5, NumPy, and SpikeInterface versions.
- Add tests that simulate `FileNotFoundError` and `CalledProcessError` from
  `subprocess.check_output` while creating v2 analysis artifacts.

### 9. Medium: container execution configuration is powerful but under-discovered

The code has a strong `execution_params` model, but the docs mostly mention the
shipped Singularity MS4 preset and say Docker is a user-insertable row away
(`docs/src/Features/SpikeSortingV2.md:309-316`). The notebook uses preset
listing, but does not show `describe_pipeline_preset()`'s execution block or a
custom Docker/Singularity insert example.

Impact: users can select containerized presets without easily seeing the exact
image/install mode, or may try putting `docker_image`/`singularity_image` into
the rejected params/job-kwargs locations.

Recommended fix:

- Document `SorterParameters.execution_params` fields and show one Docker and
  one Singularity row insert.
- In the notebook, call `describe_pipeline_preset(pipeline_preset)` and point to
  the `sorter_execution` block.
- Add docs tests for `execution_params`, `container_image`, and
  `describe_pipeline_preset`.

### 10. Medium-low: matcher registration is process-global mutable state

The matcher registry is module-level mutable state keyed by matcher name
(`src/spyglass/spikesorting/v2/matcher_protocol.py:100-124`). `replace=True` is
needed for built-in idempotent registration, but the table stores matcher name,
not backend module/version. UnitMatch import is cached and installs a NumPy-2
compatibility shim in the third-party package path
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:73-85`,
`src/spyglass/spikesorting/v2/_unitmatch_backend.py:394-402`).

Impact: a custom matcher name can mean different code in another process unless
the registry setup is repeated exactly. Replacement can alter the meaning of
existing matcher rows by name.

Recommended fix:

- Add a registry reset/context manager for tests and a clear fresh-process
  failure test for missing custom matcher registration.
- Persist matcher backend module/entry point/version with matcher parameters or
  outputs, or restrict replacement of referenced matcher names.

## Suggested Repair Order

1. Settle the `job_kwargs` contract: ignore ambient globals or persist the
   effective ambient layer.
2. Add analyzer cache namespace/manifest validation before wider shared-cache
   usage.
3. Add local `_assert_v2_db_safe()` guards to every v2 schema module and
   re-check high-risk mutation/destructive paths.
4. Route UnitMatch and analyzer-recompute temp dirs through Spyglass temp.
5. Document the DB guard, storage roots, cache model, `job_kwargs` precedence,
   and container `execution_params` near the first-run docs.
6. Make `conda env export` best-effort with a structured fallback.

