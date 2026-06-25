# Spike Sorting V2 Security and External-Boundary Review

Date: 2026-06-25

Scope: filesystem path handling, raw/analysis NWB filenames, analysis-file
permissions, sorter scratch permissions, container/image execution boundaries,
DataJoint string restrictions, analyzer cache deletion paths, nonlocal database
operator boundaries, and docs/tests that teach safe use. This is a different
lens from destructive-admin safety, configuration isolation, concurrency, and
scientific reproducibility, though several findings overlap those surfaces.

Method: local static code/docs/test inspection plus two independent
explorer-agent reviews. This review is read-only except for this document. I did
not run tests.

## Executive Summary

V2 has good internal hygiene in many places: deterministic selection ids,
Pydantic parameter validation, reserved execution-key rejection, dict
restrictions across most v2 table code, explicit staged-artifact cleanup, safe
delete overrides that respect cancelled safemode prompts, recompute dry-run and
current-environment gates, and path-safe analyzer waveform recipe names.

The main security risks sit at external boundaries. Shared NWB path helpers
still concatenate database filenames into filesystem paths without an explicit
"stay under root" check. V2 analysis NWBs inherit world-writable default
permissions from `AnalysisNwbfile.create()`, and sorter scratch is chmodded
world-writable for every run. Container execution is intentionally powerful, but
the production trust model is not explicit: mutable image tags, arbitrary
`extra_requirements`, and GitHub installs are fine only when parameter-row
writers are already trusted to execute code on populate workers. Analyzer-cache
helpers are mostly called safely through table APIs, but the helper boundary
itself still trusts raw path components.

## What Looks Solid

- V2 table code mostly uses dict restrictions rather than formatted SQL-like
  strings for user-facing row selection.
- `AnalyzerWaveformParameters` rejects unsafe recipe names before normal rows can
  embed them in analyzer folder names
  (`src/spyglass/spikesorting/v2/sorting.py:561-579`).
- Sorter scientific params and execution/container controls are separated:
  reserved execution keys are rejected from sorter params and `job_kwargs`, then
  container kwargs are built only from validated `execution_params`
  (`src/spyglass/spikesorting/v2/_params/sorter.py:331-591`,
  `src/spyglass/spikesorting/v2/_sorting_dispatch.py:520-558`).
- Staged recording and units NWB writers clean up partial files on write/hash
  failure (`src/spyglass/spikesorting/v2/_recording_nwb.py:167-180`,
  `src/spyglass/spikesorting/v2/_recording_nwb.py:258-275`,
  `src/spyglass/spikesorting/v2/_units_nwb.py:543-562`,
  `src/spyglass/spikesorting/v2/_units_nwb.py:735-756`).
- Destructive recompute defaults to dry-run and requires a current
  `UserEnvironment` match by default before deleting matched artifacts
  (`src/spyglass/spikesorting/v2/recompute.py:481-506`,
  `src/spyglass/spikesorting/v2/recompute.py:1163-1205`).
- `Sorting.delete()` snapshots rows before DB deletion and removes analyzer
  cache folders only for rows that actually disappeared, so cancelled safemode
  prompts do not trigger side-effect cleanup
  (`src/spyglass/spikesorting/v2/sorting.py:2084-2154`).

## Findings

### 1. High: NWB filenames can escape configured raw/analysis roots

The shared path helpers treat database filenames as path fragments. Raw NWB
resolution is `raw_dir + "/" + nwb_file_name`
(`src/spyglass/common/common_nwbfile.py:107`). Analysis-file resolution builds
paths from `analysis_file_name` with `Path(analysis_dir) / fname` and
`__get_file_parent(fname) / fname`
(`src/spyglass/utils/mixins/analysis.py:367-396`).

V2 calls these helpers for recording, sorting, curation, metrics, and UnitMatch
artifacts, for example `AnalysisNwbfile().create(...)` in
`_recording_nwb.py` and `_units_nwb.py`
(`src/spyglass/spikesorting/v2/_recording_nwb.py:172-175`,
`src/spyglass/spikesorting/v2/_units_nwb.py:543`,
`src/spyglass/spikesorting/v2/_units_nwb.py:735`). New analysis file names are
derived from `os.path.splitext(nwb_file_name)[0]` plus a random suffix
(`src/spyglass/utils/mixins/analysis.py:266-268`,
`src/spyglass/utils/mixins/analysis.py:346-359`), so path separators in the
source filename can propagate into generated analysis filenames.

Impact: a crafted `nwb_file_name` or `analysis_file_name` containing `/`, `..`,
or an absolute path can steer writes, chmods, lookup, and cleanup outside the
configured raw/analysis roots. This is most concerning in shared DataJoint
deployments where file-name rows can be inserted by users or scripts that are
not fully trusted filesystem operators.

Recommended fix:

- Validate raw and analysis filenames as basenames at insertion and at path
  helper boundaries: reject absolute paths, path separators, `..`, and empty
  names.
- Resolve the computed path and require
  `computed.resolve().relative_to(root.resolve())` before read, write, chmod,
  or unlink operations.
- Generate v2 analysis basenames from `Path(nwb_file_name).name`, then apply a
  strict filename regex.
- Add tests for `../evil.nwb`, `/tmp/evil.nwb`, `nested/evil.nwb`, and names
  with quotes/slashes through `Nwbfile.get_abs_path()`,
  `AnalysisNwbfile.create()`, and `AnalysisNwbfile.get_abs_path()`.

### 2. High in shared deployments: v2 artifacts and sorter scratch are world-writable

`AnalysisNwbfile.create()` defaults `restrict_permission=False`; the docstring
describes that as "no permission restriction (666)", and the implementation
chmods the file to `0o666` unless restricted
(`src/spyglass/utils/mixins/analysis.py:210-227`,
`src/spyglass/utils/mixins/analysis.py:300-302`). V2 artifact writers do not
pass `restrict_permission=True`
(`src/spyglass/spikesorting/v2/_recording_nwb.py:172-175`,
`src/spyglass/spikesorting/v2/_units_nwb.py:543`,
`src/spyglass/spikesorting/v2/_units_nwb.py:735`).

Sorter scratch starts as a private `TemporaryDirectory`, but `run_si_sorter()`
immediately chmods the per-sort directory to `0o777`
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:408-415`,
`src/spyglass/spikesorting/v2/_sorting_dispatch.py:480-486`). The comment says
this supports container/subprocess UID mismatches, but the mode is applied for
local backend runs too.

Impact: on shared hosts or shared storage, other OS users can mutate registered
analysis NWBs or inspect/tamper with sorter inputs and outputs during execution.
Post-registration hashes may detect some later changes, but they do not prevent
races, denial of service, or pre-registration tampering.

Recommended fix:

- Default v2 analysis artifacts to restricted permissions, ideally `0600` or a
  configurable group-safe `0640`/`0660` depending on lab policy.
- If shared write is required, use a configured group/ACL model rather than
  world-writable files.
- Keep local sorter scratch at the private `TemporaryDirectory` mode. For
  container backends, prefer running the container with the worker uid/gid,
  group ACLs, or an explicit opt-in scratch mode with docs explaining the risk.
- Add tests asserting created v2 NWB modes and local/container sorter temp-dir
  modes.

### 3. High if parameter rows are not admin-only: sorter/container rows are a code-execution boundary

`SorterParameters` accepts any installed SpikeInterface sorter name through the
validated lookup row path (`src/spyglass/spikesorting/v2/sorting.py:286`).
`execution_params` accept container backend, container image,
`installation_mode`, `spikeinterface_version`, and `extra_requirements`
(`src/spyglass/spikesorting/v2/_params/sorter.py:368-375`). Those values flow to
SpikeInterface's sorter/container execution path
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:121-134`,
`src/spyglass/spikesorting/v2/_sorting_dispatch.py:546-574`).

That is an appropriate power tool when `SorterParameters` writers are already
trusted to run code on populate workers. It is a security boundary if those rows
are writable by ordinary users in a shared database.

Impact: a user who can insert arbitrary sorter/container parameter rows can ask
workers to pull and execute arbitrary images or install arbitrary packages into
the runtime. That may expose mounted data, scratch directories, worker
credentials, or lab compute resources.

Recommended fix:

- Decide and document the policy: `SorterParameters` writers are trusted compute
  operators, or custom execution rows require admin approval.
- In production, allowlist sorter names, image registries/digests or SIF roots,
  and supported `installation_mode` values.
- Consider disabling `github` installs and arbitrary `extra_requirements` for
  non-admin rows.
- Add tests that unapproved image/install-mode combinations are rejected while
  shipped defaults pass.

### 4. Medium-high: "pinned" container docs are tag-based, not immutable

The main docs describe the MS4 Singularity recipe as a "pinned Singularity
container" (`docs/src/Features/SpikeSortingV2.md:309-315`). The shipped recipe
uses `spikeinterface/mountainsort4-base:1.0.5`
(`src/spyglass/spikesorting/v2/_recipe_catalog.py:68-78`), and tests assert that
recommended container rows pin the container-side SpikeInterface runtime, not
that the image reference itself is immutable
(`tests/spikesorting/v2/test_sorter_execution_params.py:181-188`).

Impact: a Docker tag can be retargeted. Operators may execute different sorter
code than the reviewed row implies, which is both a supply-chain issue and a
reproducibility issue.

Recommended fix:

- Distinguish "tagged" from "immutable" in docs and comments.
- Prefer digest-pinned Docker references (`@sha256:...`) or verified local SIF
  files with recorded hashes for shipped production rows.
- Store or report the resolved image digest/SIF hash in execution provenance.
- Add tests that production/recommended container rows use an immutable image
  reference or a validated local artifact hash, not merely `name:tag`.

### 5. Medium: analyzer cache path/deletion helpers rely on caller discipline

`analyzer_path()` embeds raw `sorting_id` and `waveform_params_name` into a
folder name (`src/spyglass/spikesorting/v2/_analyzer_cache.py:57-89`). Normal
`AnalyzerWaveformParameters` rows validate `waveform_params_name`, but the public
helper boundary does not enforce the same regex itself. `remove_analyzer_cache()`
uses `root.glob(f"{sorting_id}__*.zarr")` and removes every match
(`src/spyglass/spikesorting/v2/_analyzer_cache.py:138-183`). `get_analyzer()`
computes and may delete the folder before validating direct helper arguments
through a table lookup (`src/spyglass/spikesorting/v2/_sorting_analyzer.py:199-240`).

`Sorting.find_orphaned_analyzer_folders(dry_run=False)` lists every directory
under the analyzer root and deletes disk-side orphans after interactive
confirmation, without filtering to analyzer-shaped `.zarr` names
(`src/spyglass/spikesorting/v2/sorting.py:2273-2327`).

Impact: table-mediated calls are usually safe, but direct helper calls or a
misconfigured analyzer root can broaden deletion. A malicious or mistaken
`sorting_id` containing glob metacharacters could match unexpected folders under
the analyzer root.

Recommended fix:

- Validate `sorting_id` as a UUID and `waveform_params_name` with the same
  `^[A-Za-z0-9_]+$` rule inside `analyzer_path()` and
  `remove_analyzer_cache()`, not only at table insert time.
- Resolve computed analyzer paths and require they remain under the resolved
  analyzer root.
- Filter orphan cleanup to expected analyzer folder names, for example
  `<uuid>__<safe_recipe>.zarr`.
- Add traversal/glob/sentinel tests for `sorting_id="*"`,
  `sorting_id="../outside"`, and unsafe waveform names.

### 6. Medium: shared file helpers still use formatted DataJoint string restrictions

V2 table code mostly avoids raw string interpolation, but shared helpers used by
v2 still interpolate filenames and paths into SQL-like DataJoint restrictions.
Examples include `AnalysisNwbfile.get_abs_path(..., from_schema=True)` using
`filepath LIKE '%{analysis_nwb_file_name}'`
(`src/spyglass/utils/mixins/analysis.py:574-581`) and
`_update_permissions()` using `filepath = '{str(file_path)}'`
(`src/spyglass/utils/mixins/analysis.py:833-835`). The broader helper layer has
similar formatted restriction patterns in `dj_helper_fn.py`.

Impact: quotes, `%`, or `_` in filenames can broaden or misroute lookups,
permission updates, and checksum updates. This is not necessarily remote SQL
injection in the classic sense, but it is an unsafe database-boundary pattern
for file identity.

Recommended fix:

- Replace formatted string restrictions with exact dict restrictions whenever
  the relative path is known.
- If substring matching is truly required for legacy external-table rows, escape
  `LIKE` wildcards and reject quotes/path separators in the public filename.
- Add tests with quotes, `%`, `_`, parentheses, and path separators to verify
  exact-match behavior or explicit rejection.

### 7. Medium: operator docs do not surface the database and deletion safety boundaries early enough

Runtime code blocks nonlocal DataJoint hosts unless
`SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1` is set
(`src/spyglass/spikesorting/v2/utils.py:513-557`), and tests cover the override
semantics. The main V2 environment docs cover package installation but not the
nonlocal DB guard, allowed hosts, override meaning, or recommended checks before
write examples (`docs/src/Features/SpikeSortingV2.md:1042-1053`).

Similarly, recompute deletion has a dry-run/current-env/age-gated safe path, but
the main V2 guide only briefly names recompute tables and later discusses
fixture reruns (`docs/src/Features/SpikeSortingV2.md:87`,
`docs/src/Features/SpikeSortingV2.md:1124-1135`). Operators may miss the
canonical storage-management workflow and reach for ad hoc filesystem deletion.

Impact: the safe primitives exist, but users following the main tutorial may
not know when they are crossing a shared/prod DB boundary or how to reclaim
storage safely.

Recommended fix:

- Add a "Database boundary" block near setup and before write examples: confirm
  host, schema prefix, raw/analysis/temp roots, and the meaning of
  `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1`.
- Link the storage-management workflow from the main V2 page and notebook:
  `attempt_all`, `get_disk_space`, `delete_files(..., dry_run=True)`, inspect
  results, then explicit deletion.
- Add docs smoke checks that the env var and dry-run deletion workflow appear in
  the main V2 guide.

### 8. Low: one recompute cleanup path masks `rmtree` details

Most v2 cleanup paths propagate or wrap deletion failures. Analyzer reclamation
uses `shutil.rmtree(..., ignore_errors=True)` before checking whether the folder
still exists (`src/spyglass/spikesorting/v2/recompute.py:1311`). The follow-up
existence check helps prevent false success, but it loses the original
permission or filesystem exception detail.

Impact: an operator gets a less actionable error for one of the more dangerous
cleanup paths.

Recommended fix: use `ignore_errors=False`, catch and log the exception with the
folder path, and only mark `deleted=1` after confirmed removal. Add a simulated
`PermissionError` test.

## Suggested Repair Order

1. Add root-confinement validation for raw/analysis NWB path helpers and tests
   for traversal filenames.
2. Change v2 artifact permissions and sorter scratch modes to restricted
   defaults, with an explicit container/shared-storage opt-in path.
3. Decide the production trust model for sorter/container parameter rows; add
   allowlist/digest policy if ordinary users can insert rows.
4. Harden analyzer-cache helper validation and orphan-sweep name filtering.
5. Replace formatted file/path restrictions in shared analysis helpers.
6. Update docs for DB boundary, immutable container wording, and dry-run storage
   reclamation workflow.
