# Spyglass Test Suite

[![codecov](https://codecov.io/gh/LorenFrankLab/spyglass/graph/badge.svg?token=QEJIIFN2S8)](https://codecov.io/gh/LorenFrankLab/spyglass)

______________________________________________________________________

## Quick Start

```bash
# Run all tests
pytest

# Run fast tests only
pytest -m "not slow and not very_slow"

# Run specific test file
pytest tests/position/v1/test_trodes.py

# Run with coverage report
pytest --cov=spyglass --cov-report term-missing

# Debug mode (preserve database, verbose output)
# --base-dir defaults to ./tests/_data/, so it can be omitted
pytest --no-teardown -v
```

### Spike sorting v2

Use the isolated helper tier for a dependable database-free development run:

```bash
pytest tests/spikesorting/v2 --v2-tier unit --no-cov \
  -m "not slow and not very_slow and not regenerate"
```

This command starts no Docker server, fetches no datasets, and rejects
DataJoint connections and queries during collection and execution. Small
synthetic recordings and temporary files are allowed. Drop the speed filter to
run all isolated helper tests. On macOS, add `-p no:xvfb` if pytest-xvfb is
installed; its virtual display requires Linux's Xvfb executable.

Every collected v2 test has exactly one effective tier:

| Tier | Dependency and behavior |
| --- | --- |
| `unit` | Isolated helpers; no database, downloaded datasets, browser, or native sorting run. |
| `db_unit` | Database metadata, lookups and schema checks without v2 stage computation. |
| `stage` | A materialized stage, native sorter, browser, or local server; database access when needed. |
| `pipeline` | The complete sorting chain or session orchestration, including fixtures that already built it. |
| `regression_gate` | Heavy scientific fixtures, memory, lifecycle acceptance and regeneration gates. |

Run another tier with `--v2-tier stage`, for example, or every tier except
`unit` with `--v2-tier database`. These dependency tiers are independent of
speed markers; database tiers use the normal Docker/database setup. The CI
workload shard named `unit` includes light database work to balance runtime. It
is **not** the isolated `unit` tier.

On push and pull request, every CI lane (the three workload shards and the
matching and browser extras) runs only the `unit` and `db_unit` tiers plus the
tests marked `smoke`, one end-to-end check per main path. The nightly schedule
and a manual "Run workflow" dispatch run every tier; scheduled runs use the
default branch, so dispatch one before merging a branch that changes v2. To run
the pull-request selection locally:

```bash
pytest tests/spikesorting/v2 --v2-pr --v2-tier-budgets
```

To run the whole v2 suite in parallel locally, use pytest-xdist (the `test`
extra) with `--dist loadfile`, so each module's fixtures are built once:

```bash
pytest tests/spikesorting/v2 -n 8 --dist loadfile \
  --base-dir /local/disk/tests/_data
```

Each worker starts its own Docker MySQL server (`<container-name>-gw<N>`, on
its own port) and uses its own data directory (`<base-dir>/xdist/gw<N>`);
the raw session and videos are linked from `<base-dir>` instead of fetched
again. The controller caps each worker's BLAS, torch and numba threads at
cores / workers. A `--base-dir` on local disk avoids NFS latency; it must
contain a `tests` path component. `--no-docker` cannot give workers separate
servers, so with xdist it is refused except for `--v2-tier unit`. On a 96-core
machine, 8 workers ran the full suite in 36 min (4.5 h serially); the longest
module, `test_curation_evaluation.py`, bounds the wall time.

`--v2-tier-budgets` fails the run when a test exceeds its tier's time budget
(`TIER_BUDGETS_S` in `tests/spikesorting/v2/_test_tiers.py`); move such a test
to a heavier tier in `test_tiers.json` or make its setup cheaper. With
`pytest-timeout` installed (the `test` extra), every v2 test also carries its
tier's timeout (`TIER_TIMEOUTS_S`), so a hang fails instead of stalling the run.

Architecture checks require canonical owner imports and docstring-only private
package initializers. Recording dictionary and pickle round trips must preserve
independently expected traces and timestamps; SpikeInterface and DataJoint
version adapters have their own tests. See the
[architecture map](../docs/src/ForDevelopers/SpikeSortingV2Architecture.md).

Integration scenarios declare their expected outcomes before calling the public
APIs. Unit identities, accepted/excluded populations, original-clock spike
times, and match memberships come from the fixture specification and raw input
artifacts. Comparing two production readers is an additional persistence check;
it cannot establish correctness on its own. A failing scenario is investigated
against that independent truth, rather than changing expectations to match
current output.

The user scenarios are covered through the public APIs and persisted outputs:

| User scenario                                                                        | Integration coverage                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Curate an oversplit bursting neuron alongside MUA and artifacts                      | [User curation workflow](spikesorting/v2/test_user_curation_workflow.py): waveform-bearing physical sources, known refractory violations, manual/browser merges, postmerge verification, and exact accepted spike data.                                          |
| Merge units during review and resume verification                                    | [Browser journey](spikesorting/v2/test_review_browser_journey.py): real Chromium edits, save, commit, merged-child verification, restart/resume, and analysis handoff.                                                                                           |
| Match epochs with changing firing rates and missing/newly detected neurons           | [Epoch matching](spikesorting/v2/test_epoch_unitmatch.py): one stable neuron with different sorter IDs, epoch-specific singletons, reviewed nuisance exclusion, explicit automatic-only populations, original-clock readback, and reuse.                         |
| Sort same-day recordings together with one shared unit namespace                     | [Concat pipeline](spikesorting/v2/test_session_group_concat.py), `test_run_v2_pipeline_concat_mode_routes_session_group`: public run, auto-curation, per-member analysis outputs, and reuse.                                                                     |
| Match independent recordings or daily concatenations                                 | [Direct curation matching](spikesorting/v2/test_direct_curation_unitmatch.py), `test_direct_reviewed_curation_populations_reach_match_artifact`: exact reviewed generations, evaluation-pinned populations, known match memberships, and database/NWB agreement. |
| Repair a blocked matching request                                                    | [Direct curation matching](spikesorting/v2/test_direct_curation_unitmatch.py): blocks missing artifacts and changed pins before writes; restoring an artifact makes a rebuilt plan ready.                                                                        |
| Track neurons across days, including partial activity and disappearance/reappearance | [Native daily matching](spikesorting/v2/test_unitmatch_daily_concat.py): real waveform extraction and UnitMatch, independently planted identities, unrelated sorter IDs, distractors, and held-out recovery/precision gates.                                     |

Run the focused pipeline scenarios in an environment with the `test`,
`spikesorting-v2`, and `spikesorting-v2-curation` extras:

```bash
pytest \
    tests/spikesorting/v2/test_user_curation_workflow.py \
    tests/spikesorting/v2/test_epoch_unitmatch.py \
    tests/spikesorting/v2/test_session_group_concat.py::test_run_v2_pipeline_concat_mode_routes_session_group \
    tests/spikesorting/v2/test_direct_curation_unitmatch.py::test_direct_reviewed_curation_populations_reach_match_artifact \
    --v2-tier pipeline --no-dlc --no-cov
```

The focused review/matching journeys use deterministic sorter and matcher
fixtures while running the real orchestration, review commits, waveform
preparation, database writes, NWB serialization, and downstream readers. Native
sorter tests in `single_session/test_pipeline_run.py`, native UnitMatch recovery
tests in `test_unitmatch_daily_concat.py`, and the browser journeys
independently exercise those runtimes. The browser journeys also require
`spikesorting-v2-curation-test` and Playwright Chromium. CI runs the public
review journeys in its curation lane and native matching in its matching lane;
optional review paths can skip in the base environment. Heavy corrected
daily-matching acceptance runs belong to `regression_gate` and run on the
nightly/manual lane.

The small user fixtures declare irregular physical spike trains before assigning
sorter clusters. An oversplit cell shares one waveform across its fragments;
pooled MUA combines two separately refractory-safe neurons. Bursting and a high
firing rate alone do not make a unit MUA. Raw waveforms, channel footprints,
background noise, and broad transients give evaluation and review meaningful
signals. Expected merge events, refractory fractions, matched identities, and
unmatched units are fixed independently of pipeline outputs.

The native chronic benchmarks also include gradual template changes and nearby
neurons with similar templates. Recovery for those two difficult cases is
reported as diagnostics, without a passing recall requirement. Their presence in
the fixture does not establish reliable recovery under drift or ambiguity.

The motion acceptance manifests pin the benchmark runner and helper files. A
pin's `git_commit` identifies the checkout base; `files` identifies exact
working-tree bytes, including uncommitted edits. Matching hashes are not
evidence that historical gates pass the current harness.

The unpinned `_motion_acceptance_reuse.py` wrapper also requires a complete
runtime receipt and matching fingerprint before reusing results. Python,
dependency/code versions, platform, BLAS/thread settings and the explicit
benchmark job/recipe settings must match. Existing results without these
receipts must be rerun. Failed runs never attach new receipts to old metrics,
and runtime/source drift during execution leaves results unstamped.

Reviewed module defaults, function overrides and fixture minimums live in
[`spikesorting/v2/test_tiers.json`](spikesorting/v2/test_tiers.json). Collection
checks assignments against pytest's resolved fixture graph, including fixture
overrides. Adding a new unclassified module or a database-dependent fixture
fails collection. Add its assignment, or declare an explicit tier on a new
module/test. A test added to an existing module inherits its reviewed default;
if it introduces a higher-tier fixture, collection fails until its tier is
updated. Named overrides use `ClassName::test_name` for class methods and omit
parameter IDs. Use an override when lowering a module default for a pure test.

Lint rules live in `pyproject.toml` and are shared by local pre-commit and CI.
The type-checking gate covers pipeline value types, recording value types and
the matcher protocol:

```bash
python -m pip install ruff==0.14.13 mypy==2.3.1
ruff check .
mypy --config-file=pyproject.toml
```

______________________________________________________________________

## Test Markers

The test suite uses pytest markers to categorize tests by speed and type.

### Speed Markers

Use these to run subsets of tests based on execution time:

```bash
# Fast tests only (excludes slow and very_slow)
pytest -m "not slow and not very_slow"

# Include slow tests but exclude very_slow
pytest -m "not very_slow"

# Run only slow tests
pytest -m "slow"
```

**Marker Definitions** (see `pyproject.toml`):

- **`slow`**: Tests taking 10-60 seconds
- **`very_slow`**: Tests taking >60 seconds
- Unmarked tests: \<10 seconds

### Type Markers

Tests are also categorized by functionality:

- **`unit`**: Unit tests (fast, isolated)
- **`integration`**: Integration tests (moderate speed)
- **`spikesorting`**: Spike sorting pipeline tests
- **`position`**: Position tracking tests
- **`lfp`**: LFP analysis tests
- **`decoding`**: Decoding algorithm tests
- **`common`**: Common table tests

### Module Markers

Target specific modules:

```bash
# Run only DLC tests
pytest -m dlc

# Run only linearization tests
pytest -m linearization
```

**Available module markers**: `dlc`, `trodes`, `linearization`, `ripple`, `mua`,
`clusterless`, `sorted_spikes`

______________________________________________________________________

## Fixture System

The test suite uses session-scoped fixtures for expensive shared setup and
function-scoped fixtures for mutable state. Repository fixtures live in
`tests/conftest.py`; domain and suite fixtures live in their nearest
`conftest.py`, including `tests/spikesorting/v2/conftest.py` and
`tests/spikesorting/v2/single_session/conftest.py`.

### Core Data Fixtures

**Full Test File** (229.6 MB):

- `mini_path`: Path to minirec20230622.nwb
- `mini_content`: NWB file content (opened)
- `mini_copy_name`: Copy filename after processing
- `mini_dict`: Dictionary key for database queries
- `mini_restr`: Restriction string for database queries

### Database Fixtures

- `dj_conn`: DataJoint connection (with worker isolation for parallel tests)
- `server`: Docker MySQL server instance
- `teardown`: Controls whether database is preserved on exit

### Module Fixtures

- `common`: spyglass.common module
- `sgp`: spyglass.position module
- `lfp`: spyglass.lfp module
- `sgl`: spyglass.linearization module

### Shared Fixture Notes

- `imported_merge_id` (`tests/decoding/conftest.py`) appends units to a copy of
    the shared minirec `_` file and swaps it in with `os.replace`. Under the
    full suite, LFP `*_raw` fixtures hold their analysis files open read-only
    inside `NWBHDF5IO`, outside Spyglass's handle cache, and those files
    external-link the raw ephys. `close_nwb_files` cannot release them, and HDF5
    refuses `mode="a"` on a file open `"r"` in the same process. `os.replace` is
    atomic on one filesystem; readers keep the old inode.
- `fetch_spike_data` (`spikesorting/analysis/v1/group.py`) skips label filtering
    in test mode. The shared `default_exclusion` groups exclude noise/mua over
    curations carrying those labels, so filtering would empty them and break
    `test_fetch_data` and sorted-spikes decoding. `test_filter_units` covers
    filtering directly. The branch for curations without a label column is not
    gated: an exclude filter over empty labels keeps every unit.

______________________________________________________________________

## Mocking Infrastructure

Critical tests use mocking to avoid expensive external operations while
maintaining coverage.

### Mocked Operations

**Location**: `tests/decoding/conftest.py` and
`tests/spikesorting/*/conftest.py`

1. **Decoder Results I/O** (90% speedup)

    - Mocks: `ClusterlessDetector.save_results()`,
        `ClusterlessDetector.load_results()`
    - Uses: In-memory netCDF4 files instead of disk I/O

2. **Spike Sorting External Calls**

    - Mocks: Spikeinterface operations, detector computations
    - V2 tests isolate external boundaries explicitly; pipeline, persistence,
        transaction and browser tests exercise real operations where relevant.

### Using Mocked Fixtures

Mocked fixtures are automatically applied globally via
`@pytest.fixture(scope="session", autouse=True)`:

```python
# In tests/decoding/conftest.py
@pytest.fixture(scope="session", autouse=True)
def mock_detector_io_globally():
    """Globally mock detector I/O operations."""
    # Mocking implementation...
```

______________________________________________________________________

## CI/CD Pipeline

### GitHub Actions Workflow

**File**: `.github/workflows/test-conda.yml`

**Trigger Modes** (the general `run-tests` job):

- `push`: Runs fast tests only (`-m "not slow and not very_slow"`)
- `pull_request` (closed + merged): Runs full test suite
- `workflow_dispatch`: Manual trigger with mode selection (fast/full)

The `fast`/`full` selection applies only to `run-tests`. The spike sorting jobs
select by event instead; see [Spike sorting lanes](#spike-sorting-lanes).

**Lanes:** each job selects its tests with `--lane NAME`, defined in
`tests/_ci_lanes.py`, instead of listing paths:

| Lane | Job | Tests |
| --- | --- | --- |
| `core` | `run-tests` | `tests/` except `spikesorting/v0`, `v1` and `v2` |
| `legacy` | `pytest-legacy` | `spikesorting/v0`, `v1` and the v2 legacy-guard file |
| `v2-single-session`, `v2-integration`, `v2-unit` | `pytest-v2` workload shards | Every v2 test file once |
| `v2-matching`, `v2-curation` | `pytest-v2` extras lanes | Tests that need the matching or curation extras |
| `v2-supported-python` | `v2-supported-python` | DB-free contract tests, run with `--noconftest -p tests._ci_lanes` |

Run a job's selection locally with, for example,
`pytest --lane v2-unit --v2-tier unit`. Pass a value-taking option as
`--opt=value` (such as `--base-dir=/local/disk/tests/_data`) when no test path
is given; pytest otherwise reads the value as a test path and does not load
`tests/conftest.py`.

**Optimizations:**

- **Test data caching**: 231 MB cached (saves 1-2 min per run)
- **Conditional downloads**: Skip if cache hit
- **CodeCov integration**: one report. Every test job uploads its raw coverage
    data (`COVERAGE_FILE`) as an artifact; the `coverage` job combines them
    with `coverage combine` and uploads one XML, flagged `fast-tests` or
    `full-tests` by the `run-tests` mode. `[paths]` in `tests/.coveragerc` maps
    the site-packages install of `run-tests` onto `src/`.

**Manual Dispatch:**

- Via GitHub UI: Actions → test-conda.yml → Run workflow
- Select: fast or full

### Spike sorting lanes

`pytest-v2` is one matrix job (`fail-fast: false`, so one lane's failure does
not cancel the others). Tier selection by event is described under
[Spike sorting v2](#spike-sorting-v2). Each lane uploads JUnit XML with per-test
durations.

**Workload shards** (`single-session`, `integration`, `unit`)

- Balanced by runtime, not test count. The `v2-single-session` lane holds
    `single_session/`, `v2-integration` holds `INTEGRATION_MODULES` (in
    `tests/_ci_lanes.py`), and `v2-unit` every other file. Add a new full-sort
    test file to `INTEGRATION_MODULES`.
- Each shard runs `--v2-tier unit` on xdist workers, then `--v2-tier database`
    serially. Push and PR add `--v2-pr`.
- Env: `environments/environment_spikesorting_v2.yml` (Python 3.11, extras
    `test`, `spikesorting-v2`, `spikesorting-v2-validation`). NEURON and LFPy
    are left out: they only generate fixtures, which CI downloads.
- The verify step runs `pip check` and a real MountainSort5 sort, so a broken
    sorter runtime fails instead of skipping.

**Fixtures** (all v2 lanes)

- `fixtures/_fetch.py` downloads from Box, checks sha256 against
    `fixtures_manifest.json` and retries transient errors. A hash mismatch or an
    HTTP 4xx fails the job.
- No regeneration fallback: it would hide a corrupt or rotated artifact behind a
    green run and pull NEURON/LFPy into CI. Regenerate by hand with
    `generate_mearec.py` (see `fixtures/README.md`).
- The cache key covers the manifest, `_fetch.py` and the event, so per-PR and
    nightly/manual caches do not overwrite each other.
- Per event: push and PR use `mearec_polymer_smoke`; nightly adds
    `mearec_polymer_128ch_60s` and the real session `minirec20230622`; manual
    dispatch adds `mearec_neuropixels_60s` and `mearec_tetrode_60s`.
- `SPYGLASS_V2_REQUIRE_FIXTURES` lists the event's fixtures and
    `pytest_sessionstart` fails if one is missing, so a green run means their
    tests ran. Elsewhere, a test whose fixture is absent skips.
- `minirec20230622` has tetrode contacts in the x-z plane, which an x-y
    projection collapses. It drives the geometry round trip in
    `single_session/test_recording.py`; per-PR runs cover the repair on a
    synthesized all-zero tetrode.

**`matching`**

- Own conda env, so the `spikesorting-v2-matching` extra's weight stays out of
    the default env. It is the only env with UnitMatchPy, so it owns the matcher
    tests and the cross-session notebook.
- `test_optional_matching_extra_resolution` checks that the extra resolves with
    numpy >= 2.
- `SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED=1` turns the missing-extra skip into
    a failure, so the install may tolerate transient errors (`|| true`) while a
    broken extra still fails.
- `test_v2_unitmatch_polymer_mearec_ground_truth` needs
    `mearec_polymer_128ch_2sessions_{s1,s2}`, which have no URL in `_fetch.py`,
    so CI skips it; run it locally with the fixtures.

**`curation`**

- Own env, so the FigPack pins do not couple to the default env. Only lane that
    runs the FigPack table layer and the Playwright browser tests.
- The install has no `|| true`, and `SPYGLASS_REQUIRE_BROWSER_TESTS=1` fails
    browser tests on a missing figpack, Playwright or browser, so they cannot
    skip green.
- Nightly/manual: `tests/spikesorting/v2/acceptance` with `--run-acceptance`
    sorts `mearec_tetrode_60s` end to end. Without the flag these tests are
    collected as skipped. `test_database_and_file_restore` skips in CI: with
    `--no-docker` there is no Docker client for its restore.

**`motion-benchmark`** (manual dispatch only)

- `SPYGLASS_V2_MOTION_BENCHMARK=1` runs `test_motion_acceptance.py` on the
    development manifest (seeds 0-2, 117 cases, each in its own process; about
    50 min) plus the drift fixture.
- Publishes the `motion-benchmark-development` artifact (motion error, sorting
    accuracy, runtime, memory). It is development evidence, not validation: the
    manifest has no gates, so `test_benchmark_meets_manifest_gates` skips and
    only a case that fails to run fails.
- The held-out manifest is not run. A new held-out evaluation needs a new
    manifest with unused seeds, committed before it runs.

**`v2-supported-python`**

- Python 3.10 and 3.12 install the `spikesorting-v2` extra and run DB-free
    contract tests and a sorter smoke script. The full suite uses 3.11.

**`pytest-legacy`**

- Runs on Sunday's scheduled run, on manual dispatch, and on pull requests
    whose files match the `legacy-changes` paths filter (the v0/v1, analysis
    and shared spike sorting modules, the legacy env, `pyproject.toml`, this
    workflow, the lanes file and the legacy tests).
- v0/v1 compute paths call `_require_legacy_si_environment`, which raises under
    SI 0.101 and later, so the `core` lane leaves out `tests/spikesorting/v0`
    and `v1`; this job runs them on SI 0.99.
- A `sed` step relaxes the SI and probeinterface pins in its checkout so the env
    resolves SI 0.99 in one pass. Installing 0.104 and then downgrading leaves
    an incompatible numpy 2 stack.
- The verify step asserts SI < 0.100 (0.100 breaks `NwbRecordingExtractor`
    timestamps), probeinterface < 0.3 and numpy < 2.
- Also runs `tests/spikesorting/v2/test_clusterless_waveform_features.py` for
    its legacy-guard test. Its v2 imports are lazy, and the `legacy` lane
    collects no other SI 0.104 module.

______________________________________________________________________

## Environment

To facilitate headless testing of various Qt-based tools as well as Tensorflow,
`pyproject.toml` includes environment variables:

- `QT_QPA_PLATFORM`: Set to `offscreen` to prevent the need for a display
- `TF_ENABLE_ONEDNN_OPTS`: Set to `0` to disable approximate Tensorflow operations
- `TF_CPP_MIN_LOG_LEVEL`: Set to `2` to suppress Tensorflow warnings

______________________________________________________________________

## Performance Optimization Tips

### For Local Development

**Use fast tests during development:**

```bash
pytest -m "not slow and not very_slow"
```

**Preserve database between runs:**

```bash
pytest --no-teardown
```

Persists state across runs in the default `./tests/_data/` base.

**Run specific test files:**

- Only test what you're working on

```bash
pytest tests/position/v1/test_trodes.py
```

**Combine for maximum speed:**

```bash
pytest tests/common/test_session.py --no-teardown --quiet-spy
```

### For CI/CD

- Fast tests run automatically on every push
- Full test suite runs on PR merge
- Test data is cached
- Use `workflow_dispatch` to manually trigger fast or full suite

### Fixture Optimization Tips

1. **Leverage session scope**: Fixtures run once per session, not per test
2. **Mock expensive operations**: Mocking saves 9.2 min per run
3. **Request only needed fixtures**: Don't import fixtures you don't use

______________________________________________________________________

## Command-Line Options

All tests run with default parameters from `pyproject.toml`. To customize:

### Coverage Options

```bash
--cov=spyglass              # Enable coverage for spyglass package
--cov-report term-missing   # Show lines missing from coverage
--cov-report html           # Generate HTML coverage report
```

### Verbosity Options

```bash
-v                  # Verbose: list individual tests, report pass/fail
-vv                 # Extra verbose: show test details
--quiet-spy         # Silence Spyglass logging (default: False)
-s                  # No capture: show print statements
```

### Data and Database Options

```bash
--base-dir PATH     # Where to store downloaded/created files
# Default: ./tests/_data/. SPYGLASS_BASE_DIR in the environment is ignored
# by the test suite — a shell-exported value pointing at shared/production
# storage would let destructive tests (e.g. AnalysisNwbfile.cleanup) scan
# and delete real data. Pass --base-dir explicitly to override the default.
#
# The resolved --base-dir must contain a 'tests' path component; this is
# checked in conftest before anything is created or downloaded, and again
# in settings.py under test_mode, which additionally requires every
# resolved Spyglass directory to sit inside the base dir.
#
# Generated data (analysis/*.nwb plus the export, moseq, recording,
# spikesorting, and tmp trees) is removed on teardown ONLY when --base-dir
# is the repository's own tests/_data. Any other base is left untouched,
# since a 'tests' component proves location, not ownership. Nested
# analysis/<session>/*.nwb files are NOT removed and accumulate across
# runs; if that backlog grows large enough to trip the cleanup deletion
# limits, clear it with `rm -rf tests/_data/analysis`.
#
# Concurrent pytest runs (--container-name/--container-port) should use
# distinct --base-dir values. This is not currently enforced: two runs
# sharing a base can race during teardown or other filesystem mutations.

--no-teardown       # Preserve Docker database on exit (default: False)
# Useful for: inspecting database state, faster reruns.

--no-docker         # Don't launch Docker, connect to existing container
# Useful for: GitHub Actions, manual Docker management

--no-dlc            # Skip DeepLabCut tests and downloads
# Useful for: systems without DLC, faster test runs

--container-name NAME  # Docker container name (default: branch-derived)
--container-port PORT  # Host port mapped to MySQL's 3306

--container-vol-dir PATH  # Host dir for the container's MySQL data
# Useful for: keeping a populated test database off a small root disk
```

#### Container Data Directory

By default Docker stores the test database on its own root filesystem, which on
a shared machine is often the smallest disk available. A populated test database
is not small, so `--container-vol-dir` bind-mounts the container's
`/var/lib/mysql` onto a directory you choose:

```bash
pytest --container-vol-dir /path/to/roomy-disk/docker-vols/
```

Each container gets its own subdirectory, `<vol-dir>/<container-name>`, so
concurrent runs on different branches do not share data. The directory is
created if it does not exist. Without the flag, storage is left to Docker
exactly as before.

Unless `--no-teardown` is passed, this directory is deleted along with the
container at the end of the run, so a later run that reuses the same
`--container-name` always starts from an empty data dir rather than rebooting
MySQL onto stale, partially-cleaned-up data. With `--no-teardown`, both the
container and its data dir are left in place. Note that `--container-vol-dir`
only takes effect when the container is (re)created — if a container with the
matching name is already running (e.g. from a prior `--no-teardown` run), the
flag has no effect on it; a warning is logged in that case.

### Debugging Options

```bash
-s                  # No capture: enables IPython.embed() in tests
--pdb               # Drop into debugger on test failure
--sw                # Stepwise: resume from last failed test
--lf                # Last failed: rerun only tests that failed last time
--ff                # Failed first: run failed tests first, then others

-k PATTERN          # Run tests matching pattern
# Example: pytest -k "test_session or test_nwb"

tests/path/file.py  # Run specific test file
```

### Common Combinations

```bash
# Debug a specific test
pytest tests/common/test_session.py::test_insert -s --pdb

# Fast development cycle
pytest -m "not slow" --no-teardown --quiet-spy -v

# Full coverage analysis
pytest --cov=spyglass --cov-report html --cov-report term-missing

# Rerun failures with debug
pytest --lf --pdb -v
```

______________________________________________________________________

## Pytest Configuration

Key settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
  "--cov=spyglass",
  "--cov-report=term-missing",
  "--cov-report=xml",
]

markers = [
  "slow: marks tests as slow",
  "very_slow: marks tests as very slow",
  # ... (see file for full list)
]
```

______________________________________________________________________

## Writing Tests

### Marking Tests Appropriately

```python
import pytest


@pytest.mark.slow
def test_long_running_operation():
    """This test takes 30 seconds."""
    pass


@pytest.mark.very_slow
def test_extremely_long_operation():
    """This test takes 2 minutes."""
    pass


@pytest.mark.dlc
def test_dlc_feature():
    """DLC-specific test."""
    pass
```

### Using Mocked Fixtures

Mocking is automatic—no changes needed to your tests. The mocked fixtures are
globally applied:

```python
# This automatically uses mocked I/O
def test_decoder_pipeline(sgp, decoder_selection):
    sgp.v1.ClusterlessDecodingV1().populate(decoder_selection)
    # save_results() and load_results() are mocked automatically
```

______________________________________________________________________

## Troubleshooting

### Database Connection Issues

```bash
# Check if Docker container is running
docker ps | grep spyglass

# Manually start container
pytest --no-teardown  # Run once to start container
# Container persists for subsequent runs
```

Note that the container process will try to use the branch name as the database
suffix. If your branch name has special characters, consider renaming.

### Fixture Teardown and `--no-teardown`

`--no-teardown` exists to skip **expensive** rebuilds - the Docker container and
the ingestion of the shared test NWB file - so that repeated runs are fast. It
is not a license to leave state behind.

A fixture whose leftover rows or files would break a *subsequent* run must clean
up unconditionally, without consulting the `teardown` fixture:

```python
@pytest.fixture
def some_fixture(raw_dir, common):
    ...
    yield file_name

    # Not guarded by `teardown`: this entry would give populate() phantom
    # work on the next run, and the file is cheap to rebuild.
    (common.SomeTable & {"some_field": file_name}).delete(safemode=False)
```

The common case is a fixture that registers a new `Nwbfile` entry. Its Session
has no rows in the ingestion tables, so any later `populate()` call finds work
to do for it and fails. If a fixture cannot be written to survive
`--no-teardown`, it should tear itself down instead.

______________________________________________________________________

### Slow Test Runs

```bash
# 1. Use fast tests during development
pytest -m "not slow and not very_slow"

# 2. Check if database is being recreated
pytest --no-teardown  # Preserve database between runs

# 3. Check test data cache
ls -lh tests/_data/  # Should see cached files
```

### Import Errors

```bash
# Ensure you're in the correct conda environment
conda activate spyglass

# Verify spyglass is installed and editable
python -c "import spyglass; print(spyglass.__version__)"
```

### Coverage Issues

```bash
# Generate detailed HTML report
pytest --cov=spyglass --cov-report html
# Open htmlcov/index.html in browser
```

______________________________________________________________________

## Additional Resources

- **CI/CD Workflow**: `.github/workflows/test-conda.yml`
- **Pytest Configuration**: `pyproject.toml`
- **Fixture Definitions**: `tests/conftest.py`
- **Mocking Implementation**: `tests/decoding/conftest.py`,
    `tests/spikesorting/*/conftest.py`
- **Coverage Report**:
    [![codecov](https://codecov.io/gh/LorenFrankLab/spyglass/graph/badge.svg?token=QEJIIFN2S8)](https://codecov.io/gh/LorenFrankLab/spyglass)
