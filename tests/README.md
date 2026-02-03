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
pytest --no-teardown -v
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

The test suite uses session-scoped fixtures for efficiency. All fixtures are
defined in `tests/conftest.py`.

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

**Trigger Modes:**

- `push`: Runs fast tests only (`-m "not slow and not very_slow"`)
- `pull_request` (closed + merged): Runs full test suite
- `workflow_dispatch`: Manual trigger with mode selection (fast/full)

**Optimizations:**

- **Test data caching**: 231 MB cached (saves 1-2 min per run)
- **Conditional downloads**: Skip if cache hit
- **CodeCov integration**: Separate flags for fast/full test coverage

**Manual Dispatch:**

- Via GitHub UI: Actions → test-conda.yml → Run workflow
- Select: fast or full

______________________________________________________________________

## Environment

To facilitate headless testing of various Qt-based tools as well as Tensorflow,
`pyproject.toml` includes environment variables:

- `QT_QPA_PLATFORM`: Set to `offscreen` to prevent the need for a display
- `TF_ENABLE_ONEDNN_OPTS`: Set to `1` to enable Tensorflow optimizations
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
pytest --no-teardown  # Avoid container restart overhead
```

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
--base_dir PATH     # Where to store downloaded/created files
# Default: ./tests/_data/

--no-teardown       # Preserve Docker database on exit (default: False)
# Useful for: inspecting database state, faster reruns

--no-docker         # Don't launch Docker, connect to existing container
# Useful for: GitHub Actions, manual Docker management

--no-dlc            # Skip DeepLabCut tests and downloads
# Useful for: systems without DLC, faster test runs
```

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
