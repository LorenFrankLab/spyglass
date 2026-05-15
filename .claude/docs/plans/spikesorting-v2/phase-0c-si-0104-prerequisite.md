# Phase 0c — SpikeInterface 0.104 prerequisite

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#spikeinterface-099--0104-migration-cheat-sheet)

This is a required prerequisite PR before Phase 1. It is separated from Phase 0 because it changes the production v1 runtime dependency and must prove v1 still works before v2 starts landing runtime tables.

## Purpose

Move Spyglass from `spikeinterface>=0.99.1,<0.100` to `spikeinterface>=0.104,<0.105` without breaking v1. The hard blocker is v1's WaveformExtractor usage (`extract_waveforms` / `load_waveforms`): modern SpikeInterface keeps these as back-compat shims over `SortingAnalyzer`, but v2 should implement against the native `SortingAnalyzer` API rather than the shim. The port changes v1 implementation internals only; it must not alter v1 schemas, v1 public table names, or v1 user workflows.

## Executor Checklist

- Work in a dedicated `uv` virtualenv; do not test the SI 0.104 resolver or UnitMatchPy extra in a shared/base environment.
- Port v1 WaveformExtractor calls to SortingAnalyzer-compatible internals while preserving v1 public methods.
- Update v1 metric/burst helpers or adapters so existing notebook-facing behavior still works.
- Bump the SI dependency and resolver-check Python 3.10, 3.11, and 3.12.
- Prove v1 schemas are byte-identical before/after the port.
- Run the v0/v1 validation slice; Phase 1 remains blocked until this passes.
- Record exact resolved package versions and sorter availability in the PR description.

## Inputs to read first

- [pyproject.toml](../../../../pyproject.toml) — current SpikeInterface pin.
- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) — WaveformExtractor usage and metric curation surface.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py) — depends on `MetricCuration.get_waveforms`.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet).

**Global invariants apply:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) and [Code Artifact Naming](shared-contracts.md#code-artifact-naming).

## Tasks

- **Create and use an isolated resolver/test environment.**
  - Use a dedicated `uv` virtualenv for the dependency bump and validation commands.
  - Capture `python --version`, `uv pip freeze`, `spikeinterface.__version__`, `numpy.__version__`, `zarr.__version__`, `numcodecs.__version__`, `spikeinterface.sorters.installed_sorters()`, and UnitMatchPy import status in the PR description or a small resolver artifact.
  - Do not run resolver probes from base/conda; a passing base-env import is not evidence that the project dependencies resolve.

- **Port v1 WaveformExtractor usage to SortingAnalyzer-compatible code.**
  - Replace `si.extract_waveforms(...)` with `si.create_sorting_analyzer(..., format="binary_folder", sparse=True, ...)`.
  - Replace `si.load_waveforms(...)` with `si.load_sorting_analyzer(...)`.
  - Preserve `MetricCuration.get_waveforms(...)` as a v1 public method name if notebooks or `BurstPair` still call it. Returning a SortingAnalyzer-backed compatibility object is acceptable, but the caller-facing behavior must remain covered by tests.
  - Keep v1 DataJoint schemas unchanged.

- **Audit v1 metric helper compatibility.**
  - Update `metric_utils.py` functions that assume a `WaveformExtractor` object, or provide a narrow adapter that supports the methods those helpers call.
  - Confirm `BurstPair` helpers still render after the port.

- **Bump dependencies in `pyproject.toml`.**
  - Change `spikeinterface>=0.99.1,<0.100` to `spikeinterface>=0.104,<0.105`.
  - Add `mountainsort5>=0.5`.
  - Verify the MS4 runtime package in the same resolver matrix. A clean SI 0.104.3 Python 3.12 env exposes the `mountainsort4` wrapper but does not install the runtime; `pip install mountainsort4` can fail while building `isosplit5`. Phase 1 cannot ship an MS4 default row unless this phase proves `spikeinterface.sorters.installed_sorters()` includes `mountainsort4` on supported CI/dev envs, or documents the blocker and removes MS4 from the executable default set.
  - Add optional dependency group `spikesorting-v2-matching = ["UnitMatchPy>=3.3,<4", "mat73"]` only after resolver testing confirms it does not force an incompatible NumPy downgrade in the supported Python range and the UnitMatchPy import path is usable or clearly guarded.

- **Run resolver checks.**
  - Verify Python 3.10, 3.11, and 3.12 environments resolve.
  - Record SpikeInterface, NumPy, Zarr, numcodecs, `mountainsort5`, MS4 runtime status, and `spikeinterface.sorters.installed_sorters()` output in the PR description.

## Validation slice

| Test | Asserts |
| --- | --- |
| `pytest tests/spikesorting/v1/` | Existing v1 spike-sorting tests pass under SI 0.104. |
| `pytest tests/spikesorting/v0/ tests/spikesorting/v1/` | Legacy spike-sorting surfaces still import and run under the new pin. |
| `test_metric_curation_get_waveforms_compat` | `MetricCuration.get_waveforms(key)` still supports all v1 callers exercised by `MetricCuration` and `BurstPair`. |
| `test_burstpair_helpers_after_si0104_port` | `BurstPair` plotting / investigation helpers still execute against a metric-curation row. |
| `test_no_v1_schema_changes` | v1 DataJoint `definition` strings for recording, sorting, curation, metric curation, burst curation, and recompute tables are byte-identical before/after the port. |
| `test_pyproject_si_pin` | `pyproject.toml` requires `spikeinterface>=0.104,<0.105`. |
| `test_sorter_runtime_resolution` | `mountainsort5` imports, `sis.installed_sorters()` includes `mountainsort5`, and MS4 runtime status is explicit: either `sis.installed_sorters()` includes `mountainsort4` or the PR blocks Phase 1's MS4 default row with a documented resolver issue. |
| `test_optional_matching_extra_resolution` | The `spikesorting-v2-matching` extra includes both `UnitMatchPy>=3.3,<4` and `mat73`, resolves without NumPy incompatibility, and import guards produce a clear message if UnitMatchPy hits the `_tkinter` import path. |

## Commands to run

```bash
uv venv .venv-spikesorting-v2-si0104
source .venv-spikesorting-v2-si0104/bin/activate
uv pip install -e ".[test]"
python --version

pytest tests/spikesorting/v0/ tests/spikesorting/v1/ -q
pytest tests/spikesorting/v1/test_metric_curation.py tests/spikesorting/v1/test_burst.py -q
uv pip check
python - <<'PY'
import spikeinterface as si
import spikeinterface.sorters as sis
import numpy as np
import zarr
import numcodecs
from packaging.version import Version
assert Version(si.__version__) >= Version("0.104")
installed = set(sis.installed_sorters())
assert "mountainsort5" in installed
print("python resolver env ok")
print("spikeinterface", si.__version__)
print("numpy", np.__version__)
print("zarr", zarr.__version__)
print("numcodecs", numcodecs.__version__)
print("installed_sorters", sorted(installed))
PY
mkdir -p tests/spikesorting/v2
uv pip freeze > tests/spikesorting/v2/si0104-freeze.txt
git diff --check -- pyproject.toml src/spyglass/spikesorting/v1 tests/spikesorting/v1
```

## Deliberately not in this phase

- No v2 production tables.
- No v2 `Recording`, `Sorting`, `CurationV2`, `AnalyzerCuration`, `SessionGroup`, `UnitMatch`, or FigPack implementation.
- No v1 feature redesign. This is an API-compatibility port plus dependency bump only.

## Review

Before opening Phase 1, reviewers must confirm this phase has landed and that v1 remains the production path under SI 0.104. A failure here blocks Phase 1; do not skip or fold this work into the first v2 runtime PR.
