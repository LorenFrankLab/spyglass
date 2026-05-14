# Phase 0c — SpikeInterface 0.104 prerequisite

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#spikeinterface-099--0104-migration-cheat-sheet)

This is a required prerequisite PR before Phase 1. It is separated from Phase 0 because it changes the production v1 runtime dependency and must prove v1 still works before v2 starts landing runtime tables.

## Purpose

Move Spyglass from `spikeinterface>=0.99.1,<0.100` to `spikeinterface>=0.104,<0.105` without breaking v1. The hard blocker is v1's WaveformExtractor usage (`extract_waveforms` / `load_waveforms`), which was removed from modern SpikeInterface. The port changes v1 implementation internals only; it must not alter v1 schemas, v1 public table names, or v1 user workflows.

## Executor Checklist

- Port v1 WaveformExtractor calls to SortingAnalyzer-compatible internals while preserving v1 public methods.
- Update v1 metric/burst helpers or adapters so existing notebook-facing behavior still works.
- Bump the SI dependency and resolver-check Python 3.10, 3.11, and 3.12.
- Prove v1 schemas are byte-identical before/after the port.
- Run the v0/v1 validation slice; Phase 1 remains blocked until this passes.

## Inputs to read first

- [pyproject.toml](../../../../pyproject.toml) — current SpikeInterface pin.
- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) — WaveformExtractor usage and metric curation surface.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py) — depends on `MetricCuration.get_waveforms`.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet).

## Tasks

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
  - Add optional dependency group `spikesorting-v2-matching = ["UnitMatchPy>=3.3,<4"]` only after resolver testing confirms it does not force an incompatible NumPy downgrade in the supported Python range.

- **Run resolver checks.**
  - Verify Python 3.10, 3.11, and 3.12 environments resolve.
  - Record SpikeInterface, NumPy, Zarr, numcodecs, and mountainsort package versions in the PR description.

## Validation slice

| Test | Asserts |
| --- | --- |
| `pytest tests/spikesorting/v1/` | Existing v1 spike-sorting tests pass under SI 0.104. |
| `pytest tests/spikesorting/v0/ tests/spikesorting/v1/` | Legacy spike-sorting surfaces still import and run under the new pin. |
| `test_metric_curation_get_waveforms_compat` | `MetricCuration.get_waveforms(key)` still supports all v1 callers exercised by `MetricCuration` and `BurstPair`. |
| `test_burstpair_helpers_after_si0104_port` | `BurstPair` plotting / investigation helpers still execute against a metric-curation row. |
| `test_no_v1_schema_changes` | v1 DataJoint `definition` strings for recording, sorting, curation, metric curation, burst curation, and recompute tables are byte-identical before/after the port. |
| `test_pyproject_si_pin` | `pyproject.toml` requires `spikeinterface>=0.104,<0.105`. |

## Deliberately not in this phase

- No v2 production tables.
- No v2 `Recording`, `Sorting`, `CurationV2`, `AnalyzerCuration`, `SessionGroup`, `UnitMatch`, or FigPack implementation.
- No v1 feature redesign. This is an API-compatibility port plus dependency bump only.

## Review

Before opening Phase 1, reviewers must confirm this phase has landed and that v1 remains the production path under SI 0.104. A failure here blocks Phase 1; do not skip or fold this work into the first v2 runtime PR.
