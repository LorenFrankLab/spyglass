# Phase 0c — SpikeInterface 0.104 compatibility boundary

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#spikeinterface-099--0104-migration-cheat-sheet)

This is a required prerequisite checkpoint before Phase 1. It changes the Spyglass
SpikeInterface dependency boundary and must make the v1/v2 runtime split
explicit before v2 runtime tables start landing.

## Purpose

Move v2 development to `spikeinterface>=0.104,<0.105` without pretending that
the rest of Spyglass is automatically compatible. Current v0/v1 code uses
WaveformExtractor-era APIs (`extract_waveforms`, `load_waveforms`,
`WaveformExtractor`) in metric curation and burst workflows, and several
non-spike-sorting modules import SpikeInterface directly. If the global
Spyglass pin moves to SI 0.104, those active legacy runtime paths may fail or
change behavior.

Policy for this plan:

- v2 is the supported spike-sorting runtime path under the SI 0.104 environment.
- v0/v1 source stays in-tree, and existing v0/v1 rows remain queryable through
  the existing merge paths where possible.
- Active v0/v1 population, waveform extraction, MetricCuration, BurstPair,
  FigURL curation, and recompute workflows require the legacy SI 0.99 Spyglass
  environment unless Phase 0c proves a narrow compatibility shim is safe.
- Phase 0c audits every Spyglass `spikeinterface` import before the global pin
  changes, not only `src/spyglass/spikesorting/`.
- Phase 0c may port a legacy path only if the audit shows the port is small,
  schema-neutral, and behavior-preserving. Otherwise it documents the legacy
  environment boundary and adds clear runtime guards.

## Executor Checklist

- Work in a dedicated `uv` virtualenv; do not test the SI 0.104 resolver or optional extras in a shared/base environment.
- Audit all SpikeInterface usage in Spyglass and classify each affected surface as query-compatible, guarded legacy-runtime-only, safe-to-port, or v2-only.
- Add clear runtime guards / errors for legacy runtime paths that are not supported under SI 0.104.
- Update docs so users know v2 is the supported runtime path under new Spyglass/SI 0.104, while active v0/v1 processing requires a legacy SI 0.99 environment unless explicitly ported.
- Bump the SI dependency for the v2 runtime environment and resolver-check Python 3.10, 3.11, and 3.12.
- Prove legacy DataJoint schemas are unchanged.
- Run query/import smoke tests for v0/v1 under SI 0.104; do not require active legacy populate/metric workflows to pass unless this implementation explicitly ports them.
- Record exact resolved package versions, sorter availability, and legacy-runtime boundary decisions in a checked-in resolver artifact under `tests/spikesorting/v2/resolver/`.

## Inputs to read first

- [pyproject.toml](../../../../pyproject.toml) — current SpikeInterface pin.
- [src/spyglass/spikesorting/v0/](../../../../src/spyglass/spikesorting/v0/) — legacy WaveformExtractor usage.
- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) — WaveformExtractor usage and metric curation surface.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py) — depends on `MetricCuration.get_waveforms`.
- [src/spyglass/utils/waveforms.py](../../../../src/spyglass/utils/waveforms.py) — shared waveform helper surface that may still use WaveformExtractor-era APIs.
- [src/spyglass/utils/mixins/analysis.py](../../../../src/spyglass/utils/mixins/analysis.py), [src/spyglass/utils/mixins/analysis_builder.py](../../../../src/spyglass/utils/mixins/analysis_builder.py), and [src/spyglass/common/common_nwbfile.py](../../../../src/spyglass/common/common_nwbfile.py) — common analysis/NWB builder paths that import or wrap SpikeInterface objects.
- [src/spyglass/decoding/v0/clusterless.py](../../../../src/spyglass/decoding/v0/clusterless.py) and [src/spyglass/decoding/v1/waveform_features.py](../../../../src/spyglass/decoding/v1/waveform_features.py) — downstream decoding consumers affected by any waveform or sorting object API drift.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet).

**Global invariants apply:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) and [Code Artifact Naming](shared-contracts.md#code-artifact-naming).

## Tasks

- **Create and use an isolated resolver/test environment.**
  - Use a dedicated `uv` virtualenv for the dependency bump and validation commands.
  - Capture `python --version`, `uv pip freeze`, `spikeinterface.__version__`, `numpy.__version__`, `zarr.__version__`, `numcodecs.__version__`, `spikeinterface.sorters.installed_sorters()`, and optional extra import status in a checked-in resolver artifact.
  - Do not run resolver probes from base/conda; a passing base-env import is not evidence that the project dependencies resolve.

- **Audit global SpikeInterface runtime compatibility.**
  - Inventory every `import spikeinterface` / `from spikeinterface...` in `src/spyglass`, including v0/v1 spike sorting, shared waveform utilities, analysis mixins/builders, `common_nwbfile`, and decoding modules.
  - Explicitly inspect at minimum:
    - `src/spyglass/utils/waveforms.py`
    - `src/spyglass/utils/mixins/analysis.py`
    - `src/spyglass/utils/mixins/analysis_builder.py`
    - `src/spyglass/common/common_nwbfile.py`
    - `src/spyglass/decoding/v0/clusterless.py`
    - `src/spyglass/decoding/v1/waveform_features.py`
  - Inventory all references to `extract_waveforms`, `load_waveforms`, `WaveformExtractor`, legacy metric helpers, sorter APIs, recording/sorting extractor constructors, and SpikeInterface objects serialized through NWB helpers.
  - For each surface, decide:
    - **query-compatible** — existing DB rows / merge outputs can still be read under SI 0.104;
    - **guarded legacy-runtime-only** — active population/recompute/curation requires the SI 0.99 legacy environment;
    - **safe-to-port** — a small adapter preserves behavior without schema changes.
    - **v2-only** — new APIs used only by `spyglass.spikesorting.v2`.
  - Store the audit summary in docs or the PR description; update this plan only if the decision changes v2 scope.

- **Verify motion-correction API before schema freeze.**
  - In the same SI 0.104 resolver environment, inspect and record the signature and return behavior of `spikeinterface.preprocessing.correct_motion`.
  - Confirm which kwargs are accepted for the pinned 0.104 patch range and which kwargs change the return type or write side artifacts (`folder`, `output_motion`, `output_motion_info` or their current equivalents).
  - Record the deliberate MVP decision from `overview.md`: v2 Phase 3 persists only the corrected `ElectricalSeries`, sample boundaries, and hash; motion estimates / motion-info side artifacts are not queryable unless a future opt-in side table is added.
  - If the pinned SI API cannot support "corrected recording only" without untracked side artifacts, stop before Phase 1 freezes `MotionCorrectionParameters` and `ConcatenatedRecording`.

- **Add runtime guards for unsupported legacy active workflows.**
  - Guard v0/v1 active runtime paths that depend on WaveformExtractor-era APIs if they are not ported in this implementation.
  - Error messages must be explicit: "This v1/v0 spike-sorting runtime path requires the legacy SpikeInterface 0.99 environment. Use v2 for new SI 0.104+ processing, or run this workflow in the legacy Spyglass environment."
  - Do not guard read/query paths that continue to work without invoking unsupported SI APIs.
  - Keep v0/v1 DataJoint schemas unchanged.

- **Optionally port only narrow legacy shims.**
  - If a v1 compatibility adapter is genuinely small, it may replace `si.extract_waveforms(...)` with `si.create_sorting_analyzer(...)` and `si.load_waveforms(...)` with `si.load_sorting_analyzer(...)` while preserving public method names.
  - Any such port must prove notebook-facing behavior and metric values remain within documented tolerances.
  - If the port touches broad MetricCuration/BurstPair behavior, stop and keep the legacy-runtime boundary instead.

- **Bump dependencies for the v2 runtime environment.**
  - Change `spikeinterface>=0.99.1,<0.100` to `spikeinterface>=0.104,<0.105`.
  - Add `mountainsort5>=0.5`.
  - Verify the MS4 runtime package in the same resolver matrix. MS4 is known to install on Linux, so Phase 0c must record Linux resolver/runtime evidence and `spikeinterface.sorters.installed_sorters()` output before Phase 1 ships an MS4 default row. If macOS or another developer platform cannot install the runtime, document that platform-specific limitation and add a clear runtime guard instead of removing MS4 from the Linux-supported default set. Record explicitly that MS4 is **not deterministic**; resolver/runtime checks prove availability only, not repeatable spike-time output.
  - Add optional dependency group `spikesorting-v2-matching = ["UnitMatchPy>=3.3,<4", "mat73"]` only after resolver testing confirms it does not force an incompatible NumPy downgrade in the supported Python range and the UnitMatchPy import path is usable or clearly guarded.

- **Update user-facing docs.**
  - Document the environment split: v2/new processing uses SI 0.104+; active v0/v1 runtime workflows use the legacy SI 0.99 environment unless explicitly ported.
  - Preserve the promise that existing v0/v1 rows stay queryable where possible.
  - Do not describe v1 as the production runtime under the new SI pin unless this phase actually ports and validates it.

- **Run resolver checks.**
  - Verify Python 3.10, 3.11, and 3.12 environments resolve.
  - Record SpikeInterface, NumPy, Zarr, numcodecs, `mountainsort5`, MS4 Linux runtime status, any non-Linux MS4 limitations, and `spikeinterface.sorters.installed_sorters()` output in `tests/spikesorting/v2/resolver/si0104-runtime.md`.

## Validation slice

| Test | Asserts |
| --- | --- |
| Global SI import audit | `tests/spikesorting/v2/test_legacy_runtime_boundary.py` or an audit artifact lists every `spikeinterface` import under `src/spyglass` and records its classification. The required non-spike-sorting consumers above are present in the artifact. |
| Legacy import smoke | `tests/spikesorting/v2/test_legacy_runtime_boundary.py` imports v0/v1 modules and the required non-spike-sorting consumers used by existing downstream query paths under SI 0.104, or verifies unsupported active-runtime imports raise the explicit legacy-env guard. |
| Legacy merge query smoke | `tests/spikesorting/v2/test_legacy_runtime_boundary.py` verifies existing v0/v1 `SpikeSortingOutput` merge rows remain queryable for spike times / firing rates where the path does not invoke WaveformExtractor-era recomputation. It must use narrow fixtures or stubs that do not call active v0/v1 populate paths. |
| Legacy runtime guard tests | `tests/spikesorting/v2/test_legacy_runtime_boundary.py` verifies v0/v1 active runtime paths classified as legacy-only raise the clear SI 0.99 environment message under SI 0.104. |
| Optional v1 shim tests | Only if this implementation ports a narrow v1 shim: `MetricCuration.get_waveforms`, relevant metric helpers, and BurstPair callers still satisfy documented behavior. |
| `test_no_legacy_schema_changes` | v0/v1 DataJoint `definition` strings for recording, sorting, curation, metric curation, burst curation, and recompute tables are byte-identical before/after this implementation. |
| `test_pyproject_si_pin` | `pyproject.toml` requires `spikeinterface>=0.104,<0.105`. |
| `test_sorter_runtime_resolution` | `mountainsort5` imports, `sis.installed_sorters()` includes `mountainsort5`, and MS4 runtime status is explicit: Linux `sis.installed_sorters()` includes `mountainsort4`; any non-Linux runtime limitation is documented with a platform-specific guard. This test does not assert deterministic MS4 output. |
| `test_optional_matching_extra_resolution` | The `spikesorting-v2-matching` extra includes both `UnitMatchPy>=3.3,<4` and `mat73`, resolves without NumPy incompatibility, and import guards produce a clear message if UnitMatchPy hits the `_tkinter` import path. |
| `test_correct_motion_api_contract` | The pinned SI 0.104 environment exposes a `correct_motion` path compatible with Phase 3's "corrected recording only" contract, and the recorded resolver artifact lists accepted/rejected kwargs. |

## Commands to run

```bash
uv venv .venv-spikesorting-v2-si0104
source .venv-spikesorting-v2-si0104/bin/activate
uv pip install -e ".[test]"
python --version

pytest tests/spikesorting/v2/test_legacy_runtime_boundary.py -q
uv pip check
mkdir -p tests/spikesorting/v2/resolver
python - <<'PY'
import spikeinterface as si
import spikeinterface.sorters as sis
import spikeinterface.preprocessing as spre
import numpy as np
import zarr
import numcodecs
import inspect
from packaging.version import Version
assert Version(si.__version__) >= Version("0.104")
installed = set(sis.installed_sorters())
assert "mountainsort5" in installed
print("correct_motion_signature", inspect.signature(spre.correct_motion))
print("python resolver env ok")
print("spikeinterface", si.__version__)
print("numpy", np.__version__)
print("zarr", zarr.__version__)
print("numcodecs", numcodecs.__version__)
print("installed_sorters", sorted(installed))
PY
python - <<'PY' > tests/spikesorting/v2/resolver/si0104-runtime.md
import platform
import spikeinterface as si
import spikeinterface.sorters as sis
import spikeinterface.preprocessing as spre
import numpy as np
import zarr
import numcodecs
import inspect

print("# SpikeInterface 0.104 runtime resolver")
print()
print(f"- Python: {platform.python_version()}")
print(f"- Platform: {platform.platform()}")
print(f"- SpikeInterface: {si.__version__}")
print(f"- NumPy: {np.__version__}")
print(f"- Zarr: {zarr.__version__}")
print(f"- numcodecs: {numcodecs.__version__}")
print(f"- installed_sorters: {sorted(sis.installed_sorters())}")
print(f"- correct_motion_signature: `{inspect.signature(spre.correct_motion)}`")
PY
uv pip freeze > tests/spikesorting/v2/resolver/si0104-freeze.txt
rg -n "import spikeinterface|from spikeinterface" src/spyglass > tests/spikesorting/v2/resolver/si0104-spikeinterface-imports.txt
git diff --check -- pyproject.toml src/spyglass/spikesorting/v0 src/spyglass/spikesorting/v1 src/spyglass/utils src/spyglass/common src/spyglass/decoding tests/spikesorting
```

## Deliberately not in this phase

- No v2 production tables.
- No v2 `Recording`, `Sorting`, `CurationV2`, `AnalyzerCuration`, `SessionGroup`, `UnitMatch`, or FigPack implementation.
- No broad v1 redesign. The default outcome is an explicit legacy-runtime boundary, not a full v1 modernization.
- No promise that active v0/v1 population, MetricCuration, BurstPair, FigURL, or recompute workflows run under SI 0.104 unless this phase explicitly ports and validates that surface.

## Review

Before opening Phase 1, reviewers must confirm Phase 0c has landed, the SI 0.104 resolver is pinned, every Spyglass SpikeInterface import has been audited, non-spike-sorting consumers have smoke coverage or guards, the `correct_motion` API contract has been recorded, legacy runtime boundaries are documented, and unsupported v0/v1 active workflows fail with clear legacy-environment guidance. Phase 1 is blocked until this compatibility boundary is explicit; it is not blocked on a broad v1 runtime port.
