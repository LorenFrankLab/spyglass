# UnitMatchPy cross-session matching environment — resolver evidence

Durable record of the isolated environment used to verify the UnitMatchPy
cross-session matching path against the modern (SortingAnalyzer) spike-sorting
stack. Companion freeze: [`unitmatch-matching-freeze.txt`](unitmatch-matching-freeze.txt).

## Environment

- Python **3.11.15** (UnitMatchPy declares `python>=3.9,<3.13`; 3.13+ is out of range).
- Dedicated `uv` virtualenv, separate from the SI-0.99 base environment and the
  SI-0.104 dev environment. The matching extra is **not** installed into any
  shared/base environment.
- Install command (matches the `spikesorting-v2-matching` optional-dependency group):
  `uv pip install "UnitMatchPy>=3.2.6,<3.2.8" mat73` on top of
  `spikeinterface==0.104.3` + `numpy>=2,<3` + the NWB read stack.

## Resolved versions (key packages)

| Package | Version |
| --- | --- |
| unitmatchpy | 3.2.7 |
| numpy | 2.4.6 |
| spikeinterface | 0.104.3 |
| torch | 2.12.1 |
| mat73 | 0.65 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| h5py | 3.16.0 |
| pynwb | 3.1.3 |
| probeinterface | 0.3.2 |
| zarr | 2.18.7 |

`uv pip check` → **all installed packages are compatible**. UnitMatchPy 3.2.7
coexists with numpy 2.4.6 and SpikeInterface 0.104.3 at the dependency-metadata
level.

## Resolver / import notes

- **`UnitMatchPy 3.2.7` declares `mat73`, `torch`, and `spikeinterface` as real
  dependencies** (`Requires: h5py, joblib, mat73, matplotlib, mtscomp, numpy,
  pandas, scikit-learn, scipy, spikeinterface, torch, tqdm`). The "undeclared
  `mat73`" wart applies to the later 3.3.x line, not 3.2.7 — keeping `mat73` in
  the install command is harmless belt-and-suspenders for 3.2.7. Installing
  UnitMatchPy pulls `torch` transitively.
- **`import UnitMatchPy` runs `UnitMatchPy/__init__.py`, which does
  `from . import GUI`**, and `GUI.py` imports `tkinter`/`_tkinter`. On Python
  builds without Tk support this fails at import time. This uv CPython 3.11.15
  build ships `_tkinter`, so top-level import succeeds here; a headless/Tk-less
  CI image must either provide Tk or import the non-GUI submodules by path. Any
  import guard should surface this as an actionable message.
- **`UnitMatchPy/run.py` is not importable** (`import bayes_functions as bf` —
  an unqualified import that only resolves with the package directory on
  `sys.path`). It is a reference script, not a module entry point; drive the
  pipeline through the `overlord` / `bayes_functions` / `save_utils` functions
  instead.

## numpy-2 metric-path incompatibility (confirmed, with wrapper-owned fix)

UnitMatchPy 3.2.7's metric computation does **not** actually run under numpy 2
out of the box, despite the compatible dependency metadata:

- `param_functions.get_avg_waveform_per_tp` builds a per-unit good-time window
  with `np.arange(wave_duration_tmp[0], wave_duration_tmp[-1] + 1)` where
  `wave_duration_tmp` is a `np.argwhere` result (shape `(k, 1)`), so the
  endpoints are 1-element arrays.
- numpy < 2 accepted 1-element arrays as scalars; **numpy 2 raises
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`**.
- A bare `except` swallows this for every unit (printing
  `unit{i} is very likely bad, no good time points in average waveform`) and
  leaves the trajectory written at relative argwhere indices instead of absolute
  samples → every per-timepoint centroid distance becomes NaN → the candidate
  pair set is empty → the auto-threshold quantile raises and the run aborts.

This is the latent numpy-2 break behind the upstream reactive `numpy<2` repins.
The fix is a **wrapper-owned numpy-2 shim** that coerces the 1-element-array
`arange` endpoints to scalars (a no-op for scalar args), scoped to
`UnitMatchPy.param_functions`. It does **not** edit the installed package and
preserves the numpy≥2 baseline (no isolated numpy<2 subprocess required). With
the shim installed, the pipeline runs end-to-end and recovers clean cross-session
matches (verified AUC = 1.0 on the 128-channel polymer fixture; see the
cross-session notebook).
