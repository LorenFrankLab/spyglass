# SpikeInterface 0.104 runtime resolver

- Python: 3.11.13
- Platform: Linux-5.15.0-179-generic-x86_64-with-glibc2.35
- SpikeInterface: 0.104.3
- probeinterface: 0.3.2
- NumPy: 2.4.6
- Zarr: 2.18.7
- numcodecs: 0.15.1
- pynwb: 3.1.3
- hdmf: 4.3.1
- mountainsort5: 0.5.8
- installed_sorters: ['lupin', 'mountainsort5', 'simple', 'spykingcircus2', 'tridesclous2']
- correct_motion signature: `(recording: spikeinterface.core.baserecording.BaseRecording, preset: Literal['dredge', 'medicine', 'dredge_fast', 'nonrigid_accurate', 'nonrigid_fast_and_accurate', 'rigid_fast', 'kilosort_like'] = 'dredge_fast', folder: str | pathlib.Path | None = None, output_motion: bool = False, output_motion_info: bool = False, overwrite: bool = False, detect_kwargs: dict = {}, select_kwargs: dict = {}, localize_peaks_kwargs: dict = {}, estimate_motion_kwargs: dict = {}, interpolate_motion_kwargs: dict = {}, **job_kwargs)`

## Active-runtime status of SI 0.104 surfaces v0/v1 depend on

| Surface | 0.104 status | Effect on v0/v1 active runtime |
|---|---|---|
| `si.WaveformExtractor` | removed | annotation-only; deferred via `from __future__ import annotations` in Phase 0a / e5f4928a |
| `si.extract_waveforms(...)` | backwards-compat mock returning a `SortingAnalyzer` | rejects `overwrite=True/False` (must be `None`); v0/v1 callers pass `overwrite=True` so the active path fails with `AssertionError: overwrite=True/False is not supported anymore` |
| `si.load_waveforms(folder)` | backwards-compat mock | works on existing folders |
| `si.create_sorting_analyzer` / `si.load_sorting_analyzer` | present (the modern replacements) | available for any future port |
| `sq.compute_snrs` | present | usable |
| `sq.nearest_neighbors_isolation` | absent | v0/v1 metric registries already wrap with `getattr(..., None)` (commit e5f4928a) |
| `sq.nearest_neighbors_noise_overlap` | absent | same |
| `spikeinterface.qualitymetrics` module | DeprecationWarning -- will be removed in 0.105.0 (use `spikeinterface.metrics.quality`) | pin range `<0.105` keeps this safe; future SI bump will require a rename |
| `spikeinterface.core.job_tools.ChunkRecordingExecutor` | signature widened: `init_func`/`init_args` positional, plus `pool_engine`, `mp_context`, `need_worker_index` | v0/v1 `ArtifactDetection.make` calls with the old signature will fail |
| `spikeinterface.core.job_tools.ensure_n_jobs` | signature unchanged (`(recording, n_jobs=1)`) | usable |
| `spikeinterface.sortingcomponents.peak_detection.detect_peaks` | accepts `**old_kwargs` for backwards compat | usable from v0/v1 |
| `spikeinterface.sorters.run_sorter` | signature broadly compatible | usable; per-sorter parameter changes still possible |
| `spikeinterface.curation` | present (53 attrs) | basic curation read paths stable |

## Motion-correction API contract for Phase 3

- `correct_motion(recording, preset='dredge_fast', output_motion=False, output_motion_info=False, folder=None, **job_kwargs)` returns a corrected recording only when both `output_motion` and `output_motion_info` are `False`.
- Phase 3's MVP contract (overview.md): persist only the corrected `ElectricalSeries`, sample boundaries, and hash; do not persist motion estimates / motion-info side artifacts. The 0.104 API supports this directly via the `output_motion=False, output_motion_info=False` default.
- Available presets in 0.104: `dredge`, `medicine`, `dredge_fast`, `nonrigid_accurate`, `nonrigid_fast_and_accurate`, `rigid_fast`, `kilosort_like`. Phase 3's default-row choice for `MotionCorrectionParameters` lands within this set.
- `folder` accepts a path for intermediate side artifacts; passing `folder=None` keeps the recording in memory and writes nothing to disk other than what the caller writes explicitly.

## Resolver-clean evidence

`uv pip install -e ".[test]"` resolved on Python 3.11 without `--upgrade` overrides. Initial attempt failed because the legacy `probeinterface<0.3` pin conflicts with `spikeinterface>=0.104`'s requirement `probeinterface>=0.3.2`; the pin was updated to `probeinterface>=0.3.2`. No further conflicts.

The pre-bump comment behind the `<0.3` pin ("Bc some probes fail space checks") is empirically retired by probeinterface 0.3.2:

- **Frank-lab tetrode** (4 contacts, ~12.5 um pitch): builds, JSON-roundtrips via `write_probeinterface` / `read_probeinterface`, and attaches to a generated SI 0.104 recording via `recording.set_probe(...)`.
- **LLNL polymer 128c-4s6mm6cm-15um-26um-sl** (4 shanks × 32 contacts, the current implant): same.
- **Neuropixels-128-sim** (1 shank, 2-col staggered): same.

Live ingestion evidence: `pytest tests/spikesorting/v2/` is 35/35 green and exercises (a) `mini_insert` of `minirec20230622.nwb` (Frank-lab tetrode) and (b) the MEArec polymer fixture round-trip through `insert_sessions` (128-ch 4-shank polymer). Both populate `Probe`, `ProbeType`, `Probe.Shank`, and `Probe.Electrode` without errors.

## Sorter availability (Linux x86_64)

- `mountainsort5` installs via the project dependency and registers in `installed_sorters()`. **MS5 is not deterministic**; resolver/runtime checks prove availability only, not repeatable spike-time output.
- `mountainsort4` is now in the `[spikesorting-v2]` extra (Phase 1d code-review followup). The PyPI package `mountainsort4==1.0.7` pulls `isosplit5` + `pybind11` + `spikeextractors` + `scikit-learn` -- enough for the SI wrapper to **register** in `installed_sorters()` -- but it does **NOT** pull `ml_ms4alg`, MS4's actual algorithm backend. `ml_ms4alg` is a numpy<2-era package that does not install under the v2 `numpy>=2` baseline (its build needs the deprecated `sklearn` shim), so an MS4 *sort* fails with `ModuleNotFoundError: ml_ms4alg` even though the wrapper is "installed." Preflight's `sorter_runtime_available` check imports the backend and catches this, and the shipped `run_v2_pipeline` default is MS5 (which runs as-is) -- so installing the extra does **not** give a complete *runnable* sorter set on a `numpy>=2` env; MS4 needs a `numpy<2` environment. **MS4 is not deterministic either**; same caveat as MS5.
- Sorters actually in `installed_sorters()` after this install: `['lupin', 'mountainsort4', 'mountainsort5', 'simple', 'spykingcircus2', 'tridesclous2']`.
