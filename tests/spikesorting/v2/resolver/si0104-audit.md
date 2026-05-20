# SpikeInterface 0.104 compatibility audit

Inventory and classification of every `spikeinterface` import in `src/spyglass`
ahead of the SpikeInterface dependency bump from `>=0.99.1,<0.100` to
`>=0.104,<0.105`. The classification drives where runtime guards are added
in the next slice.

## Method

```bash
grep -rnE "^(import|from) spikeinterface" src/spyglass --include="*.py"
```

Returned **50 imports across 28 files**. Each file was then inspected at its
call sites for SpikeInterface 0.104-removed names — primarily
`extract_waveforms`, `load_waveforms`, `WaveformExtractor.load_from_folder`,
`nearest_neighbors_isolation`, `nearest_neighbors_noise_overlap`,
`compute_snrs` — and for any SI module whose call signatures change between
0.99 and 0.104.

## Classification categories

- **query-compatible**: imports only stable SI APIs (base extractors,
  `load_extractor`, basic preprocessing). Existing DB rows / merge outputs
  read fine under SI 0.104 without invoking the removed APIs.
- **guarded legacy-runtime-only**: contains call-site usage of an SI API
  removed or renamed in 0.104. Active populate / recompute / curation
  requires the SI 0.99 legacy environment; adds a runtime guard so callers
  get an explicit error instead of an `AttributeError`.
- **safe-to-port**: a small, schema-neutral adapter would preserve behavior
  (`extract_waveforms` → `create_sorting_analyzer`,
  `load_waveforms` → `load_sorting_analyzer`). Plan default is to skip the
  port in favor of the guard; recorded here for completeness only.
- **v2-only**: new code written against SI 0.104 from the start.

The user has selected the **strict** boundary: every guarded-legacy entry
point gets a runtime guard in the next slice; no narrow shims are ported.

## Classification

### v2-only

| File | Notes |
| --- | --- |
| `spyglass/spikesorting/v2/utils.py` | Written against SI 0.104; reads `si.get_global_job_kwargs()` |
| `spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py` | Phase 0b converter; uses `MEArecRecordingInterface` (neuroconv) and `ndx_franklab_novela` |

### guarded legacy-runtime-only

Files whose entry-point bodies directly call an API removed or changed in
SI 0.104. Empirical verification narrowed the initial broad classification
to **11 entry points across 8 files**; the remaining
``SpikeSortingRecording``, ``SpikeSorting``, ``RecordingRecompute`` makes
use only stable SI APIs (``read_nwb_recording``, ``bandpass_filter``,
``run_sorter``, ``detect_peaks`` -- all present and backwards-compatible
in 0.104) and are listed under query-compatible below.

| File | Removed-API call sites | Guarded entry points |
| --- | --- | --- |
| `spyglass/spikesorting/v0/spikesorting_curation.py` | `si.extract_waveforms` (445), `si.WaveformExtractor.load_from_folder` (496, 678) | `Waveforms.make_compute`, `Waveforms.load_waveforms`, `QualityMetrics.make_compute` |
| `spyglass/spikesorting/v0/spikesorting_burst.py` | calls `Waveforms().load_waveforms` (169) | `BurstPair.make` |
| `spyglass/spikesorting/v0/spikesorting_artifact.py` | `ChunkRecordingExecutor` (258, signature widened in 0.104) | `ArtifactDetection.make` |
| `spyglass/spikesorting/v1/metric_curation.py` | `si.extract_waveforms` (387), `si.load_waveforms` (395) | `MetricCuration.make_compute`, `MetricCuration.get_waveforms` |
| `spyglass/spikesorting/v1/burst_curation.py` | depends on `MetricCuration.get_waveforms` | `BurstPair.make` |
| `spyglass/spikesorting/v1/artifact.py` | `ChunkRecordingExecutor` (292, signature widened in 0.104) | `ArtifactDetection.make` |
| `spyglass/decoding/v0/clusterless.py` | `si.extract_waveforms` (215) | `UnitMarks.make` |
| `spyglass/decoding/v1/waveform_features.py` | `si.extract_waveforms` (217 via `_fetch_waveform`) | `UnitWaveformFeatures.make` |

Transitively guarded (no direct call site, but consumed only by the
guarded entry points above):

- `spyglass/utils/waveforms.py` (`_get_peak_amplitude`) -- duck-types on the
  WaveformExtractor protocol; every caller is guarded.
- `spyglass/utils/mixins/analysis.py` (`add_units_waveforms`) -- duck-types
  on `WaveformExtractor.get_waveforms` and `.sorting.get_unit_ids`; every
  caller is guarded.
- `spyglass/spikesorting/v1/metric_utils.py` -- annotation-only
  WaveformExtractor references (deferred in Phase 0b); only consumed by
  guarded `MetricCuration` paths.

### query-compatible

These import SpikeInterface but use only stable APIs (base extractors,
`load_extractor`, type hints, or version string) — they continue to work
under SI 0.104 without runtime changes. Existing query paths against v0/v1
merge outputs remain functional.

| File | What it uses |
| --- | --- |
| `spyglass/common/common_nwbfile.py` | `import spikeinterface as si` is unused at module level (no `si.X` call sites). Stable. |
| `spyglass/spikesorting/utils.py` | helper utilities; `si` used for type annotations only |
| `spyglass/spikesorting/v0/curation_figurl.py` | `si.BaseRecording`, `si.BaseSorting` annotations |
| `spyglass/spikesorting/v0/figurl_views/SpikeSortingRecordingView.py` | `si.BaseRecording` annotations + `SpikeSortingRecording.load_recording` (Spyglass helper) |
| `spyglass/spikesorting/v0/figurl_views/SpikeSortingView.py` | `si.BaseRecording`, `si.BaseSorting`, `si.load_extractor` -- stable in 0.104 |
| `spyglass/spikesorting/v0/figurl_views/prepare_spikesortingview_data.py` | `si.BaseRecording`, `si.BaseSorting` annotations |
| `spyglass/spikesorting/v0/sortingview_helper_fn.py` | `si.BaseRecording`, `si.BaseSorting` annotations |
| `spyglass/spikesorting/v0/merged_sorting_extractor.py` | subclasses `si.BaseSorting` / `si.BaseSortingSegment` -- base classes stable |
| `spyglass/spikesorting/v1/curation.py` | `si.curation`, `si.extractors` -- read-paths stable. Note: `CurationV1.get_sorting`, `get_recording`, and `_write_sorting_to_nwb_with_curation` paths must be confirmed under SI 0.104 in Slice B; current expectation is read-only stable. |
| `spyglass/spikesorting/v1/figurl_curation.py` | `import si` for stable types |
| `spyglass/spikesorting/v0/spikesorting_recording.py` | active populate via `extractors.read_nwb_recording` + `preprocessing.bandpass_filter` / `common_reference` / `whiten` -- all stable in 0.104 (verified in `si0104-runtime.md`). `SpikeSortingRecording.make` is not guarded. |
| `spyglass/spikesorting/v0/spikesorting_sorting.py` | `sorters.run_sorter` and `sortingcomponents.peak_detection.detect_peaks` -- both stable / backwards-compatible in 0.104. `SpikeSorting.make` is not guarded. |
| `spyglass/spikesorting/v0/spikesorting_recompute.py` | recomputes through stable SI APIs; no removed-name call sites. `RecordingRecompute.make` is not guarded. |
| `spyglass/spikesorting/v1/recording.py` | active populate uses stable preprocessing APIs only. Not guarded. |
| `spyglass/spikesorting/v1/sorting.py` | `sorters.run_sorter` + `detect_peaks` stable. Not guarded. |
| `spyglass/spikesorting/v1/recompute.py` | recomputes through stable APIs. Not guarded. |

### safe-to-port

Not pursued; the user selected the strict-guard boundary. If a future
implementer wants to revisit, the smallest credible port is the
`extract_waveforms`/`load_waveforms` -> `create_sorting_analyzer`/
`load_sorting_analyzer` swap in `v1/metric_curation.py` and the matching
load in `v0/spikesorting_curation.py`. The plan permits this only if
metric output stays within documented tolerances.

## Empirical findings (verified against `.venv-spikesorting-v2-si0104`)

Recorded in [`si0104-runtime.md`](si0104-runtime.md). Key surprises:

- `si.extract_waveforms` and `si.load_waveforms` are **importable** in
  0.104 as backwards-compatibility shims redirecting to
  `SortingAnalyzer`, but `extract_waveforms` **rejects
  `overwrite=True/False`** (must be `None`). Every v0/v1 caller passes
  `overwrite=True`, so the active populate paths still fail at runtime --
  the strict-guard classification holds.
- `si.WaveformExtractor` is fully removed; annotation-only references are
  already deferred (Phase 0a `fb571590` + Phase 0b `e5f4928a`).
- `sq.compute_snrs` is **present** in 0.104 (Phase 0b's
  `getattr(sq, "compute_snrs", None)` was defensive; the underlying name
  is fine). `sq.nearest_neighbors_isolation` and
  `sq.nearest_neighbors_noise_overlap` are absent, as expected.
- `spikeinterface.qualitymetrics` emits a `DeprecationWarning` and will be
  removed in 0.105.0 in favor of `spikeinterface.metrics.quality`. The
  pin `>=0.104,<0.105` keeps the legacy module path alive; a future SI
  bump beyond 0.105 will need this module renamed.
- `spikeinterface.core.job_tools.ChunkRecordingExecutor` is present but
  its signature has widened: `init_func` and `init_args` are positional,
  and `pool_engine`, `mp_context`, `need_worker_index` are new keywords.
  v0/v1 `ArtifactDetection.make` callers using the old signature will
  fail under 0.104 (guard required).
- `ensure_n_jobs(recording, n_jobs=1)` -- stable.
- `spikeinterface.sortingcomponents.peak_detection.detect_peaks` keeps
  `**old_kwargs` for back-compat; usable from v0/v1.
- `run_sorter` signature is broadly compatible; per-sorter parameter
  changes still possible.
- `correct_motion(recording, preset='dredge_fast', ...,
  output_motion=False, output_motion_info=False, folder=None,
  **job_kwargs)` -- supports Phase 3's "corrected recording only" MVP
  contract directly with default kwargs. Recorded as the Phase 3
  schema-freeze input in `si0104-runtime.md`.
- `installed_sorters()` after the resolver-clean install:
  `['lupin', 'mountainsort5', 'simple', 'spykingcircus2', 'tridesclous2']`.
  `mountainsort4` is not auto-installed by `[test]`; flagged as a
  documentation item (Linux runtime install is separate).

## Resolver-side change forced by the SI bump

- The legacy `probeinterface<0.3` pin in `pyproject.toml` conflicts with
  `spikeinterface>=0.104`'s requirement `probeinterface>=0.3.2`. Updated
  to `probeinterface>=0.3.2`; the pre-bump "probes fail space checks"
  caveat is empirically retired (tetrode + LLNL polymer + Neuropixels-128
  probes all build / JSON-roundtrip / attach to SI 0.104 recordings; live
  ingestion via `pytest tests/spikesorting/v2/` succeeds on both
  minirec tetrodes and the MEArec polymer fixture). See
  [`si0104-runtime.md`](si0104-runtime.md) for the probe-family
  evidence.
