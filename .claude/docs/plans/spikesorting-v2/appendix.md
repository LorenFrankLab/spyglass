# Appendix — External references

[← back to PLAN.md](PLAN.md)

Reference material for executors implementing against external code or formats. Load only when needed.

## Contents

- [SpikeInterface 0.99 → 0.104 migration cheat sheet](#spikeinterface-099--0104-migration-cheat-sheet)
- [SortingAnalyzer extension dependencies](#sortinganalyzer-extension-dependencies)
- [Quality metric renames in 0.104](#quality-metric-renames-in-0104)
- [MountainSort 5 install + params](#mountainsort-5-install--params)
- [Kilosort 4 install + params](#kilosort-4-install--params)
- [UnitMatchPy integration notes](#unitmatchpy-integration-notes)
- [FigPack vs FigURL](#figpack-vs-figurl)
- [Motion correction presets](#motion-correction-presets)
- [MEArec integration notes](#mearec-integration-notes)
- [Spyglass NWB ingestion requirements (trodes_to_nwb compatibility)](#spyglass-nwb-ingestion-requirements-trodes_to_nwb-compatibility)

---

## SpikeInterface 0.99 → 0.104 migration cheat sheet

Direct replacements when porting code from v1 to v2. Sources:
- https://spikeinterface.readthedocs.io/en/stable/whatisnew.html
- https://spikeinterface.readthedocs.io/en/stable/tutorials/waveform_extractor_to_sorting_analyzer.html

| 0.99 (v1 uses) | 0.104 (v2 uses) | Notes |
| --- | --- | --- |
| `si.extract_waveforms(recording, sorting, folder=...)` | `si.create_sorting_analyzer(sorting=sorting, recording=recording, format="binary_folder", folder=..., sparse=True)` | Treat the old call as legacy. If the installed SI 0.104 patch still exposes a shim, v2 still must not rely on it; Phase 0c decides whether any v0/v1 runtime path gets a narrow shim or remains legacy-SI-0.99-only. |
| `si.load_waveforms(folder)` | `si.load_sorting_analyzer(folder)` | Treat old WaveformExtractor folder loading as legacy. Phase 0c verifies whether legacy folders remain queryable under SI 0.104 or need a legacy-environment guard. |
| `we.get_template(unit_id)` | `analyzer.get_extension("templates").get_data(operator="average" or "median")` or SI's `get_template_extremum_*` helpers for peak-channel/amplitude queries | Use documented SortingAnalyzer APIs rather than v1 WaveformExtractor methods. |
| `we.get_waveforms(unit_id)` | `analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)` | v2 compatibility wrappers may expose `get_waveforms(unit_id)` for v1 callers. |
| `we.run_extract_waveforms(...)` | `analyzer.compute(["waveforms"], **kwargs)` | |
| `we.compute_quality_metrics(...)` | `from spikeinterface.metrics.quality import compute_quality_metrics; compute_quality_metrics(analyzer, ...)` | Takes an analyzer, not a we. |
| `return_scaled=True` | `return_in_uV=True` | Renamed across the API. |
| `is_scaled` | `is_in_uV` | |
| Manual chain: `bandpass → cmr → whiten` | `apply_preprocessing_pipeline(recording, {"bandpass_filter": {...}, "common_reference": {...}, "whiten": {...}})` | Declarative; serializable. New in 0.103. Import from `spikeinterface.preprocessing`. **Note**: v2 splits this into pre-motion (`bandpass + cmr`) and post-motion (`whiten`) stages — see `shared-contracts.md § Pydantic Parameter Schema Convention`. |
| `sic.MergeUnitsSorting(sorting, merge_groups)` | Same class still exists in SI 0.104. For analyzer-backed curation, prefer `spikeinterface.curation.apply_curation(...)` when applying a full curation model. | `apply_merges_to_sorting` is not present in SI 0.104.3. |
| `sic.remove_excess_spikes(sorting, recording)` | Same name, same module. | Still present in 0.104. |
| Auto-merge: `get_potential_auto_merge(we, ...)` | `compute_merge_unit_groups(analyzer, preset=...)` | Modern signature; preset-based. |
| `set_global_job_kwargs(n_jobs=N)` | Same. | |
| `ChunkRecordingExecutor` | Same. | |
| `concatenate_recordings([rec1, rec2])` | Same. | |
| `correct_motion(recording, preset=...)` | Same; new presets in 0.101+ (`dredge_fast`, `medicine`). | |

---

## SortingAnalyzer extension dependencies

Extensions form a DAG. Phase 1 / 2 must compute parents before children.

```
random_spikes  ───┐
                  ├─→ waveforms ──┬─→ templates ──┬─→ template_metrics
                  │               │               ├─→ unit_locations
                  │               │               └─→ template_similarity
                  │               └─→ principal_components ──→ (PCA-based metrics)
                  │
noise_levels  ────┘

spike_amplitudes (needs templates)
correlograms (needs random_spikes)
isi_histograms (needs nothing else)
spike_locations (needs random_spikes)
```

**For v2 `Sorting.make()`**: compute `random_spikes`, `noise_levels`, `templates`, `waveforms` at sort time (cheap and unblocks everything).

**For v2 `AnalyzerCuration.make()`**: add `correlograms`, `spike_amplitudes`, `template_similarity`, `unit_locations`, `template_metrics`, and `principal_components` (if metric params don't `skip_pc_metrics`). `template_similarity` is required before auto-merge presets such as `similarity_correlograms`; compute it explicitly so `compute_merge_unit_groups(..., compute_needed_extensions=False)` does not hide missing-extension drift.

**Source**: https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html

---

## Quality metric renames in 0.104

These break v1's `MetricParameters` blobs verbatim — Phase 2 introduces fresh `QualityMetricParameters` rows, does not migrate v1.

| 0.99 / v1 name | 0.104 / v2 name | Notes |
| --- | --- | --- |
| `peak_to_valley` | `peak_to_trough_duration` | Template metric. |
| `peak_trough_ratio` | `peak_to_trough_ratio` | Now absolute-valued; magnitudes differ from 0.99. |
| `snr` (mean-based) | `snr` (median-based) | Same name, different formula. Numeric thresholds shift; recalibrate. |
| `from spikeinterface.qualitymetrics import compute_snr` | Use `spikeinterface.metrics.quality.compute_quality_metrics(...)` for table-level v2 metrics; only import individual metric helpers from SI's documented metric submodules after checking the pinned 0.104 API. | Metrics were refactored in 0.104. |
| `auto_label_units` | `unitrefine_label_units` | UnitRefine rebranded. |

Source: SpikeInterface 0.104 release notes — https://spikeinterface.readthedocs.io/en/stable/releases/0.104.0.html

---

## MountainSort 5 install + params

**Install**: `pip install mountainsort5` (the `mountainsort5` PyPI package, separate from `mountainsort4`).

**Repo**: https://github.com/flatironinstitute/mountainsort5

**MS4 runtime caveat**: SpikeInterface 0.104.3 lists a `mountainsort4` wrapper even in a clean env where the runtime is not installed. In the Python 3.12 uv verification env, `pip install mountainsort4` failed while compiling `isosplit5` (`cstdint` header not found). Phase 0c must resolve this before Phase 1 treats MS4 as runnable; do not confuse `available_sorters()` with `installed_sorters()`. MS4 is also not deterministic, so availability checks must not become exact-output reproducibility checks.

**KS4 reproducibility caveat**: Kilosort4 seeds NumPy/PyTorch internally during sorting, but it runs through PyTorch on CPU/GPU and defaults to GPU when available. Treat KS4 as a scientific correctness candidate, especially for dense probes, not as a deterministic fallback for exact rerun or v1-v2 spike-time parity checks.

**Differences from MS4** (relevant to v2 default params):

- **No `tempdir` parameter** — MS5 doesn't require a tempdir; remove the v1 `sorter_params.pop("tempdir", None)` hack.
- **Requires preprocessed input** — MS5 expects bandpass-filtered + whitened input. The recording stage must do this; the sorter wrapper does NOT do it for you.
- **Sorting algorithms**: `scheme1`, `scheme2`, `scheme3`. Scheme 2 is the default; comparable to MS4's default behavior for tetrodes. Scheme 3 is faster for long recordings but more memory.

**Default params for Frank-lab tetrode (validated empirically against MS4 v1 defaults)**:

```python
{
    "scheme": "2",
    "detect_sign": -1,
    "detect_threshold": 5.5,
    "detect_time_radius_msec": 0.5,
    "snippet_T1": 20,
    "snippet_T2": 20,
    "scheme2_phase1_detect_channel_radius": 200,
    "scheme2_detect_channel_radius": 50,
}
```

Source: MS5 README + Phase 1 validation script outputs.

---

## Kilosort 4 install + params

**Install**: `pip install kilosort` (KS4 is the `kilosort` package).

**Repo**: https://github.com/MouseLand/Kilosort

**GPU required** for non-trivial runs (CPU fallback exists but is slow).

**KS4 SI wrapper config**: `sis.run_sorter(sorter_name="kilosort4", recording=rec, **params)` — `use_binary_file=True` is the 0.102+ default; do not override unless necessary.

**Key v2-relevant kwargs**:

- `Th_universal: float = 9` — universal templates threshold.
- `Th_learned: float = 8` — learned templates threshold.
- `nblocks: int = 1` — drift correction blocks (1 = rigid, >1 = nonrigid).
- `max_cluster_subset: int = 25_000` — long-recording optimization (KS4 paper claim: 65% runtime cut).
- `do_CAR: bool = True` — common average referencing (set False if your preprocessing already CMR'd).

Source: https://kilosort.readthedocs.io/en/latest/parameters.html

---

## UnitMatchPy integration notes

**Install**: `pip install UnitMatchPy mat73` (UnitMatchPy 3.3.1 imports `mat73` from `UnitMatchPy.utils` but does not declare it in the wheel metadata).

**Repo**: https://github.com/EnnyvanBeest/UnitMatch (Python port under `UnitMatch_python/`)

**PyPI page**: https://pypi.org/project/UnitMatchPy/ (current checked version during planning: 3.3.1, released April 2026; the plan pins `UnitMatchPy>=3.3,<4` which covers 3.3.x patch releases). PyPI metadata currently declares `python>=3.9,<3.13` and `numpy<2.0`, so Phase 4a must run a resolver check against the v2 SpikeInterface environment before adding the optional extra.

**Import caveat verified against UnitMatchPy 3.3.1**: the package import name is capitalized (`UnitMatchPy`), not `unitmatchpy`. `import UnitMatchPy` imports `UnitMatchPy.GUI`, which requires `_tkinter`; Python builds without Tk support fail at top-level import. Normal submodule imports also execute package `__init__.py`, so they can hit the same GUI failure. Phase 4a must either run in a Tk-enabled env or load the non-GUI modules by file path / upstream patch before schema work starts.

**Demo notebook**: `UMPy_spike_interface_demo.ipynb` in the repo — this is Phase 4's primary integration template.

**Core API**:

**PHASE4A_CONTRACT_STUB — finalized in Phase 4a.** If this marker is still
present, the UnitMatchPy API has not been verified and Phase 4b has not started.
The sketch below captures partial planning-time observations only. Phase 4a
must replace it with the exact UnitMatchPy import path, entry-point calls, input
bundle layout, output schema, and measured runtime/memory behavior before Phase
4b starts.

```python
# Phase 4a must replace this sketch with imports verified in the target env.
# In a Tk-enabled env this may be:
# from UnitMatchPy import default_params, overlord, utils as util
# In a no-Tk env, top-level UnitMatchPy imports fail because __init__.py imports GUI.

# Inputs:
# - RawWaveforms/Unit*_RawSpikes.npy files, one directory per session.
# - Per-unit waveform shape is (n_samples, n_channels, 2); the last axis is
#   the two half-recording / cross-validation waveform estimates.
# - channel_positions.npy: shape (n_channels, 2 or 3)
# - cluster_group.tsv or equivalent good-unit metadata per session

param = default_params.get_default_param()
param["KS_dirs"] = [session_dir_1, session_dir_2, ...]
wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(param["KS_dirs"])
waveform, session_id, session_switch, within_session, good_units, param = (
    util.load_good_waveforms(wave_paths, unit_label_paths, param, good_units_only=True)
)
# The current demo then calls UnitMatchPy.overlord.extract_parameters(),
# UnitMatchPy.overlord.extract_metric_scores(), and the Bayes helpers.
# Phase 4a must replace this sketch with the exact wrapper code and outputs.
```

**Key features extracted** (from the paper):
- Spatial decay
- Average waveform shape
- Centroid position (monopolar fit)
- Travelling-wave direction
- Waveform duration

**Memory**: documented to need >32 GB RAM for "large datasets" (likely >1000 units across sessions). For Frank-lab tetrode setups (typically <500 units total), fits in <8 GB.

**Tetrode caveat**: UnitMatch's published validation is Neuropixels-heavy (hundreds of channels per unit). On tetrodes (4 channels), spatial features have very low discriminative power. Phase 4 gates on the Frank-lab polymer-probe fixture; tetrode validation is informational and does not declare production tetrode matching unless the measured AUC supports it.

**DeepUnitMatch** (v2 Phase 4.1 future hook) lives in the same `UnitMatchPy` repo under `DeepUnitMatch/`. Pretrained model for inference; CNN over multi-channel waveforms. Drop-in via the same `match()` API.

Sources:
- van Beest et al. 2024 Nature Methods: https://www.nature.com/articles/s41592-024-02440-1
- DeepUnitMatch bioRxiv 2026: https://www.biorxiv.org/content/10.64898/2026.01.30.702777v1

---

## FigPack vs FigURL

**FigURL** (v1's curation UI) is the SortingView-backed web app. Requires `kachery-cloud` for state persistence. v1's `FigURLCuration` lives at [src/spyglass/spikesorting/v1/figurl_curation.py](../../../../src/spyglass/spikesorting/v1/figurl_curation.py).

**FigPack** is the intended FigURL successor UI path. Repo: https://github.com/flatironinstitute/figpack. Current packaging during plan review is a core `figpack` package plus a spike-sorting extension package named `figpack-spike-sorting` on PyPI and imported as `figpack_spike_sorting`. Phase 5 must still verify the exact view-construction API, upload/show method, and edited-curation state round trip before implementing the DataJoint table.

**Migration policy** (per resolved decision #2 in `overview.md`):
- Phase 1 ships v2 with NO curation UI table — users curate by editing `CurationV2` rows directly in Python.
- Phase 5 introduces **FigPack** as v2's curation UI, gated by a Phase 5a feasibility check. If FigPack proves unusable at Phase 5 implementation time, Phase 5 stops and escalates to the project owner — there is no silent FigURL fallback for v2.
- v1's FigURL stays usable for v1 data only; it is NOT extended to v2 curations.

---

## Motion correction presets

Available in `spikeinterface.preprocessing.correct_motion()` as of 0.104:

| Preset | Algorithm | Best for | Speed |
| --- | --- | --- | --- |
| `rigid_fast` | KS-like rigid shift, single block | Same-day recordings with mild drift | Fast |
| `kilosort_like` | KS2.5/3 algorithm | Standard Neuropixels drift | Medium |
| `dredge_fast` | DREDge fast variant | Chronic recordings, large drift | Medium |
| `dredge` | Full DREDge | Best accuracy, large drift / long records | Slow |
| `medicine` | MEDiCINe | Alternative to DREDge | Medium |
| `nonrigid_accurate` | Nonrigid with monopolar localization | High-density probes, severe drift | Very slow |
| `nonrigid_fast_and_accurate` | Nonrigid DREDge/KS-like hybrid | High-density probes, severe drift | Medium/slow |

For Phase 3 default: `rigid_fast` (same-day, fast).
For opt-in multi-day concat: caller must choose an explicit non-`auto` preset; `dredge_fast` or `dredge` are candidate presets, but sort-then-match remains the recommended cross-day workflow.

Source: https://spikeinterface.readthedocs.io/en/stable/modules/motion_correction.html

---

## MEArec integration notes

MEArec ([SpikeInterface/MEArec](https://github.com/SpikeInterface/MEArec), Buccino & Einevoll 2020 Neuroinformatics) is a biophysical simulator for extracellular recordings with absolute ground-truth spike trains. v2 uses it as the primary validation oracle — `minirec` is too short to contain real spikes and is reduced to a plumbing-only fixture.

### What MEArec generates

- `RecordingGenerator.recordings`: shape `(n_samples, n_channels)` raw traces.
- `RecordingGenerator.spiketrains`: list of Neo SpikeTrain objects, one per ground-truth unit.
- `RecordingGenerator.templates`: jittered template tensor, shape `(n_units, n_jitters, n_electrodes, n_samples)`.
- `RecordingGenerator.template_locations`: per-unit 3D soma positions.
- `RecordingGenerator.channel_positions`: probe electrode layout.
- `RecordingGenerator.cell_types`: per-unit cell-type label.

### Probes (relevant subset)

- **Linear tetrode probe** (4 channels) — primary fixture for Frank-lab tetrode validation.
- **Neuropixels-128** (128 channels, ~100 templates per cell model) — for Phase 4 UnitMatch validation against published Neuropixels-only validation.
- **Neuronexus-32** (32 channels, 30 drifting templates per cell) — used by the 2024 motion-correction benchmark paper.

### Drift simulation

Critical for Phase 3 motion-correction validation:
- **Slow drift**: continuous, ~5 μm/min default.
- **Fast drift**: discrete jumps, ~20 s interval by default.
- **Rigid** (all neurons move together) or **non-rigid** (depth-gradient — neurons at probe bottom move at 50% the speed of those at top).

### Multi-session workaround

MEArec has no native multi-session concept. For Phase 4 cross-session validation, generate **two recordings using the same `templates` file** with different `seeds.spiketrain` and a small applied drift between them. The shared templates ARE the shared biological units; UnitMatch should recover the correspondence. This is the approach the 2024 drift-benchmark paper used.

### Integration with SpikeInterface + NeuroConv

- `MEArecRecordingExtractor` exposes the recording side as a SpikeInterface `BaseRecording`.
- `MEArecSortingExtractor` exposes the ground-truth spike times as a `BaseSorting` — directly usable by `spikeinterface.comparison.compare_sorter_to_ground_truth`.
- `neuroconv.datainterfaces.MEArecRecordingInterface` writes the recording side to NWB (`ElectricalSeries` + electrodes table + probe). Install: `pip install "neuroconv[mearec]"`.
- The Units ground-truth table is NOT written by NeuroConv's interface — v2's converter adds it manually from `RecordingGenerator.spiketrains` + `template_locations`. ~20 LOC.

### Sources

- [MEArec docs — generate recordings](https://mearec.readthedocs.io/en/latest/generate_recordings.html)
- [Buccino & Einevoll 2020, MEArec paper, Neuroinformatics](https://link.springer.com/article/10.1007/s12021-020-09467-7)
- [Garcia et al. 2024, modular drift-correction benchmark, eNeuro / PMC10897502](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897502/)
- [SpikeInterface blog — ground-truth comparison and ensemble sorting](https://spikeinterface.github.io/blog/ground-truth-comparison-and-ensemble-sorting-of-a-synthetic-neuropixels-recording/)
- [NeuroConv MEArec interface docs](https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/recording/mearec.html)

---

## Spyglass NWB ingestion requirements (trodes_to_nwb compatibility)

The MEArec → NWB converter must produce files that Spyglass's `insert_sessions` ingests successfully. Reference: [LorenFrankLab/trodes_to_nwb](https://github.com/LorenFrankLab/trodes_to_nwb) — the canonical converter for Frank-lab raw data.

### ElectricalSeries naming (required)

Spyglass's `Raw` ingestion at [common_ephys.py:289-294](../../../../src/spyglass/common/common_ephys.py#L289-L294) looks for the first `ElectricalSeries` matching one of:

```python
_source_nwb_object_name = ["e-series", "electricalseries", "ephys", "electrophysiology"]
```

`trodes_to_nwb` writes to `"e-series"` (`convert_ephys.py` builds the raw acquisition with `ElectricalSeries(name="e-series", ...)`). **Match that name** in the MEArec converter — NeuroConv's default may differ. Override via `metadata={"Ecephys": {"ElectricalSeries": {"name": "e-series"}}}`.

### YAML metadata fields (from trodes_to_nwb 20230622_sample_metadata.yml)

The converter must populate these NWB-level fields (synthetic but well-formed):
- `experimenter_name` (list of "lastname, firstname")
- `lab`, `institution`
- `experiment_description`, `session_description`, `session_id`, `keywords`
- `subject`: `description`, `genotype`, `sex`, `species`, `subject_id`, `date_of_birth`, `weight`
- `data_acq_device`, `device`
- `electrode_groups`: per physical probe/electrode group, `location`, `device_type`, coordinates `(x, y, z)`, `targeted_location`
- `ntrode_electrode_group_channel_map` (one mapping per ntrode/shank map; the polymer probe-reconfiguration sample uses one electrode group for the 128-channel probe and four ntrode maps pointing at that same group)

### Probe device types Spyglass recognizes

- `tetrode_12.5` — Frank-lab default tetrode; recognized by Spyglass at [recording.py:630-643](../../../../src/spyglass/spikesorting/v1/recording.py#L630-L643) for probe-geometry handling.
- Neuropixels device types via `probeinterface`.

For MEArec fixtures, plant the device_type explicitly to match a Spyglass-recognized probe — otherwise the ingestion succeeds but downstream probe-geometry handling falls back to channel-position-only.

### Frank-lab / Novela electrode format for polymer fixtures

Spyglass ephys ingestion has a stronger path when the NWB uses `ndx_franklab_novela.Probe`, `Shank`, and `ShanksElectrode` objects. `ProbeType`, `Probe`, `Probe.Shank`, and `Probe.Electrode` ingest from those objects ([common_device.py:335-504](../../../../src/spyglass/common/common_device.py#L335-L504)). `Electrode.make()` then links each electrode row to `Probe.Electrode` only if the NWB electrode table carries the rec_to_nwb / novela-style columns `probe_shank`, `probe_electrode`, `bad_channel`, and `ref_elect_id` ([common_ephys.py:157-181](../../../../src/spyglass/common/common_ephys.py#L157-L181)).

Therefore the polymer MEArec converter must not stop at generic NeuroConv electrode metadata. It must create the Frank-lab/Novela probe object and add those electrode-table columns. Use the local `trodes_to_nwb` reference files as the source of truth: `convert_yaml.py::add_electrode_groups`, `tests/test_data/20230622_sample_metadataProbeReconfig.yml`, and `device_metadata/probe_metadata/128c-4s6mm6cm-15um-26um-sl.yml`.

- `probe_type = "128c-4s6mm6cm-15um-26um-sl"`
- One NWB `ElectrodeGroup` for the whole 128-channel probe, not one group per shank.
- 4 Novela `Shank` objects, 32 contacts per shank.
- `probe_shank` values `0..3`.
- `probe_electrode` values `0..127` globally within the probe, matching `ShanksElectrode.name` and `convert_yaml.py`'s `electrode_counter_probe`.
- `bad_channel = False` for generated fixtures unless a test intentionally marks channels bad.
- `ref_elect_id = -1` unless the fixture intentionally tests a reference electrode.
- `rel_x`, `rel_y`, `rel_z` copied from the same probe metadata used by MEArec generation.

Round-trip validation must assert `Electrode * Probe.Electrode` has 128 rows for the polymer fixture and that no `"Electrode did not match expected novela format"` warning was emitted.

### BrainRegion validation with trodes-compatible electrodes

Spyglass's `BrainRegion` table is populated from each electrode's NWB `group.location` during `Electrode.make()` ([common_ephys.py:128-140](../../../../src/spyglass/common/common_ephys.py#L128-L140)). `trodes_to_nwb` also stores `targeted_location` on the NWB `ElectrodeGroup`, but current Spyglass `Electrode.make()` does not use that field for per-electrode regions.

For the polymer fixture, keep the NWB trodes-compatible: one electrode group for the physical probe. That means NWB ingestion gives all electrodes the group-level default region. For the v2 brain-region tracing regression, plant multi-region ground truth after ingestion using an isolated-test config/update path such as `Electrode.create_from_config(...)`, assigning `Electrode.region_id` by `probe_shank` (for example shank 0 -> CA1, shank 1 -> CA3). This tests the v2 unit-to-region joins without falsifying the Frank-lab NWB structure.

### Round-trip validation

The Phase 0 fixture generator runs `Nwbfile.insert_from_relative_file_name(out_path)` immediately after writing each fixture and asserts:
- `Session & {"nwb_file_name": fixture}` exists with one row.
- `Raw & {"nwb_file_name": fixture}` exists with one row.
- `Electrode & {"nwb_file_name": fixture}` has the expected count.
- `BrainRegion & {"region_name": planted_region}` exists for each planted region.
- For polymer fixtures, `ProbeType`, `Probe`, `Probe.Shank`, and `Probe.Electrode` rows exist and `Electrode * Probe.Electrode` has 128 rows.
- Imported ground-truth Units match MEArec's `MEArecSortingExtractor` spike trains by unit id within one sample.

If round-trip fails on a freshly-generated fixture, the converter is broken — fail fast at fixture-generation time rather than at test-run time.

### Sources

- [LorenFrankLab/trodes_to_nwb GitHub](https://github.com/LorenFrankLab/trodes_to_nwb)
- [trodes_to_nwb sample metadata YAML](https://github.com/LorenFrankLab/trodes_to_nwb/blob/main/src/trodes_to_nwb/tests/test_data/20230622_sample_metadata.yml)
- Spyglass [common_ephys.py:276-330](../../../../src/spyglass/common/common_ephys.py#L276-L330) — Raw ingestion logic.
