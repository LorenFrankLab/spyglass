# Phase 6 — v2-native clusterless waveform-feature extraction

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Clusterless decoding is a primary consumer of v2 sorts, but
`UnitWaveformFeatures.make` **cannot run under the SI 0.104 (v2) environment**:
it opens with `_require_legacy_si_environment(...)` (raises under SI≥0.101) and
its waveform extraction uses the removed `si.extract_waveforms` /
`WaveformExtractor`. The v2 dispatch branch that resolves a v2 source
(`__curation_v2`) is therefore **unreachable**. This is a real regression, not a
roadmap stub — no later spikesorting-v2 phase makes the *decoding* feature path
work under 0.104 (Phases 2/4 build `SortingAnalyzer` waveform access only for
curation/matching). This phase makes clusterless waveform-feature extraction
work for v2 sorts under SI 0.104.

**Inputs to read first:**

- [decoding/v1/waveform_features.py:135-261](../../../../src/spyglass/decoding/v1/waveform_features.py#L135-L261) — `UnitWaveformFeatures.make` (legacy guard at `:137`; v2 dispatch branch `:168-194`, reachable only after the guard is fixed), `_fetch_waveform` (`:242-261`, uses removed `si.extract_waveforms`).
- [decoding/v1/waveform_features.py:263-380](../../../../src/spyglass/decoding/v1/waveform_features.py#L263-L380) — `_compute_waveform_features` and the per-feature helpers (`_get_full_waveforms`, amplitude/spike-feature fns) — all currently typed against `si.WaveformExtractor`; the v2 path must feed these from a `SortingAnalyzer` (or the helpers must be refactored to read analyzer extensions).
- `_require_legacy_si_environment` — grep its definition (likely `spikesorting/utils` or `decoding/utils`); understand what it gates and how to scope it to v0/v1 only.
- [spikesorting/v2/sorting.py:2007-2097](../../../../src/spyglass/spikesorting/v2/sorting.py#L2007-L2097) — how v2 builds its `SortingAnalyzer` (`create_sorting_analyzer(format="binary_folder", sparse=True)`, extension set). The v2 path should reuse `Sorting.get_analyzer(key)` where possible rather than re-extracting.
- [spikesorting/v2/sorting.py:1245-1280](../../../../src/spyglass/spikesorting/v2/sorting.py#L1245-L1280) — `Sorting.get_analyzer` (rebuild-on-missing, zero-unit short-circuit).
- The phase-1b plan note that *intended* the v2 branch: `.claude/docs/plans/spikesorting-v2/phase-1b-runtime-regressions.md:807` (sparse-id indexing) — confirms the indexing design but not the SI-0.104 execution path.

## Tasks

### Task 0 — Investigation spike (do first; record findings in the Decision block)

Establish exactly what clusterless decoding needs and how to produce it from a v2 `SortingAnalyzer` under SI 0.104:

1. Enumerate the waveform features clusterless decoding actually consumes (read `WaveformFeaturesParams` defaults + `_compute_waveform_features` feature keys: amplitude, spike-width, etc.). Clusterless typically needs per-spike **amplitudes** at the peak channel.
2. Determine which `SortingAnalyzer` extensions supply them under SI 0.104 (`compute(["random_spikes","waveforms","templates"])`, the `spike_amplitudes`/`amplitude_scalings` extensions). `inspect` the installed SI to confirm extension names + outputs (do NOT guess — read `spikeinterface/postprocessing` source).
3. Decide: **reuse** the v2 `Sorting` analyzer folder via `Sorting.get_analyzer(merge_key→sorting_id)` and compute any missing extensions, **vs** build a fresh analyzer in `UnitWaveformFeatures.make` from `get_recording()` + `get_sorting()`. Prefer reuse (avoids a second full extraction) if the persisted analyzer has, or can cheaply add, the needed extensions.
4. Confirm the clusterless thresholder output (peaks-as-units) flows through this path the same way a sorted result does, and that sparse/true unit_ids index correctly.

**Decision (recorded after the spike):**

*What clusterless decoding consumes.* The shipped `WaveformFeaturesParams`
default (`waveform_features.py:56-61`) and the decoding consumer's
`low_thresh_amplitude` row (`tests/decoding/conftest.py:192-213`) both request
the **`amplitude`** feature with `peak_sign="neg"`, `estimate_peak_time=False`,
and `max_spikes_per_unit=None` (ALL spikes). `_get_peak_amplitude`
(`utils/waveforms.py:39-54`) returns `waveforms[:, n_time//2]` → shape
`(n_spikes, n_channels)`: the per-spike amplitude across the sort-group channels
at the spike-time sample. The features are written 1:1 against the full NWB
`spike_times` (`_write_waveform_features_to_nwb`), so the amplitude array MUST
cover every spike, not a subsample.

*Which SI 0.104 extension supplies it.* Verified against installed SI source
(`spikeinterface/core/analyzer_extension_core.py`):
- `random_spikes` (`:57`) with `method="all"` selects **all** spikes
  (`:40-41` "ignored if method='all'") — required because clusterless needs
  every spike's amplitude.
- `waveforms` (`:165`) `get_waveforms_one_unit(unit_id)` (`:297-332`) returns
  `(n_spikes, n_samples, n_channels)`. `nbefore = int(ms_before*fs/1000)`
  (`:172-173`); for the symmetric default window (`ms_before==ms_after`)
  `n_time//2 == nbefore == the spike sample`, so the legacy `_get_peak_amplitude`
  center-index logic is exact. Empirically confirmed on a synthetic tetrode
  (`scratch spike`): `n_time//2 == nbefore == 15` and per-unit waveform count
  equals the spike-train count for both `sparse=True/False`.

*Reuse vs rebuild — REBUILD a fresh in-memory analyzer.* Reuse of the persisted
v2 `Sorting` analyzer (`Sorting.get_analyzer`) is **rejected**: its `waveforms`
extension is built with `max_spikes_per_unit=500` (subsampled) and an asymmetric
`ms_before=1.0/ms_after=2.0` window (`spikesorting/v2/sorting.py:2122-2148`),
so (a) it covers only 500 spikes/unit but clusterless needs all spikes aligned
1:1 with the full `spike_times`, and (b) recomputing the extension on the
`binary_folder` analyzer would mutate the canonical persisted analyzer that the
curation/peak-amplitude path relies on. Instead, build a fresh `format="memory"`,
`sparse=False`, `return_in_uV=True` analyzer from
`SpikeSortingOutput().get_recording(merge_key)` +
`get_sorting(merge_key)` (both dispatch to `CurationV2` via
`spikesorting_merge.py:408-423`) and compute `random_spikes(method="all")` +
`waveforms(ms_before, ms_after)`. `sparse=False` guarantees the full per-channel
mark for every unit (deterministic channel count). This mirrors the legacy
`_fetch_waveform` semantics without `si.extract_waveforms` and without touching
the persisted folder.

*Analyzer → feature mapping.* A thin `_AnalyzerWaveformAccessor` adapts the
analyzer to the minimal surface the shared helpers use (`.sorting` +
`.get_waveforms(unit_id) → get_waveforms_one_unit`), so
`_compute_waveform_features`, `_get_peak_amplitude`, `_get_full_waveform`, and
`_write_waveform_features_to_nwb` run **unchanged** on the v2 path (v0/v1
WaveformExtractor path byte-identical). Per-unit features are keyed by the
analyzer's true `sorting.get_unit_ids()` (NWB `.id`), correct for sparse
merge-applied ids. `spike_location` is NOT served by the adapter (it needs the
analyzer object) and is not consumed by clusterless decoding nor in the
validation slice → guarded with a clear error on the v2 path (follow-up:
analyzer-native spike_location for v2 if a sorted-spikes consumer needs it).

*Clusterless flow + downstream.* The clusterless thresholder labels every peak
unit_id=0 (`spikesorting/v2/sorting.py:1859-1862`); zero peaks → zero units →
the analyzer build is short-circuited and an empty-but-valid features row is
written. No further v2 adaptation is needed downstream of `UnitWaveformFeatures`:
`fetch_data`/`_convert_data` read the written NWB columns and are
source-agnostic; `ClusterlessDecodingV1` consumes `fetch_data` output (out of
scope, unchanged).

### Task 1 — Scope the legacy-SI guard to v0/v1 only

`make()` ([:137](../../../../src/spyglass/decoding/v1/waveform_features.py#L137)) calls `_require_legacy_si_environment` unconditionally, killing the v2 path. Restructure so the source is dispatched **first**, and the legacy-SI requirement applies only to the v0/v1 branches; the v2 (`__curation_v2`) branch runs under SI 0.104 without it. Keep the v0/v1 behavior byte-identical under the legacy env.

### Task 2 — v2 `SortingAnalyzer` feature path

Implement the v2 feature extraction per the Decision:

- Add a `_fetch_waveform`-equivalent for v2 that returns a `SortingAnalyzer` (or the computed feature arrays) instead of a `WaveformExtractor`. Do **not** call `si.extract_waveforms` on the v2 path.
- Either refactor `_compute_waveform_features` + the per-feature helpers to accept a `SortingAnalyzer` (reading `analyzer.get_extension(...)`), or add v2 variants. Preserve the v0/v1 `WaveformExtractor` path unchanged (don't break legacy under SI 0.99).
- Index per-unit features by the true NWB unit_id (`.id`), not positional index (the sparse-id fix the v2 branch already documents at `:172-175`).
- Write features to NWB via the existing `_write_waveform_features_to_nwb` path (confirm it doesn't require a `WaveformExtractor` — adapt if it does).

### Task 3 — Decide the fate of `extract_waveforms` usage

After Task 2, the v2 path no longer uses `si.extract_waveforms`. If the v0/v1 path still does and still runs only under legacy SI, leave it (guarded). If the helper signatures changed, update type hints (`si.WaveformExtractor` → a union or analyzer type) and the module-top comment at `:3`.

### Task 4 — Docs

CHANGELOG: clusterless waveform-feature extraction now supported for v2 sorts under SI 0.104 (was raising). Update any decoding doc that implied v2 clusterless decoding was unsupported. Remove the "document the gap" framing — the gap is being closed, not documented.

## Deliberately not in this phase

- v0/v1 waveform extraction behavior — unchanged; this phase only adds the v2 path and scopes the legacy guard.
- The downstream `ClusterlessDecodingV1` decode step itself — out of scope; this phase makes its *input* (`UnitWaveformFeatures`) computable for v2. (Confirm in Task 0 that no further v2 adaptation is needed downstream of `UnitWaveformFeatures`; if it is, log it as a follow-up, don't expand here.)
- UnitMatch / cross-session waveform handling (spikesorting-v2 Phase 4 roadmap stub).
- **Ragged multi-unit `amplitude` column write (pre-existing, follow-up).**
  `_write_waveform_features_to_nwb` writes the per-unit `amplitude` column with
  `add_unit_column(..., data=metric_values)` (no `index=True`); when units have
  different spike counts the per-unit `(n_spikes, n_ch)` arrays are ragged and
  hdmf raises an inhomogeneous-shape error. This never bit clusterless decoding
  (the thresholder emits exactly one unit, so the column is rectangular) and is
  identical on the v0/v1 path — out of scope here. The v2 sparse-id test merges
  to a single non-positional survivor to isolate the true-id keying from this
  limitation. Wiring `index=True` (with a matching `_convert_data` read) for
  multi-unit sorts is the follow-up if a multi-unit consumer of
  `UnitWaveformFeatures` appears.
- **`spike_location` for v2 sources.** The v2 `SortingAnalyzer` path serves the
  `get_waveforms`-based features (`amplitude`, `full_waveform`); `spike_location`
  raises a clear `NotImplementedError` on the v2 path. Clusterless decoding uses
  `amplitude`, so this closes the regression; analyzer-native `spike_location`
  is a follow-up if a v2 consumer needs it.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_unit_waveform_features_v2_clusterless_runs_under_si0104` (slow/integration) | Populate a v2 clusterless `Sorting`→`CurationV2`→merge_id; `UnitWaveformFeatures.make` completes under SI 0.104 (no `_require_legacy_si_environment` raise, no `extract_waveforms` AssertionError) and writes features. |
| `test_unit_waveform_features_v2_sparse_unit_ids` (slow/integration) | A v2 merge-applied sorting with sparse unit_ids: features are keyed by true unit_id, not positional index (covers #1273 for the now-reachable v2 path). |
| `test_unit_waveform_features_v0v1_unchanged_under_legacy` (slow, legacy env) | The v0/v1 path still requires + runs under legacy SI exactly as before (guarded). Runs in the `pytest-legacy` job (Phase 5 dependency — see overview); **skip cleanly** when the legacy env/job is absent so this phase can merge independently. |
| `test_unit_waveform_features_zero_unit_v2` (slow) | A zero-unit v2 curation yields an empty-but-valid features result, not a crash. |

## Fixtures

Reuse the MEArec smoke fixture with the **clusterless_thresholder** preset (`run_v2_pipeline(preset="franklab_tetrode_clusterless_thresholder")`) to get a v2 clusterless merge_id; the existing `WaveformFeaturesParams` default row supplies feature params. The legacy-path test runs in the Phase-5 `pytest-legacy` job under SI 0.99. All `slow`.

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm:
- The **Decision** block is filled with SI-source-grounded evidence (extension names verified, not guessed).
- v0/v1 legacy behavior is unchanged (the guard still protects it; legacy test passes in the legacy job).
- No `si.extract_waveforms` / `WaveformExtractor` on the v2 path; v2 features come from a `SortingAnalyzer`.
- Per-unit indexing uses true unit_id (sparse-id test passes).
- The previously-dead v2 branch is now genuinely exercised by a test (revert Task 1 locally → the v2 test should fail with the legacy-guard raise).
- CHANGELOG/docs updated in-phase; no plan/phase references in code.
