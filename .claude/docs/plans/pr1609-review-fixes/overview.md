# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All paths relative to the repo root; line numbers verified at head cc9a7953.

Recording stage (phase 3a):
- `src/spyglass/spikesorting/v2/recording.py:2101-2137` — `Recording.make_compute` body: `read_recording_nwb` → `restrict_recording` (2109-2123) → `apply_pre_motion_preprocessing` (2124-2131) → `maybe_apply_tetrode_geometry` (2132-2137). Order changes; the read and the geometry call are preserved.
- `src/spyglass/spikesorting/v2/recording.py:1796-1818` — `Recording.get_recording`: reads the artifact by stored `electrical_series_path`, annotates `is_filtered`. Untouched (geometry is persisted in the artifact instead).
- `src/spyglass/spikesorting/v2/_recording_restriction.py:413` (`restrict_recording_times`), `:494-624` (`restrict_recording`: interval intersection, then time restriction, then the `ChannelSliceRecording` block at the end). Channel slicing moves out; time restriction stays.
- `src/spyglass/spikesorting/v2/_recording_preprocessing.py:31` — `apply_pre_motion_preprocessing` (phase-shift ~93-109, bandpass 116-135, interpolation ~136-181, reference 184-244). Split into temporal and spatial halves; the single caller is `recording.py:2124`.
- `src/spyglass/spikesorting/v2/_recording_geometry.py:166-252` — `maybe_apply_tetrode_geometry`. Untouched; gains `select_distinct_plane` / `normalize_channel_locations` / `assert_unique_contact_positions` beside it, run before any probe is constructed.
- `src/spyglass/spikesorting/v2/_recording_nwb.py:184` — `write_nwb_artifact`; series write at 320-347; content hash at 350-358. Gains a channel-location write-back (no `get_probe()` call).
- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:844-863` — 3D→2D projection with a warning. Keeps the warning; gains the all-contacts-unique assertion.
- `src/spyglass/spikesorting/v2/session_group.py:1199-1265` — concat masking (`mask_member_recordings`, `silence_frame_ranges`); the concat artifact persists masked traces, so valid ranges must travel as data (phase 3b).
- `src/spyglass/spikesorting/v2/_nwb_provenance.py` — provenance scratch writers; gains the sort's statistics spans (phase 3b). `ConcatenatedRecording` additionally persists per-member boundary spans offset into concat frames.

Sorting/analyzer stage (phase 3b):
- `src/spyglass/spikesorting/v2/sorting.py:1681` `make_compute`; mask at 1768-1776 (`_apply_artifact_mask`, def 2789); sort + `_remove_excess_spikes` at 1819-1826 (def 2895); `_build_analyzer` at 2905.
- `src/spyglass/spikesorting/v2/_sorting_dispatch.py:334` `pinned_whiten`; whiten call inside `run_si_sorter` (597) at 714-726; docstring claim at 618-621; `run_clusterless_thresholder` 386 with `_clusterless_noise_levels` at 516; `remove_excess_spikes` 835.
- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:673` `build_analyzer`; whiten branch 866-874; `noise_levels` extension params 939-951.
- `src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:27` `artifact_frame_ranges`, `:292` `apply_artifact_mask`, `:305` `silence_frame_ranges` (`silence_periods(mode="zeros")` at 323-327).
- `src/spyglass/spikesorting/v2/_si_metric_patches.py:38` `_nn_noise_overlap_sparse_fixed`; noise chunks drawn from `sorting_analyzer.recording` at 96-103.
- `src/spyglass/spikesorting/v2/_observed_time.py:47` `observed_intervals` (end computed from nominal fs at 74-75), `:79` `contains_times`, `:136` `ObservationAvailability`.
- `src/spyglass/spikesorting/v2/_signal_math.py:201` `intersect_intervals`; `:593` `_segment_times_at`.

Cross-session (phase 4a):
- `src/spyglass/spikesorting/v2/_unitmatch_backend.py:126` `extract_unitmatch_bundle`; time-half loop 208-231; `match` 328 (applies `_zero_center` at 401).

Metric curation (phase 4b):
- `src/spyglass/spikesorting/v2/_metric_curation.py:108` `apply_label_rules`; policy logic 165-215.
- `src/spyglass/spikesorting/v2/metric_curation.py:269` `QualityMetricParameters` (`observed_presence_bin_duration_s=60: float` at 286); `:432` `AutoCurationRules` (shipped rule sets 594-654); `:1179` `make_compute`; `:2142` `_compute_metrics` (PC compute 2286-2300).

Shared modules (phase 2):
- `src/spyglass/spikesorting/v0/spikesorting_curation.py:485` `Waveforms.load_waveforms` (guard at 498, `si.WaveformExtractor.load_from_folder` at 500).
- `src/spyglass/spikesorting/v1/metric_curation.py:355` `MetricCuration.get_waveforms` (guard at 370; extraction/load branch 405-414).
- `src/spyglass/spikesorting/_legacy_runtime.py:33-45` message, `:48` guard. `src/spyglass/spikesorting/_si_compat.py:12` `load_extractor`.
- `src/spyglass/spikesorting/analysis/v1/unit_annotation.py:55` `add_annotation`; `:94` `audit_positional_unit_ids`; `:151` `migrate_positional_unit_ids` (mapping at 193-197); `:331` `UnitAnnotationPositionalIdMigration`.
- `src/spyglass/spikesorting/v2/sorting.py:508` `insert_default_legacy_si_sorters` (`get_default_sorter_params` at 584, `cls.insert` at 623). `src/spyglass/spikesorting/v2/_params/sorter.py:640` `_dynamic_params` vocabulary.
- `src/spyglass/utils/dj_merge_tables.py:587` `fetch_nwb`; per-source `_applicable_restriction` branch 742-758.
- `src/spyglass/spikesorting/analysis/v1/group.py:570` `get_spike_indicator` (NaN at 622-623). `src/spyglass/mua/v1/mua.py:74` `MuaEventsV1.make` (indicator → detector at 89-111).

Dependencies / CI (phases 1, 5):
- `pyproject.toml:55` networkx, `:59` jax<0.10, `:65` `numpy>=2,<3`, `:76` spikeinterface==0.104.3, `:109-118` moseq extras, `:119-141` spikesorting-v2 extra (comment at 120-121 is stale).
- `environments/environment_dlc.yml:26,32`, `environment_moseq_cpu.yml:26,32`, `environment_moseq_gpu.yml:26,32` (numpy>=2 with pytorch<1.12.0). `environments/environment.yml:39-43` (pip torch pattern).
- `.github/workflows/test-conda.yml:165-171` run-tests ignores; `:351-357` required fixtures; `:420-427` matcher fixture fetch with `|| true`; `:536` legacy sed rewriting the numpy line; `:615-617` legacy collection.
- `tests/spikesorting/v2/test_dependency_contract.py` guards the sed.
- `tests/spikesorting/v2/fixtures/_fetch.py:93-94` two-session fixture URLs (None).
- `tests/decoding/conftest.py:286-347` direct-insert `pop_unitwave` with `_StubWaveforms`.
- `tests/spikesorting/test_legacy_modern_si_coexistence.py:115-148` monkeypatched read-path test.
- `tests/spikesorting/v2/scripts/audit_lifecycle.py`, `audit_handoff.py` (acceptance probes, not collected).

Docs (phase 6): `README.md:167,172,181`; `CHANGELOG.md` `[Unreleased]` (contradictions listed in the appendix); `TODO.md`; `tests/spikesorting/v2/test_v1_parity.py:417` leakage regex.

## Scope and dependency policy

### Goals

- Statistics that drive detection thresholds and metrics (whitening covariance, noise levels, nn noise cluster) are estimated from unmasked samples only.
- Temporal filtering sees continuous data; interval restriction never creates filter transients.
- Cross-session matching compares per-unit temporal halves, never zero templates.
- Auto-curation distinguishes "metric not computable for this unit" from "metric computation failed"; the latter always raises.
- Observed-time intervals are derived from actual timestamps and validated as sorted/disjoint.
- Real Frank-lab tetrode files (x-z contact geometry) sort with four distinct contact positions and never crash at analyzer build.
- DLC and MoSeq users can install a working spyglass again; v0/v1 read paths work under SI 0.104 for binary-folder artifacts.
- CI proves the above with real artifacts, not stubs.

### Non-Goals

- The Important/Suggestion backlog in the appendix (cache-loader exception narrowing, recompute fingerprint churn, review-UI poll recovery, preset waveform-window fallback, type Literals, etc.) is NOT in this plan. Revisit trigger: after phases 1-4 merge, open a follow-up plan from the appendix's "Important" list, highest first: review-UI failed-poll freeze, `Merge.fetch_nwb` strictness, `clone_pipeline_preset` waveform window, cache-loader `except Exception`.
- Motion correction, new sorters, schema redesign beyond the two column-type changes in phase 4b.
- Hosting the two-session UnitMatch fixtures (external Box upload; phase 5 wires the CI once they exist).

### Dependency policy

Decision (see Open Question 1): the base `numpy>=2,<3` pin moves to the `spikesorting-v2` (and `spikesorting-v2-matching`) extras; base becomes `numpy>=1.26,<3` so the `dlc` and `moseq-*` extras resolve to current DeepLabCut 3.x / keypoint-moseq 0.6.x (both require numpy<2). `scipy>=1.13` is declared explicitly. `deeplabcut>=3.0` and `keypoint-moseq>=0.6` floors prevent silent backtracking. No new dependencies are introduced anywhere in the plan.

## Cross-cutting test principle

The defects this plan fixes share one shape: arrays keep the right dimensions, calls succeed, files reload, but meaning is lost at a boundary — which samples were observed, which frames are continuous in acquisition time, which unit an integer names, which units a value carries, whether a NaN is "not applicable" or "failed". The existing tests missed them because their fixtures were non-discriminating: dense ids, zero offsets, regular clocks, continuous recordings, units present throughout. Every phase's validation slice therefore includes fixtures where competing interpretations give different answers (sparse/reordered ids, x-z geometry, unequal gains and offsets, drifted clocks, disjoint intervals, drift-in/drift-out units), and the five reusable test kinds are placed in the phases that own them: validity/continuity properties (3b), semantic round trips (3a), state transitions (2, annotations), failure injection (2, 3b, 4b), and tests-of-the-tests plus execution evidence (5).

## Tracked follow-ups (in scope for spikesorting-v2, not merge-blocking)

- Specific-reference subtraction with unequal channel offsets and no bandpass loses calibration before the writer's guard (`_recording_preprocessing.py:184-218`); visible as a strict `xfail` in phase 3a's round-trip test.
- Cache-loader `except Exception` → `rmtree` on transient I/O errors (`_sorting_analyzer.py:199-227`, `_curation_analyzer.py:477-481`): a transient read failure must not be indistinguishable from a corrupt cache.
- `Merge.fetch_nwb` partial-source drop is warned (phase 2), not raised; revisit once `get_spike_times` callers can declare single-source intent.
- Seeding the sorter's own `seed` from `effective_random_seed`; `threshold_unit` vs explicit `noise_levels` contract; dead-channel inertness under `proportion_above_threshold=1.0`.

## Metrics

- Phase 3b: on a synthetic recording with 30% of samples masked, `noise_levels` from the analyzer match the unmasked recording's within 2% (currently −25%); whitened valid-sample std is 1.00 ± 0.02 (currently 1.16).
- Phase 3a: RMS of a 1.5 ms restricted sliver after a 600 Hz high-pass matches the same sliver filtered continuously to within 1% (currently 3.8×).
- Phase 4a: on the 10-seed experiment in the appendix, pooled drift-out recall ≥ 0.80 (currently 0/50), S×S false-pair rate ≤ 0.015, and paired healthy-unit recall drop ≤ 0.04 against control-fixed on the same 150 units (observed 0.033).
- Phase 3b: the 990 Hz-under-1000 Hz probe excludes 0 samples (currently 10); the MUA probe returns 1 event with a NaN gap adjacent to the event (currently 31-35).
- Phase 1: `uv pip compile --extra dlc` resolves deeplabcut ≥ 3.0 and `import deeplabcut` succeeds in a throwaway venv; `conda create --dry-run` succeeds for all three env files; `black --check .` is clean.
- Phase 3a: the real file `tests/_data/raw/minirec20230622.nwb` (x-z geometry) yields four distinct 2D contact positions after reload and `get_probe()` succeeds.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Filtering before time restriction changes every existing v2 recording artifact's bytes (content hash) | v2 is pre-production; document the recreation step in the CHANGELOG's preproduction upgrade sequence. Assert timestamp override and channel ids are unchanged; only trace values near interval edges differ. |
| Valid-sample noise injection relies on SpikeInterface's recording-property cache (`noise_level_{method}_{raw,scaled}`) | Phase 3b asserts the analyzer's stored `noise_levels` equal the injected values, and falls back to writing the extension data directly if the property route is not honored. |
| Valid frame ranges are recovered from the recording object instead of carried as data | Forbidden by design: SI stores silenced periods under `_kwargs["periods"]` as a vector, and concat artifacts persist masked traces with no wrapper. Ranges are threaded through the sort and persisted in provenance; phase 3b tests the concat reload path explicitly. |
| Statistics pieces straddle an artificial join (selection interval, concat member, or artifact edge) | Statistics spans are the intersection of timestamp-gap/member boundary spans with artifact-free ranges, computed even when nothing is masked; `sample_span_data` draws within one span at a time with sample-count weighting and no fixed chunk requirement; tested on an unmasked two-interval recording. |
| A fixed 500 ms chunk requirement rejects recordings whose valid spans are short | No chunk-length requirement for covariance/MAD (pieces are capped by span length); only nn snippets keep a per-snippet length rule. Tested with 400 ms spans. |
| Test targets contaminated by planted artifacts | Statistics targets come from the clean pre-injection twin or exactly from retained samples. |
| Per-metric missingness rules drift from SpikeInterface's actual NaN conditions | Rules are derived from SI 0.104.3 source per metric and tested against real SI output on fixtures that hit each condition; unclassifiable NaNs are attributed via SI's own warnings. |
| MUA detection on disjoint observed runs changes thresholds versus a whole-session z-score | Rate and z-score are computed once over all observed samples; only event extraction runs per run; a single-run case must equal the current detector exactly. |
| Relaxing the base numpy pin lets a v2 user install numpy 1.x without the extra | The `spikesorting-v2` extra pins numpy>=2; `test_dependency_contract` asserts the extra carries the pin and preflight already checks SI/numpy versions. |
| Per-unit halves exclude units with < 2 sampled spikes from the bundle | Excluded ids are logged and returned; they remain in the frozen matchable universe and simply produce no pairs (same outcome as today, but explicit). |
| Raising on unexpected-NaN metrics turns previously "successful" evaluations into failures | That is the intended change; the classifier treats `n_spikes < min_spikes` / `fr < min_fr` as expected, so only genuine computation failures raise. |
| Persisting patched geometry via h5py write-back after the pynwb write | Write-back happens before the content hash is computed; a read-back assertion in the same function verifies the values landed. |

## Rollout Strategy

Phases 1-5 and the required subset of phase 6 land on the `spikesorting-v2` branch before PR #1609 merges; phase 6's optional items and the deferred environment reorganization may follow. No feature flags. v2 users must recreate the preproduction database and analyzer caches after phases 3a/3b/4b (documented in CHANGELOG). v0/v1 users see no behavior change except the restored waveform reads (phase 2).

## Open Questions

1. **Base numpy pin.** DECIDED by the owner on 2026-09-18: relax base to `numpy>=1.26,<3` and pin `numpy>=2,<3` in the v2 extras (phase 1 as written). Alternative B (keep the pin, declare DLC/MoSeq unsupported) is not taken.
2. **Observed-interval end for the last sample.** Current best answer: `t_last + 1/fs` using the recording's declared rate (phase 3b design). Alternative: `t_last + median(diff)`; deferred unless drift > 1% is observed in production files.
3. **Legacy waveform reads for Zarr-format WaveformExtractors.** Current best answer: keep them gated with a message naming Zarr (SI 0.104 raises NotImplementedError). No known Frank-lab v0 Zarr waveform folders; deferred.

## Estimated Effort

Roughly +1,900 / −450 LOC across the six phases: phase 1 ~120 (config + tests), phase 2 ~350, phase 3a ~400, phase 3b ~450, phase 4a ~200, phase 4b ~250, phase 5 ~400 (fixtures + tests), phase 6 ~−300 net (deletions and rewrites). Tests are roughly half of the additions.
