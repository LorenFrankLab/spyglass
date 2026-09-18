# Phase 3b — Sorting/analyzer stage: statistics from valid samples; observed-time endpoints

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#valid-sample-statistics)

**Inputs to read first:**

- `src/spyglass/spikesorting/v2/sorting.py:1681-1830` — `make_compute`: mask (1768-1776), seed/provenance (1794-1817), sort + excess removal (1819-1826); `_apply_artifact_mask` (2789), `_run_sorter` (2809), `_remove_excess_spikes` (2895), `_build_analyzer` (2905); `source_provenance` dict (1877-1892) and `_stage_sorting_artifact`.
- `src/spyglass/spikesorting/v2/_nwb_provenance.py` — the scratch-table provenance writers (where the statistics spans will be persisted).
- `src/spyglass/spikesorting/v2/_concat_recording.py:650-700` — `build_concatenated_recording` uses `concatenate_recordings(..., ignore_times=True)` (line 695): member-internal gaps must be captured per member BEFORE concatenation and offset (design, boundary spans).
- `src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:27` (`artifact_frame_ranges`), `:292` (`apply_artifact_mask`), `:305-340` (`silence_frame_ranges`).
- `src/spyglass/spikesorting/v2/session_group.py:1199-1265` — concat masking (`mask_member_recordings` at 1220, `silence_frame_ranges` at 1261); the concat artifact persists masked traces, so reload carries no wrapper.
- `src/spyglass/spikesorting/v2/_sorting_dispatch.py:334-362` (`pinned_whiten`), `:597-760` (`run_si_sorter`; whiten at 714-726; docstring claim at 618-621), `:386-560` (`run_clusterless_thresholder`; `_clusterless_noise_levels` at 516), `:835` (`remove_excess_spikes`).
- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:673-1000` — `build_analyzer` (whiten branch 866-874; `create_sorting_analyzer` 880-895; `noise_levels` params 939-951). Other builders that must receive the ranges: `_curation_analyzer.py` (curation-scoped analyzers), `metric_curation.py:~799` (metric analyzer), `recompute.py:1251-1310` (rebuild path).
- `src/spyglass/spikesorting/v2/_si_metric_patches.py:38-120` — `_nn_noise_overlap_sparse_fixed` (noise chunks at 96-103); `metric_curation.py:2237-2260` where the patch is installed.
- SpikeInterface 0.104.3: `preprocessing/whiten.py:60-122,151-220` (`W`/`M` kwargs; `compute_whitening_matrix` internals to mirror), `preprocessing/silence_periods.py:115` (`_kwargs["periods"]`), `core/recording_tools.py:461-560,687-760` (`get_random_data_chunks`, `get_noise_levels` property cache keys), `core/analyzer_extension_core.py:762-800` (`ComputeNoiseLevels`), `core/basesorting.py:563` (`remove_empty_units`).
- `src/spyglass/spikesorting/v2/_observed_time.py:47-77` (`observed_intervals`), `:79-86` (`contains_times`), `:136-175` (`ObservationAvailability`).
- `src/spyglass/spikesorting/v2/_signal_math.py:201-221` (`intersect_intervals`), `:593` (`_segment_times_at`).

**Designs referenced:** [designs.md#valid-sample-statistics](designs.md#valid-sample-statistics).

## Tasks

- **Baseline capture**: build a synthetic 16-channel, 60 s CLEAN recording (`generate_ground_truth_recording`, seed 0) and record its `pinned_whiten` matrix `W`, MAD noise levels, and per-unit SNR — these are the targets. Then inject 800 µV transients masking ~30% of samples, and record today's `W`, analyzer `noise_levels`, and SNR on the masked recording (the biased values to beat). Never use the unmasked recording *with* artifacts as the target: its MAD is inflated by the artifacts themselves (observed 1.83 vs 0.995 on retained clean samples). Where a clean twin is unavailable (real fixtures), the target is statistics computed exactly on the retained samples.
- **Statistics spans as data**: `complement_frame_ranges`, `boundary_spans_from_timestamps`, `concat_boundary_spans`, `statistics_spans`, `sample_span_data` (exact length-proportional quotas), and the fixed-length snippet sampler in `_sorting_artifact_mask.py` per the design; `apply_artifact_mask` returns `(masked_recording, excluded_ranges)`. Boundary spans: single-session sources use `boundary_spans_from_timestamps(recording)` on the reloaded artifact (its persisted timestamps carry selection gaps); concatenated sources use the offset per-member span list that `ConcatenatedRecording.make` computes from each member's own artifact before `concatenate_recordings(..., ignore_times=True)` and persists next to the member set. `Sorting.make_compute` then computes `statistics_spans(n_samples, excluded_ranges, boundary_spans)` so selection joins and member joins are boundaries even when nothing is masked. Thread the spans through `_run_sorter` → `run_si_sorter` / `run_clusterless_thresholder`, and → `_build_analyzer` → `build_analyzer`. Persist the spans in the sorting artifact's provenance scratch and expose `Sorting.get_statistics_spans(key)`; every later analyzer builder (`_curation_analyzer`, `metric_curation`, `recompute`) obtains spans from that accessor, never from the recording object. Delete any code that inspects `SilencedPeriodsRecording` internals.
- **Whitening from within-span samples**: `pinned_whiten(..., spans=...)` per the design (`sample_span_data` with sample-count weighting, no fixed chunk requirement), mirroring `compute_whitening_matrix`'s `mode="global"` math; unmasked single-span path unchanged. Pass the spans at both call sites (`_sorting_dispatch.py:726`, `_sorting_analyzer.py:872-874`). Assert in a test that with `spans=None` `W` is bit-identical to the pre-change `W`.
- **Noise levels from within-span samples**: MAD per channel from `sample_span_data` cached under SpikeInterface's property keys immediately before `create_sorting_analyzer` in `build_analyzer` (on the recording actually handed to it) and before `_clusterless_noise_levels` (`:516`). Confirm by test that `analyzer.get_extension("noise_levels").get_data()` equals the cached array; if the extension does not consult the property, implement the documented fallback (write extension data directly) and keep the test.
- **nn noise cluster from within-span snippets**: `_si_metric_patches.py:96-103` → the fixed-length snippet sampler over spans supplied through a `ContextVar` set by `_compute_metrics` around the PC-metric compute (`metric_curation.py:2286-2300`).
- **Masked-fraction logging and docstring**: log the realized masked fraction at INFO in `apply_artifact_mask`; rewrite the docstring at `_sorting_dispatch.py:618-621` to describe the valid-sample statistics.
- **Non-finite traces fail loudly in the artifact worker** (owner's probe: an all-NaN chunk, and a single NaN propagated through filtering and median referencing to all 8,000 values of a chunk, both returned "no artifacts"). In `_artifact_compute.py:125-150`, before the threshold comparisons, `if not np.isfinite(traces_uv).all(): raise ValueError(...)` naming the chunk frame range and the count of non-finite samples per channel. The scan reads every sample already, so the check is nearly free. Test: one NaN sample in the raw fixture makes `RecordingArtifactDetection.populate` raise with the frame range in the message instead of inserting an empty interval list.
- **Zero-spike units**: in `remove_excess_spikes` (`_sorting_dispatch.py:835`) return `sic.remove_excess_spikes(sorting, recording).remove_empty_units()` and update its docstring; `Sorting._remove_excess_spikes` (2895) inherits it. `n_units` and the units NWB then exclude empty units.
- **Observed-interval endpoints** (`_observed_time.py:74-75`): replace `t + (end - first) / fs` with the data-derived end

  ```python
  t0 = float(_segment_times_at(recording, np.array([first]))[0])
  t1 = float(_segment_times_at(recording, np.array([end - 1]))[0]) + 1.0 / fs
  kept.append((t0, t1))
  ```

  and, before returning, assert the intervals are sorted by start and disjoint (`np.all(arr[1:, 0] >= arr[:-1, 1])`), raising `ValueError` with the offending pair otherwise.
- **Interval-set guard**: in `intersect_intervals` (`_signal_math.py:201`) add a private `_normalize(intervals)` that sorts by start, merges overlapping/adjacent/duplicate rows, and drops zero-length rows; apply to both operands before the identical-input fast path. In `ObservationAvailability` (`_observed_time.py:136`) add `__post_init__` that reshapes `intervals` to `(n, 2)`, rejects non-finite values, asserts sorted/disjoint, and stores a read-only array (`arr.setflags(write=False)`); `restrict()` must return shape `(0, 2)` (not `(0,)`) when `intervals is None` and the input is empty.
- **Comparison after the edits**: re-run the baseline script; `noise_levels` on the masked recording now within 2% of the CLEAN targets (was −25% at 30% masked); whitened within-span std 1.00 ± 0.02; report the SNR shift per unit. Repeat on (a) a two-interval restricted recording with no artifacts, asserting no sampled piece straddles the selection join (the review's reproduction found 23/200 crossing snippets with `[0, n)` ranges), and (b) a concatenated two-member source (build via `test_session_group_concat.py` fixtures) to prove the spans reach the concat analyzer without a recording wrapper and never cross a member boundary.
- **CHANGELOG** (`[Unreleased]` → Spike sorting v2): whitening, noise levels, and the nn-noise cluster are estimated from unmasked samples within valid spans (existing analyzer caches and evaluations must be recreated); valid frame ranges are persisted with each sort; units left with zero spikes after window trimming are dropped; observed intervals use recorded timestamps for their end and are validated as sorted/disjoint. Note the SNR values of previously computed evaluations are biased upward by roughly the masked fraction.

## Deliberately not in this phase

- Filter order / geometry (phase 3a).
- Seeding the sorter's own `seed` parameter from `effective_random_seed` (appendix Important; separate follow-up — it changes sort outputs for TDC2/SC2 rows).
- Dead-channel detection inertness under `proportion_above_threshold=1.0` (appendix Important). The trace finiteness guard is IN this phase (task above).
- `threshold_unit` / explicit `noise_levels` contract mismatch (appendix Important).
- Metric-missingness policy (phase 4b) even though `_si_metric_patches.py` is touched here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_artifact_mask.py::test_complement_frame_ranges` | for excluded `[(100,200),(500,700)]` on 1000 samples → `[(0,100),(200,500),(700,1000)]`; overlapping/unsorted excluded ranges are merged; empty → `[(0,1000)]` |
| `...::test_boundary_spans_from_timestamps` | a recording whose persisted timestamps have two gaps yields three spans; a continuous single-interval recording yields `[(0,n)]` |
| `...::test_concat_boundary_spans_offsets_members` | two members with 1 and 2 internal spans yield 3 spans offset by the members' cumulative starts |
| `...::test_statistics_spans_intersect_boundaries_and_artifacts` | with boundary spans `[(0,500),(500,1000)]` and excluded `[(450,550)]` → `[(0,450),(550,1000)]` (no span crosses 500) |
| `...::test_sample_span_data_never_crosses_a_span_boundary` | with spans `[(0,300),(600,1000)]`, every returned piece equals a contiguous slice of the parent inside one span; a piece is never longer than its span |
| `...::test_sample_span_data_uses_short_spans` | 8 s of valid data split into 400 ms spans inside a 10 s recording is sampled (no raise) and the MAD matches the clean value within 2% |
| `...::test_sample_span_data_weights_by_length` | a 900-sample span contributes exactly 9× the samples of a 100-sample span (quota apportionment) |
| `...::test_sample_span_data_budget_exceeds_valid_data` | requesting 10,000 samples from 8,000 valid samples returns exactly 8,000 rows, each valid sample once (compare against the concatenated spans) |
| `...::test_snippet_sampler_respects_length_per_snippet` | snippets of `nsamples` are drawn only from spans that admit them; a span of length `nsamples - 1` contributes none; raise only when no span admits one |
| `...::test_apply_artifact_mask_returns_ranges_matching_silenced_samples` | the returned excluded ranges are exactly the zeroed samples of the masked recording |
| `tests/spikesorting/v2/test_sorting_dispatch.py::test_pinned_whiten_unmasked_is_bit_identical_to_previous` | `W` equals the captured baseline exactly when `spans=None` |
| `...::test_whitened_valid_samples_have_unit_variance_under_masking` | 30% masked: std of whitened traces over valid samples in [0.98, 1.02] (baseline 1.16) |
| `tests/spikesorting/v2/test_sorting_analyzer.py::test_noise_levels_unbiased_by_masking` | analyzer `noise_levels` within 2% of the CLEAN recording's (pre-injection twin) for 5%, 30%, 50% masked |
| `...::test_estimates_invariant_to_excluded_sample_values` (validity property) | with statistics spans held fixed, overwrite ONLY the excluded samples with zeros, then with ±10 mV, then with NaN: `W`, `M`, `noise_levels`, and the nn noise-cluster draw are bit-identical across the three (excluded values cannot reach any estimator) |
| `tests/spikesorting/v2/test_artifact_detection.py::test_non_finite_trace_raises` | one NaN sample in the raw fixture → `populate` raises naming the frame range; no empty interval list is inserted |
| `...::test_no_piece_crosses_selection_join_when_unmasked` | on a two-interval restricted recording with no artifacts, 0 of 200 sampled pieces straddle the join |
| `...::test_noise_levels_extension_equals_cached_values` | extension data equals the cached property array exactly |
| `tests/spikesorting/v2/test_session_group_concat.py::test_concat_analyzer_receives_statistics_spans_after_reload` (`pytest.mark.slow`) | a concat sort's `Sorting.get_statistics_spans` is non-empty after reload, no span crosses a member boundary, and the metric analyzer's `noise_levels` matches the clean members within 2% |
| `...::test_concat_preserves_member_internal_gaps` | a member with internal spans `[0,500)` and `[500,1000)` yields two offset spans in the concat's persisted list (not one `[start, start+1000)`) |
| `tests/spikesorting/v2/test_artifact_mask.py::test_sample_span_data_exact_quotas_heterogeneous_noise` | with a 9 s σ=1 span and ten 100 ms σ=5 spans, returned rows == `target_samples`, each span's rows == its quota, pooled MAD within 2% of the length-weighted expectation |
| `tests/spikesorting/v2/single_session/test_recording.py::test_reloaded_two_interval_artifact_exposes_gap` | `boundary_spans_from_timestamps` on the reloaded artifact returns two spans at the persisted interval boundary |
| `tests/spikesorting/v2/test_metrics.py::test_nn_noise_cluster_excludes_masked_samples` | with 50% masked, no sampled noise clip is all-zero and none straddles a span boundary |
| `tests/spikesorting/v2/test_sorting_dispatch.py::test_remove_excess_spikes_drops_empty_units` | a unit whose only spikes fall outside the window is absent from the returned sorting |
| `tests/spikesorting/v2/test_observed_time.py::test_observed_intervals_end_from_timestamps` | 1000 samples at 990 Hz under a declared 1000 Hz: 0 samples excluded (currently 10); intervals sorted/disjoint |
| `...::test_intersect_intervals_normalizes_duplicates_and_order` | duplicate rows collapse; unsorted input gives the same result as sorted; `observed_duration_s` for `[[0,10],[0,10]]` is 10 |
| `...::test_observation_availability_rejects_unsorted` | `ObservationAvailability(intervals=[[5,6],[0,1]])` raises `ValueError` |
| `tests/spikesorting/v2/single_session/test_sorting.py::test_masked_sort_snr_matches_unmasked_within_tolerance` (`pytest.mark.slow`) | on the smoke fixture with planted artifacts (~10% masked), per-unit SNR within 5% of the unmasked sort's SNR for matched units |

## Fixtures

- Synthetic: `si.generate_ground_truth_recording` (16 ch, 60 s, 30 kHz, seed 0) plus planted 800 µV transients to drive `artifact_frame_ranges`; masked fraction parametrized by transient count.
- Smoke fixture `mearec_polymer_smoke` with `test_artifact_integration.py`'s transient injection helper; concat fixtures from `test_session_group_concat.py`.
- Timestamp-drift recording: `si.NumpyRecording` with `set_times(np.arange(n)/990)` under `sampling_frequency=1000` (the review's `probe_obs.py`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the nominal-rate endpoint arithmetic; the unguarded `intersect_intervals` fast path; any recording-introspection of silenced periods).
- User-facing documentation listed as tasks is updated, not deferred.
