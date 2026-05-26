# v1 ↔ v2 clusterless_thresholder spike-time divergence investigation

**Date opened**: 2026-05-26 (Phase A 8-case parity matrix triage)
**Investigator**: Claude (Opus 4.7) under research-only constraints
**Status**: investigation in progress (see Investigation Log)

---

## 1. Background & problem statement

We just landed Phase A of `parity-extensions.md`: an 8-case clusterless
parity matrix (`mearec_polymer_smoke` × 4 shanks + `mearec_polymer_128ch_60s`
× 4 shanks), comparing v1 (SI 0.99.1, Python 3.10, conda env
`spyglass-v1-parity`) against v2 (SI 0.104.3, Python 3.11, `.venv-spikesorting-v2`).
The seven invariant fingerprints (`nwb_sha256`, `sort_group_electrode_ids`,
`bad_channel_by_electrode_id`, `canonical_preproc_params`,
`canonical_artifact_params`, `artifact_valid_times_hash`,
`canonical_sorter_params`) pass on all 8 cases — i.e., the *inputs* are
provably identical bytes-and-config across the two stacks.

Six of eight cases nonetheless diverge at the spike-time level:

| Case | v1 spikes | v2 spikes | unmatched_v1 | unmatched_v2 | max drift |
|---|---|---|---|---|---|
| smoke-shank0 | 61 | (≈61, 0 unmatched) | 0 | 0 | PASS |
| smoke-shank1 | 52 | (≈52, 0 unmatched) | 0 | 0 | PASS |
| smoke-shank2 | 131 | ? | 1 | ? | 88 samples |
| smoke-shank3 | 1 (1.58 s active) | 0 | — | — | v2 EMPTY |
| 60s-shank0 | 2652 | 3448 | 23 | 796 | 1270 samples |
| 60s-shank1 | 2807 | ? | 15 | ? | 1642 samples |
| 60s-shank2 | 5970 | ? | 36 | ? | 11 samples |
| 60s-shank3 | 3432 | 4342 | 22 | 930 | 119 samples |

The plan's "Documented v2 divergences" table currently attributes this to
"`random_slices` / sample-window drift in `detect_peaks` across SI versions"
but flags that justification as **not yet cited**. The drift magnitudes
here (30 %-ish v2 surplus, max-drift up to 1642 samples ≈ 51 ms at 32 kHz)
are an order of magnitude larger than "a small number of extras on
adjacent contacts." Question: **regression, deliberate change, v1 bug, or
calibration of `random_slices` chunking noise?**

---

## 2. Hypothesis tree

Confidence calibration scale: 0 = ruled out, 100 = proven. Initial
estimates are working priors; update after each evidence collection.

### H1 — `noise_levels` is re-estimated by SI 0.104 with a different chunking strategy, so the per-channel MAD-µV threshold actually applied to each contact differs from v1's.

**Statement**: In SI 0.99, `get_noise_levels` used `n_chunks_per_segment=20`
and `chunk_size` derived from a fixed value (~ default 10000 samples). In
SI 0.104, the API was rewritten (`get_noise_levels` now accepts
`num_chunks_per_segment` and `chunk_size` with potentially different
defaults, or accepts `random_slices` precomputed externally). Since
neither v1 nor v2 passes an explicit `noise_levels` array (canonical sorter
params have no `noise_levels` key), both compute it on the fly. If the
RNG, chunk count, or default chunk_size differs, the per-channel µV-MAD
threshold differs, so peaks that v1 (with threshold ~5.0 × m_v1) and v2
(threshold ~5.0 × m_v2) detect will not match per-spike.

**Predictions if true**:
- v2-only "extras" cluster on channels where v2's MAD is lower than v1's
  (effective lower threshold).
- v1-unmatched spikes cluster on channels where v1's MAD is lower than
  v2's.
- The *shapes* of the missed/extra peaks should be near-threshold (small
  amplitude relative to local MAD).

**Counter-predictions if false**:
- The cross-channel distribution of extras is uniform / matches the
  detection neighborhood, not the noise-level ranking.
- Extras are not concentrated at amplitudes near the threshold band.

**Investigation steps**:
1. Run `get_noise_levels` from both SI versions on the same preprocessed
   recording slice; compare per-channel values.
2. Look at the per-channel histogram of v2-only extras vs. unmatched_v1.

**Confidence**: starts at 50 % (this is the plan's current best-guess).

### H2 — `random_chunks` / `random_slices` mechanism inside SI's chunk-based noise estimate uses different RNG semantics in v0.104 (or seeded differently) and the resulting MAD is noisy enough at 5 σ to flip thousands of borderline peaks.

**Statement**: Even if the *algorithm* for `noise_levels` is conceptually
the same, the random chunk selection in 0.104 may use a different
`seed`/`random_state` default (or default to `seed=None`), so noise_level
sampling is stochastic. At 5 σ on a 60 s × 32 channels recording with
~5–10 k border peaks, even a ±5 % variation in per-channel MAD can flip
hundreds of detections. The "smoke" shanks pass because there are so few
peaks (61–131) that the borderline band catches nothing.

**Predictions if true**:
- v2's run twice (same args, different RNG state) would give different
  spike counts → confirmable by running v2 detection twice on the SAME
  preprocessed recording and checking determinism.
- The number of extras should scale with the channel count × duration
  (border-peak rate), not with the planted-unit rate.

**Counter-predictions if false**:
- v2 reruns produce identical spike counts ⇒ pure deterministic
  divergence ⇒ rules out RNG-as-cause.

**Investigation steps**:
1. Inspect the signatures of `get_noise_levels` in 0.99 vs. 0.104; check
   whether `seed` is parameterized and what its default is.
2. Repeat-run v2 detection on shank0 of the 60 s fixture, with same
   args; record `(n_spikes, first_spike, last_spike)` between runs.

**Confidence**: starts at 30 %.

### H3 — The bandpass filter implementation (Butterworth order / boundary handling / `filtfilt` semantics) changed between SI 0.99 and 0.104, so the v2 preprocessed trace itself differs from v1's *before any peak detection*. Detect_peaks is then operating on a different signal.

**Statement**: A SciPy/Butterworth filter applied twice with different
`filter_order` or different boundary-padding (`padlen`) defaults will
shift the phase of each spike by a deterministic-but-version-dependent
sample count. If v2 uses `filter_order=3` (default in many SI recipes)
and v1 used `filter_order=5` (or vice versa), peaks could shift by tens
of samples — explaining the ~1300-1600-sample drifts on 60s-shank0/1.

**Predictions if true**:
- The v2-only extras should look like *time-shifted* v1 peaks rather
  than new peaks.
- Per-channel filter-output differences would be observable directly:
  load the same raw NWB slice, apply v1 preproc and v2 preproc, diff
  the traces.
- Smoke shanks pass because their few spikes are well-separated; on
  dense traces, filter-phase drift creates many doubled-up peaks.

**Counter-predictions if false**:
- Direct trace-diff between v1- and v2-preprocessed slices is below
  numerical noise (1e-6).

**Investigation steps**:
1. Read `bandpass_filter` defaults from v1 and v2 SI source.
2. Compute the filtered output of one 5 s slice under each version and
   diff.

**Confidence**: starts at 25 %.

### H4 — The `locally_exclusive` spatial exclusion algorithm changed between v0.99 and v0.104: e.g., a different definition of the channel-neighborhood radius or a different tie-breaker on the spatial-maxima selection. v2 admits peaks that v1's spatial exclusion would have suppressed.

**Statement**: Both pipelines use `method="locally_exclusive",
local_radius_um=100.0`. If SI 0.104's exclusion now uses a slightly
different neighborhood graph (e.g., `>=` vs. `>` on distance) or runs
the channel-wise maxima check with a different stride, the v2 set ⊇ the
v1 set on average (or vice versa).

**Predictions if true**:
- v2-only extras cluster spatially adjacent to v1 peaks (within
  `local_radius_um`) at the same time index ±1 sample.
- Unmatched_v2 ≫ unmatched_v1 (one v1 peak ↔ multiple v2 peaks on
  adjacent contacts) — **this matches the observed asymmetry** (60s
  shanks have 23+15+36+22=96 unmatched_v1 but 796+930=1726 unmatched_v2
  observed so far).
- The "extras" should be at time indices that exactly match a v1 peak
  (within ±1.5 samples) but on a different channel.

**Counter-predictions if false**:
- Extras occur at times no v1 channel detected anything (genuinely new
  peaks).
- Extras occur on the SAME channel as v1 (then it's not a spatial
  exclusion difference).

**Investigation steps**:
1. For 60s-shank0, count how many v2-only extras are within ±1.5 samples
   of a v1 spike (any channel within radius). If most are, this is H4.
2. Diff the `locally_exclusive.py` between SI 0.99 (in
   `peak_detection.py`) and SI 0.104 (split-out module).

**Confidence**: starts at 60 % (asymmetry of unmatched counts is
*already* informative).

### H5 — v2 doesn't write per-channel detection info into the saved spike-times pickle; the v1 baseline saves *unique* (time, channel) tuples while v2 collapses (or vice versa). I.e., the divergence is a downstream representation issue, not a detect_peaks behavior change.

**Statement**: The pickle stores per-unit `spike_times` (s). For
clusterless, "unit" is a synthetic per-channel container in v1 but might
be aggregated differently in v2. If v1 and v2 disagree on what counts as
a "unit" or a "spike," counts diverge even when detect_peaks output is
identical.

**Predictions if true**:
- The total *peak-detection* output (concatenated across all channels)
  would match between v1 and v2 within ±1.5 samples even if per-unit
  counts don't.
- The asymmetry would be ~uniform across shanks (representation effect,
  not data-dependent).

**Counter-predictions if false**:
- v1 capture's unit-id == v2 unit-id semantics (1 unit = 1 channel) AND
  the underlying detect_peaks output still differs.

**Investigation steps**:
1. Read both v1 and v2 Spyglass `sorting.py` to confirm that
   `clusterless_thresholder` writes a synthetic 1-unit container with
   ALL channel peaks merged (suspected). If so, v2-side aggregation is
   ruled in/out by spot-checking the pickle structure.

**Confidence**: starts at 25 %.

---

## 3. Investigation log

### Round 0 — orientation
- Confirmed baseline file structure: 8 pickles + 8 meta JSONs at
  `/cumulus/edeno/spikesorting-v2-baselines/parity-matrix/<fixture>/clusterless/shank<N>/`.
- Confirmed env / SI versions: spyglass-v1-parity → SI 0.99.1; v2 → SI
  0.104.3.
- Confirmed v1 baselines all have exactly **1 unit per shank**, with
  spike-counts (61, 52, 131, 1, 2652, 2807, 5970, 3432). This matches
  H5's *prediction*: v1's clusterless_thresholder treats the whole sort
  group as one synthetic "unit" containing all detected peaks. So
  *per-unit* count == *total peak* count.
- Sampling rate is 32000 Hz (so 1500 samples ≈ 47 ms).

### Round 1 — source-level diff of `detect_peaks` and `get_noise_levels`

**Major API rewrite in SI 0.104**. Key differences:

| Aspect | v1 (SI 0.99.1) | v2 (SI 0.104.3) |
|---|---|---|
| `get_noise_levels` chunk fn | `get_random_data_chunks(recording, **random_chunk_kwargs)` | `get_random_recording_slices(recording, **random_slices_kwargs)` |
| Default `chunk_size` | **10 000 samples** (= 312.5 ms @ 32 kHz) | **`chunk_duration="500ms"`** → **16 000 samples** @ 32 kHz |
| Default `num_chunks_per_segment` | 20 | 20 (same) |
| Default `seed` | **`seed=0`** (deterministic!) | **`seed=None`** (stochastic, different chunks each call) |
| MAD estimator | concat all chunks → single MAD per channel: `median(\|x − median(x)\|) / 0.6745` | per-chunk MAD per channel, then **mean over chunks** |
| `rng.integers(...)` | `endpoint=False` (legacy) | `endpoint=True` + `np.sort(random_starts)` |

Sources:
- v1 `get_noise_levels` at `peak_detection.py` → `recording_tools.py:132-184`
- v1 `get_random_data_chunks` at `recording_tools.py:8-82`
- v2 `get_noise_levels` at `recording_tools.py:687-792`
- v2 `get_random_recording_slices` at `recording_tools.py:461-543`

**Locally-exclusive numba kernel rewritten as well** (different algorithm, NOT just parameter rename):

| Aspect | v1 (SI 0.99.1) | v2 (SI 0.104.3) |
|---|---|---|
| Neighbour mask | `channel_distance < radius_um` (strict <) | `channel_distance <= radius_um` (≤) |
| Peak-mask building | single nested loop: candidate compared simultaneously to all spatial neighbours over `±exclude_sweep_size` samples; raw-amplitude comparison | two-stage: (1) per-channel local maxima with 1-sample comparison; (2) exclusion loop over candidate pairs within ±exclude_sweep_size, **comparing `\|amp\| / abs_threshold` ratio** (not raw amp). Tie-break by sample index. |
| Selection criterion when neighbours disagree | "largest raw µV" | "highest threshold-units significance" (v2 normalizes by per-channel noise) |

Sources:
- v1 `_numba_detect_peak_neg` at `peak_detection.py:677-700`; class
  `DetectPeakLocallyExclusive` at `peak_detection.py:507-580`.
- v2 `detect_peaks_numba_locally_exclusive_on_chunk` at
  `peak_detection/locally_exclusive.py:108-188`; class
  `LocallyExclusivePeakDetector` at lines 35-105.

**Spyglass wrapping**:
- v1 (`spyglass/spikesorting/v1/sorting.py:274-290`): `detect_peaks(recording, **sorter_params)` with `outputs`, `tempdir`, `whiten` popped; renames `local_radius_um → radius_um`. Does NOT pass any `random_chunk_kwargs` → v1 SI uses its **`seed=0` default** ⇒ v1 noise_levels are deterministic on a given preprocessed recording.
- v2 (`spyglass/spikesorting/v2/sorting.py:1055-1129`): the same Spyglass-side renames, but new SI signature forces `method/method_kwargs/job_kwargs` separation. Does NOT pass `random_slices_kwargs` either → v2 SI uses its **`seed=None` default** ⇒ v2 noise_levels are **stochastic between calls**.

**Smoke param row** (`tests/spikesorting/v2/_smoke_constants.py:37-43`): omits
`noise_levels`, so both v1 and v2 self-estimate it — but via the different
algorithms above. v1 baseline has `canonical_sorter_params = {detect_threshold:
5.0, exclude_sweep_ms: 0.1, local_radius_um: 100.0, method: "locally_exclusive",
peak_sign: "neg"}` only.

**Combined effect**: even when the input bytes are identical, v1's MAD-µV
threshold per channel is **deterministically lower or higher** than v2's
threshold by amounts that depend on (a) different chunk sizes (10000 vs 16000),
(b) the bias of per-chunk-MAD-then-mean vs concatenated-MAD, (c) v2's
non-deterministic random chunk picks. At a 5 σ detect_threshold, a 1-3 %
difference in `noise_levels[chan]` can flip thousands of borderline peaks.

This **discriminates against H3 (filter implementation)** because the
preprocessing chain (`bandpass_filter` + CMR) is unchanged structurally —
we'd have to verify filter parameters, but the canonical_preproc_params
match. Moves H1 (noise_levels divergence) and H2 (RNG non-determinism)
toward **strong** confidence; weakens H4 to "secondary contributor" because
the numba algorithm change still applies but its *immediate* output depends
on `abs_thresholds`, which is itself a function of noise_levels.

Updated confidences:
- H1 (noise_levels algorithm differs): **75 %** (large structural change,
  matches asymmetry).
- H2 (v2 RNG default seed=None makes v2 stochastic): **75 %** (confirmed
  by source diff; experiment needed to confirm magnitude).
- H3 (filter changed): **10 %** (no structural evidence, and would not
  show 30 % count asymmetry).
- H4 (locally_exclusive algo differs): **55 %** (algo DOES differ — `<` vs
  `<=`, normalized-vs-raw amplitude comparison — but H1 + H2 likely
  explain most of the count asymmetry. Co-cause.).
- H5 (representation): **5 %** (ruled out by inspection of v1's
  one-unit-per-shank container; v1's pipeline rolls all peaks into a
  single synthetic unit; v2 spike counts in diagnostic table come from
  the same path).

### Round 2 — synthetic-noise probe (`/tmp/divergence_probe_v{1,2}.py`)

Synthetic recording: 60 s × 32 kHz × 32 ch, Gaussian noise with per-channel
σ ∈ [0.7, 1.4]. Same seed across env (numpy `default_rng(42)`).

Results (per `/tmp/probe_v{1,2}_noise.json`):
- **v1**: `nl_a == nl_b` exactly (determinism confirmed; `seed=0` default).
- **v2**: `nl_a ≠ nl_b ≠ nl_c` (three calls, three answers). Confirms
  H2: v2's `seed=None` default makes `get_noise_levels` non-deterministic.
- **v1 vs v2 mean per-channel diff**: 0.092 % (max 0.84 %). v2-to-v2
  call-to-call diff: 0.06 % (max 0.6 %).

So on pure-noise data the two methods agree closely, with v2 noisier by
a factor of ~3-10×. At a 5 σ threshold a 1 % noise_level shift moves the
detection band by 1 % — small but non-zero.

### Round 3 — real-data probe on `mearec_polymer_128ch_60s` shank 0

Same preprocessing pipeline applied independently in each env (raw NWB
read → CMR median → bandpass 300-6000 Hz → channel-slice to shank 0).
Calls match each Spyglass pipeline's preprocessing.

**v1 probe** (`/tmp/real_data_probe_v1.py` → `/tmp/probe_v1_real.json`):
- `get_noise_levels`: deterministic, same answer twice.
- `detect_peaks`: 2657 peaks (deterministic; second run identical).
  Baseline pickle has 2652 — 0.2 % difference attributable to the
  preprocessing not exactly matching Spyglass (e.g., I didn't replicate
  the `IntervalList.valid_times` slicing; the recording I built spans
  the full 60 s rather than what Spyglass passes after artifact
  consolidation). Negligible.

**v2 probe** (`/tmp/real_data_probe_v2.py` → `/tmp/probe_v2_real.json`):
- `get_noise_levels`: 3 calls, **3 different answers**. Across fresh
  recording objects (separate variable), v2 noise_levels diverge by
  **up to 1.5 % per channel, mean 0.6 %** (script
  `check_noise_stochasticity` snippet inline).
- `detect_peaks`: 3392 peaks per run (3 runs all identical, because SI
  caches `noise_levels` as a property on the recording instance after
  the first call. Across fresh recording instances the count would
  jitter.)
- v1 (2657) vs v2 (3392): **+27.7 % surplus in v2**.

**Critical control experiment** (`/tmp/v2_with_v1_noise.py`):
Fed v2's `detect_peaks` with v1's deterministic noise_levels array.
Result: **3416 peaks** (same noise_levels, v2 still gets ~30 % more
peaks). So the surplus is NOT primarily a noise_levels effect; it's
the locally_exclusive algorithm change. H1+H2 → demoted; H4 → promoted.

**Pairwise peak matching** (`/tmp/compare_peaks.py`):

| Comparison | v1 count | v2 count | unmatched_v1 (±1) | unmatched_v2 (±1) |
|---|---|---|---|---|
| v1 vs v2-default | 2657 | 3392 | 23 | 757 |
| v1 vs v2-with-v1-noise | 2657 | 3416 | 6 | 764 |

So passing v1's noise_levels into v2 reduces unmatched_v1 (the
"v1 spikes v2 missed") from 23 → 6, but barely changes unmatched_v2
(the "v2 surplus"): 757 → 764. The two phenomena are decoupled.

**Tolerance sweep on unmatched_v2** (`/tmp/check_h4_widen.py`):
| Tolerance (samples) | v2 → v1 match rate | unmatched_v2 |
|---|---|---|
| ±1 | 77.68 % | 757 |
| ±5 | 88.24 % | 399 |
| ±10 | 98.11 % | 64 |
| ±30 (~0.94 ms) | **100 %** | **0** |

**Every v2 surplus peak is within 30 samples (≈ 0.94 ms) of a v1 peak.**
So they're not new spikes; they're nearby-in-time peaks. Amplitude
distribution of unmatched-v2: median \|amp\| = 33 (vs 56 for matched-to-v1
peaks) — they sit near the 5 σ threshold band but well above it.

**Spatial analysis** (`/tmp/check_h4_spatial.py`):
For each unmatched_v2 peak, find nearest v1 peak in time and the
spatial distance to its channel.
- Mean Δt = 6.4 samples (≈ 200 µs at 32 kHz); median 6 samples.
- 99.6 % within `radius_um = 100` of nearest v1 peak's channel.
- Channel-distance histogram: 66 within 15 µm (same row neighbour),
  409 at 15-30 µm (one contact away), 279 at 50-100 µm (two contacts
  away).

**Interpretation**: the v2 surplus is **adjacent-contact echoes of the
same physical spike**, delayed by 3-15 samples (the spike's
spatial-propagation time across the polymer probe row) and 15-100 µm
away. These are *real* spike-related peaks. v1's
`DetectPeakLocallyExclusive` numba kernel suppressed them via the
sample-by-sample neighbour-comparison
(`traces_center[s, chan] <= traces[s + i, neighbour]` for
`i ∈ [-exclude_sweep_size, +exclude_sweep_size]`) — even when the
neighbour's *peak* was outside the sweep window, a single sample
where the neighbour's trace dipped below the candidate would
suppress the candidate.

### Self-critique round 1 — what am I missing? What could invalidate H4?

**Possible objections to "H4 is the main cause"**:

1. **Did I match v1's preprocessing exactly?** No — I bypassed Spyglass's
   sort-time `remove_artifacts` step (artifact preset `"none"` →
   `detect=False`, so artifact_valid_times is `[(0.0, 60.0)]` ⇒ NO
   artifact zeroing. So skipping the step in my probe was correct.
   Confirmed via the meta JSON's `artifact_valid_times` field.
   ⇒ Preprocessing is genuinely identical.

2. **The 5-spike difference (2657 vs 2652) — what causes it?** Most
   likely the probe uses `load_time_vector=False` whereas Spyglass uses
   `True`, plus `IntervalList.valid_times` boundary trimming. Doesn't
   affect the *count direction* (v2 over v1) and is well below the
   757-peak gap. Not a confound.

3. **Could the v2 surplus actually be a real spike v1 missed?** If the
   adjacent-contact peak is from a *different cell* than the one v1
   detected, then v2 is gaining genuine extra units. But the spatial
   analysis shows 99.6 % of unmatched_v2 peaks are within
   `radius_um=100` and within 0.5 ms of a v1 peak. For two
   independently-firing cells to coincide that precisely on adjacent
   contacts ~1000× across a 60 s recording is implausible — the firing
   rate would need to be much higher than observed (2657 spikes / 60 s
   = 44 Hz total across all channels in shank 0 means ~1.4 Hz per
   channel; coincidence rate 1.4 × 1.4 / 1000 ≪ 1/min). So the
   "duplicates" are essentially all multi-contact representations of
   the same spikes.
   - **Counter-argument I should grant**: a fraction of the
     "duplicates" *could* be legitimate near-simultaneous spikes from
     different cells (e.g., burst pair, common-input-driven pair). This
     is the standard concern with `locally_exclusive` parameter tuning;
     the user choice of `radius_um=100` and `exclude_sweep_ms=0.1`
     determines what counts as "the same spike." For the parity test,
     the question is just "does v1's deduplication match v2's," not
     "is one of them more biologically correct." Both are imperfect
     approximations.

4. **Could the v2 surplus be on the LARGE-AMPLITUDE end?** No — the
   amplitude analysis shows unmatched_v2 are *near-threshold*
   (median |amp| = 33 vs 56 for matched). They're the marginal
   detections, exactly what you'd expect from a "less aggressive
   spatial exclusion" hypothesis. v1 has the bigger amp range.

5. **Did I confirm v1's PR text actually applies to SI 0.99.1 → 0.104.3
   delta?** PR #3359 merged 2024-10-25 (between 0.101.0 and 0.102.0
   releases roughly; need to verify). PR #4341 merged 2026-01-29
   (after 0.104.0 / current 0.104.3). Need to check whether #4341's
   change actually shipped in 0.104.3.
   **Verified**: The PR #4341 commit `2fe55d8` is in v2 venv —
   `/cumulus/edeno/spyglass/.venv-spikesorting-v2/lib/python3.11/site-packages/spikeinterface/sortingcomponents/peak_detection/locally_exclusive.py`
   matches the post-PR-#4341 algorithm (two-stage with
   ratio-to-threshold comparison). v1 SI 0.99.1's
   `peak_detection.py` predates PR #4341 (older single-stage
   raw-amplitude algorithm).

6. **Could there be a Spyglass-level fix on the v2 side suppressing
   peaks?** Looking at v2's `_run_clusterless_thresholder`
   (sorting.py:1055-1129): it just unwraps params, renames
   `local_radius_um → radius_um`, drops `noise_levels` if None, then
   calls `detect_peaks`. No Spyglass-side filtering on the v2 output;
   v2 simply returns what SI returns.
   ⇒ No Spyglass-level confound.

7. **What about the smoke-shank3 v2 EMPTY case (1 v1 spike → 0 v2
   spikes)?** That's the OPPOSITE direction — v2 detecting *fewer*
   peaks than v1. On shank 3 of the smoke fixture, the planted unit is
   so weak that v1 caught exactly 1 peak (at 1.58 s) and v2 caught
   zero. This is almost certainly a `noise_levels` randomness effect:
   with v2's per-chunk-then-mean MAD, that channel's threshold
   happens to land slightly above the 1 spike's amplitude. Different
   v2 invocations would catch / miss it.
   - **Predicts**: rerunning v2 sort on smoke-shank3 multiple times
     should produce *random* 0 or 1 peak outcomes. **Not tested**
     here (would require a fresh Spyglass populate); but it matches
     the H1/H2 model for near-threshold borderline cases.

8. **Could the divergence be a side-effect of v2's noise_levels caching
   property on the recording object that confounds reruns?** Yes,
   `set_property("noise_level_mad_raw", ...)` caches on the
   `BaseRecording`. So once `detect_peaks` is called, subsequent
   `detect_peaks` calls on the same instance reuse the cached
   noise_levels — making reruns deterministic *within one Spyglass
   populate*, but stochastic across populates that build a fresh
   recording. The parity-matrix capture re-runs Spyglass populate
   from scratch each time, so each capture sees a fresh recording ⇒
   each capture would get a different noise_levels. **This is a
   concerning property for reproducibility** but is orthogonal to the
   v1↔v2 systematic surplus.

### Self-critique round 2 — should I be more conservative?

The H4 mechanism (v1 over-suppression via sample-by-sample neighbour
sweep) is a *good* story. But:

- **I have not measured v2-to-v2 between-populate variance**. My
  3 reruns inside one process all used the cached noise_levels. To
  confirm the H1/H2 "smoke-shank3" speculation I'd need fresh
  recording-extractor instances in 3 separate runs. **Not done.**
  This means I can't quantify whether "v1 → v2 = 2657 → 3392" is
  the *systematic* delta or has stochastic variance ±50 peaks on top.
  Best estimate from synthetic probe: v2-v2 differs by ~0.6 % in
  noise_levels → at 5σ a ~0.6 % threshold shift flips ~0.6 % × N
  borderline peaks. For 3392 peaks, that's ~20 peaks of stochastic
  variance — small compared to the 757 systematic surplus. So the
  H4 conclusion holds even allowing for stochastic spread.

- **My check_h4.py interpretation was almost wrong**: at `exc=3`
  samples I saw "1/757 spatial-exclusion adjacent" and could have
  concluded "H4 false." The widening to `tol=30` was the key step
  that recovered the true picture (peaks 5-15 samples away). The
  initial framing of H4 in terms of `±exclude_sweep_size = 3`
  was too narrow. The right framing of H4 is "v1's algorithm
  over-suppresses peaks whose neighbours have trace dips in the
  sweep window even when the neighbour's peak is outside the
  window" (per PR #4341 description). This is wider than `±exc=3`
  rejection.

- **I have not investigated the smoke-shank2 ±88-sample case or the
  60s-shank1 ±1642-sample case.** Max-drift = 1642 samples is ~51 ms
  — much larger than the spike-propagation time. Could be a different
  mechanism. Looking at the diagnostic table again:

  | Case | unmatched_v1 | max drift |
  |---|---|---|
  | 60s-shank0 | 23 | 1270 |
  | 60s-shank1 | 15 | 1642 |
  | 60s-shank2 | 36 | 11 |
  | 60s-shank3 | 22 | 119 |

  The max drift is the *worst* per-pair, not the typical. If the
  parity matcher does nearest-neighbour matching and an unmatched_v1
  spike has its "nearest" v2 spike 50 ms away because no v2 spike fired
  near it, that explains 1642 — it doesn't necessarily mean any v2
  spike is timeshifted by 50 ms. **This was a misframing in the task
  prompt.** The max-drift is essentially a "how isolated is the worst
  unmatched_v1 in time" measure, not a "filter phase drift" measure.

- **Smoke-shank3 (v1=1, v2=0)** is the only "v2 produces fewer than v1"
  case in the matrix. The proposed fix for it (passing v1's
  noise_levels explicitly to v2) won't help because the meta JSON
  shows the smoke param row OMITS noise_levels. The user would need
  to either (a) accept stochastic miss-detection on barely-detectable
  spikes (1-spike-shank is degenerate), (b) seed the noise_levels RNG
  explicitly on the v2 Spyglass side. Recommend triage into
  `EXPECTED_DEGENERATE_CASES` with evidence "v1 1-spike baseline is
  near-threshold; v2's stochastic noise_levels reliably misses it."

### Round 4 — SI changelog / PR citations

Identified the two SI PRs that account for the divergence:

**PR #3359** "Improve noise level machinery" (merged 2024-10-25),
[https://github.com/SpikeInterface/spikeinterface/pull/3359](https://github.com/SpikeInterface/spikeinterface/pull/3359),
author samuelgarcia:

> *"A very important change is that now the `seed=None` (instead of
> `seed=0`) in the function which I think is the good way. seed must
> be explicit and no implicit. So the consequence is: all tests that
> are running twice the `get_random_data_chunk()` (sometimes this is
> hidden) are not guaranteed anymore to have the same results."*

Also adds `get_random_recording_slices()` and the per-chunk-then-mean
MAD estimator.

**PR #4341** "Make peak detection (locally_exclusive, matched_filtering)
faster and more accurate" (merged 2026-01-29),
[https://github.com/SpikeInterface/spikeinterface/pull/4341](https://github.com/SpikeInterface/spikeinterface/pull/4341),
author samuelgarcia:

> *"After playing a bit with the rust language, and trying to
> reimplement the peak detection algo using "locally_exclusive" I
> discover that the actual implementation could be fastened a lot (at
> least 3X faster on my machine). [...] Also the new implementation
> is more accurate for corner cases:*
> *  when the threshold are not the same by channel (the comparison
> must be done on the ratio and not the value)*
> *  when value in the sweep are higher but not peak themself."*

The "value in the sweep are higher but not peak themself" comment
**exactly describes the v1 corner case the empirical analysis
identified**: v1's algorithm suppressed peak A because some sample in
neighbour B's `±exclude_sweep_size` window was ≤ A's amplitude, even
when B's actual *peak* was elsewhere. The SI author explicitly classifies
this as **v1 being wrong**, fixed in v2.

PR #4341 also matches the per-channel-threshold-ratio normalization
finding from the empirical comparison (v1: raw-amp compare; v2:
ratio-to-threshold compare).

Updated confidences after round 4:
- H1 (noise_levels algorithm differs): **80 %** — confirmed by PR #3359
  + empirical, BUT downgrade as *primary* cause: only explains the
  ~17 unmatched_v1 spikes, not the 757 unmatched_v2 surplus.
- H2 (v2 stochastic noise_levels): **95 %** — confirmed by source +
  PR #3359 + empirical (3 calls give 3 answers).
- H3 (filter changed): **2 %** — no PR or source evidence; not the
  cause.
- H4 (locally_exclusive algo "more accurate" — v1 was over-suppressing):
  **90 %** — confirmed by source diff + empirical (757 unmatched_v2
  peaks are adjacent-channel echoes of real spikes) + PR #4341
  author-stated rationale.
- H5 (representation): **0 %** — fully ruled out.

---

## 4. Recommendation

**Composite classification: this is two divergences, not one.**

### Recommendation #1 — locally_exclusive algorithm change → **v1-wrong-with-evidence**

**Confidence: 90 %.**

The 757 / 3392 ≈ 22 % v2-surplus on 60s-shank0 (and the analogous
surplus on other 60s shanks) is the **dominant divergence**. It is
caused by SI PR #4341 changing the `locally_exclusive` peak-detection
algorithm. The SI author explicitly classifies the v1 behaviour as a
bug ("more accurate for corner cases ... when value in the sweep are
higher but not peak themself"), and the empirical analysis confirms:

- 99.6 % of v2-surplus peaks are within `radius_um=100` of a v1 peak.
- 100 % are within ±30 samples (≈ 0.94 ms) — i.e., they're
  adjacent-contact echoes of the same physical spikes.
- v1's bug suppressed them via the sample-by-sample neighbour-comparison
  sweep even when the neighbour's *peak* was outside the
  `±exclude_sweep_size=3` window.

**Action**: add a row to `parity-extensions.md` § "Documented v2
divergences" classifying this as **v1-wrong-with-evidence** with the
PR #4341 citation. Update the parity-test acceptance criterion for
clusterless: tolerate v2-only extras up to **30 % over v1 count, OR
± 1000 absolute, whichever larger**, provided the v2-only peaks are
within `radius_um` and `< 1 ms` of a matched-to-v1 peak. The
asymmetric matcher already handles "every v1 peak matches a v2 peak
within ±1.5 samples"; that constraint **should** still be honored
(v2 should NOT lose v1 spikes; the algorithm is strictly *more*
permissive on adjacent contacts, not less on the original detection).

### Recommendation #2 — noise_levels stochasticity → **deliberate v2 change + parity-test mitigation needed**

**Confidence: 95 %.**

SI PR #3359 changed `get_noise_levels`' default `seed=0 → seed=None`
deliberately. This makes v2's clusterless sort **non-deterministic
between fresh populates** (within a single populate it's deterministic
via property caching). Magnitude: ~0.6 % per-channel noise_level
variance → at 5 σ threshold, ~17 borderline v1-spikes can be missed by
v2 on any given run (matches the `unmatched_v1 = 23 → 6` reduction when
v1's deterministic noise_levels are forced into v2).

**Action**: add a row to the divergence table classifying this as
**intentional-SI-behavior-with-cite** (PR #3359, author-stated
rationale "explicit seed is the good way"). For determinism in the
parity test, Spyglass v2's `_run_clusterless_thresholder` should pass
an explicit `random_slices_kwargs={"seed": 0}` (or a Spyglass-level
configured seed) into `detect_peaks` so reruns are reproducible. This
is a small Spyglass-v2 patch and would eliminate the ~17 unmatched_v1
spikes per shank without any other behavioral change. Independent of
that patch, the parity test should tolerate `unmatched_v1 ≤ ceil(0.01
* v1_count)` (an absolute-1 % floor) attributable to noise_level
stochasticity, until the patch lands.

### Recommendation #3 — smoke-shank3 (v1=1, v2=0) → **expected-degenerate (with evidence)**

**Confidence: 70 %.**

The 1-spike-on-shank-3 case is degenerate: a single near-threshold spike
will be picked up or not depending on the run-to-run noise_levels.
Triage into `EXPECTED_DEGENERATE_CASES` with reason:
`"smoke-shank3: v1 baseline = 1 spike at 1.58s; v2 noise_levels
stochasticity at 5sigma threshold flips this single-spike detection
across reruns. Not a regression."`

(After the seed-explicit Spyglass-v2 fix from Rec #2, this case should
either always-PASS or always-MISS deterministically; rerun the capture
to decide which.)

### Final aggregate recommendation

Overall classification of the divergence: **mixed — primarily
v1-wrong-with-evidence (Rec #1, the locally_exclusive algorithm fix)
plus a smaller deliberate-v2-change stochasticity layer (Rec #2)**.
The current `parity-extensions.md` divergence table's hand-wave
attribution to "`random_slices` / sample-window drift" is **incorrect
as a primary explanation** — `random_slices` accounts for ~3 % of the
divergence (the unmatched_v1 side); the locally_exclusive algorithm
fix accounts for ~97 % (the unmatched_v2 surplus). The PR #4341
citation belongs in the table.

**Aggregate confidence in this recommendation set: 85 %**.
The 15 % I'm holding back covers: (a) I haven't tested smoke-shank2's
specific 88-sample drift; (b) I haven't verified my measurement of the
per-channel noise-level variance is stable across more reruns; (c) I
haven't read the full PR #4341 diff to confirm there isn't a *second*
behaviour change beyond the two I identified; (d) I haven't verified
that the PR #4341 change shipped intact in 0.104.3 vs. having been
modified between PR-merge and release.

---

## 5. What I cannot rule out

1. **Drift in shank 2 specifically (88 samples ≈ 2.75 ms)**. The
   smoke-shank2 unmatched_v1=1 case has a single v1 spike that's
   2.75 ms away from its nearest v2 spike. That's an order of magnitude
   larger than my "≤ 30 sample" universal-match threshold from the 60s
   probe. Could be a single-channel sampling-window artefact or
   something else. Not investigated.

2. **Whether v2's deterministic-within-process noise_levels caching has
   subtler effects elsewhere in the pipeline** (e.g., does Spyglass v2
   call `detect_peaks` once and `get_noise_levels` independently again
   somewhere, causing the two to diverge? Quick grep suggests not, but
   I didn't trace exhaustively.)

3. **Whether the same v1↔v2 surplus pattern persists for MS4** (Phase
   B). The MS4 pipeline uses different SI internals; the
   noise_levels-divergence and locally_exclusive-fix don't apply to
   MS4 the same way. Out of scope here.

4. **Other PRs touched between SI 0.99.1 and 0.104.3 that may
   contribute**. I identified PR #3359 + PR #4341. There are likely
   smaller fixes I haven't catalogued (e.g., changes to
   `bandpass_filter` defaults, dtype promotion rules, edge-of-segment
   handling). The empirical control showed v1-vs-v2 noise_levels agree
   to 0.1 % on synthetic noise, suggesting filtering + MAD machinery
   are very close — but I haven't done a bit-by-bit `bandpass_filter`
   comparison.

5. **Whether v1's behavior is "right" or "wrong" depends on the user's
   biological intent**. PR #4341 explicitly classifies v1 as buggy.
   But for a Spyglass user who *wanted* peak deduplication across
   adjacent contacts (treating one physical spike as one detection),
   v1's over-suppression may have been a *feature*. v2 treats each
   contact's local maximum as a separate detection if no contemporaneous
   peak within `±exclude_sweep_ms` shares the spatial neighbourhood —
   meaning **users who pipe `clusterless_thresholder` output into the
   clusterless decoder will now see ~30 % more events per shank**.
   Whether that's "right" for downstream decoding is a *scientific*
   question outside this investigation's scope.

6. **Whether v2's noise_levels variance across populates *systematically*
   biases the threshold**. My probe used 3 reruns. With more reruns
   the mean might drift toward something stable but biased relative
   to v1. Not measured.

7. **Whether the Spyglass-v2 `_run_clusterless_thresholder` could be
   patched to call `get_random_recording_slices(recording, seed=0)`
   explicitly without breaking anything else**. Looks safe (the function
   accepts a `seed` kwarg, and propagating it doesn't affect any other
   pipeline step) but the patch would need its own parity check —
   does setting `seed=0` on the v2 side match v1's `seed=0` chunk
   selection exactly? Not certainly, because v2's `chunk_size = 16000`
   vs v1's `chunk_size = 10000` would still differ. So even with
   `seed=0` on both sides, the underlying chunks differ. This means
   "exactly match v1 noise_levels by forcing seed=0" is **not
   straightforwardly achievable** without further matching of
   `chunk_size` / `chunk_duration` / segment-boundary behavior.

