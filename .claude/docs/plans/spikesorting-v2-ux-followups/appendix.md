# Appendix — SpikeInterface 0.104.3 deep reference

[← back to PLAN.md](PLAN.md)

Source-grounded reference for the SI internals this plan (and the later
motion/curation phases) implement against. All facts verified against SI
**0.104.3** source in the v2 env
(`/Users/edeno/miniconda3/envs/spyglass_spikesorting_v2/lib/python3.11/site-packages/spikeinterface/`)
and confirmed empirically where noted. Load this only when implementing against SI.

## Sorters — and the `detect_threshold` resolution

**MS4/MS5 `detect_threshold` is a standard-deviation multiple of the ZCA-whitened
signal — NOT a MAD multiplier.** Traced: `mountainsort4/ms4alg.py:60` detects via a
raw `max_vals >= detect_threshold` with **no internal normalization** (grep for
`whiten|std|MAD|median|variance` in `ms4alg.py` → nothing); the SI wrapper
(`sorters/external/mountainsort4.py:103-106`) calls `whiten(dtype="float32")`
*first*; SI `whiten` is std-based ZCA (`cov = data.T@data/N`,
`preprocessing/whiten.py:256-257`; `W = U·diag(1/√(S+eps))·Uᵀ`, `:232-233`) →
unit-variance output. Empirically: post-whiten per-channel std = 1.000, fraction
beyond ±3 = 0.0027 (= Gaussian 3σ). MS5 (`sorting_scheme1.py:39`,
`external/mountainsort5.py:142-147`) is the same scale; its `5.5` default is a more
conservative *tuning*, not a different unit. **The MAD label belongs to the
clusterless `detect_peaks` path only** (see Peak detection below). On whitened
Gaussian noise σ and MAD·1.4826 coincide (≈1), so values still behave — but the
mechanism is std-via-ZCA.

**`filter`/`whiten` flags unconditionally apply** inside the sorter wrapper — they
do **not** detect prior preprocessing. SI defaults `filter=True, whiten=True`. v2
sets `filter=False` (recording already bandpassed) and runs whitening once,
externally, at float64 with a pinned seed (`_sorting_dispatch.py`), then
passes `whiten=False` to the sorter. Sound (avoids double-filter/double-whiten).
MS4 also needs a `np.Inf=np.inf` shim under numpy≥2 (v2 patches it).

| Param | MS4 default | Units / meaning |
| --- | --- | --- |
| `detect_sign` | −1 | −1 neg / 1 pos / 0 both. MS4 SI desc lists only −1/1 (supports 0); MS5 desc lists 0. |
| `adjacency_radius` | −1 (SI); v2 sets 100 | **µm, spatial, rate-independent**; −1 = all channels (`ms4alg.py:69-73`). |
| `clip_size` | 50 (SI); franklab 40/27 | **samples** per waveform (rate-tied). |
| `detect_threshold` | 3 | **σ of the ZCA-whitened signal** (see above), not absolute V. |
| `detect_interval` | 10 | **samples**, per-channel dead-time (rate-tied). MS5 uses `detect_time_radius_msec=0.5` (ms, rate-*independent*) instead. |
| `freq_min/max` | 600/6000 (v2) | Hz; only used if `filter=True` (off in v2). |

MS5 schemes: 1 = single-pass (short/stationary); 2 = default, train-on-subset
(`scheme2_training_duration_sec=300`) then classify full (`phase1_detect_channel_radius=200`,
`detect_channel_radius=50`); 3 = per-block (`scheme3_block_duration_sec=1800`) for
long/drifting. KS4: `Th_universal`/`Th_learned` (detection sensitivity), `nblocks`
(drift: 0=off, 1=rigid, >1=nonrigid; `do_correction=False` silently forces
`nblocks=0`); KS4 defaults are pulled live from the installed `kilosort` package
(not frozen in SI) — verify with `Kilosort4Sorter.default_params()` in a kilosort env.

## Preprocessing

- **`bandpass_filter`** (`preprocessing/filter.py`): `freq_min=300, freq_max=6000`, Butterworth `filter_order=5` **doubled to effective 10** (zero-phase `forward-backward` `sosfiltfilt`, acausal). `margin_ms='auto'`≈16.7 ms at 300 Hz. **Raises if `freq_min<100` and `ignore_low_freq_error=False`** (v2 doesn't pass it → a custom low-band preset would fail). v2 forces `dtype=float64`.
- **`common_reference`** (`common_reference.py`): `reference="global"`, `operator="median"`. `reference="global"` subtracts the per-sample median across channels. **With `groups`, "global" becomes per-group**, but v2 passes no `groups` (each sort group is its own recording), so it's genuinely all-channels-in-the-group.
- **Filter↔reference is non-commutative on the median-CAR branch** (median is non-linear across channels). v1 referenced-then-filtered; **v2 filters-then-references** → v1/v2 differ numerically on the `global_median` path, identical on `specific`/`none`. Documented, intentional.
- **`whiten`** (`whiten.py`): std-based ZCA. **eps default is `1e-16` in code (docstring wrongly says `1e-8`)**; auto-scales to `median(data²)·1e-3` for small-float data. Seed for the covariance random-chunks defaults to `None` (non-deterministic) — v2 pins it. v2 **defers whitening to the sorter stage** (not applied at preprocessing) — correct, because whitening must not precede motion estimation and would double-whiten internal-whitening sorters.
- **`phase_shift`** (`phase_shift.py`): **raises `AssertionError` if the recording lacks `inter_sample_shift`** — the "safe no-op on tetrodes" is a v2 guard, not SI behavior. SI `margin_ms=40`; v2 presets use 100. Only `default_neuropixels` enables it.

## SortingAnalyzer + extensions (SI ≥0.101)

`create_sorting_analyzer(sorting, recording, format="memory"|"binary_folder"|"zarr", sparse=True, sparsity=None, **sparsity_kwargs)`. `binary_folder`/`zarr` persist a recording **link (provenance), not traces** — extensions needing traces (`spike_amplitudes`, hard-merge recompute) require the upstream recording present. Zero-unit sortings **cannot** build an analyzer (`estimate_sparsity`→`np.concatenate([])`); v2 short-circuits and raises `ZeroUnitAnalyzerError` — any analyzer loop must pre-filter `n_units>0`.

**Extension dependency graph** (build order):
```
random_spikes → waveforms → templates → { spike_amplitudes, spike_locations,
                                          unit_locations, template_similarity,
                                          template_metrics, amplitude_scalings }
random_spikes → waveforms → principal_components
noise_levels (needs recording, order-free) ; correlograms/isi (sorting-only)
```
`templates` needs `random_spikes | waveforms` (either). **Silent-delete cascade on
recompute:** recomputing `random_spikes` deletes `waveforms,templates,PCA`;
`waveforms` deletes `templates,PCA`; `templates` deletes all template-derived
extensions (recursive). Changing an extension's params silently deletes its
children and wipes its on-disk data.

**Defaults that matter:** `random_spikes` `max_spikes_per_unit=500, seed=None`;
`waveforms` `ms_before=1.0, ms_after=2.0`. **Set A uses `0.5/0.5` and `20000`** — a
recompute, which cascades. **Sparsity (`sparse=True`, radius 100µm default) is the
primary lever against 50+GB analyzer folders** (~50-100× on a 384-ch probe;
≈no-op on a tetrode). `merge_units(merging_mode="soft")` derives merged data
(needs `sparsity_overlap≥0.75`, else errors); `"hard"` recomputes from traces.

**`compute_quality_metrics(analyzer, metric_names=, metric_params={name:{...}}, ...)`**
lives in `spikeinterface.metrics` now (the `qualitymetrics` import is a deprecated
shim, removed 0.105). PCA metrics run **only if `principal_components` is already
computed** (silent skip otherwise); snr/amplitude metrics silently use
`noise_levels`/`spike_amplitudes`/`templates` if present — so **`depend_on=[]` does
not protect you; the recipe must explicitly compute the required extensions and
assert their presence**, or metric values silently change/omit.

## Quality metrics & curation (Phase 2b)

**`isi_violation` (Spyglass ≠ SI) — verified by reading both formulas:** SI's
`isi_violations_ratio` (`misc_metrics.py` `isi_violations`) is the **Hill/UMS2000
contamination-rate estimate** `n_viol·T / (2·N²·(τ−min_isi))` — an *estimated
fraction of spikes that are contamination*, **unbounded, can exceed 1**, and SI's
own docstring says it "breaks down... especially for highly contaminated units."
Spyglass deliberately ignores that ratio: it takes SI's **stable violation count**
and computes its own **bounded observed fraction** `count / (num_spikes − 1)`
(`v1/metric_utils.py:16-38`) — the literal fraction of consecutive-spike ISIs that
are too short, ∈ [0,1], no model assumptions, no `N²`/duration dependence. (This
is why: the SI *ratio* is a model estimator whose formula has varied across the
ecosystem and misbehaves on contaminated units; the count is robust.) So the Set-A
gate `> 0.0025` = "more than 0.25% of consecutive ISIs violate the 1.5 ms
threshold," and the mooted **2% = `> 0.02`**. **If Phase 2b ports Set A onto a SI `SortingAnalyzer` via
`compute_quality_metrics`, the `isi_violation` column will be SI's ratio (a
different number, can exceed 1) unless the custom `count/(n−1)` fraction is
replicated** — the gate would otherwise change meaning silently. Edge bug (#1556
class): a 1-spike unit gives `0/0 = nan` (caught by NaN sanitization); a 0-spike
unit hits the **high-level** `compute_isi_violations` count=−1 sentinel (the
*low-level* `isi_violations` returns nan; Spyglass calls the high-level one) →
`−1/(0−1) = 1.0`, a finite spurious value that survives NaN sanitization.

**`nn_isolation`/`nn_noise_overlap` are deprecated names** in SI 0.104.3 — now the
two output columns of one metric **`nn_advanced`** (`metrics/quality/pca_metrics.py`).
Requesting the old names **raises ValueError**; the v1 `getattr(sq,
"nearest_neighbors_isolation")` chain resolves only under SI 0.99. Phase 2b must
request `nn_advanced` and read the two columns. Set-A params differ from SI
defaults: `n_components=7` (SI 10), `n_neighbors=5` (SI registers 4),
`max_spikes=20000` (SI 1000), `seed=0` (SI `None`) — **pass all explicitly**.

**Low-spike NaN behavior (relevant to #1556):** `nn_*` → NaN when `n_spikes <
min_spikes` (10, kept in Set A) so low-spike noisy units are **not** auto-rejected
(NaN comparisons are False); `amplitude_cutoff` → NaN below ~500 spikes;
`firing_rate`/`presence_ratio` → NaN at 0 spikes; `snr` → `inf` if noise=0;
`num_spikes`/`peak_offset`/`peak_channel` never NaN. **Effective defaults come from
the registration `metric_params` dict, not the function signatures/docstrings**
(several docstrings are wrong, e.g. `snr.peak_sign` is "both" not "neg") — pin
params explicitly.

**`peak_offset`/`peak_channel` are Frank-lab custom** (`v1/metric_utils.py:41-74`),
not SI metrics; `peak_offset` is `|template_peak − nbefore|` in **samples**.
v2 already persists a peak channel at sort time, so `peak_channel` may be redundant.

**Curation API:** `compute_merge_unit_groups` (replaces deprecated
`get_potential_auto_merge`; presets `similarity_correlograms` default,
`temporal_splits`, `x_contaminations`, ...); `CurationSorting`/`MergeUnitsSorting`
(`delta_time_ms=0.4`); `apply_curation` applies **labels→remove→merge→split**;
`model_based_label_units` (replaces `auto_label_units`). The "auto→manual
merge→re-curate" workflow = compute metrics → propose merges/labels → merge
(new analyzer) → **recompute metrics on the merged analyzer** for final numbers.
Spyglass v1's Set-A path calls SI metric *functions* directly with a
`[op, value, [labels]]` triple, not SI's curation module.

## Motion / comparison / peak detection (later phases)

- **`correct_motion(recording, preset=, ...)`** code default `dredge_fast` (docstring stale). Pipeline: detect→localize→estimate→interpolate; per-step `*_kwargs` **merge over** the preset.
- **Torch dependence:** `dredge`/`dredge_fast`/**`rigid_fast`** all use `dredge_ap` which **hard-requires torch** (`dredge.py:110-113`). Only **`kilosort_like`** (`iterative_template`) is torch-free; `nonrigid_accurate`/`nonrigid_fast_and_accurate` (`decentralized`) are torch-**optional** (numpy fallback). **No preset requires GPU** (CPU fallback). v2's shipped concat default is `rigid_fast` → torch needed (v2 already pins torch).
- **`estimate_motion` is single-segment only** (`assert num_segments==1`) — concat must estimate on one concatenated segment. **Whiten AFTER motion** (SI explicit: do not whiten before motion estimation) — aligns with v2's deferred whitening.
- **`compare_sorter_to_ground_truth`** defaults `match_score=0.5, delta_time=0.4 ms`. `accuracy = tp/(tp+fn+fp)`, `recall = tp/(tp+fn)`, `precision = tp/(tp+fp)`. **Unmatched GT units report all-zeros (not miss_rate=1)** → a pooled average is dragged toward 0 by undetected units; use `get_performance("by_unit")` per unit for the gate. `delta_frames` truncates; matching is boundary-inclusive.
- **`detect_peaks`** (clusterless): `abs_thresholds = noise_levels × detect_threshold`. Estimated `noise_levels` → per-channel **MAD** → `detect_threshold` is a MAD multiplier; `noise_levels=[1.0]` + `scale_to_uV` → a true **µV** threshold. **v2's `threshold_unit` ('uv'/'mad') matches this exactly** (`_sorting_dispatch.py`); the `'uv'` path raises if the recording carries no channel gains. Concat motion + `ConcatenatedRecording` are still `NotImplementedError`-gated in v2.

## Cross-cutting: determinism

`random_spikes`, `whiten`, and `nn_advanced` all default `seed=None` →
non-deterministic across runs (subsampling / random covariance chunks). v2 pins
seeds (default 0). **Any new analyzer/metric/curation code that recomputes these
must re-pin the seed** or per-unit metrics, peak amplitudes, PCA, and fixture
hashes drift. (Ties to the recording_h5_sha256 / fixture-hash non-determinism work.)
