# Appendix — Upstream pipeline references (AIND, IBL)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The motivating implementations in the Allen (AIND) and IBL Neuropixels
pipelines, with repo + commit + `file:line`. Line numbers are verified at the
commits below; upstream drifts, so the commit hash is the anchor (re-clone at
that commit, or read the named function). **These are Neuropixels-tuned**;
thresholds and the phase-shift step are NP-specific — adapt, don't copy
verbatim (see [overview](overview.md) Non-Goals).

To read the source: `git clone https://github.com/<owner>/<repo>` then
`git checkout <commit>`.

## Feature → reference map

| This plan's phase | AIND reference | IBL reference |
| --- | --- | --- |
| [phase 1 — phase-shift](phase-1-adc-phase-shift.md) | [A.1](#a1-adc-phase-shift) | [B.3](#b3-adc-phase-shift-fshift) |
| [phase 2 — detection](phase-2-bad-channel-detection.md) | [A.2](#a2-bad-channel-detection--removal) | [B.5](#b5-bad-channel-detection) |
| [phase 3 — interpolate/remove](phase-3-bad-channel-handling.md) | [A.2](#a2-bad-channel-detection--removal) (remove) | [B.4](#b4-bad-channel-interpolation) (interpolate) |
| [phase 4 — drift QC](phase-4-drift-qc.md) | [A.4](#a4-motion-computed-not-applied) | — |
| order validation (filter→reference) | [A.3](#a3-filter-order--reference) | [B.1](#b1-destripe-chain-order) |

## A. AIND — `aind-ephys` pipeline (Neuropixels, SpikeInterface 0.103.2)

Repos / commits (cloned + verified):

- `AllenNeuralDynamics/aind-ephys-preprocessing` @ `6611cf9` — the preprocessing
  capsule (all the logic). The pipeline pins it at `1724d7fb`; the agent
  verified `code/params.json` is byte-identical between the two, so the refs
  below at `6611cf9` are authoritative for the default config.
- `AllenNeuralDynamics/aind-ephys-pipeline` @ `0eaf64a` — Nextflow orchestration
  + sorter params.
- `AllenNeuralDynamics/aind-ephys-spikesort-kilosort4` @ `03d3522` — the KS4
  capsule (where drift/whitening actually happen for the default path).

The pipeline builds a dict `DEFAULT_PREPROCESSING_PIPELINE` and applies steps in
**dict insertion order** via `spre.apply_preprocessing_pipeline` —
`aind-ephys-preprocessing/code/run_capsule.py:428`. So `code/params.json` key
order *is* the operation order. Default config: `denoising_strategy="cmr"`,
`filter_type="highpass"`, motion computed-not-applied.

### A.1 ADC phase-shift

- `code/params.json:14-16` — `"phase_shift": {"margin_ms": 100.0}` (first key).
- `code/run_capsule.py:395-397` — dropped when the recording lacks the
  `inter_sample_shift` property: `if "inter_sample_shift" not in
  recording.get_property_keys(): preprocessing_pipeline.pop("phase_shift", None)`.
  **This is the exact gating phase 1 mirrors.**
- `docs/source/parameters.rst:305-307` — rationale ("multiplexed ADCs").

### A.2 Bad-channel detection + removal

- `code/params.json:26-35` — `"detect_and_remove_bad_channels"`: one combined
  detect+remove step, `"method": "coherence+psd"`, with the dead/noisy/outside
  thresholds and `channel_filters` (phase 2 borrows these defaults).
- `code/run_capsule.py:409-414` — splits the toggles (`remove_out_channels`
  vs `remove_bad_channels`); `:476-484` — skips the whole recording if removed
  channels exceed `max_bad_channel_fraction` (default 0.5).
- AIND **removes** (channel-slices out) bad channels — it does not interpolate.
- `docs/source/parameters.rst:309-311` — coherence+psd rationale.

### A.3 Filter order + reference

- `code/params.json:17-19` — `"highpass_filter": {"freq_min": 300.0,
  "margin_ms": 5.0}` (Butterworth, zero-phase); `:21-25` — a `bandpass_filter`
  (300–6000) alternative, dropped by default.
- `code/params.json:36-38` — `"common_reference": {"reference": "global",
  "operator": "median"}` — global CMR; `:40-48` — `"highpass_spatial_filter"`
  (destripe) alternative, mutually exclusive with CMR.
- Filter (step ~2) precedes reference (step ~4); bad-channel removal sits
  between. **Validates our filter→reference order.**
- `docs/source/parameters.rst:344-347` — "median is robust to outlier channels."

### A.4 Motion (computed, not applied)

- `code/params.json:52` — motion `"preset": "dredge_fast"`.
- `code/run_capsule.py:490-625` — motion handled *outside*
  `apply_preprocessing_pipeline`, after the recording is saved;
  `compute_motion` at `:542-551`; estimate kwargs `:509-526`; default
  `COMPUTE_MOTION=True, APPLY_MOTION=False`.
- `docs/source/parameters.rst:360-367` — "motion is always saved … even if it
  is not applied" (apply default `false`). **This is the model phase 4 adopts.**
- `aind-ephys-spikesort-kilosort4/code/run_capsule.py:264-270` — KS4 does its
  own drift correction (`do_correction`) for the default path.

## B. IBL — `ibl-sorter` / `ibldsp` destripe (Neuropixels, pykilosort 2.5)

Repos / commits (cloned + verified):

- `int-brain-lab/ibl-sorter` @ `1ec099e` — the sorter; preprocessing imports
  from `ibldsp`.
- `int-brain-lab/ibl-neuropixel` @ `3a3a2a5` — provides `ibldsp`, where the
  destripe chain lives.

Entry point: `ibl-sorter/iblsorter/preprocess.py:13` —
`from ibldsp.voltage import decompress_destripe_cbin, destripe,
detect_bad_channels`. Production chain writes `proc.dat` via
`decompress_destripe_cbin`; `destripe()` is the array form used for the
whitening covariance. Both apply the same step order.

### B.1 Destripe chain order

- `ibl-neuropixel/src/ibldsp/voltage.py:484-545` — `destripe()` (array form):
  filter → fshift → interpolate → spatial filter.
- `ibl-neuropixel/src/ibldsp/voltage.py:592` — `decompress_destripe_cbin()`;
  inner per-batch loop `:757-799` is the authoritative production order
  (filter `:757`, fshift `:762`, interpolate `:773`, spatial `:776-780`,
  whitening `:798-799`).
- Filter is unambiguously first, before any spatial op. **Validates
  filter→reference.**

### B.2 Highpass filter

- `voltage.py:459` — `butter_kwargs = {"N": 3, "Wn": 300/fs*2,
  "btype": "highpass"}` (3rd-order Butterworth, 300 Hz).
- `voltage.py:525` / `:757` — applied zero-phase via
  `scipy.signal.sosfiltfilt`.

### B.3 ADC phase-shift (fshift)

- `voltage.py:529-532` / `:762` — `fourier.fshift(x, h["sample_shift"], ...)`,
  a frequency-domain per-channel sub-sample shift. NP multiplexing correction
  (`sample_shift` from the probe header).
- **Order note:** IBL applies `fshift` **after** the temporal high-pass filter
  (`:525` then `:529`), whereas AIND ([A.1](#a1-adc-phase-shift)) applies
  phase-shift **first**. The two are equivalent: a fractional-sample delay
  (`fshift`/`phase_shift`) and a zero-phase Butterworth bandpass are both LTI
  per channel and commute (modulo edge margins). [Phase 1](phase-1-adc-phase-shift.md)
  adopts the AIND "first" order as a deliberate choice, not a correctness
  requirement — so "matches AIND/IBL" is more precisely "AIND order; equivalent
  to IBL's post-filter order."

### B.4 Bad-channel interpolation

- `voltage.py:413-452` — `interpolate_bad_channels()`: inverse-distance kriging
  (`p=1.3`, `kriging_distance_um=20`); fills labels **1 (dead)** and **2
  (noisy)** only (`:437`).
- `voltage.py:538` / `:773` — applied after the filter, before the spatial
  filter; outside-brain channels (label 3) excluded from the spatial step
  (`inside_brain`, `:539`/`:774`). **IBL interpolates rather than removing** —
  the model phase 3's `interpolate` option mirrors (we fill the sort group's
  curated-bad channels).

### B.5 Bad-channel detection

- `voltage.py:847-1035` — `detect_bad_channels()`: the coherence/PSD method.
  `channels_similarity` (xcorr with the median trace) `:931-956`; `psd_hf`
  `:979`; labeling (dead/noisy/outside) `:983-1023`. **Same `coherence+psd`
  method SpikeInterface exposes and phase 2 wraps.**
- `ibl-sorter/iblsorter/preprocess.py:227` — run over snippets and the per-
  channel **mode** taken for robustness (`get_good_channels`, `:212-250`).

### B.6 Spatial filter ("k-filter", not plain CAR)

- `voltage.py:472-481` — `apply_spatial_filter` dispatcher (k-filter default,
  CAR fallback); `kfilt()` `:192-267` (Butterworth high-pass along the channel
  axis); `car()` `:168-189` (median/mean subtraction).
- `voltage.py:540-544` / `:776-780` — applied **per shank**
  (`collection=h["shank"]`). Beyond this plan (a future `reference_mode`); cited
  for the per-shank principle our reference already follows.

### B.7 Whitening (in preprocessing, baked into proc.dat)

- `ibl-sorter/iblsorter/preprocess.py:172-209` — `get_whitening_matrix` (ZCA,
  local `whiteningRange=32`); applied at `voltage.py:798-799`. (IBL whitens in
  preprocessing; we defer to the sorter — cited for contrast, not adopted.)
