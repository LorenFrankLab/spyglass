# Designs

[← back to PLAN.md](PLAN.md)

Sections: [filter-before-restriction](#filter-before-restriction) · [valid-sample-statistics](#valid-sample-statistics) · [geometry-normalization](#geometry-normalization) · [per-unit-halves](#per-unit-halves) · [metric-missingness](#metric-missingness) · [mua-contiguous-runs](#mua-contiguous-runs)

## filter-before-restriction

**Problem.** `Recording.make_compute` (`recording.py:2101-2137`) restricts time first (`restrict_recording`, which frame-slices each selected interval and concatenates), then filters (`apply_pre_motion_preprocessing`). The lazy `bandpass_filter` sees a concatenated signal with step discontinuities at every join; its `margin_ms` only covers chunk boundaries of a continuous parent. With the shipped `min_segment_length: 0.0015` (`_recipe_catalog.py:180,192`) a 45-sample sliver is entirely filter transient (measured 3.8× RMS, 96 µV peak error after a 600 Hz high-pass).

**Design.** Reorder to: read → channel-select → temporal preprocessing (phase-shift, bandpass) on the continuous channel-sliced recording → time restriction (frame-slice + concatenate) → spatial preprocessing (bad-channel interpolation, reference). SpikeInterface's `FrameSliceRecording` of a `BandpassFilterRecording` pulls the filter margin from the continuous parent, so every restricted sample is filtered with its true temporal context. Referencing and interpolation are per-sample spatial operations and are unaffected by concatenation.

Three code moves, one caller:

1. In `_recording_restriction.py`, extract the `ChannelSliceRecording` block at the end of `restrict_recording` (`:494-624`, from `assert_reference_not_member(...)` through `return`) into

```python
def select_sort_group_channels(
    recording,
    nwb_file_name: str,
    sort_group_channel_ids: list,
    reference_mode: str,
    reference_electrode_id: int | None,
    *,
    bad_channel_handling: str = "remove",
    bad_channel_ids=(),
):
    """Channel-slice the full-source recording to the sort group's surface.

    Members plus the ``specific`` reference (sliced in for subtraction, dropped
    after referencing) plus, on ``interpolate``, the interior curated-bad
    channels. Time is untouched; call before any temporal preprocessing so
    filters see the continuous source.
    """
    # body = the existing assert_reference_not_member / extra / slice_ids /
    # spikeinterface_channel_ids / ChannelSliceRecording code, verbatim.
    return ChannelSliceRecording(recording, channel_ids=si_ids, renamed_channel_ids=slice_ids)
```

`restrict_recording` keeps its interval intersection + `restrict_recording_times` call and returns `(recording, timestamps_override, n_intervals)`; drop its channel parameters (`sort_group_channel_ids`, `reference_mode`, `reference_electrode_id`, `bad_channel_handling`, `bad_channel_ids`) — after the move it no longer uses them. Update its docstring (delete the "reference channel is included in the slice" and "ChannelSliceRecording" paragraphs).

2. In `_recording_preprocessing.py`, split `apply_pre_motion_preprocessing` (`:31`) into

```python
def apply_temporal_preprocessing(recording, validated) -> tuple:
    """Phase-shift then bandpass on a CONTINUOUS recording; returns (recording, applied_steps)."""
    # steps 0 and 1 verbatim (lines ~93-135), applied_steps = {"phase_shift": bool}

def apply_spatial_preprocessing(
    recording, reference_mode, reference_electrode_id, validated,
    bad_channel_handling="remove", bad_channel_ids=(),
) -> tuple:
    """Bad-channel interpolation then reference on the time-restricted recording; returns (recording, applied_steps)."""
    # steps 1b and 2 verbatim (lines ~136-244), applied_steps = {"bad_channels": {...}}
```

Delete `apply_pre_motion_preprocessing` (single caller). Keep `filtering_description` (`:252`) reading `applied_steps["phase_shift"]`; the caller merges the two `applied_steps` dicts.

3. In `recording.py:2101-2137`:

```python
recording = read_recording_nwb(raw_path, load_time_vector=load_time_vector,
                               electrical_series_path=raw_series_path)
sampling_frequency = float(recording.get_sampling_frequency())
recording = select_sort_group_channels(
    recording, nwb_file_name=nwb_file_name, sort_group_channel_ids=channel_ids,
    reference_mode=reference_mode, reference_electrode_id=reference_electrode_id,
    bad_channel_handling=preprocessing_params.bad_channel_handling,
    bad_channel_ids=bad_channel_ids,
)
recording = normalize_channel_locations(recording)  # see geometry-normalization
recording, temporal_steps = apply_temporal_preprocessing(recording, preprocessing_params)
recording, timestamps_override, n_selected_intervals = restrict_recording(
    recording=recording, nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name, sort_valid_times=sort_valid_times,
    raw_valid_times=raw_valid_times,
    min_segment_length=preprocessing_params.min_segment_length,
)
recording, spatial_steps = apply_spatial_preprocessing(
    recording, reference_mode=reference_mode, reference_electrode_id=reference_electrode_id,
    validated=preprocessing_params, bad_channel_handling=preprocessing_params.bad_channel_handling,
    bad_channel_ids=bad_channel_ids,
)
applied_steps = {**temporal_steps, **spatial_steps}
recording = maybe_apply_tetrode_geometry(...)  # unchanged
assert_unique_contact_positions(recording)      # see geometry-normalization
```

**Invariants to assert in tests.** `timestamps_override` values, `n_selected_intervals`, and `recording.get_channel_ids()` are identical before and after the reorder on the same fixture. `restrict_recording_times` (`:413`) is unchanged; its non-regular branch (`select_channels` + `reset_times`) operates on the filtered recording and must still produce the same frames (SpikeInterface preprocessors delegate `get_times` to the parent). Correctness of the filtered traces is checked against a continuously filtered reference (filter the whole channel-sliced recording, then take the same sample ranges), never against the old restrict-then-filter output: the two orders differ at every interval edge by construction.

**Tolerance for the reference comparison.** SpikeInterface's lazy filter uses a finite `margin_ms` (default 5 ms) per `get_traces` request, so a sliced request and a whole-recording request to the same continuous filter differ by roughly 1e-4 (float64, observed 1.5e-4) at request edges. Two controls: request the reference with the same frame boundaries as the restricted recording's chunks (call `reference.get_traces(start, end)` per interval rather than one whole-recording request), and compare with `max_abs_diff <= 1e-3 * rms(reference)` plus `rms` agreement within 0.1%. Do not use `1e-9`.

**Not a fix on its own:** raising `min_segment_length` (a 50 ms floor still leaves discontinuities between longer intervals). Keep the recipe floors as they are.

## valid-sample-statistics

**Problem.** The sort stage rebinds `recording` to the artifact-masked recording (`sorting.py:1768-1776`; `silence_periods(mode="zeros")` at `_sorting_artifact_mask.py:323-327`). Three estimators then treat zeroed samples as data: the whitening covariance (`pinned_whiten`, `_sorting_dispatch.py:334`, called at 714-726), the analyzer `noise_levels` extension (`_sorting_analyzer.py:939-951`), and the nn-noise-overlap noise cluster (`_si_metric_patches.py:96-103`). Measured bias: noise −10% at 5% masked, −30% at 50%; whitening gain +7% at 10% masked, +23% at 50%. A second, independent problem: even an unmasked sort recording contains artificial joins (between selected intervals from `restrict_recording`, and between members of a concatenated recording), and any chunk that straddles a join is contaminated.

**Design.** Thread an explicit, authoritative list of *statistics spans* — half-open, sorted, disjoint frame ranges that are both artifact-free AND lie within a single acquisition/selection/member span — from the sort stage to every estimator, and never rediscover either kind of boundary from the recording object. Reasons: SpikeInterface 0.104.3's `SilencedPeriodsRecording` stores its periods under `_kwargs["periods"]` as a per-segment vector (`silence_periods.py:115`), not a list of ranges; concatenated recordings are masked in memory (`session_group.py:1220,1261`) with the masked traces persisted into the concat artifact, so the reloaded recording carries no wrapper; and selection joins are visible only through the persisted timestamps. `mode="noise"` is NOT equivalent (it synthesizes independent per-channel noise, which dilutes cross-channel covariance).

1. **Boundary spans.** Where the joins are visible differs by source, so they are computed where they are visible and carried forward as frame ranges:
   - Single-session `Recording` artifacts persist the original wall-clock timestamps (`timestamps_override`), so on the reloaded recording `base_intervals_and_gaps(recording)` (`_signal_math.py:736`) returns the `gap_after` frame indices of every selection join; spans are `[0, g0+1), [g0+1, g1+1), …, [gk+1, n)`, and a single-interval recording yields `[(0, n)]`. A test must confirm a reloaded two-interval artifact exposes its gap this way.
   - Concatenated recordings are built with `concatenate_recordings(recordings, ignore_times=True)` (`_concat_recording.py:695`) and persisted as a synthetic continuous timeline, so member-internal gaps are NOT recoverable from the concat artifact. `ConcatenatedRecording.make` must therefore compute each member's boundary spans from that member's own `Recording` artifact (real timestamps) BEFORE concatenation, offset them by the member's cumulative start frame (`cumulative_member_boundaries`, `_concat_recording.py:269`), and persist the offset list next to the member set. Member joins are implied by the offsets (each member's spans start at its own cumulative start).

   ```python
   def boundary_spans_from_timestamps(recording) -> list[tuple[int, int]]:
       """Half-open spans that never cross a wall-clock gap in ``recording``'s persisted timestamps."""
       n = recording.get_num_samples()
       cuts = sorted({0, n, *(int(g) + 1 for g in base_intervals_and_gaps(recording).gap_after)})
       return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


   def concat_boundary_spans(member_recordings, member_starts) -> list[tuple[int, int]]:
       """Union of each member's timestamp-derived spans, offset into concat frame coordinates."""
       out = []
       for rec, start in zip(member_recordings, member_starts):
           out.extend((start + a, start + b) for a, b in boundary_spans_from_timestamps(rec))
       return out
   ```

   `statistics_spans(...)` below takes the boundary spans as an argument; `Sorting.make_compute` obtains them from `boundary_spans_from_timestamps(recording)` for a single-session source and from the persisted concat list for a concatenated source. Test: a member with internal spans `[0,500)` and `[500,1000)` must still yield two spans after concatenation (the timestamp-only approach collapsed them into `[0,1000)`).

2. **Statistics spans = boundary spans ∩ artifact-free ranges.** `apply_artifact_mask` (`_sorting_artifact_mask.py:292`) returns `(masked_recording, excluded_ranges)`; then

   ```python
   def statistics_spans(n_samples, excluded_ranges, boundary_spans) -> list[tuple[int, int]]:
       """Artifact-free frame ranges that lie within one boundary span each."""
       valid = complement_frame_ranges(excluded_ranges, n_samples)   # [(0,n)] when nothing is excluded
       spans = boundary_spans
       out = []
       for a, b in valid:
           for c, d in spans:
               lo, hi = max(a, c), min(b, d)
               if lo < hi:
                   out.append((lo, hi))
       if not out:
           raise ValueError("no artifact-free samples inside any acquisition span")
       return sorted(out)
   ```

   `complement_frame_ranges` sorts/merges `excluded_ranges` and returns `[(0, n)]` for an empty list. `statistics_spans` must NEVER merge two output spans that are adjacent in frame coordinates: adjacency across a boundary span (a selection join, a member join) is exactly the information the spans exist to preserve, so no generic interval-union pass is applied to the result. Three distinct notions are kept separate throughout: *observed* (a sample was recorded and is inside the selected intervals), *continuous* (two frames are adjacent in acquisition time), and *statistically eligible* (observed, continuous with its neighbors in the piece, and artifact-free). A sample can be observed yet ineligible; two eligible spans can be adjacent yet discontinuous. `Sorting.make_compute` computes `statistics_spans` once (boundary spans from timestamps for a single-session source; the persisted offset member list for a concatenated source), passes the list to `_run_sorter` → `run_si_sorter` / `run_clusterless_thresholder` and to `_build_analyzer` → `build_analyzer`, and persists it in the sorting artifact's provenance scratch (the `_nwb_provenance` writers); `Sorting.get_statistics_spans(key)` returns the stored list for every later analyzer builder (`_curation_analyzer`, `metric_curation`, `recompute`). The 50% masked-fraction guard (`_signal_math.py:91`) bounds the excluded fraction; short spans are handled below, not rejected.

3. **Sampling within spans, without a fixed chunk length.** Covariance and MAD do not need contiguous half-second windows; only waveform snippets need a fixed length. Sample a target number of *samples* (not chunks), allocate them across spans proportionally to span length, and take contiguous pieces no longer than the span:

   ```python
   def sample_span_data(recording, spans, *, target_samples, max_piece, seed, return_in_uV):
       """Contiguous pieces drawn inside spans; each span contributes EXACTLY its length-proportional quota.

       ``max_piece`` caps a piece (e.g. 0.5 s worth of samples); the last piece of a span
       is trimmed so the span's total equals its quota. Returns (sum(quota), n_channels) float data.
       """
       import numpy as np
       rng = np.random.default_rng(seed)
       lengths = np.array([b - a for a, b in spans], dtype=np.int64)
       total_valid = int(lengths.sum())
       effective_target = min(int(target_samples), total_valid)
       if effective_target == total_valid:
           # The budget covers every valid sample: read each span once, no overlapping draws.
           data = np.concatenate(
               [recording.get_traces(start_frame=a, end_frame=b, return_in_uV=return_in_uV) for a, b in spans], axis=0
           )
           assert data.shape[0] == total_valid
           return data
       # Largest-remainder apportionment: quotas sum to effective_target exactly and never
       # exceed a span's length.
       raw = effective_target * lengths / total_valid
       quota = np.minimum(np.floor(raw).astype(np.int64), lengths)
       for i in np.argsort(-(raw - np.floor(raw))):
           if quota.sum() >= effective_target:
               break
           if quota[i] < lengths[i]:
               quota[i] += 1
       pieces = []
       for (a, b), q in zip(spans, quota):
           remaining = int(q)
           while remaining > 0:
               piece = min(max_piece, remaining, b - a)
               s = int(rng.integers(a, b - piece + 1))
               pieces.append(recording.get_traces(start_frame=s, end_frame=s + piece, return_in_uV=return_in_uV))
               remaining -= piece
       data = np.concatenate(pieces, axis=0)
       assert data.shape[0] == effective_target
       return data
   ```

   Contract: the returned row count is `min(target_samples, total_valid_samples)`; when the budget covers all valid data every valid sample is read exactly once. Weighting is by sample count and is enforced exactly: a span's share of the returned data equals its share of valid data (the earlier round-up-to-whole-pieces version let spans holding 9% of the data contribute 51% of the samples). With `target_samples = 20 * int(0.5 * fs)` (what SI's default 20 × 500 ms chunks amount to) and `max_piece = int(0.5 * fs)` the unmasked single-span case draws the same amount of data as today. Tests: 8 s of valid data split into 400 ms spans inside a 10 s recording must be sampled, not rejected; and on heterogeneous spans with different noise levels (e.g. a 9 s span at σ=1 and ten 100 ms spans at σ=5) the returned sample count equals `target_samples`, each span's contribution equals its quota, and the pooled MAD matches the length-weighted expectation within 2%.

   Fixed-length snippets (the nn noise cluster) keep a length requirement but per snippet, not per span: draw snippet starts uniformly from the set of positions `s` with `s + nsamples <= span_end` across all spans (weighted by each span's count of admissible starts); spans shorter than one snippet contribute no starts; raise only if no span admits a snippet.

4. **Whitening.** `pinned_whiten(recording, *, random_seed=0, spans=None)`: with spans, `data = sample_span_data(recording, spans, target_samples=20*int(0.5*fs), max_piece=int(0.5*fs), seed=random_seed, return_in_uV=False)`, then reproduce `compute_whitening_matrix`'s `mode="global"` math on `data` (read `whiten.py:151-220`: optional mean subtraction with `apply_mean=False`, `cov = data.T @ data / n`, eigendecomposition with SI's `eps` rule) to get `W, M`, and `sip.whiten(recording, dtype=np.float64, W=W, M=M)`. With `spans=None`, keep today's `sip.whiten(recording, dtype=np.float64, seed=random_seed)` unchanged so unmasked single-span sorts are bit-identical to before. Both call sites (`_sorting_dispatch.py:726`, `_sorting_analyzer.py:872-874`) pass the spans.

5. **Noise levels.** MAD per channel from `sample_span_data(...)` (the estimator `get_noise_levels` uses: `median(abs(x - median)) / 0.6744897501960817`), for both `return_in_uV=True` and `False`, cached on the recording under SpikeInterface's property keys `noise_level_mad_scaled` / `noise_level_mad_raw` (`recording_tools.py:687-760`) immediately before `create_sorting_analyzer` in `build_analyzer` and before `_clusterless_noise_levels` (`_sorting_dispatch.py:516`). The `noise_levels` extension calls `get_noise_levels(analyzer.recording, ...)`, which returns the cached property. Assert in a test that the extension's stored data equals the cached array; if the extension does not consult the property (check `ComputeNoiseLevels._run`, `analyzer_extension_core.py:762-800`), write the extension data directly after `compute` (`ext.data["noise_levels"] = levels; ext.save()`) and keep the test.

6. **nn noise cluster.** In `_nn_noise_overlap_sparse_fixed` (`_si_metric_patches.py:96-103`) replace `get_random_data_chunks(recording, ...)` with the fixed-length snippet sampler over the spans supplied through a `contextvars.ContextVar` set by `_compute_metrics` around the PC-metric compute (`metric_curation.py:2286-2300`); `None` → today's behavior.

7. Log the realized masked fraction and the number of statistics spans at INFO in `apply_artifact_mask`; rewrite the docstring at `_sorting_dispatch.py:618-621`.

**Baseline for tests.** The correctness target for masked-recording statistics is the *clean* recording before artifacts are injected (or, equivalently, statistics computed exactly on the retained samples). Never use the unmasked recording *with* planted artifacts as the target: its MAD is inflated by the artifacts themselves (observed 1.83 vs 0.995 on retained clean samples).

## geometry-normalization

**Problem.** Real Frank-lab tetrode files (e.g. `tests/_data/raw/minirec20230622.nwb`) store contacts in the x-z plane (rel_y = 0, rel_z = ±6.25). SpikeInterface reads rel_x/rel_y/rel_z as 3D locations and, when it constructs a probe (`get_probe()` → `create_dummy_probe_from_locations`, and inside `create_sorting_analyzer`), collapses to x-y first, producing duplicate positions and `ValueError: Contact positions must be unique within a probe`. `maybe_apply_tetrode_geometry` (`_recording_geometry.py:166-252`) repairs only the exact 4-channel `tetrode_12.5` case, in memory, and the repair is not persisted; `_sorting_analyzer.py:844-863` projects with `axes="xy"` and only warns.

**Design.** Normalize the raw 3D channel locations to a 2D plane *before* any probe is constructed, at the recording stage, on the channel-sliced recording (so groups with removed/excluded channels are normalized on the channels they actually contain). Persist the normalized 2D locations. Require every contact to be unique.

```python
def normalize_channel_locations(recording):
    """Reduce 3D electrode locations to the 2D plane that keeps every contact distinct.

    Prefers x-y; falls back to x-z when y is constant across the group and z is
    not; raises when no axis pair separates the contacts. Sets the recording's
    channel locations to the chosen 2D coordinates (no probe object is built here).
    """
    import numpy as np
    loc = recording.get_channel_locations(axes="xyz") if recording.get_property("location").shape[1] == 3 else recording.get_channel_locations()
    if loc.shape[1] == 2:
        return recording
    candidates = {"xy": loc[:, [0, 1]], "xz": loc[:, [0, 2]], "yz": loc[:, [1, 2]]}
    for axes in ("xy", "xz", "yz"):
        pos = candidates[axes]
        if len(np.unique(np.round(pos, 6), axis=0)) == len(pos):
            recording.set_channel_locations(pos)
            return recording
    return recording  # leave 3D; assert_unique_contact_positions raises after the tetrode repair


def assert_unique_contact_positions(recording) -> None:
    """Every contact must have a distinct 2D position (n_channels > 1)."""
    import numpy as np
    pos = np.asarray(recording.get_channel_locations())
    if len(pos) > 1 and len(np.unique(np.round(pos, 6), axis=0)) < len(pos):
        raise ValueError(
            "Recording.make: contacts share a 2D position after geometry normalization "
            f"(locations={pos.tolist()}). Fix Probe.Electrode rel_x/rel_y/rel_z for this sort group; "
            "the tetrode_12.5 repair applies only to 4-channel single-group tetrodes."
        )
```

Single-channel groups pass trivially. Confirm SpikeInterface's `get_channel_locations(axes="xyz")` / `set_channel_locations` semantics on the installed version (the `location` property may be 2D or 3D). Order in `Recording.make_compute`: `select_sort_group_channels` → `normalize_channel_locations` → temporal preprocessing → restriction → spatial preprocessing → `maybe_apply_tetrode_geometry` → `assert_unique_contact_positions`. The tetrode repair still applies (it overrides an all-zero legacy geometry that normalization cannot fix); the assertion runs after it so the *effective* geometry is what is checked.

**Persistence.** In `write_nwb_artifact` (`_recording_nwb.py:184`), after `io.write(nwbfile)` and before the content hash (350-358), write the recording's 2D channel locations (not a probe — `get_channel_locations()` works whether or not a probe is attached) into the artifact's electrodes table rows referenced by the series: `rel_x`, `rel_y` from the 2D coordinates and `rel_z = 0`. Row order equals `recording.get_channel_ids()` order (the order `electrode_table_region` used); assert equal lengths, read back, assert equality. On reload, SpikeInterface sees a constant-z 3D table and its x-y projection is exactly the normalized geometry. The `_sorting_analyzer.py:844-863` projection branch then only needs the uniqueness assertion (keep the warning for genuinely non-planar probes).

**Preflight.** In the existing `Probe.Electrode` geometry check in `_pipeline_preflight.py` (grep `rel_x`), compute the *effective* 2D geometry the same way (`normalize` on the group's electrode rows, then the tetrode repair predicate from `maybe_apply_tetrode_geometry`'s gates) and report an error when any two contacts coincide, naming file, sort group, and positions. Share the plane-selection code with `normalize_channel_locations` by factoring the pure array function `select_distinct_plane(loc3d) -> (axes, loc2d) | None` into `_recording_geometry.py`.

## per-unit-halves

**Problem.** `extract_unitmatch_bundle` (`_unitmatch_backend.py:208-231`) builds UnitMatch's two cross-validation template halves from the first and second halves of the *recording*; SpikeInterface zero-fills a unit's template when it has no sampled spikes in a half. Measured: drift-in/drift-out units get 0/50 true matches; calibration (match prior 0.054→0.043, non-match kernels) shifts session-wide.

**Design.** One dense analyzer on the whole session; split each unit's sampled spikes by temporal order (first half → cv 0, second half → cv 1), as UnitMatchPy's own `extract_raw_data` does. Reference implementation and measured results: [appendix-unitmatch-experiment.py](appendix-unitmatch-experiment.py) (the review's 10-seed experiment; its FIXED condition is this design) and [appendix-unitmatch-results.md](appendix-unitmatch-results.md).

```python
analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
analyzer.compute("random_spikes", method="uniform",
                 max_spikes_per_unit=2 * max_spikes_per_unit, seed=random_seed)
analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after, **compute_job_kwargs)
wf_ext = analyzer.get_extension("waveforms")
selected = analyzer.get_extension("random_spikes").get_random_spikes()  # fields: sample_index, unit_index, segment_index
unit_ids = np.asarray(sorting.get_unit_ids(), dtype=int)
n_samples = wf_ext.nbefore + wf_ext.nafter
avg_waves = np.zeros((len(unit_ids), n_samples, n_channels, 2))
excluded: list[int] = []
for unit_index, unit_id in enumerate(sorting.get_unit_ids()):
    wfs = wf_ext.get_waveforms_one_unit(unit_id, force_dense=True)   # (n_sel, n_samples, n_channels), selection order
    sample_index = selected["sample_index"][selected["unit_index"] == unit_index]
    order = np.argsort(sample_index, kind="stable")
    wfs = wfs[order]
    if len(wfs) < 2:
        excluded.append(int(unit_id))
        continue
    half = len(wfs) // 2
    avg_waves[unit_index, :, :, 0] = wfs[:half].mean(axis=0)
    avg_waves[unit_index, :, :, 1] = wfs[half:].mean(axis=0)
keep = np.array([i for i, u in enumerate(unit_ids) if int(u) not in excluded], dtype=np.intp)
if keep.size == 0:
    raise ValueError(
        f"extract_unitmatch_bundle: every unit in {session_dir} has fewer than two sampled "
        "spikes; the session cannot be matched. Lower max_spikes_per_unit or exclude the session."
    )
avg_waves, unit_ids = avg_waves[keep], unit_ids[keep]
assert not np.any(np.all(avg_waves == 0, axis=(1, 2))), "a template half is all-zero"
```

Then the existing `save_avg_waveforms` / `channel_positions.npy` / `cluster_group.tsv` writes with the kept `unit_ids`. `keep` is an explicit integer index (an empty Python list would otherwise become a float64 array and `avg_waves[keep]` would raise `IndexError`). A session with no matchable units raises rather than writing an empty bundle, so the session ↔ bundle mapping in `UnitMatch.make` is never silently shortened.

**Exclusion route.** `extract_unitmatch_bundle` returns the `excluded` list. Its caller is `UnitMatch.make` (`unit_matching.py:1277`), which extracts every session's bundle *before* calling `UnitMatchBackend.match` (`_unitmatch_backend.py:328`, returns `list[MatchPair]`; there is no receipt object). `make` collects `{session_key: excluded}` from the extraction loop, logs one WARNING per session with exclusions, and passes nothing extra to `match` — the backend reads the bundles, and excluded units are simply absent from `cluster_group.tsv`. The frozen matchable universe (`MatchableUnit`) is not altered; excluded units produce no pairs. `_zero_center` stays where it is (`match`, line 401). `max_spikes_per_unit` keeps its "per half" meaning for users (draw 2×, split); update the docstring and the `UnitMatchParamsSchema` description.

Verify `get_waveforms_one_unit` returns waveforms in `selected` order for that unit (SpikeInterface sorts selected indices by sample index; the `argsort` above is a no-op guard).

## metric-missingness

**Problem.** `apply_label_rules` (`_metric_curation.py:108`) treats every non-finite metric as "not assessable"; all four shipped rule sets use `missing_policy="pass"` (`metric_curation.py:594-654`), so a metric that failed for every unit yields "no units flagged" with one warning, and `"fail"` labels every unit with no warning. SpikeInterface swallows metric exceptions to NaN both in the calculator (with a "Error computing metric" warning) and inside the nn metrics (silently, `pca_metrics.py:185-194`). `_compute_metrics` also computes *template* metrics (`QualityMetricParameters.template_metric_columns`, `metric_curation.py:284`, `trough_half_width` by default), and SpikeInterface's template metrics return NaN silently whenever the required waveform feature is absent (`metrics/template/metrics.py:83-91,265`).

**Design (narrowed, owner 2026-09-18).** The regression is an auto-curation *rule* going silently inert, so the failure/missingness distinction is enforced only on columns that the evaluated rule set references. All other computed columns — custom quality metrics, template metrics, anything not in a rule — are left exactly as SpikeInterface produced them (NaN included) and never cause a failure. Within rule-referenced columns: a NaN for a unit that meets the metric's own preconditions is a computation failure and raises regardless of `missing_policy`; a NaN for a unit that does not meet them is expected and is governed by the policy. A rule that references a column with no registered eligibility rule fails closed only when a NaN is present, with a message telling the user to register an eligibility rule for that metric (or to switch to a metric that has one). This keeps custom metrics usable and avoids reverse-engineering every SI metric.

| column | legitimate NaN when (SI 0.104.3 source) | eligibility inputs |
| --- | --- | --- |
| `nn_isolation`, `nn_noise_overlap` | `n_spikes < min_spikes` or `fr < min_fr`, where `fr = n_spikes / sorting_analyzer.get_total_duration()` (`pca_metrics.py:150-200`) | n_spikes, SI total duration (recording duration, NOT observed duration) |
| `presence_ratio` | recording duration shorter than `bin_duration_s` (`misc_metrics.py:95-110`) | SI total duration, `bin_duration_s` |
| `amplitude_cutoff` | `n_spikes / num_histogram_bins < amplitudes_bins_min_ratio` (`misc_metrics.py:~1670`, returns NaN; defaults 100 bins × ratio 5 = 500 spikes), or SI's "no cutoff found" path (NaN with a warning) | n_spikes, `num_histogram_bins`, `amplitudes_bins_min_ratio`, captured SI warnings |
| `isi_violation` (Spyglass's `isi_violation_fraction`, `_metric_curation.py:239-292`, `count / (n_spikes - 1)`) | `n_spikes <= 1` | n_spikes |
| `snr`, `firing_rate`, `num_spikes` | never (confirm by reading each `compute_*`) | — |

That table is the initial registry: the metrics the shipped rule sets reference (`nn_noise_overlap`, `isi_violation`) plus the common ones whose rules are cheap. Template metrics and every other SI metric are deliberately NOT registered; they stay NaN-tolerant unless a rule references them, in which case the fail-closed message above applies.

Implement as a registry of per-column eligibility predicates in `_metric_curation.py` (DB-free):

```python
_ELIGIBILITY_RULES: dict[str, Callable] = {...}   # the table above


def expected_missing_units(rule_columns, *, n_spikes_by_unit, total_duration_s, metric_kwargs,
                           si_warned_units) -> dict[str, set[int]]:
    """For each rule-referenced column, unit ids for which SI legitimately returns NaN.

    Rules mirror SI 0.104.3's own guards. ``total_duration_s`` is the analyzer's recording
    duration (``sorting_analyzer.get_total_duration()``). ``si_warned_units`` maps column ->
    unit ids SI warned about. A column without a registered rule maps to ``None``.
    """


def assert_rule_metrics_computed(metrics_df, rule_columns, expected_missing) -> None:
    """Raise if a rule-referenced column is non-finite where a value is required.

    Registered column: NaN outside ``expected_missing[col]`` -> ValueError listing units.
    Unregistered column: any NaN -> ValueError telling the user to register an eligibility
    rule for that metric or choose a metric that has one. Columns not referenced by a rule
    are never inspected.
    """
```

Test the registered rules against real SI output on synthetic sortings that hit each condition (2 s recording for presence_ratio; a 100-spike unit for amplitude_cutoff with default bins/ratio; a 5-spike unit for nn), asserting the classifier agrees with where SI produced NaN and that replacing one finite value with NaN is flagged; test that a custom/template column not in any rule may be all-NaN without any failure; test the unregistered-rule-column message.

`apply_label_rules` gains `expected_missing: dict[str, set[int] | None] | None = None`; with it, a non-finite value for a unit outside a registered column's expected set raises regardless of policy, and the all-units-missing branch warns for `"fail"` as well as `"pass"`. `make_compute` supplies `n_spikes_by_unit` from the analyzer's sorting, `total_duration_s = analyzer.get_total_duration()`, the captured SI warnings, and `rule_columns` from the evaluated rule set.

Belt-and-braces, scoped to the same promise: wrap both `compute_quality_metrics` calls and the template-metric compute in `_compute_metrics` with `warnings.catch_warnings(record=True)`; escalate an "Error computing metric <name>" warning to `ValueError` only when `<name>` produces a rule-referenced column; log the rest at WARNING. Attribute metric-specific warnings (amplitude cutoff) to units for the classifier. An unreferenced metric that raises inside SI therefore yields a NaN column and a log line, never an aborted evaluation.

Schema (allowed): `observed_presence_bin_duration_s=60: double` (`metric_curation.py:286`) and `AutoCurationRules.Rule.threshold` → `double`; bump the params schema versions following the existing convention (grep `params_schema_version` in `_params/metric_curation.py`).

## mua-contiguous-runs

**Problem.** `MuaEventsV1.make` (`mua/v1/mua.py:89-111`) sums `get_spike_indicator` output, which now carries NaN in unobserved bins for v2-containing groups, and passes it to `multiunit_HSE_detector`. Dropping the NaN bins and concatenating what remains removes the NaN but joins samples across missing time: in a reproduction, two bursts separated by a 380 ms unobserved interval merged into one 499 ms event spanning the gap; detecting on each contiguous run separately produced two ~60 ms events.

**Design — one algorithm, preserving the detector's parameters.** The installed `ripple_detection.multiunit_HSE_detector` (`detectors.py:650-664`) takes `speed_threshold`, `minimum_duration`, `zscore_threshold`, `smoothing_sigma`, `close_event_threshold`, `use_speed_threshold_for_zscore` (deprecated), `normalization_method` (`"zscore"` | `"median_mad"`), `normalization_mask`, and `normalization_time_range`. Its selector precedence and validation live in `core.py:452` (`_validate_normalization_params`: supplying both `normalization_mask` and `normalization_time_range` is an error) and `core.py:496` (`_get_normalization_mask`: an explicit mask or time range wins; the speed threshold is used only when neither explicit selector is supplied). All of these must keep their meaning — reuse those two functions; do NOT intersect the selectors. (1) Smooth each contiguous observed run separately, so the Gaussian kernel never crosses a gap. (2) Normalize ONCE across ALL observed smoothed samples (including runs too short to hold an event — they still carry baseline information) with the detector's own normalization function and `normalization_method`, using the mask that `_get_normalization_mask` returns for the observed samples (a user-supplied `normalization_mask` is sliced to the observed samples first; a `normalization_time_range` is applied to the observed times). (3) Extract events separately per run from the shared normalized rate, using the detector's extraction with `zscore_threshold`, `minimum_duration`, `close_event_threshold`, and the speed threshold; runs shorter than `minimum_duration` are skipped only at this step.

```python
spike_indicator, valid = SortedSpikesGroup.get_spike_indicator(key, time, return_validity=True)
spike_indicator = spike_indicator.sum(axis=1, keepdims=True)          # (n_time, 1)
mask = np.zeros_like(time, dtype=bool)
for start, end in valid_times:
    mask |= (time >= start) & (time <= end)
mask &= valid
dt = 1.0 / sampling_frequency
breaks = np.r_[True, (~mask[:-1]) | (~mask[1:]) | (np.diff(time) > 1.5 * dt)]
run_ids = np.cumsum(breaks)
runs = [np.flatnonzero(mask & (run_ids == r)) for r in np.unique(run_ids[mask])]   # ALL observed runs
observed = np.concatenate(runs) if runs else np.array([], dtype=int)

# (1) smooth per run; get_multiunit_population_firing_rate returns a 1-D array (core.py:904-926)
rate = np.full(time.shape, np.nan)
for sel in runs:
    rate[sel] = get_multiunit_population_firing_rate(spike_indicator[sel], sampling_frequency, smoothing_sigma)

# (2) one normalization over ALL observed samples, with the detector's own selector precedence
_validate_normalization_params(normalization_mask, normalization_time_range)          # core.py:452
user_mask = None if normalization_mask is None else np.asarray(normalization_mask)[observed]
norm_mask = _get_normalization_mask(                                                  # core.py:496
    time[observed], speed[observed], speed_threshold, use_speed_threshold_for_zscore,
    user_mask, normalization_time_range,
)
normalized = np.full(time.shape, np.nan)
normalized[observed] = normalize(rate[observed], method=normalization_method, mask=norm_mask)

# (3) events per run, shared normalization; only here are runs too short for an event skipped
min_run = int(np.ceil(minimum_duration * sampling_frequency))
mua_times = [
    extract_events(time[sel], normalized[sel], speed[sel], sampling_frequency,
                   speed_threshold=speed_threshold, minimum_duration=minimum_duration,
                   zscore_threshold=zscore_threshold, close_event_threshold=close_event_threshold)
    for sel in runs if sel.size >= min_run
]
if mua_times:
    # Each run's table restarts its 1-based ``event_number`` index; two runs would yield
    # duplicate indices ([1, 1]) that survive into the NWB DynamicTable. Renumber
    # chronologically, one-based, after concatenation.
    mua_times = pd.concat(mua_times, ignore_index=True).sort_values("start_time", kind="stable")
    mua_times.index = pd.RangeIndex(1, len(mua_times) + 1, name="event_number")
else:
    mua_times = empty_events_frame()
```

`normalize` and `extract_events` are the normalization and event-extraction pieces of `multiunit_HSE_detector` (read `detectors.py:650-760` and `core.py:452-600`; import ripple_detection's functions directly — `_validate_normalization_params`, `_get_normalization_mask`, the `_normalize_*` dispatch — rather than re-implementing them; match their exact signatures, which the executor must read, since the argument order above is illustrative). The helper as written is executed in its regression tests, not only described. Tests: with a single fully observed run the result equals the current `multiunit_HSE_detector` output exactly, for the default parameters AND for each of `use_speed_threshold_for_zscore=True`, `normalization_method="median_mad"`, a `normalization_time_range`, and an explicit `normalization_mask`; supplying both `normalization_mask` and `normalization_time_range` raises the same error the detector raises; with two runs separated by a gap, no event spans the gap and the concatenated table's `event_number` index is unique, one-based, and chronological (`[1, 2]`, not `[1, 1]`), verified through the `MuaEventsV1` NWB write and read-back; a run shorter than `minimum_duration` yields no events but IS part of the normalization population (removing it from normalization would change another run's peak z-score — observed 5.60 → 5.84 in a probe); zero events on every run yields the current empty-result shape.
