# Spike Sorting V2 Signal Units / Preprocessing Semantics Review

Date: 2026-06-25

Scope: signal scaling, preprocessing order, reference semantics, bad-channel handling, sorter/analyzer unit conventions, peak polarity, probe geometry, Units NWB signal metadata, and tests/docs that protect those contracts.

Method: main-agent source review plus two independent read-only agents. I also ran one local SpikeInterface 0.104 runtime probe with the `spyglass_spikesorting_v2` conda Python to check how `bandpass_filter` and `common_reference` propagate channel offsets.

## Findings

### 1. Medium-high: `no_filter` + referencing can persist a re-referenced signal with the original DC offset reattached

Evidence:

- `bandpass_filter=None` is a supported preprocessing mode and still allows referencing: [src/spyglass/spikesorting/v2/_params/preprocessing.py:143](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_params/preprocessing.py:143), [tests/spikesorting/v2/test_preprocessing_order.py:181](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_preprocessing_order.py:181).
- The runtime applies `common_reference` directly on that path: [src/spyglass/spikesorting/v2/_recording_preprocessing.py:187](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_preprocessing.py:187), [src/spyglass/spikesorting/v2/_recording_preprocessing.py:227](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_preprocessing.py:227).
- The NWB writer then writes transformed traces with `return_in_uV=False`, but derives `ElectricalSeries.conversion` and `offset` from the post-preprocessing recording and persists both: [src/spyglass/spikesorting/v2/_recording_nwb.py:182](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_nwb.py:182), [src/spyglass/spikesorting/v2/_recording_nwb.py:204](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_nwb.py:204), [src/spyglass/spikesorting/v2/_recording_nwb.py:240](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_nwb.py:240).
- `resolve_conversion_and_offset` intentionally preserves a uniform nonzero offset: [src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:45](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:45), [tests/spikesorting/v2/test_conversion_offset.py:39](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_conversion_offset.py:39).

Runtime probe result: in the current SI environment, `sip.bandpass_filter` resets channel offsets to zero, but `sip.common_reference` preserves the parent offsets while subtracting traces. That makes the default filter-then-reference path safe, because the filter has already zeroed offsets. It leaves the supported `bandpass_filter=None` path exposed: for a uniform-offset recording, `specific` or `global_median` reference mathematically cancels the offset, but readback with `return_in_uV=True`, artifact detection, clusterless `scale_to_uV`, and display analyzers can add it back.

Impact: no-filter referenced recordings with nonzero acquisition offsets can have artifact thresholds, clusterless thresholds, display waveforms, and standalone NWB readback biased by a DC offset that should have cancelled during re-referencing.

Fix direction:

- After any `common_reference` step, set channel offsets to zero on the referenced recording, while preserving gains.
- Add a hermetic test with nonzero offsets, `bandpass_filter=None`, and `reference_mode in {"specific", "global_median"}` asserting `get_channel_offsets()` and `get_traces(return_in_uV=True)` reflect the referenced signal with no reattached DC.
- Keep the no-filter/no-reference path preserving offsets; that path has not cancelled the offset.

### 2. Medium-high: SNR quality metrics assume negative peaks while unit attribution honors sorter polarity

Evidence:

- Default metric rows hard-code `snr: {"peak_sign": "neg"}`: [src/spyglass/spikesorting/v2/metric_curation.py:254](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:254), [src/spyglass/spikesorting/v2/metric_curation.py:272](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:272).
- Metric computation passes those kwargs to SI unchanged: [src/spyglass/spikesorting/v2/metric_curation.py:1102](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:1102).
- Sort-time `Sorting.Unit` peak channel/amplitude attribution resolves the sorter polarity from `peak_sign` / `detect_sign`: [src/spyglass/spikesorting/v2/sorting.py:2599](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:2599), [src/spyglass/spikesorting/v2/utils.py:354](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/utils.py:354).

Impact: positive-going or bidirectional detections can have sort-time peak metadata computed with one polarity but SNR/auto-curation computed with another. That can understate SNR or threshold the wrong extremum, especially for clusterless `peak_sign="pos"` / `"both"` or MountainSort `detect_sign=1` / `0`.

Fix direction:

- Derive the SNR `peak_sign` from the sorter's resolved polarity unless the metric row explicitly overrides it.
- Alternatively, validate `QualityMetricParameters` against the sorter row and require an explicit mismatch acknowledgement.
- Add a synthetic positive-going unit test that verifies SNR and auto-curation labels follow `resolve_peak_sign`.

### 3. Medium: Units NWBs preserve spike trains but omit the per-unit signal metadata advertised by the API/docs

Evidence:

- The docs/notebook advertise per-unit quick metadata such as `n_spikes`, `peak_amplitude_uv`, `peak_electrode_id`, and `brain_region`: [docs/src/Features/SpikeSortingV2.md:77](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:77), [notebooks/py_scripts/10_Spike_SortingV2.py:232](/Users/edeno/Documents/GitHub/spyglass/notebooks/py_scripts/10_Spike_SortingV2.py:232).
- The `Sorting.Unit` part table stores `peak_amplitude_uv` and `n_spikes` in the DB: [src/spyglass/spikesorting/v2/sorting.py:1162](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:1162).
- The sort-time Units NWB write path adds only `curation_label` and `spike_sample_index` columns before `add_unit`: [src/spyglass/spikesorting/v2/_units_nwb.py:626](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:626), [src/spyglass/spikesorting/v2/_units_nwb.py:649](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:649).
- The curated Units NWB path similarly centers labels/sample indices, not peak amplitude/electrode/region metadata: [src/spyglass/spikesorting/v2/_units_nwb.py:864](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:864).

Impact: a detached exported Units NWB can preserve spike times but not enough signal context to interpret amplitude, peak electrode, or region without the Spyglass DB. That weakens portability and makes the `peak_amplitude_uv` unit convention DB-only.

Fix direction:

- Add Units columns for `peak_amplitude_uv`, `peak_electrode_id`, `n_spikes`, and region where available, or document explicitly that these are DB-only summaries.
- Add PyNWB round-trip tests for sort-time and curated exports.

### 4. Medium: non-planar 3D probe geometry is silently projected to 2D apart from a warning

Evidence:

- Analyzer construction fetches the probe and projects any 3D probe with `probe.to_2d()`: [src/spyglass/spikesorting/v2/_sorting_analyzer.py:555](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:555), [src/spyglass/spikesorting/v2/_sorting_analyzer.py:574](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:574).
- Non-degenerate z only logs a warning: [src/spyglass/spikesorting/v2/_sorting_analyzer.py:561](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:561), [src/spyglass/spikesorting/v2/_sorting_analyzer.py:566](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:566).

Impact: dropping real z-axis structure changes channel distances, sparsity, and unit-location metrics. A warning is easy to miss in batch populate, so a non-planar probe can produce scientifically different analyzer geometry without an explicit operator decision.

Fix direction:

- Fail by default when z is non-degenerate, or require an explicit projection policy.
- Record the projection policy in analyzer provenance.
- Add a non-planar probe test asserting hard failure or declared projection behavior.

### 5. Medium: preprocessing implementation order changed without a schema/identity bump

Evidence:

- The schema docs state the runtime order changed from reference-then-filter to filter-then-reference, but `schema_version` was not bumped because the blob shape was unchanged: [src/spyglass/spikesorting/v2/_params/preprocessing.py:116](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_params/preprocessing.py:116).
- `RecordingSelection` identity hashes the preprocessing row name, not an implementation/order version: [src/spyglass/spikesorting/v2/_selection_identity.py:40](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_selection_identity.py:40), [src/spyglass/spikesorting/v2/recording.py:768](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:768).
- The current docs/tests correctly pin filter-before-reference behavior: [docs/src/Features/SpikeSortingV2.md:59](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:59), [tests/spikesorting/v2/test_preprocessing_order.py:141](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_preprocessing_order.py:141).

Impact: an older and newer environment can interpret the same preprocessing row name/blob as different scientific signal transforms. The persisted `ElectricalSeries.filtering` and content fingerprint help after materialization, but selection identity and parameter provenance do not distinguish the two semantics before populate/recompute.

Fix direction:

- Add an explicit preprocessing implementation/order version to the params blob or the selection identity, or bump the schema version when transform semantics change even if the shape does not.
- Add a stale-row diagnosis test so old rows fail or report why their semantics are ambiguous.

### 6. Medium-low: peak-amplitude integration coverage admits it is not discriminating for off-center extrema

Evidence:

- The integration test itself notes that the current fixture passes with or without the `mode="extremum"` fix because templates are trough-aligned: [tests/spikesorting/v2/test_unit_peak_amplitude.py:17](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_unit_peak_amplitude.py:17).
- The code now correctly asks SI for `mode="extremum"` and the resolved peak sign: [src/spyglass/spikesorting/v2/sorting.py:2613](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:2613), [src/spyglass/spikesorting/v2/sorting.py:2619](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:2619).

Impact: a regression to center-sample/`at_index` amplitude behavior could still pass the main integration fixture for trough-aligned data.

Fix direction:

- Add a hermetic synthetic analyzer where the true extremum is deliberately off-center and on a polarity-dependent channel, then assert stored/row-builder amplitude and electrode use the extremum.

### 7. Low: clusterless smoke row name still says `5uv` while behavior is explicitly MAD

Evidence:

- `SMOKE_CLUSTERLESS_PARAM_NAME = "smoke_clusterless_5uv"` but the params set `threshold_unit="mad"`: [tests/spikesorting/v2/_smoke_constants.py:24](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/_smoke_constants.py:24), [tests/spikesorting/v2/_smoke_constants.py:38](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/_smoke_constants.py:38).
- A nearby test comment still carries stale native-count/microvolt language even though the runtime now has an explicit unit contract: [tests/spikesorting/v2/single_session/test_sorting.py:451](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_sorting.py:451).

Impact: this is test/support naming, not production behavior, but it invites future copy-paste unit confusion around `5 uV`, raw counts, and `5x MAD`.

Fix direction:

- Rename to `smoke_clusterless_5mad` or keep a compatibility alias with clear comments.
- Add a tiny assertion that smoke row names and `threshold_unit` do not contradict each other.

## Positives

- The default filter-then-reference path is protected by runtime order tests and Nyquist checks: [src/spyglass/spikesorting/v2/_recording_preprocessing.py:128](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_recording_preprocessing.py:128), [tests/spikesorting/v2/test_preprocessing_order.py:119](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_preprocessing_order.py:119).
- Raw-count/microvolt handling is much stronger than before: the NWB writer preserves conversion and offset, and heterogeneous gains/offsets fail loudly: [src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:15](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:15), [tests/spikesorting/v2/test_conversion_offset.py:48](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_conversion_offset.py:48).
- Artifact detection and clusterless `threshold_unit="uv"` explicitly use stored gain/offset: [src/spyglass/spikesorting/v2/_artifact_compute.py:121](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_artifact_compute.py:121), [src/spyglass/spikesorting/v2/_sorting_dispatch.py:365](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_dispatch.py:365).
- Peak sign now flows into sort-time unit attribution, and there are hermetic tests for positive-going attribution: [src/spyglass/spikesorting/v2/sorting.py:2609](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:2609), [tests/spikesorting/v2/test_peak_sign_resolution.py:93](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_peak_sign_resolution.py:93).
- Bad-channel remove/interpolate behavior and interpolation ordering have explicit tests and docs: [tests/spikesorting/v2/test_bad_channel_handling.py:1](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_bad_channel_handling.py:1), [docs/src/Features/SpikeSortingV2.md:703](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:703).
