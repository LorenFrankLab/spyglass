# Feature parity with v1

[← back to PLAN.md](PLAN.md)

This page is the parity contract. It distinguishes behavior v2 must preserve from intentional table/API departures. When in doubt, implementers should preserve downstream behavior and user workflows, not v1's exact table names.

Runtime compatibility boundary: v2 is the supported runtime path under the
SpikeInterface 0.104 environment introduced by Phase 0c. Existing v0/v1 rows
remain queryable where possible, but active v0/v1 population, MetricCuration,
BurstPair, FigURL, and recompute workflows require the legacy SI 0.99
environment unless Phase 0c explicitly ports a narrow compatibility shim. The
parity targets below describe user-facing workflow successors in v2, not a
promise that v1 runtime code is modernized under SI 0.104.

| v1 surface | v2 parity target | Notes |
| --- | --- | --- |
| `SortGroup.set_group_by_shank` | `SortGroupV2.set_group_by_shank` plus column-based grouping | v2 adds guarded deletion and non-shank grouping. Silent overwrite is intentionally not preserved. |
| `SpikeSortingPreprocessingParameters`, `SpikeSortingRecordingSelection`, `SpikeSortingRecording` | `PreprocessingParameters`, `RecordingSelection`, `Recording` | Same workflow shape; v2 adds single-dict `insert_selection()` returns and missing-cache rebuild. |
| `ArtifactDetectionParameters`, `ArtifactSelection`, `ArtifactDetection` | Same named v2 concepts | v2 writes the artifact-removed interval to `common.IntervalList` under name `f"artifact_{artifact_id}"` using `insert1()` (not v1's bare-UUID name + `skip_duplicates=True`); `SharedArtifactGroup` detections write one row per member `nwb_file_name`. `get_artifact_removed_intervals()` is a thin `IntervalList.fetch1("valid_times")` wrapper. |
| `SpikeSorterParameters` | `SorterParameters` | v2 ships dedicated defaults for MS4, MS5, KS4, `clusterless_thresholder`, `spykingcircus2`, and `tridesclous2`. `clusterless_thresholder` remains Spyglass' SI-peak-detection special case rather than an SI registered sorter. Other SpikeInterface sorters are supported through explicit custom parameter rows, not auto-inserted defaults for every installed sorter. |
| `SpikeSortingSelection`, `SpikeSorting` | `SortingSelection`, `Sorting` | Same single-session output behavior plus SortingAnalyzer folder and `Sorting.Unit` metadata. |
| `CurationV1` | `CurationV2` | Same lineage role. v2 auto-registers into `SpikeSortingOutput`, validates labels, and normalizes labels into `UnitLabel`. Preserve notebook-facing accessors including `get_sorting`, `get_merged_sorting`, and `get_sort_group_info`; v2's sort-group info returns all electrodes rather than v1's `fetch(limit=1)` sample. |
| `MetricParameters`, `WaveformParameters`, `MetricCurationParameters`, `MetricCuration` | `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCuration` | v2 consolidates metric computation and auto-curation around SortingAnalyzer extensions. Preserve the user-facing fetch/promote surface: `get_waveforms`, `get_metrics`, `get_labels`, `get_merge_groups`, and `materialize_curation()` as the explicit analog of `CurationV1.insert_metric_curation()`. |
| `BurstPairParams`, `BurstPairSelection`, `BurstPair` | No table clone; folded into `AnalyzerCuration` | Preserve the user-facing auto-merge behavior and visualization helpers, including `insert_by_curation_id` analog, `plot_by_sort_group_ids`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time`. |
| `FigURLCurationSelection`, `FigURLCuration` | `FigPackCurationSelection`, `FigPackCuration` | Intentional backend replacement. v1 FigURL remains for v1 data; v2 does not extend FigURL. Preserve an explicit UI-builder helper analogous to `generate_curation_uri`, with the exact FigPack API pinned by the Phase 5 feasibility check. |
| `RecordingRecompute*` | `RecordingArtifactRecompute*` plus `SortingAnalyzerRecompute*` | v2 uses new names to avoid v1 table-name collisions and extends the pattern to analyzer folders. Preserve the operational/admin surface unless explicitly dropped in Phase 2: attempt-all, matched-row removal/deletion, disk-space reporting, name joins, parent-key lookup, recheck, and secondary-field update equivalents. |
| `ImportedSpikeSorting` | Existing shared `ImportedSpikeSorting` remains canonical | Do not duplicate this as a v2 table. Imported NWB Units continue to enter `SpikeSortingOutput.ImportedSpikeSorting` and can be used beside v2 rows by downstream consumers and validation fixtures. |
| `SpikeSortingOutput` merge dispatch | Existing merge table plus `CurationV2` part | `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, and `get_firing_rate` must work for v2 merge IDs. |
| `get_restricted_merge_ids` and `get_spiking_sorting_v1_merge_ids` convenience lookup | `SpikeSortingOutput.get_restricted_merge_ids(..., sources=['v2'])` | v2 must support interpretable restrictions across recording, artifact, sorting, curation, and analyzer-curation stages. No v2-specific duplicate of the v1 utility is required. |
| `MetricParameters.show_available_metrics()` | `QualityMetricParameters.show_available_metrics()` or documented `available_quality_metrics()` helper | Preserve a discoverable notebook helper for the metric names v2 supports. It may report SortingAnalyzer/SpikeInterface metric names rather than v1's exact custom metric set. |

## Explicit non-parity

- v2 does not promise identical table names or primary keys.
- v2 does not preserve v1's list-returning `insert_selection()` behavior.
- v2 does not auto-create default rows for every installed SpikeInterface sorter.
- v2 does not migrate existing v1 curation rows into `CurationV2`.
- v2 does not guarantee active v0/v1 runtime workflows under the SI 0.104 environment. Legacy runtime workflows require the SI 0.99 environment unless explicitly ported by Phase 0c.
- v2 does not implement `SpikeSortingRecording.update_ids()`. That v1 helper was a transition/backfill tool for recompute-era NWB files; v2's zero-migration policy requires final row fields when each table is introduced.
- v2 does not clone the standalone `spike_times_to_valid_samples()` helper. The boundary-spike behavior is preserved inside `Sorting.make()` by running SpikeInterface's `remove_excess_spikes()` and by the Phase 1 boundary invariant test.
- v2 does not remove, deprecate, or rewrite v0/v1.
