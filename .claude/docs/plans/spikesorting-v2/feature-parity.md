# Feature parity with v1

[← back to PLAN.md](PLAN.md)

This page is the parity contract. It distinguishes behavior v2 must preserve from intentional table/API departures. When in doubt, implementers should preserve downstream behavior and user workflows, not v1's exact table names.

| v1 surface | v2 parity target | Notes |
| --- | --- | --- |
| `SortGroup.set_group_by_shank` | `SortGroupV2.set_group_by_shank` plus column-based grouping | v2 adds guarded deletion and non-shank grouping. Silent overwrite is intentionally not preserved. |
| `SpikeSortingPreprocessingParameters`, `SpikeSortingRecordingSelection`, `SpikeSortingRecording` | `PreprocessingParameters`, `RecordingSelection`, `Recording` | Same workflow shape; v2 adds single-dict `insert_selection()` returns and missing-cache rebuild. |
| `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` | Same named v2 concepts | v2 does not write artifact-removed intervals to `IntervalList`; intervals live on `ArtifactDetection.Interval` and are exposed through `get_artifact_removed_intervals()`. |
| `SpikeSorterParameters` | `SorterParameters` | v2 ships dedicated defaults for MS4, MS5, KS4, `clusterless_thresholder`, `spykingcircus2`, and `tridesclous2`. Other SpikeInterface sorters are supported through explicit custom parameter rows, not auto-inserted defaults for every installed sorter. |
| `SpikeSortingSelection`, `SpikeSorting` | `SortingSelection`, `Sorting` | Same single-session output behavior plus SortingAnalyzer folder and `Sorting.Unit` metadata. |
| `CurationV1` | `CurationV2` | Same lineage role. v2 auto-registers into `SpikeSortingOutput`, validates labels, and normalizes labels into `UnitLabel`. |
| `MetricParameters`, `WaveformParameters`, `MetricCurationParameters`, `MetricCuration` | `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCuration` | v2 consolidates metric computation and auto-curation around SortingAnalyzer extensions. |
| `BurstPairParams`, `BurstPairSelection`, `BurstPair` | No table clone; folded into `AnalyzerCuration` | Preserve the user-facing auto-merge behavior and visualization helpers, including `plot_by_sort_group_ids`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time`. |
| `FigURLCurationSelection`, `FigURLCuration` | `FigPackCurationSelection`, `FigPackCuration` | Intentional backend replacement. v1 FigURL remains for v1 data; v2 does not extend FigURL. |
| `RecordingRecompute*` | `RecordingArtifactRecompute*` plus `SortingAnalyzerRecompute*` | v2 uses new names to avoid v1 table-name collisions and extends the pattern to analyzer folders. |
| `ImportedSpikeSorting` | Existing shared `ImportedSpikeSorting` remains canonical | Do not duplicate this as a v2 table. Imported NWB Units continue to enter `SpikeSortingOutput.ImportedSpikeSorting` and can be used beside v2 rows by downstream consumers and validation fixtures. |
| `SpikeSortingOutput` merge dispatch | Existing merge table plus `CurationV2` part | `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, and `get_firing_rate` must work for v2 merge IDs. |
| `get_restricted_merge_ids` and `get_spiking_sorting_v1_merge_ids` convenience lookup | `SpikeSortingOutput.get_restricted_merge_ids(..., sources=['v2'])` | v2 must support interpretable restrictions across recording, artifact, sorting, curation, and analyzer-curation stages. No v2-specific duplicate of the v1 utility is required. |

## Explicit non-parity

- v2 does not promise identical table names or primary keys.
- v2 does not preserve v1's list-returning `insert_selection()` behavior.
- v2 does not write artifact-derived intervals into `IntervalList`.
- v2 does not auto-create default rows for every installed SpikeInterface sorter.
- v2 does not migrate existing v1 curation rows into `CurationV2`.
- v2 does not remove, deprecate, or rewrite v0/v1.
