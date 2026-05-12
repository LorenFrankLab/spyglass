# Feature parity with v1

[← back to PLAN.md](PLAN.md)

This page is the parity contract. It distinguishes behavior v3 must preserve from intentional table/API departures. When in doubt, implementers should preserve downstream behavior and user workflows, not v1's exact table names.

| v1 surface | v3 parity target | Notes |
| --- | --- | --- |
| `SortGroup.set_group_by_shank` | `SortGroupV3.set_group_by_shank` plus column-based grouping | v3 adds guarded deletion and non-shank grouping. Silent overwrite is intentionally not preserved. |
| `SpikeSortingPreprocessingParameters`, `SpikeSortingRecordingSelection`, `SpikeSortingRecording` | `PreprocessingParameters`, `RecordingSelection`, `Recording` | Same workflow shape; v3 adds single-dict `insert_selection()` returns and missing-cache rebuild. |
| `ArtifactDetectionParameters`, `ArtifactDetectionSelection`, `ArtifactDetection` | Same named v3 concepts | v3 does not write artifact-removed intervals to `IntervalList`; intervals live on `ArtifactDetection.Interval` and are exposed through `get_artifact_removed_intervals()`. |
| `SpikeSorterParameters` | `SorterParameters` | v3 ships dedicated defaults for MS4, MS5, KS4, `clusterless_thresholder`, `spykingcircus2`, and `tridesclous2`. Other SpikeInterface sorters are supported through explicit custom parameter rows, not auto-inserted defaults for every installed sorter. |
| `SpikeSortingSelection`, `SpikeSorting` | `SortingSelection`, `Sorting` | Same single-session output behavior plus SortingAnalyzer folder and `Sorting.Unit` metadata. |
| `CurationV1` | `CurationV3` | Same lineage role. v3 auto-registers into `SpikeSortingOutput`, validates labels, and normalizes labels into `UnitLabel`. |
| `MetricParameters`, `WaveformParameters`, `MetricCurationParameters`, `MetricCuration` | `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCuration` | v3 consolidates metric computation and auto-curation around SortingAnalyzer extensions. |
| `BurstPairParams`, `BurstPairSelection`, `BurstPair` | No table clone; folded into `AnalyzerCuration` | Preserve the user-facing auto-merge behavior and visualization helpers, including `plot_by_sort_group_ids`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time`. |
| `FigURLCurationSelection`, `FigURLCuration` | `FigPackCurationSelection`, `FigPackCuration` | Intentional backend replacement. v1 FigURL remains for v1 data; v3 does not extend FigURL. |
| `RecordingRecompute*` | `RecordingArtifactRecompute*` plus `SortingAnalyzerRecompute*` | v3 uses new names to avoid v1 table-name collisions and extends the pattern to analyzer folders. |
| `ImportedSpikeSorting` | Existing shared `ImportedSpikeSorting` remains canonical | Do not duplicate this as a v3 table. Imported NWB Units continue to enter `SpikeSortingOutput.ImportedSpikeSorting` and can be used beside v3 rows by downstream consumers and validation fixtures. |
| `SpikeSortingOutput` merge dispatch | Existing merge table plus `CurationV3` part | `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, and `get_firing_rate` must work for v3 merge IDs. |
| `get_restricted_merge_ids` and `get_spiking_sorting_v1_merge_ids` convenience lookup | `SpikeSortingOutput.get_restricted_merge_ids(..., sources=['v3'])` | v3 must support interpretable restrictions across recording, artifact, sorting, curation, and analyzer-curation stages. No v3-specific duplicate of the v1 utility is required. |

## Explicit non-parity

- v3 does not promise identical table names or primary keys.
- v3 does not preserve v1's list-returning `insert_selection()` behavior.
- v3 does not write artifact-derived intervals into `IntervalList`.
- v3 does not auto-create default rows for every installed SpikeInterface sorter.
- v3 does not migrate existing v1 curation rows into `CurationV3`.
- v3 does not remove, deprecate, or rewrite v0/v1.
