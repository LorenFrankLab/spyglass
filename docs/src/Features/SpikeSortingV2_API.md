# Spike Sorting v2 API map

A map of what to import from `spyglass.spikesorting.v2` and what each entry
point is for. It lists names and import paths only; signatures and parameter
documentation are in the generated
[API reference](../api/spikesorting/v2/pipeline.md), and worked examples are in
the [Quickstart](./SpikeSortingV2_Quickstart.md), the full
[Spike Sorting v2](./SpikeSortingV2.md) guide, and the v2 notebooks.

## What is supported

- **Supported:** the names listed on this page, imported from the paths shown,
    and the public (no leading underscore) methods of the listed table classes.
- **Internal:** every module whose name starts with an underscore
    (`spyglass.spikesorting.v2._*`). These carry no stability guarantee and may
    change without notice; import their supported names from the public paths
    below (for example, the `pipeline` facade re-exports the orchestration
    functions). A helper a public module imports for its own use is not API
    unless this page lists it. The same goes for the `*Fetched` / `*Computed`
    classes in the table modules: they carry data between a table's
    `make_fetch`, `make_compute`, and `make_insert` steps and are not called
    directly.
- **Where to start:** most work needs only the package root, the `pipeline`
    facade, and the curation/review handles. Reach for the table classes when
    you need to query results or compose a stage by hand.

## Package root

`from spyglass.spikesorting.v2 import ...`

| Name                        | Use for                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `initialize_v2_defaults`    | Install every default parameter row the pipeline needs, in one idempotent call. Run it first. |
| `verify_v2_default_catalog` | Report stored default rows whose content differs from the shipped defaults.                   |
| `CurationLabel`             | The canonical set of curation labels (`accept`, `mua`, `noise`, ...).                         |

The package root also re-exports the main `pipeline` entry points below (for
example `from spyglass.spikesorting.v2 import run_v2_pipeline`). They load on
first use, so `import spyglass.spikesorting.v2` stays lightweight.

## Run the pipeline

`from spyglass.spikesorting.v2.pipeline import ...`

| Name                                                      | Use for                                                                                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_v2_pipeline`                                         | Sort one sort group (or a same-day concatenation) end to end: recording, artifacts, sort, curation, optional auto-curation. Returns a `RunResult`. |
| `run_v2_pipeline_session`                                 | Sort every (or selected) sort group of a session in one call.                                                                                      |
| `preflight_v2_pipeline` / `preflight_v2_pipeline_session` | Read-only check of a planned run (or whole session) before any compute; returns a `PreflightReport` / `PreflightSessionReport`.                    |
| `estimate_motion`                                         | Save the motion estimate of a run's source without sorting, for inspection before applying it.                                                     |
| `describe_sort_groups` / `plot_sort_group_geometry`       | Inspect a session's sort groups and their electrode geometry before choosing one.                                                                  |
| `list_pipeline_presets` / `describe_pipeline_presets`     | List the shipped presets / show them as a catalog table.                                                                                           |
| `describe_pipeline_preset`                                | Unpack one preset into its full, validated parameter values.                                                                                       |
| `describe_recommendation_status`                          | Explain the `recommendation_status` column of the preset catalog.                                                                                  |
| `register_pipeline_preset` / `clone_pipeline_preset`      | Add a custom preset, or copy one with some parameter values changed.                                                                               |
| `describe_parameter_rows`                                 | Catalog the parameter rows currently in the database.                                                                                              |
| `describe_run`                                            | Render a run summary as a receipt: stages, warnings, effective sorter configuration, merge ids.                                                    |
| `describe_units`                                          | Per-unit, sort-time quality snapshot for one sort.                                                                                                 |
| `describe_unit_match_choices`                             | Show which curations can be pinned for each member of a session group.                                                                             |
| `plan_v2_unit_match` / `plan_v2_unit_match_from_sorts`    | Build a reviewable plan that pins one curation per member (or per named sort) for cross-session matching.                                          |
| `run_v2_unit_match`                                       | Match units across sessions and derive tracked-unit identities in one call.                                                                        |
| `select_units_for_analysis`                               | Hand a reviewed curation to downstream analysis under a named label policy (see [Hand off to analysis](#hand-off-to-analysis)).                    |
| `open_curation_analyzer`                                  | Open a disk-backed working copy of a curation's SpikeInterface analyzer, for debugging.                                                            |

The typed result and plan classes these functions return -- `RunResult`,
`PreflightReport`, `PreflightSessionReport`, `UnitMatchPlan`,
`EstimateMotionReceipt`, and the `RunV2*` summaries -- are importable from
`spyglass.spikesorting.v2.pipeline` for type hints and `isinstance` checks.

## Curate and review

### Curation handles

`from spyglass.spikesorting.v2.curation_api import ...` (the most common names
are also importable from `pipeline`)

| Name                                                                  | Use for                                                                                                                                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RunResult`                                                           | What `run_v2_pipeline` returns: a mapping with `root_curation`, `auto_labeled_curation`, and `start_review(...)`.                                                           |
| `CurationRef`                                                         | A curation pinned to one exact row generation. Methods cover lineage, `evaluate`, `start_review`, `preview_merges`, `commit_merges`, `open_analyzer`, and subtree deletion. |
| `save_manual_curation`                                                | Save a manual child curation (labels and/or merges) from a `CurationRef` parent.                                                                                            |
| `preview_merges` / `commit_merges` / `merge_and_evaluate`             | Function forms of the merge operations; `merge_and_evaluate` commits a merge and re-evaluates the merged child.                                                             |
| `create_initial_curation`                                             | Create the root curation of a sorting.                                                                                                                                      |
| `EvaluationSpec`, `EvaluationResult`, `EvaluationPlots`               | The metric/rule recipe of an evaluation, its result snapshot, and its plotting helpers.                                                                                     |
| `MergedCuration`, `MergeEvaluateReceipt`                              | Results of merge operations.                                                                                                                                                |
| `CurationDeletePreview`, `CurationDeleteReceipt`, `CurationOperation` | What a subtree deletion would remove / did remove; provenance of one curation operation.                                                                                    |

### Browser review (FigPack)

`from spyglass.spikesorting.v2.review_api import ...` (`FigPackReview` is also
importable from `pipeline`). Requires the `spikesorting-v2-curation` extra.

| Name                                                             | Use for                                                                                                                                                 |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start_review`                                                   | Evaluate a pinned curation with a review profile and build its browser view. Usually called as `run.start_review(...)` or `curation.start_review(...)`. |
| `FigPackReview`                                                  | Resumable handle for one review: `summary`, `open`, `preview_import`, `commit_panel`, `result`, `find`, `resume`.                                       |
| `CurationChangeSet`, `ReviewImportReceipt`, `MergeLabelConflict` | The pending edits to import, the import result, and a label conflict to resolve before a merge.                                                         |
| `ReviewProfileRef`, `ReviewStageStatus`                          | A resolved review profile and the derived status of a review.                                                                                           |

### Unit annotations

`from spyglass.spikesorting.v2.unit_annotation import ...`

| Name                                          | Use for                                                                      |
| --------------------------------------------- | ---------------------------------------------------------------------------- |
| `UnitAnnotationDefinition`                    | Table of versioned definitions of custom (non-label) unit properties.        |
| `CurationUnitAnnotationSet`                   | Table of immutable annotation sets computed over one exact curation's units. |
| `from_dataframe` / `to_dataframe`             | Write an annotation set from a DataFrame / read one back.                    |
| `read_unit_properties`                        | Read selected built-in and custom properties by unit id.                     |
| `AnnotationDefinitionRef`, `AnnotationSetRef` | References to one definition / one annotation set.                           |

`spyglass.spikesorting.v2.annotation_api` exposes `read_unit_properties` and the
two reference classes without importing the annotation tables.

### Plots and exports

`from spyglass.spikesorting.v2 import visualization as ssviz`

| Name                                                                                                          | Use for                                                              |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `available_visualizations`                                                                                    | Catalog every helper below with the key it takes.                    |
| `recording_key_for_sorting`                                                                                   | Resolve a sorting key to its preprocessed `Recording` key.           |
| `plot_recording_traces`, `plot_recording_probe_map`                                                           | Plot a saved preprocessed recording.                                 |
| `plot_sorting_summary`, `plot_unit_summary`, `plot_waveforms`, `plot_spikes_on_traces`, `plot_unit_locations` | Unit-level plots of an exact curation.                               |
| `plot_metrics`, `plot_si_quality_metrics`, `plot_si_template_metrics`, `plot_suggested_merges`                | Quality-metric and merge-suggestion plots.                           |
| `export_si_report`, `export_to_phy`                                                                           | Write a SpikeInterface report folder or a Phy folder for a curation. |

## Hand off to analysis

| Name                                                                  | Import from                                | Use for                                                                                                                                                 |
| --------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `select_units_for_analysis`                                           | `spyglass.spikesorting.v2.pipeline`        | The supported handoff: apply a label policy to a curation and build the `SortedSpikesGroup` downstream analyses read. Returns a `UnitSelectionReceipt`. |
| `UnitSelectionReceipt`, `SelectedGroup`, `V2_UNIT_SELECTION_POLICIES` | `spyglass.spikesorting.v2.pipeline`        | The handoff receipt (`describe`, `fetch_spike_data`, `group_key`), one created group, and the shipped policies.                                         |
| `SpikeSortingOutput`                                                  | `spyglass.spikesorting.spikesorting_merge` | The merge table v2 curations register on: `get_spike_times`, `get_recording`, `get_sorting`, `get_unit_brain_regions`, ...                              |
| `get_spike_sorting_v2_merge_ids`                                      | `spyglass.spikesorting.v2.utils`           | Resolve the merge ids of v2 curations matching a restriction.                                                                                           |

## Other helpers

| Name                                                            | Import from                                 | Use for                                                                                               |
| --------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `suggest_bad_channels`                                          | `spyglass.spikesorting.v2.bad_channels`     | Suggest a session's bad channels for review, and optionally persist the reviewed report.              |
| `detect_bad_channels`                                           | `spyglass.spikesorting.v2.bad_channels`     | Lower-level detection on a filtered SpikeInterface recording: bad channel ids and per-channel labels. |
| Exception classes (e.g. `PipelineStageError`, `PreflightError`) | `spyglass.spikesorting.v2.exceptions`       | Catch specific v2 failures.                                                                           |
| `MatcherProtocol`, `register_matcher`                           | `spyglass.spikesorting.v2.matcher_protocol` | Implement and register a new cross-session matcher backend.                                           |

## Tables

The DataJoint tables, by module. Fill a pipeline `*Selection` table with its
`insert_selection` method, then `populate()` the computed table;
`run_v2_pipeline` does this for you. The recompute selection tables are planned
with `attempt_all` instead. Parameter tables provide `insert_default()`. See
[Tables](./SpikeSortingV2.md#tables) in the full guide for what each one stores.

| Module (`spyglass.spikesorting.v2.`) | Tables                                                                                                                                                                                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `recording`                          | `SortGroupV2`, `PreprocessingParameters`, `RecordingSelection`, `Recording`, `DriftEstimate` (QC only)                                                                                                                                                        |
| `artifact`                           | `ArtifactDetectionParameters`, `RecordingArtifactSelection`, `RecordingArtifactDetection`, `SharedArtifactGroup`, `SharedGroupArtifactSelection`, `SharedGroupArtifactDetection`                                                                              |
| `sorting`                            | `SorterParameters`, `AnalyzerWaveformParameters`, `SortingSelection`, `Sorting`                                                                                                                                                                               |
| `curation`                           | `CurationV2`                                                                                                                                                                                                                                                  |
| `metric_curation`                    | `QualityMetricParameters`, `AutoCurationRules`, `CurationEvaluationSelection`, `CurationEvaluation`                                                                                                                                                           |
| `review_profile`                     | `CurationReviewProfile`                                                                                                                                                                                                                                       |
| `figpack_curation`                   | `FigPackCurationSelection`, `FigPackCuration` (the lower layer under `FigPackReview`)                                                                                                                                                                         |
| `motion`                             | `MotionEstimationParameters`, `MotionInterpolationParameters`, `MotionCorrectionParameters`, `MotionEstimateSelection`, `MotionEstimate`, `MotionCorrectedRecordingSelection`, `MotionCorrectedRecording`                                                     |
| `session_group`                      | `SessionGroup`, `ConcatenatedRecordingSelection`, `ConcatenatedRecording`                                                                                                                                                                                     |
| `concat_member_curation`             | `ConcatMemberCuration`                                                                                                                                                                                                                                        |
| `unit_matching`                      | `MatcherParameters`, `UnitMatchSelection`, `UnitMatch`, `TrackedUnit`                                                                                                                                                                                         |
| `recompute`                          | `RecordingArtifactVersions`, `RecordingArtifactRecomputeSelection`, `RecordingArtifactRecompute`, `SortingAnalyzerVersions`, `SortingAnalyzerRecomputeSelection`, `SortingAnalyzerRecompute` (see [Storage Management](./SpikeSortingV2StorageManagement.md)) |

`ArtifactDetectionOutput` (`artifact_output`) is the internal merge over the two
artifact-detection tables; `SortingSelection` resolves it for you, so user code
does not need it.

The remaining public names in the table modules and in `utils` -- insert guards,
integrity audits, and helpers the tables call while computing -- support the
tables themselves and are not entry points.
