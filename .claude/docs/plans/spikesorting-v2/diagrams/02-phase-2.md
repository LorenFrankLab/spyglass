# Phase 2 — Analyzer curation + recompute verification

[← Phase 1](01-phase-1.md) · [README](README.md) · [next: Phase 3 →](03-phase-3.md)

Adds quality metrics, auto-curation, burst-pair detection (consolidated into one table), and the recompute machinery that powers verified storage reclamation. **No Phase 1 schemas change.**

## What ships in Phase 2

| New table | Tier | Purpose |
| --- | --- | --- |
| `QualityMetricParameters` | Lookup | Metric names + per-metric kwargs; SI 0.104 `compute_quality_metrics`. |
| `AutoCurationRules` | Lookup | Label thresholds + auto-merge preset name. |
| `AnalyzerCurationSelection` | Manual | One row per (curation, metrics params, rules) tuple. |
| `AnalyzerCuration` | Computed | Walks SortingAnalyzer extensions; writes metrics + merge suggestions + proposed labels to NWB. Replaces v1 `MetricCuration` + `BurstPair`. |
| `RecordingArtifactVersions` | Computed | Inventories the NWB / SI versions used to materialize each `Recording`. |
| `RecordingArtifactRecomputeSelection` | Manual | Pair (`Recording`, `UserEnvironment`) for recompute attempt. |
| `RecordingArtifactRecompute` (+ `Name`, `Hash` parts) | Computed | Compares regenerated artifact byte-hash to stored `cache_hash`. Gates `delete_files()`. |
| `SortingAnalyzerVersions` | Computed | Same shape for `Sorting`'s analyzer folder. |
| `SortingAnalyzerRecomputeSelection` | Manual | Selection row for analyzer recompute. |
| `SortingAnalyzerRecompute` (+ `Name`, `Hash` parts) | Computed | Verified analyzer folder regeneration. |

## ER diagram — analyzer curation

```mermaid
erDiagram
    %% ===== From Phase 1 =====

    %% ===== NEW IN PHASE 2 =====
    QualityMetricParameters {
        varchar metric_params_name PK
        blob metric_names
        blob metric_kwargs
        bool skip_pc_metrics
    }
    AutoCurationRules {
        varchar auto_curation_rules_name PK
        varchar auto_merge_preset
        blob auto_merge_kwargs
    }
    AutoCurationRules_Rule {
        varchar auto_curation_rules_name PK
        int rule_index PK
        varchar rule_name
        varchar metric_name
        enum operator
        float threshold
        varchar label
    }
    AnalyzerCurationSelection {
        uuid analyzer_curation_id PK
        uuid sorting_id FK
        int curation_id FK
        varchar metric_params_name FK
        varchar auto_curation_rules_name FK
    }
    AnalyzerCuration {
        uuid analyzer_curation_id PK
        varchar analysis_file_name FK
        varchar metrics_object_id
        varchar merge_suggestions_object_id
        varchar proposed_labels_object_id
    }

    CurationV2 ||--o{ AnalyzerCurationSelection : "FK"
    QualityMetricParameters ||--o{ AnalyzerCurationSelection : "FK"
    AutoCurationRules ||--o{ AnalyzerCurationSelection : "FK"
    AutoCurationRules ||--o{ AutoCurationRules_Rule : "part"
    AnalyzerCurationSelection ||--|| AnalyzerCuration : "Computed"
    AnalysisNwbfile ||--o{ AnalyzerCuration : "metrics + labels + merges NWB"

    %% ===== Materialize back into CurationV2 lineage =====
    %% NOTE: This is a programmatic linkage (materialize_curation creates a child
    %% CurationV2 row with parent_curation_id set), NOT a DataJoint FK.
    %% CurationV2.parent_curation_id is a plain int, not a self-referential FK.
    AnalyzerCuration ||--o{ CurationV2 : "materialize_curation (programmatic, not FK)"
```

## ER diagram — recompute machinery

```mermaid
erDiagram
    %% ===== From Phase 1 / baseline =====

    %% ===== NEW IN PHASE 2 =====
    RecordingArtifactVersions {
        uuid recording_id PK
        blob nwb_deps
        char cache_hash
    }
    RecordingArtifactRecomputeSelection {
        uuid recording_id PK
        varchar env_id PK
        int rounding PK
        bool logged_at_creation
        varchar xfail_reason
    }
    RecordingArtifactRecompute {
        uuid recording_id PK
        varchar env_id PK
        int rounding PK
        bool matched
        varchar err_msg
        datetime created_at
        bool deleted
    }
    RecordingArtifactRecompute_Name {
        uuid recording_id PK
        varchar env_id PK
        int rounding PK
        varchar name PK
        enum missing_from
    }
    RecordingArtifactRecompute_Hash {
        uuid recording_id PK
        varchar env_id PK
        int rounding PK
        varchar name PK
    }

    Recording ||--|| RecordingArtifactVersions : "Computed"
    RecordingArtifactVersions ||--o{ RecordingArtifactRecomputeSelection : ""
    UserEnvironment ||--o{ RecordingArtifactRecomputeSelection : ""
    RecordingArtifactRecomputeSelection ||--|| RecordingArtifactRecompute : "Computed"
    RecordingArtifactRecompute ||--o{ RecordingArtifactRecompute_Name : "part"
    RecordingArtifactRecompute ||--o{ RecordingArtifactRecompute_Hash : "part"

    %% ===== Mirror tables for SortingAnalyzer folders =====
    SortingAnalyzerVersions {
        uuid sorting_id PK
        blob si_deps
        blob analyzer_manifest
        char_64 analyzer_hash
    }
    SortingAnalyzerRecomputeSelection {
        uuid sorting_id PK
        varchar env_id PK
        int rounding PK
        bool logged_at_creation
        varchar xfail_reason
    }
    SortingAnalyzerRecompute {
        uuid sorting_id PK
        varchar env_id PK
        int rounding PK
        bool matched
        varchar err_msg
        bool deleted
    }
    SortingAnalyzerRecompute_Name {
        uuid sorting_id PK
        varchar env_id PK
        int rounding PK
        varchar name PK
        enum missing_from
    }
    SortingAnalyzerRecompute_Hash {
        uuid sorting_id PK
        varchar env_id PK
        int rounding PK
        varchar name PK
    }

    Sorting ||--|| SortingAnalyzerVersions : "Computed"
    SortingAnalyzerVersions ||--o{ SortingAnalyzerRecomputeSelection : ""
    UserEnvironment ||--o{ SortingAnalyzerRecomputeSelection : ""
    SortingAnalyzerRecomputeSelection ||--|| SortingAnalyzerRecompute : "Computed"
    SortingAnalyzerRecompute ||--o{ SortingAnalyzerRecompute_Name : "part"
    SortingAnalyzerRecompute ||--o{ SortingAnalyzerRecompute_Hash : "part"
```

## Populate flow

```mermaid
flowchart LR
    subgraph "Curation flow"
        C[CurationV2 row exists] --> S[AnalyzerCurationSelection.insert]
        S --> A[AnalyzerCuration.populate]
        A --> M[materialize_curation creates child CurationV2]
        M --> R[SpikeSortingOutput.CurationV2 auto-register]
    end

    subgraph "Recompute flow per artifact"
        REC[Recording row] --> V[RecordingArtifactVersions.populate]
        V --> RS[RecordingArtifactRecomputeSelection.insert]
        RS --> RR[RecordingArtifactRecompute.populate]
        RR --> DEL[delete_files only if matched=1 in current env]
    end
```

## Critical design points

- **One table replaces two**: `AnalyzerCuration` consolidates v1's `MetricCuration` + `BurstPair`. The burst-pair cross-correlogram-asymmetry logic becomes one auto-merge preset; the visualization helpers (`plot_correlograms_by_sort_group`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, `plot_peak_over_time`) become methods on `AnalyzerCuration`.
- **`materialize_curation()` is explicit** — auto-curation never silently writes a new `CurationV2` row. The user calls `AnalyzerCuration.materialize_curation(key)` to commit, which inserts a child `CurationV2` row with `parent_curation_id` set and `metrics_source='analyzer_curation'`.
- **Fetch-helper parity**: `AnalyzerCuration` exposes `get_waveforms`, `get_metrics`, `get_labels`, `get_merge_groups` to match v1's `MetricCuration` notebook surface.
- **NaN sanitization**: serialized metric tables coerce non-finite values to `None` in the JSON-bound path; in-memory DataFrames preserve NaN semantics (issue #1556).
- **Two distinct recompute families**: `RecordingArtifactRecompute*` verifies the NWB `ElectricalSeries` byte-hash; `SortingAnalyzerRecompute*` verifies the analyzer folder contents. Same lifecycle, different artifacts.
- **`delete_files()` gates on `matched=1` in the current `UserEnvironment`** — storage reclamation requires a verified recompute round-trip in the environment doing the deletion. Historic matches from stale environments are audit evidence only unless explicitly force-overridden with justification. `matched=0` cannot delete.
- **Admin surface**: `attempt_all`, `remove_matched`, `with_names`, `get_parent_key`, `recheck`, `get_disk_space`, `update_secondary` are preserved from v1's `RecordingRecompute` family, or explicitly listed as non-parity in [feature-parity.md](../feature-parity.md).
