"""DRAFT v2 schema declarations — for plan validation only.

This module exists for `code_graph.py` to statically check the v2 schema
designs against existing Spyglass FK targets BEFORE we commit to
implementation. Every class has a `definition` string but the `make()`
bodies all raise `NotImplementedError`.

Do NOT decorate with `@schema`. Do NOT import from production code.
Will be git-rm'd or formalized into individual modules in Phase 0.

See `.claude/docs/plans/spikesorting-v2/` for the design plan.
"""

import datajoint as dj

from spyglass.common import (
    Session,
    Nwbfile,
    IntervalList,
    Raw,
    Electrode,
    LabTeam,
    BrainRegion,
    UserEnvironment,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import SpyglassMixin, SpyglassMixinPart


# ============================================================================
# Phase 1: Recording side
# ============================================================================


class SortGroupV2(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    sort_group_id: int
    ---
    sort_reference_electrode_id = -1: int
    """

    class SortGroupElectrode(SpyglassMixinPart):
        definition = """
        -> master
        -> Electrode
        """


class PreprocessingParameters(SpyglassMixin, dj.Lookup):
    definition = """
    preproc_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=1: int
    """


class RecordingSelection(SpyglassMixin, dj.Manual):
    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
    -> IntervalList
    -> PreprocessingParameters
    -> LabTeam
    """


class Recording(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(40)
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(64)
    """


# ============================================================================
# Phase 1: Artifact detection
# ============================================================================


class SharedArtifactGroup(SpyglassMixin, dj.Manual):
    definition = """
    shared_artifact_group_name: varchar(64)
    ---
    -> Session
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> Recording
        """


class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=1: int
    """


class ArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    definition = """
    artifact_id: uuid
    ---
    -> [nullable] Recording
    -> [nullable] SharedArtifactGroup
    -> ArtifactDetectionParameters
    """


class ArtifactDetection(SpyglassMixin, dj.Computed):
    definition = """
    -> ArtifactDetectionSelection
    """

    class Interval(SpyglassMixinPart):
        definition = """
        -> master
        interval_index: int
        ---
        start_time: double
        end_time: double
        """


# ============================================================================
# Phase 1: Concat scaffolding (declared in Phase 1; make() raises until Phase 3)
# ============================================================================


class SessionGroup(SpyglassMixin, dj.Manual):
    definition = """
    session_group_name: varchar(64)
    ---
    description: varchar(255)
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        member_index: int
        ---
        -> Session
        -> SortGroupV2
        -> IntervalList
        -> LabTeam
        recording_date: date
        """


class MotionCorrectionParameters(SpyglassMixin, dj.Lookup):
    definition = """
    motion_correction_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=1: int
    """


class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual):
    definition = """
    concat_recording_id: uuid
    ---
    -> SessionGroup
    -> PreprocessingParameters
    -> MotionCorrectionParameters
    """


class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    definition = """
    -> ConcatenatedRecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(40)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    member_segment_boundaries: blob
    cache_hash: char(64)
    """


# ============================================================================
# Phase 1: Sorting
# ============================================================================


class SorterParameters(SpyglassMixin, dj.Lookup):
    definition = """
    sorter: varchar(64)
    sorter_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=1: int
    """


class SortingSelection(SpyglassMixin, dj.Manual):
    definition = """
    sorting_id: uuid
    ---
    -> [nullable] Recording
    -> [nullable] ConcatenatedRecording
    -> [nullable] ArtifactDetection
    -> SorterParameters
    """


class Sorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)
    analyzer_folder: varchar(255)
    n_units: int
    time_of_sort: datetime
    """

    class Unit(SpyglassMixinPart):
        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uV: float
        n_spikes: int
        """


# ============================================================================
# Phase 1: Curation
# ============================================================================


class CurationV2(SpyglassMixin, dj.Manual):
    definition = """
    -> Sorting
    curation_id=0: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied=0: tinyint
    metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack', 'imported')
    description: varchar(255)
    """

    class Unit(SpyglassMixinPart):
        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uV: float
        n_spikes: int
        """

    class UnitLabel(SpyglassMixinPart):
        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)
        """


# ============================================================================
# Phase 2: Analyzer curation
# ============================================================================


class QualityMetricParameters(SpyglassMixin, dj.Lookup):
    definition = """
    metric_params_name: varchar(64)
    ---
    metric_names: blob
    metric_kwargs: blob
    skip_pc_metrics=1: tinyint
    params_schema_version=1: int
    """


class AutoCurationRules(SpyglassMixin, dj.Lookup):
    definition = """
    auto_curation_rules_name: varchar(64)
    ---
    label_rules: blob
    auto_merge_preset: varchar(32)
    auto_merge_kwargs: blob
    params_schema_version=1: int
    """


class AnalyzerCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    analyzer_curation_id: uuid
    ---
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    """


class AnalyzerCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> AnalyzerCurationSelection
    ---
    -> AnalysisNwbfile
    metrics_object_id: varchar(40)
    merge_suggestions_object_id: varchar(40)
    proposed_labels_object_id: varchar(40)
    """


class RecordingArtifactVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> Recording
    ---
    nwb_deps=null: blob
    cache_hash: char(64)
    """


class RecordingArtifactRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingArtifactVersions
    -> UserEnvironment
    rounding=4: int
    ---
    logged_at_creation=0: tinyint
    xfail_reason=NULL: varchar(127)
    """


class RecordingArtifactRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingArtifactRecomputeSelection
    ---
    matched: tinyint
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: tinyint
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """


class SortingAnalyzerVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> Sorting
    ---
    si_deps=null: blob
    analyzer_manifest=null: blob
    analyzer_hash: char(32)
    """


class SortingAnalyzerRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortingAnalyzerVersions
    -> UserEnvironment
    rounding=4: int
    ---
    logged_at_creation=0: tinyint
    xfail_reason=NULL: varchar(127)
    """


class SortingAnalyzerRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingAnalyzerRecomputeSelection
    ---
    matched: tinyint
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: tinyint
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """


# ============================================================================
# Phase 4: Cross-session matching
# ============================================================================


class MatcherParameters(SpyglassMixin, dj.Lookup):
    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)
    params: blob
    params_schema_version=1: int
    """


class UnitMatchSelection(SpyglassMixin, dj.Manual):
    definition = """
    unitmatch_id: uuid
    ---
    -> SessionGroup
    -> MatcherParameters
    curation_set_hash: char(64)
    """

    class MemberCuration(SpyglassMixinPart):
        definition = """
        -> master
        -> SessionGroup.Member
        ---
        -> CurationV2
        """


class UnitMatch(SpyglassMixin, dj.Computed):
    definition = """
    -> UnitMatchSelection
    ---
    -> AnalysisNwbfile
    pairs_object_id: varchar(40)
    n_pairs: int
    matcher_runtime_s: float
    """

    class Pair(SpyglassMixinPart):
        definition = """
        -> master
        pair_index: int
        ---
        -> CurationV2.Unit.proj(
            session_a_sorting_id='sorting_id',
            session_a_curation_id='curation_id',
            unit_a_id='unit_id')
        -> CurationV2.Unit.proj(
            session_b_sorting_id='sorting_id',
            session_b_curation_id='curation_id',
            unit_b_id='unit_id')
        match_probability: float
        drift_estimate_um=0.0: float
        fdr_estimate=NULL: float
        """


class TrackedUnit(SpyglassMixin, dj.Computed):
    definition = """
    -> UnitMatch
    tracked_unit_id: int
    ---
    n_sessions_observed: int
    median_match_probability=NULL: float
    n_transitive_only_edges=0: int
    policy_used: enum('strict', 'transitive', 'transitive_fallback')
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> CurationV2.Unit
        """


# ============================================================================
# Phase 5: FigPack curation
# ============================================================================


class FigPackCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    figpack_curation_id: uuid
    ---
    -> CurationV2
    figpack_config_hash: char(64)
    label_options: blob
    metrics: blob
    upload: tinyint
    ephemeral: tinyint
    """


class FigPackCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(512)
    """
