"""Convenience orchestration across the spike sorting tables.

``run_v2_pipeline`` chains the recording -> artifact -> sort ->
curation stages into one call so notebook users can populate an
end-to-end single-session sort without writing the per-stage
insert_selection / populate boilerplate. The same orchestrator also
runs metric-driven auto-curation (``auto_curate=True``), same-day
concatenated sorts (concat mode over a ``SessionGroup``), and a
publishable browser curation view (``figpack=True``); cross-session
unit matching is its own helper, ``run_v2_unit_match``.

Pipeline presets are Pydantic-validated bundles of Lookup-row names; the
orchestrator looks them up at first call. The shipped presets are the dated
franklab production recipes -- the MountainSort4 family by target region
(hippocampus 600 Hz / cortex 300 Hz high-pass) and sampling rate, a
MountainSort5 preset, and a clusterless preset. The ``run_v2_pipeline``
default is the MountainSort5 hippocampus recipe
``franklab_probe_hippocampus_30khz_ms5_2026_06``: it runs under the v2
``numpy>=2`` baseline, whereas MountainSort4's ``ml_ms4alg`` backend needs
``numpy<2`` (see the MS4 preset notes). Call ``describe_pipeline_presets()``
for the catalog and ``list_pipeline_presets()`` for the names.

The orchestrator is idempotent: re-running with the same inputs finds
existing rows via the insert_selection helpers and returns the same
run summary (with the same root_merge_id) without inserting duplicates.

This module is a thin facade: the implementation lives in the
``_pipeline_*`` submodules (presets, geometry, preflight, reporting, run),
split for readability. Every public name is re-exported here so notebook and
user import paths (``from spyglass.spikesorting.v2.pipeline import ...``) are
unchanged.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._pipeline_geometry import (
    describe_sort_groups,
    plot_sort_group_geometry,
)
from spyglass.spikesorting.v2._pipeline_preflight import (
    PreflightCheck,
    PreflightReport,
    PreflightSessionReport,
    preflight_v2_pipeline,
    preflight_v2_pipeline_session,
)
from spyglass.spikesorting.v2._pipeline_presets import (
    _PIPELINE_PRESETS,
    _PipelinePreset,
    clone_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_preset,
    list_pipeline_presets,
    register_preset,
)
from spyglass.spikesorting.v2._pipeline_reporting import (
    describe_parameter_rows,
    describe_run,
    describe_units,
)
from spyglass.spikesorting.v2._pipeline_run import (
    describe_unit_match_choices,
    run_v2_pipeline,
    run_v2_pipeline_session,
    run_v2_unit_match,
)
from spyglass.spikesorting.v2._pipeline_types import (
    PipelineOutcome,
    PipelineStageSeconds,
    RunV2ConcatSummary,
    RunV2PipelineInputs,
    RunV2PipelineSessionFailed,
    RunV2PipelineSessionInputs,
    RunV2PipelineSessionOk,
    RunV2PipelineSessionRequiredInputs,
    RunV2PipelineSessionResult,
    RunV2PipelineSummary,
    RunV2SingleSessionSummary,
    RunV2UnitMatchSummary,
    SourceMode,
    StageStatus,
    UnitMatchCurationChoice,
    UnitMatchMemberChoices,
    UnitMatchStageSeconds,
)
