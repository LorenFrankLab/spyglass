"""Run one sort group or concatenated session group through the v2 pipeline."""

from __future__ import annotations

import uuid
from typing import Any
import importlib

from spyglass.spikesorting.v2._orchestration.preflight import (
    resolve_motion_recipe,
    resolve_preset_sort_config,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    DEFAULT_PIPELINE_PRESET,
)
from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.motion import (
    _run_motion_correction,
    _run_motion_estimate,
)
from spyglass.spikesorting.v2._orchestration.source import (
    _RunRequest,
    _build_run_source,
    _run_preflight,
    _validate_run_request,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _run_auto_curation_stage,
    _run_member_curation_stage,
    _run_root_curation_stage,
    _run_sort_group_keys,
    _run_sorting_stage,
)
from spyglass.spikesorting.v2.curation_api import RunResult
from spyglass.spikesorting.v2.exceptions import PipelineInputError
from spyglass.spikesorting.v2._orchestration.preflight import (
    describe_scientific_setup,
)
from spyglass.utils import logger


def run_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    *,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    curation_description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id: "uuid.UUID | str | None" = None,
) -> "RunResult":
    """End-to-end sort of one sort group: select + populate every stage, then curate.

    Stages: recording -> optional artifact detection -> [motion] -> sort ->
    curation. To sort a concatenated session group instead, use
    :func:`run_v2_concat_pipeline`, which takes the same options.

    Idempotent: re-running reuses the same outputs and PKs without duplicating
    rows. Reported statuses and timings describe the current call.

    Prerequisites (set these up first, in order)
    --------------------------------------------
    1. ``initialize_v2_defaults()``.
    2. A ``LabTeam`` row for ``team_name``.
    3. ``SortGroupV2.set_group_by_shank(nwb_file_name=...)`` (or
       ``set_group_by_electrode_table_column``).

    ``preflight_v2_pipeline(...)`` checks these without running.

    Parameters
    ----------
    nwb_file_name
        Session whose data will be sorted. The session must already be
        ingested via ``insert_sessions``.
    sort_group_id
        ID of an existing ``SortGroupV2`` row for this session.
    interval_list_name
        Name of the IntervalList row to sort. Typically ``"raw data
        valid times"`` for a full-session sort.
    team_name
        Existing ``LabTeam`` owning the sort.
    pipeline_preset
        Pipeline-preset name. Default
        ``franklab_probe_hippocampus_30khz_ms4_2026_06`` (MountainSort4); the
        ``..._ms4_singularity_2026_06`` preset runs it in Singularity /
        Apptainer. See ``describe_pipeline_presets()``.
    curation_description
        Free-text description passed to ``CurationV2.insert_curation``.
    require_units
        If False (default), a zero-unit sort produces an empty curation with
        a warning (a quiet shank is legitimate). If True, it raises
        ``ZeroUnitSortError``.
    auto_curate
        If False (default), stop at the root curation. If True, score it with
        the preset's metric and rule rows (``CurationEvaluation``, the
        heaviest populate) and commit a child curation with those labels; the
        summary adds ``curation_evaluation_id`` and ``auto_curation_status``
        and fills the ``auto_labeled_*`` keys. The child still holds every
        unit; select the analysis population with
        ``select_units_for_analysis``.
    preflight
        If True (default), run ``preflight_v2_pipeline`` first; a failure
        raises ``PreflightError``.
    manual_excluded_times
        Immutable half-open [start, stop) exclusions in absolute session
        seconds (the ``IntervalList.valid_times`` clock, not seconds from the
        recording start), composed with automatic artifact detection. Each
        must overlap the sorted time, else preflight (or the artifact
        selection) raises. Manual exclusions also apply when the preset
        disables automatic detection.
    motion_mode
        The motion stage, run on the sort's masked source:

        - ``"off"`` (default): no motion stage.
        - ``"estimate"``: also save a ``MotionEstimate`` for QC; the sort is
          the ``"off"`` sort.
        - ``"apply"``: also save a ``MotionCorrectedRecording`` and sort it.

        Experimental. A motion stage failure raises ``PipelineStageError``
        with no fallback to the uncorrected source. ``"apply"`` rejects a
        sorter row that runs its own motion correction.
    motion_correction_params_name
        The ``MotionCorrectionParameters`` recipe (e.g. ``dredge_v1``,
        ``dredge_fast_v1``). Required iff ``motion_mode`` is not ``"off"``.
    motion_estimate_id
        With ``motion_mode="apply"`` only: apply this saved ``MotionEstimate``
        (never recomputed). It must be an estimate of this run's source and
        mask with the recipe's estimation row; a mismatch fails preflight or
        the ``motion_estimate`` stage before any sort.

    Returns
    -------
    RunResult
        Mapping-compatible summary wrapping a ``RunV2SingleSessionSummary`` or
        ``RunV2ConcatSummary``, with generation-pinned ``root_curation`` and
        ``auto_labeled_curation`` accessors.

        Always present:
            ``pipeline_preset``          : the pipeline-preset name
            ``source_mode``              : ``"single_session"`` or ``"concat"``
            ``sorting_id``               : SortingSelection PK
            ``root_curation_id``         : the ROOT (uncurated) CurationV2 PK
            ``root_merge_id``            : the root's SpikeSortingOutput PK;
                ``None`` for a concat run
            ``auto_labeled_curation_id``     : the auto-labeled child CurationV2
                PK, or ``None`` on a root-only run
            ``auto_labeled_merge_id``        : that child's SpikeSortingOutput
                PK, or ``None`` on a root-only or concat run
            ``root_curation_uuid`` / ``auto_labeled_curation_uuid`` : the
                accessors' generation UUIDs
            ``sorter_config``            : ``EffectiveSortConfig.as_dict()``
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Single-session mode adds:
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : RecordingArtifactSelection PK, or
                ``None`` when the preset runs no artifact detection
        :func:`run_v2_concat_pipeline` adds:
            ``member_recording_ids``     : the per-member RecordingSelection PKs
            ``concat_recording_id``      : ConcatenatedRecording PK
            ``member_merge_ids``         : frozen ``member_index`` to
                wall-clock-aligned SpikeSortingOutput PK; points to the
                auto-curated child when ``auto_curate=True``, otherwise the root
        Motion keys (always present; ``None`` where they do not apply):
            ``motion_mode`` / ``motion_correction_params_name`` : the request
            ``motion_estimate_id``       : MotionEstimateSelection PK
                (``"estimate"`` / ``"apply"``)
            ``motion_estimate_supplied`` : whether the caller passed
                ``motion_estimate_id``
            ``motion_estimation_preset`` : the resolved SI estimation preset
            ``motion_corrected_recording_id`` :
                MotionCorrectedRecordingSelection PK (``"apply"``)
            ``motion_removed_channel_ids`` : channels the ``remove_channels``
                border mode dropped (``"apply"``)
            ``motion_spans_without_evidence`` : continuity spans with no
                peak; a non-empty list also adds a warning
        Per-stage keys:
            ``*_status``          : ``"computed"``, ``"reused"``, or
                ``"skipped"`` (see ``StageStatus``)
            ``stage_seconds``     : wall-clock seconds per stage this call
            ``warnings``          : advisories raised during the run

        No merge id is a filtered unit set; pass a curation to
        ``select_units_for_analysis`` for analysis.

    Raises
    ------
    PipelineInputError
        Before any insert or compute: a source input is ``None``, the preset
        is unknown, or the motion request is contradictory.
    PreflightError
        If ``preflight=True`` and a prerequisite is missing.
    PipelineStageError
        If a compute stage fails. Names the stage and carries the partial run
        summary; the original error is chained.
    ZeroUnitSortError
        If the sort finds zero units and ``require_units=True``.
    ValueError
        If a required parameter row is missing; run
        ``initialize_v2_defaults()``.
    datajoint.errors.IntegrityError
        If an upstream row is missing and ``preflight=False``.
    """
    # A bad request fails before any schema activates.
    request = _validate_run_request(
        "run_v2_pipeline",
        is_concat=False,
        source_inputs={
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_list_name,
            "team_name": team_name,
        },
        pipeline_preset=pipeline_preset,
        curation_description=curation_description,
        require_units=require_units,
        auto_curate=auto_curate,
        preflight=preflight,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
    )
    return _run_pipeline(request)


def run_v2_concat_pipeline(
    session_group_owner: str,
    session_group_name: str,
    *,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    curation_description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id: "uuid.UUID | str | None" = None,
) -> "RunResult":
    """End-to-end sort of a concatenated session group, then curate.

    Stages: member recordings -> member artifact masks ->
    ``ConcatenatedRecording`` -> [motion] -> sort -> curation. Artifact masks
    apply per member. :func:`run_v2_pipeline` documents the shared options.

    Parameters
    ----------
    session_group_owner, session_group_name
        The primary key of an existing ``SessionGroup``.
    manual_excluded_times
        A mapping of each ``member_index`` to its half-open [start, stop)
        exclusions in that member's absolute session seconds, composed with
        automatic artifact detection. Each must overlap that member's sorted
        time.
    preflight
        If True (default), first run ``assert_concat_preflight``; a failure
        raises ``PreflightError``.
    motion_estimate_id
        As in :func:`run_v2_pipeline`; the estimate must match this group's
        preprocessing recipe, members and member masks.
    pipeline_preset, curation_description, require_units, auto_curate, \
motion_mode, motion_correction_params_name
        As in :func:`run_v2_pipeline`.

    Returns
    -------
    RunResult
        Wraps a ``RunV2ConcatSummary``. Root and auto-labeled merge ids stay
        ``None``; use ``member_merge_ids``.

    Raises
    ------
    PipelineInputError, PreflightError, PipelineStageError, ZeroUnitSortError, \
ValueError
        As in :func:`run_v2_pipeline`.
    """
    # A bad request fails before any schema activates.
    request = _validate_run_request(
        "run_v2_concat_pipeline",
        is_concat=True,
        source_inputs={
            "concat_session_group_owner": session_group_owner,
            "concat_session_group_name": session_group_name,
        },
        pipeline_preset=pipeline_preset,
        curation_description=curation_description,
        require_units=require_units,
        auto_curate=auto_curate,
        preflight=preflight,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
    )
    return _run_pipeline(request)


def _run_pipeline(request: _RunRequest) -> "RunResult":
    """Shared body of ``run_v2_pipeline`` / ``run_v2_concat_pipeline``."""
    bundle = request.bundle
    is_concat = request.is_concat
    source_inputs = request.source_inputs
    pipeline_preset = request.pipeline_preset
    motion_mode = request.motion_mode

    # A schema that cannot activate should fail before any compute.
    for schema_module in (
        "spyglass.spikesorting.spikesorting_merge",
        "spyglass.spikesorting.v2.sorting",
        "spyglass.spikesorting.v2.curation",
        "spyglass.spikesorting.v2.concat_member_curation",
    ):
        importlib.import_module(schema_module)

    preflight_warnings: list[str] = []
    if request.preflight:
        preflight_warnings = _run_preflight(request)

    run_summary: dict[str, Any] = {
        "pipeline_preset": pipeline_preset,
        "motion_mode": motion_mode,
        "motion_correction_params_name": request.motion_correction_params_name,
        "motion_estimate_id": None,
        "motion_estimate_supplied": request.motion_estimate_id is not None,
        "motion_corrected_recording_id": None,
        "motion_estimation_preset": None,
        "motion_removed_channel_ids": None,
        "motion_spans_without_evidence": None,
    }
    # Fails before any populate when preflight is off.
    motion_recipe = None
    if motion_mode != "off":
        try:
            motion_recipe = resolve_motion_recipe(
                request.motion_correction_params_name
            )
        except ValueError as exc:
            raise PipelineInputError(f"{request.caller}: {exc}") from exc
    # Resolved up front so a reused sort still reports its configuration.
    run_summary["sorter_config"] = resolve_preset_sort_config(bundle)
    stage_seconds: dict[str, float] = {}
    # Shared up front so a failure's shallow-copied partial summary has
    # timings and the preflight warnings.
    run_summary["stage_seconds"] = stage_seconds
    warnings_list: list[str] = list(preflight_warnings)
    run_summary["warnings"] = warnings_list
    context = request.stage_context

    run_summary["scientific_config"] = describe_scientific_setup(
        bundle,
        _run_sort_group_keys(
            is_concat,
            nwb_file_name=source_inputs.get("nwb_file_name"),
            sort_group_id=source_inputs.get("sort_group_id"),
            concat_session_group_owner=source_inputs.get(
                "concat_session_group_owner"
            ),
            concat_session_group_name=source_inputs.get(
                "concat_session_group_name"
            ),
        ),
        run_summary["sorter_config"],
        manual_excluded_times=request.manual_excluded_times,
        concat=is_concat,
        motion_mode=motion_mode,
        motion_recipe=motion_recipe,
    )
    source = _build_run_source(
        request,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
        warnings_list=warnings_list,
    )
    # Only ``"apply"`` changes what the sort reads.
    corrected: dict = {}
    if motion_recipe is not None:
        estimate_key = _run_motion_estimate(
            source.selection_fields,
            motion_recipe,
            run_summary,
            stage_seconds,
            warnings_list,
            context=context,
            motion_estimate_id=request.motion_estimate_id,
        )
        if motion_mode == "apply":
            corrected = _run_motion_correction(
                estimate_key,
                motion_recipe,
                run_summary,
                stage_seconds,
                warnings_list,
                context=context,
            )
    sorting_key, n_units = _run_sorting_stage(
        source.selection_fields,
        corrected,
        bundle,
        require_units=request.require_units,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
        warnings_list=warnings_list,
        context=context,
    )

    # Set before curation so a curation failure's partial summary has it.
    run_summary["n_units"] = n_units

    curation_key = _run_root_curation_stage(
        sorting_key,
        curation_description=request.curation_description,
        pipeline_preset=pipeline_preset,
        is_concat=is_concat,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
        warnings_list=warnings_list,
        context=context,
    )
    # None on a root-only run so downstream code cannot mistake the root.
    run_summary["auto_labeled_curation_id"] = None
    run_summary["auto_labeled_merge_id"] = None
    run_summary["auto_labeled_curation_uuid"] = None

    if request.auto_curate:
        _run_auto_curation_stage(
            sorting_key,
            curation_key,
            bundle,
            pipeline_preset=pipeline_preset,
            is_concat=is_concat,
            run_summary=run_summary,
            stage_seconds=stage_seconds,
            warnings_list=warnings_list,
            context=context,
        )

    # Concat curations reach analysis through one merge row per member.
    if is_concat:
        _run_member_curation_stage(
            sorting_key,
            source.concat_key,
            run_summary,
            stage_seconds,
            context=context,
        )

    logger.info(
        f"{context.label}: done in {sum(stage_seconds.values()):.1f} s, "
        f"n_units={n_units}, {len(warnings_list)} warning(s). "
        "result.summary() or describe_run(result) shows the receipt."
    )
    return RunResult(run_summary)
