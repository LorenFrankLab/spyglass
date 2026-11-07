"""Estimate and apply motion for a v2 pipeline source.

Sorting runs its motion stages through the same helpers."""

from __future__ import annotations

import uuid
from typing import Any, cast

from spyglass.spikesorting.v2._orchestration.preflight import (
    resolve_motion_recipe,
    supplied_motion_estimate_problem,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    DEFAULT_PIPELINE_PRESET,
)
from spyglass.spikesorting.v2._orchestration.types import EstimateMotionReceipt
from spyglass.spikesorting.v2._orchestration.source import (
    _build_run_source,
    _run_preflight,
    _validate_run_request,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _populate_once,
    _run_stage,
)
from spyglass.spikesorting.v2._recording.source import SourceLineage
from spyglass.utils import logger


def _run_motion_estimate(
    source: dict,
    motion_recipe,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
    motion_estimate_id=None,
) -> dict:
    """Select and populate the motion estimate of a run's source.

    Records the estimate id, preset, status, seconds, and spans without
    evidence (warned when non-empty) in ``run_summary`` / ``stage_seconds``.

    Parameters
    ----------
    source : dict
        The ``MotionEstimateSelection`` source (``_RunSource.selection_fields``).
    motion_recipe : MotionRecipe
        The resolved ``MotionCorrectionParameters`` recipe.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).
    warnings_list : list
        The run's warnings (appended to).
    motion_estimate_id : uuid.UUID or str, optional
        A populated estimate of ``source`` with the recipe's estimation row
        to reuse; never recomputed.

    Returns
    -------
    dict
        ``{"motion_estimate_id": ...}``.

    Raises
    ------
    PipelineStageError
        If the selection is refused, the estimation fails, or the supplied
        estimate does not match the source or recipe (stage
        ``"motion_estimate"``).
    """
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )

    def _preset() -> str:
        return (MotionEstimate & estimate_key).fetch1("resolved_params")[
            "preset"
        ]

    if motion_estimate_id is not None:
        estimate_key = {
            "motion_estimate_id": uuid.UUID(str(motion_estimate_id))
        }
        if "concat_recording_id" in source:
            lineage = SourceLineage(
                "concatenated_recording", dict(source), None
            )
        else:
            lineage = SourceLineage(
                "recording",
                {"recording_id": source["recording_id"]},
                source["artifact_detection_id"],
            )

        def _estimate() -> str:
            problem = supplied_motion_estimate_problem(
                motion_estimate_id, motion_recipe, source_lineage=lineage
            )
            if problem is not None:
                raise ValueError(problem)
            return _preset()

        exists = True
    else:
        # Inside ``_run_stage`` so a refused selection is a PipelineStageError.
        estimate_key, _, _ = _run_stage(
            "motion_estimate",
            False,
            lambda: MotionEstimateSelection.insert_selection(
                {
                    **source,
                    "motion_estimation_params_name": motion_recipe.recipe[
                        "motion_estimation_params_name"
                    ],
                }
            ),
            run_summary,
        )

        def _estimate() -> str:
            _populate_once(MotionEstimate, estimate_key)
            return _preset()

        exists = bool(MotionEstimate & estimate_key)

    (
        run_summary["motion_estimation_preset"],
        run_summary["motion_estimate_status"],
        stage_seconds["motion_estimate"],
    ) = _run_stage("motion_estimate", exists, _estimate, run_summary)
    run_summary["motion_estimate_id"] = estimate_key["motion_estimate_id"]
    # Warn, not refuse: dropped-frame gaps can leave spans with no peak.
    empty_spans = MotionEstimate().get_spans_without_evidence(estimate_key)
    run_summary["motion_spans_without_evidence"] = empty_spans
    if empty_spans:
        empty_span_warning = (
            f"Motion estimate {estimate_key['motion_estimate_id']}: "
            f"{len(empty_spans)} continuity span(s) kept no peaks "
            "(source times "
            + ", ".join(
                f"{span['source_start_s']:.3f}-{span['source_end_s']:.3f} s"
                for span in empty_spans
            )
            + "); the displacement there rests on the estimator's "
            "temporal prior only, so a correction applied to them is not "
            "evidence-based. See run_summary"
            "['motion_spans_without_evidence'] or "
            "MotionEstimate().get_spans_without_evidence(...)."
        )
        logger.warning(empty_span_warning)
        warnings_list.append(empty_span_warning)
    return dict(estimate_key)


def _run_motion_correction(
    estimate_key: dict,
    motion_recipe,
    run_summary: dict,
    stage_seconds: dict,
) -> dict:
    """Select and populate the corrected recording of a saved estimate.

    Records the corrected recording id, the channels its border mode removed,
    its status and seconds in ``run_summary`` / ``stage_seconds``.

    Parameters
    ----------
    estimate_key : dict
        ``{"motion_estimate_id": ...}`` of a populated ``MotionEstimate``.
    motion_recipe : MotionRecipe
        The resolved ``MotionCorrectionParameters`` recipe, whose
        interpolation row is applied.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    dict
        ``{"motion_corrected_recording_id": ...}``, the key fragment the sort
        selection needs.

    Raises
    ------
    PipelineStageError
        If the selection is refused or the interpolation fails (stage
        ``"motion_corrected_recording"``).
    """
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
    )

    corrected_key, _, _ = _run_stage(
        "motion_corrected_recording",
        False,
        lambda: MotionCorrectedRecordingSelection.insert_selection(
            {
                "motion_estimate_id": estimate_key["motion_estimate_id"],
                "motion_interpolation_params_name": motion_recipe.recipe[
                    "motion_interpolation_params_name"
                ],
            }
        ),
        run_summary,
    )

    def _correct() -> list:
        _populate_once(MotionCorrectedRecording, corrected_key)
        return list(
            (MotionCorrectedRecording & corrected_key).fetch1(
                "removed_channel_ids"
            )
        )

    (
        run_summary["motion_removed_channel_ids"],
        run_summary["motion_corrected_recording_status"],
        stage_seconds["motion_corrected_recording"],
    ) = _run_stage(
        "motion_corrected_recording",
        bool(MotionCorrectedRecording & corrected_key),
        _correct,
        run_summary,
    )
    run_summary["motion_corrected_recording_id"] = corrected_key[
        "motion_corrected_recording_id"
    ]
    return dict(corrected_key)


def estimate_motion(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    *,
    motion_correction_params_name: str,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    preflight: bool = True,
    manual_excluded_times=None,
) -> "EstimateMotionReceipt":
    """Save the motion estimate of one sort group's source, without sorting.

    Builds the source as ``run_v2_pipeline`` does and saves its
    ``MotionEstimate``. Inspect it, then sort with exactly that estimate::

        receipt = estimate_motion(..., motion_correction_params_name=name)
        run_v2_pipeline(
            ...,
            motion_mode="apply",
            motion_correction_params_name=name,
            motion_estimate_id=receipt["motion_estimate_id"],
        )

    Idempotent: the estimate id is content-addressed.

    Experimental: no motion recipe is validated for a probe.

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name
        The sort group, as in :func:`run_v2_pipeline`.
    pipeline_preset
        The preset whose preprocessing and artifact rows build the source; use
        the preset of the run that will apply the estimate.
    preflight
        If True (default), run the read-only preflight for
        ``motion_mode="estimate"`` without sorter checks. A failure raises
        ``PreflightError``.
    manual_excluded_times
        Manual exclusions, as in :func:`run_v2_pipeline`; they are part of
        the mask the estimate is made under.
    motion_correction_params_name
        The ``MotionCorrectionParameters`` recipe (required). Its estimation
        row is estimated here; its interpolation row is the one the later
        ``"apply"`` run uses.

    Returns
    -------
    EstimateMotionReceipt
        ``motion_estimate_id``, the estimation preset, spans without
        evidence, ``motion_diagnostics``, source-stage ids and statuses,
        ``stage_seconds``, and ``warnings``.

    Raises
    ------
    PipelineInputError
        Before any database access: if ``pipeline_preset`` is unknown, or a
        source input or ``motion_correction_params_name`` is ``None``.
    PreflightError
        If ``preflight=True`` and a prerequisite is missing.
    PipelineStageError
        If a source stage or the estimation fails; names the stage and
        carries the partial summary.
    ValueError
        If the recipe row is missing and ``preflight=False``.
    """
    return _estimate_motion(
        "estimate_motion",
        is_concat=False,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        motion_correction_params_name=motion_correction_params_name,
        pipeline_preset=pipeline_preset,
        preflight=preflight,
        manual_excluded_times=manual_excluded_times,
    )


def estimate_concat_motion(
    session_group_owner: str,
    session_group_name: str,
    *,
    motion_correction_params_name: str,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    preflight: bool = True,
    manual_excluded_times=None,
) -> "EstimateMotionReceipt":
    """Save the motion estimate of a session group's concatenation.

    Concat counterpart of :func:`estimate_motion`. Apply it with
    ``run_v2_concat_pipeline(..., motion_mode="apply",
    motion_estimate_id=receipt["motion_estimate_id"])``.

    Parameters
    ----------
    session_group_owner, session_group_name
        The primary key of an existing ``SessionGroup``.
    manual_excluded_times
        A mapping of each ``member_index`` to its exclusions, as in
        :func:`run_v2_concat_pipeline`.
    motion_correction_params_name, pipeline_preset, preflight
        As in :func:`estimate_motion`.

    Returns
    -------
    EstimateMotionReceipt
        As in :func:`estimate_motion`, with the member and concat stage ids.

    Raises
    ------
    PipelineInputError, PreflightError, PipelineStageError, ValueError
        As in :func:`estimate_motion`.
    """
    return _estimate_motion(
        "estimate_concat_motion",
        is_concat=True,
        concat_session_group_owner=session_group_owner,
        concat_session_group_name=session_group_name,
        motion_correction_params_name=motion_correction_params_name,
        pipeline_preset=pipeline_preset,
        preflight=preflight,
        manual_excluded_times=manual_excluded_times,
    )


def _estimate_motion(
    caller: str,
    *,
    is_concat: bool,
    nwb_file_name: "str | None" = None,
    sort_group_id: "int | None" = None,
    interval_list_name: "str | None" = None,
    team_name: "str | None" = None,
    concat_session_group_owner: "str | None" = None,
    concat_session_group_name: "str | None" = None,
    motion_correction_params_name: str,
    pipeline_preset: str,
    preflight: bool,
    manual_excluded_times,
) -> "EstimateMotionReceipt":
    """Shared body of ``estimate_motion`` / ``estimate_concat_motion``."""
    bundle, manual_excluded_times, source_inputs = _validate_run_request(
        caller,
        is_concat=is_concat,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        concat_session_group_owner=concat_session_group_owner,
        concat_session_group_name=concat_session_group_name,
        pipeline_preset=pipeline_preset,
        motion_mode="estimate",
        motion_correction_params_name=motion_correction_params_name,
        manual_excluded_times=manual_excluded_times,
    )
    warnings_list: list[str] = []
    if preflight:
        warnings_list = _run_preflight(
            caller,
            is_concat=is_concat,
            source_inputs=source_inputs,
            bundle=bundle,
            pipeline_preset=pipeline_preset,
            auto_curate=False,
            manual_excluded_times=manual_excluded_times,
            motion_mode="estimate",
            motion_correction_params_name=motion_correction_params_name,
            sort_checks=False,
        )
    from spyglass.spikesorting.v2.motion import MotionEstimate

    stage_seconds: dict[str, float] = {}
    run_summary: dict[str, Any] = {
        "pipeline_preset": pipeline_preset,
        "motion_correction_params_name": motion_correction_params_name,
        "motion_estimate_id": None,
        "motion_estimation_preset": None,
        "motion_spans_without_evidence": None,
        "stage_seconds": stage_seconds,
        "warnings": warnings_list,
    }
    motion_recipe = resolve_motion_recipe(motion_correction_params_name)
    source = _build_run_source(
        is_concat=is_concat,
        source_inputs=source_inputs,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
    )
    estimate_key = _run_motion_estimate(
        source.selection_fields,
        motion_recipe,
        run_summary,
        stage_seconds,
        warnings_list,
    )
    diagnostics = (
        (MotionEstimate & estimate_key)
        .proj(
            "n_peaks_detected",
            "n_peaks_kept",
            "max_abs_displacement_um",
            "n_temporal_bins",
        )
        .fetch1()
    )
    run_summary["motion_diagnostics"] = {
        "n_peaks_detected": int(diagnostics["n_peaks_detected"]),
        "n_peaks_kept": int(diagnostics["n_peaks_kept"]),
        "max_abs_displacement_um": float(
            diagnostics["max_abs_displacement_um"]
        ),
        "n_temporal_bins": int(diagnostics["n_temporal_bins"]),
    }
    return cast("EstimateMotionReceipt", run_summary)
