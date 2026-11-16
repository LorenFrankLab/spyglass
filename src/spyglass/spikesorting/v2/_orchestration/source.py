"""Build and validate the recording source shared by sorting and motion runs.

Schema tables are imported only when their stage is requested."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

from spyglass.spikesorting.v2._orchestration.preflight import (
    assert_concat_preflight,
    motion_request_problem,
    preflight_v2_pipeline,
)
from spyglass.spikesorting.v2._orchestration.presets import (
    _find_pipeline_preset,
    _unknown_pipeline_preset_message,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _populate_once,
    _run_stage,
)
from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._artifacts.manual import (
    artifact_recipe_with_manual_exclusions,
    resolve_manual_exclusions,
)
from spyglass.spikesorting.v2._core.signal_math import artifact_frame_ranges
from spyglass.spikesorting.v2.exceptions import (
    PipelineInputError,
    PreflightError,
)
from spyglass.utils import logger


@dataclass(frozen=True)
class _RunRequest:
    """A validated run request, read by every step below the entry points.

    Built only by :func:`_validate_run_request`, the one place the run options
    are named, so a step cannot fall back to an option's default.

    Attributes
    ----------
    caller : str
        The public entry point, named in errors and logged advisories.
    is_concat : bool
        True for a concat entry point.
    source_inputs : Mapping[str, Any]
        Read-only source fields: ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and ``team_name``, or
        ``concat_session_group_owner`` and ``concat_session_group_name``.
    pipeline_preset : str
        The preset name.
    bundle : _PipelinePreset
        The preset, with a ``"none"`` artifact recipe when manual exclusions
        need a detection output and the preset scans for none.
    manual_excluded_times : list or dict
        The normalized manual exclusions (a member-index mapping for concat).
    curation_description : str
    require_units : bool
    auto_curate : bool
    preflight : bool
    motion_mode : MotionMode
    motion_correction_params_name : str or None
    motion_estimate_id : uuid.UUID, str or None
        The caller's options, unchanged.
    """

    caller: str
    is_concat: bool
    source_inputs: Mapping[str, Any]
    pipeline_preset: str
    bundle: Any
    manual_excluded_times: Any
    curation_description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    motion_mode: MotionMode
    motion_correction_params_name: "str | None"
    motion_estimate_id: Any


def _validate_run_request(
    caller: str,
    *,
    is_concat: bool,
    source_inputs: dict,
    pipeline_preset: str,
    curation_description: str,
    require_units: bool,
    auto_curate: bool,
    preflight: bool,
    manual_excluded_times,
    motion_mode: MotionMode,
    motion_correction_params_name: "str | None",
    motion_estimate_id,
) -> _RunRequest:
    """Validate a run request before any stage runs.

    Checks the source inputs, preset name and motion request, and folds
    manual exclusions into the preset's artifact recipe. Runs before table
    imports. A shipped preset name needs no database; any other name is
    looked up in ``PipelinePreset``.

    Parameters
    ----------
    caller : str
        The public entry point, named in every error message.
    is_concat : bool
        True when ``caller`` is a concat entry point, which supplies the
        concat inputs instead of the single-session ones.
    source_inputs : dict
        ``nwb_file_name``, ``sort_group_id``, ``interval_list_name`` and
        ``team_name``, or ``concat_session_group_owner`` and
        ``concat_session_group_name``; any may be ``None`` (rejected here).
    pipeline_preset : str
        A pipeline-preset name.
    curation_description, require_units, auto_curate, preflight
        The caller's run options, carried unchanged.
    manual_excluded_times
        The caller's manual exclusions (intervals, or a member-index mapping
        for concat).
    motion_mode, motion_correction_params_name, motion_estimate_id
        The motion request (see :func:`motion_request_problem`).

    Returns
    -------
    _RunRequest

    Raises
    ------
    PipelineInputError
        On a missing source input, an unknown preset, or a contradictory
        motion request.
    """
    missing = [
        name.removeprefix("concat_")
        for name, value in source_inputs.items()
        if value is None
    ]
    if missing:
        raise PipelineInputError(
            f"{caller}: required input(s) {', '.join(missing)} cannot be None."
        )

    preset = _find_pipeline_preset(pipeline_preset)
    if preset is None:
        raise PipelineInputError(
            _unknown_pipeline_preset_message(
                pipeline_preset,
                caller=caller,
                hint=(
                    "Call spyglass.spikesorting.v2.pipeline."
                    "describe_pipeline_presets() to see what each preset "
                    "does, or list_pipeline_presets() for just the names."
                ),
            )
        )
    motion_problem = motion_request_problem(
        motion_mode, motion_correction_params_name, motion_estimate_id
    )
    if motion_problem is not None:
        raise PipelineInputError(f"{caller}: {motion_problem}")

    manual_excluded_times = resolve_manual_exclusions(
        manual_excluded_times, concat=is_concat
    )
    bundle = artifact_recipe_with_manual_exclusions(
        preset, manual_excluded_times
    )
    return _RunRequest(
        caller=caller,
        is_concat=is_concat,
        source_inputs=MappingProxyType(dict(source_inputs)),
        pipeline_preset=pipeline_preset,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        curation_description=curation_description,
        require_units=require_units,
        auto_curate=auto_curate,
        preflight=preflight,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
    )


def _run_preflight(
    request: _RunRequest, *, sort_checks: bool = True
) -> list[str]:
    """Run the mode's read-only preflight; return its advisories.

    :func:`preflight_v2_pipeline` for single-session,
    :func:`assert_concat_preflight` for concat. Each logged advisory and
    each concat preflight error names ``request.caller``.

    Parameters
    ----------
    request : _RunRequest
        The validated request.
    sort_checks : bool
        False skips the checks only a sort needs (see
        :func:`preflight_v2_pipeline`).

    Returns
    -------
    list[str]
        Non-blocking advisories, each also logged.

    Raises
    ------
    PreflightError
        If a prerequisite is missing.
    """
    if not request.is_concat:
        report = preflight_v2_pipeline(
            **request.source_inputs,
            pipeline_preset=request.pipeline_preset,
            auto_curate=request.auto_curate,
            manual_excluded_times=request.manual_excluded_times,
            motion_mode=request.motion_mode,
            motion_correction_params_name=request.motion_correction_params_name,
            motion_estimate_id=request.motion_estimate_id,
            sort_checks=sort_checks,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))
        warnings = list(report.warnings)
        for warning in warnings:
            logger.warning(f"{request.caller} preflight: {warning}")
        return warnings
    return assert_concat_preflight(
        request.source_inputs["concat_session_group_owner"],
        request.source_inputs["concat_session_group_name"],
        request.bundle,
        auto_curate=request.auto_curate,
        manual_excluded_times=request.manual_excluded_times,
        motion_mode=request.motion_mode,
        motion_correction_params_name=request.motion_correction_params_name,
        motion_estimate_id=request.motion_estimate_id,
        caller=request.caller,
        sort_checks=sort_checks,
    )


class _RunSource(NamedTuple):
    """The built source of a run, keyed for the stages that read it.

    Attributes
    ----------
    selection_fields : dict
        Source fields shared by ``MotionEstimateSelection`` and
        ``SortingSelection``: ``recording_id`` and ``artifact_detection_id``,
        or ``concat_recording_id``.
    concat_key : dict or None
        The ``ConcatenatedRecordingSelection`` PK in concat mode.
    """

    selection_fields: dict
    concat_key: "dict | None"


def _build_run_source(
    request: _RunRequest, *, run_summary: dict, stage_seconds: dict
) -> _RunSource:
    """Select and populate the source stages a sort reads.

    Records each stage's id, status and seconds as it goes, so a failure's
    partial summary carries every completed stage.

    Parameters
    ----------
    request : _RunRequest
        The validated request; its source inputs, preset and exclusions.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    _RunSource

    Raises
    ------
    PipelineStageError
        If a source stage's populate fails.
    PipelineInputError
        If concat manual exclusions name an absent member.
    """
    build = (
        _build_concat_source
        if request.is_concat
        else _build_single_session_source
    )
    return build(request, run_summary=run_summary, stage_seconds=stage_seconds)


def _build_single_session_source(
    request: _RunRequest, *, run_summary: dict, stage_seconds: dict
) -> _RunSource:
    """Select and populate the ``Recording`` and its artifact detection.

    The artifact-detection stage is ``"skipped"`` when the preset runs none.
    Parameters and return as in :func:`_build_run_source`.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    source_inputs = request.source_inputs
    bundle = request.bundle
    nwb_file_name = source_inputs["nwb_file_name"]
    sort_group_id = source_inputs["sort_group_id"]
    run_summary["source_mode"] = "single_session"
    recording_key = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": source_inputs["interval_list_name"],
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": source_inputs["team_name"],
        }
    )
    (
        _,
        run_summary["recording_status"],
        stage_seconds["recording"],
    ) = _run_stage(
        "recording",
        bool(Recording & recording_key),
        lambda: _populate_once(Recording, recording_key),
        run_summary,
    )
    run_summary["recording_id"] = recording_key["recording_id"]

    if bundle.artifact_detection_params_name is None:
        artifact_detection_id = None
        run_summary["artifact_detection_status"] = "skipped"
        stage_seconds["artifact_detection"] = 0.0
    else:
        artifact_detection_key = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": recording_key["recording_id"],
                "artifact_detection_params_name": bundle.artifact_detection_params_name,
                "manual_excluded_times": request.manual_excluded_times,
            }
        )
        (
            _,
            run_summary["artifact_detection_status"],
            stage_seconds["artifact_detection"],
        ) = _run_stage(
            "artifact_detection",
            bool(RecordingArtifactDetection & artifact_detection_key),
            lambda: _populate_once(
                RecordingArtifactDetection, artifact_detection_key
            ),
            run_summary,
        )
        artifact_detection_id = artifact_detection_key["artifact_detection_id"]
    run_summary["artifact_detection_id"] = artifact_detection_id
    source = {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_id,
    }
    return _RunSource(source, None)


def _build_concat_source(
    request: _RunRequest, *, run_summary: dict, stage_seconds: dict
) -> _RunSource:
    """Build each member's recording and artifact mask, then the concat.

    Parameters, return and errors as in :func:`_build_run_source`.
    """
    run_summary["source_mode"] = "concat"
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    bundle = request.bundle
    manual_excluded_times = request.manual_excluded_times
    concat_session_group_owner = request.source_inputs[
        "concat_session_group_owner"
    ]
    concat_session_group_name = request.source_inputs[
        "concat_session_group_name"
    ]
    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    # Order matches the concat identity and snapshot ordering.
    members = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    unknown_members = set(manual_excluded_times) - {
        int(m["member_index"]) for m in members
    }
    if unknown_members:
        raise PipelineInputError(
            f"Manual exclusions name absent concat members: {sorted(unknown_members)}"
        )
    member_recording_keys = _build_member_recordings(
        members, bundle, run_summary, stage_seconds
    )
    artifact_ids = _build_member_artifacts(
        members,
        member_recording_keys,
        bundle,
        manual_excluded_times,
        run_summary,
        stage_seconds,
    )

    concat_key = ConcatenatedRecordingSelection.insert_selection(
        {
            "session_group_owner": concat_session_group_owner,
            "session_group_name": concat_session_group_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
        },
        artifact_detection_ids=artifact_ids,
    )
    (
        _,
        run_summary["concat_recording_status"],
        stage_seconds["concat_recording"],
    ) = _run_stage(
        "concat_recording",
        bool(ConcatenatedRecording & concat_key),
        lambda: _populate_once(ConcatenatedRecording, concat_key),
        run_summary,
    )
    run_summary["concat_recording_id"] = concat_key["concat_recording_id"]
    concat_row = (ConcatenatedRecording & concat_key).fetch1()
    valid_duration = sum(
        end - start for start, end in concat_row["obs_intervals"]
    )
    run_summary["artifact_masked_duration_s"] = float(
        concat_row["total_duration_s"] - valid_duration
    )
    source = {"concat_recording_id": concat_key["concat_recording_id"]}
    return _RunSource(source, dict(concat_key))


def _build_member_recordings(
    members: list[dict],
    bundle,
    run_summary: dict,
    stage_seconds: dict,
) -> list[dict]:
    """Select and populate every concat member's ``Recording`` as one stage.

    Parameters
    ----------
    members : list[dict]
        The ``SessionGroup.Member`` rows, in ``member_index`` order.
    bundle : _PipelinePreset
        The validated preset.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    list[dict]
        The members' ``RecordingSelection`` keys, in member order.
    """
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    member_recording_keys = [
        RecordingSelection.insert_selection(
            {
                "nwb_file_name": member["nwb_file_name"],
                "sort_group_id": int(member["sort_group_id"]),
                "interval_list_name": member["interval_list_name"],
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": member["team_name"],
            }
        )
        for member in members
    ]

    def _populate_member_recordings():
        for key in member_recording_keys:
            _populate_once(Recording, key)

    (
        _,
        run_summary["member_recording_status"],
        stage_seconds["member_recording"],
    ) = _run_stage(
        "member_recording",
        len(Recording & member_recording_keys) == len(member_recording_keys),
        _populate_member_recordings,
        run_summary,
    )
    run_summary["member_recording_ids"] = [
        key["recording_id"] for key in member_recording_keys
    ]
    return member_recording_keys


def _build_member_artifacts(
    members: list[dict],
    member_recording_keys: list[dict],
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> dict:
    """Select and populate every member's artifact detection as one stage.

    Records each member's detection id, status and masked duration in
    ``run_summary["member_artifacts"]``.

    Parameters
    ----------
    members : list[dict]
        The ``SessionGroup.Member`` rows, in ``member_index`` order.
    member_recording_keys : list[dict]
        The members' ``RecordingSelection`` keys, in the same order.
    bundle, manual_excluded_times
        As validated by :func:`_validate_run_request` (exclusions keyed by
        ``member_index``).
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    dict
        ``member_index`` to ``artifact_detection_id`` (``None`` without
        artifact detection), the concat selection's member masks.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording

    artifact_ids = {int(member["member_index"]): None for member in members}
    run_summary["member_artifacts"] = []
    if bundle.artifact_detection_params_name is None:
        run_summary["member_artifact_detection_status"] = "skipped"
        stage_seconds["member_artifact_detection"] = 0.0
    else:
        artifact_keys = [
            RecordingArtifactSelection.insert_selection(
                {
                    **recording_key,
                    "artifact_detection_params_name": bundle.artifact_detection_params_name,
                    "manual_excluded_times": manual_excluded_times.get(
                        int(member["member_index"]), []
                    ),
                }
            )
            for member, recording_key in zip(
                members, member_recording_keys, strict=True
            )
        ]

        def _populate_member_artifacts():
            for member, artifact_key, recording_key in zip(
                members, artifact_keys, member_recording_keys, strict=True
            ):
                reused = bool(RecordingArtifactDetection & artifact_key)
                _populate_once(RecordingArtifactDetection, artifact_key)
                artifact_id = artifact_key["artifact_detection_id"]
                artifact_ids[int(member["member_index"])] = artifact_id
                member_recording = Recording().get_recording(recording_key)
                kept = (
                    RecordingArtifactDetection().get_artifact_removed_intervals(
                        artifact_key
                    )
                )
                # Count frames actually masked, excluding wall-clock gaps.
                excluded = artifact_frame_ranges(
                    member_recording,
                    kept,
                    artifact_detection_id=artifact_id,
                    recording_id=recording_key["recording_id"],
                )
                masked_duration = (
                    sum(end - start for start, end in excluded)
                    / member_recording.get_sampling_frequency()
                )
                run_summary["member_artifacts"].append(
                    {
                        "member_index": int(member["member_index"]),
                        "artifact_detection_id": artifact_id,
                        "status": "reused" if reused else "computed",
                        "masked_duration_s": masked_duration,
                    }
                )

        (
            _,
            run_summary["member_artifact_detection_status"],
            stage_seconds["member_artifact_detection"],
        ) = _run_stage(
            "member_artifact_detection",
            len(RecordingArtifactDetection & artifact_keys)
            == len(artifact_keys),
            _populate_member_artifacts,
            run_summary,
        )
    return artifact_ids
