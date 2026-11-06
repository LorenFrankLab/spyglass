"""Build and validate the recording source shared by sorting and motion runs.

Schema tables are imported only when their stage is requested."""

from __future__ import annotations

from typing import Any, NamedTuple

from spyglass.spikesorting.v2._orchestration.preflight import (
    assert_concat_preflight,
    motion_request_problem,
    preflight_v2_pipeline,
)
from spyglass.spikesorting.v2._orchestration.presets import (
    _PIPELINE_PRESETS,
    _unknown_pipeline_preset_message,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _populate_once,
    _run_stage,
)


def _validate_run_request(
    caller: str,
    *,
    is_concat: bool,
    nwb_file_name,
    sort_group_id,
    interval_list_name,
    team_name,
    concat_session_group_owner,
    concat_session_group_name,
    pipeline_preset: str,
    motion_mode,
    motion_correction_params_name,
    manual_excluded_times,
    motion_estimate_id=None,
) -> tuple[Any, Any, dict]:
    """Validate a run request without touching the database.

    Checks the source inputs, preset name and motion request, and folds
    manual exclusions into the preset's artifact recipe. Runs before table
    imports, so a bad request fails even offline.

    Parameters
    ----------
    caller : str
        The public entry point, named in every error message.
    is_concat : bool
        True when ``caller`` is a concat entry point, which supplies the
        concat inputs instead of the single-session ones.
    nwb_file_name, sort_group_id, interval_list_name, team_name
        The single-session inputs (``None`` in concat mode).
    concat_session_group_owner, concat_session_group_name
        The concat inputs (``None`` in single-session mode).
    pipeline_preset : str
        A ``_PIPELINE_PRESETS`` name.
    motion_mode, motion_correction_params_name
        The motion request (see :func:`motion_request_problem`).
    manual_excluded_times
        The caller's manual exclusions (intervals, or a member-index mapping
        for concat).
    motion_estimate_id
        A saved estimate to apply (see :func:`motion_request_problem`).

    Returns
    -------
    bundle : _PipelinePreset
        The preset, with a ``"none"`` artifact recipe when manual exclusions
        need a detection output and the preset scans for none.
    manual_excluded_times : list or dict
        The normalized manual exclusions.
    source_inputs : dict
        The mode's source fields: ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and ``team_name``, or
        ``concat_session_group_owner`` and ``concat_session_group_name``.

    Raises
    ------
    PipelineInputError
        On a missing source input, an unknown preset, or a contradictory
        motion request.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    single_named = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "team_name": team_name,
    }
    concat_named = {
        "concat_session_group_owner": concat_session_group_owner,
        "concat_session_group_name": concat_session_group_name,
    }
    source_inputs = concat_named if is_concat else single_named
    missing = [
        name.removeprefix("concat_")
        for name, value in source_inputs.items()
        if value is None
    ]
    if missing:
        raise PipelineInputError(
            f"{caller}: required input(s) {', '.join(missing)} cannot be None."
        )

    if pipeline_preset not in _PIPELINE_PRESETS:
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
    from spyglass.spikesorting.v2._artifacts.manual import (
        artifact_recipe_with_manual_exclusions,
        resolve_manual_exclusions,
    )

    manual_excluded_times = resolve_manual_exclusions(
        manual_excluded_times, concat=is_concat
    )
    bundle = artifact_recipe_with_manual_exclusions(
        _PIPELINE_PRESETS[pipeline_preset], manual_excluded_times
    )
    return bundle, manual_excluded_times, source_inputs


def _run_preflight(
    caller: str,
    *,
    is_concat: bool,
    source_inputs: dict,
    bundle,
    pipeline_preset: str,
    auto_curate: bool,
    manual_excluded_times,
    motion_mode,
    motion_correction_params_name,
    motion_estimate_id=None,
    sort_checks: bool = True,
) -> list[str]:
    """Run the mode's read-only preflight; return its advisories.

    :func:`preflight_v2_pipeline` for single-session,
    :func:`assert_concat_preflight` for concat.

    Parameters
    ----------
    caller : str
        The public entry point, prefixed to each logged advisory and each
        concat preflight error.
    is_concat : bool
        The input mode.
    source_inputs : dict
        The mode's source fields: ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and ``team_name``, or
        ``concat_session_group_owner`` and ``concat_session_group_name``.
    bundle, pipeline_preset, auto_curate, manual_excluded_times
        As validated by :func:`_validate_run_request`.
    motion_mode, motion_correction_params_name, motion_estimate_id
        The motion request.
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
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.utils import logger

    if not is_concat:
        report = preflight_v2_pipeline(
            **source_inputs,
            pipeline_preset=pipeline_preset,
            auto_curate=auto_curate,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
            sort_checks=sort_checks,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))
        warnings = list(report.warnings)
        for warning in warnings:
            logger.warning(f"{caller} preflight: {warning}")
        return warnings
    return assert_concat_preflight(
        source_inputs["concat_session_group_owner"],
        source_inputs["concat_session_group_name"],
        bundle,
        auto_curate=auto_curate,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
        caller=caller,
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
    *,
    is_concat: bool,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> _RunSource:
    """Select and populate the source stages a sort reads.

    Records each stage's id, status and seconds as it goes, so a failure's
    partial summary carries every completed stage.

    Parameters
    ----------
    is_concat : bool
        The input mode.
    source_inputs : dict
        The mode's source fields (see :func:`_run_preflight`).
    bundle, manual_excluded_times
        As validated by :func:`_validate_run_request`.
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
    if not is_concat:
        return _build_single_session_source(
            source_inputs=source_inputs,
            bundle=bundle,
            manual_excluded_times=manual_excluded_times,
            run_summary=run_summary,
            stage_seconds=stage_seconds,
        )
    return _build_concat_source(
        source_inputs=source_inputs,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
    )


def _build_single_session_source(
    *,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
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
                "manual_excluded_times": manual_excluded_times,
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
    *,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> _RunSource:
    """Build each member's recording and artifact mask, then the concat.

    Parameters, return and errors as in :func:`_build_run_source`.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    run_summary["source_mode"] = "concat"
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    concat_session_group_owner = source_inputs["concat_session_group_owner"]
    concat_session_group_name = source_inputs["concat_session_group_name"]
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
            from spyglass.spikesorting.v2._sorting.artifact_mask import (
                artifact_frame_ranges,
            )

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
