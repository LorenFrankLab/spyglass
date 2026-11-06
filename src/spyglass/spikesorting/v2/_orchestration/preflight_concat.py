"""The raising preflight a concat run executes (``assert_concat_preflight``)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.preflight_checks import (
    _check_auto_curation_rows,
    _check_motion_stage,
    _check_sampling_rate,
    _check_session_rows,
    _check_sort_group,
    _raising_check,
    assert_preset_compute_rows,
)
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    expected_source_ids,
    supplied_motion_estimate_problem,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def assert_concat_preflight(
    concat_session_group_owner,
    concat_session_group_name,
    bundle,
    *,
    auto_curate: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id=None,
    caller: str = "run_v2_concat_pipeline",
    sort_checks: bool = True,
) -> list[str]:
    """Raise ``PreflightError`` if a concat run's prerequisites are missing.

    Concat counterpart of :func:`preflight_v2_pipeline`, using the same
    ``_check_*`` helpers for the group, each member, the preset rows, and the
    motion stage. A supplied ``motion_estimate_id`` must match the
    concatenation this run would build. Member failures name the member.
    Returns advisory warnings.
    """
    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    members = _assert_session_group_members(group_key, caller)

    # Member FKs do not cover Raw, valid times, or sort-group contents.
    for member in members:
        _assert_concat_member_inputs(
            member, bundle, caller=caller, sort_checks=sort_checks
        )

    if auto_curate:
        _check_auto_curation_rows(_raising_check(caller), bundle)

    sorter_row = assert_preset_compute_rows(
        bundle, caller=caller, sort_checks=sort_checks
    )
    if motion_mode != "off":
        _assert_concat_motion_stage(
            bundle,
            members,
            group_key,
            sorter_row=sorter_row,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
            manual_excluded_times=manual_excluded_times,
            caller=caller,
        )
    if (
        bundle.artifact_detection_params_name in (None, "none")
        and not manual_excluded_times
    ):
        return ["No artifact masking selected for the concatenated members."]
    return []


def _assert_session_group_members(group_key: dict, caller: str) -> list[dict]:
    """Raise ``PreflightError`` unless the session group has members.

    Returns
    -------
    list[dict]
        The ``SessionGroup.Member`` rows' ``member_index``,
        ``nwb_file_name``, ``sort_group_id``, ``interval_list_name`` and
        ``team_name``.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    if not (SessionGroup & group_key):
        raise PreflightError(
            f"{caller}: SessionGroup {group_key} does not exist. "
            "Create it with SessionGroup.create_group(...) first."
        )
    if not (SessionGroup.Member & group_key):
        raise PreflightError(
            f"{caller}: SessionGroup {group_key} has no members."
        )
    return (SessionGroup.Member & group_key).fetch(
        "member_index",
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        as_dict=True,
    )


def _member_check(
    member: dict, caller: str
) -> "Callable[[str, Any, str], bool]":
    """A raising check whose failures name the concat member.

    Parameters
    ----------
    member : dict
        A ``SessionGroup.Member`` row (needs ``member_index``).
    caller : str
        The public entry point, prefixed to the message.

    Returns
    -------
    Callable[[str, Any, str], bool]
        A :func:`_raising_check` prefixed ``"{caller}: concat member {i}: "``.
    """
    return _raising_check(f"{caller}: concat member {member['member_index']}")


def _assert_concat_member_inputs(
    member: dict, bundle, *, caller: str, sort_checks: bool
) -> None:
    """Raise ``PreflightError`` if a member cannot be built as a recording.

    Runs the single-session session-row, sort-group and (with
    ``sort_checks``) sampling-rate checks on the member.
    """
    check = _member_check(member, caller)
    nwb_file_name = member["nwb_file_name"]
    _check_session_rows(
        check,
        nwb_file_name,
        member["interval_list_name"],
        member["team_name"],
    )
    _check_sort_group(check, nwb_file_name, int(member["sort_group_id"]))
    if sort_checks:
        _check_sampling_rate(check, bundle, nwb_file_name, None)


def _assert_concat_motion_stage(
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    sorter_row: "dict | None",
    motion_mode,
    motion_correction_params_name: str,
    motion_estimate_id,
    manual_excluded_times,
    caller: str,
) -> None:
    """Raise ``PreflightError`` if a concat run's motion stage cannot run.

    Runs :func:`_check_motion_stage` over every member's sort group, and
    checks a supplied ``motion_estimate_id`` against the concatenation.
    """
    # assert_preset_compute_rows already checked the preprocessing row.
    motion_recipe = _check_motion_stage(
        _raising_check(caller),
        bundle,
        sort_groups=[
            (
                _member_check(member, caller),
                member["nwb_file_name"],
                int(member["sort_group_id"]),
            )
            for member in members
        ],
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        preprocessing_params_exist=True,
        sorter_row=sorter_row,
    )
    if motion_estimate_id is not None:
        _assert_supplied_concat_estimate(
            motion_estimate_id,
            motion_recipe,
            bundle,
            members,
            group_key,
            manual_excluded_times=manual_excluded_times,
            caller=caller,
        )


def _assert_supplied_concat_estimate(
    motion_estimate_id,
    motion_recipe: "MotionRecipe",
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    manual_excluded_times,
    caller: str,
) -> None:
    """Raise ``PreflightError`` unless the estimate matches this concat run.

    Its frozen members must be the recordings and artifact detections this
    run would select.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError

    concat_members = []
    for member in members:
        recording_id, artifact_detection_id = expected_source_ids(
            bundle,
            nwb_file_name=member["nwb_file_name"],
            sort_group_id=int(member["sort_group_id"]),
            interval_list_name=member["interval_list_name"],
            team_name=member["team_name"],
            manual_excluded_times=(manual_excluded_times or {}).get(
                int(member["member_index"]), []
            ),
        )
        concat_members.append(
            {
                "member_index": int(member["member_index"]),
                "recording_id": recording_id,
                "artifact_detection_id": artifact_detection_id,
            }
        )
    problem = supplied_motion_estimate_problem(
        motion_estimate_id,
        motion_recipe,
        concat_source={
            **group_key,
            "preprocessing_params_name": (bundle.preprocessing_params_name),
        },
        concat_members=concat_members,
    )
    if problem is not None:
        raise PreflightError(f"{caller}: {problem}")
