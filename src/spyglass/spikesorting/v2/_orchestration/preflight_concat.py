"""The concat preflight checks, as a report or raised by a concat run."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.preflight_checks import (
    _check_auto_curation_rows,
    _check_manual_exclusions,
    _check_motion_stage,
    _check_preset_compute_rows,
    _check_sampling_rate,
    _check_session_rows,
    _check_sort_group,
    _collecting_check,
    kilosort4_double_reference_warning,
)
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    expected_source_ids,
    supplied_motion_estimate_problem,
)
from spyglass.spikesorting.v2._orchestration.preflight_report import (
    PreflightCheck,
    preflight_blocked_message,
)
from spyglass.spikesorting.v2._orchestration.stages import _run_target_label
from spyglass.spikesorting.v2.exceptions import PreflightError

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

    Runs every check of :func:`collect_concat_checks` and raises once with
    all failures (see ``preflight_blocked_message``), a member's prefixed
    with the member. Returns advisory warnings.
    """
    checks: list[PreflightCheck] = []
    warnings: list[str] = []
    collect_concat_checks(
        _collecting_check(checks),
        warnings,
        concat_session_group_owner,
        concat_session_group_name,
        bundle,
        pipeline_preset=None,
        auto_curate=auto_curate,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
        sort_checks=sort_checks,
    )
    errors = [check.fix for check in checks if not check.ok]
    if errors:
        target = _run_target_label(
            {
                "concat_session_group_owner": concat_session_group_owner,
                "concat_session_group_name": concat_session_group_name,
            }
        )
        raise PreflightError(preflight_blocked_message(caller, target, errors))
    return warnings


def collect_concat_checks(
    check: "Callable[[str, Any, str], bool]",
    warnings: list[str],
    concat_session_group_owner,
    concat_session_group_name,
    bundle,
    *,
    pipeline_preset: "str | None",
    auto_curate: bool,
    manual_excluded_times,
    motion_mode: MotionMode,
    motion_correction_params_name: "str | None",
    motion_estimate_id,
    sort_checks: bool,
) -> "tuple[list[dict], dict | None, MotionRecipe | None]":
    """Run every concat preflight check through ``check``.

    Concat counterpart of :func:`preflight_v2_pipeline`, using the same
    ``_check_*`` helpers for the group, each member, the preset rows, and the
    motion stage. A supplied ``motion_estimate_id`` must match the
    concatenation this run would build. Member failures and member
    advisories are prefixed ``"concat member {i}: "``.

    Parameters
    ----------
    check : Callable[[str, Any, str], bool]
        The check callback; a collecting one keeps going after a failure.
    warnings : list of str
        Advisories are appended here.
    concat_session_group_owner, concat_session_group_name : str
        The ``SessionGroup`` primary key.
    bundle : _PipelinePreset
        The resolved preset, with manual exclusions folded in.
    pipeline_preset : str or None
        The preset name for messages; ``None`` says "the preset".
    auto_curate, manual_excluded_times, motion_mode, \
motion_correction_params_name, motion_estimate_id, sort_checks
        As in :func:`preflight_v2_concat_pipeline`.

    Returns
    -------
    members : list of dict
        The ``SessionGroup.Member`` rows; empty when the group is missing or
        has no members (nothing else is checked then).
    sorter_row : dict or None
        The preset's ``SorterParameters`` row when it was checked and exists.
    motion_recipe : MotionRecipe or None
        The resolved motion recipe, when a motion stage was requested.
    """
    failed: list[str] = []
    outer_check = check

    def check(name: str, ok, fix: str) -> bool:
        passed = outer_check(name, ok, fix)
        if not passed:
            failed.append(name)
        return passed

    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    members = _session_group_members(check, group_key)
    if not members:
        return [], None, None
    exclusions = manual_excluded_times or {}
    member_indexes = sorted(int(m["member_index"]) for m in members)
    unknown_members = sorted(set(exclusions) - set(member_indexes))
    check(
        "manual_exclusion_members_exist",
        not unknown_members,
        f"manual_excluded_times names member_index {unknown_members}, absent "
        f"from SessionGroup {group_key}. Its members are {member_indexes}; "
        "key the exclusions by those member_index values.",
    )

    # Member FKs do not cover Raw, valid times, or sort-group contents.
    for member in members:
        _check_concat_member_inputs(
            _member_check(check, member),
            member,
            bundle,
            pipeline_preset=pipeline_preset,
            sort_checks=sort_checks,
            warnings=warnings,
            manual_excluded_times=exclusions.get(
                int(member["member_index"]), []
            ),
        )

    if auto_curate:
        _check_auto_curation_rows(check, bundle)

    sorter_row = _check_preset_compute_rows(
        check, bundle, sort_checks=sort_checks
    )
    motion_recipe = None
    if motion_mode != "off":
        motion_recipe = _check_concat_motion_stage(
            check,
            bundle,
            members,
            group_key,
            sorter_row=sorter_row,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
            manual_excluded_times=manual_excluded_times,
            inputs_ok=not failed,
        )
    for member in members:
        double_reference = kilosort4_double_reference_warning(
            sorter_row, member["nwb_file_name"], member["sort_group_id"]
        )
        if double_reference is not None:
            warnings.append(f"{_member_label(member)}: {double_reference}")
    if (
        bundle.artifact_detection_params_name in (None, "none")
        and not manual_excluded_times
    ):
        warnings.append(
            "No artifact masking selected for the concatenated members."
        )
    return members, sorter_row, motion_recipe


def _session_group_members(check, group_key: dict) -> list[dict]:
    """Check the session group exists and has members; return them.

    Returns
    -------
    list[dict]
        The ``SessionGroup.Member`` rows' ``member_index``,
        ``nwb_file_name``, ``sort_group_id``, ``interval_list_name`` and
        ``team_name`` in ``member_index`` order; empty when either check
        fails.
    """
    from spyglass.spikesorting.v2.session_group import SessionGroup

    if not check(
        "session_group_exists",
        SessionGroup & group_key,
        f"SessionGroup {group_key} does not exist. Create it with "
        "SessionGroup.create_group(...) first.",
    ):
        return []
    if not check(
        "session_group_has_members",
        SessionGroup.Member & group_key,
        f"SessionGroup {group_key} has no members.",
    ):
        return []
    return list(
        (SessionGroup.Member & group_key).fetch(
            "member_index",
            "nwb_file_name",
            "sort_group_id",
            "interval_list_name",
            "team_name",
            as_dict=True,
            order_by="member_index",
        )
    )


def _member_label(member: dict) -> str:
    """``"concat member {i}"``, the prefix of a member's messages."""
    return f"concat member {member['member_index']}"


def _member_check(
    check: "Callable[[str, Any, str], bool]", member: dict
) -> "Callable[[str, Any, str], bool]":
    """A check whose failures name the concat member.

    Parameters
    ----------
    check : Callable[[str, Any, str], bool]
        The run-level check callback.
    member : dict
        A ``SessionGroup.Member`` row.

    Returns
    -------
    Callable[[str, Any, str], bool]
        ``check`` with each fix prefixed by :func:`_member_label`.
    """
    label = _member_label(member)

    def _check(name: str, ok, fix: str) -> bool:
        return check(name, ok, f"{label}: {fix}")

    return _check


def _check_concat_member_inputs(
    check: "Callable[[str, Any, str], bool]",
    member: dict,
    bundle,
    *,
    pipeline_preset: "str | None",
    sort_checks: bool,
    manual_excluded_times,
    warnings: list[str],
) -> None:
    """Check a member can be built as a recording.

    Runs the single-session session-row, sort-group, manual-exclusion and
    (with ``sort_checks``) sampling-rate checks on the member through its
    member-prefixed ``check``. ``manual_excluded_times`` are this member's
    normalized intervals; sampling-rate advisories are appended to
    ``warnings`` with the member's label.
    """
    nwb_file_name = member["nwb_file_name"]
    _check_session_rows(
        check,
        nwb_file_name,
        member["interval_list_name"],
        member["team_name"],
    )
    _check_sort_group(
        check,
        nwb_file_name,
        int(member["sort_group_id"]),
        bundle.preprocessing_params_name,
    )
    _check_manual_exclusions(
        check,
        manual_excluded_times,
        nwb_file_name=nwb_file_name,
        interval_list_name=member["interval_list_name"],
        preprocessing_params_name=bundle.preprocessing_params_name,
    )
    if sort_checks:
        member_warnings: list[str] = []
        _check_sampling_rate(
            check, bundle, nwb_file_name, pipeline_preset, member_warnings
        )
        warnings.extend(
            f"{_member_label(member)}: {warning}" for warning in member_warnings
        )


def _check_concat_motion_stage(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    sorter_row: "dict | None",
    motion_mode,
    motion_correction_params_name: str,
    motion_estimate_id,
    manual_excluded_times,
    inputs_ok: bool,
) -> "MotionRecipe | None":
    """Check a concat run's motion stage can run; return its recipe.

    Runs :func:`_check_motion_stage` over every member's sort group, and
    checks a supplied ``motion_estimate_id`` against the concatenation when
    every earlier check passed (its ids read the member rows).
    """
    # _check_preset_compute_rows already checked the preprocessing row.
    motion_recipe = _check_motion_stage(
        check,
        bundle,
        sort_groups=[
            (
                _member_check(check, member),
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
    if (
        motion_estimate_id is not None
        and motion_recipe is not None
        and inputs_ok
    ):
        problem = _supplied_concat_estimate_problem(
            motion_estimate_id,
            motion_recipe,
            bundle,
            members,
            group_key,
            manual_excluded_times=manual_excluded_times,
        )
        check("motion_estimate_applicable", problem is None, problem or "")
    return motion_recipe


def _supplied_concat_estimate_problem(
    motion_estimate_id,
    motion_recipe: "MotionRecipe",
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    manual_excluded_times,
) -> "str | None":
    """Why the supplied estimate does not match this concat run, or None.

    Its frozen members must be the recordings and artifact detections this
    run would select.
    """
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
    return supplied_motion_estimate_problem(
        motion_estimate_id,
        motion_recipe,
        concat_source={
            **group_key,
            "preprocessing_params_name": (bundle.preprocessing_params_name),
        },
        concat_members=concat_members,
    )
