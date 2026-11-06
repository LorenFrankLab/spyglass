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

    Concat counterpart of :func:`preflight_v2_pipeline`: the group, its members,
    and -- because each member is sorted through the same single-session
    ``Recording`` build -- each member's session / sort-group / sampling-rate
    prerequisites, run through the single-session ``_check_*`` helpers (so
    the two preflights cannot drift), plus the
    ``auto_curate`` metric/rule/metric-waveform rows
    (when opted in), and the compute-time param rows + sorter binary via
    :func:`assert_preset_compute_rows`, including member artifact parameters.
    With a motion mode, also the recipe, that the preset's preprocessing
    recipe (the concatenation's) applies a temporal filter, each member's
    geometry against the estimation recipe, and (for ``"apply"``) the
    sorter's own motion correction, as in :func:`preflight_v2_pipeline`.
    A supplied ``motion_estimate_id`` must be a populated estimate, made with
    the recipe's estimation row, of a concatenation of this session group under
    the preset's preprocessing recipe whose frozen members are the member
    recordings and artifact detections this run would select.
    ``sort_checks=False`` skips the sorter-only checks (the sorter rows and
    runtime, the display analyzer recipe, each member's preset sampling
    rate and the sorter's own motion correction), as in
    :func:`preflight_v2_pipeline`. A failing per-member check names the
    member (``"concat member {i}: ..."``). Fails before member/concat
    populate. Returns advisory warnings (including explicitly disabled artifact
    masking) for symmetry with
    :func:`preflight_v2_pipeline`.
    """
    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    members = _assert_session_group_members(group_key, caller)

    # Each member is sorted through the same single-session Recording build, so
    # run preflight_v2_pipeline's per-session checks for EVERY member;
    # SessionGroup.Member's FK set validates only that the master rows exist, not
    # Raw / 'raw data valid times' / a non-empty, distinct-contact sort group /
    # a matching rate, so a partially-ingested or empty-sort-group member would
    # otherwise fail deep in the member populate with an opaque error.
    for member in members:
        _assert_concat_member_inputs(
            member, bundle, caller=caller, sort_checks=sort_checks
        )

    # Auto-curation prerequisites (preset-level, same rows as single-session),
    # only when the caller opts into auto_curate -- so a concat auto-curate run
    # fails fast on a missing metric / rule / metric-waveform recipe rather than
    # after the member + concat + sort compute.
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

    Runs the single-session session-row and sort-group checks (``Raw``,
    ``'raw data valid times'``, electrodes, distinct contact geometry) and,
    with ``sort_checks``, the preset sampling-rate check on the member. The
    session / interval / team / sort-group rows themselves are FK-guaranteed
    by ``SessionGroup.Member``, so those checks always pass here.
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

    Runs the single-session motion checks (:func:`_check_motion_stage`) with
    every member's sort group: the preset's preprocessing recipe must filter,
    the recipe must resolve, every member's geometry must support the
    estimation recipe, and an ``"apply"`` sort must not run the sorter's own
    motion correction (checked when ``sorter_row`` is given). A supplied
    ``motion_estimate_id`` must also match the concatenation the run would
    build.
    """
    # The concatenation is built with the preset's preprocessing recipe
    # (checked to exist by assert_preset_compute_rows, which
    # assert_concat_preflight runs before this function).
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

    It must be a populated estimate, made with the recipe's estimation row,
    of a concatenation of this session group under the preset's
    preprocessing recipe whose frozen members are the member recordings and
    artifact detections this run would select.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError

    # The member recordings and masks the run would select, derived
    # like the single-session preview (manual exclusions per member).
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
