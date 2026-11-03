"""The raising preflight a concat run executes (``assert_concat_preflight``)."""

from __future__ import annotations

from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.preflight_checks import (
    _SAMPLING_RATE_TOLERANCE,
    _check_auto_curation_rows,
    _raising_check,
    assert_preset_compute_rows,
)
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    expected_source_ids,
    motion_geometry_problem,
    resolve_motion_recipe,
    sorter_motion_correction_problem,
    supplied_motion_estimate_problem,
)


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
    ``Recording`` build -- the per-member ``Raw`` / ``'raw data valid times'`` /
    sort-group-electrode / sampling-rate prerequisites, plus the
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
    runtime, the display analyzer recipe and each member's preset sampling
    rate), as in :func:`preflight_v2_pipeline`. Fails before member/concat
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
    # mirror preflight_v2_pipeline's per-session prerequisites for EVERY member;
    # SessionGroup.Member's FK set validates only that the master rows exist, not
    # Raw / 'raw data valid times' / a non-empty sort group / a matching rate, so
    # a partially-ingested or empty-sort-group member would otherwise fail deep
    # in the member populate with an opaque error.
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

    assert_preset_compute_rows(bundle, caller=caller, sort_checks=sort_checks)
    if motion_mode != "off":
        _assert_concat_motion_stage(
            bundle,
            members,
            group_key,
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


def _assert_concat_member_inputs(
    member: dict, bundle, *, caller: str, sort_checks: bool
) -> None:
    """Raise ``PreflightError`` if a member cannot be built as a recording.

    Checks the member's ``Raw`` row, its ``'raw data valid times'`` interval,
    its sort-group electrodes and, with ``sort_checks``, that it samples at
    the preset's rate.
    """
    from spyglass.common import IntervalList, Raw
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb = member["nwb_file_name"]
    sort_group_id = int(member["sort_group_id"])
    tag = (
        f"concat member {member['member_index']} "
        f"({nwb!r}, sort_group_id={sort_group_id})"
    )
    if not (Raw & {"nwb_file_name": nwb}):
        raise PreflightError(
            f"{caller}: {tag} has no Raw electrical-series row (the "
            "session is ingested but its Raw data is not). Re-run ingestion "
            "(populate_all_common / insert_sessions)."
        )
    if not (
        IntervalList
        & {
            "nwb_file_name": nwb,
            "interval_list_name": "raw data valid times",
        }
    ):
        raise PreflightError(
            f"{caller}: {tag} is missing IntervalList 'raw data "
            "valid times', which the recording build reads for the raw "
            "sample bounds. Re-run ingestion."
        )
    if not (
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb, "sort_group_id": sort_group_id}
    ):
        raise PreflightError(
            f"{caller}: {tag} SortGroupV2 has zero electrode "
            "members; Recording.populate would raise 'has zero electrodes'. "
            "Recreate it with SortGroupV2.set_group_by_shank(nwb_file_name="
            "...)."
        )
    if sort_checks and bundle.sampling_rate_hz is not None:
        actual_rate = float(
            (Raw & {"nwb_file_name": nwb}).fetch1("sampling_rate")
        )
        if (
            abs(actual_rate - bundle.sampling_rate_hz)
            > _SAMPLING_RATE_TOLERANCE * bundle.sampling_rate_hz
        ):
            raise PreflightError(
                f"{caller}: {tag} samples at {actual_rate:g} Hz but "
                f"the concat preset is tuned for {bundle.sampling_rate_hz} "
                "Hz (the rate-keyed sorter row "
                f"{bundle.sorter_params_name!r} holds its clip_size / "
                "detect_interval snippet window at that rate). Every member "
                "must share the preset's acquisition rate."
            )


def _assert_concat_motion_stage(
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    motion_mode,
    motion_correction_params_name: str,
    motion_estimate_id,
    manual_excluded_times,
    caller: str,
) -> None:
    """Raise ``PreflightError`` if a concat run's motion stage cannot run.

    The recipe must resolve, the preset's preprocessing recipe must filter,
    every member's geometry must support the estimation recipe, an
    ``"apply"`` sort must not run the sorter's own motion correction, and a
    supplied ``motion_estimate_id`` must match the concatenation the run
    would build.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.motion import (
        preprocessing_filter_problem,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    try:
        motion_recipe = resolve_motion_recipe(motion_correction_params_name)
    except ValueError as exc:
        raise PreflightError(f"{caller}: {exc}") from exc
    # The concatenation is built with the preset's preprocessing recipe
    # (checked to exist by assert_preset_compute_rows, which
    # assert_concat_preflight runs before this function).
    problem = preprocessing_filter_problem(bundle.preprocessing_params_name)
    if problem is not None:
        raise PreflightError(
            f"{caller}: motion_mode={motion_mode!r}: {problem}"
        )
    for member in members:
        problem = motion_geometry_problem(
            member["nwb_file_name"],
            int(member["sort_group_id"]),
            motion_recipe.resolved_estimation,
        )
        if problem is not None:
            raise PreflightError(
                f"{caller}: concat member {member['member_index']} "
                f"{problem}"
            )
    if motion_mode == "apply":
        sorter_params = (
            SorterParameters
            & {
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
            }
        ).fetch1("params")
        problem = sorter_motion_correction_problem(
            bundle.sorter, sorter_params, bundle.sorter_params_name
        )
        if problem is not None:
            raise PreflightError(f"{caller}: {problem}")
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
