"""Plan, inspect, and run cross-session unit matching."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from spyglass.spikesorting.v2._matching.planning import (
    UnitMatchInputPlan,
    UnitMatchPlan,
)
from spyglass.spikesorting.v2._orchestration.types import (
    RunV2UnitMatchSummary,
    UnitMatchInputSummary,
    UnitMatchMemberChoices,
    UnitMatchStageSeconds,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _populate_once,
    _run_stage,
)
from spyglass.spikesorting.v2._matching.planning import (
    build_unit_match_input_plan,
    build_unit_match_plan,
    member_choices_to_dataframe,
)
from spyglass.spikesorting.v2._matching.preflight import (
    preflight_v2_unit_match,
    unlabeled_units_warning,
)
from spyglass.spikesorting.v2.analysis_selection import UnitSelectionSnapshot
from spyglass.spikesorting.v2.curation_api import CurationRef
from spyglass.spikesorting.v2.exceptions import (
    PipelineInputError,
    PreflightError,
)

if TYPE_CHECKING:
    import pandas as pd


def run_v2_unit_match(
    plan: "UnitMatchPlan | UnitMatchInputPlan | None" = None,
    *,
    session_group_owner: "str | None" = None,
    session_group_name: "str | None" = None,
    matcher_params_name: str = "unitmatch_default",
    curation_choices: "dict | None" = None,
    unit_policy: "str | None" = None,
) -> RunV2UnitMatchSummary:
    """Match units across matching inputs in one call, then track them.

    A matching input is one curated sort of a recording or a same-day
    concatenation. Call forms:

    - ``run_v2_unit_match(plan)`` with a plan from :func:`plan_v2_unit_match`,
      :func:`plan_v2_unit_match_from_curations`, or
      :func:`plan_v2_unit_match_from_sorts` (recommended).
    - ``run_v2_unit_match(session_group_owner=..., session_group_name=...,
      curation_choices={...})``: the explicit group form.

    Runs selection, ``UnitMatch`` (pairwise matches), then ``TrackedUnit``
    (identity across sessions). Idempotent. The explicit form requires
    ``curation_choices`` because an implicit "latest" curation is not
    reproducible; build it from :func:`describe_unit_match_choices`.

    Parameters
    ----------
    plan : UnitMatchPlan or UnitMatchInputPlan, optional
        Reviewable plan returned by :func:`plan_v2_unit_match` or
        :func:`plan_v2_unit_match_from_sorts`. Recommended.
    session_group_owner, session_group_name : str, optional
        Identify the ``SessionGroup`` whose members are matched in the explicit
        form.
    matcher_params_name : str
        ``MatcherParameters`` row to use. Default ``"unitmatch_default"`` (the
        ``UnitMatchPy`` backend; ``MatcherParameters.insert_default()`` seeds it).
        The same row supplies ``TrackedUnit``'s threshold / budget.
    curation_choices : dict
        ``{member_index: {"sorting_id": ..., "curation_id": ...}}``, one
        committed curation per member. Required in the explicit form.
    unit_policy : str, optional
        Explicit form only: the named unit eligibility policy frozen for each
        member; it must exclude ``mua``. Default ``None`` uses
        ``"v2_accepted_single_units"``; choose
        ``"v2_unflagged_single_units"`` for automatic-only output. A plan
        carries its own policy.

    Returns
    -------
    RunV2UnitMatchSummary
        Group identity (``None`` for named sorts), ``matcher_params_name``,
        ``unit_match_id``, ``inputs`` (one :class:`UnitMatchInputSummary` per
        input, in ``input_index`` order), ``n_pairs``, ``n_tracked_units``,
        per-stage status (``"computed"`` / ``"reused"``), ``stage_seconds``,
        and advisory ``warnings``.

    Raises
    ------
    PipelineInputError
        If ``curation_choices`` is ``None`` or ``matcher_params_name`` is not a
        known ``MatcherParameters`` row.
    PipelineInputError
        Also if ``plan`` is not a plan, is combined with the explicit
        arguments, or could not pin a curation for every member / sort.
    ValueError
        From ``insert_selection`` (group form) on a missing/extra member
        choice, a non-existent curation, or a curation that does not belong to
        its member; or from ``insert_inputs`` (both forms) on an invalid input
        set, e.g. a curation with unapplied merges, inputs reusing original
        raw samples (``OverlappingMatchInputError``), a multi-day concatenation input,
        or a channel-geometry mismatch across inputs.
    PreflightError
        A planned input's generation, frozen unit population, coverage,
        geometry, or matcher availability changed. Raised before any write.
    PipelineStageError
        If ``UnitMatch`` / ``TrackedUnit`` populate fails (names the stage,
        carries the partial summary).
    """
    input_curations = None
    input_unit_selections = None
    expected_input_set_hash = None
    preflight_warnings = []
    if plan is not None:
        if not isinstance(plan, (UnitMatchPlan, UnitMatchInputPlan)):
            raise PipelineInputError(
                "run_v2_unit_match: plan must be a UnitMatchPlan from "
                "plan_v2_unit_match or a UnitMatchInputPlan from "
                "plan_v2_unit_match_from_sorts, or omit plan and pass "
                "session_group_owner= and session_group_name=."
            )
        # matcher_params_name at its default counts as unset.
        if (
            session_group_owner is not None
            or session_group_name is not None
            or curation_choices is not None
            or unit_policy is not None
            or matcher_params_name != "unitmatch_default"
        ):
            raise PipelineInputError(
                "run_v2_unit_match: pass a UnitMatchPlan OR the explicit "
                "(session_group_owner, session_group_name, matcher_params_name, "
                "curation_choices) form -- not both. The plan already carries "
                "the group, matcher, and pinned curations."
            )
        is_input_plan = isinstance(plan, UnitMatchInputPlan)
        if not plan.ok:
            raise PipelineInputError(
                "run_v2_unit_match: the plan could not pin curations or validate prerequisites for "
                f"every {'sort' if is_input_plan else 'member'} "
                f"(curation_strategy={plan.curation_strategy!r}):\n  - "
                + "\n  - ".join(plan.errors)
            )
        matcher_params_name = plan.matcher_params_name
        if is_input_plan:
            input_curations = plan.curations
        else:
            session_group_owner = plan.session_group_owner
            session_group_name = plan.session_group_name
            curation_choices = plan.curation_choices
        pins = input_curations if is_input_plan else curation_choices.values()
        if any(pin.get("curation_uuid") is None for pin in pins):
            raise PipelineInputError(
                "run_v2_unit_match: the plan lacks a curation_uuid generation "
                "pin. Rebuild it with plan_v2_unit_match or "
                "plan_v2_unit_match_from_sorts before matching."
            )
    elif session_group_owner is None or session_group_name is None:
        raise PipelineInputError(
            "run_v2_unit_match: session_group_owner and session_group_name "
            "are required in the explicit form (or pass a UnitMatchPlan "
            "from plan_v2_unit_match)."
        )

    if input_curations is None and curation_choices is None:
        raise PipelineInputError(
            "run_v2_unit_match requires either a UnitMatchPlan (from "
            "plan_v2_unit_match) or explicit curation_choices "
            "(member_index -> {'sorting_id': ..., 'curation_id': ...}); it never "
            "auto-picks a curation. List each member's options with "
            "describe_unit_match_choices(session_group_owner, "
            "session_group_name)."
        )

    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    if not (MatcherParameters & {"matcher_params_name": matcher_params_name}):
        raise PipelineInputError(
            "run_v2_unit_match: unknown matcher_params_name "
            f"{matcher_params_name!r}. Seed defaults with "
            "MatcherParameters.insert_default() (ships 'unitmatch_default'), or "
            "register a custom matcher row first."
        )

    if plan is not None:
        report = preflight_v2_unit_match(plan)
        if not report.ok:
            raise PreflightError(report.summary())
        input_unit_selections = list(report.unit_selections)
        expected_input_set_hash = report.input_set_hash
        preflight_warnings = list(report.warnings)
    else:
        from spyglass.spikesorting.v2.analysis_selection import (
            resolve_unit_selection,
        )

        input_unit_selections = [
            resolve_unit_selection(
                pin, policy=unit_policy or "v2_accepted_single_units"
            )
            for pin in curation_choices.values()
        ]
        preflight_warnings = [
            warning
            for selection in input_unit_selections
            if (warning := unlabeled_units_warning(selection.snapshot()))
        ]

    # Partial receipt for PipelineStageError.partial_run_summary.
    run_summary: dict[str, Any] = {
        "session_group_owner": session_group_owner,
        "session_group_name": session_group_name,
        "matcher_params_name": matcher_params_name,
    }
    stage_seconds: dict[str, float] = {}

    # Validates inputs and geometry before any populate.
    if input_curations is not None:
        selection = UnitMatchSelection.insert_inputs(
            input_curations,
            matcher_params_name,
            unit_selections=input_unit_selections,
            expected_input_set_hash=expected_input_set_hash,
        )
    else:
        selection = UnitMatchSelection.insert_selection(
            session_group_owner,
            session_group_name,
            matcher_params_name,
            curation_choices,
            unit_selections=input_unit_selections,
            expected_input_set_hash=expected_input_set_hash,
        )
    # Receipts spell it ``unit_match_id``; the table PK is ``unitmatch_id``.
    run_summary["unit_match_id"] = selection["unitmatch_id"]

    # Recomputed so a reused selection still reports the divergence.
    warnings: list[str] = list(preflight_warnings)
    divergent = UnitMatchSelection._divergent_electrode_space_members(
        UnitMatchSelection.pinned_curations(selection)
    )
    if divergent:
        warnings.append(
            UnitMatchSelection._divergent_electrode_space_message(divergent)
        )
    # Set before the stages so a failure snapshot carries it.
    run_summary["warnings"] = warnings

    (
        _,
        run_summary["unit_match_status"],
        stage_seconds["unit_match"],
    ) = _run_stage(
        "unit_match",
        bool(UnitMatch & selection),
        lambda: _populate_once(UnitMatch, selection),
        run_summary,
    )
    run_summary["n_pairs"] = int((UnitMatch & selection).fetch1("n_pairs"))
    run_summary["inputs"] = _unit_match_input_summaries(selection)

    # Separate stage so a tracked-unit failure (e.g. node budget) is distinct.
    (
        _,
        run_summary["tracked_unit_status"],
        stage_seconds["tracked_unit"],
    ) = _run_stage(
        "tracked_unit",
        bool(TrackedUnit & selection),
        lambda: _populate_once(TrackedUnit, selection),
        run_summary,
    )
    run_summary["n_tracked_units"] = len(TrackedUnit & selection)

    return RunV2UnitMatchSummary(
        session_group_owner=session_group_owner,
        session_group_name=session_group_name,
        matcher_params_name=matcher_params_name,
        unit_match_id=run_summary["unit_match_id"],
        inputs=run_summary["inputs"],
        unit_match_status=run_summary["unit_match_status"],
        n_pairs=run_summary["n_pairs"],
        tracked_unit_status=run_summary["tracked_unit_status"],
        n_tracked_units=run_summary["n_tracked_units"],
        stage_seconds=UnitMatchStageSeconds(
            unit_match=stage_seconds["unit_match"],
            tracked_unit=stage_seconds["tracked_unit"],
        ),
        warnings=warnings,
    )


def _unit_match_input_summaries(
    selection: dict,
) -> tuple[UnitMatchInputSummary, ...]:
    """The receipt's per-input records for one populated match run.

    Reads the frozen selection parts only; never opens the analysis NWB.
    """
    from spyglass.spikesorting.v2.unit_matching import UnitMatch

    inputs, recordings = UnitMatch().get_input_provenance(selection)
    summaries = []
    for row in inputs.to_dict(orient="records"):
        input_index = int(row["input_index"])
        members = recordings[recordings["input_index"] == input_index]
        summaries.append(
            UnitMatchInputSummary(
                input_index=input_index,
                sorting_id=row["sorting_id"],
                curation_id=int(row["curation_id"]),
                curation_uuid=row["curation_uuid"],
                source_kind=row["source_kind"],
                source_id=row["source_id"],
                nwb_file_names=tuple(members["nwb_file_name"]),
                interval_list_names=tuple(members["interval_list_name"]),
                n_recordings=len(members),
                motion_corrected_recording_id=row[
                    "motion_corrected_recording_id"
                ],
                waveform_traces=str(row["waveform_traces"]),
            )
        )
    return tuple(summaries)


def plan_v2_unit_match(
    session_group_owner: str,
    session_group_name: str,
    *,
    curation_strategy: str,
    matcher_params_name: str = "unitmatch_default",
    manual_curation_choices: "dict | None" = None,
    unit_policy: str = "v2_accepted_single_units",
) -> "UnitMatchPlan":
    """Build a reviewable plan pinning one curation per member for matching.

    Applies ``curation_strategy`` to each member's committed curations;
    inspect the plan before the expensive match::

        plan = plan_v2_unit_match(
            owner, name, curation_strategy="final_curated"
        )
        display(plan.as_dataframe())   # one row per member
        summary = run_v2_unit_match(plan)

    Parameters
    ----------
    session_group_owner, session_group_name : str
        Identify the ``SessionGroup`` whose members are matched.
    curation_strategy : str
        Required, so a match never silently pins a "latest" curation:

        - ``"final_curated"``: each member's single terminal non-root
          curation; zero or several is a plan error.
        - ``"auto_curated"``: each member's auto-curated child.
        - ``"root"``: each member's root curation (uncurated; warns).
        - ``"manual"``: pin ``manual_curation_choices``.
    matcher_params_name : str, optional
        ``MatcherParameters`` row carried onto the plan (default
        ``"unitmatch_default"``).
    manual_curation_choices : dict, optional
        Only for ``curation_strategy="manual"``:
        ``{member_index: {"sorting_id": ..., "curation_id": ...}}``.
    unit_policy : str, optional
        Named unit eligibility policy; it must exclude ``mua``. Default
        ``"v2_accepted_single_units"`` matches reviewed single units; choose
        ``"v2_unflagged_single_units"`` for automatic-only output, whose
        included unlabeled units are reported.

    Returns
    -------
    UnitMatchPlan
        Carries ``curation_choices`` (the pins), ``warnings`` / ``errors``, and
        ``as_dataframe()``. ``plan.ok`` is ``False`` if any member is
        unresolved; ``run_v2_unit_match(plan)`` then raises with the errors.
    """
    members = _unit_match_member_choices(
        session_group_owner, session_group_name
    )
    plan = build_unit_match_plan(
        session_group_owner=session_group_owner,
        session_group_name=session_group_name,
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        members=members,
        manual_curation_choices=manual_curation_choices,
    )
    return _preflight_match_plan(plan, unit_policy=unit_policy)


def plan_v2_unit_match_from_sorts(
    sorting_ids,
    *,
    curation_strategy: str,
    matcher_params_name: str = "unitmatch_default",
    manual_curation_choices: "dict | None" = None,
    unit_policy: str = "v2_accepted_single_units",
) -> "UnitMatchInputPlan":
    """Build a reviewable plan pinning one curation per named sort.

    Group-less counterpart of :func:`plan_v2_unit_match`; each sort (of a
    recording or a same-day concatenation) is one matching input::

        plan = plan_v2_unit_match_from_sorts(
            [day1_sorting_id, day2_sorting_id],
            curation_strategy="final_curated",
        )
        display(plan.as_dataframe())   # one row per matching input
        summary = run_v2_unit_match(plan)

    The read-only preflight checks coverage, sample overlap, ordering,
    geometry, and matcher availability. Disjoint epochs in one NWB can be
    matched; repeated original samples cannot.

    Parameters
    ----------
    sorting_ids : sequence of str or uuid.UUID
        The ``sorting_id`` of each sort to match, each named once.
    curation_strategy : str
        Required; as in :func:`plan_v2_unit_match`, per sort.
    matcher_params_name : str, optional
        ``MatcherParameters`` row carried onto the plan (default
        ``"unitmatch_default"``).
    manual_curation_choices : dict, optional
        Only for ``curation_strategy="manual"``: ``{sorting_id:
        curation_id}`` for every named sort.
    unit_policy : str, optional
        Named unit eligibility policy; it must exclude ``mua``. Default
        ``"v2_accepted_single_units"`` matches reviewed single units; choose
        ``"v2_unflagged_single_units"`` for automatic-only output.

    Returns
    -------
    UnitMatchInputPlan
        Carries ``curations`` (the pins), ``warnings`` / ``errors``, and
        ``as_dataframe()``. ``plan.ok`` is ``False`` if any sort is
        unresolved; ``run_v2_unit_match(plan)`` then raises with the errors.

    Raises
    ------
    PipelineInputError
        If a ``sorting_id`` is not a UUID or names no ``SortingSelection``.
    ValueError
        From the planner on no sorts, a sort named twice, or an invalid
        strategy / manual pin.
    """
    plan = build_unit_match_input_plan(
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        sorts=_unit_match_sort_choices(sorting_ids),
        manual_curation_choices=manual_curation_choices,
    )
    return _preflight_match_plan(plan, unit_policy=unit_policy)


def plan_v2_unit_match_from_curations(
    curations: "list[CurationRef]",
    *,
    unit_policy: str | None = None,
    matcher_params_name: str = "unitmatch_default",
    unit_selections: "list[UnitSelectionSnapshot] | None" = None,
) -> UnitMatchInputPlan:
    """Plan matching directly from the exact curations a reviewer selected.

    Each curation pins its immutable generation, and a named eligibility
    policy freezes the included units.

    - ``unit_policy``: default ``'v2_accepted_single_units'``; auto-only
      workflows choose ``'v2_unflagged_single_units'``. Must exclude ``mua``.
    - ``unit_selections``: optional, one :class:`UnitSelectionSnapshot` per
      curation, each pinning that curation. Their policies apply when
      ``unit_policy`` is omitted; an explicit ``unit_policy`` must match.

    The preflight is read-only; ``run_v2_unit_match(plan)`` revalidates before
    any write.
    """
    refs = list(curations)
    if not refs:
        raise ValueError(
            "plan_v2_unit_match_from_curations needs at least one curation."
        )
    if any(not isinstance(ref, CurationRef) for ref in refs):
        raise TypeError(
            "plan_v2_unit_match_from_curations requires CurationRef objects. "
            "Resolve an explicit key with CurationRef.from_key first."
        )
    repeated = len({ref.sorting_id for ref in refs}) != len(refs)
    if repeated:
        raise ValueError(
            "Each sorting may be matched through one curation generation only; "
            "choose one CurationRef per sorting."
        )
    selections = ()
    if unit_selections is not None:
        selections = tuple(unit_selections)
        if len(selections) != len(refs):
            raise ValueError("Supply exactly one unit selection per curation.")
        for ref, selection in zip(refs, selections, strict=True):
            if not isinstance(selection, UnitSelectionSnapshot):
                raise TypeError(
                    "unit_selections must contain UnitSelectionSnapshot objects."
                )
            if selection.curation != ref:
                raise ValueError(
                    "A unit selection pins a different curation generation."
                )
        policy_names = {selection.policy_name for selection in selections}
        if unit_policy is not None and policy_names != {unit_policy}:
            raise ValueError(
                "unit_policy conflicts with the supplied selection policies."
            )
        unit_policy = (
            next(iter(policy_names))
            if len(policy_names) == 1
            else "provided selections"
        )
    elif unit_policy is None:
        unit_policy = "v2_accepted_single_units"
    # Copy the pin so a stale ref becomes a preflight finding, not a repin.
    pins = [
        {
            "sorting_id": str(ref.sorting_id),
            "curation_id": ref.curation_id,
            "curation_uuid": str(ref.curation_uuid),
        }
        for ref in refs
    ]
    plan = UnitMatchInputPlan(
        matcher_params_name=matcher_params_name,
        curation_strategy="explicit_curations",
        curations=pins,
        rows=[{**pin, "status": "pinned"} for pin in pins],
        unit_selections=selections,
    )
    return _preflight_match_plan(plan, unit_policy=unit_policy)


def _preflight_match_plan(plan, *, unit_policy: str):
    """Freeze eligibility and attach the canonical read-only prerequisites."""
    plan.unit_policy = unit_policy
    if not plan.ok:
        return plan
    report = preflight_v2_unit_match(plan)
    plan.preflight = report
    plan.unit_selections = report.unit_selections
    plan.input_set_hash = report.input_set_hash
    plan.errors.extend(report.errors)
    plan.warnings.extend(
        value for value in report.warnings if value not in plan.warnings
    )
    selections = {}
    # preflight_v2_unit_match keeps only typed UnitSelectionSnapshot objects.
    for selection in report.unit_selections:
        snapshot = selection.snapshot()
        selections[
            (str(snapshot["sorting_id"]), int(snapshot["curation_id"]))
        ] = snapshot
    resolved = {
        (str(item["sorting_id"]), int(item["curation_id"])): item
        for item in (report._prepared or {}).get("inputs", [])
    }
    pins = plan.pinned_curations
    requested_indices = {
        (str(pin["sorting_id"]), int(pin["curation_id"])): index
        for index, pin in enumerate(pins)
    }
    for row in plan.rows:
        identity = (str(row["sorting_id"]), int(row["curation_id"]))
        snapshot = selections.get(identity)
        item = resolved.get(identity)
        input_index = (
            item["input_index"]
            if item is not None
            else requested_indices[identity]
        )
        row.update(
            unit_policy=(
                snapshot["policy_name"] if snapshot is not None else unit_policy
            ),
            preflight_status="ready" if report.ok else "blocked",
            checks=", ".join(
                f"{check.name}: {check.status}"
                for check in report.checks
                if check.input_index is None or check.input_index == input_index
            ),
        )
        if snapshot is not None:
            row.update(
                included_unit_ids=tuple(snapshot["included_unit_ids"]),
                excluded_units=dict(snapshot["excluded_units"]),
                n_included_units=len(snapshot["included_unit_ids"]),
                n_excluded_units=len(snapshot["excluded_units"]),
                selection_hash=snapshot["selection_hash"],
            )
        if item is not None:
            recordings = item["recordings"]
            row.update(
                input_index=int(item["input_index"]),
                source_kind=item["source_kind"],
                source_id=str(item["source_id"]),
                nwb_file_names=tuple(
                    value["nwb_file_name"] for value in recordings
                ),
                interval_list_names=tuple(
                    value["interval_list_name"] for value in recordings
                ),
                motion_corrected_recording_id=item.get(
                    "motion_corrected_recording_id"
                ),
                waveform_traces=(
                    "motion_corrected_recording"
                    if item.get("motion_corrected_recording_id")
                    else item["source_kind"]
                ),
                source_coverage=tuple(
                    deepcopy(value["source_coverage"]) for value in recordings
                ),
            )
    if isinstance(plan, UnitMatchInputPlan) and report.ok:
        plan.rows.sort(key=lambda row: row["input_index"])
    return plan


def _unit_match_sort_choices(sorting_ids) -> list[dict]:
    """Each named sort's source, constituent recordings and curations.

    In the order named. Curations are not filtered for match validity;
    ``UnitMatchSelection.insert_inputs`` validates.

    Raises
    ------
    PipelineInputError
        If a ``sorting_id`` is not a UUID or names no ``SortingSelection``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sorts = []
    for sorting_id in sorting_ids:
        try:
            key = {"sorting_id": uuid.UUID(str(sorting_id))}
        except ValueError as exc:
            raise PipelineInputError(
                f"unit-match sorts: sorting_id {sorting_id!r} is not a UUID."
            ) from exc
        if not (SortingSelection & key):
            raise PipelineInputError(
                f"unit-match sorts: no SortingSelection for sorting_id "
                f"{key['sorting_id']}; sort it first (run_v2_pipeline)."
            )
        source = SortingSelection.resolve_source(key)
        if source.kind == "recording":
            source_id = source.key["recording_id"]
            recordings = (RecordingSelection & source.key).fetch(
                "nwb_file_name", "interval_list_name", as_dict=True
            )
        else:
            source_id = source.key["concat_recording_id"]
            recordings = (
                ConcatenatedRecordingSelection.MemberSnapshot & source.key
            ).fetch(
                "nwb_file_name",
                "interval_list_name",
                as_dict=True,
                order_by="member_index",
            )
        sorts.append(
            {
                "sorting_id": str(key["sorting_id"]),
                "source_kind": source.kind,
                "source_id": str(source_id),
                "nwb_file_names": tuple(
                    row["nwb_file_name"] for row in recordings
                ),
                "interval_list_names": tuple(
                    row["interval_list_name"] for row in recordings
                ),
                "choices": (CurationV2 & key).fetch(
                    "sorting_id",
                    "curation_id",
                    "curation_uuid",
                    "parent_curation_id",
                    "curation_source",
                    "description",
                    as_dict=True,
                    order_by="curation_id",
                ),
            }
        )
    return sorts


def _unit_match_member_choices(
    session_group_owner: str,
    session_group_name: str,
) -> "list[UnitMatchMemberChoices]":
    """Structured per-member curation choices (the DB-fetching core).

    Backs :func:`describe_unit_match_choices` and :func:`plan_v2_unit_match`.
    A member with no sort yet has empty ``choices``. Curations are not
    filtered for match validity; ``UnitMatchSelection.insert_selection``
    validates.

    Parameters
    ----------
    session_group_owner, session_group_name : str
        Identify the ``SessionGroup``.

    Returns
    -------
    list of UnitMatchMemberChoices
        One entry per member (``member_index`` order): the member identity plus
        a ``choices`` list of ``{sorting_id, curation_id, parent_curation_id,
        curation_source, description}`` (``parent_curation_id == -1`` marks a
        root curation; ``curation_source`` is the CurationV2 enum, e.g.
        ``'curation_evaluation'`` for an auto-curated child).

    Raises
    ------
    PipelineInputError
        If the ``SessionGroup`` has no members.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import SortingSelection

    group_key = {
        "session_group_owner": session_group_owner,
        "session_group_name": session_group_name,
    }
    members = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    if len(members) == 0:
        raise PipelineInputError(
            "unit-match choices: no SessionGroup.Member rows for "
            f"{group_key}. Create the group first via "
            "SessionGroup.create_group()."
        )

    out: list[dict] = []
    for member in members:
        member_id = {
            "nwb_file_name": member["nwb_file_name"],
            "sort_group_id": int(member["sort_group_id"]),
            "interval_list_name": member["interval_list_name"],
            "team_name": member["team_name"],
        }
        # Include team_name: the ownership validator rejects other teams' sorts.
        member_sortings = SortingSelection.RecordingSource * (
            RecordingSelection & member_id
        )
        choices = (CurationV2 & member_sortings).fetch(
            "sorting_id",
            "curation_id",
            "curation_uuid",
            "parent_curation_id",
            "curation_source",
            "description",
            as_dict=True,
            order_by="sorting_id, curation_id",
        )
        out.append(
            {
                "member_index": int(member["member_index"]),
                **member_id,
                "choices": choices,
            }
        )
    return cast("list[UnitMatchMemberChoices]", out)


def describe_unit_match_choices(
    session_group_owner: str,
    session_group_name: str,
) -> "pd.DataFrame":
    """Table of the curations you can pin per SessionGroup member.

    One row per (member, curation); ``parent_curation_id == -1`` marks a root.
    A member with no sort yet appears with null curation columns. Pass one
    ``{sorting_id, curation_id}`` per member as ``curation_choices``, or use
    :func:`plan_v2_unit_match`. Read-only.

    Raises
    ------
    PipelineInputError
        If the ``SessionGroup`` has no members.
    """
    return member_choices_to_dataframe(
        _unit_match_member_choices(session_group_owner, session_group_name)
    )
