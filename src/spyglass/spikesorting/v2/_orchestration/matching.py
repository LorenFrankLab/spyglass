"""Plan, inspect, and run cross-session unit matching.

Owns curation-choice discovery and plan/run orchestration. Matcher computation
and tracked-unit derivation remain in their domain table and pure helpers."""

from __future__ import annotations

import uuid
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

if TYPE_CHECKING:
    import pandas as pd


def run_v2_unit_match(
    plan: "UnitMatchPlan | UnitMatchInputPlan | None" = None,
    *,
    session_group_owner: "str | None" = None,
    session_group_name: "str | None" = None,
    matcher_params_name: str = "unitmatch_default",
    curation_choices: "dict | None" = None,
) -> RunV2UnitMatchSummary:
    """Match units across matching inputs in one call, then track them.

    A matching input is one curated sort, of a single recording or of a
    same-day concatenation. Three call forms:

    - ``run_v2_unit_match(plan)`` with a :class:`UnitMatchPlan` from
      :func:`plan_v2_unit_match` (the recommended plan-then-run path for a
      ``SessionGroup``; the plan already pinned one curation per member via a
      curation strategy).
    - ``run_v2_unit_match(plan)`` with a :class:`UnitMatchInputPlan` from
      :func:`plan_v2_unit_match_from_sorts` (named sorts, no group; the plan
      pinned one curation per sort). The pinned curations go to
      ``UnitMatchSelection.insert_inputs``.
    - ``run_v2_unit_match(session_group_owner=..., session_group_name=...,
      matcher_params_name=..., curation_choices={...})`` -- the explicit
      group form, for power users.

    A plan that could not pin every curation (``plan.ok is False``) raises
    here. Chains the cross-session unit-matching stages into one call: the
    selection (``UnitMatchSelection.insert_selection`` for a group,
    ``insert_inputs`` for named sorts) -> ``UnitMatch`` (pairwise matches) ->
    ``TrackedUnit`` (biological-unit identity across sessions). Idempotent:
    re-running with the same inputs reuses the existing rows.

    In the explicit form ``curation_choices`` is REQUIRED -- this helper never
    auto-picks a "latest" curation (which would make the match irreproducible
    the moment a source session gains a new curation). Build it from
    :func:`describe_unit_match_choices`, or use the plan form (recommended):
    :func:`plan_v2_unit_match` pins the curations for you by a named curation
    strategy and returns the ``UnitMatchPlan`` you pass here.

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
        ``{member_index: {"sorting_id": ..., "curation_id": ...}}`` -- exactly
        one committed curation pinned per member. ``None`` (the default) raises.

    Returns
    -------
    RunV2UnitMatchSummary
        ``session_group_owner`` / ``session_group_name`` (the group matched;
        ``None`` for a plan of named sorts) / ``matcher_params_name``,
        the ``unit_match_id`` selection PK, ``inputs`` (one
        :class:`UnitMatchInputSummary` per matching input, in chronological
        ``input_index`` order, read from the frozen selection), ``n_pairs`` (pairwise matches) and
        ``n_tracked_units`` (cross-session biological units), the per-stage
        ``unit_match_status`` / ``tracked_unit_status`` (``"computed"`` /
        ``"reused"`` -- stems match the ``stage_seconds`` keys so ``describe_run``
        fills the receipt), ``stage_seconds`` (keys ``unit_match`` /
        ``tracked_unit``), and ``warnings`` (advisory notes, e.g. an
        electrode-space divergence across members; also logged).

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
        set, e.g. a curation with unapplied merges, two inputs sharing a
        session (``SameSessionMatchError``), a multi-day concatenation input,
        or a channel-geometry mismatch across inputs.
    PipelineStageError
        If ``UnitMatch`` / ``TrackedUnit`` populate fails (names the stage,
        carries the partial summary). A missing optional matcher backend (e.g.
        ``UnitMatchPy``) surfaces here with the backend's install hint.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    # Accept a plan (from plan_v2_unit_match or plan_v2_unit_match_from_sorts)
    # OR the explicit keyword form. Unwrap a plan into the explicit inputs; a
    # plan that could not pin every curation (not ok) raises here rather than
    # running a partial match.
    input_curations = None
    if plan is not None:
        if not isinstance(plan, (UnitMatchPlan, UnitMatchInputPlan)):
            raise PipelineInputError(
                "run_v2_unit_match: plan must be a UnitMatchPlan from "
                "plan_v2_unit_match or a UnitMatchInputPlan from "
                "plan_v2_unit_match_from_sorts, or omit plan and pass "
                "session_group_owner= and session_group_name=."
            )
        # A plan already carries the group + matcher + choices, so the explicit
        # args must not ALSO be given -- otherwise one would be silently
        # overridden. (matcher_params_name at its default is treated as unset.)
        if (
            session_group_owner is not None
            or session_group_name is not None
            or curation_choices is not None
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
                "run_v2_unit_match: the plan could not pin a curation for "
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

    # curation_choices is REQUIRED and explicit: an implicit "latest curation"
    # lookup would make the match irreproducible the moment a source session
    # gains a new curation. Validate DB-free, before any table import.
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

    # Partial receipt accumulated for error reporting: _run_stage snapshots
    # it into PipelineStageError.partial_run_summary when a stage fails. The
    # complete RunV2UnitMatchSummary is assembled explicitly at the end.
    run_summary: dict[str, Any] = {
        "session_group_owner": session_group_owner,
        "session_group_name": session_group_name,
        "matcher_params_name": matcher_params_name,
    }
    stage_seconds: dict[str, float] = {}

    # The selection pins one curation per input and validates the inputs /
    # geometry BEFORE any populate (for a group, insert_selection also checks
    # coverage and per-member ownership), so a bad pin raises here, not deep
    # in the matcher.
    if input_curations is not None:
        selection = UnitMatchSelection.insert_inputs(
            input_curations, matcher_params_name
        )
    else:
        selection = UnitMatchSelection.insert_selection(
            session_group_owner,
            session_group_name,
            matcher_params_name,
            curation_choices,
        )
    # Public orchestration receipts use the clearer ``unit_match_id`` spelling;
    # the DataJoint table PK remains ``unitmatch_id`` for schema stability.
    run_summary["unit_match_id"] = selection["unitmatch_id"]

    # Surface the advisory electrode-space divergence in the receipt (it is also
    # logged inside insert_selection / make_fetch). Recomputed here from the
    # selection's pinned inputs so the summary carries it even on a reused
    # selection.
    warnings: list[str] = []
    divergent = UnitMatchSelection._divergent_electrode_space_members(
        UnitMatchSelection.pinned_curations(selection)
    )
    if divergent:
        warnings.append(
            UnitMatchSelection._divergent_electrode_space_message(divergent)
        )
    # Record now (not after the stages) so a PipelineStageError snapshot from a
    # failing stage still carries the advisory -- it is known pre-stage.
    run_summary["warnings"] = warnings

    # Pairwise cross-session match.
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

    # Biological-unit identity across sessions, derived from the pair graph with
    # the same matcher params. Its own stage so a tracked-unit failure (e.g. the
    # strict-node budget) is reported distinctly from the match itself.
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

    Everything comes from the frozen selection parts through the database
    form of ``UnitMatch.get_input_provenance`` (which derives
    ``waveform_traces`` from the frozen ``motion_corrected_recording_id``),
    so building the receipt never opens the run's analysis NWB.
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
) -> "UnitMatchPlan":
    """Build a reviewable plan pinning one curation per member for matching.

    The recommended plan-then-run entry point for cross-session matching: it
    lists each member's committed curations (via
    :func:`describe_unit_match_choices`) and applies the named
    ``curation_strategy`` to pin exactly one per member, returning a
    ``UnitMatchPlan`` you inspect BEFORE the expensive match::

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
        REQUIRED (no default -- your declaration of intent, so a match never
        silently pins a "latest" curation):

        - ``"final_curated"`` -- each member's single terminal curated
          (non-root) curation; a member with zero or several is a plan error.
        - ``"auto_curated"`` -- each member's auto-curated child (sort members
          with ``run_v2_pipeline(auto_curate=True)`` first).
        - ``"root"`` -- each member's root curation (UNCURATED; warns loudly).
        - ``"manual"`` -- pin ``manual_curation_choices`` explicitly (required
          with this curation strategy), validated against each member's committed
          curations.
    matcher_params_name : str, optional
        ``MatcherParameters`` row carried onto the plan (default
        ``"unitmatch_default"``).
    manual_curation_choices : dict, optional
        Only for ``curation_strategy="manual"``:
        ``{member_index: {"sorting_id": ..., "curation_id": ...}}``.

    Returns
    -------
    UnitMatchPlan
        Carries ``curation_choices`` (the pins), ``warnings`` / ``errors``, and
        ``as_dataframe()``. ``plan.ok`` is ``False`` if any member is
        unresolved; ``run_v2_unit_match(plan)`` then raises with the errors.
    """
    from spyglass.spikesorting.v2._matching.planning import (
        build_unit_match_plan,
    )

    members = _unit_match_member_choices(
        session_group_owner, session_group_name
    )
    return build_unit_match_plan(
        session_group_owner=session_group_owner,
        session_group_name=session_group_name,
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        members=members,
        manual_curation_choices=manual_curation_choices,
    )


def plan_v2_unit_match_from_sorts(
    sorting_ids,
    *,
    curation_strategy: str,
    matcher_params_name: str = "unitmatch_default",
    manual_curation_choices: "dict | None" = None,
) -> "UnitMatchInputPlan":
    """Build a reviewable plan pinning one curation per named sort.

    The group-less counterpart of :func:`plan_v2_unit_match`: each named
    sort -- of a single recording or of a same-day concatenation -- is one
    matching input, so daily concatenation sorts can be matched directly::

        plan = plan_v2_unit_match_from_sorts(
            [day1_sorting_id, day2_sorting_id],
            curation_strategy="final_curated",
        )
        display(plan.as_dataframe())   # one row per matching input
        summary = run_v2_unit_match(plan)

    The plan lists each sort's committed curations and applies
    ``curation_strategy`` within them. It does not check that the sorts can
    be matched together; ``UnitMatchSelection.insert_inputs`` (run by
    ``run_v2_unit_match``) validates that (no shared session, a
    concatenation within one day, shared channel geometry) and orders the
    inputs chronologically.

    Parameters
    ----------
    sorting_ids : sequence of str or uuid.UUID
        The ``sorting_id`` of each sort to match, each named once.
    curation_strategy : str
        REQUIRED; the same strategies as :func:`plan_v2_unit_match`, applied
        to each sort's own curations: ``"final_curated"``,
        ``"auto_curated"``, ``"root"`` (UNCURATED; warns loudly) or
        ``"manual"``.
    matcher_params_name : str, optional
        ``MatcherParameters`` row carried onto the plan (default
        ``"unitmatch_default"``).
    manual_curation_choices : dict, optional
        Only for ``curation_strategy="manual"``: ``{sorting_id:
        curation_id}`` for every named sort.

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
    from spyglass.spikesorting.v2._matching.planning import (
        build_unit_match_input_plan,
    )

    return build_unit_match_input_plan(
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        sorts=_unit_match_sort_choices(sorting_ids),
        manual_curation_choices=manual_curation_choices,
    )


def _unit_match_sort_choices(sorting_ids) -> list[dict]:
    """Each named sort's source, constituent recordings and curations.

    The DB-fetching core of :func:`plan_v2_unit_match_from_sorts`. For each
    sort, in the order named: its ``SortingSelection`` source kind and id,
    the ``nwb_file_name`` / ``interval_list_name`` of its constituent
    recordings in recording order (the recording itself, or the
    concatenation's frozen members), and every committed ``CurationV2`` row
    of the sort. The curations are not pre-filtered for match-validity;
    ``UnitMatchSelection.insert_inputs`` is the validation boundary.

    Raises
    ------
    PipelineInputError
        If a ``sorting_id`` is not a UUID or names no ``SortingSelection``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
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

    Backs both :func:`describe_unit_match_choices` (the DataFrame view) and
    :func:`plan_v2_unit_match`. For each member (ordered by ``member_index``) it walks
    the member's recordings -> sortings -> committed ``CurationV2`` rows and
    returns every curation you may pin, so you never hand-query the join or rely
    on an implicit "latest". Pick one entry per member and build
    ``curation_choices = {member_index: {"sorting_id": ..., "curation_id": ...}}``.

    A member with no sort/curation yet returns an empty ``choices`` list (sort it
    first). The returned curations are not pre-filtered for match-validity;
    ``run_v2_unit_match`` / ``UnitMatchSelection.insert_selection`` is the
    validation boundary (it rejects e.g. a preview curation or a wrong-member
    pin).

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
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
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
        # Match the member's recordings on the FULL member identity, INCLUDING
        # team_name. UnitMatchSelection's ownership validator compares the whole
        # (nwb_file_name, sort_group_id, interval_list_name, team_name) tuple, so
        # a curation sorted under a different team tag would be rejected at run
        # time -- surfacing it here would offer an un-pickable choice. One
        # relational read: every single-session sort of any of the member's
        # recordings, then that sort set's curations (empty when the member
        # has no recording or no sort yet).
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

    Read-only discovery helper (the ``describe_*`` family -- returns a
    ``DataFrame``). One row per (member, pinnable curation): the member identity
    (``member_index`` / ``nwb_file_name`` / ``sort_group_id`` /
    ``interval_list_name`` / ``team_name``) plus the curation fields
    (``sorting_id`` / ``curation_id`` / ``parent_curation_id`` /
    ``curation_source`` / ``description``; ``parent_curation_id == -1`` marks a
    root). A member with no sort/curation yet still appears as one row with null
    curation columns, so it is visibly "sort me first".

    Pick one curation per member and pass its ``{sorting_id, curation_id}`` as
    ``run_v2_unit_match``'s ``curation_choices``; or, recommended, declare intent
    with :func:`plan_v2_unit_match` and inspect the resulting plan first. The
    returned curations are not pre-filtered for match-validity;
    ``UnitMatchSelection.insert_selection`` is the validation boundary.

    Raises
    ------
    PipelineInputError
        If the ``SessionGroup`` has no members.
    """
    from spyglass.spikesorting.v2._matching.planning import (
        member_choices_to_dataframe,
    )

    return member_choices_to_dataframe(
        _unit_match_member_choices(session_group_owner, session_group_name)
    )
