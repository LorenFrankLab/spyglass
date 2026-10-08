"""DB-free curation-choice planning for cross-session unit matching.

``run_v2_unit_match`` requires every matching input to pin exactly one
committed curation, so a match never silently depends on an implicit "latest"
curation. Hand-building those pins is error-prone, so this module turns the
per-candidate curation lists into pins via a named, intent-declaring curation
strategy and returns a plan (with warnings / errors) to inspect BEFORE the
expensive match:

- :func:`build_unit_match_plan` -- one pin per ``SessionGroup`` member (the
  structured form ``describe_unit_match_choices`` tabulates), returning a
  :class:`UnitMatchPlan`;
- :func:`build_unit_match_input_plan` -- one pin per named sort (a sort of a
  single recording or of a same-day concatenation), returning a
  :class:`UnitMatchInputPlan` with one row per matching input.

This module is deliberately DB-free: it operates on already-fetched curation
lists, so the curation-strategy logic is unit-tested without a database. The
DataJoint plumbing (fetching the candidates, running the plan) lives in
the matching orchestration module. The low-level ``UnitMatchSelection`` stays explicit and
pinned; this module is only the layer that assembles the pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from spyglass.spikesorting.v2._core.lookup_validation import lossless_int

if TYPE_CHECKING:
    import pandas as pd

# Every curation strategy is an explicit declaration of intent; there is
# deliberately no default (an implicit "latest" is exactly what pinning
# exists to prevent).
STRATEGIES: tuple[str, ...] = (
    "final_curated",
    "auto_curated",
    "root",
    "manual",
)

_ROOT_PARENT = -1
_AUTO_SOURCE = "curation_evaluation"

# Column order for the ``describe_unit_match_choices`` DataFrame view: the member
# identity first, then the per-choice curation fields.
_MEMBER_COLUMNS = (
    "member_index",
    "nwb_file_name",
    "sort_group_id",
    "interval_list_name",
    "team_name",
)
_CHOICE_COLUMNS = (
    "sorting_id",
    "curation_id",
    "curation_uuid",
    "parent_curation_id",
    "curation_source",
    "description",
)
# Column order for ``UnitMatchInputPlan.as_dataframe``: the sort and its
# constituent recordings (in recording order), then the pin.
_INPUT_PLAN_COLUMNS = (
    "sorting_id",
    "source_kind",
    "source_id",
    "nwb_file_names",
    "interval_list_names",
    "curation_id",
    "curation_uuid",
    "status",
)


def member_choices_to_dataframe(members: list[dict]) -> "pd.DataFrame":
    """Flatten per-member curation choices to one row per (member, choice).

    The human-facing view behind ``describe_unit_match_choices``: each pinnable
    curation becomes a row carrying the member identity plus the choice fields.
    A member with no sort yet still appears as a single row with null curation
    columns, so it stays visible as "sort me first" rather than vanishing.
    DB-free; ``pandas`` is imported lazily to keep this module import-light.
    """
    import pandas as pd

    rows: list[dict] = []
    for member in members:
        member_id = {col: member[col] for col in _MEMBER_COLUMNS}
        choices = member.get("choices") or []
        if choices:
            rows.extend(
                {**member_id, **{c: choice[c] for c in _CHOICE_COLUMNS}}
                for choice in choices
            )
        else:
            rows.append({**member_id, **{c: None for c in _CHOICE_COLUMNS}})
    df = pd.DataFrame(rows, columns=list(_MEMBER_COLUMNS + _CHOICE_COLUMNS))
    # Keep the id columns as nullable ``Int64`` so a sortless member's ``None``
    # placeholder row does not upcast valid ids to float (a curation_id would
    # otherwise display / copy as ``0.0`` instead of ``0`` -- the exact
    # copy-paste footgun this table exists to prevent).
    for col in (
        "member_index",
        "sort_group_id",
        "curation_id",
        "parent_curation_id",
    ):
        df[col] = df[col].astype("Int64")
    return df


@dataclass
class UnitMatchPlan:
    """A reviewable plan pinning one curation per SessionGroup member.

    ``curation_choices`` maps each member index to ``sorting_id``,
    ``curation_id`` and ``curation_uuid``. The UUID pins the reviewed
    generation, so deleting and recreating a numeric curation ID invalidates
    the plan. ``run_v2_unit_match`` consumes these pins. ``errors`` (blocking)
    is non-empty when
    the curation strategy could not pin exactly one curation for some member;
    ``ok`` is then ``False`` and running the plan raises. ``warnings`` are
    advisory (e.g. the ``root`` curation strategy pins uncurated curations).
    ``as_dataframe()`` renders one row per member for review.
    """

    session_group_owner: str
    session_group_name: str
    matcher_params_name: str
    curation_strategy: str
    curation_choices: dict[int, dict[str, Any]]
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the strategy pinned exactly one curation per member."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def as_dataframe(self):
        """One row per member (``member_index`` order) for notebook review."""
        import pandas as pd

        return pd.DataFrame(
            self.rows,
            columns=[
                "member_index",
                "nwb_file_name",
                "sorting_id",
                "curation_id",
                "curation_uuid",
                "status",
            ],
        )


@dataclass
class UnitMatchInputPlan:
    """A reviewable plan pinning one curation per named sort for matching.

    Each named sort -- of a single recording or of a same-day concatenation
    -- is one matching input. ``curations`` is the list of pinned
    ``{"sorting_id", "curation_id", "curation_uuid"}`` in the order the sorts
    were named; the UUID pins the reviewed generation, even if the numeric
    curation ID is later reused.
    ``run_v2_unit_match`` passes it to ``UnitMatchSelection.insert_inputs``,
    which numbers the inputs chronologically. ``errors`` (blocking) is
    non-empty when the curation strategy could not pin exactly one curation
    for some sort; ``ok`` is then ``False`` and running the plan raises.
    ``warnings`` are advisory. ``as_dataframe()`` renders one row per input.
    """

    matcher_params_name: str
    curation_strategy: str
    curations: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the strategy pinned exactly one curation per sort."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def as_dataframe(self):
        """One row per matching input (in the order named) for review."""
        import pandas as pd

        return pd.DataFrame(self.rows, columns=list(_INPUT_PLAN_COLUMNS))


def _leaf_keys(choices: list[dict]) -> set[tuple]:
    """``(sorting_id, curation_id)`` of curations that are no other's parent.

    A leaf is a terminal curation in its sort's lineage -- nothing was curated
    from it. Computed within each sort (a ``parent_curation_id`` is only unique
    within a ``sorting_id``).
    """
    parent_keys = {(c["sorting_id"], c["parent_curation_id"]) for c in choices}
    return {
        (c["sorting_id"], c["curation_id"])
        for c in choices
        if (c["sorting_id"], c["curation_id"]) not in parent_keys
    }


def _candidates(choices: list[dict], curation_strategy: str) -> list[dict]:
    """The curations a curation strategy considers pinnable for one member."""
    if curation_strategy == "root":
        return [c for c in choices if c["parent_curation_id"] == _ROOT_PARENT]
    if curation_strategy == "auto_curated":
        return [c for c in choices if c["curation_source"] == _AUTO_SOURCE]
    if curation_strategy == "final_curated":
        leaves = _leaf_keys(choices)
        return [
            c
            for c in choices
            if c["parent_curation_id"] != _ROOT_PARENT
            and (c["sorting_id"], c["curation_id"]) in leaves
        ]
    raise AssertionError(
        f"unhandled curation_strategy {curation_strategy!r}"
    )  # pragma: no cover


#: Per-subject fix hints for a candidate list an automatic strategy cannot
#: resolve: ``{subject: {curation_strategy: hint}}``.
_NONE_HINTS = {
    "member": {
        "auto_curated": "sort the member with auto_curate=True",
        "final_curated": "curate the member first (see the Curation how-to)",
        "root": "sort the member first",
    },
    "sort": {
        "auto_curated": "auto-curate the sort first",
        "final_curated": "curate the sort first (see the Curation how-to)",
        "root": "curate the sort first",
    },
}


def _describe_none(curation_strategy: str, subject: str = "member") -> str:
    """The 'no candidate' fix hint for an automatic curation strategy.

    ``subject`` is what is being pinned: a SessionGroup ``"member"`` or a
    named ``"sort"``.
    """
    hint = _NONE_HINTS[subject][curation_strategy]
    if curation_strategy == "auto_curated":
        return (
            "has no auto-curated curation (curation_source="
            f"'curation_evaluation'); {hint}, or pin one explicitly with "
            "curation_strategy='manual'"
        )
    if curation_strategy == "final_curated":
        return (
            f"has no curated (non-root) leaf curation; {hint}, or pin one "
            "explicitly with curation_strategy='manual'"
        )
    return f"has no curation yet; {hint}"  # root


def _pin_one(
    label: str,
    subject: str,
    choices: list[dict],
    curation_strategy: str,
    chosen: dict | None,
    listing_hint: str,
) -> tuple[dict | None, list[str], list[str]]:
    """Pin one curation from a candidate list per the curation strategy.

    Parameters
    ----------
    label : str
        Names the candidate in messages, e.g. ``"member 0 (a.nwb)"``.
    subject : str
        ``"member"`` or ``"sort"``, the noun used in messages.
    choices : list of dict
        The candidate's committed curations.
    curation_strategy : str
        One of :data:`STRATEGIES`.
    chosen : dict or None
        For ``curation_strategy="manual"``: the caller's normalized
        ``{"sorting_id", "curation_id"}`` pin, or ``None`` if none was given.
    listing_hint : str
        Where to list the candidate's curations, quoted when a manual pin is
        not among them.

    Returns
    -------
    tuple
        ``(pinned_or_None, warnings, errors)`` where ``pinned`` is
        ``{"sorting_id", "curation_id", "curation_uuid"}``.
    """
    warnings: list[str] = []

    if curation_strategy == "manual":
        if chosen is None:
            return (
                None,
                warnings,
                [
                    f"{label}: curation_strategy='manual' requires an "
                    "explicit manual_curation_choices entry for this "
                    f"{subject}; none was provided."
                ],
            )
        available = {(c["sorting_id"], c["curation_id"]) for c in choices}
        if (chosen["sorting_id"], chosen["curation_id"]) not in available:
            return (
                None,
                warnings,
                [
                    f"{label}: pinned curation "
                    f"(sorting_id={chosen['sorting_id']}, "
                    f"curation_id={chosen['curation_id']}) is not among this "
                    f"{subject}'s committed curations ({listing_hint})."
                ],
            )
        pick = next(
            choice
            for choice in choices
            if (choice["sorting_id"], choice["curation_id"])
            == (chosen["sorting_id"], chosen["curation_id"])
        )
        return _generation_pin(pick, label, warnings)

    candidates = _candidates(choices, curation_strategy)
    if len(candidates) == 0:
        return (
            None,
            warnings,
            [
                f"{label}: curation_strategy={curation_strategy!r} "
                f"{_describe_none(curation_strategy, subject)}."
            ],
        )
    if len(candidates) > 1:
        ids = sorted((c["sorting_id"], c["curation_id"]) for c in candidates)
        return (
            None,
            warnings,
            [
                f"{label}: curation_strategy={curation_strategy!r} is "
                f"ambiguous -- {len(candidates)} candidate curations {ids}. "
                "Pin one explicitly with curation_strategy='manual'."
            ],
        )
    pick = candidates[0]
    if curation_strategy == "root":
        warnings.append(
            f"{label}: curation_strategy='root' pins the "
            "UNCURATED root curation; matching uncurated units is rarely what "
            "you want -- prefer curation_strategy='final_curated' or "
            "'auto_curated'."
        )
    return _generation_pin(pick, label, warnings)


def _generation_pin(pick, label, warnings):
    """Pin the chosen row's generation; refuse candidate lists without it."""
    if pick.get("curation_uuid") is None:
        return (
            None,
            warnings,
            [
                f"{label}: the chosen curation has no curation_uuid; fetch "
                "its generation and rebuild the plan before matching."
            ],
        )
    return (
        {
            "sorting_id": pick["sorting_id"],
            "curation_id": int(pick["curation_id"]),
            "curation_uuid": str(UUID(str(pick["curation_uuid"]))),
        },
        warnings,
        [],
    )


def _resolve_member(
    member: dict, curation_strategy: str, manual_curation_choices: dict | None
) -> tuple[dict | None, list[str], list[str]]:
    """Pin one curation for a SessionGroup member per the curation strategy.

    Returns ``(pinned_or_None, warnings, errors)`` where ``pinned`` is
    ``{"sorting_id", "curation_id", "curation_uuid"}``.
    """
    idx = member["member_index"]
    chosen = None
    if curation_strategy == "manual":
        raw = (manual_curation_choices or {}).get(idx)
        if raw is not None:
            chosen = {
                "sorting_id": raw["sorting_id"],
                "curation_id": lossless_int(
                    raw["curation_id"], f"member {idx} curation_id"
                ),
            }
    return _pin_one(
        f"member {idx} ({member['nwb_file_name']})",
        "member",
        member["choices"],
        curation_strategy,
        chosen,
        "see describe_unit_match_choices",
    )


def _check_strategy_arguments(
    caller: str, curation_strategy: str, manual_curation_choices
) -> None:
    """Reject an unknown strategy, or manual pins with an automatic one."""
    if curation_strategy not in STRATEGIES:
        raise ValueError(
            f"{caller}: unknown curation_strategy "
            f"{curation_strategy!r}; choose one of {STRATEGIES}."
        )
    # manual_curation_choices is only consulted by curation_strategy='manual';
    # passing it with an automatic curation strategy would silently do nothing
    # (the planner picks), so reject it up front rather than let it look
    # intentional.
    if manual_curation_choices is not None and curation_strategy != "manual":
        raise ValueError(
            f"{caller}: manual_curation_choices is only used by "
            f"curation_strategy='manual', not {curation_strategy!r}; the "
            "planner picks the curations for the other strategies."
        )


def build_unit_match_plan(
    *,
    session_group_owner: str,
    session_group_name: str,
    matcher_params_name: str,
    curation_strategy: str,
    members: list[dict],
    manual_curation_choices: dict | None = None,
) -> UnitMatchPlan:
    """Assemble a pinned :class:`UnitMatchPlan` from per-member curation lists.

    ``members`` is the structured per-member choices (one entry per member:
    ``member_index`` / ``nwb_file_name`` / ``choices`` list of ``{sorting_id,
    curation_id, curation_uuid, parent_curation_id, curation_source,
    description}``) --
    ``describe_unit_match_choices`` tabulates the same data. DB-free.
    ``curation_strategy`` is REQUIRED:

    - ``final_curated`` -- the member's single terminal curated (non-root)
      curation; errors if a member has zero or more than one.
    - ``auto_curated`` -- the member's auto-curated child
      (``curation_source='curation_evaluation'``); errors on zero / multiple.
    - ``root`` -- the member's root curation, with a loud advisory (uncurated).
    - ``manual`` -- pins ``manual_curation_choices[member_index]`` per member,
      validated against the member's committed curations.

    A member the curation strategy cannot resolve to exactly one curation becomes a
    blocking ``error`` (``plan.ok is False``); other members still resolve, so
    ``plan.as_dataframe()`` shows the whole picture at once.
    """
    _check_strategy_arguments(
        "build_unit_match_plan", curation_strategy, manual_curation_choices
    )

    if manual_curation_choices is not None:
        # Keys go through the same lossless rule as the ids: True / 1.0 hash
        # equal to 1 and would otherwise silently select member 1.
        manual_curation_choices = {
            lossless_int(key, "manual_curation_choices member index"): value
            for key, value in manual_curation_choices.items()
        }
    curation_choices: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for member in sorted(members, key=lambda m: m["member_index"]):
        pinned, member_warnings, member_errors = _resolve_member(
            member, curation_strategy, manual_curation_choices
        )
        warnings.extend(member_warnings)
        errors.extend(member_errors)
        if pinned is not None:
            curation_choices[member["member_index"]] = pinned
        rows.append(
            {
                "member_index": member["member_index"],
                "nwb_file_name": member["nwb_file_name"],
                "sorting_id": pinned["sorting_id"] if pinned else None,
                "curation_id": pinned["curation_id"] if pinned else None,
                "curation_uuid": pinned["curation_uuid"] if pinned else None,
                "status": "pinned" if pinned else "UNRESOLVED",
            }
        )

    # Manual coverage is EXACT: a manual_curation_choices entry for a member
    # index that is not in the group (stale / mistyped) is silently unused
    # above, so flag it as a blocking error rather than let the plan look ok.
    if curation_strategy == "manual" and manual_curation_choices:
        member_indexes = {m["member_index"] for m in members}
        extra = sorted(set(manual_curation_choices) - member_indexes)
        if extra:
            errors.append(
                "manual_curation_choices has entries for member index(es) "
                f"{extra} "
                f"that are not SessionGroup members {sorted(member_indexes)} "
                "(stale or mistyped index)."
            )

    return UnitMatchPlan(
        session_group_owner=session_group_owner,
        session_group_name=session_group_name,
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        curation_choices=curation_choices,
        rows=rows,
        warnings=warnings,
        errors=errors,
    )


def build_unit_match_input_plan(
    *,
    matcher_params_name: str,
    curation_strategy: str,
    sorts: list[dict],
    manual_curation_choices: dict | None = None,
) -> UnitMatchInputPlan:
    """Assemble a pinned :class:`UnitMatchInputPlan` from per-sort curations.

    Each entry of ``sorts`` is one matching input: a sort of a single
    recording or of a same-day concatenation, as ``{sorting_id,
    source_kind, source_id, nwb_file_names, interval_list_names, choices}``
    where ``nwb_file_names`` / ``interval_list_names`` list the sort's
    constituent recordings in recording order and ``choices`` is the sort's
    committed curations (``{sorting_id, curation_id, curation_uuid,
    parent_curation_id, curation_source, description}``). DB-free.
    ``curation_strategy`` is
    REQUIRED and resolves exactly as in :func:`build_unit_match_plan`, within
    each sort's own curations:

    - ``final_curated`` -- the sort's single terminal curated (non-root)
      curation; errors on zero or several.
    - ``auto_curated`` -- the sort's auto-curated child
      (``curation_source='curation_evaluation'``); errors on zero / several.
    - ``root`` -- the sort's root curation, with a loud advisory (uncurated).
    - ``manual`` -- pins ``manual_curation_choices[sorting_id]`` (a
      ``curation_id``) per sort, validated against the sort's committed
      curations; an entry for a sort that is not named is a blocking error.

    A sort the strategy cannot resolve to exactly one curation becomes a
    blocking ``error`` (``plan.ok is False``); the other sorts still resolve,
    so ``plan.as_dataframe()`` shows every input at once.

    Raises
    ------
    ValueError
        On an unknown ``curation_strategy``, ``manual_curation_choices`` with
        an automatic strategy, no sorts, a sort named twice, or a manual
        ``curation_id`` that is not an integer.
    """
    _check_strategy_arguments(
        "build_unit_match_input_plan",
        curation_strategy,
        manual_curation_choices,
    )
    if not sorts:
        raise ValueError(
            "build_unit_match_input_plan: name at least one sort to match."
        )
    sorting_ids = [str(sort["sorting_id"]) for sort in sorts]
    repeated = sorted(
        {
            sorting_id
            for sorting_id in sorting_ids
            if sorting_ids.count(sorting_id) > 1
        }
    )
    if repeated:
        raise ValueError(
            "build_unit_match_input_plan: each sort is one matching input and "
            f"may be named once; named more than once: {repeated}."
        )
    manual = None
    if manual_curation_choices is not None:
        manual = {
            str(sorting_id): lossless_int(
                curation_id, f"sorting {sorting_id} curation_id"
            )
            for sorting_id, curation_id in manual_curation_choices.items()
        }

    curations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for sort in sorts:
        sorting_id = str(sort["sorting_id"])
        chosen = None
        if manual is not None and sorting_id in manual:
            chosen = {
                "sorting_id": sorting_id,
                "curation_id": manual[sorting_id],
            }
        choices = [
            {**choice, "sorting_id": str(choice["sorting_id"])}
            for choice in sort["choices"]
        ]
        pinned, sort_warnings, sort_errors = _pin_one(
            f"sort {sorting_id} ({sort['source_kind']}: "
            f"{', '.join(sort['nwb_file_names'])})",
            "sort",
            choices,
            curation_strategy,
            chosen,
            "see its CurationV2 rows",
        )
        warnings.extend(sort_warnings)
        errors.extend(sort_errors)
        if pinned is not None:
            curations.append(pinned)
        rows.append(
            {
                "sorting_id": sorting_id,
                "source_kind": sort["source_kind"],
                "source_id": str(sort["source_id"]),
                "nwb_file_names": tuple(sort["nwb_file_names"]),
                "interval_list_names": tuple(sort["interval_list_names"]),
                "curation_id": pinned["curation_id"] if pinned else None,
                "curation_uuid": pinned["curation_uuid"] if pinned else None,
                "status": "pinned" if pinned else "UNRESOLVED",
            }
        )

    # Manual coverage is EXACT: an entry for a sort that was not named
    # (stale / mistyped) would be silently unused, so it blocks the plan.
    if manual:
        extra = sorted(set(manual) - set(sorting_ids))
        if extra:
            errors.append(
                "manual_curation_choices has entries for sorting_id(s) "
                f"{extra} that are not among the named sorts."
            )

    return UnitMatchInputPlan(
        matcher_params_name=matcher_params_name,
        curation_strategy=curation_strategy,
        curations=curations,
        rows=rows,
        warnings=warnings,
        errors=errors,
    )
