"""DB-free curation-choice planning for cross-session unit matching.

``run_v2_unit_match`` requires an explicit ``curation_choices`` dict -- pinning
exactly one committed curation per SessionGroup member -- so a match never
silently depends on an implicit "latest" curation. Hand-building that dict is
error-prone, so :func:`build_unit_match_plan` turns the per-member curation
lists (the structured form ``describe_unit_match_choices`` tabulates) into that
dict via a named, intent-declaring curation strategy and returns a
:class:`UnitMatchPlan` (with per-member warnings / errors) to inspect BEFORE the
expensive match.

This module is deliberately DB-free: it operates on already-fetched per-member
curation lists, so the curation-strategy logic is unit-tested without a database. The
DataJoint plumbing (fetching the members, running the plan) lives in
``_pipeline_run``. The low-level ``UnitMatchSelection`` stays explicit and
pinned -- this only redesigns the UX layer that assembles the pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# Every curation strategy is an explicit declaration of intent; there is
# deliberately no default (an implicit "latest" is exactly what the pinned
# design forbids).
STRATEGIES: tuple[str, ...] = (
    "single_leaf_curated",
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
    "parent_curation_id",
    "curation_source",
    "description",
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

    ``curation_choices`` is the ``{member_index: {"sorting_id", "curation_id"}}``
    dict ``run_v2_unit_match`` consumes. ``errors`` (blocking) is non-empty when
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
        """True when the curation strategy pinned exactly one curation for every member."""
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
                "status",
            ],
        )


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
    if curation_strategy == "single_leaf_curated":
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


def _describe_none(curation_strategy: str) -> str:
    """The per-member 'no candidate' fix hint for an automatic curation strategy."""
    if curation_strategy == "auto_curated":
        return (
            "has no auto-curated curation (curation_source="
            "'curation_evaluation'); sort the member with auto_curate=True, or "
            "pin one explicitly with curation_strategy='manual'"
        )
    if curation_strategy == "single_leaf_curated":
        return (
            "has no curated (non-root) leaf curation; curate the member first "
            "(see the Curation how-to), or pin one explicitly with "
            "curation_strategy='manual'"
        )
    return "has no curation yet; sort the member first"  # root


def _resolve_member(
    member: dict, curation_strategy: str, manual_choices: dict | None
) -> tuple[dict | None, list[str], list[str]]:
    """Pin one curation for a member per the curation strategy.

    Returns ``(pinned_or_None, warnings, errors)`` where ``pinned`` is
    ``{"sorting_id", "curation_id"}``.
    """
    idx = member["member_index"]
    nwb = member["nwb_file_name"]
    choices = member["choices"]
    warnings: list[str] = []

    if curation_strategy == "manual":
        chosen = (manual_choices or {}).get(idx)
        if chosen is None:
            return (
                None,
                warnings,
                [
                    f"member {idx} ({nwb}): curation_strategy='manual' "
                    "requires an explicit curation_choices entry for this "
                    "member; none was provided."
                ],
            )
        pinned = {
            "sorting_id": chosen["sorting_id"],
            "curation_id": int(chosen["curation_id"]),
        }
        available = {(c["sorting_id"], c["curation_id"]) for c in choices}
        if (pinned["sorting_id"], pinned["curation_id"]) not in available:
            return (
                None,
                warnings,
                [
                    f"member {idx} ({nwb}): pinned curation "
                    f"(sorting_id={pinned['sorting_id']}, "
                    f"curation_id={pinned['curation_id']}) is not among this "
                    "member's committed curations (see describe_unit_match_choices)."
                ],
            )
        return pinned, warnings, []

    candidates = _candidates(choices, curation_strategy)
    if len(candidates) == 0:
        return (
            None,
            warnings,
            [
                f"member {idx} ({nwb}): "
                f"curation_strategy={curation_strategy!r} "
                f"{_describe_none(curation_strategy)}."
            ],
        )
    if len(candidates) > 1:
        ids = sorted((c["sorting_id"], c["curation_id"]) for c in candidates)
        return (
            None,
            warnings,
            [
                f"member {idx} ({nwb}): "
                f"curation_strategy={curation_strategy!r} is ambiguous -- "
                f"{len(candidates)} candidate curations {ids}. Pin one "
                "explicitly with curation_strategy='manual'."
            ],
        )
    chosen = candidates[0]
    if curation_strategy == "root":
        warnings.append(
            f"member {idx} ({nwb}): curation_strategy='root' pins the "
            "UNCURATED root curation; matching uncurated units is rarely what "
            "you want -- prefer curation_strategy='single_leaf_curated' or "
            "'auto_curated'."
        )
    return (
        {
            "sorting_id": chosen["sorting_id"],
            "curation_id": int(chosen["curation_id"]),
        },
        warnings,
        [],
    )


def build_unit_match_plan(
    *,
    session_group_owner: str,
    session_group_name: str,
    matcher_params_name: str,
    curation_strategy: str,
    members: list[dict],
    manual_choices: dict | None = None,
) -> UnitMatchPlan:
    """Assemble a pinned :class:`UnitMatchPlan` from per-member curation lists.

    ``members`` is the structured per-member choices (one entry per member:
    ``member_index`` / ``nwb_file_name`` / ``choices`` list of ``{sorting_id,
    curation_id, parent_curation_id, curation_source, description}``) --
    ``describe_unit_match_choices`` tabulates the same data. DB-free.
    ``curation_strategy`` is REQUIRED:

    - ``single_leaf_curated`` -- the member's single terminal curated (non-root)
      curation; errors if a member has zero or more than one.
    - ``auto_curated`` -- the member's auto-curated child
      (``curation_source='curation_evaluation'``); errors on zero / multiple.
    - ``root`` -- the member's root curation, with a loud advisory (uncurated).
    - ``manual`` -- pins ``manual_choices[member_index]`` per member, validated
      against the member's committed curations.

    A member the curation strategy cannot resolve to exactly one curation becomes a
    blocking ``error`` (``plan.ok is False``); other members still resolve, so
    ``plan.as_dataframe()`` shows the whole picture at once.
    """
    if curation_strategy not in STRATEGIES:
        raise ValueError(
            "build_unit_match_plan: unknown curation_strategy "
            f"{curation_strategy!r}; choose one of {STRATEGIES}."
        )
    # manual_choices is only consulted by curation_strategy='manual'; passing it
    # with an automatic curation strategy would silently do nothing (the planner picks), so
    # reject it up front rather than let it look intentional.
    if manual_choices is not None and curation_strategy != "manual":
        raise ValueError(
            "build_unit_match_plan: curation_choices is only used by "
            f"curation_strategy='manual', not {curation_strategy!r}; the "
            "planner picks the curations for the other strategies."
        )

    curation_choices: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for member in sorted(members, key=lambda m: m["member_index"]):
        pinned, member_warnings, member_errors = _resolve_member(
            member, curation_strategy, manual_choices
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
                "status": "pinned" if pinned else "UNRESOLVED",
            }
        )

    # Manual coverage is EXACT: a curation_choices entry for a member index that
    # is not in the group (stale / mistyped) is silently unused above, so flag
    # it as a blocking error rather than let the plan look ok.
    if curation_strategy == "manual" and manual_choices:
        member_indexes = {m["member_index"] for m in members}
        extra = sorted(set(manual_choices) - member_indexes)
        if extra:
            errors.append(
                f"curation_choices has entries for member index(es) {extra} "
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
