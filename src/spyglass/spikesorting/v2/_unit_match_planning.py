"""DB-free curation-choice planning for cross-session unit matching.

``run_v2_unit_match`` requires an explicit ``curation_choices`` dict -- pinning
exactly one committed curation per SessionGroup member -- so a match never
silently depends on an implicit "latest" curation. Hand-building that dict is
error-prone, so :func:`build_unit_match_plan` turns the per-member curation
lists (from ``describe_unit_match_choices``) into that dict via a named,
intent-declaring STRATEGY and returns a reviewable :class:`UnitMatchPlan`
(with per-member warnings / errors) to inspect BEFORE the expensive match.

This module is deliberately DB-free: it operates on already-fetched per-member
curation lists, so the strategy logic is unit-tested without a database. The
DataJoint plumbing (fetching the members, running the plan) lives in
``_pipeline_run``. The low-level ``UnitMatchSelection`` stays explicit and
pinned -- this only redesigns the UX layer that assembles the pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Every strategy is an explicit declaration of intent; there is deliberately no
# default (an implicit "latest" is exactly what the pinned design forbids).
STRATEGIES: tuple[str, ...] = (
    "single_leaf_curated",
    "auto_curated",
    "root",
    "manual",
)

_ROOT_PARENT = -1
_AUTO_SOURCE = "curation_evaluation"


@dataclass
class UnitMatchPlan:
    """A reviewable plan pinning one curation per SessionGroup member.

    ``curation_choices`` is the ``{member_index: {"sorting_id", "curation_id"}}``
    dict ``run_v2_unit_match`` consumes. ``errors`` (blocking) is non-empty when
    the strategy could not pin exactly one curation for some member; ``ok`` is
    then ``False`` and running the plan raises. ``warnings`` are advisory (e.g.
    the ``root`` strategy pins uncurated curations). ``as_dataframe()`` renders
    one row per member for review.
    """

    session_group_owner: str
    session_group_name: str
    matcher_params_name: str
    strategy: str
    curation_choices: dict[int, dict[str, Any]]
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the strategy pinned exactly one curation for every member."""
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


def _candidates(choices: list[dict], strategy: str) -> list[dict]:
    """The curations a strategy considers pinnable for one member."""
    if strategy == "root":
        return [c for c in choices if c["parent_curation_id"] == _ROOT_PARENT]
    if strategy == "auto_curated":
        return [c for c in choices if c["curation_source"] == _AUTO_SOURCE]
    if strategy == "single_leaf_curated":
        leaves = _leaf_keys(choices)
        return [
            c
            for c in choices
            if c["parent_curation_id"] != _ROOT_PARENT
            and (c["sorting_id"], c["curation_id"]) in leaves
        ]
    raise AssertionError(f"unhandled strategy {strategy!r}")  # pragma: no cover


def _describe_none(strategy: str) -> str:
    """The per-member 'no candidate' fix hint for an auto strategy."""
    if strategy == "auto_curated":
        return (
            "has no auto-curated curation (curation_source="
            "'curation_evaluation'); sort the member with auto_curate=True, or "
            "pin one explicitly with strategy='manual'"
        )
    if strategy == "single_leaf_curated":
        return (
            "has no curated (non-root) leaf curation; curate the member first "
            "(see the Curation how-to), or pin one explicitly with "
            "strategy='manual'"
        )
    return "has no curation yet; sort the member first"  # root


def _resolve_member(
    member: dict, strategy: str, manual_choices: dict | None
) -> tuple[dict | None, list[str], list[str]]:
    """Pin one curation for a member per the strategy.

    Returns ``(pinned_or_None, warnings, errors)`` where ``pinned`` is
    ``{"sorting_id", "curation_id"}``.
    """
    idx = member["member_index"]
    nwb = member["nwb_file_name"]
    choices = member["choices"]
    warnings: list[str] = []

    if strategy == "manual":
        chosen = (manual_choices or {}).get(idx)
        if chosen is None:
            return (
                None,
                warnings,
                [
                    f"member {idx} ({nwb}): strategy='manual' requires an explicit "
                    "curation_choices entry for this member; none was provided."
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

    candidates = _candidates(choices, strategy)
    if len(candidates) == 0:
        return (
            None,
            warnings,
            [
                f"member {idx} ({nwb}): strategy={strategy!r} {_describe_none(strategy)}."
            ],
        )
    if len(candidates) > 1:
        ids = sorted((c["sorting_id"], c["curation_id"]) for c in candidates)
        return (
            None,
            warnings,
            [
                f"member {idx} ({nwb}): strategy={strategy!r} is ambiguous -- "
                f"{len(candidates)} candidate curations {ids}. Pin one explicitly "
                "with strategy='manual'."
            ],
        )
    chosen = candidates[0]
    if strategy == "root":
        warnings.append(
            f"member {idx} ({nwb}): strategy='root' pins the UNCURATED root "
            "curation; matching uncurated units is rarely what you want -- "
            "prefer strategy='single_leaf_curated' or 'auto_curated'."
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
    strategy: str,
    members: list[dict],
    manual_choices: dict | None = None,
) -> UnitMatchPlan:
    """Assemble a pinned :class:`UnitMatchPlan` from per-member curation lists.

    ``members`` is the ``describe_unit_match_choices`` output (one entry per
    member: ``member_index`` / ``nwb_file_name`` / ``choices`` list of
    ``{sorting_id, curation_id, parent_curation_id, curation_source,
    description}``). DB-free. ``strategy`` is REQUIRED:

    - ``single_leaf_curated`` -- the member's single terminal curated (non-root)
      curation; errors if a member has zero or more than one.
    - ``auto_curated`` -- the member's auto-curated child
      (``curation_source='curation_evaluation'``); errors on zero / multiple.
    - ``root`` -- the member's root curation, with a loud advisory (uncurated).
    - ``manual`` -- pins ``manual_choices[member_index]`` per member, validated
      against the member's committed curations.

    A member the strategy cannot resolve to exactly one curation becomes a
    blocking ``error`` (``plan.ok is False``); other members still resolve, so
    ``plan.as_dataframe()`` shows the whole picture at once.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"build_unit_match_plan: unknown strategy {strategy!r}; choose one "
            f"of {STRATEGIES}."
        )

    curation_choices: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for member in sorted(members, key=lambda m: m["member_index"]):
        pinned, member_warnings, member_errors = _resolve_member(
            member, strategy, manual_choices
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

    return UnitMatchPlan(
        session_group_owner=session_group_owner,
        session_group_name=session_group_name,
        matcher_params_name=matcher_params_name,
        strategy=strategy,
        curation_choices=curation_choices,
        rows=rows,
        warnings=warnings,
        errors=errors,
    )
