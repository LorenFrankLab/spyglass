"""Read-only matching preflight shared by reviewed plans and their executor.

Preflight never seeds defaults, inserts rows, rebuilds assets, or extracts
waveforms. It resolves unit eligibility before the canonical matching-input
validator checks source frames, coverage, ordering, geometry, and the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from spyglass.spikesorting.v2._matching.planning import (
    UnitMatchInputPlan,
    UnitMatchPlan,
    _reviewed_pin,
    unit_match_plan_structure_errors,
)
from spyglass.spikesorting.v2.exceptions import CurationNotFoundError


@dataclass(frozen=True)
class UnitMatchPreflightCheck:
    """One verified, blocking, or explicitly unexecuted prerequisite."""

    name: str
    status: Literal["passed", "blocked", "not_run"]
    message: str
    input_index: int | None = None


@dataclass(frozen=True)
class UnitMatchPreflightReport:
    """Read-only prerequisite findings and the exact checked input identity."""

    matcher_params_name: str
    checks: tuple[UnitMatchPreflightCheck, ...]
    unit_selections: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    input_set_hash: str | None = None
    _prepared: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def errors(self) -> list[str]:
        return [
            check.message for check in self.checks if check.status == "blocked"
        ]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(
            check.status == "passed" for check in self.checks
        )

    def __bool__(self) -> bool:
        return self.ok

    def as_dataframe(self):
        """One row per check, including prerequisites that could not run."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "check": check.name,
                    "input_index": check.input_index,
                    "status": check.status,
                    "message": check.message,
                }
                for check in self.checks
            ],
            columns=["check", "input_index", "status", "message"],
        )

    def summary(self) -> str:
        """State readiness, actionable fixes, and remaining unexecuted checks."""
        lines = [
            f"Unit-match preflight {'ready' if self.ok else 'blocked'}: "
            f"{self.matcher_params_name}."
        ]
        eligibility = [
            check for check in self.checks if check.name == "unit_eligibility"
        ]
        if eligibility:
            lines.append(
                f"Eligibility: {sum(check.status == 'passed' for check in eligibility)}/"
                f"{len(eligibility)} inputs passed."
            )
        lines.extend(f"Fix: {value}" for value in self.errors)
        not_run = [
            check.name for check in self.checks if check.status == "not_run"
        ]
        if not_run:
            lines.append(
                "Not established (remaining checks not run): "
                + ", ".join(not_run)
                + "."
            )
        elif self.ok:
            lines.append(
                "Source/frame coverage and applicable geometry/backend requirements verified."
            )
        lines.extend(f"Warning: {value}" for value in self.warnings[:3])
        if len(self.warnings) > 3:
            lines.append(
                f"{len(self.warnings) - 3} more warnings in report.warnings."
            )
        return "\n".join(lines)


def _not_run_checks() -> list[UnitMatchPreflightCheck]:
    return [
        UnitMatchPreflightCheck(
            name,
            "not_run",
            f"{title} have not been established; shared validation stopped at "
            "the blocking finding above. Repair it and rebuild the plan.",
        )
        for name, title in (
            ("source_frames_and_coverage", "Source/frame and coverage checks"),
            ("required_assets", "Required local artifact checks"),
            ("geometry", "Channel-geometry checks"),
            ("backend", "Matcher-backend availability checks"),
        )
    ]


def _validate_group_plan(plan: UnitMatchPlan) -> None:
    from spyglass.spikesorting.v2._matching.inputs import (
        _validate_member_curations,
        normalize_curation_choices,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup

    group = {
        "session_group_owner": plan.session_group_owner,
        "session_group_name": plan.session_group_name,
    }
    if not (SessionGroup & group):
        raise ValueError(f"SessionGroup {group} is missing; rebuild the plan.")
    members = (SessionGroup.Member & group).fetch(as_dict=True)
    _validate_member_curations(
        members, normalize_curation_choices(plan.curation_choices)
    )


def preflight_v2_unit_match(
    plan: UnitMatchPlan | UnitMatchInputPlan,
) -> UnitMatchPreflightReport:
    """Recheck frozen eligibility and source requirements without any writes.

    Missing files and caches are blocking findings with recovery instructions;
    preflight never heals them. Resolver errors are collected per input. The
    shared source/geometry/backend validator stops at its first error, and
    subsequent checks are explicitly marked ``not_run``.
    """
    if not isinstance(plan, (UnitMatchPlan, UnitMatchInputPlan)):
        raise TypeError(
            "preflight_v2_unit_match requires a reviewed unit-match plan."
        )
    structure_errors = unit_match_plan_structure_errors(plan)
    if structure_errors:
        return UnitMatchPreflightReport(
            plan.matcher_params_name,
            tuple(
                UnitMatchPreflightCheck("requested_inputs", "blocked", error)
                for error in structure_errors
            )
            + tuple(_not_run_checks()),
        )

    from spyglass.spikesorting.v2.analysis_selection import (
        UnitSelectionSnapshot,
        resolve_unit_selection,
    )

    curations = (
        plan.curations
        if isinstance(plan, UnitMatchInputPlan)
        else [pin for _, pin in sorted(plan.curation_choices.items())]
    )
    checks = []
    selections = []
    warnings = []
    for index, pin in enumerate(curations):
        try:
            if plan.unit_selections:
                if index >= len(plan.unit_selections):
                    raise ValueError(
                        "A frozen unit selection is missing for this input; rebuild the plan."
                    )
                selection = plan.unit_selections[index]
            else:
                selection = resolve_unit_selection(pin, policy=plan.unit_policy)
            if not isinstance(selection, UnitSelectionSnapshot):
                raise ValueError(
                    "Reviewed plans require typed UnitSelectionSnapshot objects. "
                    "Resolve the population again and rebuild the plan; primitive "
                    "dictionaries cannot revalidate policy/evidence drift."
                )
            if plan.unit_selections:
                if _reviewed_pin(
                    {
                        "sorting_id": selection.curation.sorting_id,
                        "curation_id": selection.curation.curation_id,
                        "curation_uuid": selection.curation.curation_uuid,
                    }
                ) != _reviewed_pin(pin):
                    raise ValueError(
                        "Frozen unit selection does not pin this input generation; "
                        "rebuild the plan with selections aligned to its curations."
                    )
                selection.revalidate()
            snapshot = selection.snapshot()
            included = tuple(snapshot["included_unit_ids"])
            if not included:
                raise ValueError(
                    f"No units satisfy policy {snapshot['policy_name']!r}. "
                    "Accept the intended units in curation review, or choose "
                    "unit_policy='v2_unflagged_units' explicitly for auto-only "
                    "output, then rebuild the plan."
                )
            unlabeled = [
                unit
                for unit in included
                if not snapshot["labels_by_unit"].get(str(unit), [])
            ]
            if unlabeled:
                warnings.append(
                    f"Curation {pin['sorting_id']}/{pin['curation_id']}: "
                    f"{len(unlabeled)} included units are unlabeled "
                    f"under policy {snapshot['policy_name']!r}."
                )
            selections.append(selection)
            checks.append(
                UnitMatchPreflightCheck(
                    "unit_eligibility",
                    "passed",
                    f"Frozen {len(included)} included and "
                    f"{len(snapshot['excluded_units'])} excluded units under "
                    f"{snapshot['policy_name']!r}.",
                    index,
                )
            )
        except (ValueError, LookupError, FileNotFoundError, OSError) as exc:
            recovery = (
                " The input has changed curation_uuid since planning or was "
                "removed; rebuild and review the plan."
                if isinstance(exc, CurationNotFoundError)
                else ""
            )
            checks.append(
                UnitMatchPreflightCheck(
                    "unit_eligibility",
                    "blocked",
                    f"Input {index} ({pin}): {type(exc).__name__}: {exc}{recovery}",
                    index,
                )
            )
    if not curations:
        checks.append(
            UnitMatchPreflightCheck(
                "inputs",
                "blocked",
                "No curations are pinned; choose matching inputs first.",
            )
        )
    if plan.unit_selections and len(plan.unit_selections) != len(curations):
        checks.append(
            UnitMatchPreflightCheck(
                "unit_eligibility",
                "blocked",
                "Frozen unit selections do not exactly cover the pinned curations; rebuild the plan.",
            )
        )
    if any(check.status == "blocked" for check in checks):
        return UnitMatchPreflightReport(
            plan.matcher_params_name,
            tuple(checks + _not_run_checks()),
            tuple(selections),
            tuple(warnings),
        )

    from spyglass.spikesorting.v2._matching.inputs import prepare_match_inputs

    try:
        if isinstance(plan, UnitMatchPlan):
            _validate_group_plan(plan)
        prepared = prepare_match_inputs(
            curations,
            plan.matcher_params_name,
            unit_selections=[
                value.snapshot() if hasattr(value, "snapshot") else dict(value)
                for value in selections
            ],
            allow_rebuild=False,
            validate_backend=True,
        )
        if plan.input_set_hash is not None and (
            prepared["input_set_hash"] != plan.input_set_hash
        ):
            raise ValueError(
                "Matching source, frame coverage, geometry, or eligibility changed "
                "after planning. Rebuild and review the plan before matching."
            )
    except (
        ValueError,
        LookupError,
        FileNotFoundError,
        OSError,
        ImportError,
    ) as exc:
        checks.append(
            UnitMatchPreflightCheck(
                "matching_inputs", "blocked", f"{type(exc).__name__}: {exc}"
            )
        )
        checks.extend(_not_run_checks())
        return UnitMatchPreflightReport(
            plan.matcher_params_name,
            tuple(checks),
            tuple(selections),
            tuple(warnings),
        )

    order = {
        (str(item["sorting_id"]), int(item["curation_id"])): int(
            item["input_index"]
        )
        for item in prepared["inputs"]
    }
    input_indices = {
        index: order[(str(pin["sorting_id"]), int(pin["curation_id"]))]
        for index, pin in enumerate(curations)
    }
    checks = [
        (
            replace(check, input_index=input_indices[check.input_index])
            if check.input_index is not None
            else check
        )
        for check in checks
    ]
    checks.append(
        UnitMatchPreflightCheck(
            "matching_inputs",
            "passed",
            "Verified curation generations, frozen eligibility, required local artifacts, source/frame coverage, "
            "chronological ordering, and matcher registration. "
            + (
                "Verified backend availability and its channel-geometry requirements."
                if len(curations) >= 2
                else "One input needs no pair inference or channel-geometry comparison."
            ),
        )
    )
    return UnitMatchPreflightReport(
        plan.matcher_params_name,
        tuple(checks),
        tuple(selections),
        tuple(warnings),
        prepared["input_set_hash"],
        prepared,
    )
