"""The ``Preflight*`` report types a preflight returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pformat
from typing import Any


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check outcome: name, pass/fail, and the fix on fail."""

    name: str  # e.g. "session_exists", "sorter_installed"
    ok: bool
    fix: str  # empty when ok; the actionable fix when not ok


@dataclass(frozen=True)
class PreflightReport:
    """Result of ``preflight_v2_pipeline``: a pre-populate config check.

    Truthy when the configuration is runnable (``ok is True``).

    Attributes
    ----------
    ok
        True when no blocking problem was found (``errors`` is empty).
    errors
        Blocking-problem messages (the ``fix`` of each failed check in
        ``checks``); non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories (e.g. ``artifact_detection_params_name="none"``).
    resolved_pipeline_preset
        The pipeline-preset name that was checked.
    expected_ids
        The selection PKs ``run_v2_pipeline`` would produce, e.g.
        ``{"recording_id": {"id": UUID(...), "exists": False,
        "computed_exists": False}, ...}``. ``exists`` means the selection row
        exists; ``computed_exists`` means the output row exists. Empty when
        the preset is unknown. Excludes ``curation_id``, which is not
        content-addressed. Motion entries, and an ``"apply"`` run's
        ``sorting_id``, have ``id=None`` and a ``pending`` reason until the
        recording is computed.
    checks
        Per-check detail.
    resource_notes
        Informational: per-unit display waveform bound, cache and scratch
        locations, and worker / chunk settings.
    effective_config
        What the sort stage would execute, as
        ``EffectiveSortConfig.as_dict()``, resolved by the dispatcher's own
        ``resolve_sort_config``. ``None`` when the sorter row is missing.
    """

    warnings: list[str]
    resolved_pipeline_preset: str
    expected_ids: dict
    checks: list["PreflightCheck"]
    effective_config: "dict | None" = None
    resource_notes: list[str] = field(default_factory=list)
    scientific_config: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[str]:
        """The fix of each failed check, in check order."""
        return [c.fix for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        """``True`` when no check failed."""
        return not self.errors

    def __bool__(self) -> bool:
        """Return ``True`` when the configuration is runnable (``ok``)."""
        return self.ok

    def summary(self) -> str:
        """Render the execution plan and actionable findings without DB reads."""
        state = "ready" if self.ok else "blocked"
        return "\n".join(
            [f"Preflight {state}: {self.resolved_pipeline_preset}"]
            + _preflight_details(
                self.errors,
                self.warnings,
                self.expected_ids,
                self.effective_config,
                self.resource_notes,
                self.scientific_config,
            )
        )


@dataclass(frozen=True)
class PreflightSessionReport:
    """Result of ``preflight_v2_pipeline_session``: a whole-session check.

    Truthy when every target group is runnable.

    Attributes
    ----------
    ok
        True when no target group has a blocking problem (``errors`` empty).
    errors
        Blocking-problem messages across all groups, each prefixed with its
        ``sort_group_id``; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories across all groups, each prefixed with its
        ``sort_group_id``.
    resolved_pipeline_preset
        The pipeline-preset name that was checked for every group.
    group_reports
        One plain-dict entry per target sort group, with keys
        ``sort_group_id``, ``ok``, ``errors``, ``warnings``, ``expected_ids``,
        ``checks``, ``effective_config``, ``resource_notes``, and
        ``scientific_config``.
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    group_reports: list[dict[str, Any]]

    def __bool__(self) -> bool:
        """Return ``True`` when every target group is runnable (``ok``)."""
        return self.ok

    def summary(self) -> str:
        """Render the checked targets and their execution plans."""
        state = "ready" if self.ok else "blocked"
        lines = [
            f"Session preflight {state}: {self.resolved_pipeline_preset}",
            "Target sort groups: "
            + ", ".join(
                str(row["sort_group_id"]) for row in self.group_reports
            ),
        ]
        for row in self.group_reports:
            lines.append(f"Sort group {row['sort_group_id']}:")
            lines.extend(
                _preflight_details(
                    row["errors"],
                    row["warnings"],
                    row["expected_ids"],
                    row["effective_config"],
                    row["resource_notes"],
                    row["scientific_config"],
                )
            )
        return "\n".join(lines)


def _preflight_details(
    errors,
    warnings,
    expected_ids,
    effective_config,
    resources,
    scientific_config=None,
):
    """Format fields shared by single-group and session preflight reports."""
    lines = [f"  ERROR: {message}" for message in errors]
    lines.extend(f"  Warning: {message}" for message in warnings)
    if scientific_config:
        lines.append("  Scientific setup:")
        lines.extend(
            f"    {line}"
            for line in pformat(scientific_config, width=76).splitlines()
        )
    for key, selection in expected_ids.items():
        if selection.get("pending"):
            action = f"compute (id known once {selection['pending']})"
        elif selection["id"] is None:
            action = "skip"
        elif selection["computed_exists"]:
            action = "reuse completed output"
        else:
            action = "compute"
        lines.append(f"  {key.removesuffix('_id')}: {action}")
    if effective_config is not None:
        lines.append("  Effective sorter configuration:")
        lines.extend(
            f"    {line}"
            for line in pformat(effective_config, width=76).splitlines()
        )
    if resources:
        lines.append("  Resources:")
        lines.extend(f"    {note}" for note in resources)
    return lines
