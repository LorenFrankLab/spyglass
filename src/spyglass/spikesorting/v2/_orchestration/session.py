"""Run the v2 pipeline across the selected sort groups of one session.

Owns session preflight, sequential group execution, failure handling, and
aggregate receipts. Each group delegates to the individual pipeline runner."""

from __future__ import annotations

from typing import cast

from spyglass.spikesorting.v2._orchestration.preflight import (
    _resolve_session_sort_group_ids,
    motion_request_problem,
    preflight_v2_pipeline_session,
)
from spyglass.spikesorting.v2._orchestration.reporting import (
    _session_outcome_counts,
)
from spyglass.spikesorting.v2._orchestration.types import (
    MotionMode,
    RunV2PipelineSessionFailed,
    RunV2PipelineSessionOk,
    RunV2PipelineSessionResult,
)
from spyglass.spikesorting.v2._orchestration.run import run_v2_pipeline
from spyglass.spikesorting.v2.curation_api import RunResult


def run_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str,
    sort_group_ids: "list[int] | None" = None,
    curation_description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    continue_on_error: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
) -> list[RunV2PipelineSessionResult]:
    """Sort every (or selected) sort group in a session in one call.

    A thin batch wrapper over :func:`run_v2_pipeline`: it resolves the
    session's target ``SortGroupV2`` rows and runs the single-group
    orchestrator on each, returning one result entry per group. ``run_v2_pipeline``
    already parallelizes the heavy ``populate`` internally; this wrapper loops
    the groups **sequentially**.

    Unlike :func:`run_v2_pipeline`, an explicit ``pipeline_preset`` is required
    (a whole-session run infers no default). Choose one from
    ``describe_pipeline_presets()``.

    Preflight and error handling
    ----------------------------
    With ``preflight=True`` (default), :func:`preflight_v2_pipeline_session`
    runs once up front. If any group fails preflight and
    ``continue_on_error=False``, a :class:`PreflightError` is raised before any
    group is sorted; with ``continue_on_error=True``, the failed groups get a
    ``outcome="failed"`` entry and only the preflight-passing groups are run.
    Either way, the groups that *are* run pass ``preflight=False`` to
    :func:`run_v2_pipeline` (the session preflight already covered the checks;
    with ``preflight=False`` here the caller has opted out entirely).

    ``continue_on_error`` makes the batch resilient to per-group *preflight* and
    *sort* failures only. Exactly :class:`PipelineStageError`,
    :class:`PreflightError`, and :class:`ZeroUnitSortError` are caught per
    group; everything else -- :class:`PipelineInputError` from input validation,
    a bare ``ValueError`` from a missing Lookup row, ``datajoint``'s
    ``IntegrityError`` from a missing upstream when ``preflight=False``, or any
    unexpected bug -- propagates and stops the batch, since those signal a
    misconfiguration or DB-state change that should not be silently skipped.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, curation_description, require_units, auto_curate
        As in :func:`run_v2_pipeline`; applied to every group.
    motion_mode, motion_correction_params_name
        As in :func:`run_v2_pipeline`; applied to every group. A
        contradictory pair raises ``PipelineInputError`` before any database
        access.
    pipeline_preset
        Required pipeline-preset name (no default). See
        ``describe_pipeline_presets()``.
    sort_group_ids
        Optional explicit subset of sort groups to run. ``None`` (default)
        runs every ``SortGroupV2`` row for the session.
    preflight
        If True (default), run the whole-session preflight once before compute
        (see above). If False, skip it and run each group with
        ``preflight=False``.
    continue_on_error
        If False (default), the first per-group preflight/sort failure
        propagates (fail-fast). If True, failed groups yield an
        ``outcome="failed"`` entry and the batch continues.

    Returns
    -------
    list[RunV2PipelineSessionResult]
        One entry per target group, in ascending ``sort_group_id`` order.
        A successful entry is a :class:`RunResult` (a ``dict`` subclass with
        the generation-pinned ``root_curation`` / ``auto_labeled_curation``
        accessors) whose keys are the single-group run summary (see
        :func:`run_v2_pipeline`) plus ``sort_group_id`` and ``outcome="ok"``
        (``RunV2PipelineSessionOk``). A failed entry is a plain dict with
        the ``RunV2PipelineSessionFailed`` keys and no accessors:
        ``{"sort_group_id", "pipeline_preset",
        "outcome": "failed", "error_type", "error", "stage",
        "original_error_type", "partial_run_summary", "warnings"}``. For a stage
        failure (:class:`PipelineStageError`) ``stage`` names the failing stage,
        ``original_error_type`` is the underlying error it wrapped (e.g.
        ``"IndexError"``), and ``partial_run_summary`` carries the stages
        completed before the failure (including their ``stage_seconds`` timing);
        all three are ``None`` for a preflight or zero-unit failure. ``warnings``
        carries this group's preflight advisories so the batch warning count does
        not under-report failed groups. Wrap with ``describe_run(results)`` for a
        receipt table.

    Raises
    ------
    PipelineInputError
        From the shared target resolver: ``pipeline_preset`` is ``None`` or
        unknown, the session has no sort groups, or a requested
        ``sort_group_ids`` entry is absent; or a contradictory motion request.
        Never suppressed by ``continue_on_error``.
    PreflightError
        If ``preflight=True``, a group fails preflight, and
        ``continue_on_error=False`` -- raised before any group is sorted, with
        the aggregated per-group fixes.
    PipelineStageError, ZeroUnitSortError
        Propagated from a per-group :func:`run_v2_pipeline` when
        ``continue_on_error=False``.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    motion_problem = motion_request_problem(
        motion_mode, motion_correction_params_name
    )
    if motion_problem is not None:
        raise PipelineInputError(f"run_v2_pipeline_session: {motion_problem}")
    targets = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=sort_group_ids,
        caller="run_v2_pipeline_session",
    )

    results: list[RunV2PipelineSessionResult] = []
    failed_preflight_ids: set[int] = set()
    preflight_warnings_by_group: dict[int, list[str]] = {}

    # Up-front, read-only whole-session preflight (when requested).
    if preflight:
        _run_session_preflight(
            nwb_file_name=nwb_file_name,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
            targets=targets,
            auto_curate=auto_curate,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            continue_on_error=continue_on_error,
            results=results,
            failed_preflight_ids=failed_preflight_ids,
            preflight_warnings_by_group=preflight_warnings_by_group,
        )

    # Per-group compute. Groups covered by the session preflight (or skipped via
    # preflight=False) run with preflight=False so the DB-only checks are not
    # repeated; a per-group failure is caught only for the three pipeline error
    # types, and only when continue_on_error is set.
    for sort_group_id in targets:
        if sort_group_id in failed_preflight_ids:
            continue
        results.append(
            _run_session_group(
                sort_group_id,
                nwb_file_name=nwb_file_name,
                interval_list_name=interval_list_name,
                team_name=team_name,
                pipeline_preset=pipeline_preset,
                curation_description=curation_description,
                require_units=require_units,
                auto_curate=auto_curate,
                manual_excluded_times=manual_excluded_times,
                motion_mode=motion_mode,
                motion_correction_params_name=motion_correction_params_name,
                continue_on_error=continue_on_error,
                preflight_warnings_by_group=preflight_warnings_by_group,
            )
        )

    # Stable, group-ordered result (preflight-failed entries were appended
    # first; restore ascending sort_group_id order).
    results.sort(key=lambda entry: entry["sort_group_id"])

    _log_session_receipt(results)
    return results


def _run_session_preflight(
    *,
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str,
    targets: list[int],
    auto_curate: bool,
    manual_excluded_times,
    motion_mode,
    motion_correction_params_name: "str | None",
    continue_on_error: bool,
    results: list,
    failed_preflight_ids: set[int],
    preflight_warnings_by_group: dict[int, list[str]],
) -> None:
    """Run the whole-session preflight once and record its outcome.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, pipeline_preset, auto_curate
        As in :func:`run_v2_pipeline_session`.
    manual_excluded_times, motion_mode, motion_correction_params_name
        As in :func:`run_v2_pipeline_session`.
    targets : list[int]
        The sort groups to check.
    continue_on_error : bool
        If False, any failed group raises; if True, each failed group gets a
        ``outcome="failed"`` entry.
    results : list
        The batch's entries (failed-preflight entries appended).
    failed_preflight_ids : set[int]
        The groups that failed preflight (added to).
    preflight_warnings_by_group : dict[int, list[str]]
        Each group's non-blocking advisories (filled in).

    Raises
    ------
    PreflightError
        If a group fails preflight and ``continue_on_error`` is False.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.utils import logger

    session_report = preflight_v2_pipeline_session(
        nwb_file_name=nwb_file_name,
        interval_list_name=interval_list_name,
        team_name=team_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=targets,
        auto_curate=auto_curate,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
    )
    # Capture each group's non-blocking advisories. OK groups run in
    # _run_session_group with preflight=False (the DB checks are not
    # repeated), so without this their preflight warnings would never reach
    # the run summary and the batch warning count would under-report.
    for row in session_report.group_reports:
        if row.get("warnings"):
            preflight_warnings_by_group[row["sort_group_id"]] = list(
                row["warnings"]
            )
    if not session_report.ok:
        if not continue_on_error:
            raise PreflightError("\n".join(session_report.errors))
        # continue_on_error: record the failed groups, run only the rest.
        for row in session_report.group_reports:
            if row["ok"]:
                continue
            sort_group_id = row["sort_group_id"]
            failed_preflight_ids.add(sort_group_id)
            logger.warning(
                "run_v2_pipeline_session: sort_group_id="
                f"{sort_group_id} failed preflight; skipping. "
                f"{row['errors']}"
            )
            results.append(
                RunV2PipelineSessionFailed(
                    sort_group_id=sort_group_id,
                    pipeline_preset=pipeline_preset,
                    outcome="failed",
                    error_type="PreflightError",
                    error="\n".join(row["errors"]),
                    # A preflight failure is not a stage failure -- keep the
                    # structured fields present (shape-consistent) but None.
                    stage=None,
                    original_error_type=None,
                    partial_run_summary=None,
                    # Carry this group's advisories too, so describe_run /
                    # the batch warning count do not under-report failures.
                    warnings=list(row.get("warnings", [])),
                )
            )


def _run_session_group(
    sort_group_id: int,
    *,
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str,
    curation_description: str,
    require_units: bool,
    auto_curate: bool,
    manual_excluded_times,
    motion_mode,
    motion_correction_params_name: "str | None",
    continue_on_error: bool,
    preflight_warnings_by_group: dict[int, list[str]],
) -> RunV2PipelineSessionResult:
    """Run one sort group of a session batch; return its result entry.

    The group runs :func:`run_v2_pipeline` with ``preflight=False``; its
    preflight advisories are folded into the entry either way.

    Raises
    ------
    PipelineStageError, PreflightError, ZeroUnitSortError
        From the group's run when ``continue_on_error`` is False; with it
        True they become an ``outcome="failed"`` entry.
    """
    from spyglass.spikesorting.v2.exceptions import (
        PipelineStageError,
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.utils import logger

    try:
        summary = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
            curation_description=curation_description,
            require_units=require_units,
            auto_curate=auto_curate,
            preflight=False,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
        )
    except (
        PipelineStageError,
        PreflightError,
        ZeroUnitSortError,
    ) as exc:
        if not continue_on_error:
            raise
        logger.warning(
            "run_v2_pipeline_session: sort_group_id="
            f"{sort_group_id} failed: {exc!r}"
        )
        return RunV2PipelineSessionFailed(
            sort_group_id=sort_group_id,
            pipeline_preset=pipeline_preset,
            outcome="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            # Surface the failing STAGE and the underlying error type (a
            # PipelineStageError wraps e.g. an IndexError) so a batch
            # caller can triage without re-parsing the message. Both are
            # None for non-stage failures (preflight / zero-unit).
            stage=getattr(exc, "stage", None),
            original_error_type=getattr(exc, "original_type", None),
            partial_run_summary=getattr(exc, "partial_run_summary", None),
            # This group passed preflight (ran with preflight=False) but
            # failed mid-run; keep its preflight advisories visible. Any
            # stage warnings live on partial_run_summary, which the
            # receipt counts read too.
            warnings=preflight_warnings_by_group.get(sort_group_id, []),
        )
    # Fold this group's preflight advisories (captured by
    # _run_session_preflight) into its run summary; the per-group run did no
    # preflight, so there is no overlap with the stage warnings it carries.
    group_preflight_warnings = preflight_warnings_by_group.get(
        sort_group_id, []
    )
    for warning in group_preflight_warnings:
        logger.warning(
            "run_v2_pipeline_session: sort_group_id="
            f"{sort_group_id} preflight: {warning}"
        )
    # The successful entry is the RunResult itself (keeps the
    # curation accessors); its keys are RunV2PipelineSessionOk.
    ok_entry = RunResult(
        {
            **summary,
            "sort_group_id": sort_group_id,
            "outcome": "ok",
            "warnings": list(summary.get("warnings", []))
            + group_preflight_warnings,
        }
    )
    return cast(RunV2PipelineSessionOk, ok_entry)


def _log_session_receipt(results: list) -> None:
    """Log a session batch's end-of-run outcome counts.

    Surfaces the outcomes that are easy to miss when scrolling a long run --
    not just failures but zero-unit sorts and warnings too.
    ``describe_run(results)`` is the richer table form.
    """
    from spyglass.utils import logger

    n_ok, n_failed, n_zero, n_warn = _session_outcome_counts(results)
    failed_details = [
        f"sort_group_id={entry['sort_group_id']}: "
        f"{entry.get('error_type') or 'Error'}"
        for entry in results
        if entry["outcome"] == "failed"
    ]
    failed_suffix = f" ({', '.join(failed_details)})" if failed_details else ""
    logger.info(
        f"run_v2_pipeline_session: {len(results)} group(s): {n_ok} ok, "
        f"{n_failed} failed{failed_suffix}, {n_zero} zero-unit, "
        f"{n_warn} with warnings. "
        "Call describe_run(results) for the per-group receipt."
    )
