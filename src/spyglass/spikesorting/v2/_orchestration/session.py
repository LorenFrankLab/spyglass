"""Run the v2 pipeline across the selected sort groups of one session."""

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
from spyglass.spikesorting.v2.exceptions import (
    PipelineInputError,
    PipelineStageError,
    PreflightError,
    ZeroUnitSortError,
)
from spyglass.utils import logger


def run_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    *,
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

    Runs :func:`run_v2_pipeline` on each target group sequentially. Requires
    an explicit ``pipeline_preset``.

    Preflight and error handling
    ----------------------------
    With ``preflight=True``, :func:`preflight_v2_pipeline_session` runs once
    up front; groups then run with ``preflight=False``. ``continue_on_error``
    catches only :class:`PipelineStageError`, :class:`PreflightError`, and
    :class:`ZeroUnitSortError` per group; other errors signal
    misconfiguration and stop the batch.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, curation_description, require_units, auto_curate
        As in :func:`run_v2_pipeline`; applied to every group.
    motion_mode, motion_correction_params_name
        As in :func:`run_v2_pipeline`; applied to every group.
    pipeline_preset
        Required pipeline-preset name (no default). See
        ``describe_pipeline_presets()``.
    sort_group_ids
        Optional explicit subset of sort groups to run. ``None`` (default)
        runs every ``SortGroupV2`` row for the session.
    preflight
        If True (default), run the whole-session preflight once before compute.
    continue_on_error
        If False (default), the first per-group failure propagates. If True,
        failed groups yield an ``outcome="failed"`` entry.

    Returns
    -------
    list[RunV2PipelineSessionResult]
        One entry per target group, in ascending ``sort_group_id`` order.
        Success is a :class:`RunResult` with the run summary plus
        ``sort_group_id`` and ``outcome="ok"``. Failure is a
        ``RunV2PipelineSessionFailed`` dict; ``stage``,
        ``original_error_type`` and ``partial_run_summary`` are ``None``
        unless a stage failed. See ``describe_run(results)``.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is ``None`` or unknown, the session has no sort
        groups, a requested group is absent, or the motion request is
        contradictory.
    PreflightError
        If a group fails preflight and ``continue_on_error=False``; raised
        before any group is sorted.
    PipelineStageError, ZeroUnitSortError
        Propagated from a per-group :func:`run_v2_pipeline` when
        ``continue_on_error=False``.
    """
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

    # Preflight-failed entries were appended first.
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
    # Groups run with preflight=False, so carry advisories forward here.
    for row in session_report.group_reports:
        if row.get("warnings"):
            preflight_warnings_by_group[row["sort_group_id"]] = list(
                row["warnings"]
            )
    if not session_report.ok:
        if not continue_on_error:
            raise PreflightError("\n".join(session_report.errors))
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
                    stage=None,
                    original_error_type=None,
                    partial_run_summary=None,
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
            stage=getattr(exc, "stage", None),
            original_error_type=getattr(exc, "original_type", None),
            partial_run_summary=getattr(exc, "partial_run_summary", None),
            # Stage warnings live on partial_run_summary.
            warnings=preflight_warnings_by_group.get(sort_group_id, []),
        )
    group_preflight_warnings = preflight_warnings_by_group.get(
        sort_group_id, []
    )
    for warning in group_preflight_warnings:
        logger.warning(
            "run_v2_pipeline_session: sort_group_id="
            f"{sort_group_id} preflight: {warning}"
        )
    # RunResult keeps the curation accessors.
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

    Includes zero-unit and warned groups, which are easy to miss.
    """
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
