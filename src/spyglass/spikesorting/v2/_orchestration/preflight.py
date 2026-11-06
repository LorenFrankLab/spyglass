"""Read-only configuration checks run before any pipeline populate.

Holds ``preflight_v2_pipeline`` / ``preflight_v2_pipeline_session`` and
re-exports the public names of its ``preflight_*`` siblings:
``preflight_concat`` (the raising ``assert_concat_preflight`` a concat run
executes), ``preflight_checks`` (the per-stage checks, and
``assert_preset_compute_rows``, which the concat preflight shares),
``preflight_runtime`` (sorter and container runtime probes),
``preflight_motion`` / ``preflight_geometry`` (motion and contact-geometry
checks), ``preflight_plan`` (the ids a run would compute or reuse),
``preflight_scientific_setup`` (``describe_scientific_setup``) and
``preflight_report`` (the ``Preflight*`` report types). Patch a private name in
the module that looks it up. At import time these modules depend only on
``_orchestration.presets``, ``_orchestration.types``, and
``_core.recipe_catalog``; table modules are imported inside the functions that
query them.
"""

from __future__ import annotations

from typing import Any

from spyglass.spikesorting.v2._core.recipe_catalog import (
    DEFAULT_PIPELINE_PRESET,
)
from spyglass.spikesorting.v2._orchestration.preflight_checks import (
    _check_auto_curation_rows,
    _check_display_waveform_params,
    _check_motion_stage,
    _check_sampling_rate,
    _check_session_rows,
    _check_sort_group,
    _check_sorter_param_rows,
    _check_source_param_rows,
    assert_preset_compute_rows,
)
from spyglass.spikesorting.v2._orchestration.preflight_concat import (
    assert_concat_preflight,
)
from spyglass.spikesorting.v2._orchestration.preflight_geometry import (
    sort_group_geometry_problem,
)
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    expected_source_ids,
    motion_geometry_problem,
    motion_geometry_problem_from_contacts,
    motion_request_problem,
    resolve_motion_recipe,
    sorter_motion_correction_problem,
    supplied_motion_estimate_problem,
)
from spyglass.spikesorting.v2._orchestration.preflight_plan import (
    _group_resource_notes,
    _preflight_expected_ids,
)
from spyglass.spikesorting.v2._orchestration.preflight_report import (
    PreflightCheck,
    PreflightReport,
    PreflightSessionReport,
)
from spyglass.spikesorting.v2._orchestration.preflight_runtime import (
    _check_sorter_execution,
    _installed_sorters_reused,
)
from spyglass.spikesorting.v2._orchestration.preflight_scientific_setup import (
    MOTION_RECIPE_STATUS,
    describe_scientific_setup,
    resolve_preset_sort_config,
)
from spyglass.spikesorting.v2._orchestration.presets import (
    _PIPELINE_PRESETS,
    _unknown_pipeline_preset_message,
)
from spyglass.spikesorting.v2._orchestration.types import (
    MotionMode,
)


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    *,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    auto_curate: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id=None,
    sort_checks: bool = True,
) -> PreflightReport:
    """Read-only pre-populate configuration check for ``run_v2_pipeline``.

    Verifies -- in ~1 s, inserting nothing and never calling ``populate``
    -- that every prerequisite a subsequent ``run_v2_pipeline(...,
    pipeline_preset=pipeline_preset)`` needs is in place: the session /
    interval / team / sort-group rows exist, the pipeline preset's
    parameter Lookup rows exist, and
    the sorter binary is installed. Returns a structured
    :class:`PreflightReport` instead of failing minutes into ``populate``
    with an opaque foreign-key or SpikeInterface error.

    Every check is a read-only restriction (``& {...}``) or a pure call. Most
    checks run even after one fails, so the report lists every problem at once.
    An unknown ``pipeline_preset`` short-circuits before any database access,
    because the later checks need the resolved param names.

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`.
    auto_curate
        Match ``run_v2_pipeline(auto_curate=...)``. When True, also verify the
        auto-curation prerequisites (the preset's ``QualityMetricParameters`` /
        ``AutoCurationRules`` rows and the whitened metric analyzer recipe), so
        a missing one fails preflight rather than after the upstream compute.
    motion_mode, motion_correction_params_name
        Match ``run_v2_pipeline``. A contradictory pair (a recipe with
        ``"off"``, no recipe otherwise, or an unknown mode) fails the
        ``motion_request_valid`` check before any database access. With a
        motion mode, a preset whose preprocessing recipe applies no temporal
        filter fails ``motion_source_filtered``: motion is estimated on
        filtered, unwhitened traces.
    motion_estimate_id
        Match ``run_v2_pipeline``. Given with a mode other than ``"apply"``
        (or not a UUID), it fails ``motion_request_valid`` before any database
        access. Otherwise the ``motion_estimate_applicable`` check requires a
        populated estimate of this run's recording and artifact mask made
        with the recipe's estimation row, and ``expected_ids`` previews the
        corrected recording and sort built on it.
    sort_checks
        If False, skip the checks only a sort needs -- the sorter row and its
        parameters, the display analyzer recipe, the preset's sampling rate
        and the sorter runtime or container -- for a caller that builds the
        source without sorting it (``estimate_motion``). Default True.

    Returns
    -------
    PreflightReport
        Truthy when the configuration is runnable. ``report.errors`` lists
        each blocking problem with the action to fix it;
        ``report.expected_ids`` holds the deterministic selection PKs the
        run would produce (see :class:`PreflightReport`).
    """
    checks: list[PreflightCheck] = []
    warnings: list[str] = []

    def _check(name: str, ok, fix: str) -> bool:
        ok = bool(ok)
        checks.append(PreflightCheck(name, ok, "" if ok else fix))
        return ok

    # pipeline_preset_known. Short-circuit before any DB access on failure: the
    # remaining checks (and expected_ids) all derive from the resolved
    # bundle's param names, which are unknown for a bogus pipeline preset.
    if pipeline_preset not in _PIPELINE_PRESETS:
        _check(
            "pipeline_preset_known",
            False,
            _unknown_pipeline_preset_message(pipeline_preset),
        )
        return _blocked_preflight_report(pipeline_preset, checks, warnings)
    _check("pipeline_preset_known", True, "")
    # A contradictory motion request short-circuits the same way: the motion
    # checks below need a valid mode and recipe name, and this one is DB-free.
    motion_problem = motion_request_problem(
        motion_mode, motion_correction_params_name, motion_estimate_id
    )
    if not _check(
        "motion_request_valid", motion_problem is None, motion_problem
    ):
        return _blocked_preflight_report(pipeline_preset, checks, warnings)
    bundle = _PIPELINE_PRESETS[pipeline_preset]
    from spyglass.spikesorting.v2._artifacts.manual import (
        artifact_recipe_with_manual_exclusions,
        resolve_manual_exclusions,
    )

    manual_excluded_times = resolve_manual_exclusions(manual_excluded_times)
    bundle = artifact_recipe_with_manual_exclusions(
        bundle, manual_excluded_times
    )

    sort_group_id = int(sort_group_id)

    _check_session_rows(_check, nwb_file_name, interval_list_name, team_name)
    sort_group_exists = _check_sort_group(_check, nwb_file_name, sort_group_id)
    preprocessing_params_exist = _check_source_param_rows(_check, bundle)
    # The sorter-only checks (the sorter row and its params, the display
    # analyzer row, the sampling rate and the sorter execution) are skipped
    # for a caller that builds no sort (``sort_checks=False``).
    sorter_params_exist, sorter_row = _check_sorter_param_rows(
        _check, bundle, sort_checks=sort_checks
    )
    # What run_sorter would receive, resolved by the dispatcher's own resolver.
    effective_config = (
        resolve_preset_sort_config(bundle) if sorter_params_exist else None
    )
    display_waveform_params_name = _check_display_waveform_params(
        _check, bundle, sort_checks=sort_checks
    )
    # Auto-curation prerequisites -- only when the caller opts into
    # auto_curate, so a default run is unchanged.
    if auto_curate:
        _check_auto_curation_rows(_check, bundle)
    if sort_checks:
        _check_sampling_rate(_check, bundle, nwb_file_name, pipeline_preset)
        _check_sorter_execution(
            _check,
            warnings,
            bundle,
            pipeline_preset,
            sorter_row,
        )
    motion_recipe = None
    if motion_mode != "off":
        motion_recipe = _check_motion_stage(
            _check,
            bundle,
            sort_groups=(
                [(_check, nwb_file_name, sort_group_id)]
                if sort_group_exists
                else []
            ),
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            preprocessing_params_exist=preprocessing_params_exist,
            sorter_row=sorter_row,
        )

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    # Manual exclusions still mask the sort, so they suppress it too.
    if (
        bundle.artifact_detection_params_name in (None, "none")
        and not manual_excluded_times
    ):
        warnings.append(
            f"artifact_detection_params_name={bundle.artifact_detection_params_name!r}: "
            "no artifact masking will be applied for this run."
        )

    # expected_ids: the deterministic selection PKs this run would produce, via
    # the SAME payload builders insert_selection uses so they cannot drift.
    # ``exists`` is a read-only & pk check. The recording_id folds in the
    # resolved input hash (membership + reference + interpolate bad-channels)
    # exactly as build_recording_selection_plan does, so preflight's expected id
    # matches the id insert_selection will mint. That resolution reads the sort
    # group and preprocessing-params rows, so the ids are only computable once
    # those exist; when a blocking input is missing the run cannot proceed
    # anyway, so leave expected_ids empty (as with an unknown preset) rather than
    # raising a bare fetch1 that would mask the actionable missing-input check.
    if not (sort_group_exists and preprocessing_params_exist):
        expected_ids = {}
    else:
        expected_ids = _preflight_expected_ids(
            _check,
            bundle,
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_recipe=motion_recipe,
            motion_estimate_id=motion_estimate_id,
        )

    resource_notes = _group_resource_notes(
        display_waveform_params_name,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        effective_config=effective_config,
    )
    return PreflightReport(
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids=expected_ids,
        checks=checks,
        effective_config=effective_config,
        resource_notes=resource_notes,
        scientific_config=describe_scientific_setup(
            bundle,
            [{"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}],
            effective_config,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_recipe=motion_recipe,
        ),
    )


def _blocked_preflight_report(
    pipeline_preset: str, checks: list[PreflightCheck], warnings: list[str]
) -> PreflightReport:
    """The report of a request that fails before any database check."""
    return PreflightReport(
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids={},
        checks=checks,
    )


def _resolve_session_sort_group_ids(
    nwb_file_name: str,
    pipeline_preset: str,
    sort_group_ids: "list[int] | None",
    caller: str,
) -> list[int]:
    """Validate session-runner inputs; return the target ``sort_group_id`` list.

    Shared by :func:`preflight_v2_pipeline_session` and
    :func:`run_v2_pipeline_session` so their input validation -- and its
    wording -- is identical. Unlike the single-group helpers, the session
    helpers do **not** infer a default preset: an explicit ``pipeline_preset``
    is required. The preset checks run before any ``SortGroupV2`` access, so an
    unknown/missing preset is rejected DB-free.

    Parameters
    ----------
    nwb_file_name
        Session whose ``SortGroupV2`` rows define the candidate targets.
    pipeline_preset
        Required pipeline-preset name; ``None`` is rejected.
    sort_group_ids
        Optional explicit subset. ``None`` means "every sort group in the
        session". Normalized to a sorted, de-duplicated ``list[int]``.
    caller
        Name of the calling helper, used to prefix error messages.

    Returns
    -------
    list[int]
        Target ``sort_group_id`` values in ascending order.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is ``None`` or unknown, the session has no
        ``SortGroupV2`` rows, or a requested ``sort_group_ids`` entry is
        absent.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    if pipeline_preset is None:
        raise PipelineInputError(
            f"{caller}: pipeline_preset is required -- a whole-session run does "
            "not infer a default. Call describe_pipeline_presets() to choose "
            "one, then pass pipeline_preset=..."
        )
    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            _unknown_pipeline_preset_message(
                pipeline_preset,
                caller=caller,
                hint=(
                    "Call describe_pipeline_presets() to see what each preset "
                    "does."
                ),
            )
        )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    available = sorted(
        int(g)
        for g in (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
            "sort_group_id"
        )
    )
    if not available:
        raise PipelineInputError(
            f"{caller}: no SortGroupV2 rows for {nwb_file_name!r}. Create sort "
            "groups first with "
            "SortGroupV2.set_group_by_shank(nwb_file_name=...)."
        )

    if sort_group_ids is None:
        return available

    available_set = set(available)
    requested = sorted({int(g) for g in sort_group_ids})
    missing = [g for g in requested if g not in available_set]
    if missing:
        raise PipelineInputError(
            f"{caller}: sort_group_ids {missing} not found for "
            f"{nwb_file_name!r}. Available sort_group_ids: {available}."
        )
    return requested


def preflight_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    *,
    pipeline_preset: str,
    sort_group_ids: "list[int] | None" = None,
    auto_curate: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
) -> PreflightSessionReport:
    """Read-only preflight for every target sort group in a session.

    Runs :func:`preflight_v2_pipeline` once per target ``SortGroupV2`` row and
    aggregates the per-group reports. Read-only and cheap: it inserts nothing,
    calls no ``populate``, and -- unlike the single-group helper -- requires an
    explicit ``pipeline_preset`` (it infers no default). It reuses the
    single-group checks rather than duplicating them, so the two helpers cannot
    drift.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`, except ``pipeline_preset``
        is required (no default).
    sort_group_ids
        Optional explicit subset of sort groups to check. ``None`` (default)
        checks every ``SortGroupV2`` row for the session.
    auto_curate
        Match ``run_v2_pipeline_session(auto_curate=...)``; when True, each
        group's check also verifies the auto-curation prerequisite rows.
    motion_mode, motion_correction_params_name
        Match ``run_v2_pipeline_session``; checked for every group.

    Returns
    -------
    PreflightSessionReport
        Truthy when every target group is runnable. ``report.group_reports``
        holds one plain-dict entry per group; ``report.errors`` /
        ``report.warnings`` aggregate the per-group messages, each prefixed
        with its ``sort_group_id``.

    Raises
    ------
    PipelineInputError
        From the shared target resolver: ``pipeline_preset`` is ``None`` or
        unknown, the session has no sort groups, or a requested
        ``sort_group_ids`` entry is absent. This is *not* swallowed -- a
        misconfigured request is a caller error, not a per-group preflight
        failure.
    """
    targets = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=sort_group_ids,
        caller="preflight_v2_pipeline_session",
    )

    group_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    # Every group checks the same preset's sorter, so probe the installed
    # sorters once for the whole session rather than once per group.
    with _installed_sorters_reused():
        for sort_group_id in targets:
            report = preflight_v2_pipeline(
                nwb_file_name=nwb_file_name,
                sort_group_id=sort_group_id,
                interval_list_name=interval_list_name,
                team_name=team_name,
                pipeline_preset=pipeline_preset,
                auto_curate=auto_curate,
                manual_excluded_times=manual_excluded_times,
                motion_mode=motion_mode,
                motion_correction_params_name=motion_correction_params_name,
            )
            group_reports.append(
                {
                    "sort_group_id": sort_group_id,
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "expected_ids": report.expected_ids,
                    "checks": report.checks,
                    "effective_config": report.effective_config,
                    "resource_notes": report.resource_notes,
                    "scientific_config": report.scientific_config,
                }
            )
            errors.extend(
                f"sort_group_id={sort_group_id}: {e}" for e in report.errors
            )
            warnings.extend(
                f"sort_group_id={sort_group_id}: {w}" for w in report.warnings
            )

    return PreflightSessionReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        group_reports=group_reports,
    )
